"""export_target.py — 把 qlib 选股 CSV 转成 live_trade.py 的 target_YYYYMMDD.json

用法:
    .venv/Scripts/python scripts/export_target.py \\
        --config config/models/alpha158_momentum_volume_k6_dd10_overlay.yaml

每个交易日 T 上午运行（Tushare 已含 T-1 收盘）：
  1. 用最新 factor_data 日期作为 scoring.end_date 覆盖 YAML 固定日期
  2. 调 score_from_config() → 写 selections.csv
  3. 取最近调仓日那批持仓 → 等权 → 写 target_<今天>.json

JSON 输出路径：qmt/targets/target_YYYYMMDD.json（日期 = 今天，供 live_trade 当日执行）
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── 加载 .env（TUSHARE_TOKEN 等）────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
_env_path = _PROJECT_ROOT / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

# ── 日志 ────────────────────────────────────────────────────
def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _abort(msg: str):
    print(f"[ABORT] {msg}", flush=True)
    sys.exit(1)


# ── 代码归一化（同 qmt/target_portfolio.normalize_symbol）───
def _normalize(sym: str) -> str:
    """SH600000 / 600000 / 600000.SH  →  600000.SH"""
    s = str(sym).strip().upper()
    if "." in s:
        code, suffix = s.split(".", 1)
        return f"{code}.{suffix}"
    if s.startswith(("SH", "SZ")):
        return f"{s[2:]}.{s[:2]}"
    suffix = "SH" if s.startswith("6") else "SZ"
    return f"{s}.{suffix}"


# ── 找 factor_data.parquet 最新日期 ─────────────────────────
def _latest_factor_date() -> str:
    import pandas as pd
    factor_path = _PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "factor_data.parquet"
    if not factor_path.exists():
        _abort(f"factor_data.parquet 不存在: {factor_path}（请先跑 main.py update）")
    df = pd.read_parquet(factor_path, columns=["datetime"])
    latest = pd.to_datetime(df["datetime"]).max()
    return latest.strftime("%Y-%m-%d")


# ── 目标构造（与对账脚本共用，单一真相源）──────────────────
def build_target(cfg, batch, exposure=None, normalizer=_normalize) -> dict:
    """从一批已排序且截断到 topk 的选股(batch DataFrame，需含 'symbol' 列)构造 target 主体。

    - 等权分配；末位补齐浮点误差，使 sum(weight)==1.0
    - invested_pct = position.params.stock_pct × exposure（exposure 默认 1.0 = 满仓）
      exposure 来自 overlay 复刻（见 compute_live_exposure）；overlay 未启用时传 None
    返回 {strategy, invested_pct, positions:[{symbol, weight}]}（不含 date，由调用方补）
    """
    k = len(batch)
    if k == 0:
        raise ValueError("build_target: batch 为空")
    weight = round(1.0 / k, 6)
    weights = [weight] * k
    weights[-1] = round(1.0 - weight * (k - 1), 6)
    positions = [
        {"symbol": normalizer(row["symbol"]), "weight": w}
        for (_, row), w in zip(batch.iterrows(), weights)
    ]
    stock_pct = float(cfg.get("position", {}).get("params", {}).get("stock_pct", 0.80))
    exp = 1.0 if exposure is None else float(exposure)
    invested_pct = round(stock_pct * exp, 6)
    return {
        "strategy": cfg["name"],
        "invested_pct": invested_pct,
        "positions": positions,
    }


def compute_live_exposure(cfg, base_return_history) -> float:
    """复刻回测 overlay 的动态仓位：用策略自身基准日收益历史（截至 T-1，因果）算出
    下一交易日的 exposure(0~1)。overlay 未启用或历史为空 → 返回 1.0（满仓）。

    与回测完全一致的关键：overlay 的回撤闸是从序列起点累计 nav/peak 的**路径依赖**量，
    因此 base_return_history 必须从策略建仓起点给起（用 overlay_results.csv 的 base_return
    全历史），不能只给近窗，否则 drawdown 基准不同会导致 exposure 与回测对不上。

    因果性：compute_overlay_frame 计算第 i 天 exposure 时只用前 i-1 天的 overlay 收益，
    故在历史末尾追加一个占位"次日"行后，最后一行 exposure 即为下一交易日的仓位，
    且不依赖该占位行自身的收益值。
    """
    overlay_cfg = dict(cfg.get("overlay", {}))
    if not bool(overlay_cfg.get("enabled", False)):
        return 1.0
    import pandas as pd
    hist = pd.Series(base_return_history, dtype=float).dropna().sort_index()
    if hist.empty:
        return 1.0
    from modules.modeling.predictive_signal import (
        load_bond_overlay_returns,
        load_market_overlay_returns,
    )
    from modules.modeling.portfolio_overlay import build_overlay_config, compute_overlay_frame

    bond = load_bond_overlay_returns()
    if bond is None:
        bond = 0.03 / 252
    config = build_overlay_config(overlay_cfg)
    market_returns = None
    if int(overlay_cfg.get("market_trend_lookback", 0) or 0) > 0:
        market_returns = load_market_overlay_returns(
            str(overlay_cfg.get("market_index_code", "000300.SH"))
        )
    next_idx = pd.to_datetime(hist.index[-1]) + pd.Timedelta(days=1)
    probe = pd.concat([hist, pd.Series([0.0], index=[next_idx])])
    frame = compute_overlay_frame(probe, bond, config, market_returns=market_returns)
    return float(frame["exposure"].iloc[-1])


def _load_base_return_history(cfg):
    """读取该策略最近一次回测 overlay_results.csv 的 base_return 全历史（因果，从建仓起点）。
    供 compute_live_exposure 复刻 overlay 动态仓位。

    前向 live（回测区间之外的新交易日）应在此基础上，把账户每日实现的 base 收益逐日追加
    进来；当前实现仅用已落盘的回测历史，对回测区间内的离线对账完全够用。
    """
    import pandas as pd
    from modules.modeling.predictive_signal import overlay_results_path

    p = Path(overlay_results_path(cfg))
    if not p.exists():
        _log(f"[WARN] 找不到 {p}，overlay 仓位回退满仓(exposure=1.0)；建议先跑一次回测")
        return pd.Series(dtype=float)
    df = pd.read_csv(p)
    return pd.Series(
        df["base_return"].astype(float).values, index=pd.to_datetime(df["date"])
    ).dropna()


# ── 主流程 ──────────────────────────────────────────────────
def run(config_path: str, target_dir: str, dry_run: bool = False):
    import pandas as pd
    from modules.modeling.predictive_signal import (
        load_predictive_config,
        score_from_config,
        selection_path,
    )

    # 1. 加载配置，覆盖 scoring.end_date 为最新数据日
    cfg = load_predictive_config(config_path)
    latest = _latest_factor_date()
    cfg["scoring"]["end_date"] = latest
    # 回溯窗口取最近 90 天，避免全量重算过慢
    from datetime import datetime, timedelta
    start_dt = (datetime.strptime(latest, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    cfg["scoring"]["start_date"] = start_dt
    _log(f"config: {cfg['name']}  scoring: {start_dt} → {latest}")

    # 2. 打分 → 写 selections.csv
    _log("运行 score_from_config ...")
    summary = score_from_config(cfg)
    sel_path = selection_path(cfg)
    _log(f"[OK] 选股 CSV: {sel_path}  rows={summary.get('selection_rows','-')}")

    # 3. 读 selections.csv，取最大调仓日那批持仓
    df = pd.read_csv(sel_path, dtype={"date": str})
    if df.empty:
        _abort("selections.csv 为空，无持仓可输出")

    last_date = df["date"].max()
    _log(f"最近调仓日: {last_date}")
    batch = df[df["date"] == last_date].copy()

    # 4. 按 rank 升序取前 topk 只
    topk = int(cfg.get("selection", {}).get("topk", 6))
    if "rank" in batch.columns:
        batch = batch.sort_values("rank").head(topk)
    elif "score" in batch.columns:
        batch = batch.sort_values("score", ascending=False).head(topk)
    else:
        batch = batch.head(topk)

    if batch.empty:
        _abort("筛选后持仓为空")

    # 5. 复刻 overlay 动态仓位（overlay 启用时 invested_pct 随波动/回撤/趋势动态调整，
    #    与回测一致；未启用则 exposure=None → invested_pct = stock_pct）
    exposure = None
    if bool(cfg.get("overlay", {}).get("enabled", False)):
        exposure = compute_live_exposure(cfg, _load_base_return_history(cfg))
        _log(f"overlay 动态仓位 exposure={exposure:.4f}")

    # 6. 等权 + invested_pct（build_target 与对账脚本共用同一套逻辑）
    k = len(batch)
    target = build_target(cfg, batch, exposure=exposure)
    invested_pct = target["invested_pct"]
    positions = target["positions"]

    # 7. 组装目标 JSON，date = 今天（live_trade 执行当日）
    today = time.strftime("%Y%m%d")
    target = {"date": today, **target}

    if dry_run:
        _log("[DRY-RUN] 不写文件，仅打印 target:")
        print(json.dumps(target, ensure_ascii=False, indent=2))
        return target

    # 8. 写文件
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"target_{today}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(target, f, ensure_ascii=False, indent=2)

    _log(f"[OK] 写入: {out_path}")
    _log(f"     策略: {target['strategy']}  仓位: {invested_pct*100:.0f}%  持仓: {k} 只")
    for p in positions:
        _log(f"       {p['symbol']}  {p['weight']:.4f}")
    return target


# ── CLI ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="qlib 选股 → live_trade target JSON 桥接")
    parser.add_argument(
        "--config", "-c",
        default="config/models/alpha158_momentum_volume_k6_dd10_overlay.yaml",
        help="策略 YAML 路径（相对 qlib_quant 根目录）",
    )
    parser.add_argument(
        "--target-dir", "-o",
        default=r"C:\Users\Administrator\Documents\stock\qmt\targets",
        help="target JSON 输出目录（默认 qmt/targets）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印不写文件",
    )
    args = parser.parse_args()

    # 切换到 qlib_quant 根目录，确保相对路径正确
    os.chdir(_PROJECT_ROOT)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    if not config_path.exists():
        _abort(f"配置文件不存在: {config_path}")

    run(str(config_path), args.target_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
