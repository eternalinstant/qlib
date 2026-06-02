#!/usr/bin/env python3
"""reconcile_backtest_vs_live.py — 回测 ↔ 模拟盘 三层一致性对账（离线历史重放）

把同一份 selections.csv 同时喂给「回测真值」和「qmt 实盘下单逻辑」，逐个历史调仓日对账，
量化两侧的系统性/结构性差异。**纯离线**：不连 MiniQMT、不依赖 xtquant。

三层：
  Layer 1 目标持仓：export_target 的 build_target() 产出 vs selections 选股；
                    并比对 invested_pct(复刻 overlay) vs 回测 overlay_results.exposure。
  Layer 2 下单/成交：复用 qmt rebalancer.plan_orders 在合成行情上出单，比对 blocked
                    名单/计数 vs 回测引擎逐日 blocked_*，并报成交后权重对目标的整手漂移。
  Layer 3 逐日盈亏：股数级账户每日 mark-to-market 的日收益 vs 回测 base_return
                    （相关系数 / 跟踪误差 / 累计缺口）。

口径对齐（与回测引擎逐字一致，全部复用 qlib_engine 的 helper）：
  - 成交时点：选股日 T 的次一交易日开盘（T+1 open）
  - 涨跌停价：prev_close×(1±pct)，pct 按 10%/20% 区分创业板/科创（_get_limit_prices）
  - 成交价：开盘价（deal price）；交易成本用策略 YAML 的 trading 段（与回测同口径）

用法：
  .venv/Scripts/python scripts/reconcile_backtest_vs_live.py \
      --config config/models/alpha158_momentum_volume_k6.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from modules.backtest.qlib_engine import _get_limit_prices, _load_raw_trade_quotes
from modules.modeling.predictive_signal import (
    load_predictive_config,
    overlay_results_path,
    selection_path,
)
import export_target as et  # build_target / compute_live_exposure / _normalize

# qmt 下单逻辑离线复用：路径放到**末尾**，避免 qmt/config.py 覆盖 qlib 的 config 包
QMT_ROOT = PROJECT_ROOT.parent / "qmt"
if str(QMT_ROOT) not in sys.path:
    sys.path.append(str(QMT_ROOT))
import rebalancer as qmt_rb  # plan_orders / summarize_turnover（顶部只 import math+live_config）


try:  # Windows 控制台默认 GBK，含 ↔ 等字符会编码失败
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _log(msg: str):
    print(msg, flush=True)


# ── 数据装载 ────────────────────────────────────────────────
def load_selection_batches(cfg) -> list[tuple[str, pd.DataFrame]]:
    """读 selections.csv，返回 [(选股日, 该日 rank 排序截断 topk 的 batch)]，按日期升序。"""
    sel = pd.read_csv(selection_path(cfg), dtype={"date": str})
    topk = int(cfg.get("selection", {}).get("topk", 6))
    out = []
    for date, grp in sel.groupby("date"):
        if "rank" in grp.columns:
            batch = grp.sort_values("rank").head(topk)
        elif "score" in grp.columns:
            batch = grp.sort_values("score", ascending=False).head(topk)
        else:
            batch = grp.head(topk)
        out.append((date, batch.reset_index(drop=True)))
    return sorted(out, key=lambda x: x[0])


def load_backtest_truth(cfg):
    """回测真值：overlay_results.csv（base_return/exposure/overlay_return）+ 引擎明细 CSV
    （blocked_* 逐日计数）。返回 (overlay_df 索引为 date, detail_df 索引为 date 或 None)。"""
    ov = pd.read_csv(overlay_results_path(cfg))
    ov["date"] = pd.to_datetime(ov["date"])
    ov = ov.set_index("date").sort_index()

    # 引擎明细 CSV：results/backtest_<name>_*_<ts>.csv，取该策略最新一个
    results_dir = PROJECT_ROOT / "results"
    cands = sorted(results_dir.glob(f"backtest_{cfg['name']}_historical_*csi*.csv"))
    detail = None
    if cands:
        detail = pd.read_csv(cands[-1])
        detail["date"] = pd.to_datetime(detail["date"])
        detail = detail.set_index("date").sort_index()
        _log(f"  引擎明细 CSV: {cands[-1].name}")
    else:
        _log("  [WARN] 未找到引擎明细 CSV，Layer 2 跳过与引擎 blocked 计数的比对")
    return ov, detail


def build_price_panel(symbols, start, end):
    """加载 raw 行情，返回 open/close/prev_close 三张 (date×symbol) 透视表（close 前向填充）。"""
    raw = _load_raw_trade_quotes(sorted(set(symbols)), start, end)
    if raw.empty:
        raise SystemExit("[ABORT] raw 行情为空，无法重放")
    raw = raw.reset_index()
    op = raw.pivot(index="datetime", columns="instrument", values="open").sort_index()
    cl = raw.pivot(index="datetime", columns="instrument", values="close").sort_index()
    pc = raw.pivot(index="datetime", columns="instrument", values="prev_close").sort_index()
    return op, cl.ffill(), pc


# ── 账户与成本 ──────────────────────────────────────────────
def cost_rates(cfg) -> tuple[float, float, float, float]:
    t = cfg.get("trading", {})
    buy_comm = float(t.get("buy_commission_rate", 0.0003))
    sell_comm = float(t.get("sell_commission_rate", 0.0003))
    stamp = float(t.get("sell_stamp_tax_rate", 0.001))
    slip_impact = (float(t.get("slippage_bps", 5)) + float(t.get("impact_bps", 5))) / 1e4
    min_comm = float(t.get("min_buy_commission", 5.0))
    return buy_comm + slip_impact, sell_comm + stamp + slip_impact, min_comm, min_comm


def reconcile(cfg, exposure_cadence="daily", order_cap=None):
    """exposure_cadence: overlay 仓位在 live 端的跟随频率
         "daily"     —— 每个交易日跟随回测当日 exposure（export_target 每日运行的真实行为）
         "rebalance" —— 只在选股调仓日更新 exposure，期间持平（低频 overlay，省换手成本）
         "off"       —— 不跟 overlay，固定 stock_pct
       order_cap: 覆盖 live_config.MAX_ORDER_NOTIONAL 单笔上限（None=用 live_config 默认）。
    """
    name = cfg["name"]
    stock_pct = float(cfg.get("position", {}).get("params", {}).get("stock_pct", 0.80))
    overlay_on = bool(cfg.get("overlay", {}).get("enabled", False))
    initial_capital = 500_000.0

    import live_config as _lc
    _cap_saved = _lc.MAX_ORDER_NOTIONAL
    if order_cap is not None:
        _lc.MAX_ORDER_NOTIONAL = float(order_cap)

    batches = load_selection_batches(cfg)
    ov, detail = load_backtest_truth(cfg)
    trading_days = list(ov.index)

    all_syms = sorted({s for _, b in batches for s in b["symbol"].astype(str)})
    start = min(d for d, _ in batches)
    end = trading_days[-1].strftime("%Y-%m-%d")
    op, cl, pc = build_price_panel(all_syms, start, end)

    # 选股日 T → 执行日（次一交易日，与引擎一致）
    def exec_day(T):
        Tts = pd.Timestamp(T)
        future = [d for d in trading_days if d > Tts]
        return future[0] if future else None

    base_hist = pd.Series(ov["base_return"].astype(float).values, index=ov.index)
    buy_rate, sell_rate, min_buy, min_sell = cost_rates(cfg)

    layer1, layer2 = [], []
    account = {"cash": initial_capital, "holdings": {}}  # holdings: {sym: shares}

    def mtm(day):
        v = account["cash"]
        for s, sh in account["holdings"].items():
            px = cl.at[day, s] if (day in cl.index and s in cl.columns) else np.nan
            if not np.isnan(px):
                v += sh * px
        return v

    rebalance_events = []
    for T, batch in batches:
        e = exec_day(T)
        if e is not None:
            rebalance_events.append((T, e, batch))

    # ── 逐日推进 ──
    # live 真实行为：export_target 每个交易日都跑 → invested_pct = stock_pct×exposure_T 每日更新，
    # 选股(picks)双周才换。故账户**逐日**按当日 exposure 调仓（overlay 是日频风控层），
    # 选股事件日额外记录 Layer 1/2 的选股决策对账。overlay 的 exposure 直接取回测 overlay_results
    # 的 exposure 列（Layer 1 已证其 == compute_live_exposure，逐日重算等价但慢）。
    daily_rows = []
    first_exec = rebalance_events[0][1]
    prev_pv = initial_capital
    pick_day = {e: (T, batch) for (T, e, batch) in rebalance_events}
    current_batch, current_syms = None, []
    current_exp = None  # 选股日锁定的 exposure（"rebalance" 频率用）

    for day in trading_days:
        if day < first_exec:
            continue
        is_pick_day = day in pick_day
        if is_pick_day:
            T, current_batch = pick_day[day]
            current_syms = [str(s) for s in current_batch["symbol"]]
            current_exp = float(ov.at[day, "exposure"]) if (overlay_on and day in ov.index) else None
        if current_batch is None:
            continue

        # 开盘总资产（持仓按开盘价 MtM）
        total_asset = account["cash"]
        for s, sh in account["holdings"].items():
            o = op.at[day, s] if s in op.columns and day in op.index else np.nan
            o = o if not np.isnan(o) else (cl.at[day, s] if s in cl.columns else np.nan)
            if not np.isnan(o):
                total_asset += sh * o

        # 当日 overlay 仓位：按 cadence 决定跟随频率
        if not overlay_on or exposure_cadence == "off":
            exp_today = None
        elif exposure_cadence == "rebalance":
            exp_today = current_exp
        else:  # daily
            exp_today = float(ov.at[day, "exposure"]) if day in ov.index else current_exp
        target_eng = et.build_target(cfg, current_batch, exposure=exp_today, normalizer=lambda s: s)

        # 合成行情 → qmt plan_orders（engine 格式 symbol 全程一致）
        union = set(account["holdings"]) | set(current_syms)
        quotes, details = {}, {}
        for s in union:
            o = op.at[day, s] if s in op.columns and day in op.index else np.nan
            if np.isnan(o) or o <= 0:
                continue  # 无开盘价 → 停牌，plan_orders 视为不可交易
            quotes[s] = {"lastPrice": float(o)}
            prevc = pc.at[day, s] if s in pc.columns and day in pc.index else np.nan
            up, down = _get_limit_prices(s, day, float(prevc) if not np.isnan(prevc) else np.nan)
            details[s] = {
                "UpStopPrice": 0.0 if pd.isna(up) else float(up),
                "DownStopPrice": 0.0 if pd.isna(down) else float(down),
            }
        positions = {s: {"volume": int(sh), "can_use_volume": int(sh)}
                     for s, sh in account["holdings"].items() if sh > 0}
        orders, blocked, warnings = qmt_rb.plan_orders(
            target_eng, total_asset, account["cash"], positions, quotes, details
        )

        # 成交：股数用 qmt 计划，成交价用开盘价（deal price），成本用 cfg trading
        for o in orders:
            s, sh = o["symbol"], int(o["shares"])
            px = float(op.at[day, s]) if s in op.columns else float(o["limit_price"])
            notional = sh * px
            if o["side"] == "sell":
                fee = max(notional * sell_rate, min_sell)
                account["cash"] += notional - fee
                account["holdings"][s] = account["holdings"].get(s, 0) - sh
                if account["holdings"][s] <= 0:
                    account["holdings"].pop(s, None)
            else:
                fee = max(notional * buy_rate, min_buy)
                account["cash"] -= notional + fee
                account["holdings"][s] = account["holdings"].get(s, 0) + sh

        # ── 选股事件日：记录 Layer 1（持仓决策）+ Layer 2（成交对账）──
        if is_pick_day:
            exposure_live = (
                et.compute_live_exposure(cfg, base_hist[base_hist.index < day])
                if overlay_on else None
            )
            target_live = et.build_target(cfg, current_batch, exposure=exposure_live, normalizer=et._normalize)
            picks_live = {p["symbol"] for p in target_live["positions"]}
            picks_sel = {et._normalize(s) for s in current_syms}
            invested_expected = round(stock_pct * (exp_today if exp_today is not None else 1.0), 6)
            layer1.append({
                "select_date": T, "exec_date": day.strftime("%Y-%m-%d"),
                "picks_match": picks_live == picks_sel,
                "n_picks": len(picks_live),
                "invested_pct_live": target_live["invested_pct"],
                "invested_pct_expected": invested_expected,
                "invested_pct_err": abs(target_live["invested_pct"] - invested_expected),
                "exposure_bt": round(float(exp_today), 6) if exp_today is not None else 1.0,
            })

            pv_open = total_asset
            tgt_w = {p["symbol"]: p["weight"] * target_eng["invested_pct"] for p in target_eng["positions"]}
            post_val = {s: sh * float(op.at[day, s]) for s, sh in account["holdings"].items()
                        if s in op.columns and not np.isnan(op.at[day, s])}
            max_w_drift = 0.0
            for s in set(tgt_w) | set(post_val):
                w_now = post_val.get(s, 0.0) / pv_open if pv_open > 0 else 0.0
                max_w_drift = max(max_w_drift, abs(w_now - tgt_w.get(s, 0.0)))
            row = {
                "exec_date": day.strftime("%Y-%m-%d"),
                "n_orders": len(orders),
                "sells": sum(1 for o in orders if o["side"] == "sell"),
                "buys": sum(1 for o in orders if o["side"] == "buy"),
                "blocked_qmt": len(blocked),
                "turnover_qmt": round(qmt_rb.summarize_turnover(orders, total_asset), 4),
                "max_weight_drift": round(max_w_drift, 5),
                "n_warnings": len(warnings),
            }
            if detail is not None and day in detail.index:
                drow = detail.loc[day]
                row["blocked_bt"] = int(drow.get("blocked_sell_count", 0)) + \
                    int(drow.get("blocked_buy_count", 0)) + int(drow.get("t1_locked_count", 0))
            layer2.append(row)

        pv = mtm(day)
        daily_rows.append({"date": day, "live_return": pv / prev_pv - 1.0 if prev_pv > 0 else 0.0})
        prev_pv = pv

    # ── Layer 3：日收益对账 ──
    # 模拟账户已按 overlay 缩仓（invested_pct 含 exposure），故对标回测 overlay_return；
    # base_return 仅作旁注（纯股票腿、未缩仓）。
    live = pd.DataFrame(daily_rows).set_index("date")["live_return"]
    ref = pd.Series(ov["overlay_return"].astype(float).values, index=ov.index).reindex(live.index).fillna(0.0)
    base = base_hist.reindex(live.index).fillna(0.0)
    diff = live - ref
    corr = float(np.corrcoef(live.values, ref.values)[0, 1]) if len(live) > 2 else float("nan")
    worst = diff.abs().sort_values(ascending=False).head(5)
    layer3 = {
        "days": int(len(live)),
        "reference": "overlay_return",
        "corr_live_vs_ref": round(corr, 5),
        "tracking_error_daily": round(float(diff.std()), 6),
        "tracking_error_annual": round(float(diff.std() * np.sqrt(252)), 5),
        "live_cum": round(float((1 + live).prod() - 1), 4),
        "ref_cum": round(float((1 + ref).prod() - 1), 4),
        "base_cum_noverlay": round(float((1 + base).prod() - 1), 4),
        "cum_gap": round(float((1 + live).prod() - (1 + ref).prod()), 4),
        "worst_abs_diff_days": [
            {"date": d.strftime("%Y-%m-%d"), "abs_diff": round(float(v), 5),
             "live": round(float(live.loc[d]), 5), "ref": round(float(ref.loc[d]), 5)}
            for d, v in worst.items()
        ],
    }

    _lc.MAX_ORDER_NOTIONAL = _cap_saved  # 还原单笔上限，避免影响后续调用

    return {
        "config": name, "overlay_enabled": overlay_on, "stock_pct": stock_pct,
        "exposure_cadence": exposure_cadence,
        "order_cap": (order_cap if order_cap is not None else _cap_saved),
        "n_rebalances": len(layer1),
        "layer1": pd.DataFrame(layer1), "layer2": pd.DataFrame(layer2), "layer3": layer3,
    }


# ── 报告 ────────────────────────────────────────────────────
def print_report(res):
    l1, l2, l3 = res["layer1"], res["layer2"], res["layer3"]
    bar = "=" * 70
    _log(f"\n{bar}\n  回测 ↔ 模拟盘 一致性对账：{res['config']}")
    _log(f"  overlay={'启用' if res['overlay_enabled'] else '关闭'}  stock_pct={res['stock_pct']}  "
         f"调仓事件={res['n_rebalances']}\n{bar}")

    _log("\n── Layer 1 目标持仓一致 ──")
    pm = int(l1["picks_match"].sum())
    _log(f"  选股一致(picks_match): {pm}/{len(l1)}  "
         f"({'全部一致' if pm == len(l1) else '存在不一致!'})")
    max_inv_err = float(l1["invested_pct_err"].max())
    _log(f"  invested_pct 复刻误差(vs 回测 stock_pct×exposure): max={max_inv_err:.2e}  "
         f"({'吻合' if max_inv_err < 1e-6 else '偏差!'})")
    _log(f"  invested_pct 实际范围: [{l1['invested_pct_live'].min():.3f}, "
         f"{l1['invested_pct_live'].max():.3f}]  均值={l1['invested_pct_live'].mean():.3f}")

    _log("\n── Layer 2 下单/成交一致（选股事件日）──")
    _log(f"  qmt 计划订单合计: 卖{int(l2['sells'].sum())} 买{int(l2['buys'].sum())}  "
         f"qmt-blocked合计={int(l2['blocked_qmt'].sum())}(停牌/涨跌停/T+1)")
    if "blocked_bt" in l2.columns:
        _log(f"  注: 回测引擎用不同口径记不可交易(cash_slot/missing), 引擎 blocked_*合计="
             f"{int(l2['blocked_bt'].sum())}, 与 qmt 口径不可直接相等")
    _log(f"  成交后权重对目标最大漂移: max={l2['max_weight_drift'].max():.4f}  "
         f"中位={l2['max_weight_drift'].median():.4f}  (整手取整 + 单笔上限导致)")
    _log(f"  qmt 平均换手/次: {l2['turnover_qmt'].mean():.3f}  现金不足/超单笔上限告警合计={int(l2['n_warnings'].sum())}")

    _log("\n── Layer 3 逐日盈亏一致 ──")
    _log(f"  对账天数: {l3['days']}  (live 已按 overlay 缩仓, 对标 overlay_return)")
    _log(f"  live 日收益 vs 回测 overlay_return  相关系数: {l3['corr_live_vs_ref']}")
    _log(f"  日跟踪误差: {l3['tracking_error_daily']}  (年化 {l3['tracking_error_annual']})")
    _log(f"  累计收益  live={l3['live_cum']:+.2%}  overlay_return={l3['ref_cum']:+.2%}  缺口={l3['cum_gap']:+.2%}")
    _log(f"  (旁注: 未缩仓 base_return 累计={l3['base_cum_noverlay']:+.2%})")
    _log("  偏离最大的 5 天:")
    for r in l3["worst_abs_diff_days"]:
        _log(f"    {r['date']}  |live-ref|={r['abs_diff']}  live={r['live']}  ref={r['ref']}")
    _log(f"\n{bar}")


def save_outputs(res):
    out_dir = PROJECT_ROOT / "results" / "reconcile" / res["config"]
    out_dir.mkdir(parents=True, exist_ok=True)
    res["layer1"].to_csv(out_dir / "layer1_targets.csv", index=False)
    res["layer2"].to_csv(out_dir / "layer2_orders.csv", index=False)
    summary = {
        "config": res["config"], "overlay_enabled": res["overlay_enabled"],
        "n_rebalances": res["n_rebalances"],
        "layer1": {
            "picks_match_rate": float(res["layer1"]["picks_match"].mean()),
            "invested_pct_max_err": float(res["layer1"]["invested_pct_err"].max()),
            "invested_pct_range": [float(res["layer1"]["invested_pct_live"].min()),
                                   float(res["layer1"]["invested_pct_live"].max())],
        },
        "layer2": {
            "blocked_qmt_total": int(res["layer2"]["blocked_qmt"].sum()),
            "max_weight_drift": float(res["layer2"]["max_weight_drift"].max()),
            "avg_turnover": float(res["layer2"]["turnover_qmt"].mean()),
        },
        "layer3": res["layer3"],
    }
    with open(out_dir / "reconcile_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _log(f"  [OK] 明细已写: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="回测 ↔ 模拟盘 三层一致性对账（离线重放）")
    parser.add_argument("--config", "-c",
                        default="config/models/alpha158_momentum_volume_k6.yaml",
                        help="策略 YAML（相对 qlib_quant 根目录）")
    parser.add_argument("--exposure-cadence", choices=["daily", "rebalance", "off"],
                        default="daily", help="overlay 仓位 live 跟随频率（默认 daily）")
    parser.add_argument("--order-cap", type=float, default=None,
                        help="覆盖 live_config.MAX_ORDER_NOTIONAL 单笔上限（元）")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_predictive_config(str(config_path))
    res = reconcile(cfg, exposure_cadence=args.exposure_cadence, order_cap=args.order_cap)
    print_report(res)
    save_outputs(res)


if __name__ == "__main__":
    main()
