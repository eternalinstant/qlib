#!/usr/bin/env python3
"""
策略 Leaderboard — 扫描已有回测结果，输出可排序/可过滤的策略指标表。

Usage:
    python3 scripts/strategy_leaderboard.py               # 按 sharpe 降序全显
    python3 scripts/strategy_leaderboard.py --top 20      # 前 20
    python3 scripts/strategy_leaderboard.py --sort calmar --min-sharpe 1.0
    python3 scripts/strategy_leaderboard.py --grep push25 --no-save
    python3 scripts/strategy_leaderboard.py --format markdown
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _ROOT / "results"
AGG_METRICS_PATH = RESULTS_DIR / "analysis" / "all_return_series_metrics.csv"
CONFIG_DIRS = [_ROOT / "config" / "models", _ROOT / "config" / "strategies"]
LEADERBOARD_DIR = RESULTS_DIR / "leaderboard"

# ── ANSI colors ───────────────────────────────────────────────────────────────
C_RESET  = "\033[0m"
C_RED    = "\033[91m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN   = "\033[96m"
C_BOLD   = "\033[1m"
C_DIM    = "\033[2m"

# ── known universe suffixes (longest first for greedy match) ──────────────────
_UNIVERSE_TAGS = [
    "historical_csi300", "historical_csi500", "historical_csi800",
    "historical_all", "csi300", "csi500", "csi800", "all",
]

# OOS cutoff: rows before this date are "in-sample"
OOS_START = "2026-01-01"

# ── slug parsing ──────────────────────────────────────────────────────────────

def _parse_slug(fname: str) -> Tuple[Optional[str], Optional[str]]:
    """
    backtest_{slug}_{universe}_{YYYYMMDD_HHMMSS}.csv
    Returns (slug, universe) or (None, None).
    """
    stem = fname
    if stem.endswith(".csv"):
        stem = stem[:-4]
    if not stem.startswith("backtest_"):
        return None, None
    body = stem[len("backtest_"):]

    # strip timestamp suffix _YYYYMMDD_HHMMSS
    ts_pat = re.compile(r"_\d{8}_\d{6}$")
    m = ts_pat.search(body)
    if m:
        body = body[: m.start()]

    # strip universe tag from right
    for tag in _UNIVERSE_TAGS:
        if body.endswith("_" + tag):
            slug = body[: -(len(tag) + 1)]
            return slug, tag

    # no known universe tag — treat whole remainder as slug
    return body, None


# ── metric calculation ────────────────────────────────────────────────────────

def _cagr(pv: pd.Series) -> float:
    if pv.empty or len(pv) < 2:
        return float("nan")
    days = (pv.index[-1] - pv.index[0]).days
    if days <= 0:
        return float("nan")
    terminal = float(pv.iloc[-1])
    if terminal <= 0:
        return -1.0
    return terminal ** (365.0 / days) - 1.0


def _max_drawdown(pv: pd.Series) -> float:
    if pv.empty:
        return float("nan")
    roll_max = pv.cummax()
    dd = pv / roll_max - 1.0
    return float(dd.min())


def _sharpe(rets: pd.Series) -> float:
    if rets.empty or rets.std() == 0:
        return float("nan")
    return float(rets.mean() / rets.std() * np.sqrt(252))


def _drawdown_recovery_days(pv: pd.Series) -> Optional[int]:
    """Days from max-drawdown trough to next all-time high. None if not recovered."""
    if pv.empty:
        return None
    roll_max = pv.cummax()
    dd = pv / roll_max - 1.0
    trough_idx = dd.idxmin()
    peak_before = roll_max.loc[:trough_idx].iloc[-1]
    after = pv.loc[trough_idx:]
    recovered = after[after >= peak_before]
    if recovered.empty:
        return None
    return (recovered.index[0] - trough_idx).days


def compute_metrics_from_daily(path: Path) -> dict:
    """从日收益 CSV 重算完整指标集。"""
    try:
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    except Exception:
        return {}

    if "return" not in df.columns:
        return {}

    rets = df["return"].dropna()
    if rets.empty:
        return {}

    pv = (1 + rets).cumprod()

    # full-period
    full_cagr   = _cagr(pv)
    full_dd     = _max_drawdown(pv)
    full_sharpe = _sharpe(rets)
    full_calmar = full_cagr / abs(full_dd) if full_dd != 0 else float("nan")

    # OOS
    oos_rets = rets[rets.index >= OOS_START]
    if not oos_rets.empty:
        oos_pv     = (1 + oos_rets).cumprod()
        oos_cagr   = _cagr(oos_pv)
        oos_dd     = _max_drawdown(oos_pv)
        oos_sharpe = _sharpe(oos_rets)
        oos_calmar = oos_cagr / abs(oos_dd) if oos_dd != 0 else float("nan")
    else:
        oos_cagr = oos_dd = oos_sharpe = oos_calmar = float("nan")

    # turnover (annualised)
    if "buy_count" in df.columns and "sell_count" in df.columns:
        total_trades = (df["buy_count"].fillna(0) + df["sell_count"].fillna(0)).sum()
        n_days = len(rets)
        annualised_turnover = (total_trades / n_days * 252) if n_days > 0 else float("nan")
    else:
        annualised_turnover = float("nan")

    # cost ratio vs gross return
    if "cost" in df.columns and "gross_return" in df.columns:
        total_cost  = df["cost"].fillna(0).sum()
        total_gross = df["gross_return"].fillna(0).sum()
        cost_ratio  = total_cost / total_gross if total_gross > 0 else float("nan")
    else:
        cost_ratio = float("nan")

    # win rate & profit factor
    win_rate = float((rets > 0).mean())
    pos_rets = rets[rets > 0]
    neg_rets = rets[rets < 0]
    if not pos_rets.empty and not neg_rets.empty:
        profit_factor = float(pos_rets.mean() / abs(neg_rets.mean()))
    else:
        profit_factor = float("nan")

    # drawdown recovery
    dd_recovery = _drawdown_recovery_days(pv)

    return {
        "full_cagr":         full_cagr,
        "full_max_dd":       full_dd,
        "full_sharpe":       full_sharpe,
        "full_calmar":       full_calmar,
        "oos_cagr":          oos_cagr,
        "oos_max_dd":        oos_dd,
        "oos_sharpe":        oos_sharpe,
        "oos_calmar":        oos_calmar,
        "turnover":          annualised_turnover,
        "cost_ratio":        cost_ratio,
        "win_rate":          win_rate,
        "profit_factor":     profit_factor,
        "dd_recovery_days":  dd_recovery,
        "full_start":        str(rets.index[0].date()),
        "full_end":          str(rets.index[-1].date()),
    }


# ── YAML metadata ─────────────────────────────────────────────────────────────

_DERIV_SUFFIXES = ("__holdout", "__valid", "__baseline")


def _find_yaml(slug: str, config_dirs: List[Path]) -> Optional[Path]:
    """优先精确匹配；失败时剥离 __holdout/__valid/__baseline 等衍生后缀再找。"""
    candidates_slugs = [slug]
    for sfx in _DERIV_SUFFIXES:
        if slug.endswith(sfx):
            candidates_slugs.append(slug[: -len(sfx)])
            break
    for s in candidates_slugs:
        for d in config_dirs:
            p = d / f"{s}.yaml"
            if p.exists():
                return p
            for candidate in d.rglob(f"{s}.yaml"):
                return candidate
    return None


def _count_factors(cfg: dict) -> Optional[int]:
    """因子数量统计：
    - 因子策略（factors.{alpha,risk,enhance}）：累加各桶 list 长度
    - 模型策略 alpha158：data.feature_columns 长度
    - 模型策略 hybrid：data.parquet_feature_columns + alpha158_feature_columns 长度
    无法判断时返回 None。
    """
    facs = cfg.get("factors")
    if isinstance(facs, dict):
        total = 0
        seen = False
        for bucket in ("alpha", "risk", "enhance"):
            v = facs.get(bucket)
            if isinstance(v, list):
                total += len(v)
                seen = True
        if seen:
            return total

    data = cfg.get("data") or {}
    # 累加 data 下所有以 *feature_columns 结尾的列表（覆盖 alpha158/hybrid/parquet/qlib 等）
    total = 0
    seen = False
    for k, v in data.items():
        if k.endswith("feature_columns") and isinstance(v, list):
            total += len(v)
            seen = True
    return total if seen else None


def _load_yaml_meta(slug: str, config_dirs: List[Path]) -> dict:
    yaml_path = _find_yaml(slug, config_dirs)
    if yaml_path is None:
        return {}
    try:
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f) or {}
        sel = cfg.get("selection", {})
        return {
            "freq":     sel.get("freq", "?"),
            "topk":     sel.get("topk", "?"),
            "universe": sel.get("universe", "?"),
            "factor_count": _count_factors(cfg),
        }
    except Exception:
        return {}


# ── scanner ───────────────────────────────────────────────────────────────────

def scan_backtest_csvs(results_dir: Path) -> Dict[str, dict]:
    """
    Scan results/ for backtest CSVs.
    Returns {slug: {path, mtime, universe, ...}}, keeping only the most-recent
    file per slug.
    """
    candidates: Dict[str, dict] = {}
    for p in results_dir.glob("backtest_*.csv"):
        slug, universe = _parse_slug(p.name)
        if slug is None:
            continue
        mtime = p.stat().st_mtime
        if slug not in candidates or mtime > candidates[slug]["mtime"]:
            candidates[slug] = {
                "path":     p,
                "mtime":    mtime,
                "universe": universe or "?",
                "filename": p.name,
            }
    return candidates


def load_agg_metrics(agg_path: Path) -> Optional[pd.DataFrame]:
    if not agg_path.exists():
        return None
    try:
        df = pd.read_csv(agg_path)
        return df
    except Exception:
        return None


# ── leaderboard builder ───────────────────────────────────────────────────────

def build_leaderboard(
    results_dir: Path = RESULTS_DIR,
    agg_metrics_path: Path = AGG_METRICS_PATH,
    config_dirs: List[Path] = None,
    stale_days: int = 30,
) -> pd.DataFrame:
    """
    主数据源：all_return_series_metrics.csv（587 个策略）。
    补充数据：results/backtest_*.csv（换手率/成本/胜率等日频指标）。
    """
    if config_dirs is None:
        config_dirs = CONFIG_DIRS

    # ── 1. 主数据源：聚合表 ──────────────────────────────────────────────────
    agg_df = load_agg_metrics(agg_metrics_path)
    if agg_df is None or agg_df.empty:
        # fallback：只有顶级 CSV
        agg_df = None
    else:
        agg_df = agg_df[
            (agg_df["series_kind"] == "backtest") & (agg_df["status"] == "ok")
        ]

    if agg_df is None or agg_df.empty:
        print(f"{C_YELLOW}[WARN] 聚合表为空，退化为扫描顶级 CSV{C_RESET}")
        scanned = scan_backtest_csvs(results_dir)
        if not scanned:
            return pd.DataFrame()
        base_rows = []
        for slug, info in scanned.items():
            m = compute_metrics_from_daily(info["path"])
            if not m:
                continue
            base_rows.append({"strategy_name": slug, **m,
                               "mtime": info["mtime"], "source": "csv"})
        agg_df = pd.DataFrame(base_rows)
    else:
        # 每个 strategy_name 只保留 mtime 最新的一行
        agg_df = (agg_df
                  .sort_values("mtime", ascending=False)
                  .drop_duplicates("strategy_name")
                  .reset_index(drop=True))

    # ── 2. 补充数据：顶级 backtest CSV（换手率 / 成本 / 胜率）──────────────
    csv_extras: Dict[str, dict] = {}
    for info in scan_backtest_csvs(results_dir).values():
        slug = info["path"].name  # 用来对应 strategy_name
        # 解析 slug
        s, _ = _parse_slug(info["path"].name)
        if s is None:
            continue
        m = compute_metrics_from_daily(info["path"])
        if m:
            csv_extras[s] = m

    # ── 3. 组装输出行 ────────────────────────────────────────────────────────
    rows = []
    for _, r in agg_df.iterrows():
        slug = r["strategy_name"]
        mtime_val = float(r.get("mtime", 0) or 0)
        last_run = datetime.fromtimestamp(mtime_val) if mtime_val > 0 else datetime(2000, 1, 1)
        is_stale = (datetime.now() - last_run).days > stale_days

        full_cagr   = r.get("full_cagr",   float("nan"))
        full_dd     = r.get("full_max_dd",  float("nan"))
        full_sharpe = r.get("full_sharpe",  float("nan"))
        full_calmar = r.get("full_calmar",  float("nan"))
        oos_cagr    = r.get("oos_cagr",     float("nan"))
        oos_dd      = r.get("oos_max_dd",   float("nan"))
        oos_sharpe  = r.get("oos_sharpe",   float("nan"))
        oos_calmar  = r.get("oos_calmar",   float("nan"))

        # OOS decay
        if (not np.isnan(float(full_sharpe) if full_sharpe is not None else float("nan"))
                and not np.isnan(float(oos_sharpe) if oos_sharpe is not None else float("nan"))
                and float(full_sharpe) != 0):
            oos_decay = float(oos_sharpe) / float(full_sharpe)
        else:
            oos_decay = float("nan")

        # extras from daily CSV (re-computed with current OOS_START, overrides agg_df OOS)
        extras = csv_extras.get(slug, {})
        turnover     = extras.get("turnover")
        cost_ratio   = extras.get("cost_ratio")
        win_rate     = extras.get("win_rate")
        profit_factor = extras.get("profit_factor")
        dd_recovery  = extras.get("dd_recovery_days")
        if extras.get("oos_cagr") is not None:
            oos_cagr   = extras["oos_cagr"]
            oos_dd     = extras.get("oos_max_dd", oos_dd)
            oos_sharpe = extras["oos_sharpe"]
            oos_calmar = extras.get("oos_calmar", oos_calmar)
            if (not np.isnan(float(full_sharpe)) and not np.isnan(float(oos_sharpe))
                    and float(full_sharpe) != 0):
                oos_decay = float(oos_sharpe) / float(full_sharpe)
            else:
                oos_decay = float("nan")

        # YAML metadata
        yaml_meta = _load_yaml_meta(slug, config_dirs)
        has_config = bool(yaml_meta)
        freq     = yaml_meta.get("freq",     "?")
        topk     = yaml_meta.get("topk",     "?")
        universe = yaml_meta.get("universe", "?")
        factor_count = yaml_meta.get("factor_count")
        # fallback: infer universe from agg path column
        if universe == "?" and "path" in r:
            _, uni = _parse_slug(str(r["path"]))
            if uni:
                universe = uni

        rows.append({
            "strategy":      slug,
            "has_config":    has_config,
            "cagr":          _safe_float(full_cagr),
            "max_dd":        _safe_float(full_dd),
            "sharpe":        _safe_float(full_sharpe),
            "calmar":        _safe_float(full_calmar),
            "oos_cagr":      _safe_float(oos_cagr),
            "oos_max_dd":    _safe_float(oos_dd),
            "oos_sharpe":    _safe_float(oos_sharpe),
            "oos_decay":     oos_decay,
            "turnover":      turnover,
            "cost_ratio":    cost_ratio,
            "win_rate":      win_rate,
            "profit_factor": profit_factor,
            "dd_recovery":   dd_recovery,
            "freq":          freq,
            "topk":          topk,
            "factor_count":  factor_count,
            "universe":      universe,
            "last_run":      last_run.strftime("%Y-%m-%d"),
            "stale":         is_stale,
        })

    return pd.DataFrame(rows)


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


# ── filtering ─────────────────────────────────────────────────────────────────

def apply_filters(
    df: pd.DataFrame,
    min_sharpe: Optional[float] = None,
    max_dd: Optional[float] = None,
    not_stale_days: Optional[int] = None,
    grep: Optional[str] = None,
    has_config: bool = False,
) -> pd.DataFrame:
    if df.empty:
        return df
    if has_config and "has_config" in df.columns:
        df = df[df["has_config"] == True]
    if min_sharpe is not None:
        df = df[df["sharpe"] >= min_sharpe]
    if max_dd is not None:
        df = df[df["max_dd"] >= max_dd]
    if not_stale_days is not None:
        cutoff = datetime.now() - timedelta(days=not_stale_days)
        df = df[pd.to_datetime(df["last_run"]) >= cutoff]
    if grep:
        df = df[df["strategy"].str.contains(grep, case=False, na=False)]
    return df


# ── CLI table rendering ───────────────────────────────────────────────────────

def _fmt_pct(v, decimals=1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  --  "
    return f"{v*100:+.{decimals}f}%"


def _fmt_f(v, decimals=2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  --  "
    return f"{v:.{decimals}f}"


def _fmt_int(v) -> str:
    if v is None or v == "?" or (isinstance(v, float) and np.isnan(v)):
        return " ?"
    try:
        return str(int(v))
    except (ValueError, TypeError):
        return str(v)


def _row_color(row: pd.Series) -> str:
    if row.get("stale", False):
        return C_YELLOW
    sharpe = row.get("sharpe")
    if sharpe is not None and not np.isnan(sharpe) and sharpe < 0:
        return C_RED
    oos_decay = row.get("oos_decay")
    if oos_decay is not None and not np.isnan(oos_decay) and oos_decay < 0.5:
        return C_RED
    return ""


_HEADERS = [
    ("strategy",      "Strategy",       32, "l"),
    ("cagr",          "CAGR",            8, "r"),
    ("max_dd",        "MaxDD",           8, "r"),
    ("sharpe",        "Sharpe",          7, "r"),
    ("calmar",        "Calmar",          7, "r"),
    ("oos_cagr",      "OOS_CAGR",        9, "r"),
    ("oos_sharpe",    "OOS_Sharpe",     10, "r"),
    ("oos_decay",     "OOS_Decay",       9, "r"),
    ("turnover",      "Turnover",        9, "r"),
    ("cost_ratio",    "CostRatio",       9, "r"),
    ("win_rate",      "WinRate",         8, "r"),
    ("freq",          "Freq",            6, "c"),
    ("topk",          "K",               3, "r"),
    ("factor_count",  "Factors",         7, "r"),
    ("universe",      "Univ",            8, "l"),
    ("last_run",      "LastRun",        10, "l"),
    ("stale",         "Stale",           5, "c"),
]


def _cell(val, key: str, width: int, align: str) -> str:
    if key == "cagr":
        s = _fmt_pct(val)
    elif key == "max_dd":
        s = _fmt_pct(val)
    elif key in ("sharpe", "calmar", "oos_decay", "profit_factor"):
        s = _fmt_f(val)
    elif key in ("oos_cagr", "oos_sharpe"):
        s = _fmt_pct(val) if key == "oos_cagr" else _fmt_f(val)
    elif key in ("turnover",):
        s = _fmt_f(val, 1)
    elif key == "cost_ratio":
        s = _fmt_pct(val)
    elif key == "win_rate":
        s = _fmt_pct(val)
    elif key == "stale":
        s = "Y" if val else "N"
    elif key == "topk":
        s = _fmt_int(val)
    elif key == "factor_count":
        s = _fmt_int(val)
    else:
        s = str(val) if val is not None else "--"

    if align == "r":
        return s.rjust(width)
    elif align == "c":
        return s.center(width)
    else:
        return s.ljust(width)[:width]


def render_cli_table(df: pd.DataFrame) -> None:
    if df.empty:
        print(f"{C_YELLOW}[WARN] 没有符合条件的策略{C_RESET}")
        return

    sep = "  "
    header_parts = []
    for key, label, width, align in _HEADERS:
        if align == "r":
            header_parts.append(label.rjust(width))
        elif align == "c":
            header_parts.append(label.center(width))
        else:
            header_parts.append(label.ljust(width)[:width])
    header_line = sep.join(header_parts)
    divider = "─" * len(header_line)

    print(f"\n{C_BOLD}{header_line}{C_RESET}")
    print(C_DIM + divider + C_RESET)

    for _, row in df.iterrows():
        color = _row_color(row)
        cells = [_cell(row.get(key), key, width, align) for key, _, width, align in _HEADERS]
        line = sep.join(cells)
        if color:
            print(f"{color}{line}{C_RESET}")
        else:
            print(line)

    print(C_DIM + divider + C_RESET)
    print(f"  {C_DIM}共 {len(df)} 条  {C_YELLOW}黄=过期{C_RESET} {C_RED}红=Sharpe<0 或 OOS衰减<0.5{C_RESET}")


def render_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无符合条件的策略_\n"

    cols = [
        ("strategy",   "策略名称"),
        ("cagr",       "年化收益"),
        ("max_dd",     "最大回撤"),
        ("sharpe",     "夏普比率"),
        ("calmar",     "卡玛比率"),
        ("oos_cagr",   "OOS年化"),
        ("oos_sharpe", "OOS夏普"),
        ("oos_decay",  "OOS衰减"),
        ("turnover",   "年化换手"),
        ("cost_ratio", "成本占比"),
        ("win_rate",   "日胜率"),
        ("freq",       "调仓频率"),
        ("topk",       "持股数"),
        ("factor_count","因子数"),
        ("universe",   "股票池"),
        ("last_run",   "最近回测"),
        ("stale",      "是否过期"),
    ]

    def fmt(val, key):
        if key in ("cagr", "oos_cagr", "max_dd", "oos_max_dd", "cost_ratio", "win_rate"):
            return _fmt_pct(val).strip()
        if key in ("sharpe", "calmar", "oos_sharpe", "oos_decay", "profit_factor", "turnover"):
            return _fmt_f(val).strip()
        if key == "stale":
            return "是" if val else "否"
        if key in ("topk", "factor_count"):
            return _fmt_int(val)
        return str(val) if val is not None else "--"

    header = "| " + " | ".join(label for _, label in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    lines  = [header, sep]
    for _, row in df.iterrows():
        cells = [fmt(row.get(k), k) for k, _ in cols]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


# ── snapshot persistence ──────────────────────────────────────────────────────

def save_snapshot(df: pd.DataFrame, out_dir: Path = LEADERBOARD_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # historical snapshot
    csv_path = out_dir / f"leaderboard_{ts}.csv"
    df.to_csv(csv_path, index=False)

    # latest.csv (overwrite)
    latest_path = out_dir / "latest.csv"
    df.to_csv(latest_path, index=False)

    # STRATEGIES.md (git-trackable)
    md_path = out_dir / "STRATEGIES.md"
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"# Strategy Leaderboard\n\n_Generated: {ts_str}_\n\n"
    with open(md_path, "w") as f:
        f.write(header)
        f.write(render_markdown_table(df))

    print(f"\n[OK] 快照已保存:")
    print(f"     {csv_path}")
    print(f"     {latest_path}")
    print(f"     {md_path}")


# ── main CLI ──────────────────────────────────────────────────────────────────

SORT_FIELDS = [
    "sharpe", "cagr", "calmar", "max_dd", "oos_sharpe",
    "oos_cagr", "oos_decay", "cost_ratio", "last_run", "turnover",
]


def run_leaderboard(
    sort_by: str = "sharpe",
    top: Optional[int] = None,
    min_sharpe: Optional[float] = None,
    max_dd: Optional[float] = None,
    not_stale_days: Optional[int] = None,
    grep: Optional[str] = None,
    has_config: bool = False,
    no_save: bool = False,
    fmt: str = "cli",
    stale_days: int = 30,
    results_dir: Path = RESULTS_DIR,
    agg_metrics_path: Path = AGG_METRICS_PATH,
    config_dirs: Optional[List[Path]] = None,
) -> pd.DataFrame:
    print(f"{C_CYAN}[INFO] 扫描回测结果...{C_RESET}", end="", flush=True)
    df = build_leaderboard(
        results_dir=results_dir,
        agg_metrics_path=agg_metrics_path,
        config_dirs=config_dirs,
        stale_days=stale_days,
    )
    print(f"\r{C_CYAN}[INFO] 扫描完成，共 {len(df)} 条策略{C_RESET}")

    if df.empty:
        print(f"{C_RED}[ERROR] 未找到回测数据，请先运行 python3 main.py backtest{C_RESET}")
        return df

    # sort
    ascending = sort_by in ("max_dd",)  # max_dd: less negative = better
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending, na_position="last")
    else:
        print(f"{C_YELLOW}[WARN] 未知排序字段 '{sort_by}'，使用 sharpe{C_RESET}")
        df = df.sort_values("sharpe", ascending=False, na_position="last")

    # filter
    df = apply_filters(df, min_sharpe, max_dd, not_stale_days, grep, has_config)
    if df.empty:
        print(f"{C_YELLOW}[WARN] 过滤后无结果{C_RESET}")
        return df

    # top-N
    if top:
        df = df.head(top)

    if fmt == "markdown":
        print(render_markdown_table(df))
    else:
        render_cli_table(df)

    if not no_save:
        save_snapshot(df, LEADERBOARD_DIR)

    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="策略 Leaderboard — 扫描回测结果，排序/过滤/快照",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sort", default="sharpe", choices=SORT_FIELDS,
                   help="排序字段 (默认: sharpe)")
    p.add_argument("--top", type=int, default=None,
                   help="只显示前 N 条")
    p.add_argument("--min-sharpe", type=float, default=None,
                   dest="min_sharpe", help="最低 Sharpe 过滤")
    p.add_argument("--max-dd", type=float, default=None,
                   dest="max_dd",
                   help="最大回撤下限，如 -0.10 表示回撤不超过 -10%%")
    p.add_argument("--not-stale", type=int, default=None,
                   dest="not_stale_days",
                   help="仅显示最近 N 天内有回测的策略")
    p.add_argument("--stale-days", type=int, default=30,
                   dest="stale_days", help="过期阈值天数 (默认: 30)")
    p.add_argument("--grep", default=None,
                   help="按策略名关键字过滤")
    p.add_argument("--has-config", action="store_true",
                   dest="has_config", help="只显示有 YAML 配置文件（可重测）的策略")
    p.add_argument("--no-save", action="store_true",
                   dest="no_save", help="只展示不存快照")
    p.add_argument("--format", choices=["cli", "markdown"], default="cli",
                   dest="fmt", help="输出格式 (默认: cli)")
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_leaderboard(
        sort_by=args.sort,
        top=args.top,
        min_sharpe=args.min_sharpe,
        max_dd=args.max_dd,
        not_stale_days=args.not_stale_days,
        grep=args.grep,
        has_config=args.has_config,
        no_save=args.no_save,
        fmt=args.fmt,
        stale_days=args.stale_days,
    )
