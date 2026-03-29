#!/usr/bin/env python3
"""
审计 Qlib 价格字段（open/high/low/close）与 Tushare 原始日线的一致性。

默认从最近若干个调仓日的 top15 选股里抽样，输出：
1. 逐行明细 CSV
2. 摘要 Markdown
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import tushare as ts

from core.qlib_init import init_qlib, load_features_safe


DEFAULT_STRATEGIES = [
    "top15_amp_day",
    "top15_core_day",
    "top15_core_trend",
    "top15_core_v2",
    "top15_robust_ma5_fixed",
    "top15_robust_ema5_fixed",
]


def _selection_path(strategy: str) -> Path:
    return Path("data/selections") / f"{strategy}.csv"


def collect_sample_symbols(
    strategies: Iterable[str],
    lookback_rebalances: int,
    sample_size: int,
) -> List[str]:
    counts = Counter()
    for strategy in strategies:
        path = _selection_path(strategy)
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        recent_dates = sorted(df["date"].drop_duplicates())[-lookback_rebalances:]
        recent = df[df["date"].isin(recent_dates)]
        counts.update(recent["symbol"].tolist())
    return [symbol for symbol, _ in counts.most_common(sample_size)]


def infer_date_window(strategies: Iterable[str], days: int) -> tuple[str, str]:
    max_date = None
    for strategy in strategies:
        path = _selection_path(strategy)
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["date"], parse_dates=["date"])
        strategy_max = df["date"].max()
        if max_date is None or strategy_max > max_date:
            max_date = strategy_max

    if max_date is None:
        raise FileNotFoundError("找不到任何选股文件，无法推断审计区间")

    start_date = (pd.Timestamp(max_date) - pd.Timedelta(days=days * 2)).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(max_date).strftime("%Y-%m-%d")
    return start_date, end_date


def load_raw_quotes(ts_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    pro = ts.pro_api()
    if hasattr(pro, "_DataApi__timeout"):
        pro._DataApi__timeout = 30

    start_compact = pd.Timestamp(start_date).strftime("%Y%m%d")
    end_compact = pd.Timestamp(end_date).strftime("%Y%m%d")
    parts = []
    for code in ts_codes:
        df = pro.daily(
            ts_code=code,
            start_date=start_compact,
            end_date=end_compact,
            fields="ts_code,trade_date,open,high,low,close,pre_close",
        )
        if df is not None and len(df) > 0:
            parts.append(df)

    if not parts:
        return pd.DataFrame(columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close"])

    raw = pd.concat(parts, ignore_index=True)
    raw["trade_date"] = raw["trade_date"].astype(str)
    return raw


def load_qlib_quotes(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    qlib_symbols = [symbol[:2].lower() + symbol[2:] for symbol in symbols]

    init_qlib()
    qlib_df = load_features_safe(
        qlib_symbols,
        ["$open", "$high", "$low", "$close"],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    ).reset_index()
    if qlib_df.empty:
        return qlib_df

    qlib_df["symbol"] = qlib_df["instrument"].str[:2].str.upper() + qlib_df["instrument"].str[2:]
    qlib_df["trade_date"] = qlib_df["datetime"].dt.strftime("%Y%m%d")
    qlib_df["ts_code"] = qlib_df["instrument"].str[2:] + "." + qlib_df["instrument"].str[:2].str.upper()
    return qlib_df


def build_audit_frame(symbols: List[str], raw: pd.DataFrame, qlib_df: pd.DataFrame) -> pd.DataFrame:
    merged = qlib_df.merge(raw, on=["ts_code", "trade_date"], how="outer", suffixes=("_qlib", "_ts"))
    if "symbol" not in merged.columns:
        merged["symbol"] = merged["ts_code"].str.split(".").str[1].fillna("").str.upper() + merged["ts_code"].str.split(".").str[0].fillna("")

    for col in ["open", "high", "low", "close"]:
        q_col = f"${col}"
        if q_col not in merged.columns:
            merged[q_col] = pd.NA
        merged[f"{col}_factor"] = merged[q_col] / merged[col]

    merged["invalid_high_low"] = (
        (merged["$high"] < merged[["$open", "$close"]].max(axis=1))
        | (merged["$low"] > merged[["$open", "$close"]].min(axis=1))
        | (merged["$high"] < merged["$low"])
    )
    merged["factor_gap_open_close"] = (merged["open_factor"] - merged["close_factor"]).abs()
    merged["factor_gap_high_close"] = (merged["high_factor"] - merged["close_factor"]).abs()
    merged["factor_gap_low_close"] = (merged["low_factor"] - merged["close_factor"]).abs()
    merged = merged[merged["symbol"].isin(symbols)].copy()
    return merged.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def summarize_audit(audit_df: pd.DataFrame, expected_rows: int) -> dict:
    matched = audit_df[audit_df["close"].notna()].copy()
    summary = {
        "expected_rows": expected_rows,
        "qlib_rows": int(audit_df[["$open", "$high", "$low", "$close"]].notna().any(axis=1).sum()),
        "raw_rows": int(audit_df["close"].notna().sum()),
        "merged_rows": int(len(audit_df)),
        "raw_match_ratio": float(audit_df["close"].notna().mean()) if len(audit_df) else 0.0,
        "qlib_open_non_null_ratio": float(audit_df["$open"].notna().mean()) if len(audit_df) else 0.0,
        "qlib_high_non_null_ratio": float(audit_df["$high"].notna().mean()) if len(audit_df) else 0.0,
        "qlib_low_non_null_ratio": float(audit_df["$low"].notna().mean()) if len(audit_df) else 0.0,
        "qlib_close_non_null_ratio": float(audit_df["$close"].notna().mean()) if len(audit_df) else 0.0,
        "invalid_high_low_ratio": float(audit_df["invalid_high_low"].mean()) if len(audit_df) else 0.0,
    }

    for key in ["factor_gap_open_close", "factor_gap_high_close", "factor_gap_low_close"]:
        series = matched[key].dropna()
        summary[f"{key}_median"] = float(series.median()) if not series.empty else None
        summary[f"{key}_p95"] = float(series.quantile(0.95)) if not series.empty else None

    return summary


def write_outputs(
    strategies: List[str],
    symbols: List[str],
    start_date: str,
    end_date: str,
    audit_df: pd.DataFrame,
    summary: dict,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"price_field_audit_{stamp}.csv"
    md_path = output_dir / f"price_field_audit_{stamp}.md"

    audit_df.to_csv(csv_path, index=False)

    bad_rows = audit_df[audit_df["invalid_high_low"]].head(10)
    lines = [
        f"# 价格字段审计 {stamp}",
        "",
        "## 范围",
        f"- 策略样本：{', '.join(strategies)}",
        f"- 抽样标的数：{len(symbols)}",
        f"- 标的列表：{', '.join(symbols)}",
        f"- 区间：{start_date} ~ {end_date}",
        "",
        "## 摘要",
        f"- 期望行数：`{summary['expected_rows']}`",
        f"- 合并后行数：`{summary['merged_rows']}`",
        f"- Qlib `$open/$high/$low/$close` 非空比例：`{summary['qlib_open_non_null_ratio']:.2%}` / `{summary['qlib_high_non_null_ratio']:.2%}` / `{summary['qlib_low_non_null_ratio']:.2%}` / `{summary['qlib_close_non_null_ratio']:.2%}`",
        f"- 能与 Tushare 原始日线匹配的比例：`{summary['raw_match_ratio']:.2%}`",
        f"- `high/low` 逻辑异常比例：`{summary['invalid_high_low_ratio']:.2%}`",
        f"- `|open_factor - close_factor|` 中位数 / P95：`{summary['factor_gap_open_close_median']}` / `{summary['factor_gap_open_close_p95']}`",
        f"- `|high_factor - close_factor|` 中位数 / P95：`{summary['factor_gap_high_close_median']}` / `{summary['factor_gap_high_close_p95']}`",
        f"- `|low_factor - close_factor|` 中位数 / P95：`{summary['factor_gap_low_close_median']}` / `{summary['factor_gap_low_close_p95']}`",
        "",
        "## 结论",
        "- 如果 `close` 大面积为空，或 `high < max(open, close)` / `low > min(open, close)` 的比例明显非零，就不能把 Qlib 的 OHLC 当成可靠的成交约束输入。",
        "- 这类结果下，涨跌停可成交约束不应静默启用，必须接入独立的原始日线数据源。",
        "",
        "## 异常样本",
    ]

    if bad_rows.empty:
        lines.append("- 没有发现 `high/low` 逻辑异常样本。")
    else:
        for row in bad_rows.to_dict("records"):
            lines.append(
                f"- `{row['symbol']} {row['trade_date']}`: "
                f"`open={row.get('$open')}` "
                f"`high={row.get('$high')}` "
                f"`low={row.get('$low')}` "
                f"`close={row.get('$close')}` "
                f"`raw_open={row.get('open')}` `raw_close={row.get('close')}`"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(description="审计 Qlib 价格字段质量")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES), help="策略列表，逗号分隔")
    parser.add_argument("--lookback-rebalances", type=int, default=20, help="回看最近多少个调仓日来抽样")
    parser.add_argument("--sample-size", type=int, default=20, help="抽样标的数")
    parser.add_argument("--days", type=int, default=5, help="审计最近多少个自然日窗口")
    parser.add_argument("--start-date", default=None, help="显式开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="显式结束日期 YYYY-MM-DD")
    parser.add_argument("--output-dir", default="results", help="输出目录")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    symbols = collect_sample_symbols(strategies, args.lookback_rebalances, args.sample_size)
    if not symbols:
        raise FileNotFoundError("没有抽到任何样本标的，请先生成选股结果")

    if args.start_date and args.end_date:
        start_date, end_date = args.start_date, args.end_date
    else:
        start_date, end_date = infer_date_window(strategies, args.days)

    raw = load_raw_quotes([s[2:] + "." + s[:2] for s in symbols], start_date, end_date)
    qlib_df = load_qlib_quotes(symbols, start_date, end_date)
    audit_df = build_audit_frame(symbols, raw, qlib_df)

    trade_dates = sorted(raw["trade_date"].drop_duplicates()) if not raw.empty else []
    expected_rows = len(symbols) * len(trade_dates)
    summary = summarize_audit(audit_df, expected_rows)
    csv_path, md_path = write_outputs(
        strategies=strategies,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        audit_df=audit_df,
        summary=summary,
        output_dir=Path(args.output_dir),
    )

    print(f"[OK] audit csv: {csv_path}")
    print(f"[OK] audit md: {md_path}")


if __name__ == "__main__":
    main()
