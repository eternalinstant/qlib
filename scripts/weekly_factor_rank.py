#!/usr/bin/env python3
"""
周频单因子真实口径排行

复用 single_factor_backtest 的真实执行口径，固定 week 调仓，
输出完整 CSV 排行和一份可读的 Markdown 摘要。
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from config.config import CONFIG
from scripts.factor_scan import get_all_factors
from scripts.single_factor_backtest import (
    ScanConfig,
    load_base_data,
    load_factor_batch,
    single_factor_backtest,
)


PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
WINDOW_FUNCS = {"Mean", "Std", "EMA", "Ref", "Slope", "Min", "Max"}
INT_LITERAL_RE = re.compile(r"^\d+$")


def _resolve_end_date(value: str) -> str:
    if not value or value == "auto":
        return pd.Timestamp.today().strftime("%Y-%m-%d")
    return value


def _build_scan_config(args: argparse.Namespace) -> ScanConfig:
    factor_names = [x.strip() for x in args.factors.split(",") if x.strip()] or None
    return ScanConfig(
        start_date=args.start,
        end_date=_resolve_end_date(args.end),
        freq="week",
        topk=args.topk,
        buffer=args.buffer,
        sticky=args.sticky,
        churn_limit=args.churn_limit,
        margin_stable=not args.disable_margin_stable,
        score_smoothing_days=args.score_smoothing_days,
        entry_rank=args.entry_rank,
        exit_rank=args.exit_rank,
        entry_persist_days=args.entry_persist_days,
        exit_persist_days=args.exit_persist_days,
        min_hold_days=args.min_hold_days,
        stock_pct=args.stock_pct,
        initial_capital=args.initial_capital,
        universe=args.universe,
        min_market_cap=args.min_market_cap,
        exclude_st=not args.keep_st,
        exclude_new_days=args.exclude_new_days,
        buy_commission_rate=args.buy_commission_rate,
        sell_commission_rate=args.sell_commission_rate,
        sell_stamp_tax_rate=args.sell_stamp_tax_rate,
        slippage_bps=args.slippage_bps,
        impact_bps=args.impact_bps,
        factor_names=factor_names,
        output_tag=args.output_tag.strip(),
    )


def parse_args() -> tuple[ScanConfig, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=_resolve_end_date(CONFIG.get("end_date", "auto")))
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--buffer", type=int, default=20)
    parser.add_argument("--sticky", type=int, default=5)
    parser.add_argument("--churn-limit", type=int, default=2)
    parser.add_argument("--score-smoothing-days", type=int, default=1)
    parser.add_argument("--entry-rank", type=int)
    parser.add_argument("--exit-rank", type=int)
    parser.add_argument("--entry-persist-days", type=int, default=1)
    parser.add_argument("--exit-persist-days", type=int, default=1)
    parser.add_argument("--min-hold-days", type=int, default=0)
    parser.add_argument("--stock-pct", type=float, default=0.88)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--min-market-cap", type=float, default=50.0)
    parser.add_argument("--exclude-new-days", type=int, default=120)
    parser.add_argument("--universe", default="all", choices=["all", "csi300"])
    parser.add_argument("--buy-commission-rate", type=float, default=0.0003)
    parser.add_argument("--sell-commission-rate", type=float, default=0.0003)
    parser.add_argument("--sell-stamp-tax-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--impact-bps", type=float, default=5.0)
    parser.add_argument("--disable-margin-stable", action="store_true")
    parser.add_argument("--keep-st", action="store_true")
    parser.add_argument("--factors", default="")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--window-scale", type=int, default=5)
    args = parser.parse_args()
    return _build_scan_config(args), args.top_n, args.window_scale


def _split_top_level_args(text: str) -> list[str]:
    parts = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def _transform_expression(expr: str, window_scale: int) -> str:
    if window_scale <= 1:
        return expr

    def walk(text: str) -> str:
        out = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch.isalpha() or ch == "_":
                j = i + 1
                while j < n and (text[j].isalnum() or text[j] == "_"):
                    j += 1
                name = text[i:j]
                if j < n and text[j] == "(":
                    depth = 1
                    k = j + 1
                    while k < n and depth > 0:
                        if text[k] == "(":
                            depth += 1
                        elif text[k] == ")":
                            depth -= 1
                        k += 1
                    inner = text[j + 1 : k - 1]
                    args = [walk(arg) for arg in _split_top_level_args(inner)]
                    if name in WINDOW_FUNCS and len(args) >= 2 and INT_LITERAL_RE.fullmatch(args[1]):
                        args[1] = str(int(args[1]) * window_scale)
                    out.append(f"{name}(" + ", ".join(args) + ")")
                    i = k
                    continue
            out.append(ch)
            i += 1
        return "".join(out)

    return walk(expr)


def _weeklyize_factors(factor_dict: dict[str, str], window_scale: int) -> dict[str, str]:
    return {name: _transform_expression(expr, window_scale) for name, expr in factor_dict.items()}


def _merge_daily_ir_reference(df_results: pd.DataFrame) -> pd.DataFrame:
    ref_path = RESULTS_DIR / "factor_scan_v2.csv"
    if not ref_path.exists():
        return df_results

    ref = pd.read_csv(ref_path)
    keep_cols = ["因子", "IC均值", "IR", "|IR|", "IC胜率", "方向"]
    available_cols = [c for c in keep_cols if c in ref.columns]
    if "因子" not in available_cols:
        return df_results

    ref = ref[available_cols].rename(
        columns={
            "IC均值": "日频IC均值",
            "IR": "日频IR",
            "|IR|": "日频|IR|",
            "IC胜率": "日频IC胜率",
            "方向": "日频IC方向",
        }
    )
    return df_results.merge(ref, on="因子", how="left")


def _scan_weekly_factors(
    cfg: ScanConfig,
    window_scale: int,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    valid, df_ret, all_dates, rebalance_dates, bond_etf_returns, mv_series, mv_floor = load_base_data(
        cfg
    )

    base_factor_dict = get_all_factors()
    if cfg.factor_names:
        factor_set = set(cfg.factor_names)
        base_factor_dict = {k: v for k, v in base_factor_dict.items() if k in factor_set}
        print(f"[INFO] 过滤后因子数: {len(base_factor_dict)}")

    factor_dict = _weeklyize_factors(base_factor_dict, window_scale)
    print(f"[INFO] 因子窗口倍率: x{window_scale}")

    df_all, failed = load_factor_batch(valid, factor_dict, cfg)
    print(f"[INFO] 开始周频回测: {len(df_all.columns)} 个因子")

    results = []
    total = len(df_all.columns)
    for i, col in enumerate(df_all.columns, 1):
        factor_values = df_all[col]

        r_pos = single_factor_backtest(
            factor_values,
            df_ret,
            all_dates,
            rebalance_dates,
            cfg,
            bond_etf_returns,
            mv_series,
            mv_floor,
            negate=False,
        )
        r_neg = single_factor_backtest(
            factor_values,
            df_ret,
            all_dates,
            rebalance_dates,
            cfg,
            bond_etf_returns,
            mv_series,
            mv_floor,
            negate=True,
        )

        if r_pos is None and r_neg is None:
            print(f"[{i:02d}/{total:02d}] {col:30s} SKIP")
            continue

        if r_pos is None:
            best, best_dir = r_neg, "反向"
        elif r_neg is None:
            best, best_dir = r_pos, "正向"
        else:
            best, best_dir = (r_neg, "反向") if r_neg["sharpe"] > r_pos["sharpe"] else (r_pos, "正向")

        row = {
            "因子": col,
            "方向": best_dir,
            "窗口倍率": window_scale,
            "年化收益": best["annual_return"],
            "夏普比率": best["sharpe"],
            "最大回撤": best["max_drawdown"],
            "总收益": best["total_return"],
            "扣费前总收益": best["gross_total_return"],
            "日胜率": best["win_rate"],
            "平均换手": best["avg_turnover"],
            "累计费用": best["total_fee_amount"],
            "总买卖次数": best["sum_turns"],
            "活跃调仓日": best["active_rebalance_days"],
            "原表达式": base_factor_dict[col],
            "周频表达式": factor_dict[col],
        }
        for year in sorted(best["yearly"].keys()):
            row[f"收益_{year}"] = best["yearly"][year]["ret"]
            row[f"夏普_{year}"] = best["yearly"][year]["sharpe"]
        results.append(row)

        print(
            f"[{i:02d}/{total:02d}] {col:30s} {best_dir} "
            f"年化:{best['annual_return']:+.2%} "
            f"夏普:{best['sharpe']:.3f} "
            f"回撤:{best['max_drawdown']:+.2%}"
        )

    df_results = pd.DataFrame(results)
    if df_results.empty:
        return df_results, failed

    df_results = df_results.sort_values(
        ["夏普比率", "年化收益", "最大回撤"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    df_results.insert(0, "排名", range(1, len(df_results) + 1))
    return df_results, failed


def _format_value(column: str, value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int,)) or column in {"排名", "总买卖次数", "活跃调仓日"}:
        return str(int(value))
    if column in {"年化收益", "最大回撤", "总收益", "扣费前总收益", "日胜率", "日频IC均值", "日频IC胜率"}:
        return f"{float(value):+.2%}"
    if column in {"夏普比率", "平均换手", "日频IR", "日频|IR|"}:
        return f"{float(value):.3f}"
    if column == "累计费用":
        return f"{float(value):,.0f}"
    return str(value)


def _frame_to_markdown(df: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = [c for c in columns if c in df.columns]
    if not cols or df.empty:
        return "_无数据_"

    header = "| " + " | ".join(cols) + " |"
    split = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df[cols].iterrows():
        rows.append("| " + " | ".join(_format_value(col, row[col]) for col in cols) + " |")
    return "\n".join([header, split, *rows])


def _build_markdown_report(
    df_results: pd.DataFrame,
    cfg: ScanConfig,
    csv_path: Path,
    failed: list[tuple[str, str]],
    top_n: int,
    window_scale: int,
) -> str:
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_sharpe = df_results.head(top_n)
    positive = df_results[df_results["夏普比率"] > 0].head(top_n)

    lines = [
        "# 周频因子真实口径排行",
        "",
        f"生成时间：{now_text}",
        "",
        "## 扫描口径",
        "",
        f"- 区间：`{cfg.start_date} ~ {cfg.end_date}`",
        f"- 频率：`{cfg.freq}`",
        f"- `topk={cfg.topk}`",
        f"- `buffer={cfg.buffer}`",
        f"- `sticky={cfg.sticky}`",
        f"- `churn_limit={cfg.churn_limit}`",
        f"- `score_smoothing_days={cfg.score_smoothing_days}`",
        f"- `window_scale={window_scale}`",
        f"- `entry_rank={cfg.entry_rank}`",
        f"- `exit_rank={cfg.exit_rank}`",
        f"- `entry_persist_days={cfg.entry_persist_days}`",
        f"- `exit_persist_days={cfg.exit_persist_days}`",
        f"- `min_hold_days={cfg.min_hold_days}`",
        f"- 股票池：`{cfg.universe}`",
        f"- 最小市值：`{cfg.min_market_cap}` 亿元",
        f"- 排除新股：`{cfg.exclude_new_days}` 天",
        f"- 排除 ST：`{cfg.exclude_st}`",
        f"- 股票仓位：`{cfg.stock_pct:.2f}`",
        "",
        f"完整结果文件：[{csv_path.name}]({csv_path})",
        "",
        "## Top By Sharpe",
        "",
        _frame_to_markdown(
            top_sharpe,
            [
                "排名",
                "因子",
                "方向",
                "年化收益",
                "夏普比率",
                "最大回撤",
                "总收益",
                "平均换手",
                "累计费用",
                "日频IR",
                "周频表达式",
            ],
        ),
        "",
        "## Positive Sharpe Factors",
        "",
        _frame_to_markdown(
            positive,
            [
                "排名",
                "因子",
                "方向",
                "年化收益",
                "夏普比率",
                "最大回撤",
                "总收益",
                "平均换手",
                "累计费用",
            ],
        ),
    ]

    if failed:
        lines.extend(
            [
                "",
                "## Failed Factors",
                "",
                f"- 失败因子数：`{len(failed)}`",
            ]
        )

    return "\n".join(lines) + "\n"


def _output_base_name(cfg: ScanConfig) -> str:
    base = (
        f"weekly_factor_rank_{cfg.start_date[:4]}_{cfg.end_date[:4]}_"
        f"k{cfg.topk}_b{cfg.buffer}_c{cfg.churn_limit}"
        f"_s{cfg.score_smoothing_days}"
    )
    if cfg.output_tag:
        base += f"_{cfg.output_tag}"
    return base


def main() -> None:
    cfg, top_n, window_scale = parse_args()
    start_ts = time.time()

    print("=" * 88)
    print("  周频单因子真实口径排行")
    print("=" * 88)
    print(
        f"区间={cfg.start_date} ~ {cfg.end_date} | freq={cfg.freq} | "
        f"topk={cfg.topk} | buffer={cfg.buffer} | sticky={cfg.sticky} | "
        f"churn_limit={cfg.churn_limit} | stock_pct={cfg.stock_pct:.2f} | "
        f"window_scale={window_scale}"
    )

    df_results, failed = _scan_weekly_factors(cfg, window_scale)
    if df_results.empty:
        raise RuntimeError("周频扫描结果为空")

    df_results = _merge_daily_ir_reference(df_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_name = _output_base_name(cfg)
    if window_scale != 1:
        base_name += f"_ws{window_scale}"
    csv_path = RESULTS_DIR / f"{base_name}.csv"
    md_path = RESULTS_DIR / f"{base_name}.md"

    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    md_path.write_text(
        _build_markdown_report(df_results, cfg, csv_path, failed, top_n, window_scale),
        encoding="utf-8",
    )

    display_cols = [
        "排名",
        "因子",
        "方向",
        "年化收益",
        "夏普比率",
        "最大回撤",
        "总收益",
        "平均换手",
        "累计费用",
    ]
    print("\nTop 15:")
    print(df_results[display_cols].head(15).to_string(index=False))
    print(f"\nCSV: {csv_path}")
    print(f"MD : {md_path}")
    print(f"耗时: {(time.time() - start_ts) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
