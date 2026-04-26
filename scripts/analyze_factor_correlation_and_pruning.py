#!/usr/bin/env python3
"""Analyze parquet factor correlation and produce pruning suggestions.

The script is read-only for market/factor data. It writes one CSV report with:
- the average cross-sectional Spearman correlation matrix over the latest
  two years available in factor_data.parquet;
- factor Rank IC against 20-day forward return;
- redundant pairs where average Spearman rho is above the configured threshold.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QLIB_DATA_DIR = PROJECT_ROOT / "data" / "qlib_data" / "cn_data"
FACTOR_PATH = QLIB_DATA_DIR / "factor_data.parquet"
OUTPUT_PATH = PROJECT_ROOT / "results" / "analysis" / "factor_correlation_pruning_report.csv"

PARQUET_FACTORS = [
    "ocf_to_ev",
    "ocf_to_ev_size_neut",
    "fcff_to_mv",
    "roe_fina",
    "roe_fina_size_neut",
    "current_ratio_fina",
    "n_cashflow_act",
    "rank_value_profit_core",
    "rank_balance_core",
    "qvf_core_interaction",
    "retained_earnings",
    "retained_earnings_size_neut",
    "ebit_to_mv",
    "ebit_to_mv_size_neut",
    "roa_fina",
    "roa_fina_size_neut",
    "book_to_market",
    "book_to_market_size_neut",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cross-sectional factor correlation and pruning candidates."
    )
    parser.add_argument("--factor-path", default=str(FACTOR_PATH), help="factor_data.parquet path")
    parser.add_argument("--qlib-data-dir", default=str(QLIB_DATA_DIR), help="Qlib provider data directory")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output CSV report path")
    parser.add_argument("--lookback-years", type=int, default=2, help="Use latest N years in parquet data")
    parser.add_argument("--horizon-days", type=int, default=20, help="Forward return horizon for Rank IC")
    parser.add_argument("--corr-threshold", type=float, default=0.7, help="rho threshold for redundant pairs")
    parser.add_argument("--min-cross-section", type=int, default=50, help="Minimum stocks per daily cross-section")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception as exc:
        logging.warning("Unable to inspect parquet schema with pyarrow: %s", exc)
        sample = pd.read_parquet(path, engine="auto").head(0)
        return list(sample.columns)


def load_recent_factor_frame(
    factor_path: Path,
    factor_names: list[str],
    lookback_years: int,
) -> tuple[pd.DataFrame, list[str], list[str], pd.Timestamp, pd.Timestamp]:
    if not factor_path.exists():
        raise FileNotFoundError(f"factor parquet not found: {factor_path}")

    columns = parquet_columns(factor_path)
    required_keys = {"datetime", "instrument"}
    missing_keys = sorted(required_keys - set(columns))
    if missing_keys:
        raise ValueError(f"factor parquet missing required columns: {missing_keys}")

    available = [name for name in factor_names if name in columns]
    missing = [name for name in factor_names if name not in columns]
    if not available:
        raise ValueError("None of the requested factor columns exist in factor_data.parquet")

    read_columns = ["datetime", "instrument"] + available
    logging.info("Reading parquet columns: %d factors + datetime/instrument", len(available))
    df = pd.read_parquet(factor_path, columns=read_columns)
    df["datetime"] = pd.to_datetime(df["datetime"])

    latest_date = pd.Timestamp(df["datetime"].max()).normalize()
    start_date = latest_date - pd.DateOffset(years=int(lookback_years))
    df = df[(df["datetime"] >= start_date) & (df["datetime"] <= latest_date)].copy()
    if df.empty:
        raise ValueError(f"No factor rows in recent {lookback_years} year window ending {latest_date.date()}")

    before = len(df)
    df = df.drop_duplicates(["datetime", "instrument"], keep="last")
    if len(df) != before:
        logging.info("Dropped duplicate datetime/instrument rows: %d", before - len(df))

    df = df.set_index(["datetime", "instrument"]).sort_index()
    df[available] = df[available].apply(pd.to_numeric, errors="coerce")

    logging.info(
        "Factor sample: %s to %s, rows=%d, instruments=%d, dates=%d",
        start_date.date(),
        latest_date.date(),
        len(df),
        df.index.get_level_values("instrument").nunique(),
        df.index.get_level_values("datetime").nunique(),
    )
    if missing:
        logging.warning("Missing factor columns skipped: %s", ", ".join(missing))
    return df, available, missing, start_date, latest_date


def average_cross_sectional_spearman(
    factor_frame: pd.DataFrame,
    factors: list[str],
    min_cross_section: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    n = len(factors)
    corr_sum = np.zeros((n, n), dtype=float)
    valid_count = np.zeros((n, n), dtype=float)
    used_dates = 0

    for _, cross in factor_frame[factors].groupby(level="datetime", sort=True):
        if len(cross) < min_cross_section:
            continue
        ranked = cross.rank(pct=True)
        corr = ranked.corr(method="pearson", min_periods=min_cross_section)
        values = corr.reindex(index=factors, columns=factors).to_numpy(dtype=float)
        mask = ~np.isnan(values)
        corr_sum[mask] += values[mask]
        valid_count[mask] += 1
        used_dates += 1

    if used_dates == 0:
        raise ValueError("No daily cross-section passed min_cross_section")

    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.divide(corr_sum, valid_count, out=np.full_like(corr_sum, np.nan), where=valid_count > 0)
    corr_df = pd.DataFrame(avg, index=factors, columns=factors)
    count_df = pd.DataFrame(valid_count.astype(int), index=factors, columns=factors)
    np.fill_diagonal(corr_df.values, 1.0)
    logging.info("Computed average Spearman correlation from %d daily cross-sections", used_dates)
    return corr_df, count_df, used_dates


def init_qlib(qlib_data_dir: Path) -> None:
    if not qlib_data_dir.exists():
        raise FileNotFoundError(f"Qlib data directory not found: {qlib_data_dir}")

    os.environ["JOBLIB_START_METHOD"] = "fork"
    try:
        import qlib
        from qlib.config import REG_CN
    except ImportError as exc:
        raise ImportError("qlib is required to load $close and compute forward returns") from exc

    qlib.init(provider_uri=str(qlib_data_dir), region=REG_CN)
    try:
        from qlib.config import C

        C.n_jobs = 1
    except Exception:
        pass


def load_forward_returns(
    qlib_data_dir: Path,
    instruments: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    horizon_days: int,
) -> pd.Series:
    logging.info("Loading %d-day forward returns from Qlib $close", horizon_days)
    init_qlib(qlib_data_dir)

    from qlib.data import D

    expr = f"Ref($close, -{int(horizon_days)}) / $close - 1"
    df = D.features(
        instruments,
        [expr],
        start_time=start_date.strftime("%Y-%m-%d"),
        end_time=end_date.strftime("%Y-%m-%d"),
        freq="day",
    )
    df.columns = ["forward_return"]
    if list(df.index.names) == ["instrument", "datetime"]:
        df = df.swaplevel().sort_index()
    else:
        df = df.sort_index()
    returns = df["forward_return"].astype(float)
    logging.info("Forward return rows loaded: %d", len(returns.dropna()))
    return returns


def compute_factor_rank_ic(
    factor_frame: pd.DataFrame,
    forward_returns: pd.Series,
    factors: list[str],
    min_cross_section: int,
) -> pd.DataFrame:
    labeled = factor_frame[factors].join(forward_returns.rename("forward_return"), how="left")
    rows: list[dict[str, object]] = []

    for factor in factors:
        daily_ics: list[float] = []
        sample = labeled[[factor, "forward_return"]].dropna()
        for _, cross in sample.groupby(level="datetime", sort=True):
            if len(cross) < min_cross_section:
                continue
            value = cross[factor].corr(cross["forward_return"], method="spearman")
            if pd.notna(value):
                daily_ics.append(float(value))

        ic_arr = np.asarray(daily_ics, dtype=float)
        mean_ic = float(np.mean(ic_arr)) if len(ic_arr) else np.nan
        std_ic = float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else np.nan
        rows.append(
            {
                "factor": factor,
                "rank_ic_20d": mean_ic,
                "rank_ic_std_20d": std_ic,
                "rank_ic_ir_20d": mean_ic / std_ic if std_ic and std_ic > 0 else np.nan,
                "ic_days": int(len(ic_arr)),
                "ic_samples": int(len(sample)),
            }
        )

    ic_df = pd.DataFrame(rows).set_index("factor")
    logging.info("Computed Rank IC for %d factors", len(ic_df))
    return ic_df


def redundant_pair_rows(
    corr_df: pd.DataFrame,
    count_df: pd.DataFrame,
    ic_df: pd.DataFrame,
    threshold: float,
) -> list[dict[str, object]]:
    factors = list(corr_df.index)
    rows: list[dict[str, object]] = []

    for i, factor_a in enumerate(factors):
        for factor_b in factors[i + 1 :]:
            rho = corr_df.loc[factor_a, factor_b]
            if pd.isna(rho) or float(rho) <= threshold:
                continue

            ic_a = ic_df.loc[factor_a, "rank_ic_20d"] if factor_a in ic_df.index else np.nan
            ic_b = ic_df.loc[factor_b, "rank_ic_20d"] if factor_b in ic_df.index else np.nan
            if pd.isna(ic_a) and pd.isna(ic_b):
                keep = ""
                reason = "both IC values are missing"
            elif pd.isna(ic_b) or (pd.notna(ic_a) and float(ic_a) >= float(ic_b)):
                keep = factor_a
                reason = f"{factor_a} has higher/equal Rank IC"
            else:
                keep = factor_b
                reason = f"{factor_b} has higher Rank IC"

            rows.append(
                {
                    "section": "redundant_pair",
                    "pair_factor_1": factor_a,
                    "pair_factor_2": factor_b,
                    "avg_spearman_rho": float(rho),
                    "corr_observation_days": int(count_df.loc[factor_a, factor_b]),
                    "redundant_candidate": True,
                    "factor_1_rank_ic_20d": ic_a,
                    "factor_2_rank_ic_20d": ic_b,
                    "recommended_keep": keep,
                    "recommendation_reason": reason,
                }
            )

    rows.sort(key=lambda row: float(row["avg_spearman_rho"]), reverse=True)
    logging.info("Redundant candidate pairs rho>%.2f: %d", threshold, len(rows))
    return rows


def build_report_frame(
    corr_df: pd.DataFrame,
    count_df: pd.DataFrame,
    ic_df: pd.DataFrame,
    pair_rows: list[dict[str, object]],
    missing_factors: list[str],
    used_dates: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    threshold: float,
) -> pd.DataFrame:
    factors = list(corr_df.columns)
    rows: list[dict[str, object]] = []

    rows.append(
        {
            "section": "metadata",
            "notes": (
                f"window={start_date.date()}~{end_date.date()}; "
                f"used_corr_dates={used_dates}; redundant_rule=avg_spearman_rho>{threshold:.2f}"
            ),
        }
    )
    if missing_factors:
        rows.append({"section": "missing_factors", "notes": ",".join(missing_factors)})

    for factor in factors:
        row: dict[str, object] = {"section": "correlation_matrix", "row_factor": factor}
        row.update({col: corr_df.loc[factor, col] for col in factors})
        rows.append(row)

    for factor, item in ic_df.iterrows():
        rows.append(
            {
                "section": "factor_ic_summary",
                "row_factor": factor,
                "factor_rank_ic_20d": item["rank_ic_20d"],
                "factor_rank_ic_std_20d": item["rank_ic_std_20d"],
                "factor_rank_ic_ir_20d": item["rank_ic_ir_20d"],
                "factor_ic_days": item["ic_days"],
                "factor_ic_samples": item["ic_samples"],
            }
        )

    rows.extend(pair_rows)

    ordered_columns = [
        "section",
        "row_factor",
        *factors,
        "pair_factor_1",
        "pair_factor_2",
        "avg_spearman_rho",
        "corr_observation_days",
        "redundant_candidate",
        "factor_1_rank_ic_20d",
        "factor_2_rank_ic_20d",
        "recommended_keep",
        "recommendation_reason",
        "factor_rank_ic_20d",
        "factor_rank_ic_std_20d",
        "factor_rank_ic_ir_20d",
        "factor_ic_days",
        "factor_ic_samples",
        "notes",
    ]
    report = pd.DataFrame(rows)
    for col in ordered_columns:
        if col not in report.columns:
            report[col] = np.nan
    return report[ordered_columns]


def main() -> None:
    setup_logging()
    args = parse_args()

    factor_path = Path(args.factor_path)
    qlib_data_dir = Path(args.qlib_data_dir)
    output_path = Path(args.output)
    if not factor_path.is_absolute():
        factor_path = PROJECT_ROOT / factor_path
    if not qlib_data_dir.is_absolute():
        qlib_data_dir = PROJECT_ROOT / qlib_data_dir
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    logging.info("Factor parquet: %s", factor_path)
    logging.info("Qlib data dir: %s", qlib_data_dir)

    factor_frame, factors, missing, start_date, end_date = load_recent_factor_frame(
        factor_path=factor_path,
        factor_names=PARQUET_FACTORS,
        lookback_years=int(args.lookback_years),
    )
    corr_df, count_df, used_dates = average_cross_sectional_spearman(
        factor_frame=factor_frame,
        factors=factors,
        min_cross_section=int(args.min_cross_section),
    )

    instruments = sorted(factor_frame.index.get_level_values("instrument").unique())
    forward_returns = load_forward_returns(
        qlib_data_dir=qlib_data_dir,
        instruments=instruments,
        start_date=start_date,
        end_date=end_date,
        horizon_days=int(args.horizon_days),
    )
    ic_df = compute_factor_rank_ic(
        factor_frame=factor_frame,
        forward_returns=forward_returns,
        factors=factors,
        min_cross_section=int(args.min_cross_section),
    )

    pair_rows = redundant_pair_rows(
        corr_df=corr_df,
        count_df=count_df,
        ic_df=ic_df,
        threshold=float(args.corr_threshold),
    )
    report = build_report_frame(
        corr_df=corr_df,
        count_df=count_df,
        ic_df=ic_df,
        pair_rows=pair_rows,
        missing_factors=missing,
        used_dates=used_dates,
        start_date=start_date,
        end_date=end_date,
        threshold=float(args.corr_threshold),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    logging.info("Report written: %s", output_path)

    if pair_rows:
        logging.info("Top redundant candidates:")
        for row in pair_rows[:10]:
            logging.info(
                "  %s vs %s rho=%.3f keep=%s",
                row["pair_factor_1"],
                row["pair_factor_2"],
                row["avg_spearman_rho"],
                row["recommended_keep"],
            )
    else:
        logging.info("No redundant pairs above threshold.")


if __name__ == "__main__":
    main()
