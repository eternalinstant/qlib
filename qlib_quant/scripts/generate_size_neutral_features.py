#!/usr/bin/env python3
"""
Generate size-neutralized factor columns in data/tushare/factor_data.parquet.

Each datetime cross-section is neutralized independently with OLS residuals:
- retained_earnings: factor ~ log(total_mv) + log(total_mv)^2
- other configured factors: factor ~ log(total_mv)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACTOR_PATH = PROJECT_ROOT / "data" / "tushare" / "factor_data.parquet"
MIN_CROSS_SECTION = 10
MARKET_CAP_COL = "total_mv"
DATE_COL = "datetime"


@dataclass(frozen=True)
class NeutralizationSpec:
    factor: str
    quadratic: bool = False

    @property
    def output_col(self) -> str:
        return f"{self.factor}_size_neut"


SPECS = [
    NeutralizationSpec("retained_earnings", quadratic=True),
    NeutralizationSpec("ebit_to_mv"),
    NeutralizationSpec("roa_fina"),
    NeutralizationSpec("book_to_market"),
    NeutralizationSpec("ocf_to_ev"),
    NeutralizationSpec("roe_fina"),
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_factor_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"factor_data.parquet not found: {path}")

    logging.info("Loading factor data: %s", path)
    df = pd.read_parquet(path)
    logging.info("Loaded shape: rows=%d cols=%d", len(df), len(df.columns))
    return df


def validate_columns(df: pd.DataFrame) -> None:
    required = {DATE_COL, MARKET_CAP_COL}
    required.update(spec.factor for spec in SPECS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"factor_data.parquet missing required columns: {missing}")


def design_matrix(log_mv: np.ndarray, quadratic: bool) -> np.ndarray:
    if quadratic:
        return np.column_stack((np.ones(len(log_mv)), log_mv, log_mv * log_mv))
    return np.column_stack((np.ones(len(log_mv)), log_mv))


def residualize_one_factor(
    factor_values: np.ndarray,
    log_mv_values: np.ndarray,
    valid_mv: np.ndarray,
    date_groups: dict[object, np.ndarray],
    spec: NeutralizationSpec,
) -> tuple[np.ndarray, dict[str, int]]:
    result = np.full(len(factor_values), np.nan, dtype="float64")
    stats = {
        "dates_total": 0,
        "dates_done": 0,
        "dates_small": 0,
        "dates_rank_deficient": 0,
        "dates_linalg_error": 0,
        "valid_rows": 0,
    }
    required_rank = 3 if spec.quadratic else 2

    for _, idx in date_groups.items():
        stats["dates_total"] += 1
        y = factor_values[idx]
        valid = valid_mv[idx] & np.isfinite(y)
        valid_count = int(valid.sum())

        if valid_count < MIN_CROSS_SECTION:
            stats["dates_small"] += 1
            continue

        row_idx = idx[valid]
        x = design_matrix(log_mv_values[row_idx], spec.quadratic)
        if np.linalg.matrix_rank(x) < required_rank:
            stats["dates_rank_deficient"] += 1
            continue

        try:
            beta, *_ = np.linalg.lstsq(x, y[valid], rcond=None)
        except np.linalg.LinAlgError:
            stats["dates_linalg_error"] += 1
            continue

        result[row_idx] = y[valid] - x @ beta
        stats["dates_done"] += 1
        stats["valid_rows"] += valid_count

    return result, stats


def add_size_neutral_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    validate_columns(df)
    original_col_count = len(df.columns)
    new_output_cols = [spec.output_col for spec in SPECS if spec.output_col not in df.columns]

    mv = pd.to_numeric(df[MARKET_CAP_COL], errors="coerce").to_numpy(dtype="float64")
    valid_mv = np.isfinite(mv) & (mv > 0)
    log_mv = np.full(len(df), np.nan, dtype="float64")
    log_mv[valid_mv] = np.log(mv[valid_mv])

    logging.info("Preparing datetime cross-sections")
    date_groups = df.groupby(DATE_COL, sort=True).indices
    logging.info(
        "Date groups=%d, valid market-cap rows=%d/%d",
        len(date_groups),
        int(valid_mv.sum()),
        len(df),
    )

    for spec in SPECS:
        factor = pd.to_numeric(df[spec.factor], errors="coerce").to_numpy(dtype="float64")
        model_desc = "log(MV) + log(MV)^2" if spec.quadratic else "log(MV)"
        logging.info("Neutralizing %s with %s", spec.factor, model_desc)

        neutralized, stats = residualize_one_factor(
            factor_values=factor,
            log_mv_values=log_mv,
            valid_mv=valid_mv,
            date_groups=date_groups,
            spec=spec,
        )
        df[spec.output_col] = neutralized

        logging.info(
            "%s -> %s: done_dates=%d/%d, skipped_small=%d, "
            "rank_deficient=%d, linalg_error=%d, non_null=%d",
            spec.factor,
            spec.output_col,
            stats["dates_done"],
            stats["dates_total"],
            stats["dates_small"],
            stats["dates_rank_deficient"],
            stats["dates_linalg_error"],
            int(np.isfinite(neutralized).sum()),
        )

    expected_col_count = original_col_count + len(new_output_cols)
    actual_col_count = len(df.columns)
    if actual_col_count != expected_col_count:
        raise RuntimeError(
            f"Unexpected column count: expected {expected_col_count}, got {actual_col_count}"
        )

    logging.info(
        "Column count verified: before=%d, added=%d, after=%d",
        original_col_count,
        len(new_output_cols),
        actual_col_count,
    )
    return df, len(new_output_cols)


def write_factor_data(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        logging.info("Writing updated parquet to temporary file: %s", tmp_path)
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
        logging.info("Updated parquet written: %s", path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> int:
    setup_logging()
    logging.info("Input parquet: %s", FACTOR_PATH)
    try:
        df = load_factor_data(FACTOR_PATH)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        alt_path = PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "factor_data.parquet"
        if alt_path.exists():
            logging.error(
                "Found alternate factor_data.parquet at %s; not using it because this script is pinned to data/tushare/.",
                alt_path,
            )
        return 1

    df, added_cols = add_size_neutral_columns(df)
    write_factor_data(df, FACTOR_PATH)
    logging.info("Done. Added new columns: %d", added_cols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
