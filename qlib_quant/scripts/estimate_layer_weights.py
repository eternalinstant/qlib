#!/usr/bin/env python3
"""Estimate Alpha/Risk/Enhance layer weights with the Grinold rule.

Score is computed as IR * sqrt(BR), then normalized to recommended weights.
If an analysis CSV under results/analysis contains explicit layer and IR
columns, the script uses those IR values; otherwise it falls back to the
documented defaults.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"
OUTPUT_PATH = ANALYSIS_DIR / "layer_weight_estimate.csv"

DEFAULT_LAYERS = {
    "Alpha": {"factor_count": 5, "ir": 0.30, "br": 300.0},
    "Risk": {"factor_count": 2, "ir": 0.45, "br": 300.0},
    "Enhance": {"factor_count": 3, "ir": 0.25, "br": 200.0},
}

CURRENT_WEIGHTS = {"Alpha": 0.55, "Risk": 0.20, "Enhance": 0.25}

LAYER_ALIASES = {
    "alpha": "Alpha",
    "alpha层": "Alpha",
    "risk": "Risk",
    "risk层": "Risk",
    "defensive": "Risk",
    "风险": "Risk",
    "enhance": "Enhance",
    "enhance层": "Enhance",
    "enhancement": "Enhance",
    "增强": "Enhance",
}

IR_COLUMNS = [
    "ir",
    "IR",
    "ic_ir",
    "IC_IR",
    "rank_ic_ir",
    "Rank IC IR",
    "rank_ic_ir_20d",
    "factor_rank_ic_ir_20d",
]
LAYER_COLUMNS = ["layer", "Layer", "层", "category", "Category", "group", "Group", "类别"]
BR_COLUMNS = ["br", "BR", "breadth", "Breadth"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate layer weights from IR and breadth.")
    parser.add_argument("--analysis-dir", default=str(ANALYSIS_DIR), help="Directory with analysis CSV files")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output CSV path")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the estimate without writing a CSV",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def normalize_layer(value: object) -> str | None:
    key = str(value).strip().lower()
    return LAYER_ALIASES.get(key)


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    exact = {col: col for col in columns}
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def discover_layer_ir(analysis_dir: Path) -> tuple[dict[str, dict[str, float]], str | None]:
    if not analysis_dir.exists():
        return {}, None

    for csv_path in sorted(analysis_dir.glob("*.csv")):
        try:
            sample = pd.read_csv(csv_path, nrows=5)
        except Exception:
            continue
        columns = list(sample.columns)
        layer_col = first_existing(columns, LAYER_COLUMNS)
        ir_col = first_existing(columns, IR_COLUMNS)
        br_col = first_existing(columns, BR_COLUMNS)
        if not layer_col or not ir_col:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        df["_layer"] = df[layer_col].map(normalize_layer)
        df["_ir"] = pd.to_numeric(df[ir_col], errors="coerce")
        df = df.dropna(subset=["_layer", "_ir"])
        if df.empty:
            continue

        found: dict[str, dict[str, float]] = {}
        for layer, group in df.groupby("_layer"):
            found[layer] = {
                "ir": float(group["_ir"].abs().mean()),
                "factor_count": int(len(group)),
            }
            if br_col:
                br_values = pd.to_numeric(group[br_col], errors="coerce").dropna()
                if not br_values.empty:
                    found[layer]["br"] = float(br_values.mean())

        if found:
            return found, str(csv_path)

    return {}, None


def build_estimate_rows(discovered: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for layer, defaults in DEFAULT_LAYERS.items():
        source = "default"
        values = defaults.copy()
        if layer in discovered:
            values.update(discovered[layer])
            source = "analysis"

        ir = float(values["ir"])
        br = float(values["br"])
        score = ir * math.sqrt(br)
        rows.append(
            {
                "layer": layer,
                "source": source,
                "factor_count": int(values["factor_count"]),
                "ir": ir,
                "breadth": br,
                "grinold_score": score,
                "current_weight": float(CURRENT_WEIGHTS[layer]),
            }
        )

    df = pd.DataFrame(rows)
    total_score = float(df["grinold_score"].sum())
    if total_score <= 0:
        raise ValueError("Total Grinold score must be positive")

    df["recommended_weight"] = df["grinold_score"] / total_score
    df["weight_delta"] = df["recommended_weight"] - df["current_weight"]
    return df


def print_table(df: pd.DataFrame) -> None:
    view = df.copy()
    for col in ["ir", "breadth", "grinold_score", "current_weight", "recommended_weight", "weight_delta"]:
        view[col] = view[col].astype(float)

    print("\nLayer weight estimate (w ∝ IR × sqrt(BR))")
    print("-" * 86)
    print(
        f"{'Layer':<10} {'Source':<9} {'IR':>7} {'BR':>8} {'Score':>8} "
        f"{'Current':>10} {'Grinold':>10} {'Delta':>10}"
    )
    for _, row in view.iterrows():
        print(
            f"{row['layer']:<10} {row['source']:<9} "
            f"{row['ir']:>7.2f} {row['breadth']:>8.0f} {row['grinold_score']:>8.2f} "
            f"{row['current_weight']:>9.1%} {row['recommended_weight']:>9.1%} "
            f"{row['weight_delta']:>+9.1%}"
        )
    print("-" * 86)
    print(f"Current sum: {view['current_weight'].sum():.1%}")
    print(f"Recommended sum: {view['recommended_weight'].sum():.1%}")


def main() -> None:
    setup_logging()
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    output_path = Path(args.output)
    if not analysis_dir.is_absolute():
        analysis_dir = PROJECT_ROOT / analysis_dir
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    discovered, source_path = discover_layer_ir(analysis_dir)
    if source_path:
        logging.info("Using layer IR data from %s", source_path)
    else:
        logging.info("No usable layer IR CSV found under %s; using default estimates", analysis_dir)

    estimate = build_estimate_rows(discovered)
    print_table(estimate)

    if not args.no_write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        estimate.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.6f")
        logging.info("Weight estimate written: %s", output_path)


if __name__ == "__main__":
    main()
