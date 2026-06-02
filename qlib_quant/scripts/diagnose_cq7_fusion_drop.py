#!/usr/bin/env python3
"""Diagnose why cq7+alpha5 fusion underperforms cq7."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from modules.modeling.predictive_signal import (
    assemble_labeled_frame,
    load_close_series,
    load_feature_frame,
    load_predictive_config,
    slice_frame_by_date,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "model_signals" / "validation_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def calc_selection_overlap() -> pd.DataFrame:
    s1 = pd.read_csv(PROJECT_ROOT / "results" / "model_signals" / "push25_cq7_k8d2_very_tight" / "selections.csv")
    s2 = pd.read_csv(
        PROJECT_ROOT / "results" / "model_signals" / "push25_cq7_alpha5_fusion_k8d2_very_tight" / "selections.csv"
    )
    a = s1[s1["rank"] <= 8][["date", "symbol"]].copy()
    b = s2[s2["rank"] <= 8][["date", "symbol"]].copy()
    rows = []
    for d in sorted(set(a["date"]).intersection(set(b["date"]))):
        sa = set(a.loc[a["date"] == d, "symbol"])
        sb = set(b.loc[b["date"] == d, "symbol"])
        inter = len(sa & sb)
        uni = len(sa | sb)
        rows.append({"date": d, "overlap": inter, "jaccard": inter / uni if uni else 0.0})
    overlap = pd.DataFrame(rows)
    overlap["year"] = pd.to_datetime(overlap["date"]).dt.year
    overlap.to_csv(OUT_DIR / "cq7_fusion_overlap_daily.csv", index=False)
    overlap.groupby("year", as_index=False)["jaccard"].mean().to_csv(OUT_DIR / "cq7_fusion_overlap_yearly.csv", index=False)
    return overlap


def _daily_rank_ic(df: pd.DataFrame, feature: str) -> float:
    vals: list[float] = []
    for _, gd in df.groupby("datetime", sort=False):
        if len(gd) < 20:
            continue
        corr = spearmanr(gd[feature], gd["label"], nan_policy="omit").correlation
        if np.isfinite(corr):
            vals.append(float(corr))
    return float(np.mean(vals)) if vals else np.nan


def calc_feature_ic_and_corr() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_predictive_config(PROJECT_ROOT / "config" / "models" / "push25_cq7_alpha5_fusion_k8d2_very_tight.yaml")
    features = list(cfg["data"]["parquet_feature_columns"]) + list(cfg["data"]["alpha158_feature_columns"])

    feature_frame, _, _ = load_feature_frame(
        start_date=str(cfg["training"]["train_start"]),
        end_date=str(cfg["scoring"]["end_date"]),
        rebalance_freq=str(cfg["selection"]["freq"]),
        feature_columns=features,
        data_cfg=cfg["data"],
        selection_cfg=cfg["selection"],
    )
    close_series = load_close_series(
        instruments=sorted(feature_frame.index.get_level_values("instrument").unique().tolist()),
        start_date=str(cfg["training"]["train_start"]),
        end_date=str(cfg["scoring"]["end_date"]),
        horizon_days=int(cfg["label"]["horizon_days"]),
    )
    labeled = assemble_labeled_frame(feature_frame, close_series, int(cfg["label"]["horizon_days"]))
    labeled = labeled.reset_index()
    labeled["year"] = labeled["datetime"].dt.year

    rows = []
    for f in features:
        tmp = labeled[["datetime", "year", f, "label"]].dropna()
        for y, g in tmp.groupby("year", sort=True):
            rows.append({"feature": f, "year": int(y), "mean_rank_ic": _daily_rank_ic(g, f)})
    ic_df = pd.DataFrame(rows)
    ic_df.to_csv(OUT_DIR / "cq7_fusion_feature_ic_yearly.csv", index=False)

    train_valid = slice_frame_by_date(
        assemble_labeled_frame(feature_frame, close_series, int(cfg["label"]["horizon_days"])),
        str(cfg["training"]["train_start"]),
        str(cfg["training"]["valid_end"]),
    )
    corr = train_valid[features].corr(method="spearman")
    corr.to_csv(OUT_DIR / "cq7_fusion_feature_corr_train_valid.csv")
    return ic_df, corr


def main() -> None:
    overlap = calc_selection_overlap()
    ic_df, corr = calc_feature_ic_and_corr()

    print(
        "OVERLAP_SUMMARY",
        {
            "mean_overlap": float(overlap["overlap"].mean()),
            "mean_jaccard": float(overlap["jaccard"].mean()),
            "jaccard_p25": float(overlap["jaccard"].quantile(0.25)),
            "jaccard_p50": float(overlap["jaccard"].quantile(0.5)),
            "jaccard_p75": float(overlap["jaccard"].quantile(0.75)),
        },
    )
    print("OVERLAP_BY_YEAR")
    print(overlap.groupby("year")["jaccard"].mean().to_string())

    focus = ["ROC20", "RSV20", "RANK20", "CORD20", "VSUMD20", "rank_value_profit_core", "qvf_core_interaction"]
    piv = ic_df[ic_df["feature"].isin(focus)].pivot(index="feature", columns="year", values="mean_rank_ic")
    print("IC_YEARLY_FOCUS")
    print(piv.to_string())

    print("TOP_CORR_FOR_NEW_FACTORS")
    for nf in ["RSV20", "RANK20", "CORD20", "VSUMD20"]:
        s = corr[nf].drop(index=nf).sort_values(key=lambda x: x.abs(), ascending=False).head(6)
        print(nf, s.to_dict())


if __name__ == "__main__":
    main()
