#!/usr/bin/env python3
"""
对 qlib 已提供、但正式目录还未覆盖的原始字段做正式单因子回测。

范围：
- factor_data.parquet 与 qlib feature bin 同时存在
- formal_single_factor_catalog.csv 里还没有作为 qlib expression 正式覆盖

口径：
- 日频正式单因子模板
- 正向/反向各跑一次
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.formal_single_factor_batch import FactorSpec, run_one


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CATALOG_CSV = RESULTS_DIR / "formal_single_factor_catalog.csv"
OUTPUT_CANDIDATES = RESULTS_DIR / "new_qlib_overlap_factor_candidates.csv"
OUTPUT_RESULTS = RESULTS_DIR / "new_qlib_overlap_factor_results.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "new_qlib_overlap_factor_summary.md"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _get_qlib_feature_names() -> set[str]:
    features_root = PROJECT_ROOT / "data/qlib_data/cn_data/features"
    if not features_root.exists():
        return set()
    return {path.stem.replace(".day", "") for path in features_root.rglob("*.day.bin")}


def collect_specs() -> list[FactorSpec]:
    parquet_cols = set(pd.read_parquet(PROJECT_ROOT / "data/qlib_data/cn_data/factor_data.parquet").columns)
    parquet_cols -= {"datetime", "instrument"}
    qlib_feature_names = _get_qlib_feature_names()

    catalog = pd.read_csv(CATALOG_CSV)
    covered = set(catalog.loc[catalog["source"] == "qlib", "expression"].astype(str))
    names = sorted(name for name in (parquet_cols & qlib_feature_names) if f"${name}" not in covered)

    return [
        FactorSpec(
            freq="day",
            name=name,
            expression=f"${name}",
            source="qlib",
            window_scale=1,
            origins=("qlib_overlap_uncovered",),
        )
        for name in names
    ]


def _dedupe_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = ["freq", "factor_name", "negate", "source", "expression", "window_scale"]
    keep_cols = [col for col in keys if col in df.columns]
    df = df.drop_duplicates(subset=keep_cols, keep="last").copy()
    order = {
        "status": True,
        "sharpe_ratio": False,
        "annual_return": False,
        "factor_name": True,
        "negate": True,
    }
    sort_cols = [col for col in ["status", "sharpe_ratio", "annual_return", "factor_name", "negate"] if col in df.columns]
    ascending = [order[col] for col in sort_cols]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending)
    return df.reset_index(drop=True)


def merge_existing_results() -> pd.DataFrame:
    frames = []
    for path in sorted(RESULTS_DIR.glob("new_qlib_overlap_factor_results*.csv")):
        df = _load_csv(path)
        if not df.empty:
            frames.append(df)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    merged = _dedupe_results(merged)
    merged.to_csv(OUTPUT_RESULTS, index=False)
    return merged


def write_summary(candidates: list[FactorSpec], results: pd.DataFrame) -> None:
    lines = []
    lines.append("# 新 Qlib Overlap 因子研究")
    lines.append("")
    lines.append(f"- 原始候选字段: {len(candidates)}")
    lines.append(f"- 已完成回测: {len(results)}")
    lines.append("")

    if results.empty:
        OUTPUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
        return

    success = results[results["status"] == "success"].copy()
    lines.append(f"- 成功: {len(success)}")
    lines.append(f"- 失败: {len(results) - len(success)}")
    lines.append("")

    if not success.empty:
        top = success.sort_values(["sharpe_ratio", "annual_return"], ascending=[False, False]).head(20)
        show = top[["factor_name", "direction", "annual_return", "sharpe_ratio", "max_drawdown", "strategy_name"]].copy()
        show["annual_return"] = show["annual_return"].map(lambda x: f"{x:.4%}")
        show["max_drawdown"] = show["max_drawdown"].map(lambda x: f"{x:.4%}")
        show["sharpe_ratio"] = show["sharpe_ratio"].map(lambda x: f"{x:.4f}")
        lines.append("## Top 20 By Sharpe")
        lines.append("")
        try:
            lines.append(show.to_markdown(index=False))
        except Exception:
            lines.append("```")
            lines.append(show.to_string(index=False))
            lines.append("```")
        lines.append("")

    OUTPUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="qlib 原始字段正式单因子回测")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    specs = collect_specs()
    pd.DataFrame(
        [
            {
                "freq": spec.freq,
                "name": spec.name,
                "expression": spec.expression,
                "source": spec.source,
                "window_scale": spec.window_scale,
                "expr_hash": spec.expr_hash,
                "origins": " | ".join(spec.origins),
            }
            for spec in specs
        ]
    ).to_csv(OUTPUT_CANDIDATES, index=False)

    if args.merge_only:
        merged = merge_existing_results()
        write_summary(specs, merged)
        return 0

    shard_results = (
        OUTPUT_RESULTS if args.shards == 1 else RESULTS_DIR / f"new_qlib_overlap_factor_results_shard{args.shard}.csv"
    )
    selected_specs = [spec for i, spec in enumerate(specs) if i % args.shards == args.shard]

    existing = _load_csv(shard_results)
    global_existing = _load_csv(OUTPUT_RESULTS)
    done_keys = set()
    for frame in [existing, global_existing]:
        if frame.empty:
            continue
        for row in frame.itertuples(index=False):
            done_keys.add((row.factor_name, bool(row.negate)))

    rows = []
    if not existing.empty:
        rows.extend(existing.to_dict("records"))

    total = len(selected_specs) * 2
    done = 0
    for spec in selected_specs:
        for negate in (False, True):
            if (spec.name, negate) in done_keys:
                continue
            done += 1
            print(f"[{done}/{total}] shard={args.shard}/{args.shards} {spec.name} {'反向' if negate else '正向'}")
            rows.append(run_one(spec, negate, force_selections=True))
            pd.DataFrame(rows).to_csv(shard_results, index=False)

    results = pd.DataFrame(rows)
    results.to_csv(shard_results, index=False)
    if args.shards == 1:
        write_summary(specs, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
