#!/usr/bin/env python3
"""
未覆盖 parquet 原始字段的正式单因子回测。

范围：
- factor_data.parquet 中存在
- formal_single_factor_catalog.csv 里还没有作为 parquet expression 正式覆盖

口径：
- 日频正式单因子模板
- 正向/反向各跑一次

输出：
- results/new_parquet_factor_candidates.csv
- results/new_parquet_factor_results.csv
- results/new_parquet_factor_summary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.formal_single_factor_batch import FactorSpec, run_one

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CATALOG_CSV = RESULTS_DIR / "formal_single_factor_catalog.csv"
OUTPUT_CANDIDATES = RESULTS_DIR / "new_parquet_factor_candidates.csv"
OUTPUT_RESULTS = RESULTS_DIR / "new_parquet_factor_results.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "new_parquet_factor_summary.md"
EXCLUDED_OVERLAP = RESULTS_DIR / "new_parquet_factor_excluded_qlib_overlap.csv"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


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
    for path in sorted(RESULTS_DIR.glob("new_parquet_factor_results*.csv")):
        df = _load_csv(path)
        if not df.empty:
            frames.append(df)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    merged = _dedupe_results(merged)
    merged.to_csv(OUTPUT_RESULTS, index=False)
    return merged


def _get_qlib_feature_names() -> set[str]:
    features_root = PROJECT_ROOT / "data/qlib_data/cn_data/features"
    if not features_root.exists():
        return set()
    return {path.stem.replace(".day", "") for path in features_root.rglob("*.day.bin")}


def collect_new_specs() -> list[FactorSpec]:
    cols = set(pd.read_parquet(PROJECT_ROOT / "data/qlib_data/cn_data/factor_data.parquet").columns)
    cols -= {"datetime", "instrument"}

    catalog = pd.read_csv(CATALOG_CSV)
    covered = set(catalog.loc[catalog["source"] == "parquet", "expression"].astype(str))
    qlib_feature_names = _get_qlib_feature_names()
    names = sorted(cols - covered - qlib_feature_names)

    excluded = sorted((cols - covered) & qlib_feature_names)
    pd.DataFrame({"expression": excluded}).to_csv(EXCLUDED_OVERLAP, index=False)

    specs = [
        FactorSpec(
            freq="day",
            name=name,
            expression=name,
            source="parquet",
            window_scale=1,
            origins=("factor_data_uncovered",),
        )
        for name in names
    ]
    return specs


def write_summary(candidates: list[FactorSpec], results: pd.DataFrame) -> None:
    lines = []
    lines.append("# 新 parquet 因子研究")
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
    parser = argparse.ArgumentParser(description="未覆盖 parquet 原始字段正式单因子回测")
    parser.add_argument("--shards", type=int, default=1, help="总分片数")
    parser.add_argument("--shard", type=int, default=0, help="当前分片编号，从 0 开始")
    parser.add_argument("--merge-only", action="store_true", help="仅合并现有分片结果并生成汇总")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    specs = collect_new_specs()
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
        OUTPUT_RESULTS
        if args.shards == 1
        else RESULTS_DIR / f"new_parquet_factor_results_shard{args.shard}.csv"
    )

    selected_specs = [spec for i, spec in enumerate(specs) if i % args.shards == args.shard]

    existing = _load_csv(shard_results)
    done_keys = set()
    global_existing = _load_csv(OUTPUT_RESULTS)
    if not existing.empty:
        for row in existing.itertuples(index=False):
            done_keys.add((row.factor_name, bool(row.negate)))
    if not global_existing.empty:
        for row in global_existing.itertuples(index=False):
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
