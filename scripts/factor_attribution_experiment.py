#!/usr/bin/env python3
"""
合成因子归因实验：分解 full_24 相对 core_12 的增量贡献。

实验组:
1. core_12 (基线)
2. core_12 + rank_core_4 (4个 rank 合成因子)
3. core_12 + qvf_core_2 (2个 qvf 交互因子)
4. core_12 + all_synthetic_6 (全部6个合成因子)
5. core_12 + deleted_raw_6 (被砍掉的6个原始因子)
6. full_24 (完整24因子对照)

用法:
    python3 scripts/factor_attribution_experiment.py
    python3 scripts/factor_attribution_experiment.py --base strategy/configs/models/qvf_alpha158_core12.yaml
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_CONFIG = PROJECT_ROOT / "strategy" / "configs" / "models" / "qvf_alpha158_core12.yaml"
FULL_CONFIG = PROJECT_ROOT / "strategy" / "configs" / "models" / "qvf_alpha158_hybrid24_prune_v2.yaml"

# 因子分组定义
RANK_SYNTHETIC_4 = [
    "rank_value_profit_core",
    "rank_flow_momentum_core",
    "rank_growth_quality_core",
    "rank_balance_core",
]
QVF_SYNTHETIC_2 = [
    "qvf_core_alpha",
    "qvf_core_interaction",
]
DELETED_RAW_6 = [
    "ebitda_to_mv",
    "fcff_to_mv",
    "roe_dt_fina",
    "current_ratio_fina",
    "net_margin",
    "n_cashflow_act",
]


def clone_config(base_cfg: dict, name_suffix: str, extra_parquet: list[str]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["name"] = f"{base_cfg['name']}__{name_suffix}"
    cfg["output"]["root"] = str(
        Path(base_cfg["output"]["root"]).parent / cfg["name"]
    )
    # 合并因子列表
    existing = set(cfg["data"].get("parquet_feature_columns", []))
    cfg["data"]["parquet_feature_columns"] = sorted(existing | set(extra_parquet))
    return cfg


def run_single_experiment(cfg: dict) -> dict | None:
    from strategy.producers.predictive_signal import (
        backtest_from_config,
        load_predictive_config,
        save_json,
        score_from_config,
        train_from_config,
    )

    try:
        print(f"\n  训练: {cfg['name']} ({len(cfg['data'].get('parquet_feature_columns', []))} parquet + {len(cfg['data'].get('alpha158_feature_columns', []))} alpha158)")
        train_summary = train_from_config(cfg)
        score_summary = score_from_config(cfg)
        bt_result, bt_summary = backtest_from_config(cfg)

        result = {
            "name": cfg["name"],
            "parquet_features": cfg["data"].get("parquet_feature_columns", []),
            "alpha158_features": cfg["data"].get("alpha158_feature_columns", []),
            "feature_count": len(cfg["data"].get("parquet_feature_columns", [])) + len(cfg["data"].get("alpha158_feature_columns", [])),
            "valid_rank_ic": train_summary.get("metrics", {}).get("valid_mean_rank_ic"),
        }
        if bt_summary:
            result.update({
                "annual_return": bt_summary.get("annual_return"),
                "max_drawdown": bt_summary.get("max_drawdown"),
                "sharpe_ratio": bt_summary.get("sharpe_ratio"),
                "avg_exposure": bt_summary.get("avg_exposure"),
            })
        return result
    except Exception as e:
        print(f"  FAIL: {e}")
        return {"name": cfg["name"], "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="合成因子归因实验")
    parser.add_argument("--base", default=str(BASE_CONFIG))
    parser.add_argument("--full", default=str(FULL_CONFIG))
    parser.add_argument("--skip-core12", action="store_true", help="跳过 core_12 基线（已跑过）")
    parser.add_argument("--skip-full24", action="store_true", help="跳过 full_24 对照（已跑过）")
    args = parser.parse_args()

    with open(args.base) as f:
        base_cfg = yaml.safe_load(f)

    experiments = []

    # 实验 1: core_12 基线
    if not args.skip_core12:
        experiments.append(("core12_baseline", base_cfg, []))
    else:
        print("跳过 core_12 基线（--skip-core12）")

    # 实验 2: core_12 + rank_core_4
    experiments.append((
        "core12_plus_rank4",
        clone_config(base_cfg, "plus_rank4", RANK_SYNTHETIC_4),
        RANK_SYNTHETIC_4,
    ))

    # 实验 3: core_12 + qvf_core_2
    experiments.append((
        "core12_plus_qvf2",
        clone_config(base_cfg, "plus_qvf2", QVF_SYNTHETIC_2),
        QVF_SYNTHETIC_2,
    ))

    # 实验 4: core_12 + all_synthetic_6
    experiments.append((
        "core12_plus_synthetic6",
        clone_config(base_cfg, "plus_synthetic6", RANK_SYNTHETIC_4 + QVF_SYNTHETIC_2),
        RANK_SYNTHETIC_4 + QVF_SYNTHETIC_2,
    ))

    # 实验 5: core_12 + deleted_raw_6
    experiments.append((
        "core12_plus_raw6",
        clone_config(base_cfg, "plus_raw6", DELETED_RAW_6),
        DELETED_RAW_6,
    ))

    # 实验 6: full_24
    if not args.skip_full24:
        with open(args.full) as f:
            full_cfg = yaml.safe_load(f)
        full_cfg_copy = copy.deepcopy(full_cfg)
        full_cfg_copy["output"]["root"] = str(
            Path(full_cfg["output"]["root"]).parent / f"{full_cfg['name']}__full24_baseline"
        )
        experiments.append(("full24_baseline", full_cfg_copy, []))
    else:
        print("跳过 full_24 对照（--skip-full24）")

    # 运行实验
    print(f"\n{'=' * 60}")
    print(f"合成因子归因实验: {len(experiments)} 组")
    print(f"{'=' * 60}")

    results = []
    for label, cfg, extra_factors in experiments:
        print(f"\n--- 实验: {label} ---")
        if extra_factors:
            print(f"  额外因子: {extra_factors}")
        result = run_single_experiment(cfg)
        if result:
            results.append(result)

    # 汇总输出
    print(f"\n{'=' * 80}")
    print("归因实验汇总")
    print(f"{'=' * 80}")
    print(f"{'实验':>30} {'因子数':>6} {'IC':>10} {'年化':>10} {'回撤':>10} {'夏普':>8}")
    print("-" * 80)

    import json
    for r in results:
        name = r.get("name", "?")[-30:]
        n = r.get("feature_count", "?")
        ic = r.get("valid_rank_ic", None)
        ann = r.get("annual_return", None)
        dd = r.get("max_drawdown", None)
        sh = r.get("sharpe_ratio", None)
        ic_str = f"{ic:+.4f}" if ic is not None else "N/A"
        ann_str = f"{ann:+.2%}" if ann is not None else "N/A"
        dd_str = f"{dd:+.2%}" if dd is not None else "N/A"
        sh_str = f"{sh:.2f}" if sh is not None else "N/A"
        print(f"{name:>30} {n:>6} {ic_str:>10} {ann_str:>10} {dd_str:>10} {sh_str:>8}")

    output_dir = PROJECT_ROOT / "results" / "attribution"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "factor_attribution_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n输出: {out_path}")


if __name__ == "__main__":
    main()
