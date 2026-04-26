"""单因子 IC 扫描：逐个因子训练 LightGBM，记录验证段 Rank IC"""
import sys, yaml, tempfile, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.modeling.predictive_signal import load_predictive_config, train_from_config, _alpha158_feature_map

with open("config/models/push25_cq10_k8d2_very_tight.yaml") as f:
    cq10_cfg = yaml.safe_load(f)

# Parquet 候选因子
parquet_candidates = [
    "ocf_to_ev", "fcff_to_mv", "roe_fina", "current_ratio_fina", "n_cashflow_act",
    "rank_value_profit_core", "rank_balance_core", "qvf_core_interaction",
    "ocf_to_mv", "ebit_to_mv", "ebitda_to_mv", "roa_fina",
    "roic_proxy", "book_to_market", "retained_earnings",
    "qvf_core_alpha", "qvf_core_dynamic", "qvf_group_alpha",
    "quality_value_interaction", "cashflow_flow_interaction",
    "qvf_dynamic_blend",
]

# Alpha158 候选因子（去掉绝对价格类）
fmap = _alpha158_feature_map(cq10_cfg["data"]["alpha158"])
alpha158_candidates = [k for k in sorted(fmap.keys()) if not k.endswith("0")]

base_cfg = {
    "data": {"start_date": "2019-01-01", "end_date": "2026-04-15"},
    "label": {"horizon_days": 20},
    "training": {"train_start": "2019-01-01", "train_end": "2022-12-31",
                 "valid_start": "2023-01-01", "valid_end": "2023-12-31"},
    "scoring": {"start_date": "2019-01-01", "end_date": "2026-04-15"},
    "model": {"preferred_backend": "lightgbm", "params": {
        "learning_rate": 0.05, "n_estimators": 200, "max_depth": 4,
        "min_child_samples": 32, "random_state": 42, "verbosity": -1}},
    "selection": {"freq": "biweek", "topk": 8, "universe": "csi300", "min_market_cap": 120,
                  "exclude_st": True, "exclude_new_days": 120, "sticky": 6, "churn_limit": 2,
                  "mode": "factor_topk"},
    "position": {"model": "fixed", "params": {"stock_pct": 0.80}},
    "overlay": {"enabled": False},
    "trading": {"buy_commission_rate": 0.0003, "sell_commission_rate": 0.0003,
                "sell_stamp_tax_rate": 0.001},
}

import json
results = {"parquet": {}, "alpha158": {}}

# --- Parquet scan ---
print("=" * 50)
print("PARQUET 单因子扫描")
print("=" * 50)
for feat in parquet_candidates:
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base_cfg.items()}
    cfg["name"] = f"scan_pq_{feat}"
    cfg["data"] = dict(base_cfg["data"])
    cfg["data"]["source"] = "parquet"
    cfg["data"]["feature_columns"] = [feat]
    cfg["output"] = {"root": f"/tmp/scan_factor/pq_{feat}"}

    tmp = f"/tmp/scan_factor/pq_{feat}.yaml"
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "w") as f:
        yaml.dump(cfg, f)

    try:
        c = load_predictive_config(tmp)
        result = train_from_config(c)
        ic = result.get("metrics", {}).get("valid_mean_rank_ic", None)
        if ic is not None:
            print(f"  {feat:<32} IC={ic:>+.4f}")
            results["parquet"][feat] = ic
    except Exception as e:
        print(f"  {feat:<32} ERR: {str(e)[:50]}")

# --- Alpha158 scan ---
print("\n" + "=" * 50)
print("ALPHA158 单因子扫描")
print("=" * 50)
for feat in alpha158_candidates:
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base_cfg.items()}
    cfg["name"] = f"scan_a158_{feat}"
    cfg["data"] = dict(base_cfg["data"])
    cfg["data"]["source"] = "alpha158"
    cfg["data"]["alpha158_universe"] = "csi300"
    cfg["data"]["feature_columns"] = [feat]
    cfg["data"]["alpha158"] = cq10_cfg["data"]["alpha158"]
    cfg["output"] = {"root": f"/tmp/scan_factor/a158_{feat}"}

    tmp = f"/tmp/scan_factor/a158_{feat}.yaml"
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "w") as f:
        yaml.dump(cfg, f)

    try:
        c = load_predictive_config(tmp)
        result = train_from_config(c)
        ic = result.get("metrics", {}).get("valid_mean_rank_ic", None)
        if ic is not None:
            print(f"  {feat:<12} IC={ic:>+.4f}")
            results["alpha158"][feat] = ic
    except Exception as e:
        print(f"  {feat:<12} ERR: {str(e)[:50]}")

# --- Summary ---
print("\n" + "=" * 50)
print("汇总排名")
print("=" * 50)
print("\n--- Parquet Top ---")
for feat, ic in sorted(results["parquet"].items(), key=lambda x: x[1], reverse=True):
    tag = " ***" if ic > 0.03 else ""
    print(f"  {feat:<32} IC={ic:>+.4f}{tag}")

print("\n--- Alpha158 Top ---")
for feat, ic in sorted(results["alpha158"].items(), key=lambda x: x[1], reverse=True):
    tag = " ***" if ic > 0.01 else ""
    print(f"  {feat:<12} IC={ic:>+.4f}{tag}")

# Save
with open("/tmp/scan_factor/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n结果已保存: /tmp/scan_factor/results.json")
