#!/usr/bin/env python3
"""CSI800 topk=8 参数优化扫描。"""
import sys, warnings, copy, json
warnings.filterwarnings('ignore')
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from modules.modeling.predictive_signal import (
    train_from_config, score_from_config, backtest_from_config, load_predictive_config
)

base_cfg = load_predictive_config('config/models/push25_cq10_k8d2_csi800.yaml')

tests = [
    # 1. 纯因子驱动，无 sticky
    {"name": "opt_s0_c3", "sel": {"sticky": 0, "churn_limit": 3, "min_hold_days": 0, "buffer": 0, "stoploss_drawdown": 0.15}},
    # 2. 激进 overlay
    {"name": "opt_aggrov", "ovb": {"target_vol": 0.22, "dd_soft": 0.020, "dd_hard": 0.050, "soft_exposure": 0.85, "hard_exposure": 0.55, "trend_exposure": 0.75}},
    # 3. 高仓位 + 激进 overlay
    {"name": "opt_p95_aggrov", "pos": 0.95, "ovb": {"target_vol": 0.22, "dd_soft": 0.020, "dd_hard": 0.050, "soft_exposure": 0.85, "hard_exposure": 0.55, "trend_exposure": 0.75}},
    # 4. 关闭 overlay
    {"name": "opt_no_overlay", "ovb": {"enabled": False}},
    # 5. 更深的树
    {"name": "opt_deep_tree", "mdl": {"max_depth": 6, "n_estimators": 500, "learning_rate": 0.03, "min_child_samples": 20}},
    # 6. sticky=0 + 关闭 overlay
    {"name": "opt_s0_no_ovb", "sel": {"sticky": 0, "churn_limit": 5, "min_hold_days": 0}, "ovb": {"enabled": False}},
]

results = []
for t in tests:
    cfg = copy.deepcopy(base_cfg)
    tag = t['name']
    cfg['name'] = f"push25_cq10_k8d2_csi800_{tag}"
    cfg['output']['root'] = f"results/model_signals/csi800_opt_20260428/{cfg['name']}"
    for k, v in t.get('sel', {}).items():
        cfg['selection'][k] = v
    for k, v in t.get('ovb', {}).items():
        cfg['overlay'][k] = v
    for k, v in t.get('mdl', {}).items():
        cfg['model']['params'][k] = v
    if 'pos' in t:
        cfg['position']['params']['stock_pct'] = t['pos']

    train_from_config(cfg)
    score_from_config(cfg)
    result, _ = backtest_from_config(cfg, engine='pybroker')
    cagr = result.annual_return
    maxdd = result.max_drawdown
    sharpe = result.sharpe_ratio
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    results.append((tag, cagr, maxdd, sharpe, calmar))
    print(f"[OK] {tag}: CAGR={cagr:.4f} DD={maxdd:.4f} Sharpe={sharpe:.4f}", flush=True)

print("\n=== CSI800 topk=8 优化扫描结果 ===")
print(f"{'Name':<20} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8}")
for r in results:
    print(f"{r[0]:<20} {r[1]:>8.2%} {r[2]:>8.2%} {r[3]:>8.2f} {r[4]:>8.2f}")
