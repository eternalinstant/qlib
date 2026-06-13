#!/usr/bin/env python3
"""P0 实盘成交假设审计：对比 baseline（close-to-close）vs realistic（T+1 开盘 + 约束）。

用法：
    python3 scripts/run_p0_audit.py                    # 审计所有默认策略
    python3 scripts/run_p0_audit.py --config qvf_alpha158_core12  # 只审计一个
    python3 scripts/run_p0_audit.py --skip-baseline    # 跳过 baseline，只跑 realistic
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategy.producers.predictive_signal import backtest_from_config, load_predictive_config

# P0 约束参数
P0_TRADING_OVERRIDES = {
    "next_open_execution": True,
    "block_limit_up_buy": True,
    "block_limit_down_sell": True,
    "t1_sell_lock_days": 1,
    "volume_cap_pct": 0.01,
    "partial_fill_min_scale": 0.20,
}

DEFAULT_STRATEGIES = [
    "qvf_alpha158_core12",
    "push25_cq10_k8d2_very_tight",
    "alpha158_momentum_volume_k6_dd10_overlay",
    "push25_cq10_v3_vol_norm",
]

AUDIT_DIR = PROJECT_ROOT / "results" / "audit_realistic_execution_20260512" / "p0_realistic_execution"


def _metrics(result) -> dict:
    return {
        "annual_return": round(result.annual_return, 4),
        "sharpe_ratio": round(result.sharpe_ratio, 4),
        "max_drawdown": round(result.max_drawdown, 4),
    }


def _run_one(cfg: dict, label: str, overrides: dict | None, engine: str) -> dict:
    run_cfg = copy.deepcopy(cfg)
    if overrides:
        run_cfg.setdefault("trading", {}).update(overrides)
    print(f"  [{label}] 运行回测...", flush=True)
    result, _ = backtest_from_config(run_cfg, engine=engine)
    m = _metrics(result)
    print(f"  [{label}] 年化={m['annual_return']:.2%} 夏普={m['sharpe_ratio']:.3f} 回撤={m['max_drawdown']:.2%}")
    return m


def audit_strategy(config_name: str, engine: str, skip_baseline: bool) -> dict:
    config_path = PROJECT_ROOT / "config" / "models" / f"{config_name}.yaml"
    if not config_path.exists():
        print(f"[SKIP] 配置不存在: {config_path}")
        return {}

    cfg = load_predictive_config(config_path)
    strategy_dir = AUDIT_DIR / config_name
    strategy_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== 审计策略: {config_name} ===")

    baseline_m = {}
    if not skip_baseline:
        # baseline：关闭所有 P0 约束
        baseline_overrides = {k: (False if isinstance(v, bool) else (0 if isinstance(v, int) else None))
                              for k, v in P0_TRADING_OVERRIDES.items()}
        baseline_m = _run_one(cfg, "baseline", baseline_overrides, engine)
        with open(strategy_dir / "metrics_baseline.json", "w") as f:
            json.dump(baseline_m, f, indent=2)

    # realistic：打开全部 P0 约束
    realistic_m = _run_one(cfg, "realistic", P0_TRADING_OVERRIDES, engine)
    with open(strategy_dir / "metrics_realistic.json", "w") as f:
        json.dump(realistic_m, f, indent=2)

    result = {"strategy": config_name, "baseline": baseline_m, "realistic": realistic_m}

    if baseline_m:
        delta = {
            k: round(realistic_m[k] - baseline_m[k], 4)
            for k in baseline_m
        }
        result["delta"] = delta

        sharpe_drop_pct = (
            (baseline_m["sharpe_ratio"] - realistic_m["sharpe_ratio"]) / abs(baseline_m["sharpe_ratio"])
            if baseline_m["sharpe_ratio"] != 0 else 0
        )
        result["sharpe_drop_pct"] = round(sharpe_drop_pct, 4)

        # 决策
        if sharpe_drop_pct > 0.30:
            verdict = "⚠️ WARN: 夏普下降 >30%，实盘 gap 严重，需要重新评估信号时点"
        elif sharpe_drop_pct > 0.15:
            verdict = "⚡ INFO: 夏普下降 15-30%，存在明显成交假设 gap"
        else:
            verdict = "✅ OK: 夏普下降 <15%，实盘执行假设影响可控"
        result["verdict"] = verdict
        print(f"  Delta 年化={delta['annual_return']:+.2%} 夏普={delta['sharpe_ratio']:+.3f} | {verdict}")

        report_lines = [
            f"# P0 审计报告：{config_name}",
            f"\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## 指标对比\n",
            "| 指标 | baseline | realistic | delta |",
            "|---|---|---|---|",
            f"| 年化收益 | {baseline_m['annual_return']:.2%} | {realistic_m['annual_return']:.2%} | {delta['annual_return']:+.2%} |",
            f"| 夏普比率 | {baseline_m['sharpe_ratio']:.3f} | {realistic_m['sharpe_ratio']:.3f} | {delta['sharpe_ratio']:+.3f} |",
            f"| 最大回撤 | {baseline_m['max_drawdown']:.2%} | {realistic_m['max_drawdown']:.2%} | {delta['max_drawdown']:+.2%} |",
            f"\n## P0 约束设置\n",
            "```yaml",
        ] + [f"{k}: {v}" for k, v in P0_TRADING_OVERRIDES.items()] + [
            "```",
            f"\n## 结论\n\n{verdict}",
            f"\n夏普下降幅度：{sharpe_drop_pct:.1%}",
        ]
        (strategy_dir / "delta_report.md").write_text("\n".join(report_lines))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=None, help="策略名称（不含 .yaml 后缀）")
    parser.add_argument("--engine", default="qlib", choices=["qlib", "pybroker"])
    parser.add_argument("--skip-baseline", action="store_true", help="跳过 baseline，只跑 realistic")
    args = parser.parse_args()

    strategies = args.config or DEFAULT_STRATEGIES
    results = []

    for name in strategies:
        try:
            r = audit_strategy(name, args.engine, args.skip_baseline)
            if r:
                results.append(r)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    if results:
        print("\n" + "=" * 60)
        print("P0 审计汇总")
        print("=" * 60)
        for r in results:
            b = r.get("baseline", {})
            re = r.get("realistic", {})
            delta = r.get("delta", {})
            v = r.get("verdict", "N/A")
            if b and re:
                print(f"{r['strategy']:45s} 夏普 {b['sharpe_ratio']:.3f}→{re['sharpe_ratio']:.3f} ({delta.get('sharpe_ratio', 0):+.3f})  {v}")
            else:
                print(f"{r['strategy']:45s} realistic 夏普={re.get('sharpe_ratio', 0):.3f}")

        summary_path = AUDIT_DIR / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] 结果已保存: {AUDIT_DIR}")


if __name__ == "__main__":
    main()
