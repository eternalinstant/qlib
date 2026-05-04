"""
Alpha158 精选因子批量回测 — 6 weights × 4 topk = 24 种变体，排名 Top20

用法:
  PYTHONPATH=. python scripts/alpha158_batch_backtest.py
"""

import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from itertools import product

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.strategy import Strategy
from modules.backtest.composite import run_strategy_backtest

# ── 精选 8 因子定义 ──────────────────────────────────────────────

FACTOR_DEFS = {
    "alpha": [
        {"name": "LOW0", "expression": "$low/$close", "source": "qlib", "ir": 0.3614},
        {"name": "VWAP0", "expression": "$vwap/$close", "source": "qlib", "ir": 0.2729},
        {
            "name": "KLOW",
            "expression": "(Less($open, $close)-$low)/$open",
            "source": "qlib",
            "ir": -0.3104,
            "negate": True,
        },
        {
            "name": "KLEN",
            "expression": "($high-$low)/$open",
            "source": "qlib",
            "ir": -0.2257,
            "negate": True,
        },
    ],
    "enhance": [
        {"name": "MIN10", "expression": "Min($low, 10)/$close", "source": "qlib", "ir": 0.2001},
        {
            "name": "QTLD5",
            "expression": "Quantile($close, 5, 0.2)/$close",
            "source": "qlib",
            "ir": 0.1767,
        },
    ],
    "risk": [
        {
            "name": "VMA60",
            "expression": "Mean($volume, 60)/($volume+1e-12)",
            "source": "qlib",
            "ir": 0.2140,
        },
        {
            "name": "CORD5",
            "expression": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",
            "source": "qlib",
            "ir": -0.1847,
            "negate": True,
        },
    ],
}

# ── 权重组合 ─────────────────────────────────────────────────────

WEIGHT_PRESETS = {
    "pure_alpha":    {"alpha": 1.0, "enhance": 0.0, "risk": 0.0},
    "pure_enhance":  {"alpha": 0.0, "enhance": 1.0, "risk": 0.0},
    "balanced":      {"alpha": 0.34, "enhance": 0.33, "risk": 0.33},
    "alpha_heavy":   {"alpha": 0.5,  "enhance": 0.3, "risk": 0.2},
    "enhance_heavy": {"alpha": 0.2,  "enhance": 0.5, "risk": 0.3},
    "risk_aware":    {"alpha": 0.3,  "enhance": 0.3, "risk": 0.4},
}

TOPK_OPTIONS = [5, 8, 10, 15, 20]


def build_variant(name: str, weights: dict, topk: int) -> Strategy:
    """动态构建一个 Strategy 变体（不依赖 YAML 文件）。

    保留 base 的 config_path 以避免 selection_dependency_paths() 回溯时
    找不到 YAML 文件。name 变更后 selections_path() 自然独立，缓存互不干扰。
    """
    base = Strategy.load("experimental/alpha158/alpha158_csi300")
    variant = replace(
        base,
        name=f"alpha158_batch__{name}_k{topk}",
        display_name=f"alpha158_{name}_k{topk}",
        description=f"Alpha158 batch variant: weights={name}, topk={topk}",
        weights=weights,
        topk=topk,
    )
    return variant


def main():
    print("=" * 70)
    print("Alpha158 精选因子批量回测")
    print(f"因子数: {sum(len(v) for v in FACTOR_DEFS.values())}")
    print(f"变体数: {len(WEIGHT_PRESETS)} weights × {len(TOPK_OPTIONS)} topk = "
          f"{len(WEIGHT_PRESETS) * len(TOPK_OPTIONS)}")
    print("=" * 70)

    results = []
    variants = list(product(WEIGHT_PRESETS.items(), TOPK_OPTIONS))

    for idx, ((wname, weights), topk) in enumerate(variants, 1):
        tag = f"{wname}_k{topk}"
        print(f"\n[{idx}/{len(variants)}] {tag} ...", flush=True)
        t0 = time.time()

        try:
            strategy = build_variant(tag, weights, topk)
            bt = run_strategy_backtest(strategy, engine="qlib")
            elapsed = time.time() - t0

            results.append({
                "variant": tag,
                "weight_preset": wname,
                "topk": topk,
                "annual_return": bt.annual_return,
                "sharpe_ratio": bt.sharpe_ratio,
                "max_drawdown": bt.max_drawdown,
                "total_return": bt.total_return,
                "elapsed_s": round(elapsed, 1),
            })
            print(f"  -> 年化={bt.annual_return:.2%}  夏普={bt.sharpe_ratio:.4f}  "
                  f"回撤={bt.max_drawdown:.2%}  ({elapsed:.1f}s)")

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  -> FAILED: {exc} ({elapsed:.1f}s)")
            results.append({
                "variant": tag,
                "weight_preset": wname,
                "topk": topk,
                "annual_return": float("nan"),
                "sharpe_ratio": float("nan"),
                "max_drawdown": float("nan"),
                "total_return": float("nan"),
                "elapsed_s": round(elapsed, 1),
            })

    if not results:
        print("\n[ERROR] 没有成功完成的变体")
        return

    # ── 汇总排名 ─────────────────────────────────────────────
    df = pd.DataFrame(results).sort_values("annual_return", ascending=False)
    top20 = df.head(20)

    output_path = PROJECT_ROOT / "results" / "alpha158_batch_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("Top20 排名（按年化收益降序）")
    print("=" * 70)
    print(top20[["variant", "annual_return", "sharpe_ratio", "max_drawdown"]].to_string(index=False))
    print(f"\n[OK] 完整结果已保存: {output_path}")


if __name__ == "__main__":
    main()
