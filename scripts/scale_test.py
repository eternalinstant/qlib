#!/usr/bin/env python3
"""500万top100 vs 1000万top300 对比回测"""
import sys, os, json, copy
sys.path.insert(0, os.path.expanduser("~/code/qlib"))
os.chdir(os.path.expanduser("~/code/qlib"))

from config.config import ConfigManager, config as global_config
from core.strategy import Strategy
from modules.backtest.qlib_engine import QlibBacktestEngine

def run_test(capital, strategy_yaml, label):
    """Run backtest with given capital and strategy"""
    # Override capital in global config
    cfg_data = copy.deepcopy(global_config._data)
    cfg_data["initial_capital"] = capital
    global_config._data.update(cfg_data)
    
    strategy = Strategy.load(strategy_yaml)
    engine = QlibBacktestEngine()
    result = engine.run(strategy=strategy)
    result.print_summary(capital)
    
    # Extract key metrics
    ret = result.daily_returns
    if len(ret) == 0:
        return None
    
    total_pct = (1 + ret).prod() - 1
    ann_ret = (1 + ret.mean()) ** 252 - 1
    ann_vol = ret.std() * (252 ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    cummax = (1 + ret).cummax()
    drawdown = (1 + ret) / cummax - 1
    max_dd = drawdown.min()
    
    n_days = len(ret)
    n_years = n_days / 252
    
    # Fee ratio
    fee_ratio = result.metadata.get("fee_ratio_to_initial", 0)
    
    metrics = {
        "label": label,
        "capital": capital,
        "topk": None,  # will fill from strategy
        "total_pct": total_pct,
        "ann_ret": ann_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "fee_ratio": fee_ratio,
        "n_days": n_days,
        "n_years": n_years,
    }
    
    # Get topk from strategy
    sel = strategy.selection if hasattr(strategy, 'selection') else {}
    metrics["topk"] = sel.get("topk", "?")
    
    return metrics

# Run both tests
print("=" * 70)
print("  规模化对比回测: 500万top100 vs 1000万top300")
print("=" * 70)

configs = [
    (5_000_000, "research/scale_test/cap5m_top100_daily_trend", "500万/top100/日频"),
    (10_000_000, "research/scale_test/cap10m_top300_daily_trend", "1000万/top300/日频"),
]

results = []
for capital, strat, label in configs:
    print(f"\n{'─' * 70}")
    print(f"  {label}")
    print(f"{'─' * 70}")
    m = run_test(capital, strat, label)
    if m:
        results.append(m)

# Print comparison
if len(results) == 2:
    r1, r2 = results
    print(f"\n{'=' * 70}")
    print(f"  对比总结")
    print(f"{'=' * 70}")
    print(f"  {'指标':<16} {'500万/top100':>16} {'1000万/top300':>16} {'变化':>12}")
    print(f"  {'─'*60}")
    
    def fmt_pct(v):
        return f"{v*100:+.2f}%"
    
    pairs = [
        ("总收益", fmt_pct(r1["total_pct"]), fmt_pct(r2["total_pct"]), fmt_pct(r2["total_pct"] - r1["total_pct"])),
        ("年化收益", fmt_pct(r1["ann_ret"]), fmt_pct(r2["ann_ret"]), fmt_pct(r2["ann_ret"] - r1["ann_ret"])),
        ("夏普比率", f"{r1['sharpe']:.3f}", f"{r2['sharpe']:.3f}", f"{r2['sharpe']-r1['sharpe']:+.3f}"),
        ("最大回撤", fmt_pct(r1["max_dd"]), fmt_pct(r2["max_dd"]), fmt_pct(r2["max_dd"] - r1["max_dd"])),
        ("手续费率", fmt_pct(r1["fee_ratio"]), fmt_pct(r2["fee_ratio"]), fmt_pct(r2["fee_ratio"] - r1["fee_ratio"])),
        ("交易天数", str(r1["n_days"]), str(r2["n_days"]), str(r2["n_days"] - r1["n_days"])),
    ]
    
    for name, v1, v2, delta in pairs:
        print(f"  {name:<16} {v1:>16} {v2:>16} {delta:>12}")
    
    # Verdict
    print(f"\n  {'─'*60}")
    ann_diff = r2["ann_ret"] - r1["ann_ret"]
    dd_diff = r2["max_dd"] - r1["max_dd"]
    sharpe_diff = r2["sharpe"] - r1["sharpe"]
    
    if ann_diff > 0.005 and dd_diff < 0.01:
        verdict = "✅ 1000万/top300 更优 — 收益更高，回撤可控"
    elif ann_diff > 0 and sharpe_diff > 0:
        verdict = "⚠️ 1000万/top300 略优 — 收益和夏普提升但幅度小"
    elif ann_diff < -0.01:
        verdict = "❌ 1000万/top300 更差 — 分散化过度稀释了alpha"
    elif dd_diff < -0.02:
        verdict = "✅ 1000万/top300 回撤更优 — 但收益可能下降"
    else:
        verdict = "📊 差异不大 — 两种规模效果接近"
    
    print(f"  结论: {verdict}")
