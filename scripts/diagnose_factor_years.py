"""Alpha360 年度因子风格诊断"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from common.platform import temp_dir

SCAN_DIR = temp_dir("scan_factor")

with open(SCAN_DIR / 'alpha360_yearly_ic.json', encoding="utf-8") as f:
    data = json.load(f)

years = ['2019','2020','2021','2022','2023','2024','2025','2026']
factors = list(data.keys())

# Per year: top factors, category dominance, window preference
print("=" * 100)
print("每年 Top-10 因子 (按 |IC| 排名)")
print("=" * 100)
for yr in years:
    ranked = sorted(factors, key=lambda f: abs(data[f].get(str(yr), 0)), reverse=True)
    top10 = []
    for f in ranked[:10]:
        ic = data[f].get(str(yr), 0)
        top10.append(f"{f}({ic:+.4f})")
    print(f"  {yr}: {', '.join(top10)}")

print()
print("=" * 100)
print("每年各类别最强因子 (按 |IC|)")
print("=" * 100)
for yr in years:
    best_per_cat = {}
    for f in factors:
        cat = ''.join(c for c in f if not c.isdigit())
        ic = abs(data[f].get(str(yr), 0))
        if cat not in best_per_cat or ic > abs(data[best_per_cat[cat]].get(str(yr), 0)):
            best_per_cat[f] = ic
    # Show best per category
    parts = []
    for cat in ['LOW','CLOSE','HIGH','VWAP','VOLUME']:
        best_f = max([f for f in factors if f.startswith(cat)], 
                     key=lambda f: abs(data[f].get(str(yr), 0)))
        parts.append(f"{best_f}({data[best_f].get(str(yr),0):+.4f})")
    print(f"  {yr}: {', '.join(parts)}")

print()
print("=" * 100)
print("每年各类别平均 |IC| + 最佳窗口区间")
print("=" * 100)
for yr in years:
    cat_avg_abs = {}
    cat_best_window_range = {}
    for cat in ['LOW','CLOSE','OPEN','HIGH','VWAP','VOLUME']:
        vals = []
        best_window = None
        best_ic = 0
        for f in factors:
            if f.startswith(cat):
                ic = data[f].get(str(yr), 0)
                vals.append(abs(ic))
                if abs(ic) > best_ic:
                    best_ic = abs(ic)
                    best_window = int(f.replace(cat, ''))
        cat_avg_abs[cat] = np.mean(vals)
        
        # Find best window range
        cat_factors = [f for f in factors if f.startswith(cat)]
        cat_factors.sort(key=lambda f: int(f.replace(cat, '')))
        
    parts = []
    for cat in ['CLOSE','LOW','HIGH','VWAP','VOLUME']:
        parts.append(f"{cat}={cat_avg_abs[cat]:.3f}")
    print(f"  {yr}: {'  '.join(parts)}")

print()
print("=" * 100)
print("VOLUME vs Price 因子 IC 方向对比 (每年)")  
print("=" * 100)
for yr in years:
    price_sign = 0
    for f in factors:
        if not f.startswith('VOLUME'):
            ic = data[f].get(str(yr), 0)
            price_sign += np.sign(ic) if ic != 0 else 0
    vol_sign = 0
    for f in factors:
        if f.startswith('VOLUME'):
            ic = data[f].get(str(yr), 0)
            vol_sign += np.sign(ic) if ic != 0 else 0
    price_dir = "反转(正)" if price_sign > 0 else "动量(负)"
    vol_dir = "放量看涨" if vol_sign > 0 else "放量看跌"
    print(f"  {yr}: 价格={price_dir}({price_sign:+.0f})  成交量={vol_dir}({vol_sign:+.0f})")

print()
print("=" * 100)
print("各窗口区间平均 |IC| 按年")
print("=" * 100)
windows = [(1,5),(6,10),(11,15),(16,20),(21,30),(31,40),(41,59)]
print(f"{'':>12}", end="")
for yr in years:
    print(f" {yr:>7}", end="")
print()
print("-" * 80)
for lo, hi in windows:
    avg_per_yr = []
    for yr in years:
        vals = []
        for f in factors:
            cat = ''.join(c for c in f if not c.isdigit())
            if cat in ['LOW','CLOSE','HIGH','VWAP']:
                win = int(f.replace(cat, ''))
                if lo <= win <= hi:
                    v = data[f].get(str(yr), 0)
                    vals.append(abs(v))
        avg_per_yr.append(np.mean(vals) if vals else 0)
    best_yr_idx = np.argmax(avg_per_yr)
    print(f"  {lo:>2}-{hi:<2}天      ", end="")
    for i, v in enumerate(avg_per_yr):
        mark = " *" if i == best_yr_idx else "  "
        print(f" {v:>+.4f}{mark}", end="")
    print()

print()
print("=" * 100)
print("年度诊断总结")
print("=" * 100)
for yr in years:
    # Dominant category
    cat_avg = {}
    for cat in ['CLOSE','LOW','HIGH','VWAP','VOLUME']:
        vals = [abs(data[f].get(yr,0)) for f in factors if f.startswith(cat)]
        cat_avg[cat] = np.mean(vals)
    dom_cat = max(cat_avg, key=cat_avg.get)
    
    # Dominant window range
    window_ic = {}
    for lo, hi in [(1,10),(11,20),(21,40),(41,59)]:
        vals = []
        for f in factors:
            cat = ''.join(c for c in f if not c.isdigit())
            if cat in ['LOW','CLOSE','HIGH','VWAP']:
                w = int(f.replace(cat, ''))
                if lo <= w <= hi:
                    vals.append(abs(data[f].get(yr, 0)))
        window_ic[(lo,hi)] = np.mean(vals) if vals else 0
    dom_win = max(window_ic, key=window_ic.get)
    
    # Signal strength
    all_abs = [abs(data[f].get(yr, 0)) for f in factors]
    strength = np.mean(all_abs)
    level = "强" if strength > 0.05 else "中" if strength > 0.03 else "弱"
    
    # Price direction
    price_vals = [data[f].get(yr, 0) for f in factors if not f.startswith('VOLUME')]
    price_mean = np.mean([v for v in price_vals if not np.isnan(v)])
    price_label = "反转" if price_mean > 0.01 else "弱反转" if price_mean > 0 else "动量" if price_mean < -0.01 else "中性"
    
    # Volume direction
    vol_vals = [data[f].get(yr, 0) for f in factors if f.startswith('VOLUME')]
    vol_mean = np.mean([v for v in vol_vals if not np.isnan(v)])
    vol_label = "放量看跌" if vol_mean < -0.01 else "放量看涨" if vol_mean > 0.01 else "中性"
    
    print(f"  {yr}: 属类={dom_cat}  窗口={dom_win[0]}-{dom_win[1]}天  强度={level}({strength:.3f})  价格={price_label}  量={vol_label}")
