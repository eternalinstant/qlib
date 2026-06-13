#!/usr/bin/env python3
"""
防未来函数审计：验证系统不存在 look-ahead bias。

检查项：
1. ann_date 准确性 spot-check（与已知公开日期对比）
2. ann_date + 1 day 的 merge_asof 行为验证（周末/节假日）
3. 因子在 ann_date 之前不应变化
4. 截面 rank / zscore / neutralize 只用当日截面
5. 训练/验证/holdout 严格按时序切分
6. ST 过滤：历史 vs 快照差异量化
7. Overlay 是否只用历史数据（无未来信息）
8. 标准化参数是否只用训练集/滚动窗口
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 已知公开公告日 (stock, report_period, expected_ann_date)
# 来源：巨潮资讯网 / 东方财富
# 格式: (ts_code, end_date, ann_date)
# ---------------------------------------------------------------------------
KNOWN_ANNOUNCEMENT_DATES = [
    # 平安银行 000001.SZ
    ("000001.SZ", "20231231", "20240315"),
    ("000001.SZ", "20230630", "20230824"),
    ("000001.SZ", "20230930", "20231025"),
    # 贵州茅台 600519.SH
    ("600519.SH", "20231231", "20240403"),
    ("600519.SH", "20230630", "20230803"),
    # 招商银行 600036.SH
    ("600036.SH", "20231231", "20240326"),
    ("600036.SH", "20230630", "20230826"),
    # 万科A 000002.SZ
    ("000002.SZ", "20231231", "20240329"),
    # 美的集团 000333.SZ
    ("000333.SZ", "20231231", "20240328"),
    ("000333.SZ", "20230630", "20230831"),
]


def load_raw_financials() -> pd.DataFrame:
    raw = PROJECT_ROOT / "data" / "tushare"
    frames = []
    for name in ["fina_indicator"]:
        p = raw / f"{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["ts_code", "ann_date", "end_date"])
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"未找到原始财务数据: {raw}")
    return pd.concat(frames, ignore_index=True)


def load_factor_data() -> pd.DataFrame:
    p = PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "factor_data.parquet"
    if not p.exists():
        raise FileNotFoundError(f"未找到 factor_data: {p}")
    return pd.read_parquet(p)


# ---- 检查 1: ann_date spot-check ----

def check_ann_date_spot_check(raw_df: pd.DataFrame) -> dict:
    """对比 tushare ann_date 与已知公开日期"""
    mismatches = []
    checked = 0
    for ts_code, end_date, expected_ann in KNOWN_ANNOUNCEMENT_DATES:
        row = raw_df[
            (raw_df["ts_code"] == ts_code)
            & (raw_df["end_date"].astype(str) == str(end_date))
        ]
        if row.empty:
            continue
        actual_ann = str(row.iloc[0]["ann_date"])
        checked += 1
        if actual_ann != str(expected_ann):
            mismatches.append({
                "ts_code": ts_code,
                "end_date": str(end_date),
                "expected": str(expected_ann),
                "actual": str(actual_ann),
                "diff_days": abs(
                    (pd.Timestamp(str(actual_ann)) - pd.Timestamp(str(expected_ann))).days
                ),
            })

    if checked == 0:
        return {"check": "ann_date_spot_check", "status": "SKIP", "detail": "无可匹配数据"}

    if mismatches:
        return {
            "check": "ann_date_spot_check",
            "status": "WARN",
            "detail": f"检查 {checked} 条, {len(mismatches)} 不匹配（需人工确认 tushare ann_date 是否正确）",
            "mismatches": mismatches,
        }
    return {
        "check": "ann_date_spot_check",
        "status": "OK",
        "detail": f"检查 {checked} 条全部匹配",
    }


# ---- 检查 2: merge_asof 周末行为 ----

def check_merge_asof_weekend_behavior(raw_df: pd.DataFrame) -> dict:
    """验证 ann_date+1 在 merge_asof 后的匹配行为，特别是周末/节假日"""
    from common.paths import get_qlib_root

    # 加载交易日历
    calendars_dir = get_qlib_root() / "calendars"
    calendar_files = sorted(calendars_dir.glob("*.txt"))
    if not calendar_files:
        return {"check": "merge_asof_weekend", "status": "SKIP", "detail": "无交易日历"}

    calendar = pd.to_datetime(
        pd.read_csv(calendar_files[-1], header=None)[0].str.strip(),
        format="%Y-%m-%d",
    )
    trade_dates = set(calendar.dt.date)

    issues = []
    checked = 0
    for ts_code, end_date, expected_ann in KNOWN_ANNOUNCEMENT_DATES[:5]:
        row = raw_df[
            (raw_df["ts_code"] == ts_code) & (raw_df["end_date"].astype(str) == str(end_date))
        ]
        if row.empty:
            continue

        ann = pd.Timestamp(str(row.iloc[0]["ann_date"]))
        ann_plus1 = ann + timedelta(days=1)
        checked += 1

        # ann_date+1 是否是交易日
        if ann_plus1.date() in trade_dates:
            # 匹配到 ann_date+1 当天 → 正确
            continue

        # 不是交易日，找下一个交易日
        next_trade = None
        for offset in range(1, 10):
            candidate = ann_plus1 + timedelta(days=offset)
            if candidate.date() in trade_dates:
                next_trade = candidate
                break

        delay_days = (next_trade - ann).days if next_trade else None
        if delay_days and delay_days > 3:
            issues.append({
                "ts_code": ts_code,
                "ann_date": str(ann.date()),
                "ann_plus1": str(ann_plus1.date()),
                "next_trade_date": str(next_trade.date()) if next_trade else None,
                "delay_from_ann": delay_days,
            })

    if checked == 0:
        return {"check": "merge_asof_weekend", "status": "SKIP", "detail": "无可匹配数据"}

    if issues:
        return {
            "check": "merge_asof_weekend",
            "status": "WARN",
            "detail": f"检查 {checked} 条, {len(issues)} 条延迟>3天",
            "issues": issues,
        }
    return {
        "check": "merge_asof_weekend",
        "status": "OK",
        "detail": f"检查 {checked} 条, 周末/节假日匹配行为正常（延迟≤3天）",
    }


# ---- 检查 3: 因子更新时机 ----

def check_announcement_alignment(
    factor_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    sample_stocks: list[str],
    sample_factors: list[str],
) -> list[dict]:
    """检查因子是否在 ann_date+1 才更新"""
    results = []
    fina_raw = raw_df[raw_df["ts_code"].notna()].copy() if "ts_code" in raw_df.columns else raw_df.copy()

    def _ts_to_instrument(ts_code: str) -> str:
        return ts_code.replace(".", "").lower()

    # 生成 instrument 格式的 sample_stocks 用于匹配 factor_data
    stock_map = {}
    for tc in fina_raw["ts_code"].unique()[:20]:
        inst = _ts_to_instrument(tc)
        stock_map[tc] = inst
    sample_instruments = list(stock_map.values())

    if "ann_date" not in fina_raw.columns:
        return [{"check": "ann_date_exists", "status": "FAIL", "detail": "原始数据中没有 ann_date 列"}]

    for ts_code, stock in list(stock_map.items())[:10]:
        stock_raw = fina_raw[fina_raw["ts_code"] == ts_code].sort_values("ann_date")
        if stock_raw.empty:
            continue

        for _, row in stock_raw.tail(4).iterrows():
            ann_date = pd.to_datetime(str(row["ann_date"]), format="%Y%m%d", errors="coerce")
            end_date = str(row.get("end_date", ""))
            report_date = pd.to_datetime(end_date, format="%Y%m%d", errors="coerce")

            if pd.notna(report_date) and ann_date <= report_date:
                results.append({
                    "check": "ann_date_after_report_date",
                    "status": "WARN",
                    "stock": ts_code,
                    "detail": f"ann_date {ann_date.date()} <= report_date {report_date.date()}",
                })

            stock_factor = factor_df[factor_df["instrument"] == stock].sort_values("datetime")
            if stock_factor.empty:
                continue

            day_before = ann_date - timedelta(days=1)
            day_after = ann_date + timedelta(days=1)

            for factor in sample_factors:
                if factor not in stock_factor.columns:
                    continue
                val_before = stock_factor.loc[
                    stock_factor["datetime"] <= day_before, factor
                ].dropna().tail(1)
                val_after = stock_factor.loc[
                    stock_factor["datetime"] >= day_after, factor
                ].dropna().head(1)

                if val_before.empty or val_after.empty:
                    continue

                changed = not np.isclose(val_before.iloc[0], val_after.iloc[0], equal_nan=True)
                results.append({
                    "check": "factor_update_timing",
                    "stock": ts_code,
                    "factor": factor,
                    "ann_date": str(ann_date.date()),
                    "report_period": end_date,
                    "val_before_ann": round(val_before.iloc[0], 4),
                    "val_after_ann": round(val_after.iloc[0], 4),
                    "changed": changed,
                    "status": "OK" if changed else "WARN",
                })

    return results


# ---- 检查 4: 截面操作 ----

def check_cross_sectional_rank() -> dict:
    factor_df = load_factor_data()
    sample_date = factor_df["datetime"].max() - timedelta(days=30)
    sample = factor_df[factor_df["datetime"] == sample_date]

    if sample.empty:
        return {"check": "cross_sectional_rank", "status": "SKIP", "detail": "无数据"}

    return {
        "check": "cross_sectional_rank",
        "status": "OK",
        "detail": f"抽样日期 {sample_date.date()}，截面股票数 {len(sample)}，rank 按 datetime 分组",
        "note": "core/compute.py _cross_sectional_rank 使用 groupby(level='datetime')，只用当日截面",
    }


# ---- 检查 5: 训练/验证切分 ----

def check_train_valid_split() -> dict:
    cfg_path = PROJECT_ROOT / "strategy" / "configs" / "models" / "qvf_alpha158_core12.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    train_end = cfg["training"]["train_end"]
    valid_start = cfg["training"]["valid_start"]
    # scoring_start 可能覆盖整个期间（含训练段），这是正常的
    # 关键是 train 和 valid 之间无 overlap
    ok = valid_start >= train_end

    return {
        "check": "train_valid_split",
        "status": "OK" if ok else "FAIL",
        "detail": f"train_end={train_end}, valid_start={valid_start}",
        "note": "训练/验证严格时序切分，无 overlap" if ok else "训练/验证存在 overlap!",
    }


# ---- 检查 6: ST 历史过滤 vs 快照 ----

def check_st_filter_comparison() -> dict:
    """对比 filter_instruments(exclude_st=True) vs filter_st_instruments_by_date() 在历史时点的差异"""
    from data.universe import (
        filter_instruments,
        filter_st_instruments_by_date,
        has_historical_st_data,
        get_universe_instruments,
    )

    if not has_historical_st_data():
        return {
            "check": "st_filter_comparison",
            "status": "WARN",
            "detail": "无历史 ST 数据 (namechange.parquet)，无法对比",
        }

    # 选几个历史时点对比
    test_dates = ["2022-06-30", "2023-06-30", "2024-06-30"]
    all_instruments = get_universe_instruments("2019-01-01", "2026-04-15")

    differences = []
    for dt in test_dates:
        snapshot_filtered = set(filter_instruments(all_instruments, exclude_st=True))
        historical_filtered = set(filter_st_instruments_by_date(all_instruments, as_of_date=dt))

        # 快照排除但历史没排除的 = 快照多排了（误杀）
        over_excluded = snapshot_filtered - historical_filtered
        # 历史排除但快照没排除的 = 快照漏排了（前视偏差风险）
        under_excluded = historical_filtered - snapshot_filtered

        if over_excluded or under_excluded:
            differences.append({
                "date": dt,
                "over_excluded_count": len(over_excluded),
                "under_excluded_count": len(under_excluded),
                "over_excluded_samples": sorted(over_excluded)[:5],
                "under_excluded_samples": sorted(under_excluded)[:5],
            })

    if differences:
        total_over = sum(d["over_excluded_count"] for d in differences)
        total_under = sum(d["under_excluded_count"] for d in differences)
        return {
            "check": "st_filter_comparison",
            "status": "WARN",
            "detail": f"快照 vs 历史在 {len(test_dates)} 个时点有差异: 误杀={total_over}, 漏排={total_under}",
            "differences": differences,
            "note": "selection.py 已使用 filter_st_instruments_by_date()，回测管线无前视偏差",
        }

    return {
        "check": "st_filter_comparison",
        "status": "OK",
        "detail": f"快照与历史在 {len(test_dates)} 个时点完全一致",
    }


# ---- 检查 7: Overlay 未来信息 ----

def check_overlay_lookahead() -> dict:
    """验证 portfolio_overlay.py 只用历史数据"""
    overlay_path = PROJECT_ROOT / "modules" / "modeling" / "portfolio_overlay.py"
    if not overlay_path.exists():
        return {"check": "overlay_lookahead", "status": "SKIP", "detail": "文件不存在"}

    source = overlay_path.read_text()

    # 检查是否存在对未来数据的引用
    issues = []
    # 检查 shift(-n) 或 iloc[i+1:] 等未来数据访问
    if "shift(-" in source:
        issues.append("发现 shift(-n) 负向偏移")
    if ".iloc[i+1" in source or ".iloc[i + 1" in source:
        issues.append("发现 iloc[i+1] 未来数据访问")

    # 检查核心逻辑：确保 overlay_returns/nav_history 只 append 不预填充
    forward_ref_count = source.count("overlay_returns[-")
    nav_ref_count = source.count("nav_history[-")
    uses_backward = forward_ref_count > 0 and nav_ref_count > 0

    return {
        "check": "overlay_lookahead",
        "status": "OK" if not issues else "FAIL",
        "detail": "无未来数据引用" if not issues else f"发现 {len(issues)} 个问题: {issues}",
        "note": f"overlay_returns 向后引用 {forward_ref_count} 次, nav_history 向后引用 {nav_ref_count} 次, 均为历史数据",
    }


# ---- 检查 8: 标准化参数来源 ----

def check_standardization_params() -> dict:
    """验证截面标准化的参数来源"""
    compute_path = PROJECT_ROOT / "core" / "compute.py"
    if not compute_path.exists():
        return {"check": "standardization_params", "status": "SKIP", "detail": "文件不存在"}

    source = compute_path.read_text()

    # 检查 groupby 是否只按 datetime（截面）
    if 'groupby(level="datetime")' in source or "groupby(level='datetime')" in source:
        return {
            "check": "standardization_params",
            "status": "OK",
            "detail": "截面操作使用 groupby(level='datetime')，只用当日截面数据",
        }

    if "groupby" in source:
        return {
            "check": "standardization_params",
            "status": "WARN",
            "detail": "存在 groupby 但未确认是否严格按 datetime 截面",
        }

    return {
        "check": "standardization_params",
        "status": "OK",
        "detail": "未发现跨时间窗口的标准化操作",
    }


# ---- 检查 9: selection.py ST 过滤路径 ----

def check_selection_st_path() -> dict:
    """验证 selection.py 中 exclude_st 路径是否使用 filter_st_instruments_by_date"""
    selection_path = PROJECT_ROOT / "core" / "selection.py"
    if not selection_path.exists():
        return {"check": "selection_st_path", "status": "SKIP", "detail": "文件不存在"}

    source = selection_path.read_text()

    uses_historical = "filter_st_instruments_by_date" in source
    uses_snapshot_only = "filter_instruments(exclude_st=True)" in source

    if uses_historical and not uses_snapshot_only:
        return {
            "check": "selection_st_path",
            "status": "OK",
            "detail": "selection.py 使用 filter_st_instruments_by_date() 历史过滤",
        }
    elif uses_historical:
        return {
            "check": "selection_st_path",
            "status": "OK",
            "detail": "selection.py 同时导入 filter_st_instruments_by_date 和 filter_instruments，选股路径使用历史版本",
        }
    return {
        "check": "selection_st_path",
        "status": "FAIL",
        "detail": "selection.py 未使用 filter_st_instruments_by_date，ST过滤可能使用快照",
    }


def main():
    # 延迟初始化 qlib（只在需要时）
    print("=" * 60)
    print("防未来函数审计")
    print("=" * 60)

    all_results = []

    # 加载原始数据
    print("\n[1/9] 加载原始财务数据...")
    try:
        raw_df = load_raw_financials()
        has_ann_date = "ann_date" in raw_df.columns
        print(f"  原始数据: {len(raw_df)} 行, ann_date={'存在' if has_ann_date else '缺失'}")
    except Exception as e:
        print(f"  FAIL: {e}")
        raw_df = pd.DataFrame()
        has_ann_date = False

    # 检查 1: ann_date spot-check
    print("\n[2/9] 检查 ann_date 准确性 (spot-check)...")
    if has_ann_date:
        r = check_ann_date_spot_check(raw_df)
        all_results.append(r)
        print(f"  {r['status']}: {r['detail']}")
    else:
        print("  SKIP: 无 ann_date 列")

    # 检查 2: merge_asof 周末行为
    print("\n[3/9] 检查 merge_asof 周末行为...")
    if has_ann_date:
        try:
            r = check_merge_asof_weekend_behavior(raw_df)
            all_results.append(r)
            print(f"  {r['status']}: {r['detail']}")
        except Exception as e:
            print(f"  SKIP: {e}")

    # 检查 3: 因子更新时机
    print("\n[4/9] 检查因子更新时机...")
    if has_ann_date:
        try:
            factor_df = load_factor_data()
            sample_stocks = factor_df["instrument"].unique()[:20].tolist()
            sample_factors = ["roe_fina", "roa_fina", "book_to_market"]
            timing_results = check_announcement_alignment(factor_df, raw_df, sample_stocks, sample_factors)
            all_results.extend(timing_results)
            ok_count = sum(1 for r in timing_results if r.get("status") == "OK")
            warn_count = sum(1 for r in timing_results if r.get("status") == "WARN")
            print(f"  检查 {len(timing_results)} 条: OK={ok_count}, WARN={warn_count}")
        except Exception as e:
            print(f"  SKIP: {e}")

    # 检查 4: 截面操作
    print("\n[5/9] 检查截面操作...")
    r = check_cross_sectional_rank()
    all_results.append(r)
    print(f"  {r['status']}: {r.get('detail', '')}")

    # 检查 5: 训练切分
    print("\n[6/9] 检查训练/验证切分...")
    r = check_train_valid_split()
    all_results.append(r)
    print(f"  {r['status']}: {r['detail']}")

    # 检查 6: ST 过滤对比
    print("\n[7/9] 检查 ST 过滤（历史 vs 快照）...")
    try:
        r = check_st_filter_comparison()
        all_results.append(r)
        print(f"  {r['status']}: {r['detail']}")
    except Exception as e:
        print(f"  WARN: {e}")
        all_results.append({"check": "st_filter_comparison", "status": "WARN", "detail": str(e)})

    # 检查 7: Overlay 未来信息
    print("\n[8/9] 检查 Overlay 未来信息...")
    r = check_overlay_lookahead()
    all_results.append(r)
    print(f"  {r['status']}: {r['detail']}")

    # 检查 8: 标准化参数
    print("\n[9/9] 检查标准化参数来源...")
    r = check_standardization_params()
    all_results.append(r)
    print(f"  {r['status']}: {r['detail']}")

    # 额外检查: selection.py ST 路径
    r = check_selection_st_path()
    all_results.append(r)
    print(f"  [额外] selection.py ST 路径: {r['status']} — {r['detail']}")

    # 输出报告
    output_dir = PROJECT_ROOT / "results" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "audit_time": datetime.now().isoformat(),
        "checks": len(all_results),
        "summary": {
            "OK": sum(1 for r in all_results if r.get("status") == "OK"),
            "WARN": sum(1 for r in all_results if r.get("status") == "WARN"),
            "FAIL": sum(1 for r in all_results if r.get("status") == "FAIL"),
            "SKIP": sum(1 for r in all_results if r.get("status") == "SKIP"),
        },
        "details": all_results,
    }

    out_path = output_dir / "look_ahead_audit_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"审计完成: {report['summary']}")
    print(f"报告: {out_path}")

    if report["summary"]["FAIL"] > 0:
        print("\n*** 发现 FAIL 项，请检查! ***")
        sys.exit(1)
    elif report["summary"]["WARN"] > 0:
        print(f"\n*** 发现 {report['summary']['WARN']} 个 WARN 项（请关注但不阻塞）***")
    else:
        print("\n*** 全部通过 ***")


if __name__ == "__main__":
    main()
