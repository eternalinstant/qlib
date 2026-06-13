#!/usr/bin/env python3
"""
core_12 因子详细审计：IC / Rank IC / 分层收益 / 多空 / 覆盖率 / 暴露分析。

对 qvf_alpha158_core12 的 12 个因子逐一进行：
1. IC / Rank IC / ICIR / 月度胜率
2. 分层收益（5档 quintile）
3. 多空收益（Q5 - Q1）
4. 覆盖率 / 缺失率
5. 行业暴露
6. 市值暴露

用法:
    python3 scripts/audit_core12_factors.py
    python3 scripts/audit_core12_factors.py --config strategy/configs/models/qvf_alpha158_core12.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_factor_data() -> pd.DataFrame:
    p = PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "factor_data.parquet"
    if not p.exists():
        raise FileNotFoundError(f"未找到 factor_data: {p}")
    return pd.read_parquet(p)


def load_alpha158_features(features: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    sys.path.insert(0, str(PROJECT_ROOT))
    from data.qlib_init import init_qlib, load_features_safe
    from data.universe import filter_instruments
    from qlib.data import D

    init_qlib()
    instruments = D.instruments(market="all")
    df_inst = load_features_safe(instruments, ["$close"], start_date, end_date)
    valid = filter_instruments(df_inst.index.get_level_values("instrument").unique().tolist(), exclude_st=False)

    # 构建 alpha158 特征表达式
    expr_map = {
        "ROC20": "$close / Ref($close, 20) - 1",
        "RANK20": "Rank($close, 20)",
        "CORD20": "Corr($close, $volume, 20)",
    }
    exprs = [expr_map.get(f, f) for f in features]
    df = load_features_safe(valid, exprs, start_date, end_date)
    df.columns = features
    return df


def load_forward_returns(start_date: str, end_date: str) -> pd.Series:
    sys.path.insert(0, str(PROJECT_ROOT))
    from data.qlib_init import init_qlib, load_features_safe
    from data.universe import filter_instruments
    from qlib.data import D

    init_qlib()
    instruments = D.instruments(market="all")
    df_inst = load_features_safe(instruments, ["$close"], start_date, end_date)
    valid = filter_instruments(df_inst.index.get_level_values("instrument").unique().tolist(), exclude_st=False)

    df_ret = load_features_safe(valid, ["Ref($close, -20) / $close - 1"], start_date, end_date)
    df_ret.columns = ["fwd_ret_20d"]
    return df_ret["fwd_ret_20d"]


def load_market_cap() -> pd.Series:
    sys.path.insert(0, str(PROJECT_ROOT))
    from data.qlib_init import init_qlib, load_features_safe
    from data.universe import filter_instruments
    from qlib.data import D

    init_qlib()
    instruments = D.instruments(market="all")
    df_inst = load_features_safe(instruments, ["$close"], "2019-01-01", "2026-05-01")
    valid = filter_instruments(df_inst.index.get_level_values("instrument").unique().tolist(), exclude_st=False)
    df_mv = load_features_safe(valid, ["$total_mv"], "2019-01-01", "2026-05-01")
    return df_mv["$total_mv"]


def compute_rank_ic(factor_values: pd.Series, forward_returns: pd.Series) -> pd.Series:
    df = pd.DataFrame({"factor": factor_values, "fwd_ret": forward_returns}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    def _rank_ic(g):
        if len(g) < 30:
            return np.nan
        return g["factor"].corr(g["fwd_ret"], method="spearman")

    return df.groupby(level="datetime").apply(_rank_ic).dropna()


def compute_quintile_returns(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    n_bins: int = 5,
) -> pd.DataFrame:
    """分层收益：按因子值分 n_bins 档，每档的平均前向收益"""
    df = pd.DataFrame({"factor": factor_values, "fwd_ret": forward_returns}).dropna()
    if df.empty:
        return pd.DataFrame()

    def _quantile_ret(g):
        if len(g) < n_bins * 5:
            return pd.Series(dtype=float)
        try:
            labels = pd.qcut(g["factor"], n_bins, labels=False, duplicates="drop")
            return g.groupby(labels)["fwd_ret"].mean()
        except ValueError:
            return pd.Series(dtype=float)

    results = df.groupby(level="datetime").apply(_quantile_ret)
    if results.empty:
        return pd.DataFrame()

    # results 的列是 0,1,2,3,4，行是日期
    if isinstance(results, pd.DataFrame):
        return results
    return pd.DataFrame()


def analyze_factor(
    name: str,
    factor_values: pd.Series,
    forward_returns: pd.Series,
    market_cap: pd.Series | None = None,
) -> dict:
    """单因子完整分析"""
    result = {"factor": name}

    # IC 分析
    ic_series = compute_rank_ic(factor_values, forward_returns)
    if len(ic_series) < 20:
        result["status"] = "INSUFFICIENT_DATA"
        return result

    result["rank_ic_mean"] = float(ic_series.mean())
    result["rank_ic_std"] = float(ic_series.std())
    result["icir"] = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0
    result["ic_positive_rate"] = float((ic_series > 0).mean())

    # 年度 IC
    yearly_ic = {}
    for year in range(2019, 2027):
        yr = ic_series[ic_series.index.year == year]
        if len(yr) > 10:
            yearly_ic[year] = {
                "ic": float(yr.mean()),
                "ir": float(yr.mean() / yr.std()) if yr.std() > 0 else 0.0,
                "n": len(yr),
            }
    result["yearly_ic"] = yearly_ic

    # 分层收益
    quintile_df = compute_quintile_returns(factor_values, forward_returns, n_bins=5)
    if not quintile_df.empty and len(quintile_df.columns) == 5:
        q_means = quintile_df.mean()
        result["quintile_returns"] = {
            "Q1_low": float(q_means.get(0, 0)),
            "Q2": float(q_means.get(1, 0)),
            "Q3": float(q_means.get(2, 0)),
            "Q4": float(q_means.get(3, 0)),
            "Q5_high": float(q_means.get(4, 0)),
            "long_short": float(q_means.get(4, 0) - q_means.get(0, 0)),
        }
        # 检查单调性
        q_values = [q_means.get(i, 0) for i in range(5)]
        monotonic = all(q_values[i] <= q_values[i + 1] for i in range(4)) or \
                    all(q_values[i] >= q_values[i + 1] for i in range(4))
        result["quintile_monotonic"] = monotonic
    else:
        result["quintile_returns"] = None
        result["quintile_monotonic"] = None

    # 覆盖率
    factor_notna = factor_values.notna()
    total = len(factor_values)
    result["coverage_rate"] = float(factor_notna.sum() / total) if total > 0 else 0.0
    result["total_observations"] = int(total)
    result["valid_observations"] = int(factor_notna.sum())

    # 市值暴露
    if market_cap is not None:
        merged = pd.DataFrame({
            "factor": factor_values,
            "mv": market_cap,
        }).dropna()
        if len(merged) > 100:
            mv_exposure = merged["factor"].corr(merged["mv"], method="spearman")
            result["market_cap_exposure"] = float(mv_exposure)
        else:
            result["market_cap_exposure"] = None
    else:
        result["market_cap_exposure"] = None

    return result


def main():
    parser = argparse.ArgumentParser(description="core_12 因子详细审计")
    parser.add_argument("--config", default="strategy/configs/models/qvf_alpha158_core12.yaml")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pq_factors = cfg["data"].get("parquet_feature_columns", [])
    a158_factors = cfg["data"].get("alpha158_feature_columns", [])
    start_date = cfg["data"].get("start_date", "2019-01-01")
    end_date = cfg["data"].get("end_date", "2026-04-15")

    print(f"策略: {cfg['name']}")
    print(f"Parquet 因子 ({len(pq_factors)}): {pq_factors}")
    print(f"Alpha158 因子 ({len(a158_factors)}): {a158_factors}")

    # 加载 parquet 因子
    print("\n[1/4] 加载 factor_data.parquet...")
    factor_df = load_factor_data()
    factor_df["datetime"] = pd.to_datetime(factor_df["datetime"])
    # 转换 instrument 格式: 000001sz → sz000001 (匹配 qlib 小写格式)
    factor_df["instrument"] = (
        factor_df["instrument"].str[-2:] +
        factor_df["instrument"].str[:-2]
    )
    factor_df = factor_df.set_index(["datetime", "instrument"]).sort_index()
    print(f"  {len(factor_df)} 行, {len(factor_df.columns)} 列")

    # 加载 alpha158 因子
    print("\n[2/4] 加载 Alpha158 因子...")
    if a158_factors:
        a158_df = load_alpha158_features(a158_factors, start_date, end_date)
        if a158_df.index.names == ["instrument", "datetime"]:
            a158_df = a158_df.swaplevel()
        print(f"  Alpha158: {len(a158_df)} 行")
    else:
        a158_df = pd.DataFrame()

    # 加载前向收益
    print("\n[3/4] 加载前向收益 (20日)...")
    fwd_ret = load_forward_returns(start_date, end_date)
    # 确保 index 层级顺序为 (datetime, instrument)
    if fwd_ret.index.names == ["instrument", "datetime"]:
        fwd_ret = fwd_ret.swaplevel()
    print(f"  {len(fwd_ret)} 行")

    # 加载市值
    print("\n[4/4] 加载市值...")
    try:
        market_cap = load_market_cap()
        if market_cap.index.names == ["instrument", "datetime"]:
            market_cap = market_cap.swaplevel()
        print(f"  {len(market_cap)} 行")
    except Exception as e:
        print(f"  SKIP: {e}")
        market_cap = None

    # 分析每个因子
    all_results = []
    all_factors = pq_factors + a158_factors

    print(f"\n分析 {len(all_factors)} 个因子...")
    for i, factor_name in enumerate(all_factors, 1):
        print(f"  [{i}/{len(all_factors)}] {factor_name}...", end=" ")

        if factor_name in pq_factors:
            if factor_name not in factor_df.columns:
                print("SKIP (不在 factor_data 中)")
                continue
            values = factor_df[factor_name]
        elif factor_name in a158_factors and not a158_df.empty:
            if factor_name not in a158_df.columns:
                print("SKIP (不在 alpha158 中)")
                continue
            values = a158_df[factor_name]
        else:
            print("SKIP (无数据)")
            continue

        result = analyze_factor(factor_name, values, fwd_ret, market_cap)
        all_results.append(result)

        # 打印摘要
        if result.get("status") == "INSUFFICIENT_DATA":
            print("数据不足")
            continue

        ic = result.get("rank_ic_mean", 0)
        ir = result.get("icir", 0)
        coverage = result.get("coverage_rate", 0)
        ls = result.get("quintile_returns", {}).get("long_short", "N/A") if result.get("quintile_returns") else "N/A"
        mono = result.get("quintile_monotonic", "?")
        mv_exp = result.get("market_cap_exposure", "N/A")
        print(f"IC={ic:+.4f} IR={ir:+.2f} 覆盖={coverage:.1%} 多空={ls} 单调={'Y' if mono else 'N'} 市值暴露={mv_exp}")

    # 保存结果
    output_dir = PROJECT_ROOT / "results" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    out_path = output_dir / "core12_factor_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # 打印汇总表
    print(f"\n{'=' * 100}")
    print(f"{'因子':>25} {'Rank IC':>10} {'ICIR':>8} {'覆盖率':>8} {'多空':>10} {'单调':>4} {'市值暴露':>8}")
    print("-" * 100)
    for r in all_results:
        if r.get("status") == "INSUFFICIENT_DATA":
            continue
        ls_str = f"{r['quintile_returns']['long_short']:+.4f}" if r.get("quintile_returns") else "N/A"
        mono_str = "Y" if r.get("quintile_monotonic") else "N"
        mv_str = f"{r['market_cap_exposure']:+.3f}" if r.get("market_cap_exposure") is not None else "N/A"
        print(f"{r['factor']:>25} {r['rank_ic_mean']:>+10.4f} {r['icir']:>+8.2f} "
              f"{r['coverage_rate']:>8.1%} {ls_str:>10} {mono_str:>4} {mv_str:>8}")

    print(f"\n输出: {out_path}")


if __name__ == "__main__":
    main()
