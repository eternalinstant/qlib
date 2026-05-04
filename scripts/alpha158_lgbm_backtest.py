"""
Alpha158 LightGBM 超参搜索 — 年化>20%, 回撤<10%

流程:
  1. 一次性加载全量因子数据 + 全部价格数据
  2. 对每组超参: 固定训练(2019-2021) → 预测(2022-至今)
  3. 构建选股 → 验证集(2022-2023) 回测
  4. 筛选年化>20% & 回撤<10% 的组合
  5. 对筛选出的组合跑全量(2019-至今)回测

增量保存: 每完成一个组合立刻追加 CSV，中途挂掉不丢数据。
支持断点续跑: 检测已有 CSV，跳过已完成的 tag。

用法:
  PYTHONPATH=. python scripts/alpha158_lgbm_backtest.py
"""

import sys
import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CONFIG
from core.strategy import Strategy
from core.selection import load_factor_data, compute_signal, extract_topk
from core.lgbm_scorer import train_lgbm_model, predict_with_model

# ── 超参搜索空间 ──────────────────────────────────────────────

SEARCH_SPACE = {
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_samples": [100, 200, 500],
    "n_estimators": [100, 200, 500],
    "forward_days": [5, 10, 20],
    "topk": [5, 10, 15, 20],
}

BASE_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
}

TRAIN_START = "2019-01-01"
TRAIN_END = "2021-12-31"
VALIDATE_START = "2022-01-01"
VALIDATE_END = "2023-12-31"
FULL_START = "2019-01-01"

OUTPUT_PATH = PROJECT_ROOT / "results" / "alpha158_lgbm_hptune.csv"
OUTPUT_COLS = [
    "tag", "num_leaves", "learning_rate", "min_child_samples",
    "n_estimators", "forward_days", "topk",
    "val_annual", "val_max_dd", "val_sharpe",
    "full_annual", "full_max_dd", "full_sharpe",
    "elapsed_s",
]


def _compute_metrics(daily_returns: pd.Series) -> dict:
    """从 daily_returns 计算年化、最大回撤、夏普。"""
    if daily_returns.empty or len(daily_returns) < 10:
        return {"annual": 0.0, "max_dd": 0.0, "sharpe": 0.0}

    pv = (1 + daily_returns).cumprod()
    days = (pv.index[-1] - pv.index[0]).days
    if days <= 0 or pv.iloc[-1] <= 0:
        return {"annual": 0.0, "max_dd": -1.0, "sharpe": 0.0}

    annual = pv.iloc[-1] ** (365 / days) - 1
    max_dd = float((pv / pv.cummax() - 1).min())
    std = daily_returns.std()
    sharpe = daily_returns.mean() / std * np.sqrt(252) if std > 0 else 0.0
    return {"annual": annual, "max_dd": max_dd, "sharpe": sharpe}


def _load_price_data(all_symbols, start_date, end_date):
    """一次性加载全部所需股票的日线 close，返回 returns_pivot (datetime × instrument)。"""
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D

    try:
        qlib.init(provider_uri=CONFIG.get("qlib_data_path"), region=REG_CN)
    except Exception:
        pass

    price_start = pd.Timestamp(start_date) - pd.Timedelta(days=10)
    price_end = pd.Timestamp(end_date) + pd.Timedelta(days=30)

    price_df = D.features(
        all_symbols, ["$close"],
        price_start.strftime("%Y-%m-%d"),
        price_end.strftime("%Y-%m-%d"),
        "day",
    )
    if price_df.empty:
        return pd.DataFrame()

    price_df.columns = ["close"]
    # qlib 返回 (instrument, datetime) 顺序
    price_df.index = pd.MultiIndex.from_tuples(
        [(dt, sym.split("/")[-1].upper() if "/" in str(sym) else str(sym).upper())
         for sym, dt in price_df.index],
        names=["datetime", "instrument"],
    )

    price_pivot = price_df["close"].unstack(level="instrument")
    price_pivot.index = pd.to_datetime(price_pivot.index)
    return price_pivot.pct_change()


# ── 价格数据缓存（懒加载） ──────────────────────────────────────
_price_cache: dict[str, pd.DataFrame] = {}


def _get_returns_pivot(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """获取 returns_pivot，按日期范围缓存。"""
    cache_key = f"{start_date}_{end_date}"
    if cache_key not in _price_cache:
        print(f"  加载价格数据: {len(symbols)} 只股票, {start_date} ~ {end_date}...")
        _price_cache[cache_key] = _load_price_data(symbols, start_date, end_date)
    return _price_cache[cache_key]


def _compute_portfolio_returns(
    df_sel: pd.DataFrame,
    returns_pivot: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """根据选股列表 + 预加载的 returns_pivot 计算等权组合 daily returns。"""
    if df_sel.empty or returns_pivot.empty:
        return pd.Series(dtype=float)

    dates = sorted(pd.to_datetime(df_sel["date"]).unique())
    if not dates:
        return pd.Series(dtype=float)

    portfolio_returns = {}
    for i in range(len(dates)):
        rebalance_date = dates[i]
        next_rebalance = (
            dates[i + 1] if i + 1 < len(dates)
            else pd.Timestamp(end_date) + pd.Timedelta(days=30)
        )

        holdings = df_sel[df_sel["date"] == rebalance_date]["symbol"].unique().tolist()
        available = [h for h in holdings if h in returns_pivot.columns]
        if not available:
            continue

        mask = (returns_pivot.index > rebalance_date) & (returns_pivot.index <= next_rebalance)
        period_returns = returns_pivot.loc[mask, available]

        for dt in period_returns.index:
            day_ret = period_returns.loc[dt].dropna()
            if len(day_ret) > 0:
                portfolio_returns[dt] = day_ret.mean()

    if not portfolio_returns:
        return pd.Series(dtype=float)

    s = pd.Series(portfolio_returns).sort_index()
    s.index.name = "datetime"
    s = s[(s.index >= pd.Timestamp(start_date)) & (s.index <= pd.Timestamp(end_date))]
    return s


def _append_result(row: dict, path: Path):
    """追加一行结果到 CSV（断点安全）。"""
    df = pd.DataFrame([row])
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


def _load_done_tags(path: Path) -> set:
    """读取已完成的 tag 集合，用于断点续跑。"""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, usecols=["tag"])
    return set(df["tag"].tolist())


def _full_backtest_for_tag(
    tag: str,
    params: dict,
    monthly_df: pd.DataFrame,
    rebalance_dates,
    strategy,
    all_instruments: list[str],
    data_end: str,
) -> dict | None:
    """对一个 tag 重新生成选股并计算全量回测指标。"""
    forward_days = params["forward_days"]
    topk = params["topk"]

    lgbm_params = {
        **BASE_LGBM_PARAMS,
        "num_leaves": params["num_leaves"],
        "learning_rate": params["learning_rate"],
        "min_child_samples": params["min_child_samples"],
        "n_estimators": params["n_estimators"],
    }

    model, feature_cols, df_neutralized = train_lgbm_model(
        monthly_df,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        forward_days=forward_days,
        lgbm_params=lgbm_params,
        neutralize_industry=True,
    )
    signal = predict_with_model(model, df_neutralized, feature_cols)
    df_sel = extract_topk(
        signal, rebalance_dates,
        topk=topk,
        exclude_new_days=strategy.exclude_new_days,
        exclude_st=strategy.exclude_st,
        universe=strategy.universe,
    )
    if df_sel.empty:
        return None

    full_returns_pivot = _get_returns_pivot(all_instruments, FULL_START, data_end)
    full_returns = _compute_portfolio_returns(df_sel, full_returns_pivot, FULL_START, data_end)
    return _compute_metrics(full_returns)


def main():
    print("=" * 70)
    print("Alpha158 LightGBM 超参搜索")
    print(f"训练集: {TRAIN_START} ~ {TRAIN_END}")
    print(f"验证集: {VALIDATE_START} ~ {VALIDATE_END}")

    param_keys = list(SEARCH_SPACE.keys())
    param_values = list(SEARCH_SPACE.values())
    total_combos = 1
    for v in param_values:
        total_combos *= len(v)
    print(f"搜索空间: {total_combos} 种组合")
    print(f"结果文件: {OUTPUT_PATH}")
    print("=" * 70)

    # Step 1: 加载策略和全量因子数据
    print("\n[Step 1] 加载因子数据...")
    t0 = time.perf_counter()

    strategy = Strategy.load("experimental/alpha158/alpha158_csi300")

    data_start = "2018-12-01"
    data_end = CONFIG.get("end_date", "2026-02-26")

    monthly_df, rebalance_dates = load_factor_data(
        registry=strategy.registry,
        start_date=data_start,
        end_date=data_end,
        rebalance_freq=strategy.rebalance_freq,
        universe=strategy.universe,
        factor_window_scale=strategy.factor_window_scale,
    )
    print(f"  因子数据: {monthly_df.shape}, 耗时 {time.perf_counter() - t0:.1f}s")

    # 获取所有涉及的股票代码（用命名级别）
    all_instruments = monthly_df.index.get_level_values("instrument").unique().tolist()
    print(f"  涉及股票: {len(all_instruments)} 只")

    # Step 1b: 断点续跑 — 加载已完成的 tag
    done_tags = _load_done_tags(OUTPUT_PATH)
    if done_tags:
        print(f"\n  断点续跑: 已完成 {len(done_tags)} 个组合，跳过")

    # Step 2: 对每组超参进行训练+预测+验证
    print(f"\n[Step 2] 开始搜索 ({total_combos} 组合)...")
    success_count = len(done_tags)
    fail_count = 0

    for combo_idx, combo in enumerate(itertools.product(*param_values)):
        params = dict(zip(param_keys, combo))
        num_leaves = params["num_leaves"]
        learning_rate = params["learning_rate"]
        min_child_samples = params["min_child_samples"]
        n_estimators = params["n_estimators"]
        forward_days = params["forward_days"]
        topk = params["topk"]

        tag = (f"nl{num_leaves}_lr{learning_rate}_mcs{min_child_samples}"
               f"_ne{n_estimators}_fd{forward_days}_k{topk}")

        # 断点续跑: 跳过已完成的
        if tag in done_tags:
            continue

        show = (combo_idx + 1) % 50 == 0 or combo_idx == 0
        if show:
            print(f"  [{combo_idx+1}/{total_combos}] {tag}", end="", flush=True)

        t1 = time.perf_counter()
        try:
            lgbm_params = {
                **BASE_LGBM_PARAMS,
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "min_child_samples": min_child_samples,
                "n_estimators": n_estimators,
            }

            # 训练
            model, feature_cols, df_neutralized = train_lgbm_model(
                monthly_df,
                train_start=TRAIN_START,
                train_end=TRAIN_END,
                forward_days=forward_days,
                lgbm_params=lgbm_params,
                neutralize_industry=True,
            )

            # 预测
            signal = predict_with_model(model, df_neutralized, feature_cols)

            # 构建选股
            df_sel = extract_topk(
                signal, rebalance_dates,
                topk=topk,
                exclude_new_days=strategy.exclude_new_days,
                exclude_st=strategy.exclude_st,
                universe=strategy.universe,
            )

            if df_sel.empty:
                if show:
                    print(" -> 无选股结果")
                fail_count += 1
                continue

            # 验证集回测 (2022-2023)
            val_returns_pivot = _get_returns_pivot(
                all_instruments, VALIDATE_START, VALIDATE_END
            )
            val_returns = _compute_portfolio_returns(
                df_sel, val_returns_pivot, VALIDATE_START, VALIDATE_END
            )
            val_metrics = _compute_metrics(val_returns)

            elapsed = time.perf_counter() - t1
            if show:
                print(f" -> val_annual={val_metrics['annual']:.2%}  "
                      f"val_dd={val_metrics['max_dd']:.2%}  ({elapsed:.1f}s)")

            # 立刻追加保存（不含 full 回测，Step 3 再补）
            row = {
                "tag": tag,
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "min_child_samples": min_child_samples,
                "n_estimators": n_estimators,
                "forward_days": forward_days,
                "topk": topk,
                "val_annual": val_metrics["annual"],
                "val_max_dd": val_metrics["max_dd"],
                "val_sharpe": val_metrics["sharpe"],
                "full_annual": float("nan"),
                "full_max_dd": float("nan"),
                "full_sharpe": float("nan"),
                "elapsed_s": round(elapsed, 1),
            }
            _append_result(row, OUTPUT_PATH)
            success_count += 1

            # 释放本轮内存
            del model, signal, df_sel, df_neutralized, val_returns

        except Exception as exc:
            if show:
                print(f" -> FAILED: {exc}")
            fail_count += 1

    print(f"\n  搜索完成: {success_count} 成功, {fail_count} 失败")

    # Step 3: 读取全部结果，筛选 + 全量回测
    if not OUTPUT_PATH.exists():
        print("\n[ERROR] 没有成功完成的组合")
        return

    print(f"\n[Step 3] 筛选验证集: 年化>20% & 回撤>-10%...")
    df_results = pd.read_csv(OUTPUT_PATH)
    filtered = df_results[
        (df_results["val_annual"] > 0.20) & (df_results["val_max_dd"] > -0.10)
    ]

    if filtered.empty:
        print("  未找到满足条件的组合，取验证集年化最高的 10 组:")
        filtered = df_results.nlargest(10, "val_annual")
    else:
        print(f"  通过筛选: {len(filtered)}/{len(df_results)} 组合")

    # Step 4: 对筛选出的组合跑全量回测
    print(f"\n[Step 4] 全量回测 (2019-至今) — {len(filtered)} 组合...")
    tags_need_full = set()

    # 找出还缺少 full 回测的 tag
    for _, r in filtered.iterrows():
        if pd.isna(r.get("full_annual", float("nan"))):
            tags_need_full.add(r["tag"])

    if tags_need_full:
        # 重建 tag → params 映射
        tag_params = {}
        for combo in itertools.product(*param_values):
            p = dict(zip(param_keys, combo))
            t = (f"nl{p['num_leaves']}_lr{p['learning_rate']}_mcs{p['min_child_samples']}"
                 f"_ne{p['n_estimators']}_fd{p['forward_days']}_k{p['topk']}")
            tag_params[t] = p

        for tag in tags_need_full:
            params = tag_params[tag]
            print(f"  全量回测: {tag}", end="", flush=True)
            t2 = time.perf_counter()
            full_metrics = _full_backtest_for_tag(
                tag, params, monthly_df, rebalance_dates,
                strategy, all_instruments, data_end,
            )
            elapsed = time.perf_counter() - t2
            if full_metrics:
                # 更新 CSV 中对应行
                df_results.loc[df_results["tag"] == tag, "full_annual"] = full_metrics["annual"]
                df_results.loc[df_results["tag"] == tag, "full_max_dd"] = full_metrics["max_dd"]
                df_results.loc[df_results["tag"] == tag, "full_sharpe"] = full_metrics["sharpe"]
                print(f" -> 年化={full_metrics['annual']:.2%}  "
                      f"回撤={full_metrics['max_dd']:.2%}  ({elapsed:.1f}s)")
            else:
                print(" -> 无选股")

        # 回写 CSV
        df_results.to_csv(OUTPUT_PATH, index=False)

    # Step 5: 输出结果
    print(f"\n[Step 5] 结果汇总...")
    final = df_results[df_results["tag"].isin(set(filtered["tag"]))].copy()
    final = final.sort_values("val_annual", ascending=False)

    print("\n" + "=" * 70)
    print("超参搜索结果（按验证集年化降序）")
    print("=" * 70)
    cols = ["tag", "val_annual", "val_max_dd", "val_sharpe",
            "full_annual", "full_max_dd", "full_sharpe"]
    print(final[cols].to_string(index=False, float_format="%.4f"))
    print(f"\n[OK] 完整结果已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
