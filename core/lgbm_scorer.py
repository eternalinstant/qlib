"""
LightGBM 打分选股模块 — 滚动训练 + 预测，替代线性加权打分

数据流：
    monthly_df (datetime×instrument, 8因子列)
        ↓
    compute_lgbm_signal() → 构造 label → 滚动训练 → 预测打分
        ↓
    signal (Series, index 同 monthly_df)
"""

import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, Optional

from config.config import CONFIG

DEFAULT_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 200,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_estimators": 200,
}


def _load_forward_returns(
    instruments: list,
    start_date: str,
    end_date: str,
    window: int = 20,
) -> pd.DataFrame:
    """加载未来 N 日个股收益率作为 label。

    Returns
    -------
    pd.DataFrame
        MultiIndex (datetime, instrument), 一列 "fwd_ret"
    """
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    from core.selection import _to_provider_instruments, _normalize_multiindex_instruments

    try:
        qlib.init(provider_uri=CONFIG.get("qlib_data_path"), region=REG_CN)
    except Exception:
        pass

    expr = f"Ref($close, -{window})/$close - 1"
    provider_instruments = _to_provider_instruments(instruments)
    df = D.features(provider_instruments, [expr], start_date, end_date, "day")
    df = _normalize_multiindex_instruments(df)
    df.columns = ["fwd_ret"]
    return df


def _cross_section_rank(series: pd.Series) -> pd.Series:
    """截面 rank（按日期分组排名，百分位 0~1）。"""
    return series.groupby(level="datetime").rank(pct=True)


def compute_lgbm_signal(
    monthly_df: pd.DataFrame,
    forward_days: int = 20,
    retrain_freq: int = 60,
    train_window: int = 500,
    lgbm_params: Optional[Dict] = None,
    neutralize_industry: bool = True,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
) -> pd.Series:
    """训练 LightGBM 并生成预测信号。

    当提供 train_start/train_end 时，使用固定训练集模式（训练一次，全部日期预测）。
    否则保持原有滚动训练逻辑。

    Parameters
    ----------
    monthly_df : pd.DataFrame
        MultiIndex (datetime, instrument)，因子列（已 negate、已中性化后的原始值）
    forward_days : int
        未来 N 日收益率作为 label
    retrain_freq : int
        每 N 个交易日重训一次模型（仅滚动模式）
    train_window : int
        训练窗口天数（仅滚动模式）
    lgbm_params : dict, optional
        LightGBM 超参，默认 DEFAULT_LGBM_PARAMS
    neutralize_industry : bool
        是否在 rank 之前做行业中性化
    train_start, train_end : str, optional
        固定训练集日期范围。提供后启用固定训练模式。

    Returns
    -------
    pd.Series
        预测得分，index 同 monthly_df
    """
    import lightgbm as lgb
    from core.selection import _load_industry_map
    from core.compute import neutralize_by_industry

    if lgbm_params is None:
        lgbm_params = DEFAULT_LGBM_PARAMS.copy()

    # 固定训练集模式
    if train_start and train_end:
        model, feature_cols, df_neutralized = train_lgbm_model(
            monthly_df,
            train_start=train_start,
            train_end=train_end,
            forward_days=forward_days,
            lgbm_params=lgbm_params,
            neutralize_industry=neutralize_industry,
        )
        return predict_with_model(model, df_neutralized, feature_cols)

    # ── 以下为原有滚动训练逻辑 ──

    # 行业中性化（与线性方法一致）
    df = monthly_df
    if neutralize_industry:
        industry_map = _load_industry_map()
        if industry_map:
            df = neutralize_by_industry(monthly_df, industry_map)
            print("[LGBM] 行业中性化完成")

    feature_cols = list(df.columns)
    all_dates = sorted(df.index.get_level_values("datetime").unique())
    all_instruments = df.index.get_level_values("instrument").unique().tolist()

    if len(all_dates) < train_window:
        print(f"[LGBM] 警告: 交易日数 ({len(all_dates)}) < 训练窗口 ({train_window})，"
              f"将使用全部可用数据训练")

    # 加载未来收益率作为 label
    label_start = time.perf_counter()
    fwd_start = (all_dates[0] - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    fwd_end = (all_dates[-1] + pd.Timedelta(days=forward_days * 2 + 10)).strftime("%Y-%m-%d")
    fwd_df = _load_forward_returns(all_instruments, fwd_start, fwd_end, window=forward_days)
    print(f"[LGBM] 未来 {forward_days} 日收益率加载完成: 用时 {time.perf_counter() - label_start:.1f}s")

    # 对 label 做截面 rank
    fwd_df["label"] = _cross_section_rank(fwd_df["fwd_ret"])

    # 合并特征和 label
    merged = df.join(fwd_df[["label"]], how="left")
    merged = merged.dropna(subset=["label"])

    # 滚动训练 + 预测
    predictions = {}
    model = None
    last_train_end_idx = -retrain_freq  # 强制首次训练

    print(f"[LGBM] 开始滚动训练: {len(all_dates)} 个交易日, "
          f"retrain_freq={retrain_freq}, train_window={train_window}")

    for i, date in enumerate(all_dates):
        # 只在 rebalance 日期做预测（与选股频率对齐）
        date_data = merged.loc[merged.index.get_level_values("datetime") == date]
        if date_data.empty:
            continue

        features = date_data[feature_cols]
        if features.dropna().empty:
            continue

        # 判断是否需要重训
        if model is None or (i - last_train_end_idx) >= retrain_freq:
            train_end_idx = max(0, i - 1)
            train_start_idx = max(0, i - train_window)

            train_dates = all_dates[train_start_idx : train_end_idx + 1]
            if len(train_dates) < 60:
                # 训练数据太少，跳过
                continue

            train_mask = merged.index.get_level_values("datetime").isin(train_dates)
            train_data = merged.loc[train_mask]

            X_train = train_data[feature_cols]
            y_train = train_data["label"]

            valid_mask = X_train.notna().all(axis=1) & y_train.notna()
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]

            if len(X_train) < 500:
                continue

            model = lgb.LGBMRegressor(**lgbm_params)
            model.fit(X_train, y_train)
            last_train_end_idx = i

            if i == 0 or (i - last_train_end_idx) == 0 or len(predictions) % 5 == 0:
                importance = dict(zip(feature_cols, model.feature_importances_))
                top_feats = sorted(importance.items(), key=lambda x: -x[1])[:5]
                print(f"[LGBM] 训练完成 @ {date.strftime('%Y-%m-%d')}: "
                      f"train_size={len(X_train)}, top features={top_feats[:3]}")

        # 预测
        if model is not None:
            valid_feature_mask = features.notna().all(axis=1)
            if valid_feature_mask.any():
                pred = pd.Series(np.nan, index=date_data.index)
                pred[valid_feature_mask] = model.predict(features[valid_feature_mask])
                predictions[date] = pred

    if not predictions:
        print("[LGBM] 警告: 没有产生任何预测，回退到零信号")
        return pd.Series(0.0, index=monthly_df.index)

    # 合并所有日期的预测
    signal = pd.concat(predictions.values())
    # 确保覆盖 monthly_df 所有 index（缺失的填 NaN）
    signal = signal.reindex(monthly_df.index)

    # 计算整体 IC
    ic_data = merged.join(signal.rename("pred"), how="inner")
    ic_data = ic_data.dropna(subset=["pred", "fwd_ret"])
    if not ic_data.empty:
        daily_ic = ic_data.groupby(level="datetime").apply(
            lambda g: g["pred"].corr(g["fwd_ret"], method="spearman")
        )
        mean_ic = daily_ic.mean()
        ic_ir = mean_ic / daily_ic.std() if daily_ic.std() > 0 else 0
        print(f"[LGBM] 预测完成: mean_IC={mean_ic:.4f}, IC_IR={ic_ir:.4f}, "
              f"覆盖 {signal.notna().sum()}/{len(signal)} 行")

    return signal


def train_lgbm_model(
    monthly_df: pd.DataFrame,
    train_start: str = "2019-01-01",
    train_end: str = "2021-12-31",
    forward_days: int = 20,
    lgbm_params: Optional[Dict] = None,
    neutralize_industry: bool = True,
) -> tuple:
    """固定训练集训练 LightGBM 模型。

    Parameters
    ----------
    monthly_df : pd.DataFrame
        MultiIndex (datetime, instrument)，因子列
    train_start, train_end : str
        训练集日期范围
    forward_days : int
        未来 N 日收益率作为 label
    lgbm_params : dict, optional
        LightGBM 超参
    neutralize_industry : bool
        是否做行业中性化

    Returns
    -------
    tuple (model, feature_cols, df_neutralized)
        model: 训练好的 LGBMRegressor
        feature_cols: list of str
        df_neutralized: DataFrame (中性化后的因子数据)
    """
    import lightgbm as lgb
    from core.selection import _load_industry_map
    from core.compute import neutralize_by_industry

    if lgbm_params is None:
        lgbm_params = DEFAULT_LGBM_PARAMS.copy()

    # 行业中性化
    df = monthly_df
    if neutralize_industry:
        industry_map = _load_industry_map()
        if industry_map:
            df = neutralize_by_industry(monthly_df, industry_map)
            print("[LGBM] 行业中性化完成")

    feature_cols = list(df.columns)
    all_instruments = df.index.get_level_values("instrument").unique().tolist()

    # 加载未来收益率
    fwd_start = (pd.Timestamp(train_start) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    fwd_end = (pd.Timestamp(train_end) + pd.Timedelta(days=forward_days * 2 + 10)).strftime("%Y-%m-%d")
    fwd_df = _load_forward_returns(all_instruments, fwd_start, fwd_end, window=forward_days)
    print(f"[LGBM] 未来 {forward_days} 日收益率加载完成 (训练标签)")

    # 截面 rank label
    fwd_df["label"] = _cross_section_rank(fwd_df["fwd_ret"])

    # 筛选训练集日期
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    train_mask = (df.index.get_level_values("datetime") >= train_start_ts) & (
        df.index.get_level_values("datetime") <= train_end_ts
    )
    train_df = df.loc[train_mask]

    # 合并 label
    merged = train_df.join(fwd_df[["label"]], how="left").dropna(subset=["label"])

    X_train = merged[feature_cols]
    y_train = merged["label"]
    valid_mask = X_train.notna().all(axis=1) & y_train.notna()
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]

    print(f"[LGBM] 训练集: {train_start} ~ {train_end}, 样本数={len(X_train)}")

    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X_train, y_train)

    importance = dict(zip(feature_cols, model.feature_importances_))
    top_feats = sorted(importance.items(), key=lambda x: -x[1])[:5]
    print(f"[LGBM] 训练完成: top features={top_feats}")

    return model, feature_cols, df


def walk_forward_train_predict(
    monthly_df: pd.DataFrame,
    train_start: str = "2019-01-01",
    data_end: str = "2026-02-26",
    train_window_years: int = 3,
    test_window_years: int = 1,
    step_years: int = 1,
    forward_days: int = 5,
    lgbm_params: Optional[Dict] = None,
    neutralize_industry: bool = True,
) -> tuple:
    """Walk-forward 训练 + 预测。

    滚动训练窗口，每个窗口训练一个模型并预测后续测试期。
    拼接所有测试期预测形成 OOS 信号，首个训练窗口的样本内预测
    用于回测连续性。

    Parameters
    ----------
    monthly_df : pd.DataFrame
        MultiIndex (datetime, instrument)，因子列
    train_start : str
        首个训练窗口起始日期
    data_end : str
        数据截止日期
    train_window_years : int
        训练窗口年数，默认 3
    test_window_years : int
        测试窗口年数，默认 1
    step_years : int
        窗口滑动步长（年），默认 1
    forward_days : int
        未来 N 日收益率作为 label
    lgbm_params : dict, optional
        LightGBM 超参
    neutralize_industry : bool
        是否做行业中性化

    Returns
    -------
    tuple (signal, oos_start, windows)
        signal: pd.Series, 全量信号 (MultiIndex datetime×instrument)
        oos_start: str, 首个 OOS 日期 (YYYY-MM-DD)，用于 OOS 指标计算
        windows: list of dict, 各窗口信息
    """
    import lightgbm as lgb
    from core.selection import _load_industry_map
    from core.compute import neutralize_by_industry

    if lgbm_params is None:
        lgbm_params = DEFAULT_LGBM_PARAMS.copy()
    else:
        # 合并默认参数
        merged = DEFAULT_LGBM_PARAMS.copy()
        merged.update(lgbm_params)
        lgbm_params = merged

    # 1. 行业中性化（全量一次完成）
    df = monthly_df
    if neutralize_industry:
        industry_map = _load_industry_map()
        if industry_map:
            df = neutralize_by_industry(monthly_df, industry_map)
            print("[WF] 行业中性化完成")

    feature_cols = list(df.columns)
    all_instruments = df.index.get_level_values("instrument").unique().tolist()

    # 2. 加载全量未来收益率作为 label
    fwd_start = (pd.Timestamp(train_start) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    fwd_end = (pd.Timestamp(data_end) + pd.Timedelta(days=forward_days * 2 + 10)).strftime(
        "%Y-%m-%d"
    )
    fwd_df = _load_forward_returns(all_instruments, fwd_start, fwd_end, window=forward_days)
    fwd_df["label"] = _cross_section_rank(fwd_df["fwd_ret"])
    print(f"[WF] 全量 label 加载完成: {len(fwd_df)} 行")

    # 3. 生成 walk-forward 窗口
    ts_train_start = pd.Timestamp(train_start)
    ts_data_end = pd.Timestamp(data_end)

    windows = []
    current_start = ts_train_start
    while True:
        train_end = current_start + pd.DateOffset(years=train_window_years) - pd.DateOffset(
            days=1
        )
        test_start = train_end + pd.DateOffset(days=1)
        test_end = min(
            test_start + pd.DateOffset(years=test_window_years) - pd.DateOffset(days=1),
            ts_data_end,
        )

        if test_start > ts_data_end:
            break

        windows.append(
            {
                "train_start": current_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
            }
        )
        current_start += pd.DateOffset(years=step_years)

    if not windows:
        print("[WF] 警告: 无有效窗口，回退到零信号")
        return pd.Series(0.0, index=monthly_df.index), "", []

    print(f"[WF] 窗口数: {len(windows)}")
    for i, w in enumerate(windows):
        print(
            f"  W{i}: train={w['train_start']}~{w['train_end']} "
            f"→ test={w['test_start']}~{w['test_end']}"
        )

    # 4. 逐窗口训练 + 预测
    all_signals = []
    oos_start = windows[0]["test_start"] if windows else ""
    first_model_feature_cols = None
    models_trained = 0

    for i, w in enumerate(windows):
        # 训练
        train_mask = (
            df.index.get_level_values("datetime") >= pd.Timestamp(w["train_start"])
        ) & (df.index.get_level_values("datetime") <= pd.Timestamp(w["train_end"]))
        train_df = df.loc[train_mask]

        if train_df.empty:
            print(f"[WF] W{i}: 训练数据为空，跳过")
            continue

        merged = train_df.join(fwd_df[["label"]], how="left").dropna(subset=["label"])

        X_train = merged[feature_cols]
        y_train = merged["label"]
        valid_mask = X_train.notna().all(axis=1) & y_train.notna()
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]

        if len(X_train) < 500:
            print(f"[WF] W{i}: 训练样本不足 ({len(X_train)}), 跳过")
            continue

        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(X_train, y_train)
        models_trained += 1

        if first_model_feature_cols is None:
            first_model_feature_cols = feature_cols.copy()

        importance = dict(zip(feature_cols, model.feature_importances_))
        top_feats = sorted(importance.items(), key=lambda x: -x[1])[:3]
        print(
            f"[WF] W{i} 训练完成: train={w['train_start']}~{w['train_end']}, "
            f"n={len(X_train)}, top={top_feats}"
        )

        # 预测测试期
        test_mask = (
            df.index.get_level_values("datetime") >= pd.Timestamp(w["test_start"])
        ) & (df.index.get_level_values("datetime") <= pd.Timestamp(w["test_end"]))
        test_df = df.loc[test_mask]

        if not test_df.empty:
            test_signal = _predict_dates(model, test_df, feature_cols)
            if test_signal is not None:
                all_signals.append(test_signal)

        # 首个窗口额外预测训练期（回测连续性）
        if i == 0:
            train_signal = _predict_dates(model, train_df, feature_cols)
            if train_signal is not None:
                all_signals.append(train_signal)

    if not all_signals:
        print("[WF] 警告: 无有效预测，回退到零信号")
        return pd.Series(0.0, index=monthly_df.index), "", windows

    # 5. 拼接全量信号
    full_signal = pd.concat(all_signals)
    full_signal = full_signal[~full_signal.index.duplicated(keep="last")]
    full_signal = full_signal.reindex(monthly_df.index)

    coverage = full_signal.notna().sum()
    print(
        f"[WF] Walk-forward 完成: {models_trained}/{len(windows)} 个模型, "
        f"OOS 起始={oos_start}, 覆盖={coverage}/{len(full_signal)} 行"
    )

    return full_signal, oos_start, windows


def _predict_dates(
    model,
    date_df: pd.DataFrame,
    feature_cols: list,
) -> Optional[pd.Series]:
    """对指定日期范围的 DataFrame 逐日预测并拼接。

    Parameters
    ----------
    model : LGBMRegressor
        训练好的模型
    date_df : pd.DataFrame
        MultiIndex (datetime, instrument)，因子数据子集
    feature_cols : list
        特征列名

    Returns
    -------
    pd.Series or None
    """
    predictions = {}
    for date in sorted(date_df.index.get_level_values("datetime").unique()):
        date_data = date_df.loc[date_df.index.get_level_values("datetime") == date]
        if isinstance(date_data, pd.Series):
            date_data = date_data.to_frame().T
        features = date_data[feature_cols]
        valid_mask = features.notna().all(axis=1)
        if valid_mask.any():
            pred = pd.Series(np.nan, index=date_data.index)
            pred[valid_mask] = model.predict(features[valid_mask].astype(float))
            predictions[date] = pred

    if not predictions:
        return None
    return pd.concat(predictions.values())


def predict_with_model(
    model,
    monthly_df: pd.DataFrame,
    feature_cols: list,
) -> pd.Series:
    """用已训练模型预测信号。

    Parameters
    ----------
    model : LGBMRegressor
        训练好的模型
    monthly_df : pd.DataFrame
        因子数据（需与训练时相同的列名）
    feature_cols : list
        特征列名

    Returns
    -------
    pd.Series
        预测得分, index 同 monthly_df
    """
    all_dates = sorted(monthly_df.index.get_level_values("datetime").unique())
    predictions = {}

    for date in all_dates:
        date_data = monthly_df.loc[monthly_df.index.get_level_values("datetime") == date]
        if date_data.empty:
            continue
        features = date_data[feature_cols]
        valid_mask = features.notna().all(axis=1)
        if valid_mask.any():
            pred = pd.Series(np.nan, index=date_data.index)
            pred[valid_mask] = model.predict(features[valid_mask])
            predictions[date] = pred

    if not predictions:
        return pd.Series(0.0, index=monthly_df.index)

    signal = pd.concat(predictions.values()).reindex(monthly_df.index)
    print(f"[LGBM] 预测完成: 覆盖 {signal.notna().sum()}/{len(signal)} 行")
    return signal
