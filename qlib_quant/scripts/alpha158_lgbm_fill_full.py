"""补跑 Top-50 全量回测"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CONFIG
from core.strategy import Strategy
from core.selection import load_factor_data, extract_topk
from core.lgbm_scorer import train_lgbm_model, predict_with_model

CSV_PATH = PROJECT_ROOT / "results" / "alpha158_lgbm_hptune.csv"

BASE_LGBM_PARAMS = {
    "objective": "regression", "metric": "mse",
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "verbose": -1,
}
TRAIN_START = "2019-01-01"
TRAIN_END = "2021-12-31"
FULL_START = "2019-01-01"


def _compute_metrics(daily_returns):
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


def _load_returns_pivot(symbols, start, end):
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    try:
        qlib.init(provider_uri=CONFIG.get("qlib_data_path"), region=REG_CN)
    except Exception:
        pass
    ps = pd.Timestamp(start) - pd.Timedelta(days=10)
    pe = pd.Timestamp(end) + pd.Timedelta(days=30)
    pdf = D.features(symbols, ["$close"], ps.strftime("%Y-%m-%d"), pe.strftime("%Y-%m-%d"), "day")
    if pdf.empty:
        return pd.DataFrame()
    pdf.columns = ["close"]
    pdf.index = pd.MultiIndex.from_tuples(
        [(dt, s.split("/")[-1].upper() if "/" in str(s) else str(s).upper())
         for s, dt in pdf.index],
        names=["datetime", "instrument"],
    )
    pivot = pdf["close"].unstack(level="instrument")
    pivot.index = pd.to_datetime(pivot.index)
    return pivot.pct_change()


def _compute_portfolio_returns(df_sel, returns_pivot, start, end):
    if df_sel.empty or returns_pivot.empty:
        return pd.Series(dtype=float)
    dates = sorted(pd.to_datetime(df_sel["date"]).unique())
    if not dates:
        return pd.Series(dtype=float)
    result = {}
    for i in range(len(dates)):
        rd = dates[i]
        nrd = dates[i + 1] if i + 1 < len(dates) else pd.Timestamp(end) + pd.Timedelta(days=30)
        holdings = [h for h in df_sel[df_sel["date"] == rd]["symbol"].unique() if h in returns_pivot.columns]
        if not holdings:
            continue
        mask = (returns_pivot.index > rd) & (returns_pivot.index <= nrd)
        pr = returns_pivot.loc[mask, holdings]
        for dt in pr.index:
            dr = pr.loc[dt].dropna()
            if len(dr) > 0:
                result[dt] = dr.mean()
    if not result:
        return pd.Series(dtype=float)
    s = pd.Series(result).sort_index()
    s.index.name = "datetime"
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    return s


def _parse_tag(tag):
    parts = tag.split("_")
    return {
        "num_leaves": int(parts[0][2:]),
        "learning_rate": float(parts[1][2:]),
        "min_child_samples": int(parts[2][3:]),
        "n_estimators": int(parts[3][2:]),
        "forward_days": int(parts[4][2:]),
        "topk": int(parts[5][1:]),
    }


def main():
    df = pd.read_csv(CSV_PATH)
    top50 = df.nlargest(50, 'val_annual')
    missing = top50[top50['full_annual'].isna()].copy()
    print(f"需要补跑: {len(missing)} 个组合")

    if missing.empty:
        print("全部已完成")
        return

    # 加载公共数据
    strategy = Strategy.load("experimental/alpha158/alpha158_csi300")
    data_end = CONFIG.get("end_date", "2026-02-26")
    monthly_df, rebalance_dates = load_factor_data(
        registry=strategy.registry, start_date="2018-12-01", end_date=data_end,
        rebalance_freq=strategy.rebalance_freq, universe=strategy.universe,
        factor_window_scale=strategy.factor_window_scale,
    )
    all_inst = monthly_df.index.get_level_values("instrument").unique().tolist()

    print(f"加载价格数据...")
    rp = _load_returns_pivot(all_inst, FULL_START, data_end)

    for idx, (_, row) in enumerate(missing.iterrows()):
        tag = row["tag"]
        p = _parse_tag(tag)
        print(f"  [{idx+1}/{len(missing)}] {tag}", end="", flush=True)
        t0 = time.perf_counter()

        lgbm_params = {**BASE_LGBM_PARAMS,
                       "num_leaves": p["num_leaves"], "learning_rate": p["learning_rate"],
                       "min_child_samples": p["min_child_samples"], "n_estimators": p["n_estimators"]}

        model, fcols, dfn = train_lgbm_model(
            monthly_df, train_start=TRAIN_START, train_end=TRAIN_END,
            forward_days=p["forward_days"], lgbm_params=lgbm_params, neutralize_industry=True,
        )
        signal = predict_with_model(model, dfn, fcols)
        df_sel = extract_topk(signal, rebalance_dates, topk=p["topk"],
                              exclude_new_days=strategy.exclude_new_days,
                              exclude_st=strategy.exclude_st, universe=strategy.universe)
        if df_sel.empty:
            print(" -> 无选股")
            continue

        fr = _compute_portfolio_returns(df_sel, rp, FULL_START, data_end)
        m = _compute_metrics(fr)
        elapsed = time.perf_counter() - t0

        # 更新 CSV
        mask = df["tag"] == tag
        df.loc[mask, "full_annual"] = m["annual"]
        df.loc[mask, "full_max_dd"] = m["max_dd"]
        df.loc[mask, "full_sharpe"] = m["sharpe"]
        print(f" -> 年化={m['annual']:.2%}  回撤={m['max_dd']:.2%}  ({elapsed:.1f}s)")

        del model, signal, df_sel, dfn

    df.to_csv(CSV_PATH, index=False)
    print(f"\n[OK] 已保存: {CSV_PATH}")

    # 打印最终 top50
    top50_final = df.nlargest(50, 'val_annual')
    cols = ['tag', 'val_annual', 'val_max_dd', 'val_sharpe', 'full_annual', 'full_max_dd', 'full_sharpe']
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n" + "=" * 70)
    print("Top 50（按验证集年化降序）")
    print("=" * 70)
    print(top50_final[cols].to_string(index=False))


if __name__ == "__main__":
    main()
