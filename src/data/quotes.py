"""原始日线行情加载。

数据层：从 data/qlib_data/raw_data/*.parquet 读取 OHLCV（含可靠 high/low）。
从 modules/backtest/common.py 上移至此，根治「数据层反向 import 引擎层 common」的循环依赖。
"""
import pandas as pd

from common.paths import raw_data_root, raw_data_path_for_instrument


def load_raw_trade_quotes(instruments, start_date: str, end_date: str) -> pd.DataFrame:
    if not instruments:
        return pd.DataFrame(columns=["open", "close", "prev_close"])

    root = raw_data_root()
    lookback_start = pd.Timestamp(start_date) - pd.Timedelta(days=10)
    end_ts = pd.Timestamp(end_date)
    frames = []
    missing_files = []

    for instrument in sorted(set(instruments)):
        path = raw_data_path_for_instrument(instrument)
        if not path.exists():
            missing_files.append(instrument)
            continue

        df = pd.read_parquet(path)
        if df.empty:
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df[(df["date"] >= lookback_start) & (df["date"] <= end_ts)].copy()
        if df.empty:
            continue

        df["instrument"] = instrument
        if "pre_close" in df.columns:
            df["prev_close"] = pd.to_numeric(df["pre_close"], errors="coerce")
        else:
            df["prev_close"] = pd.to_numeric(df["close"], errors="coerce").groupby(
                df["instrument"]
            ).shift(1)
        frames.append(df)

    if missing_files:
        preview = ", ".join(missing_files[:10])
        suffix = " ..." if len(missing_files) > 10 else ""
        print(
            f"[WARN] raw_data 缺少 {len(missing_files)} 个标的文件，按不可买卖处理: {preview}{suffix}"
        )

    if not frames:
        return pd.DataFrame(columns=["open", "close", "prev_close"])

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values(["instrument", "date"])
    raw = raw.rename(columns={"date": "datetime"})
    raw = raw[raw["datetime"] >= pd.Timestamp(start_date)].copy()
    raw = raw.drop_duplicates(subset=["datetime", "instrument"], keep="last")
    raw_cols = [c for c in ["open", "high", "low", "close", "prev_close", "amount"] if c in raw.columns]
    return raw.set_index(["datetime", "instrument"])[raw_cols].sort_index()
