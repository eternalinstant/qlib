#!/usr/bin/env python3
"""
全量重建 OHLCV bin 文件（前复权）
从 raw_data parquet 文件 + adj_factor 计算前复权价格，重建所有 6 个字段。
volume/amount 保持原始值，不做复权缩放。
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QLIB_DIR = PROJECT_ROOT / "data" / "qlib_data" / "cn_data"
RAW_DIR = PROJECT_ROOT / "data" / "qlib_data" / "raw_data"
TUSHARE_DIR = PROJECT_ROOT / "data" / "tushare"
PRICE_FIELDS = ["open", "high", "low", "close"]
VOLUME_FIELDS = ["volume", "amount"]
ALL_FIELDS = PRICE_FIELDS + VOLUME_FIELDS


def load_calendar():
    cal_file = QLIB_DIR / "calendars" / "day.txt"
    cal_dates = pd.to_datetime(
        [l.strip() for l in open(cal_file)], errors="coerce"
    ).normalize()
    date_to_idx = {d: i for i, d in enumerate(cal_dates)}
    idx_to_date = {i: d for d, i in date_to_idx.items()}
    return date_to_idx, idx_to_date, len(cal_dates) - 1


def load_adj_ratio_map(date_to_idx) -> dict:
    """加载 adj_factor 并计算前复权比例，返回 {instrument: {cal_idx: adj_ratio}}"""
    adj_path = TUSHARE_DIR / "adj_factor.parquet"
    if not adj_path.exists():
        logger.error("adj_factor.parquet 不存在，无法前复权重建。请先下载。")
        return {}

    adj_df = pd.read_parquet(adj_path, columns=["ts_code", "trade_date", "adj_factor"])
    adj_df["date"] = pd.to_datetime(adj_df["trade_date"], format="%Y%m%d").dt.normalize()
    adj_df = adj_df[adj_df["date"].isin(date_to_idx)]
    adj_df["instrument"] = adj_df["ts_code"].apply(
        lambda x: x.split(".")[1].lower() + x.split(".")[0] if "." in x else x.lower()
    )

    result = {}
    for inst, grp in adj_df.groupby("instrument"):
        grp = grp.sort_values("date")
        latest_adj = grp["adj_factor"].iloc[-1]
        if latest_adj <= 0:
            continue
        ratios = grp["adj_factor"] / latest_adj
        cal_idxs = grp["date"].map(date_to_idx)
        result[inst] = dict(zip(cal_idxs, ratios))

    logger.info(f"加载 adj_factor: {len(result)} 只股票")
    return result


def rebuild_all():
    """从 raw_data + adj_factor 全量重建所有 OHLCV bin（前复权）"""
    date_to_idx, idx_to_date, cal_last_idx = load_calendar()
    adj_maps = load_adj_ratio_map(date_to_idx)
    if not adj_maps:
        logger.error("无 adj_factor 数据，退出")
        return 0

    features = QLIB_DIR / "features"
    rebuilt = 0
    skipped = 0
    no_adj = 0

    for inst_dir in sorted(features.iterdir()):
        if not inst_dir.is_dir():
            continue
        inst = inst_dir.name

        raw_file = RAW_DIR / f"{inst}.parquet"
        if not raw_file.exists():
            skipped += 1
            continue

        inst_adj = adj_maps.get(inst)
        if inst_adj is None:
            no_adj += 1
            continue

        try:
            df = pd.read_parquet(raw_file, columns=["date"] + ALL_FIELDS)
        except Exception:
            skipped += 1
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"])
        df = df[df["date"].isin(date_to_idx)]
        if df.empty:
            skipped += 1
            continue

        df["cal_idx"] = df["date"].map(date_to_idx)
        df = df.dropna(subset=["cal_idx"])
        df["cal_idx"] = df["cal_idx"].astype(int)

        # inst_adj: {cal_idx: adj_ratio}，直接 map
        df["adj_ratio"] = df["cal_idx"].map(inst_adj)

        for fld in PRICE_FIELDS:
            if fld not in df.columns:
                continue
            fld_data = df[["cal_idx", fld, "adj_ratio"]].dropna(subset=[fld]).sort_values("cal_idx")
            if fld_data.empty:
                continue
            start_idx = int(fld_data["cal_idx"].min())
            valid_ratio = fld_data["adj_ratio"].notna() & np.isfinite(fld_data["adj_ratio"])
            valid_price = np.isfinite(fld_data[fld])
            vals = np.where(
                valid_ratio & valid_price,
                fld_data[fld] * fld_data["adj_ratio"],
                fld_data[fld],
            ).astype(np.float32)
            out = np.concatenate([[np.float32(start_idx)], vals])
            out.tofile(inst_dir / f"{fld}.day.bin")

        for fld in VOLUME_FIELDS:
            if fld not in df.columns:
                continue
            fld_data = df[["cal_idx", fld]].dropna(subset=[fld]).sort_values("cal_idx")
            if fld_data.empty:
                continue
            start_idx = int(fld_data["cal_idx"].min())
            vals = fld_data[fld].values.astype(np.float32)
            out = np.concatenate([[np.float32(start_idx)], vals])
            out.tofile(inst_dir / f"{fld}.day.bin")

        rebuilt += 1
        if rebuilt % 500 == 0:
            logger.info(f"已重建 {rebuilt} 只...")

    logger.info(f"前复权重建完成: {rebuilt} 只, {no_adj} 只无adj, {skipped} 只跳过")
    return rebuilt


if __name__ == "__main__":
    rebuild_all()
