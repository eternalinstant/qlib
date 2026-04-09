#!/usr/bin/env python3
"""
全量重建 OHLCV bin 文件
从 raw_data parquet 文件重建所有 6 个字段（open/high/low/close/volume/amount）
确保所有字段 start_idx 一致、值正确
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QLIB_DIR = PROJECT_ROOT / "data" / "qlib_data" / "cn_data"
RAW_DIR = PROJECT_ROOT / "data" / "qlib_data" / "raw_data"
FIELDS = ["open", "high", "low", "close", "volume", "amount"]


def load_calendar():
    cal_file = QLIB_DIR / "calendars" / "day.txt"
    cal_dates = pd.to_datetime(
        [l.strip() for l in open(cal_file)], errors="coerce"
    ).normalize()
    date_to_idx = {d: i for i, d in enumerate(cal_dates)}
    return date_to_idx, len(cal_dates) - 1


def rebuild_all():
    """从 raw_data 全量重建所有 OHLCV bin"""
    date_to_idx, cal_last_idx = load_calendar()
    features = QLIB_DIR / "features"
    rebuilt = 0
    skipped = 0

    errors = []

    for inst_dir in sorted(features.iterdir()):
        if not inst_dir.is_dir():
            continue
        inst = inst_dir.name

        close_bin = inst_dir / "close.day.bin"
        if not close_bin.exists():
            continue

        raw_file = RAW_DIR / f"{inst}.parquet"
        if not raw_file.exists():
            skipped += 1
            continue

        try:
            df = pd.read_parquet(raw_file, columns=["date"] + FIELDS)
        except Exception as e:
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

        for fld in FIELDS:
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

    logger.info(f"全量重建完成: {rebuilt} 只, 跳过 {skipped} 只, 错误 {errors}")
    return rebuilt


