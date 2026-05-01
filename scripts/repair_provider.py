#!/usr/bin/env python3
"""
Qlib provider 一致性修复脚本

修复三类问题：
1. close.day.bin 超过日历长度 → 截断到日历末尾
2. OHLCVA.bin 超过日历长度 → 截断到日历末尾
3. close 与 OHLCVA end_idx 不一致 → 从 raw_data 全量重建
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
FIELDS = ["open", "high", "low", "close", "volume", "amount"]


def _load_calendar():
    cal_file = QLIB_DIR / "calendars" / "day.txt"
    cal_dates = pd.to_datetime(
        [l.strip() for l in open(cal_file)], errors="coerce"
    ).normalize()
    date_to_idx = {d: i for i, d in enumerate(cal_dates)}
    return date_to_idx, len(cal_dates) - 1


def _get_end_idx(bin_path: Path) -> int:
    raw = np.fromfile(bin_path, dtype="<f4")
    if len(raw) < 2 or np.isnan(raw[0]):
        return -1
    return int(raw[0]) + len(raw) - 2


def truncate_oversized_bins():
    """截断超过日历长度的 bin 文件"""
    date_to_idx, cal_last_idx = _load_calendar()
    features = QLIB_DIR / "features"
    truncated = 0

    for inst_dir in features.iterdir():
        if not inst_dir.is_dir():
            continue
        for fld in FIELDS:
            fb = inst_dir / f"{fld}.day.bin"
            if not fb.exists():
                continue
            raw = np.fromfile(fb, dtype="<f4")
            if len(raw) < 2 or np.isnan(raw[0]):
                continue
            end_idx = int(raw[0]) + len(raw) - 2
            if end_idx <= cal_last_idx:
                continue
            start_idx = int(raw[0])
            new_len = cal_last_idx - start_idx + 2
            if new_len < 2:
                continue
            raw[:new_len].tofile(fb)
            truncated += 1

    logger.info(f"截断超范围 bin: {truncated} 个文件")
    return truncated


def _is_consistent(inst_dir: Path) -> bool:
    """检查一个股票的所有 OHLCVA 字段 end_idx 是否一致"""
    ends = set()
    for fld in FIELDS:
        fb = inst_dir / f"{fld}.day.bin"
        if not fb.exists():
            continue
        e = _get_end_idx(fb)
        if e >= 0:
            ends.add(e)
    return len(ends) <= 1


def _load_adj_ratio_map(date_to_idx) -> dict:
    """加载 adj_factor 并返回 {instrument: {cal_idx: adj_ratio}}"""
    adj_path = TUSHARE_DIR / "adj_factor.parquet"
    if not adj_path.exists():
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
    return result


def rebuild_from_raw_data():
    """从 raw_data 全量重建 OHLCVA 字段不一致的股票（前复权）"""
    date_to_idx, cal_last_idx = _load_calendar()
    features = QLIB_DIR / "features"

    idx_to_date = {i: d for d, i in date_to_idx.items()}
    adj_maps = _load_adj_ratio_map(date_to_idx)
    PRICE_FIELDS = {"open", "high", "low", "close"}

    rebuilt = 0

    for inst_dir in features.iterdir():
        if not inst_dir.is_dir():
            continue
        inst = inst_dir.name

        cb = inst_dir / "close.day.bin"
        if not cb.exists():
            continue

        if _is_consistent(inst_dir):
            continue

        raw_file = RAW_DIR / f"{inst}.parquet"
        if not raw_file.exists():
            continue

        try:
            df = pd.read_parquet(raw_file, columns=["date"] + FIELDS)
        except Exception:
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"])
        mask = df["date"].isin(set(date_to_idx.keys()))
        df = df[mask]
        if df.empty:
            continue

        df["cal_idx"] = df["date"].map(date_to_idx)
        df = df.dropna(subset=["cal_idx"])
        df["cal_idx"] = df["cal_idx"].astype(int)

        inst_adj = adj_maps.get(inst, {})

        for fld in FIELDS:
            fld_data = df[["cal_idx", fld]].dropna(subset=[fld]).sort_values("cal_idx")
            if fld_data.empty:
                continue
            start_idx = int(fld_data["cal_idx"].min())

            if fld in PRICE_FIELDS and inst_adj:
                ratios = fld_data["cal_idx"].map(inst_adj)
                valid = ratios.notna() & np.isfinite(ratios) & np.isfinite(fld_data[fld])
                vals = np.where(valid, fld_data[fld] * ratios, fld_data[fld]).astype(np.float32)
            else:
                vals = fld_data[fld].values.astype(np.float32)

            out = np.concatenate([[np.float32(start_idx)], vals])
            out.tofile(inst_dir / f"{fld}.day.bin")

        rebuilt += 1

    logger.info(f"从 raw_data 前复权重建: {rebuilt} 只股票")
    return rebuilt


def run_repair():
    logger.info("=== Qlib Provider 一致性修复 ===")
    t1 = truncate_oversized_bins()
    t2 = rebuild_from_raw_data()
    logger.info(f"完成: 截断 {t1} 个超范围 bin, 重建 {t2} 只股票")


if __name__ == "__main__":
    run_repair()
