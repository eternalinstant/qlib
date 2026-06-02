#!/usr/bin/env python3
"""
修复 OHLCV 前复权数据
4月4日 rebuild_ohlcv.py 从不复权的 raw_data 重建了 OHLCV bins，
导致 high/low/open 的复权因子与 close 不一致。

本脚本逐股票重建 open/high/low bins，使 adj_ratio = bin_close / raw_close，
再用 adj_ratio * raw_field 得到正确的前复权值。
volume/amount 直接写 raw 值，不做复权缩放。

对已有历史的处理规则：
1. raw_data 覆盖范围之前，保留现有 bin 历史
2. raw_data 覆盖范围及之后，用权威 close + raw_data 重建
3. close.bin 继续扩展到日历末端
"""
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QLIB_DIR = PROJECT_ROOT / "data" / "qlib_data" / "cn_data"
RAW_DIR = PROJECT_ROOT / "data" / "qlib_data" / "raw_data"
TUSHARE_DIR = PROJECT_ROOT / "data" / "tushare"

PRICE_FIELDS = ["open", "high", "low"]
VOLUME_FIELDS = ["volume", "amount"]
ALL_FIELDS = PRICE_FIELDS + ["close"] + VOLUME_FIELDS


def load_calendar():
    """加载日历，返回 {date: idx} 和 last_idx"""
    cal_file = QLIB_DIR / "calendars" / "day.txt"
    cal = pd.read_csv(cal_file, header=None, names=["date"], parse_dates=["date"])
    cal_dates = cal["date"].dt.normalize()
    date_to_idx = {d: i for i, d in enumerate(cal_dates)}
    return date_to_idx, len(cal) - 1


def read_bin(bin_file: Path):
    """读取 bin 文件，返回 (start_idx, values) 或 None"""
    if not bin_file.exists():
        return None
    raw = np.fromfile(bin_file, dtype="<f4")
    if len(raw) < 2 or np.isnan(raw[0]):
        return None
    start_idx = int(raw[0])
    values = raw[1:].astype(np.float32, copy=True)
    return start_idx, values


def write_bin(bin_file: Path, start_idx: int, values):
    """写入 bin 文件"""
    values = np.asarray(values, dtype="<f4")
    if values.size == 0:
        return False
    payload = np.empty(values.size + 1, dtype="<f4")
    payload[0] = np.float32(start_idx)
    payload[1:] = values
    payload.tofile(bin_file)
    return True


def _map_from_bin(bin_file: Path):
    """将 bin 文件读成 {cal_idx: value} 查找表。"""
    meta = read_bin(bin_file)
    if meta is None:
        return {}
    start_idx, values = meta
    return dict(zip(range(start_idx, start_idx + len(values)), values.tolist()))


def load_daily_basic_close(date_to_idx):
    """加载 daily_basic.parquet 的不复权 close，返回 {instrument: {cal_idx: close}}"""
    db_path = TUSHARE_DIR / "daily_basic.parquet"
    if not db_path.exists():
        logger.warning("daily_basic.parquet 不存在，无法扩展 close.bin")
        return {}

    db = pd.read_parquet(db_path, columns=["ts_code", "trade_date", "close"])
    db["date"] = pd.to_datetime(db["trade_date"], format="%Y%m%d").dt.normalize()
    db = db[db["date"].isin(date_to_idx)]
    db["cal_idx"] = db["date"].map(date_to_idx)
    # ts_code: 000001.SZ -> sz000001
    db["instrument"] = db["ts_code"].apply(
        lambda x: x.split(".")[1].lower() + x.split(".")[0] if "." in x else x.lower()
    )
    db = db[["instrument", "cal_idx", "close"]].dropna(subset=["close"])

    inst_data = {}
    for inst, g in db.groupby("instrument"):
        inst_data[inst] = dict(zip(g["cal_idx"].astype(int), g["close"]))
    return inst_data


def fix_all():
    """修复所有股票的 OHLCV bins"""
    date_to_idx, cal_last_idx = load_calendar()
    features_dir = QLIB_DIR / "features"
    db_close_map = load_daily_basic_close(date_to_idx)

    rebuilt = 0
    skipped = 0
    no_raw = 0
    close_extended = 0
    violations = 0

    inst_dirs = sorted([d for d in features_dir.iterdir() if d.is_dir()])
    total = len(inst_dirs)

    for i, inst_dir in enumerate(inst_dirs):
        inst = inst_dir.name

        # 1. 读 close.bin（前复权，可信基准）
        close_bin = inst_dir / "close.day.bin"
        close_meta = read_bin(close_bin)
        if close_meta is None:
            skipped += 1
            continue
        close_start, close_vals = close_meta
        close_end = close_start + len(close_vals) - 1

        # 2. 读 raw_data
        raw_file = RAW_DIR / f"{inst}.parquet"
        if not raw_file.exists():
            no_raw += 1
            continue

        try:
            raw_df = pd.read_parquet(raw_file, columns=["date"] + ALL_FIELDS)
        except Exception:
            skipped += 1
            continue

        raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce").dt.normalize()
        raw_df = raw_df.dropna(subset=["date"])
        raw_df = raw_df[raw_df["date"].isin(date_to_idx)]
        if raw_df.empty:
            skipped += 1
            continue

        raw_df["cal_idx"] = raw_df["date"].map(date_to_idx).astype(int)

        # 构建 raw 查找表
        raw_maps = {}
        for fld in ALL_FIELDS:
            if fld in raw_df.columns:
                sub = raw_df[["cal_idx", fld]].dropna(subset=[fld])
                raw_maps[fld] = dict(zip(sub["cal_idx"], sub[fld]))

        raw_close_map = raw_maps.get("close", {})
        if not raw_close_map:
            skipped += 1
            continue

        # 3. 构建权威 close 映射：
        # raw 覆盖之前保留旧历史；raw 覆盖及之后改用 daily_basic/raw close。
        inst_db = db_close_map.get(inst, {})
        overlap_start = min(raw_close_map) if raw_close_map else None
        if overlap_start is not None:
            close_map = dict(zip(
                range(close_start, close_start + len(close_vals)),
                close_vals.tolist(),
            ))
            authoritative_close = dict(close_map)
            close_replaced = False
            for idx in range(overlap_start, cal_last_idx + 1):
                v = inst_db.get(idx, raw_close_map.get(idx))
                if v is not None and np.isfinite(v) and v > 0:
                    authoritative_close[idx] = float(v)
                    if idx <= close_end:
                        close_replaced = True

            if close_replaced:
                close_extended += 1

            final_close_start = min(authoritative_close)
            final_close_end = max(authoritative_close)
            close_vals = np.array(
                [authoritative_close.get(idx, np.nan) for idx in range(final_close_start, final_close_end + 1)],
                dtype="<f4",
            )
            close_start = final_close_start
            close_end = final_close_end
            close_map = authoritative_close
        else:
            # 无 raw 覆盖时，仅保留既有 close，并按旧逻辑扩展到日历末端。
            if close_end < cal_last_idx:
                bin_last_close = float(close_vals[-1])
                db_last_close = inst_db.get(close_end, None)
                if db_last_close and db_last_close > 0:
                    adj_ratio_splice = bin_last_close / db_last_close
                else:
                    adj_ratio_splice = 1.0

                new_close_vals = []
                for idx in range(close_end + 1, cal_last_idx + 1):
                    v = inst_db.get(idx)
                    if v is not None and v > 0:
                        new_close_vals.append(float(v) * adj_ratio_splice)
                    else:
                        new_close_vals.append(np.nan)

                if new_close_vals:
                    close_vals = np.concatenate([close_vals, np.array(new_close_vals, dtype="<f4")])
                    close_end = cal_last_idx
                    close_extended += 1

            close_idx_range = np.arange(close_start, close_end + 1)
            close_map = dict(zip(close_idx_range.tolist(), close_vals.tolist()))

        # 5. 逐日计算 adj_ratio 并重建 price fields
        for fld in PRICE_FIELDS:
            raw_fld_map = raw_maps.get(fld, {})
            if not raw_fld_map and not raw_close_map:
                continue

            existing_field_map = _map_from_bin(inst_dir / f"{fld}.day.bin")

            target_start = close_start
            # 也考虑 raw 数据的起始点
            if raw_fld_map:
                raw_fld_start = min(raw_fld_map)
                target_start = min(target_start, raw_fld_start)
            if existing_field_map:
                target_start = min(target_start, min(existing_field_map))

            rebuilt_vals = []
            for idx in range(target_start, close_end + 1):
                raw_val = raw_fld_map.get(idx)
                bin_close = close_map.get(idx)
                raw_close = raw_close_map.get(idx)

                if (
                    raw_val is not None
                    and raw_close is not None
                    and bin_close is not None
                    and np.isfinite(raw_val)
                    and np.isfinite(raw_close)
                    and np.isfinite(bin_close)
                    and raw_close > 0
                ):
                    adj_ratio = float(bin_close) / float(raw_close)
                    derived = float(raw_val) * adj_ratio
                else:
                    existing_val = existing_field_map.get(idx)
                    if existing_val is not None and np.isfinite(existing_val):
                        derived = float(existing_val)
                    else:
                        derived = np.nan
                rebuilt_vals.append(derived)

            if rebuilt_vals:
                write_bin(inst_dir / f"{fld}.day.bin", target_start, rebuilt_vals)

        # 6. 重建 volume/amount（直接用 raw 值，不做复权）
        for fld in VOLUME_FIELDS:
            raw_fld_map = raw_maps.get(fld, {})
            existing_field_map = _map_from_bin(inst_dir / f"{fld}.day.bin")
            if not raw_fld_map and not existing_field_map:
                continue

            target_start = close_start
            if raw_fld_map:
                raw_fld_start = min(raw_fld_map)
                target_start = min(target_start, raw_fld_start)
            if existing_field_map:
                target_start = min(target_start, min(existing_field_map))

            rebuilt_vals = []
            for idx in range(target_start, close_end + 1):
                v = raw_fld_map.get(idx)
                if v is not None and np.isfinite(v):
                    rebuilt_vals.append(float(v))
                    continue
                existing_val = existing_field_map.get(idx)
                if existing_val is not None and np.isfinite(existing_val):
                    rebuilt_vals.append(float(existing_val))
                else:
                    rebuilt_vals.append(np.nan)

            if rebuilt_vals:
                write_bin(inst_dir / f"{fld}.day.bin", target_start, rebuilt_vals)

        # 7. 写入更新后的 close.bin
        write_bin(close_bin, close_start, close_vals)

        # 8. 抽样验证：检查 low <= close <= high
        if rebuilt % 200 == 0 and rebuilt > 0:
            # 快速抽样检查
            high_meta = read_bin(inst_dir / "high.day.bin")
            low_meta = read_bin(inst_dir / "low.day.bin")
            if high_meta and low_meta:
                h_start, h_vals = high_meta
                l_start, l_vals = low_meta
                c_vals_sample = close_vals[:min(20, len(close_vals))]
                h_vals_sample = h_vals[:min(20, len(h_vals))]
                l_vals_sample = l_vals[:min(20, len(l_vals))]
                for j in range(min(len(c_vals_sample), len(h_vals_sample), len(l_vals_sample))):
                    c, h, l = c_vals_sample[j], h_vals_sample[j], l_vals_sample[j]
                    if np.isfinite(c) and np.isfinite(h) and np.isfinite(l):
                        if not (l <= c + 0.001 and c <= h + 0.001):
                            violations += 1

        rebuilt += 1
        if rebuilt % 500 == 0:
            logger.info(f"已修复 {rebuilt}/{total} 只...")

    logger.info(
        f"修复完成: {rebuilt} 只重建, {close_extended} 只扩展close, "
        f"{no_raw} 只无raw_data, {skipped} 只跳过, {violations} 处OHLC违例"
    )

    # 抽样验证
    _sample_verify(inst_dirs, date_to_idx, cal_last_idx)

    return rebuilt


def _sample_verify(inst_dirs, date_to_idx, cal_last_idx):
    """抽样5只股票验证 adj_ratio 一致性和 OHLC 约束"""
    logger.info("=== 抽样验证 ===")
    sample_insts = ["sz000001", "sh600519", "sz000858", "sh601318", "sz300750"]
    violations = 0

    for inst in sample_insts:
        inst_dir = QLIB_DIR / "features" / inst
        if not inst_dir.exists():
            continue

        raw_file = RAW_DIR / f"{inst}.parquet"
        if not raw_file.exists():
            continue

        close_meta = read_bin(inst_dir / "close.day.bin")
        high_meta = read_bin(inst_dir / "high.day.bin")
        low_meta = read_bin(inst_dir / "low.day.bin")

        if not close_meta or not high_meta or not low_meta:
            logger.warning(f"  {inst}: 缺少 bin 文件")
            continue

        c_start, c_vals = close_meta
        h_start, h_vals = high_meta
        l_start, l_vals = low_meta

        # 检查结束于日历末端
        c_end = c_start + len(c_vals) - 1
        logger.info(f"  {inst}: close范围 [{c_start}..{c_end}], 日历末端={cal_last_idx}, {'OK' if c_end >= cal_last_idx else 'WARN: 未到末端'}")

        # 读 raw 数据
        raw_df = pd.read_parquet(raw_file, columns=["date", "close", "high", "low"])
        raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce").dt.normalize()
        raw_df = raw_df.dropna(subset=["date"])
        raw_df = raw_df[raw_df["date"].isin(date_to_idx)]
        raw_df["cal_idx"] = raw_df["date"].map(date_to_idx).astype(int)

        raw_close = dict(zip(raw_df["cal_idx"], raw_df["close"]))
        raw_high = dict(zip(raw_df["cal_idx"], raw_df["high"]))

        # 抽样10个点检查 adj_ratio
        check_indices = list(range(c_start, c_start + len(c_vals), max(1, len(c_vals) // 10)))[:10]
        ratios = []
        for idx in check_indices:
            offset = idx - c_start
            if offset < 0 or offset >= len(c_vals):
                continue
            bin_c = float(c_vals[offset])
            raw_c = raw_close.get(idx)
            raw_h = raw_high.get(idx)
            offset_h = idx - h_start
            bin_h = float(h_vals[offset_h]) if 0 <= offset_h < len(h_vals) else None

            if raw_c and raw_c > 0 and np.isfinite(bin_c):
                adj = bin_c / raw_c
                ratios.append(adj)

                # 验证 high 也用了同样的 adj_ratio
                if raw_h is not None and bin_h is not None and np.isfinite(raw_h) and np.isfinite(bin_h):
                    adj_h = bin_h / raw_h
                    if abs(adj - adj_h) > 0.01:  # 允许1%误差（浮点精度）
                        logger.warning(f"    {inst} idx={idx}: adj_close={adj:.6f}, adj_high={adj_h:.6f}, 差异={abs(adj-adj_h):.6f}")

        if ratios:
            logger.info(f"  {inst}: adj_ratio 范围 [{min(ratios):.4f}, {max(ratios):.4f}], 均值={np.mean(ratios):.4f}")

        # 检查 low <= close <= high
        min_len = min(len(c_vals), len(h_vals), len(l_vals))
        for j in range(min_len):
            c, h, l = float(c_vals[j]), float(h_vals[j]), float(l_vals[j])
            if np.isfinite(c) and np.isfinite(h) and np.isfinite(l):
                if not (l - 0.01 <= c <= h + 0.01):
                    violations += 1
                    if violations <= 3:
                        logger.warning(f"    {inst} idx={c_start+j}: low={l:.2f} close={c:.2f} high={h:.2f}")

    if violations > 0:
        logger.warning(f"OHLC 违例总计: {violations}")
    else:
        logger.info("OHLC 约束全部通过")


if __name__ == "__main__":
    fix_all()
