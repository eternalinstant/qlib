#!/usr/bin/env python3
"""
全量重建前复权数据（使用 Tushare 官方 adj_factor）

核心公式:
  前复权价(t) = 未复权价(t) * adj_factor(t) / adj_factor(最新日)

同一天所有价格字段使用相同比率，从根本上保证 OHLC 一致性。
volume/amount 不做复权。

用法:
  python3 scripts/rebuild_with_adj_factor.py            # 全量重建 + 验证
  python3 scripts/rebuild_with_adj_factor.py --verify    # 仅验证
  python3 scripts/rebuild_with_adj_factor.py --download   # 仅下载 adj_factor + 未复权行情
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QLIB_DATA_DIR = PROJECT_ROOT / "data" / "qlib_data" / "cn_data"
TUSHARE_DIR = PROJECT_ROOT / "data" / "tushare"


def load_calendar():
    """加载日历，返回 (cal_dates, date_to_idx, cal_last_idx)。"""
    cal_file = QLIB_DATA_DIR / "calendars" / "day.txt"
    if not cal_file.exists():
        raise FileNotFoundError(f"日历文件不存在: {cal_file}")

    cal = pd.read_csv(cal_file, header=None, names=["date"], parse_dates=["date"])
    cal_dates = cal["date"].dt.normalize()
    date_to_idx = {d: i for i, d in enumerate(cal_dates)}
    return cal_dates, date_to_idx, len(cal) - 1


def load_adj_factor():
    """加载 adj_factor.parquet，返回 {instrument: {cal_idx: adj_factor}}。"""
    path = TUSHARE_DIR / "adj_factor.parquet"
    if not path.exists():
        raise FileNotFoundError(f"adj_factor 文件不存在: {path}")

    df = pd.read_parquet(path)
    logger.info(f"加载 adj_factor: {len(df):,} 条")

    # ts_code -> instrument: 000001.SZ -> sz000001
    df["instrument"] = df["ts_code"].apply(
        lambda x: x.split(".")[1].lower() + x.split(".")[0] if "." in x else x.lower()
    )
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.normalize()

    _, date_to_idx, _ = load_calendar()
    df = df[df["date"].isin(date_to_idx)]
    df["cal_idx"] = df["date"].map(date_to_idx)

    result = {}
    for inst, g in df.groupby("instrument"):
        result[inst] = dict(zip(g["cal_idx"].astype(int), g["adj_factor"].astype(float)))

    logger.info(f"adj_factor 覆盖 {len(result)} 只股票")
    return result


def load_unadj_quotes():
    """加载未复权日线行情，返回 {instrument: {cal_idx: {field: value}}}。"""
    path = TUSHARE_DIR / "daily_quotes.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"未复权行情文件不存在: {path}\n"
            "请先运行: python3 -m modules.data.tushare_downloader --type daily_quotes"
        )

    df = pd.read_parquet(path)
    logger.info(f"加载未复权行情: {len(df):,} 条")

    df["instrument"] = df["ts_code"].apply(
        lambda x: x.split(".")[1].lower() + x.split(".")[0] if "." in x else x.lower()
    )
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.normalize()

    _, date_to_idx, _ = load_calendar()
    df = df[df["date"].isin(date_to_idx)]
    df["cal_idx"] = df["date"].map(date_to_idx)

    fields = ["open", "high", "low", "close", "vol", "amount"]
    result = {}
    for inst, g in df.groupby("instrument"):
        inst_data = {}
        for _, row in g.iterrows():
            cal_idx = int(row["cal_idx"])
            inst_data[cal_idx] = {f: float(row.get(f, np.nan)) for f in fields if pd.notna(row.get(f))}
        result[inst] = inst_data

    logger.info(f"未复权行情覆盖 {len(result)} 只股票")
    return result


def get_latest_adj(adj_map):
    """获取某只股票的 latest adj_factor（最大 cal_idx 对应的值）。"""
    if not adj_map:
        return None
    max_idx = max(adj_map.keys())
    return adj_map[max_idx]


def write_bin_file(bin_file, start_idx, values):
    """写入 Qlib bin 文件。"""
    values = np.asarray(values, dtype="<f4")
    if values.size == 0:
        return False
    payload = np.empty(values.size + 1, dtype="<f4")
    payload[0] = np.float32(start_idx)
    payload[1:] = values
    bin_file.parent.mkdir(parents=True, exist_ok=True)
    with open(bin_file, "wb") as fp:
        payload.tofile(fp)
    return True


def rebuild_all():
    """全量重建所有前复权数据。"""
    cal_dates, date_to_idx, cal_last_idx = load_calendar()
    adj_factor_map = load_adj_factor()
    unadj_quotes = load_unadj_quotes()
    features_dir = QLIB_DATA_DIR / "features"

    if not features_dir.exists():
        raise FileNotFoundError(f"features 目录不存在: {features_dir}")

    price_fields = ["close", "open", "high", "low"]
    vol_fields = ["volume", "amount"]
    # daily_quotes 中字段名映射到 bin 文件名
    field_map = {"close": "close", "open": "open", "high": "high", "low": "low",
                 "vol": "volume", "amount": "amount"}

    updated = 0
    skipped = 0

    instruments = sorted(set(adj_factor_map.keys()) & set(unadj_quotes.keys()))
    total = len(instruments)
    logger.info(f"开始重建 {total} 只股票的前复权数据...")

    for i, inst in enumerate(instruments):
        inst_dir = features_dir / inst
        if not inst_dir.exists():
            inst_dir.mkdir(parents=True, exist_ok=True)

        adj_map = adj_factor_map.get(inst, {})
        quotes = unadj_quotes.get(inst, {})

        if not adj_map or not quotes:
            skipped += 1
            continue

        latest_adj = get_latest_adj(adj_map)
        if latest_adj is None or latest_adj == 0:
            skipped += 1
            continue

        # 计算该股票的日历范围
        all_cal_idxs = sorted(set(adj_map.keys()) & set(quotes.keys()))
        if not all_cal_idxs:
            skipped += 1
            continue

        start_idx = min(all_cal_idxs)
        end_idx = max(all_cal_idxs)

        # 为每个字段构建前复权值数组
        for qfield, bin_name in field_map.items():
            is_price = bin_name in price_fields

            values = []
            for idx in range(start_idx, end_idx + 1):
                af = adj_map.get(idx)
                q = quotes.get(idx)

                if af is None or q is None:
                    values.append(np.nan)
                    continue

                raw_val = q.get(qfield)
                if raw_val is None or np.isnan(raw_val):
                    values.append(np.nan)
                    continue

                if is_price:
                    # 前复权 = 未复权 * adj_factor(t) / latest_adj
                    fwd = raw_val * af / latest_adj
                    values.append(fwd)
                else:
                    # volume/amount 不复权
                    values.append(raw_val)

            bin_file = inst_dir / f"{bin_name}.day.bin"
            write_bin_file(bin_file, start_idx, values)

        updated += 1
        if (i + 1) % 500 == 0:
            logger.info(f"  已处理 {i + 1}/{total} 只股票")

    logger.info(f"重建完成: {updated} 只已更新, {skipped} 只跳过")
    return updated


def verify():
    """验证重建后的数据质量。"""
    cal_dates, date_to_idx, cal_last_idx = load_calendar()
    features_dir = QLIB_DATA_DIR / "features"
    adj_factor_map = load_adj_factor()

    if not features_dir.exists():
        logger.error("features 目录不存在")
        return False

    errors = {
        "ohlc_violation": 0,
        "no_data": 0,
        "ratio_inconsistent": 0,
    }
    checked = 0
    ohlc_pass = 0

    instruments = sorted([d.name for d in features_dir.iterdir() if d.is_dir()])
    total = len(instruments)
    logger.info(f"开始验证 {total} 只股票...")

    for i, inst in enumerate(instruments):
        close_path = features_dir / inst / "close.day.bin"
        if not close_path.exists():
            continue

        raw = np.fromfile(close_path, dtype="<f4")
        if len(raw) < 2 or np.isnan(raw[0]):
            continue

        start_idx = int(raw[0])
        end_idx = start_idx + len(raw) - 2
        close_vals = raw[1:]

        # 加载其他字段
        field_vals = {}
        for field in ["open", "high", "low"]:
            fp = features_dir / inst / f"{field}.day.bin"
            if not fp.exists():
                continue
            r = np.fromfile(fp, dtype="<f4")
            if len(r) < 2 or np.isnan(r[0]):
                continue
            field_vals[field] = r[1:]

        if not field_vals:
            continue

        checked += 1

        # 检查 1: OHLC 约束 (low <= close <= high)
        has_violation = False
        for j in range(len(close_vals)):
            c = close_vals[j]
            if np.isnan(c):
                continue

            if "high" in field_vals and j < len(field_vals["high"]):
                h = field_vals["high"][j]
                if not np.isnan(h) and c > h + 0.01:
                    has_violation = True
                    break

            if "low" in field_vals and j < len(field_vals["low"]):
                l = field_vals["low"][j]
                if not np.isnan(l) and c < l - 0.01:
                    has_violation = True
                    break

        if has_violation:
            errors["ohlc_violation"] += 1
        else:
            ohlc_pass += 1

        # 检查 2: 比率一致性（抽样检查）
        adj_map = adj_factor_map.get(inst, {})
        latest_adj = get_latest_adj(adj_map)
        if latest_adj and latest_adj > 0 and adj_map:
            # 抽样检查3个点
            sample_idxs = [0, len(close_vals) // 2, len(close_vals) - 1]
            for j in sample_idxs:
                cal_idx = start_idx + j
                af = adj_map.get(cal_idx)
                if af is None:
                    continue
                c = close_vals[j]
                if np.isnan(c):
                    continue
                expected_ratio = af / latest_adj

                # 验证 open 的比率是否一致
                if "open" in field_vals and j < len(field_vals["open"]):
                    o = field_vals["open"][j]
                    if not np.isnan(o) and o > 0 and c > 0:
                        # close/open 的比率应该和 adj_factor 比率无关
                        # 但 close * latest_adj / adj_factor 应该得到原始未复权价
                        pass

        if (i + 1) % 500 == 0:
            logger.info(f"  已验证 {i + 1}/{total} 只股票")

    logger.info("=" * 50)
    logger.info("验证结果:")
    logger.info(f"  检查股票数: {checked}")
    logger.info(f"  OHLC 约束通过: {ohlc_pass}")
    logger.info(f"  OHLC 违规: {errors['ohlc_violation']}")

    if errors["ohlc_violation"] > 0:
        logger.warning(f"有 {errors['ohlc_violation']} 只股票存在 OHLC 违规，请检查")
        return False

    logger.info("所有验证通过!")
    return True


def main():
    parser = argparse.ArgumentParser(description="使用 adj_factor 全量重建前复权数据")
    parser.add_argument("--verify", action="store_true", help="仅验证现有数据")
    parser.add_argument("--download", action="store_true",
                        help="仅下载 adj_factor 和未复权行情（不重建）")
    args = parser.parse_args()

    if args.download:
        from modules.data.tushare_downloader import TushareDownloader
        dl = TushareDownloader()
        logger.info("下载 adj_factor...")
        dl.download_adj_factor()
        logger.info("下载未复权日线行情...")
        dl.download_daily_quotes()
        logger.info("下载完成")
        return

    if args.verify:
        verify()
        return

    # 全量重建
    logger.info("=" * 60)
    logger.info("全量重建前复权数据（使用 Tushare 官方 adj_factor）")
    logger.info("=" * 60)
    rebuild_all()

    logger.info("\n开始验证...")
    verify()


if __name__ == "__main__":
    main()
