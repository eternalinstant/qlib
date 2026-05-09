#!/usr/bin/env python3
"""
统一数据处理脚本
一条命令完成: tushare下载 → raw_data → qlib格式转换 → 验证

用法:
  python3 scripts/build_qlib_data.py build            # 全量构建
  python3 scripts/build_qlib_data.py build --force    # 强制重新下载
  python3 scripts/build_qlib_data.py build --skip-download  # 跳过下载，只做转换
  python3 scripts/build_qlib_data.py validate         # 验证数据一致性
"""

import sys
import os
import gc
import time
import logging
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TUSHARE_DIR = PROJECT_ROOT / "data" / "tushare"
RAW_DIR = PROJECT_ROOT / "data" / "qlib_data" / "raw_data"
QLIB_DIR = PROJECT_ROOT / "data" / "qlib_data" / "cn_data"
FIELDS = ["open", "high", "low", "close", "volume", "amount", "vwap"]
ALPHA158_BASE_FIELDS = ["open", "high", "low", "close", "volume", "vwap"]


# ── Step 1: Tushare 数据下载 ──────────────────────────────────────────

def step_download_tushare(start_date: str, end_date: str, workers: int):
    """下载/检查 tushare 基础数据 (9个parquet文件)"""
    from modules.data.tushare_downloader import TushareDownloader

    required = [
        "daily_basic.parquet", "adj_factor.parquet",
        "income.parquet", "balancesheet.parquet", "cashflow.parquet",
        "fina_indicator.parquet", "index_daily.parquet",
        "index_weight.parquet", "namechange.parquet",
    ]

    missing = [f for f in required if not (TUSHARE_DIR / f).exists()]
    if not missing:
        logger.info(f"[1/4] Tushare 数据已完整 ({len(required)} 个文件)，跳过下载")
        return True

    logger.info(f"[1/4] 下载 Tushare 数据 (缺 {len(missing)} 个文件)...")
    try:
        dl = TushareDownloader()
        dl.MAX_WORKERS = workers
        dl.download_all(start_date, end_date)
        return True
    except Exception as e:
        logger.error(f"Tushare 下载失败: {e}")
        return False


# ── Step 2: Raw Data 下载 ─────────────────────────────────────────────

def _get_stock_list() -> dict:
    """从 daily_basic.parquet 获取股票列表 {inst_code: ts_code}"""
    df = pd.read_parquet(TUSHARE_DIR / "daily_basic.parquet", columns=["ts_code"])
    ts_codes = df["ts_code"].unique()
    stocks = {}
    for ts_code in ts_codes:
        code, exchange = ts_code.split(".")
        stocks[exchange.lower() + code] = ts_code
    return stocks


def _download_one_stock(pro, ts_code: str, start_date: str, end_date: str):
    """下载单只股票日线数据，带重试"""
    for attempt in range(3):
        try:
            df = pro.daily(
                ts_code=ts_code, start_date=start_date, end_date=end_date,
                fields="ts_code,trade_date,open,high,low,close,vol,amount",
            )
            return df if df is not None and len(df) > 0 else None
        except Exception:
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    return None


def _save_raw_file(raw_path: Path, new_df: pd.DataFrame):
    """将新数据合并到 raw_data parquet 文件"""
    from modules.data.tushare_to_qlib import ensure_vwap

    new_df = new_df.copy()
    new_df["date"] = pd.to_datetime(new_df["trade_date"], format="%Y%m%d")
    new_df = new_df.rename(columns={"vol": "volume"})
    code, exchange = new_df["ts_code"].iloc[0].split(".")
    raw_code = exchange.lower() + code
    new_df["symbol"] = new_df["ts_code"]
    new_df = ensure_vwap(new_df)
    new_df = new_df[["date", "open", "high", "low", "close", "volume", "amount", "vwap", "symbol"]]
    new_df = new_df.sort_values("date")

    if raw_path.exists():
        existing = pd.read_parquet(raw_path)
        existing["date"] = pd.to_datetime(existing["date"])
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        combined = ensure_vwap(combined)
        combined.to_parquet(raw_path, index=False)
    else:
        new_df.to_parquet(raw_path, index=False)


def step_download_raw_data(start_date: str, end_date: str, workers: int):
    """下载 raw_data (每只股票OHLCV)"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_stocks = _get_stock_list()
    existing = {f.stem for f in RAW_DIR.glob("*.parquet")}
    missing = set(all_stocks.keys()) - existing

    logger.info(f"[2/4] Raw data: 已有 {len(existing)} 只, 缺失 {len(missing)} 只")

    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        logger.error("缺少 TUSHARE_TOKEN 环境变量")
        return False
    pro = ts.pro_api(token)

    # 更新已有股票 (最近25天)
    if existing:
        logger.info("  更新已有股票...")
        ts_codes = [all_stocks[s] for s in existing if s in all_stocks]
        _batch_download(pro, ts_codes, start_date=None, end_date=None,
                        recent_days=25, workers=workers)

    # 下载缺失股票
    if missing:
        logger.info(f"  下载 {len(missing)} 只缺失股票...")
        ts_codes = [all_stocks[s] for s in missing]
        _batch_download(pro, ts_codes, start_date=start_date, end_date=end_date,
                        workers=workers)

    final_count = len(list(RAW_DIR.glob("*.parquet")))
    logger.info(f"  Raw data 完成: {len(existing)} → {final_count} 只股票")
    return True


# ── Step 2.5: 补齐前复权依赖 ──────────────────────────────────────────

def _instrument_to_ts_code(inst: str) -> str:
    """instrument (sz000001) -> ts_code (000001.SZ)."""
    if len(inst) < 8:
        return inst
    exchange = inst[:2].upper()
    code = inst[2:]
    return f"{code}.{exchange}"


def _ts_code_to_inst(ts_code_series: pd.Series) -> pd.Series:
    """ts_code (000001.SZ) -> instrument (sz000001)."""
    return ts_code_series.apply(
        lambda x: x.split(".")[1].lower() + x.split(".")[0] if "." in x else str(x).lower()
    )


def _adj_factor_instruments() -> set[str]:
    adj_path = TUSHARE_DIR / "adj_factor.parquet"
    if not adj_path.exists():
        return set()

    try:
        adj = pd.read_parquet(adj_path, columns=["ts_code"])
    except Exception:
        return set()
    if adj.empty:
        return set()
    return set(_ts_code_to_inst(adj["ts_code"]).dropna().unique())


def _download_one_adj_factor(pro, ts_code: str, start_date: str, end_date: str):
    for attempt in range(3):
        try:
            df = pro.adj_factor(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields="ts_code,trade_date,adj_factor",
            )
            return df if df is not None and len(df) > 0 else None
        except Exception:
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    return None


def _merge_adj_factor(new_parts: list[pd.DataFrame]) -> None:
    if not new_parts:
        return

    adj_path = TUSHARE_DIR / "adj_factor.parquet"
    new_df = pd.concat(new_parts, ignore_index=True)
    new_df = new_df[["ts_code", "trade_date", "adj_factor"]].copy()

    if adj_path.exists():
        existing = pd.read_parquet(adj_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    combined = combined.sort_values(["ts_code", "trade_date"])
    combined.to_parquet(adj_path, index=False)


def step_repair_adj_factor_coverage(start_date: str, end_date: str, workers: int) -> bool:
    """补齐 raw_data 中已有股票缺失的 adj_factor。

    Alpha158 的价格字段走前复权口径；没有 adj_factor 的股票不能可靠写出
    open/high/low/close/vwap bin，所以这里在转换前补齐复权因子覆盖。
    """
    raw_instruments = {f.stem for f in RAW_DIR.glob("*.parquet")}
    if not raw_instruments:
        logger.warning("[2.5/4] raw_data 为空，跳过 adj_factor 覆盖检查")
        return True

    covered = _adj_factor_instruments()
    missing = sorted(raw_instruments - covered)
    if not missing:
        logger.info("[2.5/4] adj_factor 覆盖完整，跳过补齐")
        return True

    logger.info(f"[2.5/4] 补齐 adj_factor: {len(missing)} 只股票缺失")

    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        logger.error("缺少 TUSHARE_TOKEN，无法补齐 adj_factor")
        return False
    pro = ts.pro_api(token)

    ts_codes = [_instrument_to_ts_code(inst) for inst in missing]
    new_parts = []
    failed = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one_adj_factor, pro, ts_code, start_date, end_date): ts_code
            for ts_code in ts_codes
        }
        for future in as_completed(futures):
            ts_code = futures[future]
            df = future.result()
            if df is not None:
                new_parts.append(df)
            else:
                failed.append(ts_code)

            done = len(new_parts) + len(failed)
            if done % 100 == 0:
                logger.info(f"    adj_factor 补齐进度: {done}/{len(ts_codes)}")

    if new_parts:
        _merge_adj_factor(new_parts)
        logger.info(f"    adj_factor 新增/合并 {sum(len(x) for x in new_parts):,} 条")

    unresolved = sorted(raw_instruments - _adj_factor_instruments())
    if unresolved:
        preview = ", ".join(unresolved[:20])
        suffix = " ..." if len(unresolved) > 20 else ""
        logger.error(f"    adj_factor 仍缺 {len(unresolved)} 只: {preview}{suffix}")
        return False

    logger.info("    adj_factor 覆盖已补齐")
    return True


def _batch_download(pro, ts_codes, start_date=None, end_date=None,
                    recent_days=None, workers=4):
    """批量下载并保存，统计成功/失败数"""
    total = len(ts_codes)
    success = 0
    fail = 0

    def download_and_save(ts_code):
        code, exchange = ts_code.split(".")
        inst = exchange.lower() + code
        raw_path = RAW_DIR / f"{inst}.parquet"

        if recent_days:
            sd = (datetime.now() - pd.Timedelta(days=recent_days)).strftime("%Y%m%d")
            ed = end_date
        else:
            sd = start_date
            ed = end_date

        df = _download_one_stock(pro, ts_code, sd, ed)
        if df is not None:
            _save_raw_file(raw_path, df)
            return True
        return False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_and_save, code): code for code in ts_codes}
        for future in as_completed(futures):
            if future.result():
                success += 1
            else:
                fail += 1
            if (success + fail) % 500 == 0:
                logger.info(f"    进度: {success + fail}/{total} (成功={success}, 失败={fail})")

    if fail > 0:
        logger.warning(f"    下载完成 {total} 只: 成功={success}, 失败={fail}")
    else:
        logger.info(f"    完成 {success}/{total}")


# ── Step 3: Qlib 数据转换 ─────────────────────────────────────────────

def step_build_qlib_data(
    start_date: str,
    end_date: str,
    force: bool = False,
    build_factor_data: bool = True,
    batch_size: int = 200,
):
    """转换 tushare + raw_data → qlib 格式"""
    from modules.data.tushare_to_qlib import TushareToQlibConverter

    logger.info("[3/4] 构建 Qlib 数据...")

    # 确保目录存在
    (QLIB_DIR / "calendars").mkdir(parents=True, exist_ok=True)
    (QLIB_DIR / "features").mkdir(parents=True, exist_ok=True)

    converter = TushareToQlibConverter(
        tushare_dir=str(TUSHARE_DIR),
        qlib_dir=str(QLIB_DIR),
    )

    # 3a. 生成日历
    logger.info("  3a. 生成交易历...")
    _build_calendar(start_date, end_date)

    # 3b. 合并基本面+财务数据
    if build_factor_data:
        factor_path = QLIB_DIR / "factor_data.parquet"
        if factor_path.exists() and not force:
            logger.info(
                f"  3b. factor_data 已存在 ({factor_path.stat().st_size // 1024 // 1024}MB)，执行增量合并..."
            )
        else:
            logger.info("  3b. 合并基本面+财务数据...")
        factor_df = converter.convert()
        if factor_df is not None:
            converter.save(factor_df)
            del factor_df
            gc.collect()
    else:
        logger.info("  3b. 跳过 factor_data 合并 (Alpha158-only)")

    # 3c. 为 raw_data 中的每只股票创建 features 目录 + instruments.txt
    logger.info("  3c. 创建 features 目录 + instruments.txt...")
    raw_files = list(RAW_DIR.glob("*.parquet"))
    instruments = []
    for raw_path in raw_files:
        inst_dir = QLIB_DIR / "features" / raw_path.stem
        inst_dir.mkdir(exist_ok=True)
        instruments.append(raw_path.stem)

    (QLIB_DIR / "instruments").mkdir(parents=True, exist_ok=True)
    with open(QLIB_DIR / "instruments" / "all.txt", "w") as f:
        for inst in sorted(instruments):
            raw_path = RAW_DIR / f"{inst}.parquet"
            inst_start, inst_end = "", ""
            if raw_path.exists():
                try:
                    df = pd.read_parquet(raw_path, columns=["date"])
                    if not df.empty:
                        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
                        if not dates.empty:
                            inst_start = dates.min().strftime("%Y-%m-%d")
                            inst_end = dates.max().strftime("%Y-%m-%d")
                except Exception:
                    pass
            f.write(f"{inst}\t{inst_start}\t{inst_end}\n")
    logger.info(f"  instruments.txt: {len(instruments)} 只股票 (已写入实际日期范围)")

    # 3d. 分批计算前复权价格并写入 bin（控制内存峰值）
    logger.info("  3d. 分批计算前复权价格并写入 bin...")
    converter.build_adjusted_bins_batched(batch_size=batch_size)

    # 3e. 对没有 adj_factor 的股票写入原始价格 bin
    has_bin = {d.name for d in (QLIB_DIR / "features").iterdir()
               if d.is_dir() and (d / "close.day.bin").exists()}
    no_bin = [f for f in raw_files if f.stem not in has_bin]
    if no_bin:
        preview = ", ".join(f.stem for f in no_bin[:10])
        suffix = " ..." if len(no_bin) > 10 else ""
        logger.error(
            "  3e. 发现 %s 只股票缺少前复权 bin，拒绝回退到未复权数据: %s%s",
            len(no_bin),
            preview,
            suffix,
        )
        logger.error("  请先补齐 adj_factor 后再重新构建。")
        return False

    missing_alpha158 = _find_missing_alpha158_bins(raw_files)
    if missing_alpha158:
        preview = ", ".join(
            f"{inst}:{'/'.join(fields[:3])}{'...' if len(fields) > 3 else ''}"
            for inst, fields in missing_alpha158[:10]
        )
        suffix = " ..." if len(missing_alpha158) > 10 else ""
        logger.error(
            "  Alpha158 基础字段不完整: %s 只股票缺字段 (%s%s)",
            len(missing_alpha158),
            preview,
            suffix,
        )
        return False

    # 3f. 生成 supplement_daily.parquet（OHLCV 补充数据，供测试使用）
    logger.info("  3f. 生成 supplement_daily.parquet...")
    converter.build_supplement_daily()

    logger.info("  Qlib 数据构建完成")
    return True


def _find_missing_alpha158_bins(raw_files):
    """检查官方 Alpha158 需要的基础字段 bin 是否齐全。"""
    missing = []
    for raw_path in raw_files:
        inst_dir = QLIB_DIR / "features" / raw_path.stem
        missing_fields = [
            field for field in ALPHA158_BASE_FIELDS
            if not (inst_dir / f"{field}.day.bin").exists()
        ]
        if missing_fields:
            missing.append((raw_path.stem, missing_fields))
    return missing


def _build_calendar(start_date: str, end_date: str):
    """从 daily_basic 生成交易日历"""
    db = pd.read_parquet(TUSHARE_DIR / "daily_basic.parquet", columns=["trade_date"])
    dates = pd.to_datetime(db["trade_date"].unique(), format="%Y%m%d").sort_values()
    start = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) if end_date else dates.max()
    dates = dates[(dates >= start) & (dates <= end_dt)]

    cal_file = QLIB_DIR / "calendars" / "day.txt"
    dates.strftime("%Y-%m-%d").to_series().to_csv(cal_file, index=False, header=False)
    logger.info(f"  日历: {len(dates)} 个交易日 ({dates.min().date()} ~ {dates.max().date()})")


# ── Step 4: 验证 ──────────────────────────────────────────────────────

def step_validate():
    """运行全量数据验证"""
    logger.info("[4/4] 验证数据 (全量)...")
    validate_script = PROJECT_ROOT / "scripts" / "validate_data.py"

    result = subprocess.run(
        [sys.executable, str(validate_script), "--data-root", str(QLIB_DIR)],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.warning(f"验证脚本退出码: {result.returncode}")
        return False
    return True


# ── 主入口 ─────────────────────────────────────────────────────────────

def cmd_build(args):
    """全量构建"""
    start_time = time.time()
    end_date = args.end or datetime.now().strftime("%Y%m%d")

    logger.info("=" * 60)
    logger.info("Qlib 数据构建 (全量)")
    logger.info(f"  时间范围: {args.start} ~ {end_date}")
    logger.info(f"  并发数:   {args.workers}")
    logger.info(f"  数据目录: {PROJECT_ROOT / 'data'}")
    logger.info("=" * 60)

    if not args.skip_download:
        ok = step_download_tushare(args.start, end_date, args.workers)
        if not ok and not args.force:
            logger.error("Tushare 下载失败，退出")
            return 1

    if not args.skip_raw:
        ok = step_download_raw_data(args.start, end_date, args.workers)
        if not ok:
            logger.error("Raw data 下载失败，退出")
            return 1
    else:
        logger.info("[2/4] 跳过 raw_data 下载/更新")

    if not args.skip_adj_factor_repair:
        ok = step_repair_adj_factor_coverage(args.start, end_date, args.workers)
        if not ok:
            logger.error("adj_factor 覆盖补齐失败，退出")
            return 1
    else:
        logger.info("[2.5/4] 跳过 adj_factor 覆盖补齐")

    ok = step_build_qlib_data(
        args.start,
        end_date,
        force=args.force,
        build_factor_data=not args.skip_factor_data,
        batch_size=args.batch_size,
    )
    if not ok:
        logger.error("Qlib 数据构建失败，退出")
        return 1

    if args.validate:
        step_validate()

    elapsed = time.time() - start_time
    logger.info(f"总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    return 0


def cmd_validate(args):
    """验证数据"""
    step_validate()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Qlib 数据统一处理脚本")
    sub = parser.add_subparsers(dest="command")

    # build
    p_build = sub.add_parser("build", help="全量构建")
    p_build.add_argument("--start", default="20160101")
    p_build.add_argument("--end", default=None)
    p_build.add_argument("--workers", type=int, default=4)
    p_build.add_argument("--force", action="store_true", help="强制重新下载")
    p_build.add_argument("--skip-download", action="store_true", help="跳过 tushare 下载")
    p_build.add_argument("--skip-raw", action="store_true", help="跳过 raw_data 下载/更新")
    p_build.add_argument("--skip-adj-factor-repair", action="store_true", help="跳过缺失 adj_factor 补齐")
    p_build.add_argument("--skip-factor-data", action="store_true", help="跳过 factor_data 合并")
    p_build.add_argument("--batch-size", type=int, default=200, help="前复权 bin 构建批大小")
    p_build.add_argument("--validate", action="store_true", help="构建后验证")
    p_build.set_defaults(func=cmd_build)

    # validate
    p_val = sub.add_parser("validate", help="验证数据一致性")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
