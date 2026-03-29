#!/usr/bin/env python3
"""
数据更新脚本
一键更新 Tushare 数据：股票基本信息、财务数据、事件数据、行为数据。
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TUSHARE_DIR = PROJECT_ROOT / "data" / "tushare"
DEFAULT_DAILY_START = "20200101"
DEFAULT_FINANCIAL_START = "20100101"
DEFAULT_SUPPLEMENTAL_START = "20200101"


def _ensure_tushare_dir() -> None:
    TUSHARE_DIR.mkdir(parents=True, exist_ok=True)


def _get_pro():
    import tushare as ts

    return ts.pro_api()


def _today_str() -> str:
    return pd.Timestamp.today().strftime("%Y%m%d")


def _to_timestamp(value: Optional[object]) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def _shift_date(date_str: str, days: int) -> str:
    ts = _to_timestamp(date_str)
    if ts is None or pd.isna(ts):
        return date_str
    return (ts + pd.Timedelta(days=days)).strftime("%Y%m%d")


def _read_existing(output_path: Path) -> Optional[pd.DataFrame]:
    if output_path.exists():
        return pd.read_parquet(output_path)
    return None


def _infer_start_date(
    output_path: Path,
    date_col: str,
    fallback_start: str,
    rewind_days: int = 0,
) -> Tuple[str, Optional[pd.DataFrame]]:
    existing = _read_existing(output_path)
    if existing is None or existing.empty or date_col not in existing.columns:
        return fallback_start, existing

    max_date = _to_timestamp(existing[date_col].max())
    if max_date is None or pd.isna(max_date):
        return fallback_start, existing

    start_ts = max_date - pd.Timedelta(days=rewind_days)
    return start_ts.strftime("%Y%m%d"), existing


def _merge_and_save(
    existing: Optional[pd.DataFrame],
    fresh: pd.DataFrame,
    output_path: Path,
    dedup_subset: Sequence[str],
    sort_by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, fresh], ignore_index=True)
    else:
        combined = fresh.copy()

    combined = combined.drop_duplicates(subset=list(dedup_subset), keep="last")

    if sort_by:
        existing_cols = [col for col in sort_by if col in combined.columns]
        if existing_cols:
            combined = combined.sort_values(existing_cols)

    combined.to_parquet(output_path, index=False)
    return combined


def _print_block(title: str) -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def _print_success(output_path: Path, fresh_rows: int, extra: Optional[str] = None) -> None:
    print(f"✓ 已更新 {fresh_rows} 条记录")
    print(f"  保存到: {output_path}")
    if extra:
        print(f"  {extra}")


def _fetch_all_stock_codes(pro) -> list:
    stock_df = pro.stock_basic(list_status="L", fields="ts_code")
    return stock_df["ts_code"].dropna().astype(str).tolist()


def _get_trade_dates(start_date: str, end_date: str) -> list:
    daily_path = TUSHARE_DIR / "daily_basic.parquet"
    if daily_path.exists():
        df = pd.read_parquet(daily_path, columns=["trade_date"])
        if not df.empty:
            dates = (
                df["trade_date"]
                .astype(str)
                .dropna()
                .loc[lambda s: (s >= start_date) & (s <= end_date)]
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            if dates:
                return dates

    return [
        dt.strftime("%Y%m%d")
        for dt in pd.bdate_range(_to_timestamp(start_date), _to_timestamp(end_date))
    ]


def _update_range_endpoint(
    title: str,
    api_method: str,
    output_name: str,
    date_col: str,
    dedup_subset: Sequence[str],
    fallback_start: str,
    rewind_days: int = 0,
    extra_params: Optional[dict] = None,
    sort_by: Optional[Sequence[str]] = None,
    chunk_days: Optional[int] = None,
    force_full_refresh: bool = False,
    request_interval: float = 0.0,
) -> bool:
    _print_block(title)

    try:
        _ensure_tushare_dir()
        pro = _get_pro()
        output_path = TUSHARE_DIR / output_name
        if force_full_refresh:
            start_date = fallback_start
            existing = None
        else:
            start_date, existing = _infer_start_date(
                output_path, date_col, fallback_start, rewind_days
            )
        end_date = _today_str()

        if _to_timestamp(start_date) > _to_timestamp(end_date):
            print("✓ 无新数据")
            return True

        fresh_frames = []
        had_errors = False
        start_ts = _to_timestamp(start_date)
        end_ts = _to_timestamp(end_date)

        if chunk_days and start_ts is not None and end_ts is not None:
            chunk_ranges = []
            cursor = start_ts
            while cursor <= end_ts:
                chunk_end = min(cursor + pd.Timedelta(days=chunk_days - 1), end_ts)
                chunk_ranges.append((cursor.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
                cursor = chunk_end + pd.Timedelta(days=1)
        else:
            chunk_ranges = [(start_date, end_date)]

        for idx, (chunk_start, chunk_end) in enumerate(chunk_ranges, start=1):
            kwargs = {"start_date": chunk_start, "end_date": chunk_end}
            if extra_params:
                kwargs.update(extra_params)

            for attempt in range(2):
                try:
                    df = getattr(pro, api_method)(**kwargs)
                    if df is not None and not df.empty:
                        fresh_frames.append(df)
                    break
                except Exception as e:
                    message = str(e)
                    if "每分钟最多访问该接口" in message and attempt == 0:
                        print(f"  命中频控，暂停 65 秒后继续: {chunk_start} ~ {chunk_end}")
                        time.sleep(65)
                        continue
                    had_errors = True
                    print(f"  区间 {chunk_start} ~ {chunk_end} 失败: {e}")
                    break

            if len(chunk_ranges) > 1 and (idx % 10 == 0 or idx == len(chunk_ranges)):
                print(f"  进度: {idx}/{len(chunk_ranges)}")

            if request_interval > 0 and idx < len(chunk_ranges):
                time.sleep(request_interval)

        if not fresh_frames:
            if had_errors:
                print("✗ 未获取到有效数据")
                return False
            print("✓ 无新数据")
            return True

        fresh = pd.concat(fresh_frames, ignore_index=True)
        _merge_and_save(existing, fresh, output_path, dedup_subset, sort_by=sort_by)
        _print_success(output_path, len(fresh), f"日期范围: {start_date} ~ {end_date}")
        return True
    except Exception as e:
        print(f"✗ 更新失败: {e}")
        return False


def _update_loop_endpoint(
    title: str,
    output_name: str,
    date_col: str,
    dedup_subset: Sequence[str],
    fallback_start: str,
    fetch_dates: Iterable[str],
    fetch_fn: Callable[[object, str], pd.DataFrame],
    rewind_days: int = 0,
    sort_by: Optional[Sequence[str]] = None,
    request_interval: float = 0.0,
    force_full_refresh: bool = False,
) -> bool:
    _print_block(title)

    try:
        _ensure_tushare_dir()
        pro = _get_pro()
        output_path = TUSHARE_DIR / output_name
        if force_full_refresh:
            start_date = fallback_start
            existing = None
        else:
            start_date, existing = _infer_start_date(
                output_path, date_col, fallback_start, rewind_days
            )
        target_dates = [d for d in fetch_dates if d >= start_date]

        if not target_dates:
            print("✓ 无新数据")
            return True

        frames = []
        had_errors = False
        for idx, date_str in enumerate(target_dates, start=1):
            for attempt in range(2):
                try:
                    df = fetch_fn(pro, date_str)
                    if df is not None and not df.empty:
                        frames.append(df)
                    break
                except Exception as e:
                    message = str(e)
                    if "每分钟最多访问该接口400次" in message and attempt == 0:
                        print(f"  命中频控，暂停 65 秒后继续: {date_str}")
                        time.sleep(65)
                        continue
                    had_errors = True
                    print(f"  日期 {date_str} 失败: {e}")
                    break

            if idx % 100 == 0 or idx == len(target_dates):
                print(f"  进度: {idx}/{len(target_dates)}")

            if request_interval > 0 and idx < len(target_dates):
                time.sleep(request_interval)

        if not frames:
            if had_errors:
                print("✗ 未获取到有效数据")
                return False
            print("✓ 无新数据")
            return True

        fresh = pd.concat(frames, ignore_index=True)
        _merge_and_save(existing, fresh, output_path, dedup_subset, sort_by=sort_by)
        _print_success(output_path, len(fresh), f"抓取日期数: {len(target_dates)}")
        return True
    except Exception as e:
        print(f"✗ 更新失败: {e}")
        return False


def update_stock_basic():
    """更新股票基本信息（代码、名称、行业）"""
    _print_block("更新股票基本信息...")

    try:
        _ensure_tushare_dir()
        pro = _get_pro()

        df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry")
        output_path = TUSHARE_DIR / "stock_basic.csv"
        df.to_csv(output_path, index=False)

        print(f"✓ 已更新 {len(df)} 只股票")
        print(f"  保存到: {output_path}")

        industry_path = TUSHARE_DIR / "stock_industry.csv"
        industry_df = df[["ts_code", "industry"]].copy()
        industry_df.to_csv(industry_path, index=False)
        print(f"✓ 已保存行业数据到: {industry_path}")
        return True
    except Exception as e:
        print(f"✗ 更新失败: {e}")
        return False


def update_daily_basic():
    """更新每日基础数据"""
    return _update_range_endpoint(
        title="更新每日基础数据...",
        api_method="daily_basic",
        output_name="daily_basic.parquet",
        date_col="trade_date",
        dedup_subset=["ts_code", "trade_date"],
        fallback_start=DEFAULT_DAILY_START,
        sort_by=["trade_date", "ts_code"],
    )


def _update_financial_table(
    title: str,
    api_method: str,
    output_name: str,
    fallback_start: str = DEFAULT_FINANCIAL_START,
) -> bool:
    _print_block(title)

    try:
        _ensure_tushare_dir()
        pro = _get_pro()
        output_path = TUSHARE_DIR / output_name
        start_date, existing = _infer_start_date(output_path, "end_date", fallback_start, rewind_days=365)
        ts_codes = _fetch_all_stock_codes(pro)

        frames = []
        batch_size = 500
        for idx in range(0, len(ts_codes), batch_size):
            batch = ts_codes[idx: idx + batch_size]
            try:
                df = getattr(pro, api_method)(ts_code=",".join(batch), start_date=start_date)
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as e:
                print(f"  批次 {idx // batch_size + 1} 失败: {e}")

        if not frames:
            print("✓ 无新数据")
            return True

        fresh = pd.concat(frames, ignore_index=True)
        dedup_subset = ["ts_code", "end_date"]
        if "ann_date" in fresh.columns:
            dedup_subset.append("ann_date")
        if "report_type" in fresh.columns:
            dedup_subset.append("report_type")

        _merge_and_save(existing, fresh, output_path, dedup_subset, sort_by=["end_date", "ts_code"])
        _print_success(output_path, len(fresh))
        return True
    except Exception as e:
        print(f"✗ 更新失败: {e}")
        return False


def update_fina_indicator():
    """更新财务指标数据"""
    return _update_financial_table("更新财务指标数据...", "fina_indicator", "fina_indicator.parquet")


def update_income():
    """更新利润表数据"""
    return _update_financial_table("更新利润表数据...", "income", "income.parquet")


def update_cashflow():
    """更新现金流量表数据"""
    return _update_financial_table("更新现金流量表数据...", "cashflow", "cashflow.parquet")


def update_balancesheet():
    """更新资产负债表数据"""
    return _update_financial_table("更新资产负债表数据...", "balancesheet", "balancesheet.parquet")


def update_forecast(
    start_date: str = DEFAULT_SUPPLEMENTAL_START,
    force_full_refresh: bool = False,
):
    """更新业绩预告数据。forecast 接口需要 ann_date 或 ts_code，按公告日期循环抓取。"""
    end_date = _today_str()
    calendar_dates = [
        dt.strftime("%Y%m%d")
        for dt in pd.date_range(_to_timestamp(start_date), _to_timestamp(end_date), freq="D")
    ]
    return _update_loop_endpoint(
        title="更新业绩预告数据...",
        output_name="forecast.parquet",
        date_col="ann_date",
        dedup_subset=["ts_code", "ann_date", "end_date"],
        fallback_start=start_date,
        fetch_dates=calendar_dates,
        fetch_fn=lambda pro, date_str: pro.forecast(ann_date=date_str),
        rewind_days=7,
        sort_by=["ann_date", "ts_code"],
        request_interval=0.18,
        force_full_refresh=force_full_refresh,
    )


def update_express(
    start_date: str = DEFAULT_SUPPLEMENTAL_START,
    force_full_refresh: bool = False,
):
    """更新业绩快报数据"""
    return _update_range_endpoint(
        title="更新业绩快报数据...",
        api_method="express",
        output_name="express.parquet",
        date_col="ann_date",
        dedup_subset=["ts_code", "ann_date", "end_date"],
        fallback_start=start_date,
        rewind_days=30,
        sort_by=["ann_date", "ts_code"],
        chunk_days=30,
        force_full_refresh=force_full_refresh,
    )


def update_moneyflow(
    start_date: str = DEFAULT_SUPPLEMENTAL_START,
    force_full_refresh: bool = False,
):
    """更新资金流向数据"""
    trade_dates = _get_trade_dates(start_date, _today_str())
    return _update_loop_endpoint(
        title="更新资金流向数据...",
        output_name="moneyflow.parquet",
        date_col="trade_date",
        dedup_subset=["ts_code", "trade_date"],
        fallback_start=start_date,
        fetch_dates=trade_dates,
        fetch_fn=lambda pro, date_str: pro.moneyflow(trade_date=date_str),
        rewind_days=5,
        sort_by=["trade_date", "ts_code"],
        request_interval=0.18,
        force_full_refresh=force_full_refresh,
    )


def update_margin(
    start_date: str = DEFAULT_SUPPLEMENTAL_START,
    force_full_refresh: bool = False,
):
    """更新融资融券汇总数据"""
    return _update_range_endpoint(
        title="更新融资融券数据...",
        api_method="margin",
        output_name="margin.parquet",
        date_col="trade_date",
        dedup_subset=["trade_date", "exchange_id"],
        fallback_start=start_date,
        rewind_days=5,
        sort_by=["trade_date", "exchange_id"],
        chunk_days=30,
        force_full_refresh=force_full_refresh,
    )


def update_top_inst(
    start_date: str = DEFAULT_SUPPLEMENTAL_START,
    force_full_refresh: bool = False,
):
    """更新龙虎榜席位数据。top_inst 仅支持按交易日抓取。"""
    trade_dates = _get_trade_dates(start_date, _today_str())
    return _update_loop_endpoint(
        title="更新龙虎榜席位数据...",
        output_name="top_inst.parquet",
        date_col="trade_date",
        dedup_subset=["trade_date", "ts_code", "exalter"],
        fallback_start=start_date,
        fetch_dates=trade_dates,
        fetch_fn=lambda pro, date_str: pro.top_inst(trade_date=date_str),
        rewind_days=5,
        sort_by=["trade_date", "ts_code"],
        request_interval=0.18,
        force_full_refresh=force_full_refresh,
    )


def update_report_rc(
    start_date: str = DEFAULT_SUPPLEMENTAL_START,
    force_full_refresh: bool = False,
):
    """更新研报数据"""
    return _update_range_endpoint(
        title="更新研报数据...",
        api_method="report_rc",
        output_name="report_rc.parquet",
        date_col="report_date",
        dedup_subset=["ts_code", "report_date", "report_title"],
        fallback_start=start_date,
        rewind_days=30,
        sort_by=["report_date", "ts_code"],
        chunk_days=365,
        force_full_refresh=force_full_refresh,
    )


def update_stk_holdernumber(
    start_date: str = DEFAULT_SUPPLEMENTAL_START,
    force_full_refresh: bool = False,
):
    """更新股东户数数据"""
    return _update_range_endpoint(
        title="更新股东户数数据...",
        api_method="stk_holdernumber",
        output_name="stk_holdernumber.parquet",
        date_col="end_date",
        dedup_subset=["ts_code", "end_date"],
        fallback_start=start_date,
        rewind_days=90,
        sort_by=["end_date", "ts_code"],
        chunk_days=30,
        force_full_refresh=force_full_refresh,
    )


def update_top10_holders(
    start_date: str = DEFAULT_SUPPLEMENTAL_START,
    force_full_refresh: bool = False,
):
    """更新十大股东数据"""
    return _update_range_endpoint(
        title="更新十大股东数据...",
        api_method="top10_holders",
        output_name="top10_holders.parquet",
        date_col="end_date",
        dedup_subset=["ts_code", "end_date", "holder_name"],
        fallback_start=start_date,
        rewind_days=90,
        sort_by=["end_date", "ts_code"],
        chunk_days=30,
        force_full_refresh=force_full_refresh,
    )


def update_top10_floatholders(
    start_date: str = DEFAULT_SUPPLEMENTAL_START,
    force_full_refresh: bool = False,
):
    """更新十大流通股东数据"""
    return _update_range_endpoint(
        title="更新十大流通股东数据...",
        api_method="top10_floatholders",
        output_name="top10_floatholders.parquet",
        date_col="end_date",
        dedup_subset=["ts_code", "end_date", "holder_name"],
        fallback_start=start_date,
        rewind_days=90,
        sort_by=["end_date", "ts_code"],
        chunk_days=30,
        force_full_refresh=force_full_refresh,
    )


def update_supplemental(
    start_date: str = DEFAULT_SUPPLEMENTAL_START,
    force_full_refresh: bool = False,
    targets: Optional[Sequence[str]] = None,
):
    """更新补充数据：事件、行为、卖方、股东结构。"""
    print("\n" + "=" * 60)
    print("  补充数据更新")
    print(f"  起始日期: {start_date}")
    print("=" * 60)

    target_set = {item.strip() for item in targets} if targets else None
    jobs = [
        ("forecast", "业绩预告", update_forecast),
        ("express", "业绩快报", update_express),
        ("moneyflow", "资金流向", update_moneyflow),
        ("margin", "融资融券", update_margin),
        ("top_inst", "龙虎榜席位", update_top_inst),
        ("report_rc", "研报", update_report_rc),
        ("stk_holdernumber", "股东户数", update_stk_holdernumber),
        ("top10_holders", "十大股东", update_top10_holders),
        ("top10_floatholders", "十大流通股东", update_top10_floatholders),
    ]

    results = []
    for key, name, fn in jobs:
        if target_set and key not in target_set:
            continue
        results.append((name, fn(start_date, force_full_refresh=force_full_refresh)))

    print("\n" + "=" * 60)
    print("补充数据更新结果")
    print("=" * 60)
    success = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {name}: {'✓ 成功' if ok else '✗ 失败'}")
    print(f"\n总计: {success}/{len(results)} 成功")
    return success == len(results)


def get_stock_info(ts_code: str = None) -> pd.DataFrame:
    """获取股票基本信息"""
    path = TUSHARE_DIR / "stock_basic.csv"

    if not path.exists():
        print("股票基本信息不存在，正在更新...")
        update_stock_basic()

    df = pd.read_csv(path)
    if ts_code:
        df = df[df["ts_code"] == ts_code]
    return df


def get_stock_name(ts_code: str) -> str:
    """获取股票名称"""
    df = get_stock_info(ts_code)
    if len(df) == 0:
        return None
    return df.iloc[0]["name"]


def get_stock_industry(ts_code: str) -> str:
    """获取股票所属行业"""
    df = get_stock_info(ts_code)
    if len(df) == 0:
        return None
    return df.iloc[0]["industry"]


def get_name_to_code(name: str) -> str:
    """根据股票名称获取代码"""
    df = get_stock_info()
    matches = df[df["name"] == name]
    if len(matches) == 0:
        matches = df[df["name"].str.contains(name, na=False)]
    if len(matches) == 0:
        return None
    return matches.iloc[0]["ts_code"]


def update_all():
    """更新所有数据"""
    print("\n" + "=" * 60)
    print("  数据更新脚本")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    results = [
        ("股票基本信息", update_stock_basic()),
        ("每日基础数据", update_daily_basic()),
        ("财务指标", update_fina_indicator()),
        ("利润表", update_income()),
        ("现金流量表", update_cashflow()),
        ("资产负债表", update_balancesheet()),
        ("补充数据", update_supplemental(DEFAULT_SUPPLEMENTAL_START)),
    ]

    print("\n" + "=" * 60)
    print("更新结果汇总")
    print("=" * 60)

    success = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {name}: {'✓ 成功' if ok else '✗ 失败'}")
    print(f"\n总计: {success}/{len(results)} 成功")
    return success == len(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据更新脚本")
    parser.add_argument("--all", action="store_true", help="更新所有数据")
    parser.add_argument("--stock-basic", action="store_true", help="更新股票基本信息")
    parser.add_argument("--daily", action="store_true", help="更新每日基础数据")
    parser.add_argument("--financial", action="store_true", help="更新财务数据")
    parser.add_argument("--supplemental", action="store_true", help="更新补充数据")
    parser.add_argument(
        "--supplemental-start",
        default=DEFAULT_SUPPLEMENTAL_START,
        help="补充数据首次回补起始日期，格式 YYYYMMDD",
    )
    parser.add_argument(
        "--supplemental-targets",
        help="仅更新指定补充表，逗号分隔，可选值: "
        "forecast,express,moneyflow,margin,top_inst,report_rc,"
        "stk_holdernumber,top10_holders,top10_floatholders",
    )
    parser.add_argument(
        "--force-full-refresh",
        action="store_true",
        help="忽略已有文件，从起始日期全量重建",
    )
    parser.add_argument("--get-info", type=str, help="查询股票信息")
    parser.add_argument("--get-name", type=str, help="查询股票名称")
    parser.add_argument("--get-industry", type=str, help="查询所属行业")
    parser.add_argument("--name-to-code", type=str, help="名称转代码")

    args = parser.parse_args()

    if args.all:
        update_all()
    elif args.stock_basic:
        update_stock_basic()
    elif args.daily:
        update_daily_basic()
    elif args.financial:
        update_fina_indicator()
        update_income()
        update_cashflow()
        update_balancesheet()
    elif args.supplemental:
        targets = None
        if args.supplemental_targets:
            targets = [item.strip() for item in args.supplemental_targets.split(",") if item.strip()]
        update_supplemental(
            args.supplemental_start,
            force_full_refresh=args.force_full_refresh,
            targets=targets,
        )
    elif args.get_info:
        df = get_stock_info(args.get_info)
        print(df.to_string(index=False))
    elif args.get_name:
        print(get_stock_name(args.get_name))
    elif args.get_industry:
        print(get_stock_industry(args.get_industry))
    elif args.name_to_code:
        print(get_name_to_code(args.name_to_code))
    else:
        parser.print_help()
        print("\n示例:")
        print("  python scripts/data_update.py --all")
        print("  python scripts/data_update.py --supplemental --supplemental-start 20220101")
        print("  python scripts/data_update.py --stock-basic")
        print("  python scripts/data_update.py --get-info 000001.SZ")
