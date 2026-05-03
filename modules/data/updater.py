"""
数据更新模块
每日收盘后自动更新 Tushare 数据 → 重新计算选股
"""

import os
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from config.config import CONFIG
from modules.data.precheck import run_data_precheck
from modules.data.tushare_to_qlib import TushareToQlibConverter

logger = logging.getLogger(__name__)


PROVIDER_PRECHECK_KEYWORD = "Qlib provider 字段不一致"
BOOTSTRAP_MARKET_START = "20160101"


def get_tushare_pro():
    """
    获取 Tushare API 客户端

    Returns
    -------
    pro API 对象，或 None（如果无 token）
    """
    try:
        import tushare as ts

        token = os.environ.get("TUSHARE_TOKEN")
        if token:
            pro = ts.pro_api(token)
        else:
            # 尝试使用已配置的 token
            pro = ts.pro_api()
        if hasattr(pro, "_DataApi__timeout"):
            pro._DataApi__timeout = 30
        return pro
    except Exception as e:
        logger.warning(f"获取 Tushare API 失败: {e}")
        return None


class _RateLimiter:
    """令牌桶限速器：保证不超过 Tushare 的 500 次/分钟限制。"""

    def __init__(self, max_calls: int = 480, period: float = 60.0):
        self._max_calls = max_calls
        self._period = period
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    def wait(self):
        """阻塞直到可以安全发起新请求。"""
        while True:
            with self._lock:
                now = time.monotonic()
                # 清理过期时间戳
                self._timestamps = [t for t in self._timestamps if now - t < self._period]
                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return
                # 计算需要等待多久
                sleep_time = self._timestamps[0] + self._period - now + 0.1
            time.sleep(max(sleep_time, 0.5))


class DataUpdater:
    """数据更新器"""

    API_RETRY_COUNT = 3
    API_RETRY_BASE_SLEEP = 0.8
    MAX_WORKERS = 8  # 并发下载线程数
    _rate_limiter = _RateLimiter(max_calls=490, period=60.0)

    def __init__(self, qlib_data_path: str = None):
        configured_qlib_path = CONFIG.get(
            "paths.data.qlib_data",
            CONFIG.get("qlib_data_path", "~/code/qlib/data/qlib_data/cn_data"),
        )
        self.qlib_data_path = qlib_data_path or configured_qlib_path or "~/code/qlib/data/qlib_data/cn_data"
        self.qlib_data_path = Path(self.qlib_data_path).expanduser()
        self.qlib_data_path.mkdir(parents=True, exist_ok=True)

        # Tushare 数据目录 (与 qlib_data 同级)
        # qlib_data_path = .../data/qlib_data/cn_data → parent.parent = .../data
        data_root = self.qlib_data_path.parent.parent
        self.tushare_dir = data_root / "tushare"
        self.tushare_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir = self.qlib_data_path.parent / "raw_data"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _merge_and_save(output_path: Path, df: pd.DataFrame, subset: List[str]) -> bool:
        """合并增量数据并按主键去重保存。"""
        if df is None:
            return False

        work = df.copy()
        if output_path.exists():
            if output_path.suffix == ".parquet":
                existing = pd.read_parquet(output_path)
            else:
                existing = pd.read_csv(output_path, dtype=str)
            work = pd.concat([existing, work], ignore_index=True)

        if subset:
            work = work.drop_duplicates(subset=subset, keep="last")

        if output_path.suffix == ".parquet":
            work.to_parquet(output_path, index=False)
        else:
            work.to_csv(output_path, index=False)
        return True

    @staticmethod
    def _date_windows(start_date: str, end_date: str, step_days: int):
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        cursor = start
        while cursor <= end:
            window_end = min(cursor + timedelta(days=step_days - 1), end)
            yield cursor.strftime("%Y%m%d"), window_end.strftime("%Y%m%d")
            cursor = window_end + timedelta(days=1)

    @staticmethod
    def _next_calendar_date_str(date_value) -> str:
        """将 YYYYMMDD 推进到下一个自然日。"""
        if pd.isna(date_value):
            raise ValueError("date_value 不能为空")

        date_str = str(date_value)
        if date_str.isdigit():
            date_str = f"{int(date_str):08d}"
        else:
            date_str = pd.Timestamp(date_value).strftime("%Y%m%d")

        return (datetime.strptime(date_str, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")

    @staticmethod
    def _rewind_calendar_date_str(date_value, days: int) -> str:
        """将 YYYYMMDD 回退指定自然日，用于增量回补窗口。"""
        if pd.isna(date_value):
            raise ValueError("date_value 不能为空")
        if days < 0:
            raise ValueError("days 不能为负数")

        date_str = str(date_value)
        if date_str.isdigit():
            date_str = f"{int(date_str):08d}"
        else:
            date_str = pd.Timestamp(date_value).strftime("%Y%m%d")

        return (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=days)).strftime("%Y%m%d")

    @staticmethod
    def _required_index_weight_start() -> str:
        """返回当前策略所需的最早指数成分覆盖起点。"""
        configured_start = CONFIG.get("start_date", BOOTSTRAP_MARKET_START)
        try:
            configured_str = pd.Timestamp(configured_start).strftime("%Y%m%d")
        except Exception:
            return BOOTSTRAP_MARKET_START
        return max(BOOTSTRAP_MARKET_START, configured_str)

    def _call_tushare_api(
        self,
        api_callable,
        call_name: str,
        retry_count: int = None,
        base_sleep: float = None,
        **kwargs,
    ):
        """统一的 Tushare API 调用包装：网络/频控异常时等待重试。"""
        retry_count = retry_count or self.API_RETRY_COUNT
        base_sleep = base_sleep or self.API_RETRY_BASE_SLEEP
        last_exc = None

        for attempt in range(1, retry_count + 1):
            try:
                self._rate_limiter.wait()
                return api_callable(**kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt >= retry_count:
                    break
                wait_seconds = base_sleep * (2 ** attempt)  # 指数退避
                logger.warning(
                    "%s 失败，第 %s/%s 次重试前等待 %.1fs: %s",
                    call_name,
                    attempt,
                    retry_count,
                    wait_seconds,
                    exc,
                )
                time.sleep(wait_seconds)

        raise last_exc

    def _needs_bootstrap(self) -> bool:
        """判断当前环境是否缺少首次构建所需的核心文件。"""
        required_paths = [
            self.qlib_data_path / "calendars" / "day.txt",
            self.qlib_data_path / "instruments" / "all.txt",
            self.qlib_data_path / "factor_data.parquet",
            self.tushare_dir / "daily_basic.parquet",
            self.tushare_dir / "adj_factor.parquet",
        ]
        if any(not path.exists() for path in required_paths):
            return True
        return not any(self.raw_data_dir.glob("*.parquet"))

    def _ensure_provider_structure(self) -> int:
        """根据 raw_data 和日历补齐 Qlib provider 目录结构。"""
        cal_file = self.qlib_data_path / "calendars" / "day.txt"
        if not cal_file.exists():
            return 0

        features_dir = self.qlib_data_path / "features"
        instruments_dir = self.qlib_data_path / "instruments"
        features_dir.mkdir(parents=True, exist_ok=True)
        instruments_dir.mkdir(parents=True, exist_ok=True)

        raw_files = sorted(self.raw_data_dir.glob("*.parquet"))
        if not raw_files:
            return 0

        cal_lines = [line.strip() for line in cal_file.read_text().splitlines() if line.strip()]
        if not cal_lines:
            return 0
        cal_start = pd.Timestamp(cal_lines[0])
        cal_end = pd.Timestamp(cal_lines[-1])

        instruments = []
        for raw_path in raw_files:
            inst = raw_path.stem
            (features_dir / inst).mkdir(exist_ok=True)
            try:
                raw_df = pd.read_parquet(raw_path, columns=["date"])
            except Exception as exc:
                logger.warning("读取 %s 失败，跳过 instrument 区间同步: %s", raw_path, exc)
                continue

            raw_dates = pd.to_datetime(raw_df["date"], errors="coerce").dropna()
            if raw_dates.empty:
                logger.warning("%s 缺少有效 date，跳过 instrument 区间同步", raw_path)
                continue

            raw_start = max(raw_dates.min().normalize(), cal_start)
            raw_end = min(raw_dates.max().normalize(), cal_end)
            if raw_start > raw_end:
                logger.warning(
                    "%s 的 raw_data 日期超出当前 calendar 范围，跳过 instrument 区间同步",
                    raw_path,
                )
                continue

            instruments.append((inst, raw_start.strftime("%Y-%m-%d"), raw_end.strftime("%Y-%m-%d")))

        with open(instruments_dir / "all.txt", "w") as fp:
            for inst, inst_start, inst_end in sorted(instruments):
                fp.write(f"{inst}\t{inst_start}\t{inst_end}\n")

        return len(instruments)

    def get_last_trading_date(self) -> datetime:
        """获取本地最新交易日"""
        cal_file = self.qlib_data_path / "calendars" / "day.txt"
        if cal_file.exists():
            cal = pd.read_csv(cal_file, header=None, names=["date"], parse_dates=["date"])
            return cal["date"].iloc[-1]

        # 默认昨天
        yesterday = datetime.now() - timedelta(days=1)
        if yesterday.weekday() == 5:  # 周六
            yesterday -= timedelta(days=1)
        elif yesterday.weekday() == 6:  # 周日
            yesterday -= timedelta(days=2)
        return yesterday

    def get_remote_latest_date(self) -> Optional[datetime]:
        """
        获取远程（Tushare）最新交易日

        Returns
        -------
        datetime 或 None（如果 API 不可用）
        """
        pro = get_tushare_pro()
        if pro is None:
            return None

        try:
            # 获取最近一周的交易日历
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")

            df = self._call_tushare_api(
                pro.trade_cal,
                "trade_cal",
                start_date=start_date,
                end_date=end_date,
                is_open="1",  # 只要开市日
            )

            if df is not None and len(df) > 0:
                latest = df["cal_date"].max()
                return datetime.strptime(latest, "%Y%m%d")

        except Exception as e:
            logger.error(f"获取远程交易日历失败: {e}")

        return None

    def check_update_needed(self) -> bool:
        """检查是否需要更新"""
        local_date = self.get_last_trading_date()
        remote_date = self.get_remote_latest_date()

        if remote_date is None:
            # 无法获取远程日期，检查本地是否太旧
            days_since_update = (datetime.now().date() - local_date.date()).days
            return days_since_update > 3  # 超过3天没更新

        return remote_date.date() > local_date.date()

    def _get_index_trade_dates(self) -> Optional[pd.Series]:
        """从 index_daily.parquet 读取完整交易日列表（000300.SH 沪深300）。"""
        index_path = self.tushare_dir / "index_daily.parquet"
        if not index_path.exists():
            return None
        idx = pd.read_parquet(index_path, columns=["ts_code", "trade_date"])
        idx = idx[idx["ts_code"] == "000300.SH"]
        dates = pd.to_datetime(idx["trade_date"], format="%Y%m%d", errors="coerce").dropna()
        return dates.sort_values().drop_duplicates()

    def download_daily_basic(self, start_date: str = None) -> bool:
        """
        下载每日基础数据（增量，按交易日逐日拉取）

        Returns
        -------
        bool 是否成功
        """
        pro = get_tushare_pro()
        if pro is None:
            return False

        output_path = self.tushare_dir / "daily_basic.parquet"

        try:
            existing = None
            # 确定起始日期
            if output_path.exists():
                existing = pd.read_parquet(output_path, columns=["trade_date"])
                if start_date is None:
                    max_date = str(existing["trade_date"].max())
                    start_date = self._next_calendar_date_str(max_date)
            if start_date is None:
                start_date = BOOTSTRAP_MARKET_START

            # 从 index_daily 获取完整交易日列表
            trade_dates = self._get_index_trade_dates()
            if trade_dates is None or trade_dates.empty:
                logger.warning("缺少 index_daily.parquet，无法获取交易日列表")
                return False

            start_ts = pd.Timestamp(start_date)
            target_dates = trade_dates[trade_dates >= start_ts]
            if target_dates.empty:
                logger.info("daily_basic 已是最新")
                return True

            logger.info(f"daily_basic: 需下载 {len(target_dates)} 个交易日 (from {start_date})")

            all_data = []
            lock = threading.Lock()
            counter = [0]  # mutable counter for progress

            def _fetch_one(date_str):
                try:
                    df = self._call_tushare_api(
                        pro.daily_basic,
                        f"daily_basic {date_str}",
                        trade_date=date_str,
                    )
                    with lock:
                        counter[0] += 1
                        if counter[0] % 100 == 0:
                            logger.info(f"daily_basic 已拉取 {counter[0]}/{len(target_dates)} 个交易日")
                    return df if df is not None and len(df) > 0 else None
                except Exception as exc:
                    with lock:
                        counter[0] += 1
                    logger.warning(f"daily_basic {date_str} 失败: {exc}")
                    return None

            date_strs = [d.strftime("%Y%m%d") for d in target_dates]
            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                futures = {executor.submit(_fetch_one, ds): ds for ds in date_strs}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_data.append(result)

            if not all_data:
                logger.info("daily_basic 无新数据")
                return output_path.exists()

            result = pd.concat(all_data, ignore_index=True)
            self._merge_and_save(output_path, result, ["ts_code", "trade_date"])
            logger.info(f"已更新 daily_basic: {len(result)} 条")
            return True

        except Exception as e:
            logger.error(f"下载 daily_basic 失败: {e}")
            return False

    def download_financial_data(self) -> bool:
        """
        下载财务数据（fina_indicator, income, cashflow, balancesheet）

        Returns
        -------
        bool 是否成功
        """
        pro = get_tushare_pro()
        if pro is None:
            logger.warning("Tushare API 不可用，跳过财务数据下载")
            return False

        results = []

        # 定义要下载的数据集
        datasets = [
            ("fina_indicator", "fina_indicator.parquet", "end_date"),
            ("income", "income.parquet", "end_date"),
            ("cashflow", "cashflow.parquet", "end_date"),
            ("balancesheet", "balancesheet.parquet", "end_date"),
        ]

        # 先检查哪些数据集需要更新
        need_download = []
        threshold = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        for api_name, filename, date_col in datasets:
            output_path = self.tushare_dir / filename
            if output_path.exists():
                try:
                    existing = pd.read_parquet(output_path, columns=[date_col])
                    max_date = str(existing[date_col].max())
                    if max_date >= threshold:
                        logger.info(f"{api_name} 已是最新 (max_date={max_date})")
                        results.append(True)
                        continue
                except Exception:
                    pass
            need_download.append((api_name, filename, date_col))

        if not need_download:
            return True

        # 获取股票列表（只有确实需要下载时才调用 API）
        try:
            stock_df = self._call_tushare_api(
                pro.stock_basic,
                "stock_basic:list_status=L",
                list_status="L",
                fields="ts_code",
            )
            ts_codes = stock_df["ts_code"].tolist()
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return False

        for api_name, filename, date_col in need_download:
            output_path = self.tushare_dir / filename
            success = self._download_financial_dataset(
                pro, api_name, output_path, ts_codes, date_col
            )
            results.append(success)

        return all(results)

    def _download_financial_dataset(
        self, pro, api_name: str, output_path: Path, ts_codes: list, date_col: str
    ) -> bool:
        """下载单个财务数据集（逐只股票调用，多线程并发）"""
        try:
            # 确定起始日期
            if output_path.exists():
                existing = pd.read_parquet(output_path)
                max_date = existing[date_col].max()
                start_date = str(int(max_date) - 10000)  # 往前推确保完整
            else:
                start_date = "20100101"

            api_func = getattr(pro, api_name)
            all_data = []
            failed = [0]
            lock = threading.Lock()
            counter = [0]

            logger.info(f"{api_name}: 开始下载 {len(ts_codes)} 只股票 (start={start_date}, workers={self.MAX_WORKERS})")

            def _fetch_one(ts_code):
                try:
                    # fina_indicator 的 start_date 过滤 end_date（报告期），
                    # 传了反而限制返回范围，导致只拿到最近几个季度
                    if api_name == "fina_indicator":
                        kwargs = {"ts_code": ts_code}
                    else:
                        kwargs = {"ts_code": ts_code, "start_date": start_date}
                    df = self._call_tushare_api(
                        api_func,
                        f"{api_name} {ts_code}",
                        **kwargs,
                    )
                    with lock:
                        counter[0] += 1
                        if counter[0] % 500 == 0:
                            logger.info(f"{api_name} 已拉取 {counter[0]}/{len(ts_codes)} 只 (失败 {failed[0]})")
                    return df if df is not None and len(df) > 0 else None
                except Exception:
                    with lock:
                        counter[0] += 1
                        failed[0] += 1
                    return None

            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                futures = {executor.submit(_fetch_one, code): code for code in ts_codes}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_data.append(result)

            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                if output_path.exists():
                    combined = pd.concat([existing, df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["ts_code", date_col], keep="last")
                    combined.to_parquet(output_path, index=False)
                else:
                    df.to_parquet(output_path, index=False)

                logger.info(f"已更新 {api_name}: {len(df)} 条 (失败 {failed[0]} 只)")
            else:
                logger.info(f"{api_name} 无新数据 (失败 {failed[0]} 只)")

            return True

        except Exception as e:
            logger.error(f"下载 {api_name} 失败: {e}")
            return False

    def download_stock_basic(self) -> bool:
        """下载股票基本信息"""
        pro = get_tushare_pro()
        if pro is None:
            return False

        try:
            df = self._call_tushare_api(
                pro.stock_basic,
                "stock_basic:industry",
                exchange="",
                list_status="L",
                fields="ts_code,name,industry",
            )
            output_path = self.tushare_dir / "stock_basic.csv"
            df.to_csv(output_path, index=False)

            # 同时保存行业数据
            industry_path = self.tushare_dir / "stock_industry.csv"
            df[["ts_code", "industry"]].to_csv(industry_path, index=False)

            return True

        except Exception as e:
            logger.error(f"下载股票基本信息失败: {e}")
            return False

    def download_index_daily(self) -> bool:
        """下载主要指数日线数据。"""
        pro = get_tushare_pro()
        if pro is None:
            return False

        output_path = self.tushare_dir / "index_daily.parquet"
        indices = ["000300.SH"]

        try:
            if output_path.exists():
                existing = pd.read_parquet(output_path, columns=["trade_date"])
                max_date = str(existing["trade_date"].max())
                start_date = self._next_calendar_date_str(max_date)
            else:
                start_date = "20160101"

            end_date = datetime.now().strftime("%Y%m%d")
            all_data = []
            for index_code in indices:
                df = self._call_tushare_api(
                    pro.index_daily,
                    f"index_daily {index_code}",
                    ts_code=index_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields="ts_code,trade_date,open,high,low,close,vol,amount,pct_chg",
                )
                if df is not None and len(df) > 0:
                    all_data.append(df)

            if not all_data:
                logger.info("index_daily 无新数据")
                return True

            result = pd.concat(all_data, ignore_index=True)
            self._merge_and_save(output_path, result, ["ts_code", "trade_date"])
            logger.info(f"已更新 index_daily: {len(result)} 条")
            return True
        except Exception as e:
            logger.error(f"下载 index_daily 失败: {e}")
            return False

    def download_index_weight(self) -> bool:
        """下载历史指数成分股权重，供 csi300 历史股票池使用。"""
        pro = get_tushare_pro()
        if pro is None:
            return False

        output_path = self.tushare_dir / "index_weight.parquet"
        indices = ["000300.SH"]
        end_date = datetime.now().strftime("%Y%m%d")
        required_start = self._required_index_weight_start()

        try:
            fetch_ranges = []
            if output_path.exists():
                existing = pd.read_parquet(output_path, columns=["trade_date"])
                existing_dates = pd.to_datetime(
                    existing["trade_date"].astype(str),
                    format="%Y%m%d",
                    errors="coerce",
                ).dropna()
                if existing_dates.empty:
                    fetch_ranges.append((required_start, end_date))
                else:
                    min_date = existing_dates.min()
                    max_date = existing_dates.max()
                    required_start_ts = pd.Timestamp(required_start)
                    if min_date > required_start_ts:
                        head_end = (min_date - pd.Timedelta(days=1)).strftime("%Y%m%d")
                        fetch_ranges.append((required_start, head_end))

                    # 指数成分按月调整，尾部回拉约 3 个月以覆盖修订和重复月度快照
                    tail_start = max(
                        required_start,
                        (max_date - pd.Timedelta(days=93)).strftime("%Y%m%d"),
                    )
                    fetch_ranges.append((tail_start, end_date))
            else:
                fetch_ranges.append((required_start, end_date))

            all_data = []
            for index_code in indices:
                for range_start, range_end in fetch_ranges:
                    if range_start > range_end:
                        continue
                    for win_start, win_end in self._date_windows(range_start, range_end, step_days=366):
                        try:
                            df = self._call_tushare_api(
                                pro.index_weight,
                                f"index_weight {index_code} {win_start}-{win_end}",
                                index_code=index_code,
                                start_date=win_start,
                                end_date=win_end,
                            )
                            if df is not None and len(df) > 0:
                                all_data.append(df)
                        except Exception as exc:
                            logger.warning(
                                f"index_weight {index_code} {win_start}-{win_end} 失败: {exc}"
                            )
                        time.sleep(0.05)

            if not all_data:
                logger.info("index_weight 无新数据")
                return output_path.exists()

            result = pd.concat(all_data, ignore_index=True)
            self._merge_and_save(output_path, result, ["index_code", "con_code", "trade_date"])
            logger.info(f"已更新 index_weight: {len(result)} 条")
            return True
        except Exception as e:
            logger.error(f"下载 index_weight 失败: {e}")
            return False

    def download_namechange(self) -> bool:
        """下载历史名称变更数据，供历史 ST 过滤使用。"""
        pro = get_tushare_pro()
        if pro is None:
            return False

        output_path = self.tushare_dir / "namechange.parquet"
        end_date = datetime.now().strftime("%Y%m%d")

        try:
            if output_path.exists():
                existing = pd.read_parquet(output_path, columns=["start_date"])
                max_date = existing["start_date"].dropna()
                if not max_date.empty:
                    start_date = (
                        pd.Timestamp(str(max_date.max())) - pd.Timedelta(days=365)
                    ).strftime("%Y%m%d")
                else:
                    start_date = "20100101"
            else:
                start_date = "20100101"

            all_data = []
            for win_start, win_end in self._date_windows(start_date, end_date, step_days=365):
                try:
                    df = self._call_tushare_api(
                        pro.namechange,
                        f"namechange {win_start}-{win_end}",
                        start_date=win_start,
                        end_date=win_end,
                    )
                    if df is not None and len(df) > 0:
                        all_data.append(df)
                except Exception as exc:
                    logger.warning(f"namechange {win_start}-{win_end} 失败: {exc}")
                time.sleep(0.08)

            if not all_data:
                logger.info("namechange 无新数据")
                return output_path.exists()

            result = pd.concat(all_data, ignore_index=True)
            key_cols = [
                col
                for col in ["ts_code", "name", "start_date", "end_date", "ann_date"]
                if col in result.columns
            ]
            self._merge_and_save(output_path, result, key_cols)
            logger.info(f"已更新 namechange: {len(result)} 条")
            return True
        except Exception as e:
            logger.error(f"下载 namechange 失败: {e}")
            return False

    def download_adj_factor(self) -> bool:
        """增量下载复权因子 adj_factor（按交易日逐日拉取）"""
        pro = get_tushare_pro()
        if pro is None:
            return False

        output_path = self.tushare_dir / "adj_factor.parquet"

        try:
            # 确定起始日期：回补最近一段，避免尾部缺口永久保留
            if output_path.exists():
                existing = pd.read_parquet(output_path, columns=["trade_date"])
                max_date = str(existing["trade_date"].max())
                start_date = max("20160101", self._rewind_calendar_date_str(max_date, days=31))
            else:
                start_date = "20160101"

            # 从 index_daily 获取完整交易日列表
            trade_dates = self._get_index_trade_dates()
            if trade_dates is None or trade_dates.empty:
                logger.warning("缺少 index_daily.parquet，无法获取交易日列表")
                return False

            start_ts = pd.Timestamp(start_date)
            end_ts = trade_dates.max()
            target_dates = trade_dates[(trade_dates >= start_ts) & (trade_dates <= end_ts)]
            if target_dates.empty:
                logger.info("adj_factor 已是最新")
                return True

            logger.info(f"adj_factor: 需下载 {len(target_dates)} 个交易日 (from {start_date})")

            all_data = []
            lock = threading.Lock()
            counter = [0]

            def _fetch_one(date_str):
                try:
                    df = self._call_tushare_api(
                        pro.adj_factor,
                        f"adj_factor {date_str}",
                        trade_date=date_str,
                    )
                    with lock:
                        counter[0] += 1
                        if counter[0] % 100 == 0:
                            logger.info(f"adj_factor 已拉取 {counter[0]}/{len(target_dates)} 个交易日")
                    return df if df is not None and len(df) > 0 else None
                except Exception as exc:
                    with lock:
                        counter[0] += 1
                    logger.warning(f"adj_factor {date_str} 失败: {exc}")
                    return None

            date_strs = [d.strftime("%Y%m%d") for d in target_dates]
            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                futures = {executor.submit(_fetch_one, ds): ds for ds in date_strs}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_data.append(result)

            if not all_data:
                logger.info("adj_factor 无新数据")
                return output_path.exists()

            result = pd.concat(all_data, ignore_index=True)
            self._merge_and_save(output_path, result, ["ts_code", "trade_date"])
            logger.info(f"已更新 adj_factor: {len(result)} 条")
            return True
        except Exception as e:
            logger.error(f"下载 adj_factor 失败: {e}")
            return False

    def _get_last_bin_date(self) -> Optional[datetime]:
        """返回 close.day.bin 中最旧的最后日期（采样检测，判断是否有股票需要更新）"""
        cal_file = self.qlib_data_path / "calendars" / "day.txt"
        features_dir = self.qlib_data_path / "features"
        if not cal_file.exists() or not features_dir.exists():
            return None
        cal = pd.read_csv(cal_file, header=None, names=["date"], parse_dates=["date"])
        cal_len = len(cal)
        # 扫描全部股票，找最旧的 last_bin_date
        # 仅统计数据点数量 ≥ 250（约1年）的股票，排除极短历史的新股
        min_last_date = None
        for inst_dir in features_dir.iterdir():
            bin_f = inst_dir / "close.day.bin"
            if not bin_f.exists():
                continue
            raw = np.fromfile(bin_f, dtype="<f4")
            if len(raw) < 252 or np.isnan(raw[0]):  # 至少一年数据
                continue
            last_idx = int(raw[0]) + len(raw) - 2
            last_idx = min(last_idx, cal_len - 1)
            last_date = cal.iloc[last_idx]["date"].to_pydatetime()
            if min_last_date is None or last_date < min_last_date:
                min_last_date = last_date
        return min_last_date

    def update_price_bins(self) -> bool:
        """从 Tushare daily API 补全 qlib 价格二进制文件

        优先使用 adj_factor 前复权。如果 adj_factor 不可用，回退到 splice-point ratio。
        """
        cal_file = self.qlib_data_path / "calendars" / "day.txt"
        features_dir = self.qlib_data_path / "features"
        if not cal_file.exists() or not features_dir.exists():
            return False

        cal = pd.read_csv(cal_file, header=None, names=["date"], parse_dates=["date"])
        cal_dates = cal["date"].dt.normalize()
        cal_last_idx_global = len(cal) - 1
        idx_to_date = {i: d for i, d in enumerate(cal_dates)}

        # 扫描全部股票，找到最小的 last_bin_idx（排除极短历史新股）
        min_last_bin_idx = cal_last_idx_global
        for inst_dir in features_dir.iterdir():
            bin_f = inst_dir / "close.day.bin"
            if not bin_f.exists():
                continue
            raw0 = np.fromfile(bin_f, dtype="<f4")
            if len(raw0) < 252 or np.isnan(raw0[0]):
                continue
            idx = int(raw0[0]) + len(raw0) - 2
            if idx < min_last_bin_idx:
                min_last_bin_idx = idx

        missing_idxs = list(range(min_last_bin_idx + 1, len(cal)))
        if not missing_idxs:
            logger.info("价格 bin 已是最新")
            return True

        missing_dates = [cal.iloc[i]["date"] for i in missing_idxs]
        logger.info(f"需补全 {len(missing_dates)} 个交易日的价格 bin 数据")

        pro = get_tushare_pro()
        if pro is None:
            logger.warning("Tushare API 不可用，跳过价格 bin 更新")
            return False

        # 尝试加载 adj_factor
        adj_path = self.tushare_dir / "adj_factor.parquet"
        adj_map = {}  # {instrument: {cal_idx: adj_ratio}}
        if adj_path.exists():
            try:
                adj_df = pd.read_parquet(adj_path, columns=["ts_code", "trade_date", "adj_factor"])
                adj_df["date"] = pd.to_datetime(adj_df["trade_date"], format="%Y%m%d").dt.normalize()
                adj_df = adj_df[adj_df["date"].isin(set(cal_dates))]
                adj_df["cal_idx"] = adj_df["date"].map({d: i for i, d in enumerate(cal_dates)})
                adj_df["instrument"] = adj_df["ts_code"].apply(
                    lambda x: x.split(".")[1].lower() + x.split(".")[0] if "." in x else x.lower()
                )
                for inst, grp in adj_df.groupby("instrument"):
                    grp = grp.sort_values("cal_idx")
                    latest_adj = grp["adj_factor"].iloc[-1]
                    if latest_adj > 0:
                        adj_map[inst] = dict(zip(grp["cal_idx"], grp["adj_factor"] / latest_adj))
                logger.info(f"加载 adj_factor: {len(adj_map)} 只股票用于价格 bin 更新")
            except Exception as e:
                logger.warning(f"加载 adj_factor 失败，回退到 splice-point: {e}")
                adj_map = {}

        # 按日期批量拉取（每次一个交易日，覆盖全市场）
        daily_map: dict = {}  # instrument -> {cal_idx: {field: value}}
        for i, (cal_idx, date) in enumerate(zip(missing_idxs, missing_dates)):
            date_str = date.strftime("%Y%m%d")
            try:
                df = self._call_tushare_api(
                    pro.daily,
                    f"daily trade_date={date_str}",
                    trade_date=date_str,
                )
            except Exception as e:
                logger.warning(f"daily {date_str} 失败: {e}")
                time.sleep(1)
                continue
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                ts = str(row["ts_code"])
                if "." not in ts:
                    continue
                code, exch = ts.split(".")
                inst = exch.lower() + code
                if inst not in daily_map:
                    daily_map[inst] = {}
                daily_map[inst][cal_idx] = {
                    "close": float(row.get("close", np.nan) or np.nan),
                    "open": float(row.get("open", np.nan) or np.nan),
                    "high": float(row.get("high", np.nan) or np.nan),
                    "low": float(row.get("low", np.nan) or np.nan),
                    "volume": float(row.get("vol", np.nan) or np.nan),
                    "amount": float(row.get("amount", np.nan) or np.nan),
                    "pre_close": float(row.get("pre_close", np.nan) or np.nan),
                }
            if (i + 1) % 20 == 0:
                logger.info(f"  已拉取 {i + 1}/{len(missing_dates)} 天")
            time.sleep(0.12)

        if not daily_map:
            logger.warning("未获取到任何价格数据")
            return False

        fields_price = ["close", "open", "high", "low"]
        fields_vol = ["volume", "amount"]
        cal_last_idx = len(cal) - 1

        updated = 0
        for inst, date_data in daily_map.items():
            inst_dir = features_dir / inst
            if not inst_dir.exists():
                continue
            close_bin = inst_dir / "close.day.bin"
            if not close_bin.exists():
                continue

            raw_c = np.fromfile(close_bin, dtype="<f4")
            if len(raw_c) < 2 or np.isnan(raw_c[0]):
                continue

            stock_end_idx = int(raw_c[0]) + len(raw_c) - 2
            if stock_end_idx >= cal_last_idx:
                continue

            stock_missing_idxs = list(range(stock_end_idx + 1, cal_last_idx + 1))
            inst_adj = adj_map.get(inst, {})
            use_adj = bool(inst_adj)

            # 计算复权比例：优先 adj_factor，否则 splice-point
            if use_adj:
                # adj_factor 模式：无需 splice-point，直接用 ratio
                pass
            else:
                # splice-point fallback
                bin_last_close = float(raw_c[-1])
                first_new_cal = stock_missing_idxs[0]
                first_pre = date_data.get(first_new_cal, {}).get("pre_close", np.nan)
                if first_pre and not np.isnan(first_pre) and first_pre > 0:
                    splice_adj = bin_last_close / first_pre
                else:
                    splice_adj = 1.0

            try:
                for field in fields_price + fields_vol:
                    bin_file = inst_dir / f"{field}.day.bin"
                    if not bin_file.exists():
                        continue
                    raw = np.fromfile(bin_file, dtype="<f4")
                    if len(raw) < 2 or np.isnan(raw[0]):
                        continue
                    f_end_idx = int(raw[0]) + len(raw) - 2
                    f_missing = list(range(f_end_idx + 1, cal_last_idx + 1))
                    if not f_missing:
                        continue

                    new_vals = []
                    for cal_idx in f_missing:
                        v = date_data.get(cal_idx, {}).get(field, np.nan)
                        if not np.isnan(v) and field in fields_price:
                            if use_adj:
                                ratio = inst_adj.get(cal_idx)
                                if ratio is not None:
                                    v = v * float(ratio)
                            else:
                                if v > 0:
                                    v = v * splice_adj
                        new_vals.append(v)

                    new_arr = np.array(new_vals, dtype="<f4")
                    gap = f_missing[0] - f_end_idx - 1
                    with open(bin_file, "ab") as fp:
                        if gap > 0:
                            np.full(gap, np.nan, dtype="<f4").tofile(fp)
                        new_arr.tofile(fp)
            except Exception as e:
                logger.warning(f"更新 {inst} 失败: {e}")
                continue

            updated += 1

        logger.info(f"已更新 {updated} 只股票的价格 bin 数据 ({'adj_factor' if adj_map else 'splice-point'})")
        return updated > 0

    def update_raw_data_quotes(self, start_date: str = None, end_date: str = None, max_stocks: int = None) -> bool:
        """按股票维度增量下载 raw_data 行情，支持断点续传。"""
        import json as _json

        pro = get_tushare_pro()
        if pro is None:
            logger.warning("Tushare API 不可用，跳过 raw_data 更新")
            return False

        # ── 1. 准备阶段：交易日历 ──
        try:
            cal_df = self._call_tushare_api(
                pro.trade_cal,
                "raw_data trade_cal",
                exchange="SSE",
                is_open="1",
                fields="cal_date",
            )
            trade_cal = sorted(cal_df["cal_date"].tolist())
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return False

        if not trade_cal:
            logger.warning("交易日历为空，跳过 raw_data 更新")
            return False

        end_date = end_date or trade_cal[-1]
        # 确保 end_date 不超过已开市交易日
        if end_date > trade_cal[-1]:
            end_date = trade_cal[-1]

        full_start = start_date or BOOTSTRAP_MARKET_START

        def _next_trade_date(date_str: str) -> str:
            """返回 date_str 之后的下一个交易日。"""
            for d in trade_cal:
                if d > date_str:
                    return d
            return date_str  # 没有更晚的交易日，返回自身

        # ── 2. 获取股票列表 ──
        stock_basic_path = self.tushare_dir / "stock_basic.csv"
        if stock_basic_path.exists():
            stocks_df = pd.read_csv(stock_basic_path, dtype=str)
        else:
            logger.warning("缺少 stock_basic.csv，无法获取股票列表")
            return False

        if "ts_code" not in stocks_df.columns:
            logger.warning("stock_basic.csv 缺少 ts_code 列")
            return False

        # list_date 列不一定存在，用 BOOTSTRAP_MARKET_START 兜底
        stock_list = []
        for _, row in stocks_df.iterrows():
            ts_code = row["ts_code"]
            list_date = str(row.get("list_date", "")).strip()
            if not list_date or list_date == "nan" or not list_date.isdigit():
                list_date = BOOTSTRAP_MARKET_START
            stock_list.append((ts_code, list_date))

        if max_stocks is not None:
            stock_list = stock_list[:max_stocks]

        # ── 3. 加载状态文件 ──
        state_path = self.raw_data_dir / ".download_state.json"
        state_dict = {}
        if state_path.exists():
            try:
                with open(state_path, "r") as f:
                    state_dict = _json.load(f)
            except Exception:
                logger.warning("状态文件损坏，将重新下载所有股票")
                state_dict = {}

        # ── 4. 过滤阶段：确定需下载的股票 ──
        download_tasks = []
        skipped = 0
        for ts_code, list_date in stock_list:
            start = max(full_start, list_date)
            if start > end_date:
                skipped += 1
                continue

            cached = state_dict.get(ts_code)
            if cached and cached >= end_date:
                skipped += 1
                continue

            if cached and cached >= start:
                start = _next_trade_date(cached)

            download_tasks.append((ts_code, start, end_date))

        logger.info(
            f"raw_data 股票总数={len(stock_list)}, 需下载={len(download_tasks)}, "
            f"已跳过={skipped}, 时间范围={full_start}~{end_date}"
        )

        if not download_tasks:
            logger.info("raw_data 所有股票已是最新")
            return True

        # ── 5. 并发下载 ──
        failures = []
        completed = [0]
        completed_lock = threading.Lock()

        def _atomic_write_json(path: Path, data):
            """原子写 JSON 文件。"""
            tmp_path = path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                _json.dump(data, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    self._download_one_stock, pro, ts_code, start, end_date
                ): ts_code
                for ts_code, start, end_date in download_tasks
            }

            for future in as_completed(futures):
                ts_code = futures[future]
                try:
                    ok, covered_to, error_msg = future.result()
                except Exception as exc:
                    ok, covered_to, error_msg = False, None, str(exc)

                if ok:
                    if covered_to:
                        state_dict[ts_code] = covered_to
                else:
                    task_info = next(
                        (t for t in download_tasks if t[0] == ts_code), None
                    )
                    failures.append({
                        "ts_code": ts_code,
                        "start": task_info[1] if task_info else "unknown",
                        "error": error_msg or "unknown",
                    })

                with completed_lock:
                    completed[0] += 1
                    n = completed[0]
                    # 每 100 只或最后一只：原子写状态文件
                    if n % 100 == 0 or n == len(download_tasks):
                        _atomic_write_json(state_path, state_dict)
                    if n % 50 == 0 or n == len(download_tasks):
                        logger.info(
                            f"raw_data 进度: {n}/{len(download_tasks)}, "
                            f"失败={len(failures)}"
                        )

        # ── 6. 最终写入状态文件和失败文件 ──
        _atomic_write_json(state_path, state_dict)

        if failures:
            fail_path = self.raw_data_dir / ".download_failures.json"
            _atomic_write_json(fail_path, failures)
            logger.warning(f"raw_data 下载失败 {len(failures)} 只股票，已记录到 {fail_path}")

        logger.info(
            f"raw_data 更新完成: 成功={len(download_tasks) - len(failures)}, "
            f"失败={len(failures)}"
        )
        return (len(download_tasks) - len(failures)) > 0

    def _download_one_stock(self, pro, ts_code: str, start_date: str, end_date: str):
        """下载单只股票行情 → 校验 → 合并去重 → 原子写盘。

        Returns: (ok: bool, covered_to: str|None, error: str|None)
        """
        # a. 下载
        try:
            df = self._call_tushare_api(
                pro.daily,
                f"raw_data daily {ts_code}",
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields="ts_code,trade_date,open,high,low,close,vol,amount",
            )
        except Exception as e:
            return False, None, str(e)

        # df.empty → 不算失败，记录 covered_to 避免停牌股票重复下载
        if df is None or df.empty:
            return True, end_date, None

        # b. 校验
        required = {"ts_code", "trade_date", "open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            return False, None, f"缺少必要列: {missing}"

        if df["ts_code"].isna().any() or df["trade_date"].isna().any():
            return False, None, "ts_code 或 trade_date 含空值"

        if df.duplicated(subset=["ts_code", "trade_date"]).any():
            return False, None, "ts_code + trade_date 存在重复"

        numeric_cols = ["open", "high", "low", "close"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        bad_h = df["high"] < df[["open", "close", "low"]].max(axis=1)
        bad_l = df["low"] > df[["open", "close", "high"]].min(axis=1)
        if bad_h.any():
            return False, None, f"high < max(open,close,low) 共 {bad_h.sum()} 行"
        if bad_l.any():
            return False, None, f"low > min(open,close,high) 共 {bad_l.sum()} 行"

        # c. 格式转换
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        df["symbol"] = df["ts_code"]
        df = df.rename(columns={"vol": "volume"})
        df = df[["date", "open", "high", "low", "close", "volume", "amount", "symbol"]]
        df = df.dropna(subset=["date"])

        # d. 读旧文件并合并
        suffix = ts_code.split(".")[1].lower()
        prefix = ts_code.split(".")[0]
        raw_file = f"{suffix}{prefix}.parquet"
        path = self.raw_data_dir / raw_file

        if path.exists():
            try:
                existing = pd.read_parquet(path)
            except Exception as e:
                return False, None, f"读取旧文件失败: {e}"
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["date"], keep="last")

        df = df.sort_values("date")

        # e. 原子写盘：tmp → fsync → replace
        tmp_path = path.with_suffix(".tmp")
        try:
            df.to_parquet(tmp_path, index=False)
            with open(tmp_path, "r+b") as f:
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception as e:
            # 清理临时文件
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return False, None, f"写文件失败: {e}"
        finally:
            del df

        # f. 返回成功
        return True, end_date, None

    def bootstrap_raw_data_for_instruments(
        self,
        instruments: List[str],
        start_date: str = "20160101",
        end_date: str = None,
    ) -> bool:
        """为缺失的少量标的补齐历史 raw_data 文件。"""
        if not instruments:
            return True

        pro = get_tushare_pro()
        if pro is None:
            logger.warning("Tushare API 不可用，跳过 raw_data 历史补档")
            return False

        end_date = end_date or datetime.now().strftime("%Y%m%d")
        updated = 0
        for instrument in sorted(set(instruments)):
            path = self.raw_data_dir / f"{instrument[:2].lower()}{instrument[2:]}.parquet"
            if path.exists():
                continue

            ts_code = f"{instrument[2:]}.{instrument[:2].upper()}"
            try:
                df = self._call_tushare_api(
                    pro.daily,
                    f"bootstrap raw_data {instrument}",
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields="ts_code,trade_date,open,high,low,close,vol,amount",
                )
            except Exception as e:
                logger.warning(f"补档 {instrument} 失败: {e}")
                time.sleep(1)
                continue

            if df is None or df.empty:
                logger.warning(f"补档 {instrument} 未返回数据")
                continue

            df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
            df["symbol"] = df["ts_code"]
            df = df.rename(columns={"vol": "volume"})
            df = df[["date", "open", "high", "low", "close", "volume", "amount", "symbol"]]
            df = df.dropna(subset=["date"]).sort_values("date")
            df.to_parquet(path, index=False)
            updated += 1
            time.sleep(0.12)

        logger.info(f"raw_data 历史补档完成: {updated}/{len(set(instruments))}")
        return updated > 0 or not instruments

    def convert_to_qlib(self) -> bool:
        """
        将 Tushare 数据转换为 Qlib 格式并更新日历

        Returns
        -------
        bool 是否成功
        """
        try:
            # 1. 转换数据
            converter = TushareToQlibConverter(
                tushare_dir=str(self.tushare_dir),
                qlib_dir=str(self.qlib_data_path),
            )
            df = converter.convert()
            if df is None:
                logger.error("数据转换失败：convert() 返回 None")
                return False

            converter.save(df)

            # 2. 先更新日历，再修复 / 追加价格字段，避免漏掉最新交易日
            self._update_calendar()
            provider_count = self._ensure_provider_structure()
            if provider_count > 0:
                logger.info(f"Provider 目录已同步: {provider_count} 只股票")

            # 3. 首次 bootstrap 或缺失 bin 时，先整批写入前复权 OHLCVA
            features_dir = self.qlib_data_path / "features"
            missing_close_bins = []
            if features_dir.exists():
                for raw_path in sorted(self.raw_data_dir.glob("*.parquet")):
                    close_bin = features_dir / raw_path.stem / "close.day.bin"
                    if not close_bin.exists():
                        missing_close_bins.append(raw_path.stem)
            if missing_close_bins:
                bootstrapped = converter.build_adjusted_bins_for_instruments(
                    missing_close_bins
                )
                if bootstrapped != len(missing_close_bins):
                    logger.error(
                        "前复权 bin 初始化不完整: 期望=%s, 实际写入=%s",
                        len(missing_close_bins),
                        bootstrapped,
                    )
                    return False
                logger.info(f"前复权 bin 初始化完成: {bootstrapped} 只股票")

            repair_stats = converter.repair_price_provider()
            if any(v > 0 for v in repair_stats.values()):
                logger.info(f"Provider 修复完成: {repair_stats}")

            # 4. 更新 close.day.bin
            n = converter.update_close_bins()
            if n > 0:
                logger.info(f"价格 bin 已更新 {n} 只股票")

            # 5. 更新 open/high/low/volume/amount 的 bin 文件
            ohlcv_counts = converter.update_ohlcv_bins()
            if any(v > 0 for v in ohlcv_counts.values()):
                logger.info(f"OHLCV bin 已更新: {ohlcv_counts}")

            return True

        except Exception as e:
            logger.error(f"数据转换失败: {e}")
            return False

    def _update_calendar(self):
        """从 daily_basic.parquet 提取交易日更新日历文件"""
        daily_path = self.tushare_dir / "daily_basic.parquet"
        if not daily_path.exists():
            return

        df = pd.read_parquet(daily_path, columns=["trade_date"])
        new_dates = (
            pd.to_datetime(df["trade_date"], format="%Y%m%d").drop_duplicates().sort_values()
        )

        cal_file = self.qlib_data_path / "calendars" / "day.txt"
        if cal_file.exists():
            existing = pd.read_csv(cal_file, header=None, names=["date"], parse_dates=["date"])
            all_dates = pd.concat([existing["date"], new_dates]).drop_duplicates().sort_values()
        else:
            cal_file.parent.mkdir(parents=True, exist_ok=True)
            all_dates = new_dates

        cal_file.write_text("\n".join(all_dates.dt.strftime("%Y-%m-%d")) + "\n")
        logger.info(f"日历已更新至 {all_dates.iloc[-1].strftime('%Y-%m-%d')}")

    def regenerate_selections(self, update_days: int = 10, force: bool = False) -> bool:
        """
        重新生成选股列表（支持增量更新）

        Parameters
        ----------
        update_days : int
            增量更新时，重新计算最近 N 天的选股。默认 10 天。
            如果 CSV 不存在或 force=True，则进行全量重算。
        force : bool
            是否强制全量重算。默认 False（增量更新）。

        Returns
        -------
        bool 是否成功
        """
        try:
            from core.strategy import Strategy
            from pandas import Timestamp, Timedelta
            import os

            success = True
            strategy_names = Strategy.list_available()
            total = len(strategy_names)

            for idx, name in enumerate(strategy_names, 1):
                strategy = Strategy.load(name)
                strategy.validate_data_requirements()

                csv_path = strategy.selections_path()
                update_start_date = None

                if not force and csv_path.exists():
                    existing_df = pd.read_csv(csv_path, parse_dates=["date"])
                    if not existing_df.empty:
                        last_date = existing_df["date"].max()
                        update_start = last_date + Timedelta(days=1)
                        today = Timestamp(self.get_last_trading_date())

                        if update_start <= today:
                            update_start_date = update_start.strftime("%Y-%m-%d")
                            print(
                                f"[{idx}/{total}] 增量更新 {name}: {last_date.strftime('%Y-%m-%d')} -> {update_start_date}"
                            )
                        else:
                            print(f"[{idx}/{total}] 已是最新 {name}")
                            continue
                    else:
                        update_start_date = None
                else:
                    print(f"[{idx}/{total}] 全量重算 {name} (force=True 或 CSV 不存在)")

                if update_start_date:
                    strategy.generate_selections(
                        force=False,
                        update_start_date=update_start_date,
                    )
                else:
                    strategy.generate_selections(force=True)

            return success

        except Exception as e:
            logger.error(f"重新计算选股失败: {e}")
            return False

    def update_daily(self) -> dict:
        """
        每日更新入口（整合：检查 → 下载 → 选股）

        Returns
        -------
        dict
            更新结果 {"success": bool, "message": str, "data_updated": bool, "selections_updated": bool}
        """
        results = {
            "data_updated": False,
            "raw_data_updated": False,
            "reference_updated": False,
            "converted": False,
            "selections_updated": False,
            "precheck_ok": False,
        }

        # 1. 检查是否需要更新
        bootstrap_needed = self._needs_bootstrap()
        local_date = self.get_last_trading_date()
        remote_date = self.get_remote_latest_date()
        precheck_before = run_data_precheck(universe="csi300", require_st_history=True)
        need_market_update = bootstrap_needed or self.check_update_needed()
        need_reference_update = bootstrap_needed or not precheck_before.ok
        need_provider_repair = bootstrap_needed or any(
            PROVIDER_PRECHECK_KEYWORD in msg for msg in precheck_before.errors
        )
        need_index_daily_refresh = bootstrap_needed or need_market_update or any(
            "index_daily" in msg for msg in precheck_before.errors
        )

        print(f"[1/5] 检查数据更新...")
        print(f"      本地最新: {local_date.strftime('%Y-%m-%d')}")
        if remote_date:
            print(f"      远程最新: {remote_date.strftime('%Y-%m-%d')}")
        if bootstrap_needed:
            print("      检测到首次 bootstrap，缺少核心数据文件，将执行全量历史初始化")
        if precheck_before.errors:
            print("      预检缺口:")
            for msg in precheck_before.errors:
                print(f"        - {msg}")

        if not need_market_update and not need_reference_update:
            print("      数据已是最新，无需更新")
            return {"success": True, "message": "数据已是最新"}

        if get_tushare_pro() is None:
            message = "Tushare API 不可用，请先安装 `pip install -e .[full]` 并设置 `TUSHARE_TOKEN`。"
            print(f"      {message}")
            return {"success": False, "message": message}

        # 2. 下载 Tushare 数据
        print(f"\n[2/5] 下载 Tushare 数据...")
        if self.download_stock_basic():
            print("      股票基本信息 ✓")

        if need_market_update:
            market_start = BOOTSTRAP_MARKET_START if bootstrap_needed else None
            if self.download_daily_basic(start_date=market_start):
                print("      每日基础数据 ✓")
                results["data_updated"] = True
            else:
                print("      每日基础数据 (跳过)")

            if self.download_adj_factor():
                print("      复权因子 ✓")
            else:
                print("      复权因子 (跳过)")

            if self.download_financial_data():
                print("      财务数据 ✓")
            else:
                print("      财务数据 (跳过)")

            if self.update_raw_data_quotes(start_date=market_start):
                print("      raw_data 原始行情 ✓")
                results["raw_data_updated"] = True
            else:
                print("      raw_data 原始行情 (跳过)")
        else:
            print("      行情/财务数据已是最新，跳过增量下载")

        if need_index_daily_refresh:
            if self.download_index_daily():
                print("      指数日线 ✓")
            else:
                print("      指数日线 (跳过)")

        ref_updated = False
        if self.download_index_weight():
            print("      历史指数成分 ✓")
            ref_updated = True
        else:
            print("      历史指数成分 (跳过)")

        if self.download_namechange():
            print("      历史名称变更 ✓")
            ref_updated = True
        else:
            print("      历史名称变更 (跳过)")

        results["reference_updated"] = ref_updated

        # 3. 转换 Tushare → Qlib 格式 + 更新日历
        print(f"\n[3/5] 转换数据格式...")
        if need_market_update or need_provider_repair:
            if self.convert_to_qlib():
                print("      Tushare → Qlib 转换 ✓")
                results["converted"] = True
            else:
                print("      数据转换失败")
        else:
            print("      无需转换，沿用现有 Qlib provider")

        # 4. 正式预检
        print(f"\n[4/5] 正式数据预检...")
        precheck_after = run_data_precheck(universe="csi300", require_st_history=True)
        results["precheck_ok"] = precheck_after.ok
        if precheck_after.ok:
            print("      历史沪深300成分 + 历史 ST 预检 ✓")
        else:
            print("      数据预检失败")
            for msg in precheck_after.errors:
                print(f"        - {msg}")

        # 5. 重新计算选股
        print(f"\n[5/5] 重新计算选股...")
        if precheck_after.ok and self.regenerate_selections():
            results["selections_updated"] = True

        success = results["precheck_ok"] and (
            results["data_updated"] or results["reference_updated"] or results["selections_updated"]
        )
        message = (
            f"数据更新: {results['data_updated']}, raw_data: {results['raw_data_updated']}, 历史数据更新: {results['reference_updated']}, "
            f"转换: {results['converted']}, 预检: {results['precheck_ok']}, "
            f"选股更新: {results['selections_updated']}"
        )

        return {
            "success": success,
            "message": message,
            "data_updated": results["data_updated"],
            "raw_data_updated": results["raw_data_updated"],
            "reference_updated": results["reference_updated"],
            "converted": results["converted"],
            "precheck_ok": results["precheck_ok"],
            "selections_updated": results["selections_updated"],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    updater = DataUpdater()
    result = updater.update_daily()
    print(result)
