"""
数据更新模块
每日收盘后自动更新 Tushare 数据 → 重新计算选股
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd

from modules.data.precheck import run_data_precheck
from modules.data.tushare_to_qlib import TushareToQlibConverter

logger = logging.getLogger(__name__)


PROVIDER_PRECHECK_KEYWORD = "Qlib provider 字段不一致"


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


class DataUpdater:
    """数据更新器"""

    def __init__(self, qlib_data_path: str = None):
        self.qlib_data_path = qlib_data_path or "~/code/qlib/data/qlib_data/cn_data"
        self.qlib_data_path = Path(self.qlib_data_path).expanduser()

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
    def _format_raw_quote_frame(df: pd.DataFrame) -> pd.DataFrame:
        """标准化 raw_data 行情字段，保留原始 pre_close 供成交约束使用。"""
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "pre_close", "volume", "amount", "symbol"]
            )

        out = df.copy()
        out["date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce")
        out["symbol"] = out["ts_code"]
        out = out.rename(columns={"vol": "volume"})
        for col in ["open", "high", "low", "close", "pre_close", "volume", "amount"]:
            if col not in out.columns:
                out[col] = np.nan
        out = out[
            ["date", "open", "high", "low", "close", "pre_close", "volume", "amount", "symbol"]
        ]
        return out.dropna(subset=["date"]).sort_values("date")

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

            df = pro.trade_cal(
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

    def download_daily_basic(self) -> bool:
        """
        下载每日基础数据（增量）

        Returns
        -------
        bool 是否成功
        """
        pro = get_tushare_pro()
        if pro is None:
            return False

        output_path = self.tushare_dir / "daily_basic.parquet"

        try:
            # 确定起始日期
            if output_path.exists():
                existing = pd.read_parquet(output_path)
                max_date = existing["trade_date"].max()
                start_date = self._next_calendar_date_str(max_date)
            else:
                start_date = "20200101"

            # 下载新数据
            df = pro.daily_basic(start_date=start_date)

            if df is not None and len(df) > 0:
                if output_path.exists():
                    combined = pd.concat([existing, df], ignore_index=True)
                    combined = combined.drop_duplicates(
                        subset=["ts_code", "trade_date"], keep="last"
                    )
                    combined.to_parquet(output_path, index=False)
                else:
                    df.to_parquet(output_path, index=False)

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

        # 获取股票列表
        try:
            stock_df = pro.stock_basic(list_status="L", fields="ts_code")
            ts_codes = stock_df["ts_code"].tolist()
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return False

        for api_name, filename, date_col in datasets:
            output_path = self.tushare_dir / filename
            success = self._download_financial_dataset(
                pro, api_name, output_path, ts_codes, date_col
            )
            results.append(success)

        return all(results)

    def _download_financial_dataset(
        self, pro, api_name: str, output_path: Path, ts_codes: list, date_col: str
    ) -> bool:
        """下载单个财务数据集"""
        try:
            # 确定起始日期
            if output_path.exists():
                existing = pd.read_parquet(output_path)
                max_date = existing[date_col].max()
                start_date = str(int(max_date) - 10000)  # 往前推确保完整
            else:
                start_date = "20100101"

            all_data = []
            batch_size = 500

            for i in range(0, len(ts_codes), batch_size):
                batch = ts_codes[i : i + batch_size]
                try:
                    api_func = getattr(pro, api_name)
                    df = api_func(ts_code=",".join(batch), start_date=start_date)
                    if df is not None and len(df) > 0:
                        all_data.append(df)
                except Exception as e:
                    logger.warning(f"{api_name} 批次 {i // batch_size + 1} 失败: {e}")

            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                if output_path.exists():
                    combined = pd.concat([existing, df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["ts_code", date_col], keep="last")
                    combined.to_parquet(output_path, index=False)
                else:
                    df.to_parquet(output_path, index=False)

                logger.info(f"已更新 {api_name}: {len(df)} 条")
            else:
                logger.info(f"{api_name} 无新数据")

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
            df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry")
            output_path = self.tushare_dir / "stock_basic.csv"
            df.to_csv(output_path, index=False)

            # 同时保存行业数据
            industry_path = self.tushare_dir / "stock_industry.csv"
            if not industry_path.exists():
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
                df = pro.index_daily(
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

        try:
            if output_path.exists():
                existing = pd.read_parquet(output_path, columns=["trade_date"])
                max_date = str(existing["trade_date"].max())
                # 指数成分按月调整，回拉约 3 个月以覆盖修订和重复月度快照
                start_date = (pd.Timestamp(max_date) - pd.Timedelta(days=93)).strftime("%Y%m%d")
            else:
                start_date = "20160101"

            all_data = []
            for index_code in indices:
                for win_start, win_end in self._date_windows(start_date, end_date, step_days=366):
                    last_exc = None
                    for _ in range(3):
                        try:
                            df = pro.index_weight(
                                index_code=index_code,
                                start_date=win_start,
                                end_date=win_end,
                            )
                            if df is not None and len(df) > 0:
                                all_data.append(df)
                            last_exc = None
                            break
                        except Exception as exc:
                            last_exc = exc
                            time.sleep(0.5)
                    if last_exc is not None:
                        logger.warning(
                            f"index_weight {index_code} {win_start}-{win_end} 失败: {last_exc}"
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
                last_exc = None
                for _ in range(3):
                    try:
                        df = pro.namechange(start_date=win_start, end_date=win_end)
                        if df is not None and len(df) > 0:
                            all_data.append(df)
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        time.sleep(0.5)
                if last_exc is not None:
                    logger.warning(f"namechange {win_start}-{win_end} 失败: {last_exc}")
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

    def download_adj_factor_incremental(self) -> bool:
        """下载 adj_factor 增量数据（按交易日批量下载）。"""
        pro = get_tushare_pro()
        if pro is None:
            return False

        output_path = self.tushare_dir / "adj_factor.parquet"

        try:
            if output_path.exists():
                existing = pd.read_parquet(output_path, columns=["trade_date"])
                max_date = str(existing["trade_date"].max())
                start_date = self._next_calendar_date_str(max_date)
            else:
                start_date = "20160101"

            end_date = datetime.now().strftime("%Y%m%d")

            # 获取交易日历以确定需要下载的交易日
            try:
                cal_df = pro.trade_cal(start_date=start_date, end_date=end_date, is_open="1")
                if cal_df is None or cal_df.empty:
                    logger.info("adj_factor 无新交易日")
                    return True
                trade_dates = sorted(cal_df["cal_date"].tolist())
            except Exception as e:
                logger.warning(f"获取交易日历失败: {e}")
                return False

            if not trade_dates:
                logger.info("adj_factor 已是最新")
                return True

            all_data = []
            for i, date_str in enumerate(trade_dates):
                try:
                    df = pro.adj_factor(trade_date=date_str)
                    if df is not None and len(df) > 0:
                        all_data.append(df)
                except Exception as e:
                    logger.warning(f"adj_factor {date_str} 失败: {e}")
                    time.sleep(0.5)

                if (i + 1) % 20 == 0:
                    logger.info(f"  adj_factor 已拉取 {i + 1}/{len(trade_dates)} 天")
                time.sleep(0.12)

            if not all_data:
                logger.info("adj_factor 无新数据")
                return True

            result = pd.concat(all_data, ignore_index=True)
            # 只保留必要列
            keep_cols = [c for c in result.columns if c in ("ts_code", "trade_date", "adj_factor")]
            result = result[keep_cols]
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

        通过 pro.daily(trade_date=...) 按日期批量拉取，
        对每只股票计算调整比例（splice point ratio）后追加到 bin 文件。
        """
        cal_file = self.qlib_data_path / "calendars" / "day.txt"
        features_dir = self.qlib_data_path / "features"
        if not cal_file.exists() or not features_dir.exists():
            return False

        cal = pd.read_csv(cal_file, header=None, names=["date"], parse_dates=["date"])
        cal_last_idx_global = len(cal) - 1

        # 扫描全部股票，找到最小的 last_bin_idx（排除极短历史新股）
        min_last_bin_idx = cal_last_idx_global
        for inst_dir in features_dir.iterdir():
            bin_f = inst_dir / "close.day.bin"
            if not bin_f.exists():
                continue
            raw0 = np.fromfile(bin_f, dtype="<f4")
            if len(raw0) < 252 or np.isnan(raw0[0]):  # 至少一年数据
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

        # 按日期批量拉取（每次一个交易日，覆盖全市场）
        daily_map: dict = {}  # instrument -> {cal_idx: {field: value}}
        for i, (cal_idx, date) in enumerate(zip(missing_idxs, missing_dates)):
            date_str = date.strftime("%Y%m%d")
            try:
                df = pro.daily(trade_date=date_str)
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
                inst = exch.lower() + code  # 000001.SZ -> sz000001
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

            # 逐股确定自己的 end_idx 和需要写入的 cal_idxs
            stock_end_idx = int(raw_c[0]) + len(raw_c) - 2
            if stock_end_idx >= cal_last_idx:
                continue  # 该股已是最新，跳过

            stock_missing_idxs = list(range(stock_end_idx + 1, cal_last_idx + 1))

            # splice-point 调整比例
            bin_last_close = float(raw_c[-1])
            first_new_cal = stock_missing_idxs[0]
            first_pre = date_data.get(first_new_cal, {}).get("pre_close", np.nan)
            if first_pre and not np.isnan(first_pre) and first_pre > 0:
                adj = bin_last_close / first_pre
            else:
                adj = 1.0

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
                        if not np.isnan(v) and v > 0 and field in fields_price:
                            v = v * adj
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

        logger.info(f"已更新 {updated} 只股票的价格 bin 数据")
        return updated > 0

    def update_raw_data_quotes(self, start_date: str = None, end_date: str = None) -> bool:
        """按交易日增量更新 raw_data 原始行情文件。"""
        pro = get_tushare_pro()
        if pro is None:
            logger.warning("Tushare API 不可用，跳过 raw_data 更新")
            return False

        daily_basic_path = self.tushare_dir / "daily_basic.parquet"
        if not daily_basic_path.exists():
            logger.warning("缺少 daily_basic.parquet，无法推断 raw_data 交易日")
            return False

        try:
            daily_dates = pd.read_parquet(daily_basic_path, columns=["trade_date"])
            trade_dates = pd.to_datetime(
                daily_dates["trade_date"], format="%Y%m%d", errors="coerce"
            ).dropna()
            trade_dates = trade_dates.sort_values().drop_duplicates()
        except Exception as e:
            logger.error(f"读取 daily_basic 失败，无法更新 raw_data: {e}")
            return False

        if trade_dates.empty:
            logger.warning("daily_basic 不包含有效交易日，跳过 raw_data 更新")
            return False

        end_ts = pd.Timestamp(end_date) if end_date else trade_dates.max()
        if start_date is None:
            # 不用“全局最新文件日期”推断增量起点；少量新文件会掩盖大批陈旧文件。
            # 每次固定回补最近一段交易日，既能修复漏更，也能控制 API 开销。
            start_ts = max(trade_dates.min(), end_ts - timedelta(days=45))
        else:
            start_ts = pd.Timestamp(start_date)

        target_dates = trade_dates[(trade_dates >= start_ts) & (trade_dates <= end_ts)]
        if len(target_dates) == 0:
            logger.info("raw_data 已是最新")
            return True

        updates = {}
        for i, trade_date in enumerate(target_dates, 1):
            date_str = trade_date.strftime("%Y%m%d")
            try:
                df = pro.daily(
                    trade_date=date_str,
                    fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
                )
            except Exception as e:
                logger.warning(f"raw_data daily {date_str} 失败: {e}")
                time.sleep(1)
                continue

            if df is None or df.empty:
                continue

            df = self._format_raw_quote_frame(df)
            df["raw_file"] = df["symbol"].map(
                lambda ts_code: f"{ts_code.split('.')[1].lower()}{ts_code.split('.')[0]}.parquet"
            )

            for raw_file, grp in df.groupby("raw_file"):
                updates.setdefault(raw_file, []).append(grp.drop(columns=["raw_file"]).copy())

            if i % 10 == 0:
                logger.info(f"raw_data 已拉取 {i}/{len(target_dates)} 个交易日")
            time.sleep(0.12)

        updated_files = 0
        for raw_file, parts in updates.items():
            path = self.raw_data_dir / raw_file
            new_df = pd.concat(parts, ignore_index=True).sort_values("date")
            if path.exists():
                existing = pd.read_parquet(path)
                new_df = pd.concat([existing, new_df], ignore_index=True)
            new_df = new_df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
            new_df.to_parquet(path, index=False)
            updated_files += 1

        logger.info(f"raw_data 已更新 {updated_files} 个文件")
        return updated_files > 0

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
                df = pro.daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
                )
            except Exception as e:
                logger.warning(f"补档 {instrument} 失败: {e}")
                time.sleep(1)
                continue

            if df is None or df.empty:
                logger.warning(f"补档 {instrument} 未返回数据")
                continue

            df = self._format_raw_quote_frame(df)
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

            repair_stats = converter.repair_price_provider()
            if any(v > 0 for v in repair_stats.values()):
                logger.info(f"Provider 修复完成: {repair_stats}")

            # 3. 更新 close.day.bin
            n = converter.update_close_bins()
            if n > 0:
                logger.info(f"价格 bin 已更新 {n} 只股票")

            # 4. 更新 open/high/low/volume/amount 的 bin 文件
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
        local_date = self.get_last_trading_date()
        remote_date = self.get_remote_latest_date()
        precheck_before = run_data_precheck(universe="csi300", require_st_history=True)
        need_market_update = self.check_update_needed()
        need_reference_update = not precheck_before.ok
        need_provider_repair = any(
            PROVIDER_PRECHECK_KEYWORD in msg for msg in precheck_before.errors
        )
        need_index_daily_refresh = need_market_update or any(
            "index_daily" in msg for msg in precheck_before.errors
        )

        print(f"[1/5] 检查数据更新...")
        print(f"      本地最新: {local_date.strftime('%Y-%m-%d')}")
        if remote_date:
            print(f"      远程最新: {remote_date.strftime('%Y-%m-%d')}")
        if precheck_before.errors:
            print("      预检缺口:")
            for msg in precheck_before.errors:
                print(f"        - {msg}")

        if not need_market_update and not need_reference_update:
            print("      数据已是最新，无需更新")
            return {"success": True, "message": "数据已是最新"}

        # 2. 下载 Tushare 数据
        print(f"\n[2/5] 下载 Tushare 数据...")
        if self.download_stock_basic():
            print("      股票基本信息 ✓")

        if need_market_update:
            if self.download_daily_basic():
                print("      每日基础数据 ✓")
                results["data_updated"] = True
            else:
                print("      每日基础数据 (跳过)")

            if self.download_financial_data():
                print("      财务数据 ✓")
            else:
                print("      财务数据 (跳过)")

            if self.download_adj_factor_incremental():
                print("      复权因子 ✓")
            else:
                print("      复权因子 (跳过)")

            if self.update_raw_data_quotes():
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
