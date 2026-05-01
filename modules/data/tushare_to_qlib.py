"""
Tushare 数据转换为 Qlib 格式 (简化版)
直接合并每日数据，财务数据前向填充
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _ts_code_to_instrument(ts_code_series: pd.Series) -> pd.Series:
    """ts_code (000001.SZ) → instrument (sz000001)"""
    return ts_code_series.apply(
        lambda x: x.split('.')[1].lower() + x.split('.')[0] if '.' in x else x.lower()
    )


class TushareToQlibConverter:
    """Tushare 数据转 Qlib 格式转换器"""

    def __init__(self, tushare_dir: str = None, qlib_dir: str = None):
        self.tushare_dir = Path(tushare_dir or "~/code/qlib/data/tushare").expanduser()
        self.qlib_dir = Path(qlib_dir or "~/code/qlib/data/qlib_data/cn_data").expanduser()
        self._adj_ratio_cache = None
        self._calendar_cache = None

    def _load_calendar(self):
        if self._calendar_cache is not None:
            return self._calendar_cache
        cal_file = self.qlib_dir / "calendars" / "day.txt"
        cal = pd.read_csv(cal_file, header=None, names=["date"], parse_dates=["date"])
        cal_dates = cal["date"].dt.normalize()
        date_to_idx = {d: i for i, d in enumerate(cal_dates)}
        idx_to_date = {i: d for d, i in date_to_idx.items()}
        cal_last_idx = len(cal) - 1
        self._calendar_cache = (date_to_idx, idx_to_date, cal_last_idx)
        return self._calendar_cache

    def _load_adj_ratio_map(self) -> dict:
        """加载 adj_factor.parquet，返回 {instrument: {date: adj_ratio}}

        adj_ratio = adj_factor[date] / adj_factor[latest_date]
        前复权价格 = raw_price * adj_ratio
        """
        if self._adj_ratio_cache is not None:
            return self._adj_ratio_cache

        adj_path = self.tushare_dir / "adj_factor.parquet"
        if not adj_path.exists():
            return {}

        adj_df = pd.read_parquet(adj_path, columns=["ts_code", "trade_date", "adj_factor"])
        adj_df["date"] = pd.to_datetime(adj_df["trade_date"], format="%Y%m%d").dt.normalize()
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
            result[inst] = dict(zip(grp["date"], ratios))

        self._adj_ratio_cache = result
        logger.info(f"加载 adj_factor: {len(result)} 只股票")
        return result

    def compute_forward_adjusted_prices(self) -> dict:
        """用 adj_factor 计算前复权价格，返回 {instrument: DataFrame}

        DataFrame 列: [date, open, high, low, close, volume, amount]
        价格字段已前复权，volume/amount 保持原始值
        """
        adj_map = self._load_adj_ratio_map()
        if not adj_map:
            logger.warning("adj_factor.parquet 不存在或为空，无法计算前复权价格")
            return {}

        raw_dir = self.qlib_dir.parent / "raw_data"
        if not raw_dir.exists():
            logger.warning("raw_data 目录不存在")
            return {}

        fields = ["open", "high", "low", "close", "volume", "amount"]
        price_fields = {"open", "high", "low", "close"}
        results = {}
        raw_files = sorted(raw_dir.glob("*.parquet"))

        for raw_path in raw_files:
            inst = raw_path.stem
            if inst not in adj_map:
                continue
            inst_adj = adj_map[inst]

            try:
                df = pd.read_parquet(raw_path, columns=["date"] + fields)
            except Exception:
                continue

            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            df = df.dropna(subset=["date"])
            if df.empty:
                continue

            df["adj_ratio"] = df["date"].map(inst_adj)
            for fld in price_fields:
                if fld in df.columns:
                    mask = df["adj_ratio"].notna() & np.isfinite(df["adj_ratio"])
                    df.loc[mask, fld] = df.loc[mask, fld] * df.loc[mask, "adj_ratio"]
                    df.loc[~mask, fld] = np.nan
            df = df.drop(columns=["adj_ratio"], errors="ignore")
            df = df.dropna(subset=[c for c in fields if c in df.columns], how="all")
            results[inst] = df

        logger.info(f"计算前复权价格: {len(results)} 只股票")
        return results

    def write_adjusted_bins(self, adjusted_data: dict) -> int:
        """将前复权价格写入 qlib bin 文件

        bin 格式: [start_idx(float32), val1, val2, ...]
        """
        features_dir = self.qlib_dir / "features"
        cal_file = self.qlib_dir / "calendars" / "day.txt"

        if not cal_file.exists():
            logger.warning("日历文件不存在，无法写入 bin")
            return 0

        date_to_idx, _, _ = self._load_calendar()

        fields = ["open", "high", "low", "close", "volume", "amount"]
        written = 0

        for inst, df in adjusted_data.items():
            inst_dir = features_dir / inst
            if not inst_dir.exists():
                continue

            df = df[df["date"].isin(date_to_idx)].copy()
            if df.empty:
                continue

            df["cal_idx"] = df["date"].map(date_to_idx)
            df = df.dropna(subset=["cal_idx"])
            df["cal_idx"] = df["cal_idx"].astype(int)

            for fld in fields:
                if fld not in df.columns:
                    continue
                fld_data = df[["cal_idx", fld]].dropna(subset=[fld]).sort_values("cal_idx")
                if fld_data.empty:
                    continue
                start_idx = int(fld_data["cal_idx"].min())
                vals = fld_data[fld].values.astype(np.float32)
                payload = np.empty(vals.size + 1, dtype="<f4")
                payload[0] = np.float32(start_idx)
                payload[1:] = vals
                payload.tofile(inst_dir / f"{fld}.day.bin")

            written += 1

        logger.info(f"写入前复权 bin: {written} 只股票")
        return written

    def load_tushare_data(self, name: str) -> Optional[pd.DataFrame]:
        """加载 Tushare 数据"""
        path = self.tushare_dir / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            logger.info(f"加载 {name}: {len(df):,} 条")
            return df
        logger.warning(f"文件不存在: {path}")
        return None

    def convert(self) -> pd.DataFrame:
        """转换并合并所有数据（分批处理避免 OOM）"""
        # 1. 加载每日基本面数据 (主表)
        daily = self.load_tushare_data('daily_basic')
        if daily is None:
            return None

        daily['instrument'] = _ts_code_to_instrument(daily['ts_code'])
        daily['datetime'] = pd.to_datetime(daily['trade_date'], format='%Y%m%d')

        cols = ['instrument', 'datetime', 'turnover_rate', 'turnover_rate_f',
                'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
                'total_mv', 'circ_mv', 'free_mv', 'dv_ratio', 'dv_ttm']
        daily = daily[[c for c in cols if c in daily.columns]].copy()
        del cols

        # 2. 预加载并标准化所有季度财务数据
        quarterlies = []
        fina = self.load_tushare_data('fina_indicator')
        if fina is not None:
            fina['instrument'] = _ts_code_to_instrument(fina['ts_code'])
            fina['datetime'] = pd.to_datetime(fina['ann_date'], format='%Y%m%d', errors='coerce')
            fina = fina.dropna(subset=['datetime'])
            fina['datetime'] = fina['datetime'] + pd.Timedelta(days=1)
            fina_cols = ['instrument', 'datetime', 'roe', 'roe_dt', 'roa',
                         'current_ratio', 'quick_ratio', 'debt_to_assets',
                         'ebit', 'ebitda', 'fcff']
            fina = fina[[c for c in fina_cols if c in fina.columns]].copy()
            fina = fina.rename(columns={
                'roe': 'roe_fina', 'roe_dt': 'roe_dt_fina', 'roa': 'roa_fina',
                'current_ratio': 'current_ratio_fina', 'quick_ratio': 'quick_ratio_fina',
                'debt_to_assets': 'debt_to_assets_fina',
                'ebit': 'ebit_fina', 'ebitda': 'ebitda_fina', 'fcff': 'fcff_fina'
            })
            quarterlies.append(fina)
            del fina

        income = self.load_tushare_data('income')
        if income is not None:
            income['instrument'] = _ts_code_to_instrument(income['ts_code'])
            income['datetime'] = pd.to_datetime(income['ann_date'], format='%Y%m%d', errors='coerce')
            income = income.dropna(subset=['datetime'])
            income['datetime'] = income['datetime'] + pd.Timedelta(days=1)
            inc_cols = ['instrument', 'datetime', 'total_revenue', 'revenue',
                        'n_income', 'n_income_attr_p', 'operate_profit']
            income = income[[c for c in inc_cols if c in income.columns]].copy()
            income = income.rename(columns={
                'total_revenue': 'total_revenue_inc', 'revenue': 'revenue_inc',
                'n_income': 'n_income_inc', 'n_income_attr_p': 'n_income_attr_p_inc',
                'operate_profit': 'operate_profit_inc'
            })
            quarterlies.append(income)
            del income

        cashflow = self.load_tushare_data('cashflow')
        if cashflow is not None:
            cashflow['instrument'] = _ts_code_to_instrument(cashflow['ts_code'])
            cashflow['datetime'] = pd.to_datetime(cashflow['ann_date'], format='%Y%m%d', errors='coerce')
            cashflow = cashflow.dropna(subset=['datetime'])
            cashflow['datetime'] = cashflow['datetime'] + pd.Timedelta(days=1)
            cf_cols = ['instrument', 'datetime', 'n_cashflow_act']
            cashflow = cashflow[[c for c in cf_cols if c in cashflow.columns]].copy()
            quarterlies.append(cashflow)
            del cashflow

        balance = self.load_tushare_data('balancesheet')
        if balance is not None:
            balance['instrument'] = _ts_code_to_instrument(balance['ts_code'])
            balance['datetime'] = pd.to_datetime(balance['ann_date'], format='%Y%m%d', errors='coerce')
            balance = balance.dropna(subset=['datetime'])
            balance['datetime'] = balance['datetime'] + pd.Timedelta(days=1)
            bs_cols = ['instrument', 'datetime', 'surplus_rese', 'undistr_porfit',
                       'total_liab', 'money_cap']
            balance = balance[[c for c in bs_cols if c in balance.columns]].copy()
            quarterlies.append(balance)
            del balance

        # 3. 分批处理：按股票分组，每批 500 只
        all_instruments = daily['instrument'].unique()
        batch_size = 500
        batches = [all_instruments[i:i + batch_size] for i in range(0, len(all_instruments), batch_size)]
        logger.info(f"分批处理: {len(all_instruments)} 只股票, {len(batches)} 批")

        result_parts = []
        for bi, batch_insts in enumerate(batches):
            batch_daily = daily[daily['instrument'].isin(batch_insts)].copy()

            for q in quarterlies:
                batch_q = q[q['instrument'].isin(batch_insts)]
                if batch_q.empty:
                    continue
                batch_daily = self._forward_fill(batch_daily, batch_q, '')

            batch_daily = self._calculate_derived(batch_daily)
            result_parts.append(batch_daily)

            if (bi + 1) % 5 == 0:
                logger.info(f"  已处理 {(bi + 1) * batch_size} 只...")

            del batch_daily

        del daily
        result = pd.concat(result_parts, ignore_index=True)
        del result_parts

        logger.info(f"最终数据: {len(result):,} 条, {len(result.columns)} 列")
        return result

    def _forward_fill(self, daily: pd.DataFrame, quarterly: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """将季度财务数据前向填充到每日数据

        用 merge_asof 按最近的前一个公告日匹配季度数据到 daily，
        然后按股票分组 ffill，避免笛卡尔积导致的内存爆炸。
        """
        fina_cols = [c for c in quarterly.columns if c not in ['instrument', 'datetime']]

        # 合并季度公告到每个交易日（找 <= 交易日 的最近公告）
        merged = pd.merge_asof(
            daily.sort_values('datetime'),
            quarterly.sort_values('datetime'),
            by='instrument', on='datetime',
            direction='backward',
        )

        # 按股票分组，前向填充财务数据
        for col in fina_cols:
            merged[col] = merged.groupby('instrument')[col].ffill()

        return merged

    def _calculate_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算衍生指标"""
        # 净利率 = 归母净利润 / 营业收入
        if 'n_income_attr_p_inc' in df.columns and 'revenue_inc' in df.columns:
            df['net_margin'] = df['n_income_attr_p_inc'] / df['revenue_inc'].replace(0, np.nan)
            logger.info("+ net_margin")

        # 账面市值比 = 1 / PB
        if 'pb' in df.columns:
            df['book_to_market'] = 1 / df['pb'].replace(0, np.nan)
            logger.info("+ book_to_market")

        # ROIC 代理
        if 'roe_fina' in df.columns and 'debt_to_assets_fina' in df.columns:
            df['roic_proxy'] = df['roe_fina'] * (1 - df['debt_to_assets_fina'] / 100)
            logger.info("+ roic_proxy")

        # EBIT / 总市值 (息税前利润市值比)
        if 'ebit_fina' in df.columns and 'total_mv' in df.columns:
            df['ebit_to_mv'] = df['ebit_fina'] / df['total_mv'].replace(0, np.nan)
            logger.info("+ ebit_to_mv")

        # EBITDA / 总市值
        if 'ebitda_fina' in df.columns and 'total_mv' in df.columns:
            df['ebitda_to_mv'] = df['ebitda_fina'] / df['total_mv'].replace(0, np.nan)
            logger.info("+ ebitda_to_mv")

        # 经营现金流 / 总市值
        if 'n_cashflow_act' in df.columns and 'total_mv' in df.columns:
            df['ocf_to_mv'] = df['n_cashflow_act'] / df['total_mv'].replace(0, np.nan)
            logger.info("+ ocf_to_mv")

        # 经营现金流 / 企业价值 (EV = 总市值 + 总负债 - 货币资金)
        if all(c in df.columns for c in ['n_cashflow_act', 'total_mv', 'total_liab', 'money_cap']):
            ev = df['total_mv'] + df['total_liab'].fillna(0) - df['money_cap'].fillna(0)
            df['ocf_to_ev'] = df['n_cashflow_act'] / ev.replace(0, np.nan)
            logger.info("+ ocf_to_ev")

        # 留存收益 = 盈余公积 + 未分配利润
        if 'surplus_rese' in df.columns and 'undistr_porfit' in df.columns:
            df['retained_earnings'] = df['surplus_rese'].fillna(0) + df['undistr_porfit'].fillna(0)
            logger.info("+ retained_earnings")

        # FCFF / 总市值 (企业自由现金流市值比)
        if 'fcff_fina' in df.columns and 'total_mv' in df.columns:
            df['fcff_to_mv'] = df['fcff_fina'] / df['total_mv'].replace(0, np.nan)
            logger.info("+ fcff_to_mv")

        return df

    def update_close_bins(self) -> int:
        """增量更新 close.day.bin，优先使用 adj_factor 前复权

        如果 adj_factor.parquet 存在，用 adj_ratio = adj_factor[date] / adj_factor[latest]
        计算前复权 close 并追加。否则回退到 splice-point ratio。
        """
        raw_dir = self.qlib_dir.parent / "raw_data"
        cal_file = self.qlib_dir / "calendars" / "day.txt"
        features_dir = self.qlib_dir / "features"

        if not cal_file.exists() or not features_dir.exists():
            logger.warning("缺少必要文件，跳过 close bin 更新")
            return 0

        date_to_idx, idx_to_date, cal_last_idx = self._load_calendar()
        idx_to_date = {i: d for d, i in date_to_idx.items()}

        # 尝试使用 adj_factor
        adj_map = self._load_adj_ratio_map()
        use_adj = bool(adj_map)
        if not use_adj:
            logger.warning("adj_factor.parquet 不存在，回退到 splice-point ratio 更新 close")

        # 读取 raw_data 中的 close
        updated = 0
        raw_files = sorted(raw_dir.glob("*.parquet")) if raw_dir.exists() else []

        for raw_path in raw_files:
            inst = raw_path.stem
            bin_file = features_dir / inst / "close.day.bin"
            if not bin_file.exists():
                continue

            raw_bin = np.fromfile(bin_file, dtype="<f4")
            if len(raw_bin) < 2 or np.isnan(raw_bin[0]):
                continue

            bin_end_idx = int(raw_bin[0]) + len(raw_bin) - 2
            if bin_end_idx >= cal_last_idx:
                continue

            try:
                df = pd.read_parquet(raw_path, columns=["date", "close"])
            except Exception:
                continue

            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            df = df.dropna(subset=["date"])
            df = df[df["date"].isin(date_to_idx)]
            if df.empty:
                continue
            df["cal_idx"] = df["date"].map(date_to_idx).astype(int)

            raw_close_map = dict(zip(df["cal_idx"], df["close"]))
            missing_idxs = list(range(bin_end_idx + 1, cal_last_idx + 1))

            new_vals = []
            if use_adj and inst in adj_map:
                inst_adj = adj_map[inst]
                for idx in missing_idxs:
                    d = idx_to_date.get(idx)
                    raw_c = raw_close_map.get(idx)
                    ratio = inst_adj.get(d) if d else None
                    if raw_c is not None and ratio is not None and np.isfinite(raw_c):
                        new_vals.append(float(raw_c) * float(ratio))
                    else:
                        new_vals.append(np.nan)
            else:
                # 回退：splice-point ratio
                bin_last_close = float(raw_bin[-1])
                db_last_close = raw_close_map.get(bin_end_idx)
                adj = bin_last_close / db_last_close if db_last_close and db_last_close > 0 else 1.0
                for idx in missing_idxs:
                    v = raw_close_map.get(idx)
                    new_vals.append(float(v) * adj if v and v > 0 else np.nan)

            if new_vals:
                with open(bin_file, "ab") as fp:
                    np.array(new_vals, dtype="<f4").tofile(fp)
                updated += 1

        logger.info(f"close.day.bin 已更新 {updated} 只股票 ({'adj_factor' if use_adj else 'splice-point'})")
        return updated

    @staticmethod
    def _read_bin_file(bin_file: Path):
        raw = np.fromfile(bin_file, dtype="<f4")
        if len(raw) < 2 or np.isnan(raw[0]):
            return None
        start_idx = int(raw[0])
        values = raw[1:].astype(np.float32, copy=False)
        end_idx = start_idx + len(values) - 1
        return start_idx, end_idx, values

    @staticmethod
    def _write_bin_file(bin_file: Path, start_idx: int, values) -> bool:
        values = np.asarray(values, dtype="<f4")
        if values.size == 0:
            return False
        payload = np.empty(values.size + 1, dtype="<f4")
        payload[0] = np.float32(start_idx)
        payload[1:] = values
        with open(bin_file, "wb") as fp:
            payload.tofile(fp)
        return True

    def repair_price_provider(self) -> dict:
        """修复价格 provider 中的超范围 / 字段错位问题。

        重建价格字段时优先使用 adj_factor 前复权，确保所有价格字段用同一 adj_ratio。
        """
        cal_file = self.qlib_dir / "calendars" / "day.txt"
        features_dir = self.qlib_dir / "features"
        raw_dir = self.qlib_dir.parent / "raw_data"

        if not cal_file.exists() or not features_dir.exists():
            logger.warning("缺少日历或 features 目录，跳过 provider 修复")
            return {}

        date_to_idx, idx_to_date, cal_last_idx = self._load_calendar()
        fields = ["open", "high", "low", "close", "volume", "amount"]
        price_fields = {"open", "high", "low", "close"}

        adj_map = self._load_adj_ratio_map()

        stats = {
            "truncated_files": 0,
            "repaired_instruments": 0,
            "unresolved_instruments": 0,
        }

        for inst_dir in features_dir.iterdir():
            if not inst_dir.is_dir():
                continue

            close_path = inst_dir / "close.day.bin"
            close_meta = self._read_bin_file(close_path) if close_path.exists() else None
            if close_meta is None:
                continue

            close_start, close_end, close_values = close_meta
            original_close_start = close_start
            original_close_end = close_end
            inst = inst_dir.name

            existing_meta = {"close": close_meta}
            inconsistent = close_start > cal_last_idx or close_end > cal_last_idx
            for field in fields:
                if field == "close":
                    continue
                bin_path = inst_dir / f"{field}.day.bin"
                meta = self._read_bin_file(bin_path) if bin_path.exists() else None
                if meta is None:
                    inconsistent = True
                    continue
                existing_meta[field] = meta
                _, field_end, _ = meta
                if field_end > cal_last_idx or field_end != close_end:
                    inconsistent = True

            if not inconsistent:
                continue

            raw_path = raw_dir / f"{inst}.parquet"
            raw_df = None
            raw_last_idx = None
            raw_maps = {}
            if raw_path.exists():
                try:
                    raw_df = pd.read_parquet(raw_path, columns=["date"] + fields)
                    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce").dt.normalize()
                    raw_df = raw_df.dropna(subset=["date"])
                    raw_df = raw_df[raw_df["date"].isin(date_to_idx)]
                    if not raw_df.empty:
                        raw_df["cal_idx"] = raw_df["date"].map(date_to_idx).astype(int)
                        raw_last_idx = int(raw_df["cal_idx"].max())
                        for field in fields:
                            if field in raw_df.columns:
                                fld = raw_df[["cal_idx", field]].dropna(subset=[field])
                                raw_maps[field] = dict(zip(fld["cal_idx"], fld[field]))
                except Exception:
                    raw_df = None
                    raw_maps = {}
                    raw_last_idx = None

            raw_close_map = raw_maps.get("close", {})
            target_close_end = min(close_end, cal_last_idx)
            if raw_last_idx is not None:
                target_close_end = min(target_close_end, raw_last_idx)

            inst_adj = adj_map.get(inst, {})

            if target_close_end < close_start:
                if not raw_close_map:
                    stats["unresolved_instruments"] += 1
                    continue

                close_start = min(raw_close_map)
                close_end = min(cal_last_idx, raw_last_idx)
                close_values_list = []
                for idx in range(close_start, close_end + 1):
                    raw_c = raw_close_map.get(idx, np.nan)
                    if inst_adj:
                        d = idx_to_date.get(idx)
                        ratio = inst_adj.get(d) if d else None
                        if not np.isnan(raw_c) and ratio is not None:
                            close_values_list.append(float(raw_c) * float(ratio))
                        else:
                            close_values_list.append(np.nan)
                    else:
                        close_values_list.append(float(raw_c) if not np.isnan(raw_c) else np.nan)
                close_values = np.asarray(close_values_list, dtype="<f4")
                if self._write_bin_file(close_path, close_start, close_values):
                    stats["truncated_files"] += 1
            elif target_close_end < original_close_end:
                keep = target_close_end - close_start + 1
                close_values = close_values[:keep]
                if self._write_bin_file(close_path, close_start, close_values):
                    stats["truncated_files"] += 1
                close_end = target_close_end

            if close_start != original_close_start or close_end != original_close_end:
                existing_meta["close"] = (close_start, close_end, close_values)
            elif "close" not in existing_meta:
                existing_meta["close"] = (close_start, close_end, close_values)

            if close_end < close_start:
                stats["unresolved_instruments"] += 1
                continue

            close_index = np.arange(close_start, close_end + 1)
            close_map = dict(zip(close_index.tolist(), close_values.tolist()))

            repaired = False
            for field in fields:
                if field == "close":
                    continue

                bin_path = inst_dir / f"{field}.day.bin"
                meta = existing_meta.get(field)
                existing_map = {}
                start_candidates = [close_start]
                if meta is not None:
                    field_start, field_end, field_values = meta
                    truncated_field_end = min(field_end, close_end, cal_last_idx)
                    if truncated_field_end >= field_start:
                        keep = truncated_field_end - field_start + 1
                        existing_map.update(
                            zip(
                                range(field_start, truncated_field_end + 1),
                                field_values[:keep].tolist(),
                            )
                        )
                        start_candidates.append(field_start)

                raw_field_map = raw_maps.get(field, {})
                if raw_field_map:
                    start_candidates.append(min(raw_field_map))

                if not start_candidates:
                    continue

                target_start = min(start_candidates)
                rebuilt_values = []
                for idx in range(target_start, close_end + 1):
                    derived = None
                    raw_val = raw_field_map.get(idx)
                    if field in price_fields and inst_adj:
                        # adj_factor 模式：所有价格字段用同一 adj_ratio
                        d = idx_to_date.get(idx)
                        ratio = inst_adj.get(d) if d else None
                        if raw_val is not None and ratio is not None and np.isfinite(raw_val):
                            derived = float(raw_val) * float(ratio)
                    elif field in price_fields:
                        # fallback：用 close bin / raw_close 推导
                        close_val = close_map.get(idx)
                        raw_close = raw_close_map.get(idx)
                        if (
                            raw_val is not None
                            and raw_close is not None
                            and close_val is not None
                            and np.isfinite(raw_val)
                            and np.isfinite(raw_close)
                            and np.isfinite(close_val)
                            and raw_close > 0
                        ):
                            derived = float(close_val) * float(raw_val) / float(raw_close)
                    elif raw_val is not None and np.isfinite(raw_val):
                        derived = float(raw_val)

                    if derived is None:
                        derived = existing_map.get(idx, np.nan)
                    rebuilt_values.append(derived)

                if self._write_bin_file(bin_path, target_start, rebuilt_values):
                    repaired = True

            if repaired or target_close_end < original_close_end:
                stats["repaired_instruments"] += 1
            elif raw_path.exists():
                stats["unresolved_instruments"] += 1

        logger.info(
            "Provider 修复完成: truncated_files=%s, repaired_instruments=%s, unresolved_instruments=%s",
            stats["truncated_files"],
            stats["repaired_instruments"],
            stats["unresolved_instruments"],
        )
        return stats

    def update_ohlcv_bins(self) -> dict:
        """增量更新 open/high/low/volume/amount 的 bin 文件

        优先使用 adj_factor 前复权：所有价格字段（open/high/low）使用同一个 adj_ratio。
        如果 adj_factor 不可用，回退到 splice-point ratio。
        volume/amount 始终保持原始值，不做复权缩放。
        """
        fields = ["open", "high", "low", "volume", "amount"]
        price_fields = {"open", "high", "low"}
        raw_dir = self.qlib_dir.parent / "raw_data"
        cal_file = self.qlib_dir / "calendars" / "day.txt"
        features_dir = self.qlib_dir / "features"

        if not raw_dir.exists() or not cal_file.exists() or not features_dir.exists():
            logger.warning("缺少 raw_data / 日历 / features 目录，跳过 OHLCV bin 更新")
            return {}

        date_to_idx, idx_to_date, cal_last_idx = self._load_calendar()

        adj_map = self._load_adj_ratio_map()
        use_adj = bool(adj_map)
        if not use_adj:
            logger.warning("adj_factor.parquet 不存在，回退到 splice-point ratio 更新 OHLCV")

        counts = {f: 0 for f in fields}
        raw_files = sorted(raw_dir.glob("*.parquet"))

        for raw_path in raw_files:
            inst = raw_path.stem
            inst_dir = features_dir / inst
            if not inst_dir.exists():
                continue

            try:
                df = pd.read_parquet(raw_path, columns=["date", "close"] + fields)
            except Exception:
                continue

            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            df = df.dropna(subset=["date"])
            df = df[df["date"].isin(date_to_idx)]
            if df.empty:
                continue
            df["cal_idx"] = df["date"].map(date_to_idx)

            # 构建 raw_close 查找表（用于 splice-point fallback）
            raw_close_map = {}
            if "close" in df.columns:
                close_data = df[["cal_idx", "close"]].dropna(subset=["close"])
                raw_close_map = dict(zip(close_data["cal_idx"], close_data["close"]))

            # 构建 adj_ratio 查找表（用于 adj_factor 模式）
            inst_adj = adj_map.get(inst, {}) if use_adj else {}

            # 读一次 close.bin（splice-point fallback 需要）
            close_bin_path = features_dir / inst / "close.day.bin"
            close_bin_data = None
            if close_bin_path.exists():
                cb = np.fromfile(close_bin_path, dtype="<f4")
                if len(cb) >= 2 and not np.isnan(cb[0]):
                    close_bin_data = cb

            for field in fields:
                bin_file = inst_dir / f"{field}.day.bin"
                if not bin_file.exists():
                    continue

                raw = np.fromfile(bin_file, dtype="<f4")
                if len(raw) < 2 or np.isnan(raw[0]):
                    continue

                bin_end_idx = int(raw[0]) + len(raw) - 2
                if bin_end_idx >= cal_last_idx:
                    continue

                field_data = df[["cal_idx", field]].dropna(subset=[field])
                if field_data.empty:
                    continue
                val_map = dict(zip(field_data["cal_idx"], field_data[field]))

                new_vals = []
                if use_adj and inst_adj and field in price_fields:
                    for idx in range(bin_end_idx + 1, cal_last_idx + 1):
                        d = idx_to_date.get(idx)
                        raw_val = val_map.get(idx)
                        ratio = inst_adj.get(d) if d else None
                        if raw_val is not None and ratio is not None and np.isfinite(raw_val):
                            new_vals.append(float(raw_val) * float(ratio))
                        else:
                            new_vals.append(np.nan)
                elif field in price_fields:
                    bin_last_val = float(raw[-1])
                    db_last_close = raw_close_map.get(bin_end_idx)
                    if db_last_close and db_last_close > 0 and close_bin_data is not None:
                        cb_offset = bin_end_idx - int(close_bin_data[0])
                        if 0 <= cb_offset < len(close_bin_data) - 1:
                            bin_last_close = float(close_bin_data[cb_offset + 1])
                            adj = bin_last_close / db_last_close
                        else:
                            adj = 1.0
                    else:
                        adj = 1.0
                    for idx in range(bin_end_idx + 1, cal_last_idx + 1):
                        v = val_map.get(idx)
                        if v is not None and np.isfinite(v):
                            new_vals.append(float(v) * adj)
                        else:
                            new_vals.append(np.nan)
                else:
                    # volume/amount：保持原始值
                    for idx in range(bin_end_idx + 1, cal_last_idx + 1):
                        v = val_map.get(idx)
                        new_vals.append(float(v) if v is not None and np.isfinite(v) else np.nan)

                if new_vals:
                    with open(bin_file, "ab") as fp:
                        np.array(new_vals, dtype="<f4").tofile(fp)
                    counts[field] += 1

        logger.info(
            f"OHLCV bin 更新完成 ({'adj_factor' if use_adj else 'splice-point'}): "
            + ", ".join(f"{f}={counts[f]}" for f in fields)
        )
        return counts

    def save(self, df: pd.DataFrame, filename: str = "factor_data.parquet"):
        """保存（增量模式）"""
        path = self.qlib_dir / filename

        if path.exists():
            existing = pd.read_parquet(path)
            existing["datetime"] = pd.to_datetime(existing["datetime"])
            df["datetime"] = pd.to_datetime(df["datetime"])

            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["datetime", "instrument"],
                keep="last"
            )
            combined = combined.sort_values(["instrument", "datetime"])
            combined.to_parquet(path, index=False)
            logger.info(f"增量更新: {path}, 新增 {len(df)} 行")
        else:
            df.to_parquet(path, index=False)
            logger.info(f"保存: {path}")

        return path


def main():
    converter = TushareToQlibConverter()
    df = converter.convert()

    if df is not None:
        converter.save(df)

        print("\n" + "="*60)
        print("字段统计:")
        print("="*60)
        for col in sorted(df.columns):
            n = df[col].notna().sum()
            pct = n / len(df) * 100
            print(f"  {col:30s}: {n:>10,} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
