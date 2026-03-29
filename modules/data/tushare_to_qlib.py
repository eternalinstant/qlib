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


class TushareToQlibConverter:
    """Tushare 数据转 Qlib 格式转换器"""

    def __init__(self, tushare_dir: str = None, qlib_dir: str = None):
        self.tushare_dir = Path(tushare_dir or "~/code/qlib/data/tushare").expanduser()
        self.qlib_dir = Path(qlib_dir or "~/code/qlib/data/qlib_data/cn_data").expanduser()

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
        """转换并合并所有数据"""
        # 1. 加载每日基本面数据 (主表)
        daily = self.load_tushare_data('daily_basic')
        if daily is None:
            return None

        # 标准化股票代码
        daily['instrument'] = daily['ts_code'].str.lower().str.replace('.', '', regex=False)
        daily['datetime'] = pd.to_datetime(daily['trade_date'], format='%Y%m%d')

        # 选择列
        cols = ['instrument', 'datetime', 'turnover_rate', 'turnover_rate_f',
                'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
                'total_mv', 'circ_mv', 'free_mv', 'dv_ratio', 'dv_ttm']
        result = daily[[c for c in cols if c in daily.columns]].copy()

        # 2. 加载财务指标
        fina = self.load_tushare_data('fina_indicator')
        if fina is not None:
            fina['instrument'] = fina['ts_code'].str.lower().str.replace('.', '', regex=False)
            fina['datetime'] = pd.to_datetime(fina['ann_date'], format='%Y%m%d', errors='coerce')
            fina = fina.dropna(subset=['datetime'])
            # 公告日+1天延迟：盘后公告当天不可用，次日才能使用
            fina['datetime'] = fina['datetime'] + pd.Timedelta(days=1)
            # 不按 report_type 过滤：保留所有合并报表（含季报/半年报/年报）
            # 年报/季报由 end_date 区分，ann_date 前向填充可自然处理更新频率

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

            # 前向填充到每日
            result = self._forward_fill(result, fina, 'fina')
            logger.info(f"合并财务指标完成")

        # 3. 加载利润表
        income = self.load_tushare_data('income')
        if income is not None:
            income['instrument'] = income['ts_code'].str.lower().str.replace('.', '', regex=False)
            income['datetime'] = pd.to_datetime(income['ann_date'], format='%Y%m%d', errors='coerce')
            income = income.dropna(subset=['datetime'])
            income['datetime'] = income['datetime'] + pd.Timedelta(days=1)
            # 不按 report_type 过滤：保留所有合并报表（含季报/半年报/年报）

            inc_cols = ['instrument', 'datetime', 'total_revenue', 'revenue',
                        'n_income', 'n_income_attr_p', 'operate_profit']
            income = income[[c for c in inc_cols if c in income.columns]].copy()
            income = income.rename(columns={
                'total_revenue': 'total_revenue_inc', 'revenue': 'revenue_inc',
                'n_income': 'n_income_inc', 'n_income_attr_p': 'n_income_attr_p_inc',
                'operate_profit': 'operate_profit_inc'
            })

            result = self._forward_fill(result, income, 'inc')
            logger.info(f"合并利润表完成")

        # 4. 加载现金流量表
        cashflow = self.load_tushare_data('cashflow')
        if cashflow is not None:
            cashflow['instrument'] = cashflow['ts_code'].str.lower().str.replace('.', '', regex=False)
            cashflow['datetime'] = pd.to_datetime(cashflow['ann_date'], format='%Y%m%d', errors='coerce')
            cashflow = cashflow.dropna(subset=['datetime'])
            cashflow['datetime'] = cashflow['datetime'] + pd.Timedelta(days=1)

            cf_cols = ['instrument', 'datetime', 'n_cashflow_act']
            cashflow = cashflow[[c for c in cf_cols if c in cashflow.columns]].copy()

            result = self._forward_fill(result, cashflow, 'cf')
            logger.info("合并现金流量表完成")

        # 5. 加载资产负债表
        balance = self.load_tushare_data('balancesheet')
        if balance is not None:
            balance['instrument'] = balance['ts_code'].str.lower().str.replace('.', '', regex=False)
            balance['datetime'] = pd.to_datetime(balance['ann_date'], format='%Y%m%d', errors='coerce')
            balance = balance.dropna(subset=['datetime'])
            balance['datetime'] = balance['datetime'] + pd.Timedelta(days=1)

            bs_cols = ['instrument', 'datetime', 'surplus_rese', 'undistr_porfit',
                       'total_liab', 'money_cap']
            balance = balance[[c for c in bs_cols if c in balance.columns]].copy()

            result = self._forward_fill(result, balance, 'bs')
            logger.info("合并资产负债表完成")

        # 6. 计算衍生指标
        result = self._calculate_derived(result)

        logger.info(f"最终数据: {len(result):,} 条, {len(result.columns)} 列")
        return result

    def _forward_fill(self, daily: pd.DataFrame, quarterly: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """将季度财务数据前向填充到每日数据"""
        # 获取所有交易日
        all_dates = daily[['datetime']].drop_duplicates().sort_values('datetime')

        # 获取所有股票
        all_instruments = daily[['instrument']].drop_duplicates()

        # 创建完整的 日期×股票 笛卡尔积
        full_index = all_dates.merge(all_instruments, how='cross')

        # 合并每日数据
        result = full_index.merge(daily, on=['datetime', 'instrument'], how='left')

        # 合并季度数据
        result = result.merge(quarterly, on=['instrument', 'datetime'], how='left')

        # 按股票分组，前向填充财务数据
        fina_cols = [c for c in quarterly.columns if c not in ['instrument', 'datetime']]
        for col in fina_cols:
            result[col] = result.groupby('instrument')[col].ffill()

        return result

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
        """从 daily_basic.parquet 的 close 列更新各股 close.day.bin

        使用 splice-point ratio（断点比例）保持调整价格连续性：
          adj_ratio = bin最后收盘 / daily_basic同日收盘
        新增数据 = daily_basic新收盘 × adj_ratio

        Returns
        -------
        int : 成功更新的股票数量
        """
        db_path = self.tushare_dir / "daily_basic.parquet"
        cal_file = self.qlib_dir / "calendars" / "day.txt"
        features_dir = self.qlib_dir / "features"

        if not db_path.exists() or not cal_file.exists() or not features_dir.exists():
            logger.warning("缺少必要文件，跳过 close bin 更新")
            return 0

        # 读取日历
        cal = pd.read_csv(cal_file, header=None, names=["date"], parse_dates=["date"])
        cal_dates = cal["date"].dt.normalize()
        date_to_idx = {d: i for i, d in enumerate(cal_dates)}
        cal_last_idx = len(cal) - 1

        # 读取 daily_basic 的 close，只保留日历内的日期
        db = pd.read_parquet(db_path, columns=["ts_code", "trade_date", "close"])
        db["date"] = pd.to_datetime(db["trade_date"], format="%Y%m%d").dt.normalize()
        db = db[db["date"].isin(date_to_idx)]
        db["cal_idx"] = db["date"].map(date_to_idx)
        # instrument 格式：000001.SZ -> sz000001
        db["instrument"] = db["ts_code"].apply(
            lambda x: x.split(".")[1].lower() + x.split(".")[0] if "." in x else x.lower()
        )
        db = db[["instrument", "cal_idx", "close"]].dropna(subset=["close"])

        # 按股票分组，建立 {cal_idx: close} 查找表
        inst_data = {inst: dict(zip(g["cal_idx"], g["close"]))
                     for inst, g in db.groupby("instrument")}

        updated = 0
        for inst, close_map in inst_data.items():
            bin_file = features_dir / inst / "close.day.bin"
            if not bin_file.exists():
                continue

            raw = np.fromfile(bin_file, dtype="<f4")
            if len(raw) < 2 or np.isnan(raw[0]):
                continue

            bin_end_idx = int(raw[0]) + len(raw) - 2
            if bin_end_idx >= cal_last_idx:
                continue  # 已是最新

            # splice-point 调整比例
            bin_last_close = float(raw[-1])
            db_last_close = close_map.get(bin_end_idx, None)
            if db_last_close and db_last_close > 0:
                adj = bin_last_close / db_last_close
            else:
                adj = 1.0

            # 收集 bin_end_idx+1 到 cal_last_idx 的新值
            new_vals = []
            for idx in range(bin_end_idx + 1, cal_last_idx + 1):
                v = close_map.get(idx, None)
                new_vals.append(float(v) * adj if v and v > 0 else np.nan)

            if not new_vals:
                continue

            with open(bin_file, "ab") as fp:
                np.array(new_vals, dtype="<f4").tofile(fp)
            updated += 1

        logger.info(f"close.day.bin 已更新 {updated} 只股票")
        return updated

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
