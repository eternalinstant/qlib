"""
Tushare Pro 数据下载器 (多线程版)
下载因子所需的财务和市场数据
"""

import os
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TushareDownloader:
    """Tushare Pro 数据下载器 (多线程版)"""

    MAX_WORKERS = 8  # 并发线程数

    def __init__(self, token: str = None, data_dir: str = None):
        self.token = token or os.environ.get("TUSHARE_TOKEN")
        self.data_dir = Path(data_dir or "~/code/qlib/data/tushare").expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._counter = 0
        self._total = 0

        if not self.token:
            raise ValueError("缺少 Tushare token，请传入 token 或设置环境变量 TUSHARE_TOKEN")

        try:
            import tushare as ts
            self.pro = ts.pro_api(self.token)
            logger.info("Tushare Pro 初始化成功")
        except ImportError:
            raise ImportError("请先安装 tushare: pip install tushare")
        except Exception as e:
            raise Exception(f"Tushare 初始化失败: {e}")

    def get_all_stocks(self) -> List[str]:
        """获取所有A股股票代码"""
        logger.info("获取股票列表...")
        df = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,list_date')
        df = df[~df['ts_code'].str.startswith(('8', '4', '68'))]
        df['list_date'] = pd.to_datetime(df['list_date'])
        cutoff_date = datetime.now() - pd.Timedelta(days=60)
        df = df[df['list_date'] < cutoff_date]
        logger.info(f"共获取 {len(df)} 只股票")
        return df['ts_code'].tolist()

    def _update_progress(self):
        """更新进度"""
        with self._lock:
            self._counter += 1
            if self._counter % 100 == 0:
                logger.info(f"进度: {self._counter}/{self._total}")

    def download_daily_basic(self, start_date: str = "20160101", end_date: str = None):
        """下载每日基本面数据 (换手率、市值等)"""
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        output_file = self.data_dir / "daily_basic.parquet"

        logger.info(f"下载每日基本面数据: {start_date} ~ {end_date}")

        stocks = self.get_all_stocks()
        self._counter = 0
        self._total = len(stocks)
        all_data = []
        failed = []

        def download_one(stock):
            try:
                df = self.pro.daily_basic(
                    ts_code=stock,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,trade_date,close,turnover_rate,turnover_rate_f,'
                           'volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,'
                           'total_mv,circ_mv,free_mv,total_share,float_share,free_share'
                )
                self._update_progress()
                return df if df is not None and len(df) > 0 else None
            except Exception as e:
                with self._lock:
                    failed.append(stock)
                return None

        logger.info(f"使用 {self.MAX_WORKERS} 线程并发下载...")

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {executor.submit(download_one, stock): stock for stock in stocks}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_data.append(result)

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result.to_parquet(output_file, index=False)
            logger.info(f"每日基本面数据已保存: {output_file}")
            logger.info(f"共 {len(result)} 条记录, 失败 {len(failed)} 只")
            return result
        return None

    def download_income(self, start_date: str = "20160101", end_date: str = None):
        """下载利润表数据"""
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        output_file = self.data_dir / "income.parquet"

        logger.info(f"下载利润表数据: {start_date} ~ {end_date}")

        stocks = self.get_all_stocks()
        self._counter = 0
        self._total = len(stocks)
        all_data = []

        def download_one(stock):
            try:
                df = self.pro.income(
                    ts_code=stock,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,ann_date,f_ann_date,end_date,report_type,'
                           'total_revenue,revenue,n_income,n_income_attr_p,'
                           'oper_cost,total_cogs,admin_exp,fin_exp,'
                           'sell_exp,operate_profit,total_profit'
                )
                self._update_progress()
                return df if df is not None and len(df) > 0 else None
            except:
                return None

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {executor.submit(download_one, stock): stock for stock in stocks}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_data.append(result)

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result.to_parquet(output_file, index=False)
            logger.info(f"利润表数据已保存: {output_file}, 共 {len(result)} 条")
            return result
        return None

    def download_balancesheet(self, start_date: str = "20160101", end_date: str = None):
        """下载资产负债表数据"""
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        output_file = self.data_dir / "balancesheet.parquet"

        logger.info(f"下载资产负债表数据: {start_date} ~ {end_date}")

        stocks = self.get_all_stocks()
        self._counter = 0
        self._total = len(stocks)
        all_data = []

        def download_one(stock):
            try:
                df = self.pro.balancesheet(
                    ts_code=stock,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,ann_date,end_date,report_type,'
                           'total_assets,total_liab,total_hldr_eqy_exc_min_int,'
                           'total_hldr_eqy_inc_min_int,cap_rese,'
                           'surplus_rese,undistr_porfit,money_cap,'
                           'accounts_receiv,inventory,total_cur_assets,'
                           'total_nca,total_cur_liab,total_ncl'
                )
                self._update_progress()
                return df if df is not None and len(df) > 0 else None
            except:
                return None

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {executor.submit(download_one, stock): stock for stock in stocks}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_data.append(result)

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result.to_parquet(output_file, index=False)
            logger.info(f"资产负债表数据已保存: {output_file}, 共 {len(result)} 条")
            return result
        return None

    def download_cashflow(self, start_date: str = "20160101", end_date: str = None):
        """下载现金流量表数据"""
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        output_file = self.data_dir / "cashflow.parquet"

        logger.info(f"下载现金流量表数据: {start_date} ~ {end_date}")

        stocks = self.get_all_stocks()
        self._counter = 0
        self._total = len(stocks)
        all_data = []

        def download_one(stock):
            try:
                df = self.pro.cashflow(
                    ts_code=stock,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,ann_date,end_date,report_type,'
                           'n_cashflow_act,n_cashflow_inv_act,n_cash_flows_fnc_act,'
                           'c_fr_sale_sg,c_pay_for_tax,free_cashflow'
                )
                self._update_progress()
                return df if df is not None and len(df) > 0 else None
            except:
                return None

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {executor.submit(download_one, stock): stock for stock in stocks}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_data.append(result)

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result.to_parquet(output_file, index=False)
            logger.info(f"现金流量表数据已保存: {output_file}, 共 {len(result)} 条")
            return result
        return None

    def download_financial_indicator(self, start_date: str = "20160101", end_date: str = None):
        """下载财务指标数据 (ROE, ROA 等)"""
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        output_file = self.data_dir / "fina_indicator.parquet"

        logger.info(f"下载财务指标数据: {start_date} ~ {end_date}")

        stocks = self.get_all_stocks()
        self._counter = 0
        self._total = len(stocks)
        all_data = []

        def download_one(stock):
            try:
                df = self.pro.fina_indicator(
                    ts_code=stock,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,ann_date,end_date,'
                           'roe,roe_dt,roa,npta,'
                           'profit_dedt,op_yoy,ebt_yoy,'
                           'current_ratio,quick_ratio,cash_ratio,'
                           'ar_turn,inv_turn,ca_turn,'
                           'debt_to_assets,assets_to_eq,'
                           'ebit,ebitda,fcff'
                )
                self._update_progress()
                return df if df is not None and len(df) > 0 else None
            except:
                return None

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {executor.submit(download_one, stock): stock for stock in stocks}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_data.append(result)

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result.to_parquet(output_file, index=False)
            logger.info(f"财务指标数据已保存: {output_file}, 共 {len(result)} 条")
            return result
        return None

    def download_index_daily(self, start_date: str = "20160101", end_date: str = None):
        """下载指数日线数据 (沪深300, 中证500, 上证指数等)"""
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        output_file = self.data_dir / "index_daily.parquet"

        logger.info(f"下载指数日线数据: {start_date} ~ {end_date}")

        # 主要指数代码
        indices = [
            "000001.SH",  # 上证指数
            "000300.SH",  # 沪深300
            "000905.SH",  # 中证500
            "000852.SH",  # 中证1000
            "399001.SZ",  # 深证成指
            "399006.SZ",  # 创业板指
        ]

        all_data = []
        for idx in indices:
            try:
                df = self.pro.index_daily(
                    ts_code=idx,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,trade_date,open,high,low,close,vol,amount,pct_chg'
                )
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    logger.info(f"  {idx}: {len(df)} 条")
            except Exception as e:
                logger.warning(f"  {idx}: 下载失败 - {e}")

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result.to_parquet(output_file, index=False)
            logger.info(f"指数数据已保存: {output_file}, 共 {len(result)} 条")
            return result
        return None

    def download_index_weight(self, start_date: str = "20160101", end_date: str = None):
        """下载指数成分股权重"""
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        output_file = self.data_dir / "index_weight.parquet"

        logger.info(f"下载指数成分股权重")

        indices = ["000300.SH", "000905.SH", "000852.SH"]  # 沪深300, 中证500, 中证1000
        all_data = []

        for idx in indices:
            try:
                df = self.pro.index_weight(
                    index_code=idx,
                    start_date=start_date,
                    end_date=end_date
                )
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    logger.info(f"  {idx}: {len(df)} 条")
            except Exception as e:
                logger.warning(f"  {idx}: 下载失败 - {e}")

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result.to_parquet(output_file, index=False)
            logger.info(f"指数权重已保存: {output_file}, 共 {len(result)} 条")
            return result
        return None

    def download_namechange(self, start_date: str = "20100101", end_date: str = None):
        """下载历史名称变更数据。"""
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        output_file = self.data_dir / "namechange.parquet"

        logger.info(f"下载历史名称变更: {start_date} ~ {end_date}")

        all_data = []
        start_ts = datetime.strptime(start_date, "%Y%m%d")
        end_ts = datetime.strptime(end_date, "%Y%m%d")
        cursor = start_ts
        while cursor <= end_ts:
            win_end = min(cursor.replace(year=cursor.year + 1) - pd.Timedelta(days=1), end_ts)
            try:
                df = self.pro.namechange(
                    start_date=cursor.strftime("%Y%m%d"),
                    end_date=win_end.strftime("%Y%m%d"),
                )
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    logger.info(f"  {cursor.strftime('%Y%m%d')}~{win_end.strftime('%Y%m%d')}: {len(df)} 条")
            except Exception as e:
                logger.warning(
                    f"  {cursor.strftime('%Y%m%d')}~{win_end.strftime('%Y%m%d')}: 下载失败 - {e}"
                )
            cursor = win_end + pd.Timedelta(days=1)

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.drop_duplicates()
            result.to_parquet(output_file, index=False)
            logger.info(f"名称变更数据已保存: {output_file}, 共 {len(result)} 条")
            return result
        return None

    def download_adj_factor(self, start_date: str = "20160101", end_date: str = None):
        """下载复权因子数据 (adj_factor)"""
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        output_file = self.data_dir / "adj_factor.parquet"

        logger.info(f"下载复权因子数据: {start_date} ~ {end_date}")

        stocks = self.get_all_stocks()
        self._counter = 0
        self._total = len(stocks)
        all_data = []
        failed = []

        def download_one(stock):
            try:
                df = self.pro.adj_factor(
                    ts_code=stock,
                    start_date=start_date,
                    end_date=end_date,
                )
                self._update_progress()
                return df if df is not None and len(df) > 0 else None
            except Exception as e:
                with self._lock:
                    failed.append(stock)
                return None

        logger.info(f"使用 {self.MAX_WORKERS} 线程并发下载...")

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {executor.submit(download_one, stock): stock for stock in stocks}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_data.append(result)

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            # 只保留必要列
            for col in result.columns:
                if col not in ("ts_code", "trade_date", "adj_factor"):
                    result = result.drop(columns=[col])
            result.to_parquet(output_file, index=False)
            logger.info(f"复权因子数据已保存: {output_file}")
            logger.info(f"共 {len(result)} 条记录, 失败 {len(failed)} 只")
            return result
        return None

    def download_daily_quotes(self, ts_codes: List[str] = None, start_date: str = "20160101", end_date: str = None):
        """下载日线行情数据 (OHLCV)"""
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        output_file = self.data_dir / "daily_quotes.parquet"

        if ts_codes is None:
            ts_codes = self.get_all_stocks()

        logger.info(f"下载日线行情: {start_date} ~ {end_date}, {len(ts_codes)} 只股票")

        self._counter = 0
        self._total = len(ts_codes)
        all_data = []

        def download_one(stock):
            try:
                df = self.pro.daily(
                    ts_code=stock,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,trade_date,open,high,low,close,pre_close,vol,amount,pct_chg'
                )
                self._update_progress()
                return df if df is not None and len(df) > 0 else None
            except:
                return None

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {executor.submit(download_one, stock): stock for stock in ts_codes}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_data.append(result)

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result.to_parquet(output_file, index=False)
            logger.info(f"日线行情已保存: {output_file}, 共 {len(result)} 条")
            return result
        return None

    def download_all(self, start_date: str = "20160101", end_date: str = None):
        """下载所有数据"""
        logger.info("=" * 50)
        logger.info(f"开始下载 (并发数: {self.MAX_WORKERS})")
        logger.info(f"时间范围: {start_date} ~ {end_date or '今天'}")
        logger.info("=" * 50)

        results = {}
        results['daily_basic'] = self.download_daily_basic(start_date, end_date)
        results['adj_factor'] = self.download_adj_factor(start_date, end_date)
        results['income'] = self.download_income(start_date, end_date)
        results['balancesheet'] = self.download_balancesheet(start_date, end_date)
        results['cashflow'] = self.download_cashflow(start_date, end_date)
        results['fina_indicator'] = self.download_financial_indicator(start_date, end_date)
        results['index_daily'] = self.download_index_daily(start_date, end_date)
        results['index_weight'] = self.download_index_weight(start_date, end_date)
        results['namechange'] = self.download_namechange(start_date, end_date)

        logger.info("=" * 50)
        logger.info("下载完成!")
        logger.info(f"数据目录: {self.data_dir}")
        logger.info("=" * 50)

        return results

    def load_data(self, name: str) -> Optional[pd.DataFrame]:
        """加载已下载的数据"""
        file_path = self.data_dir / f"{name}.parquet"
        if file_path.exists():
            return pd.read_parquet(file_path)
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tushare Pro 数据下载器")
    parser.add_argument("--start", default="20160101", help="开始日期")
    parser.add_argument("--end", default=None, help="结束日期")
    parser.add_argument("--token", default=None, help="Tushare Token")
    parser.add_argument("--workers", type=int, default=8, help="并发线程数")
    parser.add_argument("--type", default="all",
                        choices=["all", "daily_basic", "adj_factor", "income", "balancesheet", "cashflow",
                                 "fina_indicator", "index_daily", "index_weight", "namechange"],
                        help="下载数据类型")

    args = parser.parse_args()

    downloader = TushareDownloader(token=args.token)
    downloader.MAX_WORKERS = args.workers

    # 方法名映射
    method_map = {
        "all": "download_all",
        "daily_basic": "download_daily_basic",
        "adj_factor": "download_adj_factor",
        "income": "download_income",
        "balancesheet": "download_balancesheet",
        "cashflow": "download_cashflow",
        "fina_indicator": "download_financial_indicator",
        "index_daily": "download_index_daily",
        "index_weight": "download_index_weight",
        "namechange": "download_namechange",
    }

    method_name = method_map.get(args.type)
    if method_name:
        getattr(downloader, method_name)(args.start, args.end)


if __name__ == "__main__":
    main()
