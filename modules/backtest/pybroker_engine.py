"""
PyBroker 回测引擎
使用共享选股 + 仓位控制进行回测
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
from decimal import Decimal
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, Set, Optional

import pybroker
from pybroker import Strategy as PyBrokerStrategy, StrategyConfig, ExecContext, register_columns
from pybroker.common import FeeInfo, PositionMode, PriceType

from config.config import CONFIG
from core.qlib_init import init_qlib
from core.universe import filter_instruments
from core.position import MarketPositionController
from core.selection import get_name_map
from modules.backtest.base import BacktestResult, BacktestEngine
from modules.backtest.qlib_engine import _load_backtest_return_frame
from utils.logger import setup_logger


# ============================================================
# 闭包工厂：消除全局变量
# ============================================================

def make_exec_fn(
    date_to_selection: Dict,
    rebalance_dates: Set,
    controller: Optional[MarketPositionController] = None,
) -> Callable:
    """创建 PyBroker 执行函数闭包"""
    import bisect

    # 预处理：转换为 numpy datetime64 避免重复 pd.Timestamp 开销
    sorted_rebal_dates = sorted(rebalance_dates)
    rebal_set = set(pd.Timestamp(d).normalize() for d in rebalance_dates)

    # 预构建每个日期的选股 frozenset，避免重复 set 查找
    norm_date_to_selection = {}
    for d, syms in date_to_selection.items():
        norm_date_to_selection[pd.Timestamp(d).normalize()] = frozenset(syms)

    norm_sorted_rebal = [pd.Timestamp(d).normalize() for d in sorted_rebal_dates]

    # 缓存仓位控制器结果（同一天只需调用一次）
    alloc_cache = {}

    def exec_fn(ctx: ExecContext):
        dt = pd.Timestamp(ctx.dt).normalize()

        # 找到当前持仓期对应的最近调仓日（二分查找）
        idx = bisect.bisect_right(norm_sorted_rebal, dt) - 1
        if idx < 0:
            return
        applicable_date = norm_sorted_rebal[idx]

        selected = norm_date_to_selection.get(applicable_date)
        if selected is None:
            return

        topk = len(selected)
        is_rebal = (dt in rebal_set)

        # 缓存仓位控制器结果
        if dt not in alloc_cache:
            if controller is not None:
                alloc = controller.get_allocation(dt, is_rebalance_day=is_rebal)
                alloc_cache[dt] = alloc.stock_pct
            else:
                alloc_cache[dt] = 1.0
        stock_pct = alloc_cache[dt]

        target_pct = stock_pct / topk

        pos = ctx.long_pos()

        if ctx.symbol in selected:
            target_shares = int(ctx.calc_target_shares(target_pct))
            current_shares = int(pos.shares) if pos else 0
            diff = target_shares - current_shares
            # 偏差超过目标仓位20%才交易，避免微小再平衡
            threshold = max(target_shares * 0.2, 100)
            if diff > threshold:
                ctx.buy_shares = diff
                ctx.buy_fill_price = PriceType.CLOSE
            elif diff < -threshold:
                ctx.sell_shares = abs(diff)
                ctx.sell_fill_price = PriceType.CLOSE
        else:
            if pos is not None:
                ctx.sell_shares = pos.shares
                ctx.sell_fill_price = PriceType.CLOSE

    return exec_fn


# ============================================================
# 自定义手续费
# ============================================================

def _default_trading_cost() -> Dict[str, float]:
    """默认交易成本，供无策略对象时回退使用。"""
    buy_commission_rate = float(
        CONFIG.get("open_cost", CONFIG.get("trading.cost.open_cost", 0.0003))
    )
    close_cost = float(
        CONFIG.get("close_cost", CONFIG.get("trading.cost.close_cost", 0.0013))
    )
    sell_stamp_tax_rate = 0.001
    sell_commission_rate = max(close_cost - sell_stamp_tax_rate, 0.0)
    min_cost = float(CONFIG.get("min_cost", CONFIG.get("trading.cost.min_cost", 5.0)))
    return {
        "open_cost": buy_commission_rate,
        "close_cost": close_cost,
        "buy_commission_rate": buy_commission_rate,
        "sell_commission_rate": sell_commission_rate,
        "sell_stamp_tax_rate": sell_stamp_tax_rate,
        "min_buy_commission": min_cost,
        "min_sell_commission": min_cost,
    }


def make_fee_fn(trading_cost: Optional[Dict[str, float]] = None) -> Callable[[FeeInfo], Decimal]:
    """按策略 trading_cost 构造手续费函数。"""
    cost = dict(_default_trading_cost())
    if trading_cost:
        cost.update(trading_cost)

    buy_commission_rate = Decimal(str(cost.get("buy_commission_rate", cost.get("open_cost", 0.0003))))
    sell_commission_rate = Decimal(str(cost.get("sell_commission_rate", 0.0003)))
    sell_stamp_tax_rate = Decimal(str(cost.get("sell_stamp_tax_rate", 0.001)))
    min_buy_commission = Decimal(str(cost.get("min_buy_commission", 5.0)))
    min_sell_commission = Decimal(str(cost.get("min_sell_commission", 5.0)))
    execution_rate = Decimal(
        str(
            (float(cost.get("slippage_bps", 0.0) or 0.0) + float(cost.get("impact_bps", 0.0) or 0.0))
            / 10000
        )
    )

    def fee_fn(fee_info: FeeInfo) -> Decimal:
        amount = fee_info.fill_price * fee_info.shares
        execution_cost = amount * execution_rate

        if fee_info.order_type == "buy":
            commission = amount * buy_commission_rate
            if commission < min_buy_commission:
                commission = min_buy_commission
            return commission + execution_cost

        commission = amount * sell_commission_rate
        if commission < min_sell_commission:
            commission = min_sell_commission
        stamp_tax = amount * sell_stamp_tax_rate
        return commission + stamp_tax + execution_cost

    return fee_fn


def _build_close_only_price_frame(df_close: pd.DataFrame) -> pd.DataFrame:
    """为 PyBroker 构造 close-to-close 口径的数据源。"""
    if df_close.empty:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

    # PyBroker 要求存在 OHLC 列；当前官方口径只在收盘价成交，因此用 close 填充其余价格列。
    work = df_close.copy()
    if "close" not in work.columns:
        raise ValueError("df_close 必须包含 close 列")
    work = work.groupby(level="instrument").ffill().dropna(subset=["close"])

    df_pybroker = work.reset_index().rename(columns={"datetime": "date", "instrument": "symbol"})
    df_pybroker["date"] = pd.to_datetime(df_pybroker["date"])
    df_pybroker["open"] = df_pybroker["close"]
    df_pybroker["high"] = df_pybroker["close"]
    df_pybroker["low"] = df_pybroker["close"]
    if "volume" not in df_pybroker.columns:
        df_pybroker["volume"] = 0.0
    df_pybroker = df_pybroker[["symbol", "date", "open", "high", "low", "close", "volume"]]
    return df_pybroker.sort_values(["symbol", "date"]).reset_index(drop=True)


# ============================================================
# PyBroker 回测引擎
# ============================================================

class PyBrokerBacktestEngine(BacktestEngine):
    """PyBroker 回测引擎"""

    def run(self, strategy=None) -> BacktestResult:
        """执行回测"""
        logger = setup_logger("pybroker", level="INFO")

        strategy_name = strategy.name if strategy else "default"
        strategy_slug = strategy.artifact_slug() if strategy else "default"
        logger.info(f"=== PyBroker 回测 [{strategy_name}] ===")

        # 初始化 Qlib
        init_qlib()

        # 加载选股列表
        logger.info("[1/4] 加载选股列表...")
        date_to_selection, rebalance_dates, controller, topk = self._prepare(strategy)
        all_selected = set()
        for s in date_to_selection.values():
            all_selected.update(s)
        logger.info(f"调仓截面数: {len(date_to_selection)}, 曾选中股票: {len(all_selected)}")

        # 加载行情数据（只加载选中股票，减少数据量）
        logger.info("[2/4] 加载行情数据...")

        df_prices, _ = _load_backtest_return_frame(
            list(all_selected),
            CONFIG.get("start_date", "2019-01-01"),
            CONFIG.get("end_date", "2026-02-26"),
        )
        if "volume" not in df_prices.columns:
            df_prices["volume"] = 0.0
        df_pybroker = _build_close_only_price_frame(df_prices)

        logger.info(f"数据行数: {df_pybroker.shape[0]:,}")

        # 运行 PyBroker 回测
        logger.info("[3/4] 运行 PyBroker 回测...")
        pybroker.disable_logging()
        pybroker.disable_progress_bar()

        initial_cash = CONFIG.get("trading.capital.initial", 500000)

        trading_cost = (
            getattr(strategy, "trading_cost", None)
            if strategy is not None
            else _default_trading_cost()
        )

        config = StrategyConfig(
            initial_cash=initial_cash,
            fee_mode=make_fee_fn(trading_cost),
            buy_delay=1,
            sell_delay=1,
            max_long_positions=topk,
            position_mode=PositionMode.LONG_ONLY,
            bars_per_year=252,
            exit_on_last_bar=True,
            exit_sell_fill_price=PriceType.CLOSE,
            exit_cover_fill_price=PriceType.CLOSE,
        )

        pb_strategy = PyBrokerStrategy(
            data_source=df_pybroker,
            start_date=CONFIG.get("start_date", "2019-01-01"),
            end_date=CONFIG.get("end_date", "2026-02-26"),
            config=config,
        )

        exec_fn = make_exec_fn(date_to_selection, rebalance_dates, controller)

        pb_strategy.add_execution(
            fn=exec_fn,
            symbols=list(all_selected),
        )

        result = pb_strategy.backtest(
            start_date=CONFIG.get("start_date", "2019-01-01"),
            end_date=CONFIG.get("end_date", "2026-02-26"),
            calc_bootstrap=False,
            disable_parallel=True,
        )

        # 转换为统一结果
        return self._build_result(
            result,
            initial_cash,
            strategy_name=strategy_name,
            strategy_slug=strategy_slug,
        )

    def _build_result(
        self,
        pybroker_result,
        initial_cash: float,
        strategy_name: str = "default",
        strategy_slug: str = "default",
    ) -> BacktestResult:
        """将 PyBroker 结果转换为统一 BacktestResult"""
        logger = setup_logger("pybroker", level="INFO")

        port_df = pybroker_result.portfolio
        if port_df is None or port_df.empty:
            logger.warning("无组合数据")
            return BacktestResult(
                daily_returns=pd.Series(dtype=float),
                portfolio_value=pd.Series(dtype=float),
            )

        equity_col = "equity" if "equity" in port_df.columns else port_df.columns[0]
        equity = port_df[equity_col].dropna()

        daily_ret = equity.pct_change().dropna()
        portfolio_value = equity / initial_cash  # 归一化为从1开始

        # 保存结果
        results_dir = Path(CONFIG.get("paths.results", "./results"))
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 每日组合汇总
        results_file = results_dir / f"pybroker_{strategy_slug}_{timestamp}.csv"
        pybroker_result.portfolio.to_csv(results_file)

        name_map = get_name_map()

        # 2. 每日持仓明细（含股票名称）
        # positions 的 index 是 [symbol, date]，需要 reset_index 才能访问 symbol
        if pybroker_result.positions is not None and not pybroker_result.positions.empty:
            pos_df = pybroker_result.positions.reset_index()
            pos_df["name"] = pos_df["symbol"].map(name_map).fillna("")
            cols = ["symbol", "name", "date"] + [
                c for c in pos_df.columns if c not in ("symbol", "name", "date")
            ]
            pos_file = results_dir / f"positions_{strategy_slug}_{timestamp}.csv"
            pos_df[cols].to_csv(pos_file, index=False)
            logger.info(f"[OK] 持仓明细: {pos_file}")

        # 3. 每笔交易明细（含股票名称）
        # trades 的 index 是 id，symbol 是普通列
        if pybroker_result.trades is not None and not pybroker_result.trades.empty:
            trades_df = pybroker_result.trades.copy()
            trades_df["name"] = trades_df["symbol"].map(name_map).fillna("")
            cols = ["symbol", "name"] + [
                c for c in trades_df.columns if c not in ("symbol", "name")
            ]
            trades_file = results_dir / f"trades_{strategy_slug}_{timestamp}.csv"
            trades_df[cols].to_csv(trades_file)
            logger.info(f"[OK] 交易明细: {trades_file}")

        return BacktestResult(
            daily_returns=daily_ret,
            portfolio_value=portfolio_value,
            metadata={
                "results_file": str(results_file),
                "strategy_name": strategy_name,
            },
        )


# ============================================================
# 主函数
# ============================================================

def main(strategy=None):
    """主函数"""
    engine = PyBrokerBacktestEngine()
    result = engine.run(strategy=strategy)
    initial_capital = CONFIG.get("trading.capital.initial", 500000)
    result.print_summary(initial_capital)

    if result.metadata.get("results_file"):
        print(f"\n  [OK] 结果已保存: {result.metadata['results_file']}")

    return result


if __name__ == "__main__":
    main()
