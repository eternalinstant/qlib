"""
规则驱动回测引擎
支持逐日 on_bar() 事件循环，用于海龟/金字塔等规则类策略
"""
from typing import Dict, List
import pandas as pd

from modules.backtest.base import BacktestResult, BacktestEngine
from modules.backtest.common import load_trade_calendar, get_limit_prices, compute_trade_cost
from core.data_provider import DataProvider
from core.qlib_data_provider import QlibDataProvider
from strategies.base import RuleStrategy, PositionState, StrategySignal


class RuleBasedEngine(BacktestEngine):

    def run(self, strategy: RuleStrategy) -> BacktestResult:
        config = self.config or {}
        initial_capital = float(config.get("initial_capital", 1_000_000))
        start_date = config.get("start_date", "2019-01-01")
        end_date = config.get("end_date", "2026-05-22")
        universe_name = strategy.config.get("selection", {}).get("universe", "csi800")

        dp: DataProvider = QlibDataProvider()
        trade_dates = load_trade_calendar(start_date, end_date)

        positions: Dict[str, PositionState] = {}
        current_value = initial_capital
        daily_returns: List[float] = []

        strategy.on_start(dp)

        for date in trade_dates:
            universe = dp.get_universe(date, universe_name)

            signals = strategy.on_bar(date, universe, positions, dp)

            day_return = 0.0
            for signal in signals:
                if signal.action == "buy":
                    positions, current_value, cost = _execute_buy(
                        signal, positions, current_value, dp, date, config
                    )
                    day_return -= cost / current_value if current_value > 0 else 0.0

                elif signal.action == "sell":
                    positions, current_value, cost = _execute_sell(
                        signal, positions, current_value, dp, date, config
                    )
                    day_return -= cost / current_value if current_value > 0 else 0.0

                elif signal.action == "add":
                    positions, current_value, cost = _execute_add(
                        signal, positions, current_value, dp, date, config
                    )
                    day_return -= cost / current_value if current_value > 0 else 0.0

            # 计算持仓浮盈亏
            pnl = _compute_position_pnl(positions, date, dp)
            if current_value > 0:
                day_return += pnl / current_value
                current_value += pnl

            daily_returns.append(day_return)

        strategy.on_end()

        returns_series = pd.Series(daily_returns, index=trade_dates)
        portfolio_value = (1 + returns_series).cumprod()

        return BacktestResult(
            daily_returns=returns_series,
            portfolio_value=portfolio_value,
            metadata={"strategy": strategy.name, "initial_capital": initial_capital},
        )


def _get_close_price(instrument: str, date: pd.Timestamp, dp: DataProvider) -> float:
    """获取单票收盘价，失败返回 0"""
    try:
        df = dp.get_ohlcv([instrument], str(date.date()), str(date.date()))
        if df.empty:
            return 0.0
        if isinstance(df.index, pd.MultiIndex):
            if instrument in df.index.get_level_values(df.index.names[-1]):
                row = df.xs(instrument, level=df.index.names[-1])
            elif instrument in df.index.get_level_values(df.index.names[0]):
                row = df.xs(instrument, level=df.index.names[0])
            else:
                return 0.0
        else:
            row = df
        if "close" not in row.columns:
            return 0.0
        val = row["close"].dropna()
        return float(val.iloc[-1]) if not val.empty else 0.0
    except Exception:
        return 0.0


def _execute_buy(
    signal: StrategySignal,
    positions: Dict[str, PositionState],
    current_value: float,
    dp: DataProvider,
    date: pd.Timestamp,
    config: dict,
) -> tuple:
    weight = signal.weight if signal.weight > 0 else 0.05
    buy_amount = current_value * weight
    price = signal.price or _get_close_price(signal.instrument, date, dp)
    if price <= 0:
        return positions, current_value, 0.0

    cost = compute_trade_cost(
        buy_value=buy_amount,
        sell_value=0,
        buy_commission_rate=config.get("trading", {}).get("buy_commission_rate", 0.0003),
        slippage_bps=config.get("trading", {}).get("slippage_bps", 5),
        impact_bps=config.get("trading", {}).get("impact_bps", 2),
    )
    actual_spend = buy_amount + cost
    if actual_spend > current_value:
        actual_spend = current_value
        buy_amount = actual_spend - cost

    pos = PositionState(
        instrument=signal.instrument,
        entry_price=price,
        layers=1,
        layer_prices=[price],
        stop_loss=price * 0.9,
        units=buy_amount / price,
    )
    positions[signal.instrument] = pos
    current_value -= actual_spend
    return positions, current_value, cost


def _execute_sell(
    signal: StrategySignal,
    positions: Dict[str, PositionState],
    current_value: float,
    dp: DataProvider,
    date: pd.Timestamp,
    config: dict,
) -> tuple:
    if signal.instrument not in positions:
        return positions, current_value, 0.0

    pos = positions[signal.instrument]
    price = signal.price or _get_close_price(signal.instrument, date, dp)
    if price <= 0:
        price = pos.entry_price

    sell_amount = pos.units * price
    cost = compute_trade_cost(
        buy_value=0,
        sell_value=sell_amount,
        sell_commission_rate=config.get("trading", {}).get("sell_commission_rate", 0.0003),
        sell_stamp_tax_rate=config.get("trading", {}).get("sell_stamp_tax_rate", 0.001),
        slippage_bps=config.get("trading", {}).get("slippage_bps", 5),
        impact_bps=config.get("trading", {}).get("impact_bps", 2),
    )
    current_value += sell_amount - cost
    del positions[signal.instrument]
    return positions, current_value, cost


def _execute_add(
    signal: StrategySignal,
    positions: Dict[str, PositionState],
    current_value: float,
    dp: DataProvider,
    date: pd.Timestamp,
    config: dict,
) -> tuple:
    if signal.instrument not in positions:
        return _execute_buy(signal, positions, current_value, dp, date, config)

    pos = positions[signal.instrument]
    add_weight = signal.weight if signal.weight > 0 else 0.05
    add_amount = current_value * add_weight
    price = signal.price or _get_close_price(signal.instrument, date, dp)
    if price <= 0:
        return positions, current_value, 0.0

    cost = compute_trade_cost(
        buy_value=add_amount,
        sell_value=0,
        buy_commission_rate=config.get("trading", {}).get("buy_commission_rate", 0.0003),
        slippage_bps=config.get("trading", {}).get("slippage_bps", 5),
        impact_bps=config.get("trading", {}).get("impact_bps", 2),
    )
    actual_spend = add_amount + cost
    if actual_spend > current_value:
        return positions, current_value, 0.0

    pos.units += add_amount / price
    pos.layers += 1
    pos.layer_prices.append(price)
    current_value -= actual_spend
    return positions, current_value, cost


def _compute_position_pnl(
    positions: Dict[str, PositionState],
    date: pd.Timestamp,
    dp: DataProvider,
) -> float:
    """计算当日持仓浮盈亏（简化：用当日 close 计算，不处理涨跌停）"""
    if not positions:
        return 0.0
    total_pnl = 0.0
    for instrument, pos in positions.items():
        close = _get_close_price(instrument, date, dp)
        if close > 0:
            total_pnl += (close - pos.entry_price) * pos.units
    return total_pnl
