"""
规则驱动回测引擎
支持逐日 on_bar() 事件循环，用于海龟/金字塔等规则类策略。

资产核算（NAV 模型）：
  每日总净值 NAV = 现金 cash + Σ(持仓 units × 当日 close)
  daily_return = (NAV_today - NAV_prev) / NAV_prev
买入把现金转为持仓（NAV 仅减交易成本），卖出把持仓转回现金。成本通过 cash 自然反映进 NAV。
"""
from typing import Dict, List
import pandas as pd

from engine.base import BacktestResult, BacktestEngine
from common.calendar import load_trade_calendar
from common.costs import compute_trade_cost
from data.provider import DataProvider
from data.qlib_provider import QlibDataProvider
from strategy.base import RuleStrategy, PositionState, StrategySignal


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
        cash = initial_capital
        prev_nav = initial_capital
        daily_returns: List[float] = []

        strategy.on_start(dp)

        for date in trade_dates:
            universe = dp.get_universe(date, universe_name)
            signals = strategy.on_bar(date, universe, positions, dp)

            for signal in signals:
                if signal.action == "buy":
                    cash = _execute_buy(signal, positions, cash, dp, date, config)
                elif signal.action == "sell":
                    cash = _execute_sell(signal, positions, cash, dp, date, config)
                elif signal.action == "add":
                    cash = _execute_add(signal, positions, cash, dp, date, config)

            # 当日总净值 = 现金 + 持仓市值
            holdings_value = 0.0
            for instrument, pos in positions.items():
                close = _get_close_price(instrument, date, dp)
                if close > 0:
                    holdings_value += pos.units * close
            nav = cash + holdings_value

            day_return = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0
            daily_returns.append(day_return)
            prev_nav = nav

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


def _trade_cfg(config: dict) -> dict:
    return config.get("trading", {})


def _execute_buy(
    signal: StrategySignal,
    positions: Dict[str, PositionState],
    cash: float,
    dp: DataProvider,
    date: pd.Timestamp,
    config: dict,
) -> float:
    """买入：现金转为持仓，返回更新后的现金。"""
    weight = signal.weight if signal.weight > 0 else 0.05
    price = signal.price or _get_close_price(signal.instrument, date, dp)
    if price <= 0 or cash <= 0:
        return cash

    trading = _trade_cfg(config)
    buy_amount = cash * weight
    cost = compute_trade_cost(
        buy_value=buy_amount, sell_value=0,
        buy_commission_rate=trading.get("buy_commission_rate", 0.0003),
        slippage_bps=trading.get("slippage_bps", 5),
        impact_bps=trading.get("impact_bps", 2),
    )
    if buy_amount + cost > cash:
        buy_amount = max(cash - cost, 0.0)
        cost = compute_trade_cost(
            buy_value=buy_amount, sell_value=0,
            buy_commission_rate=trading.get("buy_commission_rate", 0.0003),
            slippage_bps=trading.get("slippage_bps", 5),
            impact_bps=trading.get("impact_bps", 2),
        )
    if buy_amount <= 0:
        return cash

    positions[signal.instrument] = PositionState(
        instrument=signal.instrument,
        entry_price=price,
        layers=1,
        layer_prices=[price],
        stop_loss=price * 0.9,
        units=buy_amount / price,
    )
    return cash - buy_amount - cost


def _execute_sell(
    signal: StrategySignal,
    positions: Dict[str, PositionState],
    cash: float,
    dp: DataProvider,
    date: pd.Timestamp,
    config: dict,
) -> float:
    """卖出：持仓转回现金，返回更新后的现金。"""
    if signal.instrument not in positions:
        return cash

    pos = positions[signal.instrument]
    price = signal.price or _get_close_price(signal.instrument, date, dp)
    if price <= 0:
        price = pos.entry_price

    sell_amount = pos.units * price
    trading = _trade_cfg(config)
    cost = compute_trade_cost(
        buy_value=0, sell_value=sell_amount,
        sell_commission_rate=trading.get("sell_commission_rate", 0.0003),
        sell_stamp_tax_rate=trading.get("sell_stamp_tax_rate", 0.001),
        slippage_bps=trading.get("slippage_bps", 5),
        impact_bps=trading.get("impact_bps", 2),
    )
    del positions[signal.instrument]
    return cash + sell_amount - cost


def _execute_add(
    signal: StrategySignal,
    positions: Dict[str, PositionState],
    cash: float,
    dp: DataProvider,
    date: pd.Timestamp,
    config: dict,
) -> float:
    """加仓：现金转为已有持仓的新一层，返回更新后的现金。"""
    if signal.instrument not in positions:
        return _execute_buy(signal, positions, cash, dp, date, config)

    pos = positions[signal.instrument]
    add_weight = signal.weight if signal.weight > 0 else 0.05
    price = signal.price or _get_close_price(signal.instrument, date, dp)
    if price <= 0 or cash <= 0:
        return cash

    trading = _trade_cfg(config)
    add_amount = cash * add_weight
    cost = compute_trade_cost(
        buy_value=add_amount, sell_value=0,
        buy_commission_rate=trading.get("buy_commission_rate", 0.0003),
        slippage_bps=trading.get("slippage_bps", 5),
        impact_bps=trading.get("impact_bps", 2),
    )
    if add_amount + cost > cash or add_amount <= 0:
        return cash

    pos.units += add_amount / price
    pos.layers += 1
    pos.layer_prices.append(price)
    return cash - add_amount - cost
