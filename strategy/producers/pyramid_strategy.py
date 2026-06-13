"""
金字塔加仓策略
- N 日高点突破入场
- 每涨 ATR * add_factor 追加一层，最多 max_layers 层
- 跌破 entry_price - stop_atr * ATR 全部止损出场
"""
from typing import Dict, List
import pandas as pd

from strategies.base import RuleStrategy, PositionState, StrategySignal


class PyramidStrategy(RuleStrategy):

    def on_bar(
        self,
        date: pd.Timestamp,
        universe: List[str],
        positions: Dict[str, PositionState],
        data_provider,
    ) -> List[StrategySignal]:
        signals = []
        cfg = self.config.get("pyramid", {})
        entry_lookback = int(cfg.get("entry_lookback", 20))
        atr_period = int(cfg.get("atr_period", 20))
        add_factor = float(cfg.get("add_factor", 1.0))
        max_layers = int(cfg.get("max_layers", 3))
        stop_atr = float(cfg.get("stop_atr", 2.0))
        position_size = float(cfg.get("position_size", 0.05))

        lookback_start = date - pd.Timedelta(days=entry_lookback * 2)
        ohlcv = data_provider.get_ohlcv(
            list(universe), str(lookback_start.date()), str(date.date())
        )

        for instrument in universe:
            close = self._get_close(ohlcv, instrument, date)
            if close <= 0:
                continue

            if instrument not in positions:
                high_n = self._rolling_high(ohlcv, instrument, date, entry_lookback)
                if high_n is None:
                    continue
                if close > high_n:
                    atr = data_provider.get_atr(instrument, date, atr_period)
                    stop_loss = close - stop_atr * atr
                    sig = StrategySignal(
                        date=date,
                        instrument=instrument,
                        action="buy",
                        weight=position_size,
                        price=close,
                    )
                    sig.__dict__["stop_loss_price"] = stop_loss
                    signals.append(sig)
            else:
                pos = positions[instrument]
                if close <= pos.stop_loss:
                    signals.append(StrategySignal(
                        date=date,
                        instrument=instrument,
                        action="sell",
                        price=close,
                        reason="stop_loss",
                    ))
                elif pos.layers < max_layers:
                    atr = data_provider.get_atr(instrument, date, atr_period)
                    add_trigger = pos.layer_prices[-1] + atr * add_factor
                    if close >= add_trigger:
                        signals.append(StrategySignal(
                            date=date,
                            instrument=instrument,
                            action="add",
                            weight=position_size,
                            price=close,
                        ))

        return signals

    def _get_close(self, ohlcv: pd.DataFrame, instrument: str, date: pd.Timestamp) -> float:
        if ohlcv is None or ohlcv.empty:
            return 0.0
        try:
            if isinstance(ohlcv.index, pd.MultiIndex):
                names = ohlcv.index.names
                inst_level = names[-1] if names[-1] != "date" else names[0]
                if instrument in ohlcv.index.get_level_values(inst_level):
                    sub = ohlcv.xs(instrument, level=inst_level)
                    row = sub[sub.index <= date]["close"].dropna()
                    return float(row.iloc[-1]) if not row.empty else 0.0
            val = ohlcv.loc[ohlcv.index.get_level_values(-1) == instrument, "close"]
            row = val[val.index.get_level_values(0) <= date].dropna()
            return float(row.iloc[-1]) if not row.empty else 0.0
        except Exception:
            return 0.0

    def _rolling_high(
        self, ohlcv: pd.DataFrame, instrument: str, date: pd.Timestamp, lookback: int
    ):
        """返回 [date-lookback 日, date) 区间内的最高 close，不含今日"""
        if ohlcv is None or ohlcv.empty:
            return None
        try:
            if isinstance(ohlcv.index, pd.MultiIndex):
                names = ohlcv.index.names
                inst_level = names[-1] if names[-1] != "date" else names[0]
                if instrument not in ohlcv.index.get_level_values(inst_level):
                    return None
                sub = ohlcv.xs(instrument, level=inst_level)
                past = sub[sub.index < date]["close"].dropna().tail(lookback)
            else:
                past = ohlcv[ohlcv.index < date]["close"].dropna().tail(lookback)
            if past.empty:
                return None
            return float(past.max())
        except Exception:
            return None
