"""
海龟交易策略
- N 日高点突破入场（默认 20 日）
- M 日低点跌破出场（默认 10 日）
- 跌破 ATR 止损线出场
"""
from typing import Dict, List
import pandas as pd

from strategies.base import RuleStrategy, PositionState, StrategySignal


class TurtleStrategy(RuleStrategy):

    def on_bar(
        self,
        date: pd.Timestamp,
        universe: List[str],
        positions: Dict[str, PositionState],
        data_provider,
    ) -> List[StrategySignal]:
        signals = []
        cfg = self.config.get("turtle", {})
        entry_lookback = int(cfg.get("entry_lookback", 20))
        exit_lookback = int(cfg.get("exit_lookback", 10))
        atr_period = int(cfg.get("atr_period", 20))
        stop_atr = float(cfg.get("stop_atr", 2.0))
        position_size = float(cfg.get("position_size", 0.02))

        lookback_start = date - pd.Timedelta(days=max(entry_lookback, exit_lookback) * 2)
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
                # ATR 止损优先
                if close <= pos.stop_loss:
                    signals.append(StrategySignal(
                        date=date,
                        instrument=instrument,
                        action="sell",
                        price=close,
                        reason="atr_stop",
                    ))
                else:
                    # N 日低点出场
                    low_m = self._rolling_low(ohlcv, instrument, date, exit_lookback)
                    if low_m is not None and close < low_m:
                        signals.append(StrategySignal(
                            date=date,
                            instrument=instrument,
                            action="sell",
                            price=close,
                            reason="exit_low",
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
            return 0.0
        except Exception:
            return 0.0

    def _rolling_high(
        self, ohlcv: pd.DataFrame, instrument: str, date: pd.Timestamp, lookback: int
    ):
        """返回 [date-lookback 日, date) 区间内最高 close，不含今日"""
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
            return float(past.max()) if not past.empty else None
        except Exception:
            return None

    def _rolling_low(
        self, ohlcv: pd.DataFrame, instrument: str, date: pd.Timestamp, lookback: int
    ):
        """返回 [date-lookback 日, date) 区间内最低 close，不含今日"""
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
            return float(past.min()) if not past.empty else None
        except Exception:
            return None
