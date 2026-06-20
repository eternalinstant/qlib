"""
Qlib 回测引擎
使用共享选股 + 仓位控制进行回测
"""

import warnings

warnings.filterwarnings("ignore")

import hashlib
import numpy as np
import pandas as pd
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from common.config import CONFIG
from strategy.selection import SELECTION_CSV
from data.qlib_init import init_qlib, load_features_safe
from data.universe import filter_instruments, is_st_on_date
from engine.base import BacktestResult, BacktestEngine
from common.price_limit import (
    CHINEXT_REFORM_DATE,
    PRICE_LIMIT_TOL,
    round_limit_price as _round_limit_price,
    get_price_limit_pct as _get_price_limit_pct,
    get_limit_prices as _get_limit_prices,
    can_buy_at_open as _can_buy_at_open,
    can_sell_at_open as _can_sell_at_open,
)
from common.paths import (
    raw_data_root as _raw_data_root,
    raw_data_path_for_instrument as _raw_data_path_for_instrument,
)
from common.calendar import load_trade_calendar as _load_trade_calendar_slice
from data.quotes import load_raw_trade_quotes as _load_raw_trade_quotes
from strategy.overlay import compute_inverse_vol_weights
from common.logger import setup_logger, TradeLogger
from common.platform import resolve_path

# 进度回调 hook：每个交易日完成时调用。供 qlib_ui worker 实时显示用。
# 签名: fn(date: pd.Timestamp, return_value: float, portfolio_value: float, stock_pct: float, current_value: float) -> None
_DAY_PROGRESS_HOOK = None


def set_day_progress_hook(fn):
    """设置每日进度回调；传 None 清除。回调异常不会中断回测。"""
    global _DAY_PROGRESS_HOOK
    _DAY_PROGRESS_HOOK = fn

# 国债ETF数据路径
BOND_ETF_PATH = Path(__file__).parent.parent.parent / "data" / "tushare" / "bond_etf_daily.parquet"
EMPTY_RAW_DAY_QUOTES = pd.DataFrame(columns=["open", "close", "prev_close", "amount"])


@lru_cache(maxsize=1)
def _load_bond_etf_returns() -> pd.Series:
    """加载国债ETF日收益率数据"""
    if not BOND_ETF_PATH.exists():
        return None

    try:
        df = pd.read_parquet(BOND_ETF_PATH)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.sort_values("trade_date")
        df["daily_ret"] = df["pct_chg"] / 100  # 转换为小数
        return df.set_index("trade_date")["daily_ret"]
    except Exception:
        return None




def _compute_next_open_return_adj(
    raw_day_quotes: pd.DataFrame,
    actual_sell: set,
    actual_buy: set,
    topk: int,
) -> float:
    """T+1 开盘成交的收益调整量。

    基础收益使用旧持仓（old portfolio）的 close-to-close 日收益。
    卖出标的在 T+1 开盘成交：只赚隔夜段（open/prev_close - 1），
        → 需减去日内段：-(close/open - 1) / topk
    新买入标的在 T+1 开盘成交：只赚日内段（close/open - 1），
        不赚隔夜段（旧持仓里没有此标的，故基础收益贡献为 0）
        → 需加上日内段：+(close/open - 1) / topk
    """
    if topk <= 0 or (not actual_sell and not actual_buy):
        return 0.0
    adj = 0.0
    for symbol in actual_sell:
        row = _quote_row(raw_day_quotes, symbol)
        if row is not None:
            o = float(row.get("open") or 0)
            c = float(row.get("close") or 0)
            if o > 0 and c > 0:
                adj -= (c / o - 1.0) / topk
    for symbol in actual_buy:
        row = _quote_row(raw_day_quotes, symbol)
        if row is not None:
            o = float(row.get("open") or 0)
            c = float(row.get("close") or 0)
            if o > 0 and c > 0:
                adj += (c / o - 1.0) / topk
    return adj


def _ensure_tradability_constraints_supported(
    block_limit_up_buy: bool, block_limit_down_sell: bool
) -> None:
    """启用成交约束前，必须确认 raw_data 原始日线目录存在。"""
    if not (block_limit_up_buy or block_limit_down_sell):
        return
    raw_root = _raw_data_root()
    if raw_root.exists():
        return
    raise FileNotFoundError(
        f"缺少原始日线目录: {raw_root}，不能正式启用 block_limit_up_buy / block_limit_down_sell。"
    )


def _load_ranked_selection_orders(strategy=None) -> dict:
    csv_path = strategy.selections_path() if strategy else SELECTION_CSV
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"选股文件不存在: {csv_path}")

    df = pd.read_csv(csv_path, usecols=["date", "rank", "symbol"], parse_dates=["date"])
    if df.empty:
        return {}

    result = {}
    for dt, grp in df.sort_values(["date", "rank", "symbol"]).groupby("date"):
        result[pd.Timestamp(dt)] = grp["symbol"].astype(str).tolist()
    return result


def _collect_required_instruments(date_to_symbols: dict) -> list:
    """从选股结果中提取回测真正需要的股票并集。"""
    instruments = set()
    for symbols in date_to_symbols.values():
        instruments.update(symbols)
    return sorted(instruments)


def _load_provider_close_frame(instruments, start_date: str, end_date: str) -> pd.DataFrame:
    if not instruments:
        return pd.DataFrame(columns=["close", "prev_close", "daily_ret"])

    provider_px = load_features_safe(
        instruments,
        ["$close"],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    if provider_px is None or provider_px.empty:
        return pd.DataFrame(columns=["close", "prev_close", "daily_ret"])

    provider_px = provider_px.copy()
    provider_px.columns = ["close"]
    if list(provider_px.index.names) == ["instrument", "datetime"]:
        provider_px = provider_px.swaplevel().sort_index()
    provider_px.index = provider_px.index.set_names(["datetime", "instrument"])
    provider_px = provider_px.sort_index()

    try:
        calendar = _load_trade_calendar_slice(start_date, end_date)
    except Exception:
        calendar = pd.DatetimeIndex(provider_px.index.get_level_values("datetime").unique()).sort_values()

    aligned_parts = []
    for instrument, grp in provider_px.groupby(level="instrument", sort=False):
        work = grp.droplevel("instrument").sort_index().reindex(calendar)
        work["instrument"] = instrument
        aligned_parts.append(work.reset_index().rename(columns={"index": "datetime"}))

    if not aligned_parts:
        return pd.DataFrame(columns=["close", "prev_close", "daily_ret"])

    provider_frame = pd.concat(aligned_parts, ignore_index=True)
    provider_frame = provider_frame.set_index(["datetime", "instrument"]).sort_index()
    provider_frame.index = provider_frame.index.set_names(["datetime", "instrument"])
    has_any_quote = provider_frame["close"].groupby(level="datetime").transform(
        lambda s: s.notna().any()
    )
    provider_frame = provider_frame[has_any_quote]
    provider_frame["prev_close"] = provider_frame.groupby(level="instrument")["close"].shift(1)
    provider_frame["daily_ret"] = provider_frame["close"] / provider_frame["prev_close"] - 1
    provider_frame = provider_frame.replace([np.inf, -np.inf], np.nan)
    return provider_frame[["close", "prev_close", "daily_ret"]]


def _load_backtest_return_frame(
    instruments,
    start_date: str,
    end_date: str,
    include_raw: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load close-to-close returns for backtests.

    Returns use qlib provider qfq ``$close``. Raw daily bars are returned only
    for tradability constraints and should not drive performance returns.
    """
    instruments = sorted(set(instruments))
    if not instruments:
        empty_px = pd.DataFrame(columns=["close", "prev_close", "daily_ret"])
        empty_raw = pd.DataFrame(columns=["open", "close", "prev_close"])
        return empty_px, empty_raw

    df_px = _load_provider_close_frame(instruments, start_date, end_date)
    raw_quotes = (
        _load_raw_trade_quotes(instruments, start_date, end_date)
        if include_raw
        else pd.DataFrame(columns=["open", "close", "prev_close"])
    )
    return df_px, raw_quotes


def _fingerprint_raw_data(instruments: list) -> str:
    """Return a compact raw_data fingerprint for reproducibility metadata."""
    h = hashlib.md5()
    count = 0
    for inst in sorted(instruments):
        path = _raw_data_path_for_instrument(inst)
        if path.exists():
            st = path.stat()
            h.update(f"{inst}:{st.st_size}:{st.st_mtime_ns}".encode())
            count += 1
    return f"raw_n={count}_md5={h.hexdigest()[:12]}"


def _quote_row(raw_day_quotes: pd.DataFrame, instrument: str):
    if raw_day_quotes is None or raw_day_quotes.empty or instrument not in raw_day_quotes.index:
        return None
    row = raw_day_quotes.loc[instrument]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def _split_by_datetime(obj) -> dict:
    """按 datetime 预切分 MultiIndex 数据，避免循环中重复 xs()。"""
    if obj is None or len(obj) == 0:
        return {}
    if not isinstance(obj.index, pd.MultiIndex) or "datetime" not in obj.index.names:
        return {}

    result = {}
    for dt, grp in obj.groupby(level="datetime", sort=False):
        result[pd.Timestamp(dt)] = grp.droplevel("datetime")
    return result


def _sum_symbol_returns(
    day_px: pd.DataFrame,
    symbols: set,
    column: str,
    penalized_missing=None,
    apply_penalty: bool = True,
):
    if not symbols:
        return 0.0, set(), set()

    if isinstance(day_px, pd.Series):
        day_values = day_px
    else:
        day_values = day_px[column]

    symbol_set = set(symbols)
    available_mask = day_values.index.isin(symbol_set)
    available_symbols = set(day_values.index[available_mask].tolist())
    missing_symbols = symbol_set - available_symbols

    if available_symbols:
        values = np.asarray(day_values.to_numpy(copy=False)[available_mask], dtype=float)
        finite_values = values[np.isfinite(values)]
        total = float(finite_values.sum()) if finite_values.size else 0.0
    else:
        total = 0.0

    if apply_penalty and penalized_missing is not None:
        newly_missing = missing_symbols - penalized_missing
        total += -1.0 * len(newly_missing)
        penalized_missing.update(newly_missing)

    return total, available_symbols, missing_symbols


def _compute_rebalance_day(
    day_px: pd.DataFrame,
    selected: set,
    prev_selected: set,
    topk: int,
    penalized_missing=None,
    ranked_selected=None,
    raw_day_quotes: pd.DataFrame = None,
    trade_date=None,
    block_limit_up_buy: bool = False,
    block_limit_down_sell: bool = False,
    t1_locked_symbols=None,
    volume_cap_pct=None,
    per_position_notional: float = 0.0,
    partial_fill_min_scale: float = 0.20,
    enforce_lot_size: bool = False,
    lot_size: int = 100,
    min_lots: int = 1,
):
    """按 close-to-close 口径计算调仓日决策。

    这里显式不用同日 open/close 拆收益。当前本地 Qlib 日线字段中，
    $open 与 $close 不能可靠组成同日收益比值；强行拆成隔夜/日内会明显放大回测。
    T+1 开盘成交的收益调整由调用方通过 _compute_next_open_return_adj 完成。
    """
    prev_selected = set(prev_selected)
    selected = set(selected)
    ranked_selected = list(ranked_selected or sorted(selected))
    t1_locked = set(t1_locked_symbols) if t1_locked_symbols else set()

    held_symbols = set(prev_selected & selected)
    actual_sell_symbols: set = set()
    actual_buy_symbols: set = set()
    sell_count = 0
    buy_count = 0
    blocked_sell_count = 0
    blocked_buy_count = 0
    t1_locked_count = 0
    volume_capped_count = 0
    lot_blocked_count = 0
    min_buy_shares = float(lot_size) * float(min_lots) if enforce_lot_size else 0.0

    for symbol in sorted(prev_selected - selected):
        if t1_locked and symbol in t1_locked:
            held_symbols.add(symbol)
            t1_locked_count += 1
            continue
        sellable = True
        if block_limit_down_sell:
            row = _quote_row(raw_day_quotes, symbol)
            sellable = bool(
                row is not None
                and _can_sell_at_open(
                    symbol,
                    trade_date,
                    row.get("open"),
                    row.get("prev_close"),
                    is_st=is_st_on_date(symbol, trade_date),
                )
            )
        if sellable:
            sell_count += 1
            actual_sell_symbols.add(symbol)
        else:
            held_symbols.add(symbol)
            blocked_sell_count += 1

    available_buy_slots = max(topk - len(held_symbols), 0)
    for symbol in ranked_selected:
        if symbol in prev_selected or symbol in held_symbols:
            continue
        if available_buy_slots <= 0:
            break

        buyable = True
        if block_limit_up_buy:
            row = _quote_row(raw_day_quotes, symbol)
            buyable = bool(
                row is not None
                and _can_buy_at_open(
                    symbol,
                    trade_date,
                    row.get("open"),
                    row.get("prev_close"),
                    is_st=is_st_on_date(symbol, trade_date),
                )
            )
        was_volume_capped = False
        if buyable and volume_cap_pct and per_position_notional > 0:
            row = _quote_row(raw_day_quotes, symbol)
            if row is not None:
                daily_amount_yuan = float(row.get("amount") or 0) * 10000  # 万元 → 元
                if daily_amount_yuan > 0:
                    max_notional = daily_amount_yuan * volume_cap_pct
                    if max_notional < per_position_notional:
                        scale = max_notional / per_position_notional
                        if scale < partial_fill_min_scale:
                            buyable = False
                            was_volume_capped = True
                            volume_capped_count += 1

        was_lot_blocked = False
        # 最低买入手数约束：检查单标的资金能否买够 min_lots 手（默认 1 手 = 100 股）
        if buyable and enforce_lot_size and per_position_notional > 0:
            row = _quote_row(raw_day_quotes, symbol)
            if row is not None:
                open_price = float(row.get("open") or 0)
                if open_price > 0:
                    affordable_shares = per_position_notional / open_price
                    if affordable_shares < min_buy_shares:
                        buyable = False
                        was_lot_blocked = True
                        lot_blocked_count += 1

        if buyable:
            held_symbols.add(symbol)
            actual_buy_symbols.add(symbol)
            buy_count += 1
            available_buy_slots -= 1
        elif not was_volume_capped and not was_lot_blocked:
            blocked_buy_count += 1

    held_sum, available_selected, missing_symbols = _sum_symbol_returns(
        day_px,
        held_symbols,
        "daily_ret",
        penalized_missing=penalized_missing,
    )
    available_selected = set(available_selected)

    return {
        "stock_slot_return": held_sum / topk if topk > 0 else 0.0,
        "held_symbols": available_selected,
        "actual_sell": actual_sell_symbols,
        "actual_buy": actual_buy_symbols,
        "sell_count": sell_count,
        "buy_count": buy_count,
        "blocked_sell_count": blocked_sell_count,
        "blocked_buy_count": blocked_buy_count,
        "t1_locked_count": t1_locked_count,
        "volume_capped_count": volume_capped_count,
        "lot_blocked_count": lot_blocked_count,
        "cash_slot_count": max(topk - len(available_selected), 0),
        "missing_count": len(missing_symbols),
    }


def _as_positive_float(value, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return max(out, 0.0)


def _as_positive_int(value, default: int, minimum: int = 1) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    return max(out, int(minimum))


def _equal_weight_series(symbols) -> pd.Series:
    symbols = sorted(set(symbols))
    if not symbols:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(symbols), index=symbols, dtype=float)


def _parse_vol_norm_params(strategy) -> dict:
    params = dict(getattr(strategy, "position_params", {}) or {})
    return {
        "vol_method": str(params.get("vol_method", "inverse_sqrt")),
        "vol_lookback": _as_positive_int(params.get("vol_lookback", 20), default=20, minimum=2),
        "cap_max_weight": _as_positive_float(params.get("cap_max_weight", 0.0), default=0.0),
    }


def _compute_annualized_volatility_frame(df_px: pd.DataFrame, lookback: int) -> pd.Series:
    if df_px.empty or lookback <= 1:
        return pd.Series(dtype=float)
    rolling_std = (
        df_px["daily_ret"]
        .groupby(level="instrument")
        .rolling(window=int(lookback), min_periods=int(lookback))
        .std(ddof=0)
        .droplevel(0)
    )
    annualized_vol = rolling_std * np.sqrt(252)
    annualized_vol = annualized_vol.replace([np.inf, -np.inf], np.nan)
    return annualized_vol.rename("annualized_vol")


def _compute_vol_norm_target_weights(
    selected_symbols: set,
    vol_series: pd.Series,
    method: str,
    cap_max_weight: float,
) -> pd.Series:
    equal_weights = _equal_weight_series(selected_symbols)
    if equal_weights.empty:
        return equal_weights
    if vol_series is None or len(vol_series) == 0:
        return equal_weights

    vols = pd.Series(vol_series, dtype=float).reindex(equal_weights.index)
    valid = vols.notna() & np.isfinite(vols) & (vols > 0)
    # vol 缺失时整组回退等权，避免局部缺失导致权重失真。
    if not bool(valid.all()):
        return equal_weights

    weights = compute_inverse_vol_weights(
        vols,
        method=method,
        cap_max_weight=float(cap_max_weight),
    ).reindex(equal_weights.index)
    if weights.empty:
        return equal_weights
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = float(weights.sum())
    if total <= 0:
        return equal_weights
    return weights / total


def _normalize_weight_map(weight_map: dict, allowed_symbols=None) -> dict:
    out = {}
    allowed = set(allowed_symbols) if allowed_symbols is not None else None
    for symbol, weight in (weight_map or {}).items():
        if allowed is not None and symbol not in allowed:
            continue
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if np.isfinite(w) and w > 0:
            out[str(symbol)] = w
    total = float(sum(out.values()))
    if total > 1.0 + 1e-12:
        scale = 1.0 / total
        out = {symbol: weight * scale for symbol, weight in out.items()}
    return out


def _compose_post_rebalance_symbol_weights(
    prev_symbol_weights: dict,
    target_symbol_weights: pd.Series,
    held_symbols: set,
) -> dict:
    held = set(held_symbols or set())
    if not held:
        return {}

    prev = _normalize_weight_map(prev_symbol_weights, allowed_symbols=held)
    target = pd.Series(target_symbol_weights, dtype=float) if target_symbol_weights is not None else pd.Series(dtype=float)
    target = target.reindex(sorted(held)).fillna(0.0).clip(lower=0.0)
    target = target[target > 0]

    blocked_symbols = sorted(held - set(target.index))
    blocked_weights = {symbol: prev.get(symbol, 0.0) for symbol in blocked_symbols}
    blocked_total = float(sum(blocked_weights.values()))
    if blocked_total > 1.0 + 1e-12:
        scale = 1.0 / blocked_total
        blocked_weights = {symbol: weight * scale for symbol, weight in blocked_weights.items()}
        blocked_total = 1.0

    result = {symbol: weight for symbol, weight in blocked_weights.items() if weight > 0}
    remaining = max(1.0 - blocked_total, 0.0)
    target_total = float(target.sum())
    if target_total > 0 and remaining > 0:
        scaled_target = target / target_total * remaining
        for symbol, weight in scaled_target.items():
            if weight > 0:
                result[symbol] = float(weight)

    result = _normalize_weight_map(result, allowed_symbols=held)
    return result


def _compute_weighted_stock_return(
    day_px: pd.DataFrame,
    symbol_weights: dict,
    penalized_missing=None,
):
    weight_map = _normalize_weight_map(symbol_weights)
    if not weight_map:
        return 0.0, 0

    if isinstance(day_px, pd.Series):
        day_values = day_px
    else:
        day_values = day_px["daily_ret"]

    total = 0.0
    missing_symbols = set()
    for symbol, weight in weight_map.items():
        if symbol not in day_values.index:
            missing_symbols.add(symbol)
            continue
        value = day_values.loc[symbol]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        ret = float(value)
        if not np.isfinite(ret):
            # NaN/Inf 收益视为停牌（收益为0），不触发 penalized_missing
            # 与 _sum_symbol_returns 行为一致：NaN 不算 missing
            continue
        total += weight * ret

    if penalized_missing is not None:
        newly_missing = missing_symbols - penalized_missing
        total -= float(sum(weight_map.get(symbol, 0.0) for symbol in newly_missing))
        penalized_missing.update(newly_missing)

    return total, len(missing_symbols)


def _compute_weight_delta_fee(
    pre_fee_value: float,
    current_stock_pct: float,
    target_stock_pct: float,
    prev_symbol_weights: dict,
    target_symbol_weights: dict,
    buy_commission_rate: float,
    sell_commission_rate: float,
    sell_stamp_tax_rate: float,
    min_buy_commission: float,
    min_sell_commission: float,
    execution_cost_rate: float,
):
    if pre_fee_value <= 0:
        return 0.0, 0, 0

    prev = _normalize_weight_map(prev_symbol_weights)
    target = _normalize_weight_map(target_symbol_weights)
    symbols = sorted(set(prev) | set(target))

    buy_amounts = []
    sell_amounts = []
    current_pct = max(float(current_stock_pct), 0.0)
    target_pct = max(float(target_stock_pct), 0.0)

    for symbol in symbols:
        prev_notional = pre_fee_value * current_pct * prev.get(symbol, 0.0)
        target_notional = pre_fee_value * target_pct * target.get(symbol, 0.0)
        delta = target_notional - prev_notional
        if delta > 1e-8:
            buy_amounts.append(float(delta))
        elif delta < -1e-8:
            sell_amounts.append(float(-delta))

    buy_fee = sum(max(amount * buy_commission_rate, min_buy_commission) for amount in buy_amounts)
    sell_fee = sum(
        max(amount * sell_commission_rate, min_sell_commission) + amount * sell_stamp_tax_rate
        for amount in sell_amounts
    )
    execution_fee = (sum(buy_amounts) + sum(sell_amounts)) * execution_cost_rate
    fee_amount = buy_fee + sell_fee + execution_fee
    return fee_amount, len(buy_amounts), len(sell_amounts)


def _resolve_target_allocation(strategy, controller, signal_date):
    """解析某个信号日对应的目标仓位。

    Qlib 组合回测现在按 T+1 close 口径执行：
    - `signal_date` 只负责生成下一次收盘执行的目标仓位
    - 当天持仓收益由上一个已执行仓位决定
    """
    if controller is not None:
        alloc = controller.get_allocation(signal_date, is_rebalance_day=True)
        return (
            float(alloc.stock_pct),
            alloc.regime,
            alloc.opportunity_level,
            alloc.market_drawdown,
        )

    if strategy is not None and getattr(strategy, "position_model", None) in {"fixed", "vol_norm"}:
        stock_pct = float(getattr(strategy, "position_params", {}).get("stock_pct", 0.8))
        return stock_pct, str(getattr(strategy, "position_model", "fixed")), "none", 0.0

    return 1.0, "full", "none", 0.0


# ============================================================
# 回测引擎
# ============================================================


class QlibBacktestEngine(BacktestEngine):
    """Qlib 回测引擎"""

    def run(self, strategy=None) -> BacktestResult:
        """执行回测"""
        logger = setup_logger("backtest", level="INFO")
        trade_logger = TradeLogger(log_dir=CONFIG.get("paths.logs", "./logs"))
        _ = trade_logger  # 暂保留，避免未来接入明细日志时再改接口

        strategy_name = strategy.name if strategy else "default"
        strategy_slug = strategy.artifact_slug() if strategy else "default"
        logger.info(f"=== Qlib 回测 [{strategy_name}] ===")

        init_qlib()

        logger.info("[1/4] 加载选股列表...")
        date_to_symbols, rebalance_dates, controller, topk = self._prepare(strategy)
        monthly_dates_list = sorted(rebalance_dates)
        logger.info(f"调仓截面数: {len(monthly_dates_list)}")

        logger.info("[2/4] 加载收益率数据...")
        start_date = CONFIG.get("start_date", "2019-01-01")
        end_date = CONFIG.get("end_date", "2026-02-26")
        selected_instruments = _collect_required_instruments(date_to_symbols)
        if not selected_instruments:
            logger.error("选股结果为空，无法加载价格数据")
            return BacktestResult(
                daily_returns=pd.Series(dtype=float),
                portfolio_value=pd.Series(dtype=float),
            )

        valid_instruments = filter_instruments(
            selected_instruments,
            exclude_st=False,
        )
        logger.info(f"数据指纹: {_fingerprint_raw_data(valid_instruments)}")
        df_px, _ = _load_backtest_return_frame(
            valid_instruments,
            start_date=start_date,
            end_date=end_date,
            include_raw=False,
        )
        if df_px.empty:
            logger.error("无法从 qlib provider 加载任何收益率数据")
            return BacktestResult(
                daily_returns=pd.Series(dtype=float),
                portfolio_value=pd.Series(dtype=float),
            )
        dates = df_px.index.get_level_values("datetime").unique().sort_values()
        daily_ret_by_date = _split_by_datetime(df_px["daily_ret"])
        date_index = pd.DatetimeIndex(dates)

        logger.info("[3/4] 执行回测...")
        portfolio_returns = []

        # 加载国债ETF实际收益率数据
        bond_etf_returns = _load_bond_etf_returns()
        bond_etf_map = bond_etf_returns.to_dict() if bond_etf_returns is not None else {}
        if bond_etf_returns is not None:
            logger.info(f"[OK] 已加载国债ETF收益率数据: {len(bond_etf_returns)} 个交易日")
        else:
            logger.info("[INFO] 未找到国债ETF数据，使用固定债券收益率")

        # 默认债券日收益率（当没有国债ETF数据时使用）
        default_bond_daily_ret = controller.get_bond_daily_return() if controller else 0.0

        trading_cost = getattr(strategy, "trading_cost", {}) if strategy else {}
        _capital_override = trading_cost.get("initial_capital")
        if _capital_override is not None:
            initial_capital = float(_capital_override)
        else:
            initial_capital = float(CONFIG.get("initial_capital", 500000))
        current_value = initial_capital
        logger.info(f"[CAPITAL] 初始资金: {initial_capital:,.0f} 元")
        buy_commission_rate = trading_cost.get(
            "buy_commission_rate", trading_cost.get("open_cost", 0.0003)
        )
        sell_stamp_tax_rate = trading_cost.get("sell_stamp_tax_rate", 0.001)
        sell_commission_rate = trading_cost.get(
            "sell_commission_rate",
            max(trading_cost.get("close_cost", 0.0013) - sell_stamp_tax_rate, 0.0),
        )
        min_buy_commission = trading_cost.get("min_buy_commission", 5.0)
        min_sell_commission = trading_cost.get("min_sell_commission", 5.0)
        slippage_rate = float(trading_cost.get("slippage_bps", 0.0) or 0.0) / 10000
        impact_rate = float(trading_cost.get("impact_bps", 0.0) or 0.0) / 10000
        block_limit_up_buy = trading_cost.get("block_limit_up_buy", False)
        block_limit_down_sell = trading_cost.get("block_limit_down_sell", False)
        next_open_execution = bool(trading_cost.get("next_open_execution", False))
        t1_sell_lock_days = int(trading_cost.get("t1_sell_lock_days", 0) or 0)
        _volume_cap_raw = trading_cost.get("volume_cap_pct", None)
        volume_cap_pct: float | None = float(_volume_cap_raw) if _volume_cap_raw else None
        partial_fill_min_scale = float(trading_cost.get("partial_fill_min_scale", 0.20) or 0.20)
        enforce_lot_size = bool(trading_cost.get("enforce_lot_size", False))
        lot_size = int(trading_cost.get("lot_size", 100) or 100)
        min_lots = int(trading_cost.get("min_lots", 1) or 1)
        _ensure_tradability_constraints_supported(block_limit_up_buy, block_limit_down_sell)
        need_raw = (
            block_limit_up_buy
            or block_limit_down_sell
            or next_open_execution
            or bool(volume_cap_pct)
            or enforce_lot_size
        )
        ranked_selection_orders = {}
        raw_quotes = pd.DataFrame()
        if need_raw:
            ranked_selection_orders = _load_ranked_selection_orders(strategy)
            required_instruments = {
                symbol
                for ranked_symbols in ranked_selection_orders.values()
                for symbol in ranked_symbols
            }
            raw_quotes = _load_raw_trade_quotes(required_instruments, start_date, end_date)
            _active = [k for k, v in [
                ("next_open_execution", next_open_execution),
                ("block_limit_up_buy", block_limit_up_buy),
                ("block_limit_down_sell", block_limit_down_sell),
                ("t1_sell_lock", t1_sell_lock_days > 0),
                ("volume_cap", volume_cap_pct is not None),
                (f"lot_size={lot_size}x{min_lots}", enforce_lot_size),
            ] if v]
            logger.info(f"[P0] 实盘约束已启用: {', '.join(_active)}")
        raw_quotes_by_date = _split_by_datetime(raw_quotes)

        position_model = str(getattr(strategy, "position_model", "") or "").lower()
        enable_vol_norm = bool(strategy is not None and position_model == "vol_norm")
        vol_norm_params = _parse_vol_norm_params(strategy) if enable_vol_norm else {}
        vol_by_date = {}
        if enable_vol_norm:
            vol_method = str(vol_norm_params["vol_method"])
            vol_lookback = int(vol_norm_params["vol_lookback"])
            cap_max_weight = float(vol_norm_params["cap_max_weight"])
            vol_frame = _compute_annualized_volatility_frame(df_px, lookback=vol_lookback)
            vol_by_date = _split_by_datetime(vol_frame)
            logger.info(
                "[VOL_NORM] 已启用个股波动率归一化: "
                f"method={vol_method}, lookback={vol_lookback}, cap={cap_max_weight:.2f}"
            )

        current_held_symbols = set()  # 上一个收盘后真实持仓（调仓失败后会偏离 target）
        current_hold_days: dict = {}  # symbol → 持仓天数（用于 T+1 卖出锁定）
        current_symbol_weights = {}
        current_cash_slot_count = topk
        current_stock_pct = 0.0  # T+1 close 口径：首个可交易日白天仍视为现金/债券仓
        current_regime = "cash"
        current_opp = "none"
        current_mkt_dd = 0.0
        total_fee_amount = 0.0
        target_allocation_by_date = {
            rebal_date: _resolve_target_allocation(strategy, controller, rebal_date)
            for rebal_date in monthly_dates_list[:-1]
        }

        # 预加载第一个 rebal_date 的配置，避免首个交易日 current_stock_pct=0
        if monthly_dates_list and monthly_dates_list[0] in target_allocation_by_date:
            first_alloc = target_allocation_by_date[monthly_dates_list[0]]
            current_stock_pct, current_regime, current_opp, current_mkt_dd = first_alloc

        for i, rebal_date in enumerate(monthly_dates_list[:-1]):
            selected = date_to_symbols.get(rebal_date, set())
            if len(selected) < topk:
                continue

            target_stock_pct, target_regime, target_opp, target_mkt_dd = target_allocation_by_date[
                rebal_date
            ]
            target_selected = set(selected) if target_stock_pct > 0 else set()
            ranked_selected = ranked_selection_orders.get(rebal_date, sorted(target_selected))

            next_date = monthly_dates_list[i + 1]
            start_pos = date_index.searchsorted(rebal_date, side="right")
            end_pos = date_index.searchsorted(next_date, side="right")
            holding_dates = date_index[start_pos:end_pos]
            penalized_missing = set()

            for j, hd in enumerate(holding_dates):
                day_returns = daily_ret_by_date.get(hd)
                if day_returns is None:
                    continue

                is_rebal = j == 0

                stock_slot_return = 0.0
                sell_count = 0
                buy_count = 0
                blocked_sell_count = 0
                blocked_buy_count = 0
                missing_count = 0
                t1_locked_count = 0
                volume_capped_count = 0
                lot_blocked_count = 0

                bond_daily_ret = bond_etf_map.get(hd, default_bond_daily_ret)

                # T+1 lock：非调仓日对已持仓标的递增持仓天数
                if t1_sell_lock_days > 0 and not is_rebal:
                    for _s in list(current_hold_days.keys()):
                        if _s in current_held_symbols:
                            current_hold_days[_s] = current_hold_days.get(_s, 0) + 1
                        else:
                            current_hold_days.pop(_s, None)

                # ===== 调仓日：先执行调仓决策，后计算收益 =====
                raw_day_quotes_hd = None
                rebal_result = None
                next_held_symbols = current_held_symbols.copy()
                next_cash_slot_count = current_cash_slot_count
                next_symbol_weights = current_symbol_weights.copy()

                if is_rebal:
                    if need_raw:
                        raw_day_quotes_hd = raw_quotes_by_date.get(hd, EMPTY_RAW_DAY_QUOTES)

                    t1_locked = (
                        frozenset(s for s, d in current_hold_days.items() if d < t1_sell_lock_days)
                        if t1_sell_lock_days > 0 else frozenset()
                    )
                    per_pos_notional = (
                        current_value * target_stock_pct / topk
                        if (volume_cap_pct or enforce_lot_size) and topk > 0 else 0.0
                    )
                    rebal_result = _compute_rebalance_day(
                        day_returns,
                        selected=target_selected,
                        prev_selected=current_held_symbols,
                        topk=topk,
                        penalized_missing=penalized_missing,
                        ranked_selected=ranked_selected,
                        raw_day_quotes=raw_day_quotes_hd,
                        trade_date=hd,
                        block_limit_up_buy=block_limit_up_buy,
                        block_limit_down_sell=block_limit_down_sell,
                        t1_locked_symbols=t1_locked,
                        volume_cap_pct=volume_cap_pct,
                        per_position_notional=per_pos_notional,
                        partial_fill_min_scale=partial_fill_min_scale,
                        enforce_lot_size=enforce_lot_size,
                        lot_size=lot_size,
                        min_lots=min_lots,
                    )
                    next_held_symbols = rebal_result["held_symbols"]
                    next_cash_slot_count = rebal_result["cash_slot_count"]
                    sell_count = rebal_result["sell_count"]
                    buy_count = rebal_result["buy_count"]
                    blocked_sell_count = rebal_result["blocked_sell_count"]
                    blocked_buy_count = rebal_result["blocked_buy_count"]
                    t1_locked_count = rebal_result.get("t1_locked_count", 0)
                    volume_capped_count = rebal_result.get("volume_capped_count", 0)
                    lot_blocked_count = rebal_result.get("lot_blocked_count", 0)

                # ===== 计算组合收益 =====
                if enable_vol_norm:
                    stock_slot_return, missing_count = _compute_weighted_stock_return(
                        day_returns,
                        current_symbol_weights,
                        penalized_missing=penalized_missing,
                    )
                    current_cash_weight = max(1.0 - float(sum(current_symbol_weights.values())), 0.0)
                    stock_return_component = stock_slot_return + current_cash_weight * bond_daily_ret
                else:
                    held_sum, _, held_missing = _sum_symbol_returns(
                        day_returns,
                        current_held_symbols,
                        "daily_ret",
                        penalized_missing=penalized_missing,
                    )
                    stock_slot_return = held_sum / topk if topk > 0 else 0.0
                    missing_count = len(held_missing)
                    # T+1 开盘成交：调整卖出/买入标的的收益拆分
                    if (
                        is_rebal and next_open_execution
                        and raw_day_quotes_hd is not None
                        and not raw_day_quotes_hd.empty
                        and rebal_result is not None
                    ):
                        stock_slot_return += _compute_next_open_return_adj(
                            raw_day_quotes_hd,
                            rebal_result.get("actual_sell", set()),
                            rebal_result.get("actual_buy", set()),
                            topk,
                        )
                    stock_return_component = stock_slot_return + (
                        current_cash_slot_count / topk * bond_daily_ret if topk > 0 else 0.0
                    )

                gross_port_ret = (
                    current_stock_pct * stock_return_component
                    + (1 - current_stock_pct) * bond_daily_ret
                )
                pre_fee_value = current_value * (1 + gross_port_ret)

                fee_amount = 0.0
                cost_deduction = 0.0

                if is_rebal:
                    if enable_vol_norm:
                        target_symbol_weights = _compute_vol_norm_target_weights(
                            target_selected,
                            vol_by_date.get(hd),
                            method=vol_norm_params["vol_method"],
                            cap_max_weight=vol_norm_params["cap_max_weight"],
                        )
                        next_symbol_weights = _compose_post_rebalance_symbol_weights(
                            prev_symbol_weights=current_symbol_weights,
                            target_symbol_weights=target_symbol_weights,
                            held_symbols=next_held_symbols,
                        )
                        fee_amount, buy_count, sell_count = _compute_weight_delta_fee(
                            pre_fee_value=pre_fee_value,
                            current_stock_pct=current_stock_pct,
                            target_stock_pct=target_stock_pct,
                            prev_symbol_weights=current_symbol_weights,
                            target_symbol_weights=next_symbol_weights,
                            buy_commission_rate=buy_commission_rate,
                            sell_commission_rate=sell_commission_rate,
                            sell_stamp_tax_rate=sell_stamp_tax_rate,
                            min_buy_commission=min_buy_commission,
                            min_sell_commission=min_sell_commission,
                            execution_cost_rate=slippage_rate + impact_rate,
                        )
                    else:
                        per_position_value = (
                            pre_fee_value * target_stock_pct / topk if topk > 0 else 0.0
                        )
                        buy_fee_per_order = (
                            max(per_position_value * buy_commission_rate, min_buy_commission)
                            if buy_count > 0
                            else 0.0
                        )
                        sell_commission_per_order = (
                            max(per_position_value * sell_commission_rate, min_sell_commission)
                            if sell_count > 0
                            else 0.0
                        )
                        sell_stamp_tax_per_order = (
                            per_position_value * sell_stamp_tax_rate if sell_count > 0 else 0.0
                        )
                        execution_cost_per_order = per_position_value * (slippage_rate + impact_rate)
                        fee_amount = (
                            buy_count * buy_fee_per_order
                            + sell_count * (sell_commission_per_order + sell_stamp_tax_per_order)
                            + (buy_count + sell_count) * execution_cost_per_order
                        )
                    cost_deduction = fee_amount / current_value if current_value > 0 else 0.0

                end_value = pre_fee_value - fee_amount
                port_ret = end_value / current_value - 1 if current_value > 0 else 0.0

                portfolio_returns.append(
                    {
                        "date": hd,
                        "return": port_ret,
                        "gross_return": gross_port_ret,
                        "stock_slot_return": stock_slot_return,
                        "stock_pct": current_stock_pct,
                        "target_stock_pct": target_stock_pct if is_rebal else current_stock_pct,
                        "regime": current_regime,
                        "opportunity": current_opp,
                        "market_dd": current_mkt_dd,
                        "cost": cost_deduction,
                        "fee_amount": fee_amount,
                        "sell_count": sell_count if is_rebal else 0,
                        "buy_count": buy_count if is_rebal else 0,
                        "blocked_sell_count": blocked_sell_count if is_rebal else 0,
                        "blocked_buy_count": blocked_buy_count if is_rebal else 0,
                        "t1_locked_count": t1_locked_count if is_rebal else 0,
                        "volume_capped_count": volume_capped_count if is_rebal else 0,
                        "lot_blocked_count": lot_blocked_count if is_rebal else 0,
                        "cash_slot_count": current_cash_slot_count,
                        "missing_count": missing_count,
                    }
                )
                total_fee_amount += fee_amount
                current_value = end_value

                if _DAY_PROGRESS_HOOK is not None:
                    try:
                        _DAY_PROGRESS_HOOK(
                            hd,
                            port_ret,
                            current_value / initial_capital,
                            current_stock_pct,
                            current_value,
                        )
                    except Exception:
                        pass

                if is_rebal:
                    current_held_symbols = next_held_symbols.copy()
                    if enable_vol_norm:
                        current_symbol_weights = next_symbol_weights.copy()
                    current_cash_slot_count = next_cash_slot_count
                    current_stock_pct = target_stock_pct
                    current_regime = target_regime
                    current_opp = target_opp
                    current_mkt_dd = target_mkt_dd
                    # T+1 lock：更新持仓天数
                    if t1_sell_lock_days > 0 and rebal_result is not None:
                        for _s in rebal_result.get("actual_sell", set()):
                            current_hold_days.pop(_s, None)
                        for _s in rebal_result.get("actual_buy", set()):
                            current_hold_days[_s] = 0

        if not portfolio_returns:
            logger.error("无有效回测数据")
            return BacktestResult(
                daily_returns=pd.Series(dtype=float),
                portfolio_value=pd.Series(dtype=float),
            )

        df_result = pd.DataFrame(portfolio_returns).set_index("date")
        df_result.index = pd.to_datetime(df_result.index)

        # P0 约束诊断日志
        if next_open_execution or t1_sell_lock_days > 0 or volume_cap_pct or enforce_lot_size:
            total_t1 = int(df_result.get("t1_locked_count", pd.Series(0)).sum())
            total_vc = int(df_result.get("volume_capped_count", pd.Series(0)).sum())
            total_lot = int(df_result.get("lot_blocked_count", pd.Series(0)).sum())
            total_blk = int(df_result.get("blocked_buy_count", pd.Series(0)).sum())
            logger.info(
                f"[P0诊断] 涨停阻买={total_blk}次 | T+1锁定={total_t1}次 | "
                f"成交量约束={total_vc}次 | 资金不足1手={total_lot}次"
            )

        daily_returns = df_result["return"]
        portfolio_value = (1 + daily_returns).cumprod()

        results_dir = Path(CONFIG.get("paths.results", "./results"))
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        universe = getattr(strategy, "universe", "all") if strategy else "all"
        if universe == "csi300":
            universe_tag = "historical_csi300"
        elif universe == "csi800":
            universe_tag = "historical_csi800"
        elif universe == "csi1000":
            universe_tag = "historical_csi1000"
        else:
            universe_tag = "all_market"
        results_file = results_dir / f"backtest_{strategy_slug}_{universe_tag}_{timestamp}.csv"
        df_result.to_csv(results_file)

        return BacktestResult(
            daily_returns=daily_returns,
            portfolio_value=portfolio_value,
            metadata={
                "results_file": str(results_file),
                "strategy_name": strategy_name,
                "universe": universe,
                "total_fee_amount": total_fee_amount,
                "fee_ratio_to_initial": total_fee_amount / initial_capital
                if initial_capital > 0
                else 0.0,
                "data_fingerprint": _fingerprint_raw_data(valid_instruments),
            },
        )


# ============================================================
# 主程序入口
# ============================================================


def main(strategy=None):
    """主程序入口"""
    engine = QlibBacktestEngine()
    result = engine.run(strategy=strategy)
    initial_capital = CONFIG.get("initial_capital", 500000)
    result.print_summary(initial_capital)

    if result.metadata.get("results_file"):
        print(f"\n  [OK] 结果已保存: {result.metadata['results_file']}")

    return result


if __name__ == "__main__":
    main()
