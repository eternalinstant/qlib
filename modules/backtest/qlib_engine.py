"""
Qlib 回测引擎
使用共享选股 + 仓位控制进行回测
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from config.config import CONFIG
from core.selection import SELECTION_CSV
from core.qlib_init import init_qlib, load_features_safe
from core.universe import filter_instruments, is_st_on_date
from modules.backtest.base import BacktestResult, BacktestEngine
from utils.logger import setup_logger, TradeLogger


CHINEXT_REFORM_DATE = pd.Timestamp("2020-08-24")
PRICE_LIMIT_TOL = 1e-6

# 国债ETF数据路径
BOND_ETF_PATH = Path(__file__).parent.parent.parent / "data" / "tushare" / "bond_etf_daily.parquet"
EMPTY_RAW_DAY_QUOTES = pd.DataFrame(columns=["open", "close", "prev_close"])


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


def _raw_data_root() -> Path:
    qlib_root = Path(CONFIG.get("paths.qlib_data", "~/code/qlib/data/qlib_data/cn_data")).expanduser()
    return qlib_root.parent / "raw_data"


def _raw_data_path_for_instrument(instrument: str) -> Path:
    return _raw_data_root() / f"{instrument[:2].lower()}{instrument[2:]}.parquet"


def _round_limit_price(value: float) -> float:
    if pd.isna(value):
        return np.nan
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _get_price_limit_pct(instrument: str, trade_date, is_st: bool = False) -> float:
    trade_ts = pd.Timestamp(trade_date)
    if is_st:
        return 0.05
    if instrument.startswith("BJ"):
        return 0.30
    if instrument.startswith("SH688"):
        return 0.20
    if instrument.startswith("SZ300") and trade_ts >= CHINEXT_REFORM_DATE:
        return 0.20
    return 0.10


def _get_limit_prices(instrument: str, trade_date, prev_close: float, is_st: bool = False):
    if pd.isna(prev_close) or prev_close <= 0:
        return np.nan, np.nan
    pct = _get_price_limit_pct(instrument, trade_date, is_st=is_st)
    up_limit = _round_limit_price(prev_close * (1 + pct))
    down_limit = _round_limit_price(prev_close * (1 - pct))
    return up_limit, down_limit


def _can_buy_at_open(instrument: str, trade_date, open_price: float, prev_close: float, is_st: bool = False) -> bool:
    if pd.isna(open_price) or pd.isna(prev_close) or open_price <= 0 or prev_close <= 0:
        return False
    up_limit, _ = _get_limit_prices(instrument, trade_date, prev_close, is_st=is_st)
    if pd.isna(up_limit):
        return False
    return float(open_price) < float(up_limit) - PRICE_LIMIT_TOL


def _can_sell_at_open(instrument: str, trade_date, open_price: float, prev_close: float, is_st: bool = False) -> bool:
    if pd.isna(open_price) or pd.isna(prev_close) or open_price <= 0 or prev_close <= 0:
        return False
    _, down_limit = _get_limit_prices(instrument, trade_date, prev_close, is_st=is_st)
    if pd.isna(down_limit):
        return False
    return float(open_price) > float(down_limit) + PRICE_LIMIT_TOL


def _ensure_tradability_constraints_supported(block_limit_up_buy: bool, block_limit_down_sell: bool) -> None:
    """启用成交约束前，必须确认 raw_data 原始日线目录存在。"""
    if not (block_limit_up_buy or block_limit_down_sell):
        return
    raw_root = _raw_data_root()
    if raw_root.exists():
        return
    raise FileNotFoundError(
        f"缺少原始日线目录: {raw_root}，不能正式启用 "
        "block_limit_up_buy / block_limit_down_sell。"
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


def _load_raw_trade_quotes(instruments, start_date: str, end_date: str) -> pd.DataFrame:
    if not instruments:
        return pd.DataFrame(columns=["open", "close", "prev_close"])

    root = _raw_data_root()
    lookback_start = pd.Timestamp(start_date) - pd.Timedelta(days=10)
    end_ts = pd.Timestamp(end_date)
    frames = []
    missing_files = []

    for instrument in sorted(set(instruments)):
        path = _raw_data_path_for_instrument(instrument)
        if not path.exists():
            missing_files.append(instrument)
            continue

        df = pd.read_parquet(path, columns=["date", "open", "close"])
        if df.empty:
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df[(df["date"] >= lookback_start) & (df["date"] <= end_ts)].copy()
        if df.empty:
            continue

        df["instrument"] = instrument
        frames.append(df)

    if missing_files:
        preview = ", ".join(missing_files[:10])
        suffix = " ..." if len(missing_files) > 10 else ""
        print(f"[WARN] raw_data 缺少 {len(missing_files)} 个标的文件，按不可买卖处理: {preview}{suffix}")

    if not frames:
        return pd.DataFrame(columns=["open", "close", "prev_close"])

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values(["instrument", "date"])
    raw["prev_close"] = raw.groupby("instrument")["close"].shift(1)
    raw = raw.rename(columns={"date": "datetime"})
    raw = raw[raw["datetime"] >= pd.Timestamp(start_date)].copy()
    raw = raw.drop_duplicates(subset=["datetime", "instrument"], keep="last")
    return raw.set_index(["datetime", "instrument"])[["open", "close", "prev_close"]].sort_index()


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
):
    """按 close-to-close 口径计算调仓日收益。

    这里显式不用同日 open/close 拆收益。当前本地 Qlib 日线字段中，
    $open 与 $close 不能可靠组成同日收益比值；强行拆成隔夜/日内会明显放大回测。
    """
    prev_selected = set(prev_selected)
    selected = set(selected)
    ranked_selected = list(ranked_selected or sorted(selected))

    held_symbols = set(prev_selected & selected)
    sell_count = 0
    buy_count = 0
    blocked_sell_count = 0
    blocked_buy_count = 0

    for symbol in sorted(prev_selected - selected):
        sellable = True
        if block_limit_down_sell:
            row = _quote_row(raw_day_quotes, symbol)
            sellable = bool(
                row is not None and _can_sell_at_open(
                    symbol,
                    trade_date,
                    row.get("open"),
                    row.get("prev_close"),
                    is_st=is_st_on_date(symbol, trade_date),
                )
            )
        if sellable:
            sell_count += 1
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
                row is not None and _can_buy_at_open(
                    symbol,
                    trade_date,
                    row.get("open"),
                    row.get("prev_close"),
                    is_st=is_st_on_date(symbol, trade_date),
                )
            )
        if buyable:
            held_symbols.add(symbol)
            buy_count += 1
            available_buy_slots -= 1
        else:
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
        "sell_count": sell_count,
        "buy_count": buy_count,
        "blocked_sell_count": blocked_sell_count,
        "blocked_buy_count": blocked_buy_count,
        "cash_slot_count": max(topk - len(available_selected), 0),
        "missing_count": len(missing_symbols),
    }


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

    if strategy is not None and getattr(strategy, "position_model", None) == "fixed":
        stock_pct = float(getattr(strategy, "position_params", {}).get("stock_pct", 0.8))
        return stock_pct, "fixed", "none", 0.0

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

        df_px = load_features_safe(
            selected_instruments,
            ["$close"],
            start_time=start_date,
            end_time=end_date,
            freq="day",
        )
        df_px.columns = ["close"]
        df_px = df_px.sort_index()

        valid_instruments = filter_instruments(
            df_px.index.get_level_values("instrument").unique().tolist(),
            exclude_st=False,
        )
        df_px = df_px[df_px.index.get_level_values("instrument").isin(valid_instruments)].copy()
        df_px["prev_close"] = df_px.groupby(level="instrument")["close"].shift(1)
        df_px["daily_ret"] = df_px["close"] / df_px["prev_close"] - 1
        df_px = df_px.replace([np.inf, -np.inf], np.nan)
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

        initial_capital = float(CONFIG.get("initial_capital", 500000))
        current_value = initial_capital

        trading_cost = getattr(strategy, "trading_cost", {}) if strategy else {}
        buy_commission_rate = trading_cost.get("buy_commission_rate", trading_cost.get("open_cost", 0.0003))
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
        _ensure_tradability_constraints_supported(block_limit_up_buy, block_limit_down_sell)
        ranked_selection_orders = {}
        raw_quotes = pd.DataFrame()
        if block_limit_up_buy or block_limit_down_sell:
            ranked_selection_orders = _load_ranked_selection_orders(strategy)
            required_instruments = {
                symbol
                for ranked_symbols in ranked_selection_orders.values()
                for symbol in ranked_symbols
            }
            raw_quotes = _load_raw_trade_quotes(required_instruments, start_date, end_date)
        raw_quotes_by_date = _split_by_datetime(raw_quotes)

        current_held_symbols = set()  # 上一个收盘后真实持仓（调仓失败后会偏离 target）
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

        for i, rebal_date in enumerate(monthly_dates_list[:-1]):
            selected = date_to_symbols.get(rebal_date, set())
            if len(selected) < topk:
                continue

            target_stock_pct, target_regime, target_opp, target_mkt_dd = target_allocation_by_date[rebal_date]
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

                is_rebal = (j == 0)

                stock_slot_return = 0.0
                sell_count = 0
                buy_count = 0
                blocked_sell_count = 0
                blocked_buy_count = 0
                missing_count = 0
                held_sum, _, held_missing = _sum_symbol_returns(
                    day_returns,
                    current_held_symbols,
                    "daily_ret",
                    penalized_missing=penalized_missing,
                )
                stock_slot_return = held_sum / topk if topk > 0 else 0.0
                missing_count = len(held_missing)

                # 获取当天的国债ETF收益率（如果有数据）
                bond_daily_ret = bond_etf_map.get(hd, default_bond_daily_ret)

                stock_return_component = stock_slot_return + (
                    current_cash_slot_count / topk * bond_daily_ret if topk > 0 else 0.0
                )

                gross_port_ret = current_stock_pct * stock_return_component + (1 - current_stock_pct) * bond_daily_ret
                pre_fee_value = current_value * (1 + gross_port_ret)

                fee_amount = 0.0
                cost_deduction = 0.0
                next_held_symbols = current_held_symbols.copy()
                next_cash_slot_count = current_cash_slot_count
                if is_rebal:
                    raw_day_quotes = None
                    if block_limit_up_buy or block_limit_down_sell:
                        raw_day_quotes = raw_quotes_by_date.get(hd, EMPTY_RAW_DAY_QUOTES)
                    rebal_result = _compute_rebalance_day(
                        day_returns,
                        selected=target_selected,
                        prev_selected=current_held_symbols,
                        topk=topk,
                        penalized_missing=penalized_missing,
                        ranked_selected=ranked_selected,
                        raw_day_quotes=raw_day_quotes,
                        trade_date=hd,
                        block_limit_up_buy=block_limit_up_buy,
                        block_limit_down_sell=block_limit_down_sell,
                    )
                    next_held_symbols = rebal_result["held_symbols"]
                    next_cash_slot_count = rebal_result["cash_slot_count"]
                    sell_count = rebal_result["sell_count"]
                    buy_count = rebal_result["buy_count"]
                    blocked_sell_count = rebal_result["blocked_sell_count"]
                    blocked_buy_count = rebal_result["blocked_buy_count"]

                    per_position_value = pre_fee_value * target_stock_pct / topk if topk > 0 else 0.0
                    buy_fee_per_order = max(per_position_value * buy_commission_rate, min_buy_commission) if buy_count > 0 else 0.0
                    sell_commission_per_order = max(per_position_value * sell_commission_rate, min_sell_commission) if sell_count > 0 else 0.0
                    sell_stamp_tax_per_order = per_position_value * sell_stamp_tax_rate if sell_count > 0 else 0.0
                    execution_cost_per_order = per_position_value * (slippage_rate + impact_rate)
                    fee_amount = (
                        buy_count * buy_fee_per_order +
                        sell_count * (sell_commission_per_order + sell_stamp_tax_per_order) +
                        (buy_count + sell_count) * execution_cost_per_order
                    )
                    cost_deduction = fee_amount / current_value if current_value > 0 else 0.0

                end_value = pre_fee_value - fee_amount
                port_ret = end_value / current_value - 1 if current_value > 0 else 0.0

                portfolio_returns.append({
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
                    "cash_slot_count": current_cash_slot_count,
                    "missing_count": missing_count,
                })
                total_fee_amount += fee_amount
                current_value = end_value

                if is_rebal:
                    current_held_symbols = next_held_symbols.copy()
                    current_cash_slot_count = next_cash_slot_count
                    current_stock_pct = target_stock_pct
                    current_regime = target_regime
                    current_opp = target_opp
                    current_mkt_dd = target_mkt_dd

        if not portfolio_returns:
            logger.error("无有效回测数据")
            return BacktestResult(
                daily_returns=pd.Series(dtype=float),
                portfolio_value=pd.Series(dtype=float),
            )

        df_result = pd.DataFrame(portfolio_returns).set_index("date")
        df_result.index = pd.to_datetime(df_result.index)

        daily_returns = df_result["return"]
        portfolio_value = (1 + daily_returns).cumprod()

        results_dir = Path(CONFIG.get("paths.results", "./results"))
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        universe = getattr(strategy, "universe", "all") if strategy else "all"
        universe_tag = "historical_csi300" if universe == "csi300" else "all_market"
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
                "fee_ratio_to_initial": total_fee_amount / initial_capital if initial_capital > 0 else 0.0,
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
