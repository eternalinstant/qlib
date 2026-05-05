#!/usr/bin/env python3
"""
全自动实盘模拟 — Paper Trading Engine

每日收盘后运行：
1. 更新 tushare 原始数据（daily_basic, moneyflow, fina_indicator, income, cashflow, balancesheet）
2. 增量重建 factor_data.parquet
3. 用已训练模型打分，生成最新选股信号
4. 跟踪虚拟持仓，计算 P&L
5. 输出交易日志

用法:
    python3 scripts/paper_trading.py                    # 完整运行（更新+打分+交易）
    python3 scripts/paper_trading.py --skip-update       # 跳过数据更新（用现有数据）
    python3 scripts/paper_trading.py --report weekly     # 生成周报
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
TUSHARE_DIR = DATA_DIR / "tushare"
QLIB_DIR = DATA_DIR / "qlib_data" / "cn_data"
FACTOR_DATA_PATH = QLIB_DIR / "factor_data.parquet"

CONFIG_PATH = PROJECT_ROOT / "config" / "models" / "push25_cq10_k8d2_very_tight.yaml"
MODEL_BUNDLE_PATH = PROJECT_ROOT / "results" / "model_signals" / "push25_cq10_k8d2_very_tight" / "model_bundle.pkl"

PAPER_DIR = PROJECT_ROOT / "results" / "paper_trading"
POSITION_FILE = PAPER_DIR / "positions.json"
TRADE_LOG = PAPER_DIR / "trade_log.csv"
NAV_LOG = PAPER_DIR / "nav_log.csv"
REPORT_DIR = PAPER_DIR / "reports"

# Ensure dirs exist before logging
PAPER_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PAPER_DIR / "paper_trading.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
def _load_env():
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().strip().splitlines():
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


# ---------------------------------------------------------------------------
# Step 1: Update tushare data
# ---------------------------------------------------------------------------
def update_tushare_data() -> bool:
    """更新 tushare 原始数据"""
    log.info("=" * 60)
    log.info("Step 1: 更新 tushare 原始数据")
    log.info("=" * 60)

    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from data_update import (
            update_daily_basic,
            update_moneyflow,
            update_fina_indicator,
            update_income,
            update_cashflow,
            update_balancesheet,
        )

        # moneyflow 数据当前未被策略使用，跳过更新（文件损坏后重建耗时过长）
        _skip = {"moneyflow"}
        results = {}
        for name, fn in [
            ("daily_basic", update_daily_basic),
            ("moneyflow", update_moneyflow),
            ("fina_indicator", update_fina_indicator),
            ("income", update_income),
            ("cashflow", update_cashflow),
            ("balancesheet", update_balancesheet),
        ]:
            if name in _skip:
                results[name] = "SKIP"
                log.info(f"  {name}: SKIP (not used by strategy)")
                continue
            log.info(f"  更新 {name}...")
            ok = fn()
            results[name] = "OK" if ok else "FAIL"
            log.info(f"  {name}: {results[name]}")

        # moneyflow 等跳过的不影响整体判断
        failed = [k for k, v in results.items() if v == "FAIL"]
        all_ok = len(failed) == 0
        log.info(f"数据更新结果: {results}")
        if failed:
            log.error(f"数据源更新失败: {failed}")
        return all_ok
    except Exception as e:
        log.error(f"数据更新失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Step 2: Rebuild factor_data.parquet (incremental)
# ---------------------------------------------------------------------------
def rebuild_factor_data() -> bool:
    """增量重建 factor_data.parquet"""
    log.info("=" * 60)
    log.info("Step 2: 增量重建 factor_data.parquet")
    log.info("=" * 60)

    try:
        from modules.data.tushare_to_qlib import TushareToQlibConverter

        converter = TushareToQlibConverter()
        df = converter.convert()
        if df is not None:
            converter.save(df)
            # 更新 qlib close.day.bin（alpha158 需要）
            n = converter.update_close_bins()
            log.info(f"close.day.bin 已更新: {n} 只股票")
            # Verify
            latest = df["datetime"].max()
            log.info(f"factor_data.parquet 已更新，最新日期: {latest}")
            return True
        else:
            log.error("convert() 返回 None")
            return False
    except Exception as e:
        log.error(f"重建 factor_data 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Step 3: Score & generate signals
# ---------------------------------------------------------------------------
def generate_signals() -> Optional[pd.DataFrame]:
    """用已训练模型生成最新选股信号"""
    log.info("=" * 60)
    log.info("Step 3: 生成选股信号")
    log.info("=" * 60)

    try:
        # Load model bundle
        with open(MODEL_BUNDLE_PATH, "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]
        log.info(f"模型: {type(model).__name__}, 特征: {feature_columns}")

        # Load config for selection parameters
        from modules.modeling.predictive_signal import (
            load_predictive_config,
            load_feature_frame,
        )
        cfg = load_predictive_config(CONFIG_PATH)
        selection_cfg = cfg.get("selection", {})
        scoring_cfg = cfg.get("scoring", {})

        # Determine latest rebalance date
        factor_df = pd.read_parquet(FACTOR_DATA_PATH)
        factor_df["datetime"] = pd.to_datetime(factor_df["datetime"])
        latest_date = factor_df["datetime"].max()
        end_date = latest_date.strftime("%Y-%m-%d")
        start_date = (latest_date - timedelta(days=60)).strftime("%Y-%m-%d")

        log.info(f"数据范围: {start_date} ~ {end_date}")

        # Load feature frame using the proper pipeline (hybrid: parquet + alpha158)
        # Override scoring end_date to latest available
        cfg["scoring"]["end_date"] = end_date
        cfg["scoring"]["start_date"] = start_date
        rebalance_freq = str(selection_cfg.get("freq", "biweek"))

        frame, rebalance_dates, actual_columns = load_feature_frame(
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=rebalance_freq,
            feature_columns=feature_columns,
            data_cfg=cfg.get("data", {}),
            selection_cfg=selection_cfg,
        )

        if frame.empty:
            log.error("特征帧为空")
            return None

        log.info(f"特征帧: {frame.shape}, 调仓日: {len(rebalance_dates)} 个")

        # Get the latest rebalance date's data
        latest_rb = rebalance_dates.max()
        latest_features = frame.xs(latest_rb, level="datetime", drop_level=False)
        log.info(f"最新调仓日: {latest_rb.strftime('%Y-%m-%d')}, 股票数: {len(latest_features)}")

        # Predict
        features = latest_features[feature_columns].copy()
        features = features.replace([np.inf, -np.inf], np.nan)

        scores = model.predict(features)

        # Build result — scores aligns with latest_features index
        result_data = []
        instruments_idx = latest_features.index
        for i, score in enumerate(scores):
            dt, sym = instruments_idx[i]
            result_data.append({
                "date": dt.strftime("%Y-%m-%d"),
                "symbol": sym,
                "score": float(score),
            })
        result_df = pd.DataFrame(result_data)

        # Top-K selection
        topk = selection_cfg.get("topk", 8)
        result_df = result_df.dropna(subset=["score"])
        result_df = result_df.sort_values("score", ascending=False).head(topk)
        result_df.insert(2, "rank", range(1, len(result_df) + 1))

        log.info(f"选股信号 ({latest_rb.strftime('%Y-%m-%d')}):")
        for _, row in result_df.iterrows():
            log.info(f"  {row['rank']}. {row['symbol']} ({row['score']:.4f})")

        return result_df
    except Exception as e:
        log.error(f"生成信号失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Qlib init helper
# ---------------------------------------------------------------------------
_qlib_initialized = False

def _ensure_qlib_init():
    global _qlib_initialized
    if _qlib_initialized:
        return
    import qlib
    qlib.init(provider_uri=str(QLIB_DIR), region="cn")
    _qlib_initialized = True


# ---------------------------------------------------------------------------
# Step 4: Execute trades (virtual portfolio)
# ---------------------------------------------------------------------------
class PaperPortfolio:
    """虚拟持仓管理器"""

    INITIAL_CAPITAL = 1_000_000  # 100万初始资金
    COMMISSION_RATE = 0.0003     # 买卖佣金
    STAMP_TAX_RATE = 0.0010      # 卖出印花税
    SLIPPAGE_BPS = 5.0           # 滑点 5bp

    def __init__(self):
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        self.positions = self._load_positions()
        self.nav_history = self._load_nav_log()

    def _load_positions(self) -> dict:
        if POSITION_FILE.exists():
            return json.loads(POSITION_FILE.read_text())
        return {
            "cash": self.INITIAL_CAPITAL,
            "holdings": {},  # {symbol: {shares: int, cost_price: float, buy_date: str}}
            "created_at": datetime.now().isoformat(),
        }

    def _save_positions(self):
        POSITION_FILE.write_text(json.dumps(self.positions, indent=2, ensure_ascii=False))

    def _load_nav_log(self) -> pd.DataFrame:
        if NAV_LOG.exists():
            return pd.read_csv(NAV_LOG, parse_dates=["date"])
        return pd.DataFrame(columns=["date", "nav", "cash", "holdings_value", "pnl", "pnl_pct", "holdings"])

    def _save_nav_log(self):
        NAV_LOG.parent.mkdir(parents=True, exist_ok=True)
        self.nav_history.to_csv(NAV_LOG, index=False)

    def get_current_holdings(self) -> dict:
        """返回当前持仓 {symbol: {shares, cost_price, buy_date}}"""
        return dict(self.positions.get("holdings", {}))

    def execute_rebalance(self, signals: pd.DataFrame, prices: dict, trade_date: str):
        """
        根据信号调仓。
        signals: DataFrame with columns [symbol, score, rank]
        prices: {symbol: close_price} 当日收盘价
        """
        target_symbols = set(signals["symbol"].tolist())
        current_holdings = self.get_current_holdings()
        current_symbols = set(current_holdings.keys())

        # 卖出不在目标列表中的股票
        to_sell = current_symbols - target_symbols
        for sym in to_sell:
            pos = current_holdings[sym]
            price = prices.get(sym)
            if price is None or price <= 0:
                log.warning(f"  跳过卖出 {sym}: 无价格数据")
                continue
            shares = pos["shares"]
            sell_value = shares * price
            # Costs
            commission = max(sell_value * self.COMMISSION_RATE, 5.0)
            stamp_tax = sell_value * self.STAMP_TAX_RATE
            slippage = sell_value * self.SLIPPAGE_BPS / 10000
            net_proceeds = sell_value - commission - stamp_tax - slippage

            self.positions["cash"] += net_proceeds
            del self.positions["holdings"][sym]

            pnl = net_proceeds - pos["shares"] * pos["cost_price"]
            log.info(
                f"  卖出 {sym}: {shares}股 @ {price:.2f}, "
                f"费用 {(commission+stamp_tax+slippage):.2f}, "
                f"盈亏 {pnl:+.2f}"
            )
            self._log_trade(trade_date, "SELL", sym, shares, price, commission + stamp_tax + slippage, pnl)

        # 计算可用资金和目标持仓
        total_value = self._calc_total_value(prices)
        stock_pct = 0.80  # 80%仓位
        target_stock_value = total_value * stock_pct
        n_targets = len(signals)
        if n_targets == 0:
            return
        per_stock_value = target_stock_value / n_targets

        # 买入新股票 / 调整持仓
        for _, row in signals.iterrows():
            sym = row["symbol"]
            price = prices.get(sym)
            if price is None or price <= 0:
                log.warning(f"  跳过买入 {sym}: 无价格数据")
                continue

            current_pos = self.positions["holdings"].get(sym)
            if current_pos:
                # 已持有，检查是否需要调整（当前暂不加减仓，保持原仓位）
                continue

            # 计算可买股数（100股整数倍）
            buy_value = min(per_stock_value, self.positions["cash"] * 0.95)  # 留5%现金余量
            if buy_value < price * 100:
                log.warning(f"  跳过买入 {sym}: 资金不足 (可用 {self.positions['cash']:.0f}, 需要 {price*100:.0f})")
                continue

            commission = max(buy_value * self.COMMISSION_RATE, 5.0)
            slippage = buy_value * self.SLIPPAGE_BPS / 10000
            actual_buy_value = buy_value - commission - slippage
            shares = int(actual_buy_value / price / 100) * 100
            if shares <= 0:
                continue

            actual_cost = shares * price
            actual_commission = max(actual_cost * self.COMMISSION_RATE, 5.0)
            actual_slippage = actual_cost * self.SLIPPAGE_BPS / 10000
            total_cost = actual_cost + actual_commission + actual_slippage

            self.positions["cash"] -= total_cost
            self.positions["holdings"][sym] = {
                "shares": shares,
                "cost_price": price,
                "buy_date": trade_date,
            }

            log.info(
                f"  买入 {sym}: {shares}股 @ {price:.2f}, "
                f"费用 {(actual_commission+actual_slippage):.2f}, "
                f"成本 {total_cost:.2f}"
            )
            self._log_trade(trade_date, "BUY", sym, shares, price, actual_commission + actual_slippage, 0)

        self._save_positions()

    def update_nav(self, trade_date: str, prices: dict):
        """更新净值记录"""
        cash = self.positions["cash"]
        holdings_value = 0
        for sym, pos in self.positions.get("holdings", {}).items():
            price = prices.get(sym, pos["cost_price"])
            holdings_value += pos["shares"] * price

        total = cash + holdings_value
        pnl = total - self.INITIAL_CAPITAL
        pnl_pct = pnl / self.INITIAL_CAPITAL * 100

        holdings_str = ",".join(
            f"{sym}:{pos['shares']}" for sym, pos in sorted(self.positions.get("holdings", {}).items())
        )

        row = pd.DataFrame([{
            "date": trade_date,
            "nav": round(total, 2),
            "cash": round(cash, 2),
            "holdings_value": round(holdings_value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "holdings": holdings_str,
        }])
        self.nav_history = pd.concat([self.nav_history, row], ignore_index=True)
        self._save_nav_log()

        log.info(f"  NAV: {total:,.0f} | 现金: {cash:,.0f} | 持仓: {holdings_value:,.0f} | 盈亏: {pnl:+,.0f} ({pnl_pct:+.2f}%)")

    def _calc_total_value(self, prices: dict) -> float:
        cash = self.positions["cash"]
        holdings_value = 0
        for sym, pos in self.positions.get("holdings", {}).items():
            price = prices.get(sym, pos["cost_price"])
            holdings_value += pos["shares"] * price
        return cash + holdings_value

    def _log_trade(self, date: str, action: str, symbol: str, shares: int, price: float, fee: float, pnl: float):
        TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
        file_exists = TRADE_LOG.exists()
        with open(TRADE_LOG, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("date,action,symbol,shares,price,fee,pnl\n")
            f.write(f"{date},{action},{symbol},{shares},{price:.2f},{fee:.2f},{pnl:.2f}\n")

    def get_portfolio_summary(self) -> dict:
        """返回当前组合摘要"""
        holdings = self.get_current_holdings()
        last_nav = self.nav_history["nav"].iloc[-1] if not self.nav_history.empty else self.INITIAL_CAPITAL
        return {
            "nav": last_nav,
            "cash": self.positions["cash"],
            "n_holdings": len(holdings),
            "holdings": list(holdings.keys()),
            "created_at": self.positions.get("created_at", "unknown"),
            "pnl": last_nav - self.INITIAL_CAPITAL,
            "pnl_pct": (last_nav - self.INITIAL_CAPITAL) / self.INITIAL_CAPITAL * 100,
        }


# ---------------------------------------------------------------------------
# Step 5: Get prices for holdings
# ---------------------------------------------------------------------------
def get_latest_prices(symbols: list[str]) -> dict[str, float]:
    """获取最新收盘价（从 qlib features）"""
    _ensure_qlib_init()
    from qlib.data import D

    prices = {}
    # qlib uses lowercase format: sh600000, sz000001
    # Our signals use SH600000 format
    symbol_map = {s: s.lower() for s in symbols}

    end_time = datetime.now().strftime("%Y-%m-%d")
    start_time = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    for orig_sym, qlib_sym in symbol_map.items():
        try:
            fields = D.features(
                instruments=[qlib_sym],
                fields=["$close"],
                start_time=start_time,
                end_time=end_time,
            )
            if fields is not None and not fields.empty:
                latest_close = fields.iloc[-1]["$close"]
                if np.isfinite(latest_close):
                    prices[orig_sym] = float(latest_close)
        except Exception as e:
            log.debug(f"获取 {orig_sym} ({qlib_sym}) 价格失败: {e}")

    return prices


def get_prices_from_tushare(symbols: list[str]) -> dict[str, float]:
    """从 tushare daily_basic 获取价格（备选方案）"""
    try:
        db = pd.read_parquet(TUSHARE_DIR / "daily_basic.parquet")
        db["trade_date"] = pd.to_datetime(db["trade_date"], format="%Y%m%d")

        latest = db["trade_date"].max()
        latest_db = db[db["trade_date"] == latest]

        prices = {}
        for sym in symbols:
            # Convert SH600000 -> 600000.SH format
            if sym.startswith("SH"):
                ts_code = sym[2:] + ".SH"
            elif sym.startswith("SZ"):
                ts_code = sym[2:] + ".SZ"
            elif sym.startswith("BJ"):
                ts_code = sym[2:] + ".BJ"
            else:
                continue

            row = latest_db[latest_db["ts_code"] == ts_code]
            if not row.empty and "close" in row.columns:
                close = row.iloc[0]["close"]
                if pd.notna(close) and close > 0:
                    prices[sym] = float(close)

        return prices
    except Exception as e:
        log.warning(f"从 tushare 获取价格失败: {e}")
        return {}


# ---------------------------------------------------------------------------
# Weekly report
# ---------------------------------------------------------------------------
def generate_weekly_report() -> Optional[str]:
    """生成周报文本"""
    log.info("=" * 60)
    log.info("生成周报")
    log.info("=" * 60)

    nav_df = pd.read_csv(NAV_LOG, parse_dates=["date"]) if NAV_LOG.exists() else pd.DataFrame()
    trade_df = pd.read_csv(TRADE_LOG) if TRADE_LOG.exists() else pd.DataFrame()

    if nav_df.empty:
        return "暂无净值数据，周报无法生成。"

    latest = nav_df.iloc[-1]
    prev_week = nav_df.iloc[-6] if len(nav_df) >= 6 else nav_df.iloc[0]

    # Weekly return
    weekly_return = (latest["nav"] / prev_week["nav"] - 1) * 100

    # Total return
    total_return = (latest["nav"] / 1000000 - 1) * 100

    # Max drawdown
    nav_series = nav_df["nav"].values
    peak = np.maximum.accumulate(nav_series)
    drawdown = (nav_series - peak) / peak * 100
    max_dd = drawdown.min()

    # Trades this week
    if not trade_df.empty:
        trade_df["date"] = pd.to_datetime(trade_df["date"])
        week_ago = pd.Timestamp(latest["date"]) - timedelta(days=7)
        recent_trades = trade_df[trade_df["date"] >= week_ago]
    else:
        recent_trades = pd.DataFrame()

    lines = [
        f"📊 **实盘模拟周报** ({prev_week['date'].strftime('%m/%d')} ~ {latest['date'].strftime('%m/%d')})",
        "",
        f"**净值**: ¥{latest['nav']:,.0f}",
        f"**本周收益**: {weekly_return:+.2f}%",
        f"**累计收益**: {total_return:+.2f}%",
        f"**最大回撤**: {max_dd:.2f}%",
        f"**现金**: ¥{latest['cash']:,.0f} ({latest['cash']/latest['nav']*100:.1f}%)",
        f"**持仓市值**: ¥{latest['holdings_value']:,.0f}",
        "",
        f"**当前持仓** ({latest.get('holdings', '无')}):",
    ]

    holdings = json.loads(POSITION_FILE.read_text()).get("holdings", {}) if POSITION_FILE.exists() else {}
    if holdings:
        for sym, pos in sorted(holdings.items()):
            lines.append(f"  • {sym}: {pos['shares']}股, 成本 {pos['cost_price']:.2f}, 买入日 {pos['buy_date']}")
    else:
        lines.append("  空仓")

    if not recent_trades.empty:
        lines.append("")
        lines.append("**本周交易**:")
        for _, t in recent_trades.iterrows():
            action = "买入" if t["action"] == "BUY" else "卖出"
            pnl_str = f", 盈亏 {t['pnl']:+.0f}" if t["pnl"] != 0 else ""
            lines.append(f"  • {t['date'].strftime('%m/%d')} {action} {t['symbol']} {int(t['shares'])}股 @ {t['price']:.2f}{pnl_str}")

    report = "\n".join(lines)
    log.info(f"周报:\n{report}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="实盘模拟引擎")
    parser.add_argument("--skip-update", action="store_true", help="跳过数据更新")
    parser.add_argument("--report", choices=["weekly", "daily"], help="生成报告（不执行交易）")
    parser.add_argument("--init", action="store_true", help="初始化虚拟账户（重置）")
    parser.add_argument("--push", action="store_true", help="执行后推送日报到 Telegram")
    args = parser.parse_args()

    _load_env()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Init mode
    if args.init:
        if POSITION_FILE.exists():
            POSITION_FILE.unlink()
        if NAV_LOG.exists():
            NAV_LOG.unlink()
        if TRADE_LOG.exists():
            TRADE_LOG.unlink()
        log.info("虚拟账户已重置")
        return

    # Report mode
    if args.report == "weekly":
        report = generate_weekly_report()
        print(report)
        return
    elif args.report == "daily":
        nav_df = pd.read_csv(NAV_LOG, parse_dates=["date"]) if NAV_LOG.exists() else pd.DataFrame()
        if nav_df.empty:
            print("暂无净值数据")
            return
        latest = nav_df.iloc[-1]
        total_return = (latest["nav"] / 1000000 - 1) * 100
        print(
            f"日期: {latest['date'].strftime('%Y-%m-%d')} | "
            f"净值: ¥{latest['nav']:,.0f} | "
            f"累计: {total_return:+.2f}% | "
            f"持仓: {latest.get('holdings', '无')}"
        )
        return

    # Full pipeline
    log.info("🚀 实盘模拟启动")

    # Determine today's trading date
    today = pd.Timestamp.now().normalize()
    # Use tushare trade_cal for rebalance day check (qlib calendar may be outdated)
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        import tushare as ts
        pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))
        cal_df = pro.trade_cal(exchange="SSE", start_date="20260101",
                               end_date=today.strftime("%Y%m%d"))
        trading_days = set(
            pd.to_datetime(cal_df[cal_df["is_open"] == 1]["cal_date"]).dt.normalize()
        )
        is_trading_day = today in trading_days
    except Exception as e:
        log.warning(f"tushare trade_cal 获取失败: {e}, 跳过交易日检查")
        is_trading_day = True  # assume trading day

    if not is_trading_day:
        log.info(f"今天 {today.strftime('%Y-%m-%d')} 非交易日，跳过")
        sys.exit(0)

    # Check if today is a rebalance day (biweekly)
    from core.selection import compute_rebalance_dates
    rebalance_dates = compute_rebalance_dates(
        pd.Series(sorted(trading_days)), freq="biweek"
    )
    is_rebalance_day = today in rebalance_dates
    log.info(f"今日 {today.strftime('%Y-%m-%d')} | 调仓日: {'是' if is_rebalance_day else '否'}")

    portfolio = PaperPortfolio()
    current_holdings = portfolio.get_current_holdings()

    # Step 1: Update data
    if not args.skip_update:
        if not update_tushare_data():
            log.error("数据更新失败，终止")
            sys.exit(1)

    # Step 2: Rebuild factor data
    if not args.skip_update:
        if not rebuild_factor_data():
            log.error("因子数据重建失败，终止")
            sys.exit(1)

    if is_rebalance_day:
        # Rebalance day: full pipeline
        # Step 3: Generate signals
        signals = generate_signals()
        if signals is None or signals.empty:
            log.error("信号生成失败")
            sys.exit(1)

        trade_date = signals["date"].iloc[0]
        target_symbols = signals["symbol"].tolist()

        # Step 4: Get prices
        log.info("=" * 60)
        log.info("Step 4: 获取价格数据")
        log.info("=" * 60)

        all_symbols = list(set(target_symbols) | set(current_holdings.keys()))

        prices = get_latest_prices(all_symbols)
        missing = [s for s in all_symbols if s not in prices]
        if missing:
            log.warning(f"qlib 缺少价格: {missing}，尝试从 tushare 获取")
            tushare_prices = get_prices_from_tushare(missing)
            prices.update(tushare_prices)
            still_missing = [s for s in all_symbols if s not in prices]
            if still_missing:
                log.warning(f"仍缺少价格: {still_missing}")

        log.info(f"获取到 {len(prices)} 个价格")

        # Step 5: Execute rebalance
        log.info("=" * 60)
        log.info("Step 5: 执行调仓")
        log.info("=" * 60)

        portfolio.execute_rebalance(signals, prices, trade_date)
    else:
        # Non-rebalance day: just update NAV with today's prices
        log.info("今日非调仓日，仅更新净值")
        trade_date = today.strftime("%Y-%m-%d")

    # Step 6: Update NAV
    log.info("=" * 60)
    log.info("Step 6: 更新净值")
    log.info("=" * 60)

    current_after = portfolio.get_current_holdings()
    if current_after:
        all_syms = list(current_after.keys())
        final_prices = get_latest_prices(all_syms)
        missing = [s for s in all_syms if s not in final_prices]
        if missing:
            final_prices.update(get_prices_from_tushare(missing))
        portfolio.update_nav(trade_date, final_prices)
    else:
        portfolio.update_nav(trade_date, {})

    # Summary
    summary = portfolio.get_portfolio_summary()
    log.info("=" * 60)
    log.info("✅ 实盘模拟完成")
    log.info(f"  NAV: ¥{summary['nav']:,.0f}")
    log.info(f"  盈亏: {summary['pnl']:+,.0f} ({summary['pnl_pct']:+.2f}%)")
    log.info(f"  持仓: {summary['n_holdings']} 只")
    log.info("=" * 60)

    # Push to Telegram
    if args.push:
        msg = generate_daily_report()
        if msg:
            send_telegram(msg)


# ---------------------------------------------------------------------------
# Telegram push
# ---------------------------------------------------------------------------
def send_telegram(message: str):
    """发送消息到 Telegram（通过 Hermes CLI）"""
    import subprocess
    try:
        # Use hermes send_message via CLI
        # We write the message to a temp file to avoid shell escaping issues
        tmp = PAPER_DIR / "telegram_msg.txt"
        tmp.write_text(message, encoding="utf-8")
        result = subprocess.run(
            ["hermes", "send", "telegram", "--message", message],
            capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            log.info("Telegram 推送成功")
        else:
            log.warning(f"Telegram 推送失败: {result.stderr}")
        tmp.unlink(missing_ok=True)
    except Exception as e:
        log.error(f"Telegram 推送异常: {e}")


def _load_stock_name_map() -> dict:
    """加载股票代码→名称映射。symbol 格式: SH688082 / SZ000001"""
    name_csv = PROJECT_ROOT / "data" / "tushare" / "stock_names.csv"
    if name_csv.exists():
        df = pd.read_csv(name_csv)
        m = {}
        for _, row in df.iterrows():
            sym = str(row["symbol"]).zfill(6)  # 确保补零
            name = row["name"]
            # 6/9 开头 → 上交所(SH)，0/3 开头 → 深交所(SZ)
            if sym.startswith("6") or sym.startswith("9"):
                m[f"SH{sym}"] = name
            else:
                m[f"SZ{sym}"] = name
        return m
    return {}


def generate_daily_report() -> Optional[str]:
    """生成每日报告文本"""
    nav_df = pd.read_csv(NAV_LOG) if NAV_LOG.exists() else pd.DataFrame()
    trade_df = pd.read_csv(TRADE_LOG) if TRADE_LOG.exists() else pd.DataFrame()

    if nav_df.empty:
        return None

    # 加载股票名称映射
    name_map = _load_stock_name_map()

    nav_df["date"] = pd.to_datetime(nav_df["date"], format="mixed")
    latest = nav_df.iloc[-1]
    prev = nav_df.iloc[-2] if len(nav_df) >= 2 else None

    total_return = (latest["nav"] / 1000000 - 1) * 100
    day_return = ((latest["nav"] / prev["nav"]) - 1) * 100 if prev is not None else 0.0

    # Max drawdown
    nav_series = nav_df["nav"].values
    peak = np.maximum.accumulate(nav_series)
    drawdown = (nav_series - peak) / peak * 100
    max_dd = drawdown.min()

    # Sharpe (annualized, rough)
    if len(nav_df) >= 3:
        returns = nav_df["nav"].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    else:
        sharpe = 0

    # Calmar
    calmar = total_return / abs(max_dd) if max_dd < 0 else 0

    lines = [
        f"📊 **实盘模拟日报** ({latest['date'].strftime('%Y-%m-%d')})",
        "",
        f"**净值**: ¥{latest['nav']:,.0f}",
        f"**日收益**: {day_return:+.2f}%" if prev is not None else "**日收益**: N/A (首日建仓)",
        f"**累计收益**: {total_return:+.2f}%",
        f"**最大回撤**: {max_dd:.2f}%",
        f"**Sharpe**: {sharpe:.2f}  |  **Calmar**: {calmar:.2f}",
        f"**现金**: ¥{latest['cash']:,.0f} ({latest['cash']/latest['nav']*100:.1f}%)",
        f"**持仓市值**: ¥{latest['holdings_value']:,.0f}",
        "",
        f"**当前持仓**:",
    ]

    holdings = json.loads(POSITION_FILE.read_text()).get("holdings", {}) if POSITION_FILE.exists() else {}
    if holdings:
        # Get current prices for P&L calculation
        all_syms = list(holdings.keys())
        prices = get_latest_prices(all_syms)
        missing = [s for s in all_syms if s not in prices]
        if missing:
            prices.update(get_prices_from_tushare(missing))

        total_unrealized = 0
        for sym in sorted(holdings.keys()):
            pos = holdings[sym]
            shares = pos["shares"]
            cost = pos["cost_price"]
            price = prices.get(sym, cost)
            mv = shares * price
            pnl = (price - cost) / cost * 100
            total_unrealized += shares * (price - cost)
            name = name_map.get(sym, sym)
            lines.append(f"  • {name}({sym[-6:]}): {shares}股, 成本 {cost:.2f}, 现价 {price:.2f}, 盈亏 {pnl:+.1f}%")

        if total_unrealized != 0:
            lines.append(f"\n**持仓浮盈**: ¥{total_unrealized:+,.0f}")
    else:
        lines.append("  空仓")

    # Today's trades
    if not trade_df.empty:
        trade_df["date"] = pd.to_datetime(trade_df["date"])
        today_trades = trade_df[trade_df["date"].dt.strftime("%Y-%m-%d") == latest["date"].strftime("%Y-%m-%d")]
        if not today_trades.empty:
            lines.append("")
            lines.append("**今日交易**:")
            for _, t in today_trades.iterrows():
                action = "🟢 买入" if t["action"] == "BUY" else "🔴 卖出"
                pnl_str = f", 盈亏 ¥{t['pnl']:+,.0f}" if t["pnl"] != 0 else ""
                sym = t["symbol"]
                name = name_map.get(sym, sym)
                lines.append(f"  {action} {name}({sym[-6:]}) {int(t['shares'])}股 @ {t['price']:.2f}{pnl_str}")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
