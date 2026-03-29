"""
仓位控制模块 — 基于沪深300的趋势跟踪 + 机会捕捉
==========================================================
信号1: MA20/MA60 趋势 → 熊市减仓（最低 50%）
信号2: 120日高点回撤 → 大跌加仓（最高 95%）
两者通过 max() 组合：机会捕捉只向上覆盖，不向下
"""
import struct
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
# 配置
# ============================================================

@dataclass
class MarketConfig:
    # MA 周期
    ma_fast: int = 20
    ma_slow: int = 60
    peak_lookback: int = 120

    # 趋势阈值（MA 比值）— 加宽阈值以减少频繁切换
    strong_bull_threshold: float = 0.05
    mild_bull_threshold: float = 0.02
    mild_bear_threshold: float = -0.02
    strong_bear_threshold: float = -0.05

    # 趋势对应仓位
    regime_allocations: dict = field(default_factory=lambda: {
        "strong_bull": 1.00,
        "mild_bull":   0.90,
        "neutral":     0.80,
        "mild_bear":   0.65,
        "strong_bear": 0.50,
    })

    # 机会捕捉阈值: (回撤阈值, 最低仓位, 标签)
    opportunity_thresholds: list = field(default_factory=lambda: [
        (-0.10, 0.75, "light"),
        (-0.15, 0.85, "moderate"),
        (-0.20, 0.95, "heavy"),
    ])

    # 平滑
    smoothing_factor: float = 0.7   # EMA 中 target 权重
    bond_annual_return: float = 0.03

    # 数据路径
    qlib_data_path: str = None


@dataclass
class AllocationResult:
    stock_pct: float
    cash_pct: float
    regime: str
    opportunity_level: str
    market_drawdown: float
    trend_score: float


# ============================================================
# 控制器
# ============================================================

class MarketPositionController:
    def __init__(self, config: MarketConfig = None):
        self.config = config or MarketConfig()
        # 从全局配置获取数据路径
        if self.config.qlib_data_path is None:
            from config.config import CONFIG
            self.config.qlib_data_path = Path(CONFIG.get("paths.qlib_data",
                "~/code/qlib/data/qlib_data/cn_data")).expanduser()
        self.csi300_close: pd.Series = None
        self.ma_fast: pd.Series = None
        self.ma_slow: pd.Series = None
        self.ma_ratio: pd.Series = None
        self.peak: pd.Series = None
        self.drawdown: pd.Series = None
        self._prev_allocation: float = None

    # ----------------------------------------------------------
    # 数据加载方式1: 使用 Qlib API (推荐)
    # ----------------------------------------------------------
    def load_market_data_v2(self) -> None:
        """使用 Qlib API 加载沪深300数据（推荐方式）"""
        from core.qlib_init import init_qlib, load_features_safe
        init_qlib()

        instruments = ["sh399300"]
        df = load_features_safe(
            instruments, ["$close"],
            start_time="2010-01-01",
            end_time="2030-12-31",
            freq="day"
        )
        df = df.droplevel("instrument")
        df.columns = ["close"]

        self.csi300_close = df["close"].dropna()
        self._compute_indicators()
        print(f"[OK] 沪深300数据加载(API): {len(self.csi300_close)} 行, "
              f"{self.csi300_close.index[0].date()} ~ {self.csi300_close.index[-1].date()}")

    # ----------------------------------------------------------
    # 数据加载方式2: 直接读 Qlib bin 文件 (sz399300 不在 all.txt 中)
    # ----------------------------------------------------------
    def load_market_data(self) -> None:
        data_root = Path(self.config.qlib_data_path)
        cal_path = data_root / "calendars" / "day.txt"
        close_path = data_root / "features" / "sz399300" / "close.day.bin"

        if not close_path.exists():
            raise FileNotFoundError(f"沪深300数据不存在: {close_path}")

        cal_all = pd.read_csv(cal_path, header=None, names=["date"], parse_dates=["date"])

        with open(close_path, "rb") as f:
            raw = f.read()
        n_floats = len(raw) // 4
        values = struct.unpack(f"<{n_floats}f", raw)

        start_idx = int(values[0])
        data_values = values[1:]
        n_data = len(data_values)

        global_cal_len = start_idx + n_data
        local_offset = global_cal_len - len(cal_all)

        series_data = {}
        for i, row in cal_all.iterrows():
            data_pos = (local_offset + i) - start_idx
            if 0 <= data_pos < n_data:
                val = data_values[data_pos]
                if val != 0 and not np.isnan(val):
                    series_data[row["date"]] = val

        self.csi300_close = pd.Series(series_data, dtype=float).sort_index()
        self._compute_indicators()
        print(f"[OK] 沪深300数据加载: {len(self.csi300_close)} 行, "
              f"{self.csi300_close.index[0].date()} ~ {self.csi300_close.index[-1].date()}")

    # ----------------------------------------------------------
    # 指标计算
    # ----------------------------------------------------------
    def _compute_indicators(self) -> None:
        c = self.csi300_close
        self.ma_fast = c.rolling(self.config.ma_fast, min_periods=1).mean()
        self.ma_slow = c.rolling(self.config.ma_slow, min_periods=1).mean()
        self.ma_ratio = (self.ma_fast - self.ma_slow) / self.ma_slow
        self.peak = c.rolling(self.config.peak_lookback, min_periods=1).max()
        self.drawdown = (c - self.peak) / self.peak

    # ----------------------------------------------------------
    # 趋势判断
    # ----------------------------------------------------------
    def _get_regime(self, date) -> tuple:
        # 使用前一天的数据避免前视偏差（当天收盘价在决策时不可用）
        available = self.ma_ratio.loc[:date]
        ratio = available.iloc[-2] if len(available) >= 2 else available.iloc[-1]
        cfg = self.config
        if ratio > cfg.strong_bull_threshold:
            regime = "strong_bull"
        elif ratio > cfg.mild_bull_threshold:
            regime = "mild_bull"
        elif ratio > cfg.mild_bear_threshold:
            regime = "neutral"
        elif ratio > cfg.strong_bear_threshold:
            regime = "mild_bear"
        else:
            regime = "strong_bear"
        return regime, cfg.regime_allocations[regime]

    # ----------------------------------------------------------
    # 机会捕捉
    # ----------------------------------------------------------
    def _get_opportunity(self, date) -> tuple:
        # 使用前一天的数据避免前视偏差
        available = self.drawdown.loc[:date]
        dd = available.iloc[-2] if len(available) >= 2 else available.iloc[-1]
        # 从最深阈值开始匹配
        for threshold, override_pct, level in sorted(
            self.config.opportunity_thresholds, key=lambda x: x[0]
        ):
            if dd <= threshold:
                return level, override_pct
        return "none", 0.0

    # ----------------------------------------------------------
    # 主接口
    # ----------------------------------------------------------
    def get_allocation(self, date, is_rebalance_day=False) -> AllocationResult:
        ts = pd.Timestamp(date)

        # 日期超出数据范围时返回默认
        if ts < self.csi300_close.index[0] or ts > self.csi300_close.index[-1]:
            return AllocationResult(0.80, 0.20, "neutral", "none", 0.0, 0.0)

        regime, regime_alloc = self._get_regime(ts)
        opp_level, opp_override = self._get_opportunity(ts)

        # 机会捕捉只在中性或以上趋势触发，熊市不逆势加仓
        bear_regimes = {"mild_bear", "strong_bear"}
        if opp_level != "none" and regime not in bear_regimes:
            target = max(regime_alloc, opp_override)
        else:
            target = regime_alloc

        # 最高仓位封顶 90%，避免满仓押注脉冲行情
        target = min(target, 0.90)

        # EMA 平滑（调仓日不平滑）
        if self._prev_allocation is not None and not is_rebalance_day:
            smoothed = (self.config.smoothing_factor * target +
                        (1 - self.config.smoothing_factor) * self._prev_allocation)
        else:
            smoothed = target
        self._prev_allocation = smoothed

        dd_available = self.drawdown.loc[:ts]
        dd_val = float(dd_available.iloc[-2] if len(dd_available) >= 2 else dd_available.iloc[-1])
        ratio_available = self.ma_ratio.loc[:ts]
        ratio_val = float(ratio_available.iloc[-2] if len(ratio_available) >= 2 else ratio_available.iloc[-1])

        return AllocationResult(
            stock_pct=round(smoothed, 4),
            cash_pct=round(1 - smoothed, 4),
            regime=regime,
            opportunity_level=opp_level,
            market_drawdown=dd_val,
            trend_score=ratio_val,
        )

    def get_bond_daily_return(self) -> float:
        return self.config.bond_annual_return / 252


# ============================================================
# 独立测试
# ============================================================
if __name__ == "__main__":
    ctl = MarketPositionController()
    ctl.load_market_data()

    # 打印每月末的仓位信号
    dates = ctl.csi300_close.index
    monthly = pd.DatetimeIndex(
        pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M")).last().values
    )
    # 只看 2019 年以后
    monthly = monthly[monthly >= "2019-01-01"]

    print(f"\n{'日期':>12}  {'MA比值':>8}  {'市场回撤':>8}  {'趋势':>12}  {'机会':>8}  {'股票仓位':>8}")
    print("-" * 72)
    for dt in monthly:
        a = ctl.get_allocation(dt, is_rebalance_day=True)
        print(f"{dt.strftime('%Y-%m-%d'):>12}  {a.trend_score:>+8.4f}  {a.market_drawdown:>8.2%}  "
              f"{a.regime:>12}  {a.opportunity_level:>8}  {a.stock_pct:>8.0%}")
