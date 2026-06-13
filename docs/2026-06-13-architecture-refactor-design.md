# 架构重构设计文档

**日期：** 2026-06-13
**状态：** 待评审
**关联需求：** `docs/2026-06-13-architecture-refactor-requirements.md`

---

## 1. 总体架构

### 1.1 目标架构图

```
┌─────────────────────────────────────────────────────────┐
│                        main.py                          │
│              按策略类型分派到对应引擎                      │
└────────────────┬───────────────────────┬────────────────┘
                 │                       │
    ┌────────────▼──────────┐  ┌─────────▼──────────────┐
    │  QlibBacktestEngine   │  │    RuleBasedEngine      │
    │  （信号驱动，保留）     │  │   （规则驱动，新建）      │
    │  批处理 + CSV选股      │  │   逐日 on_bar() 循环    │
    └────────────┬──────────┘  └─────────┬──────────────┘
                 │                       │
                 └──────────┬────────────┘
                            │
              ┌─────────────▼──────────────┐
              │    modules/backtest/common  │
              │  涨跌停 / 交易成本 / 日历    │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │       DataProvider         │
              │  统一数据接口（新建）        │
              │  OHLCV / 因子 / 股票池      │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │       数据层（现有）         │
              │  qlib provider / parquet   │
              └────────────────────────────┘
```

### 1.2 策略目录结构

```
strategies/
├── base.py                    ← BaseStrategy 抽象类
├── factor/                    ← 信号驱动策略（现有迁移）
│   ├── __init__.py
│   ├── factor_strategy.py     ← FactorStrategy（对接现有 selection.py）
│   └── configs/               ← 现有 config/models/ 迁移至此（P2）
├── pyramid/                   ← 金字塔加仓
│   ├── __init__.py
│   ├── pyramid_strategy.py    ← PyramidStrategy(BaseStrategy)
│   └── configs/
│       ├── pyramid_atr_3layer.yaml
│       └── pyramid_breakout_2layer.yaml
├── turtle/                    ← 海龟交易（P1）
│   ├── __init__.py
│   ├── turtle_strategy.py
│   └── configs/
└── pe_timing/                 ← PE 择时（P2）
    ├── __init__.py
    ├── pe_strategy.py
    └── configs/
```

---

## 2. 核心抽象

### 2.1 BaseStrategy

```python
# strategies/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd

@dataclass
class PositionState:
    """单票仓位状态，规则驱动策略使用"""
    instrument: str
    entry_price: float
    layers: int = 1                    # 当前加仓层数
    layer_prices: List[float] = field(default_factory=list)
    stop_loss: float = 0.0
    units: float = 0.0                 # 持仓手数/市值

@dataclass
class StrategySignal:
    """策略信号，两种引擎统一输出格式"""
    date: pd.Timestamp
    instrument: str
    action: str                        # buy / sell / add / hold
    weight: float = 0.0               # 目标权重（信号驱动用）
    price: Optional[float] = None     # 触发价格（规则驱动用）
    reason: str = ""


class BaseStrategy(ABC):
    """所有策略的基类"""

    def __init__(self, config: dict):
        self.config = config
        self.name = config.get("name", self.__class__.__name__)

    @property
    def strategy_type(self) -> str:
        """返回 'signal' 或 'rule'，引擎据此分派"""
        raise NotImplementedError

    @classmethod
    def from_yaml(cls, path: str) -> "BaseStrategy":
        """从 YAML 文件加载策略"""
        import yaml
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config)
```

### 2.2 信号驱动策略接口

```python
class SignalStrategy(BaseStrategy):
    """因子选股类策略，对接现有 selection.py"""

    @property
    def strategy_type(self) -> str:
        return "signal"

    @abstractmethod
    def compute_signals(self, date: pd.Timestamp, data_provider) -> Dict[str, float]:
        """返回 {instrument: score} 字典，引擎按分数选 Top-K"""
        pass
```

### 2.3 规则驱动策略接口

```python
class RuleStrategy(BaseStrategy):
    """逐日事件驱动类策略（海龟、金字塔等）"""

    @property
    def strategy_type(self) -> str:
        return "rule"

    def on_start(self, data_provider) -> None:
        """回测开始时初始化（可选重写）"""
        pass

    @abstractmethod
    def on_bar(
        self,
        date: pd.Timestamp,
        universe: List[str],
        positions: Dict[str, PositionState],
        data_provider,
    ) -> List[StrategySignal]:
        """每日收盘后调用，返回信号列表"""
        pass

    def on_end(self) -> None:
        """回测结束时清理（可选重写）"""
        pass
```

---

## 3. 数据接口层

### 3.1 DataProvider

```python
# core/data_provider.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional

class DataProvider(ABC):

    @abstractmethod
    def get_ohlcv(
        self,
        instruments: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """返回 MultiIndex(instrument, date) 的 OHLCV DataFrame"""
        pass

    @abstractmethod
    def get_factor(
        self,
        instruments: List[str],
        factor: str,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """返回 MultiIndex(instrument, date) 的因子值 Series"""
        pass

    @abstractmethod
    def get_universe(
        self,
        date: pd.Timestamp,
        universe: str = "csi800",
    ) -> List[str]:
        """返回指定日期的合法股票池"""
        pass

    def get_atr(
        self,
        instrument: str,
        date: pd.Timestamp,
        period: int = 20,
    ) -> float:
        """ATR 计算，规则策略常用，默认实现基于 OHLCV"""
        ...
```

### 3.2 QlibDataProvider（具体实现）

```python
# core/qlib_data_provider.py
class QlibDataProvider(DataProvider):
    """基于现有 qlib + parquet 双源的数据提供者"""

    def get_ohlcv(self, instruments, start_date, end_date):
        # 复用现有 _load_raw_trade_quotes() 逻辑
        from modules.backtest.common import load_raw_trade_quotes
        return load_raw_trade_quotes(instruments, start_date, end_date)

    def get_factor(self, instruments, factor, start_date, end_date):
        # 复用现有 load_features_safe() / parquet 读取逻辑
        ...

    def get_universe(self, date, universe="csi800"):
        from core.universe import filter_instruments
        ...
```

---

## 4. 规则引擎

### 4.1 RuleBasedEngine 核心逻辑

```python
# modules/backtest/rule_engine.py
class RuleBasedEngine(BacktestEngine):

    def run(self, strategy: RuleStrategy) -> BacktestResult:
        dp = QlibDataProvider()
        trade_calendar = _load_trade_calendar(...)
        positions: Dict[str, PositionState] = {}
        portfolio_value = self.config.get("initial_capital", 1_000_000)

        strategy.on_start(dp)

        for date in trade_calendar:
            universe = dp.get_universe(date, strategy.config["selection"]["universe"])

            # 调用策略逐日逻辑
            signals = strategy.on_bar(date, universe, positions, dp)

            # 执行信号
            for signal in signals:
                portfolio_value, positions = _execute_signal(
                    signal, positions, portfolio_value, date, dp
                )

            # 记录每日净值
            ...

        strategy.on_end()
        return _build_backtest_result(daily_returns, positions)
```

### 4.2 金字塔策略实现示意

```python
# strategies/pyramid/pyramid_strategy.py
class PyramidStrategy(RuleStrategy):
    """
    金字塔加仓策略：
    - 突破 N 日高点入场
    - 每涨 ATR * add_factor 追加一层，最多 max_layers 层
    - 跌破 stop_atr * ATR 全部止损出场
    """

    def on_bar(self, date, universe, positions, dp) -> List[StrategySignal]:
        signals = []
        cfg = self.config["pyramid"]

        for instrument in universe:
            atr = dp.get_atr(instrument, date, cfg["atr_period"])
            close = dp.get_ohlcv([instrument], str(date), str(date))["close"].iloc[-1]
            high_n = dp.get_ohlcv([instrument], ..., ...)["close"].rolling(cfg["entry_lookback"]).max().iloc[-1]

            if instrument not in positions:
                # 入场：突破 N 日高点
                if close >= high_n:
                    signals.append(StrategySignal(date, instrument, "buy", price=close))
            else:
                pos = positions[instrument]
                # 止损
                if close <= pos.stop_loss:
                    signals.append(StrategySignal(date, instrument, "sell", price=close, reason="stop_loss"))
                # 加仓：涨幅超过 ATR * add_factor
                elif pos.layers < cfg["max_layers"]:
                    add_trigger = pos.layer_prices[-1] + atr * cfg["add_factor"]
                    if close >= add_trigger:
                        signals.append(StrategySignal(date, instrument, "add", price=close))

        return signals
```

### 4.3 金字塔策略 YAML 示例

```yaml
# strategies/pyramid/configs/pyramid_atr_3layer.yaml
name: pyramid_atr_3layer
strategy_class: strategies.pyramid.pyramid_strategy.PyramidStrategy

selection:
  universe: csi800
  min_market_cap: 80
  exclude_st: true

pyramid:
  entry_lookback: 20        # N 日高点突破入场
  atr_period: 20            # ATR 计算周期
  add_factor: 1.0           # 每 1 ATR 追加一层
  max_layers: 3             # 最多加仓层数
  stop_atr: 2.0             # 跌破入场价 - 2 ATR 止损
  position_size: 0.05       # 初始每票仓位比例

trading:
  buy_commission_rate: 0.0003
  sell_commission_rate: 0.0003
  sell_stamp_tax_rate: 0.001
  block_limit_up_buy: true
  block_limit_down_sell: true
```

---

## 5. 共享工具层

从现有 `qlib_engine.py` 提取到 `modules/backtest/common.py`：

| 函数 | 来源 | 用途 |
|------|------|------|
| `load_raw_trade_quotes()` | `qlib_engine._load_raw_trade_quotes` | 加载原始日线行情 |
| `get_limit_prices()` | `qlib_engine._get_limit_prices` | 涨跌停价格计算 |
| `can_buy_at_open()` | `qlib_engine._can_buy_at_open` | 是否可买判断 |
| `can_sell_at_open()` | `qlib_engine._can_sell_at_open` | 是否可卖判断 |
| `compute_trade_cost()` | `qlib_engine._compute_weight_delta_fee` | 交易成本计算 |
| `load_trade_calendar()` | `qlib_engine._load_trade_calendar_slice` | 交易日历 |

---

## 6. main.py 分派逻辑

```python
# main.py backtest 命令扩展
def run_backtest(strategy_path):
    import yaml
    with open(strategy_path) as f:
        config = yaml.safe_load(f)

    strategy_class_path = config.get("strategy_class")

    if strategy_class_path:
        # 规则驱动策略：动态加载 Python 类
        cls = _import_class(strategy_class_path)
        strategy = cls(config)
        engine = RuleBasedEngine(config)
    else:
        # 信号驱动策略：走现有路径
        strategy = Strategy.load(strategy_path)
        engine = QlibBacktestEngine(config)

    result = engine.run(strategy)
    result.save()
```

---

## 7. 实施阶段

| 阶段 | 内容 | 交付物 |
|------|------|--------|
| **Phase 1** | 提取 `common.py` + 实现 `DataProvider` + `BaseStrategy` | 共享层，不破坏现有功能 |
| **Phase 2** | 实现 `RuleBasedEngine` + `PyramidStrategy` | 金字塔策略可回测 |
| **Phase 3** | 实现 `TurtleStrategy` | 海龟策略可回测 |
| **Phase 4** | 策略目录重组（`strategies/`），迁移现有 YAML | 目录结构统一 |
| **Phase 5** | 实现 `PETimingStrategy` | PE 择时可回测 |

Phase 1-2 为最小可用版本（MVP），不影响现有任何功能。

---

## 8. 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 提取 `common.py` 引入回归 | 提取后跑全量回测对比数值，确保一致 |
| `RuleBasedEngine` 涨跌停逻辑遗漏 | 复用 `common.py` 的函数，不重写 |
| 金字塔策略参数不合理导致过拟合 | 与因子策略一样做 IS/OOS 分段验证 |
| YAML 目录迁移破坏现有脚本路径 | Phase 4 保留旧路径软链接，分批迁移 |
