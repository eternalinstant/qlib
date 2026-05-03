# Qlib 量化研究仓

这个仓库是一个面向 A 股的量化研究环境，主要做三件事：

1. 从 Qlib / Tushare 数据生成因子与选股结果
2. 用 Qlib / PyBroker 两套引擎回测策略
3. 沉淀研究结果、选股快照和回测输出

当前仓库已经从早期示例策略演进到 `top15_*` 这一批研究策略，README 以当前实际代码和数据要求为准。

## 当前目录

```text
qlib/
├── config/                    # 全局配置 + 单策略 YAML
├── core/                      # 策略核心：因子、选股、股票池、仓位
├── modules/
│   ├── backtest/              # 回测引擎（Qlib / PyBroker）
│   ├── data/                  # 数据下载、更新、Tushare -> Qlib
│   └── risk/                  # 风控模块
├── scripts/                   # 研究脚本、批量搜索、辅助工具
├── data/
│   ├── tushare/               # 原始/中间数据（parquet + 少量必要 csv）
│   ├── qlib_data/             # Qlib provider 数据
│   └── selections/            # 各策略选股结果
├── results/                   # 回测结果、图表、研究记录
└── tests/                     # 单元测试
```

## 重要文件

- [main.py](/Users/sxt/code/qlib/main.py)：命令行主入口
- [config/strategy.yaml](/Users/sxt/code/qlib/config/strategy.yaml)：全局默认策略参数
- [core/strategy.py](/Users/sxt/code/qlib/core/strategy.py)：策略 YAML 加载与生成选股入口
- [core/selection.py](/Users/sxt/code/qlib/core/selection.py)：因子加载、信号计算、Top-K 选股
- [core/universe.py](/Users/sxt/code/qlib/core/universe.py)：股票池过滤，包括历史沪深300成分过滤
- [modules/backtest/qlib_engine.py](/Users/sxt/code/qlib/modules/backtest/qlib_engine.py)：Qlib 回测引擎
- [modules/backtest/pybroker_engine.py](/Users/sxt/code/qlib/modules/backtest/pybroker_engine.py)：PyBroker 回测引擎
- [modules/data/tushare_downloader.py](/Users/sxt/code/qlib/modules/data/tushare_downloader.py)：Tushare 数据下载

## 策略分层管理

策略文件仍然统一放在 [config/strategies](/Users/sxt/code/qlib/config/strategies)，但现在支持递归分层：

- 兼容旧结构：`config/strategies/<strategy>.yaml`
- 推荐新结构：`config/strategies/<layer>/<group>/<strategy>.yaml`

当前约定：

- 根目录平铺文件：最终 `Top5` 优胜策略，方便直接挑主力候选
- `fixed/`：固定策略，作为长期基线和正式对照组
- `experimental/`：实验策略，按主题继续扩展
- `research/`：研究归档策略，按主题收纳历史方案和未晋级版本

当前根目录保留的 `Top5` 策略：

- `test_bullbear_constrained_all`
- `top15_core_trend`
- `top15_amp_day`
- `test_low_drawdown_constrained_all`
- `top15_core_day`

当前推荐目录示例：

- `fixed/balanced/core_trend`
- `fixed/balanced/core_day`
- `fixed/aggressive/amp_day`
- `fixed/portfolio/dual_guard_18`
- `fixed/portfolio/triple_diversified_18`
- `fixed/portfolio/efficient_guard_30`
- `fixed/portfolio/diversified_guard_30`
- `fixed/portfolio/efficient_guard_dynamic_20`
- `fixed/portfolio/diversified_guard_dynamic_20`
- `experimental/safety/bullbear_quality_guard_all`
- `experimental/regime/bullbear_regime_guard_all`
- `experimental/validity/bullbear_quality_regime_validity_all`
- `experimental/value_plan/value_book_ocf_stoploss_replace_15`

说明：

- `python main.py backtest --list` 会按 `winners / fixed / experimental / research` 分组列出策略。
- `python main.py backtest -s experimental/safety/bullbear_quality_guard_all` 可直接加载分层策略。
- 新分层策略的选股结果会落到 [data/selections](/Users/sxt/code/qlib/data/selections) 的同层级路径下，例如 `data/selections/experimental/safety/bullbear_quality_guard_all.csv`。
- `default` 已移动到 `fixed/reference/default`，但仍可直接用 `python main.py backtest -s default` 通过 basename 加载。
- 新增 `composition` 组合策略：会先跑成员策略，再按权重混合净值；残余权重默认视为现金。

## 快速开始

运行任何需要 Tushare 的数据更新脚本前，请先设置环境变量：

```bash
export TUSHARE_TOKEN=your_token_here
```

新机器首次启动，先执行：

```bash
pip install -e .[full]
python main.py update
```

这条命令现在同时承担两件事：

- 首次 bootstrap：从 `2016-01-01` 拉历史 Tushare 数据，生成 Qlib provider，并重算选股
- 后续更新：只做增量更新和必要修复

### 1. 列出策略

```bash
python main.py backtest --list
```

### 2. 生成选股

```bash
python main.py select -s top15_core_trend
```

### 3. 跑单策略回测

```bash
python main.py backtest -s top15_core_trend -e qlib
python main.py backtest -s top15_core_trend -e pybroker
python main.py backtest -s experimental/regime/bullbear_regime_guard_all -e qlib
```

### 4. 跑多策略对比

```bash
python main.py compare -s top15_core_trend,top15_core_day,top15_amp_day --no-benchmark
```

### 5. 更新数据

```bash
python main.py update
```

正式 `select / backtest / compare` 前会自动执行数据预检；如果缺少历史沪深300成分或历史 ST 数据，会直接失败，不会再静默降级。新环境首次执行 `python main.py update` 就会自动补这批历史数据。

## 策略配置

每个策略一个 YAML。当前支持的关键字段包括：

```yaml
name: top15_example
description: "示例策略"

weights:
  alpha: 0.10
  risk: 0.20
  enhance: 0.70

selection:
  mode: factor_topk        # factor_topk / stoploss_replace
  topk: 15
  universe: csi300         # all / csi300
  neutralize_industry: false
  min_market_cap: 80
  exclude_new_days: 120
  exclude_st: true
  buffer: 10
  stoploss_lookback_days: 20  # stoploss_replace 模式使用
  stoploss_drawdown: 0.10     # stoploss_replace 模式使用
  replacement_pool_size: 30   # stoploss_replace 模式使用，0=不限制
  hard_filters:            # 可选：绝对阈值硬过滤
    roa_fina: 0
  hard_filter_quantiles:   # 可选：分位过滤
    ocf_to_ev: 0.4

rebalance:
  freq: day                # day / week / biweek / month

position:
  model: trend             # trend / fixed / full

trading:
  slippage_bps: 5
  impact_bps: 5

validity:
  lookback_days: 60        # 最近多少个交易日做有效性检查
  min_total_return: -0.05  # 低于该区间收益触发动作
  min_sharpe: 0.00         # 低于该夏普触发动作
  max_drawdown: -0.12      # 低于该最大回撤触发动作
  action: reduce           # review / reduce / pause
  reduce_to: 0.50          # action=reduce 时建议降到的仓位比例
  apply_in_backtest: false # true 时把 validity 直接作用到回测收益

composition:
  components:
    - strategy: test_bullbear_constrained_all
      weight: 0.50
    - strategy: test_low_drawdown_constrained_all
      weight: 0.20
  cash_weight: 0.30        # 可选；不写时默认等于 1 - 成员权重和
```

### `selection.universe`

当前支持：

- `all`：全市场股票池
- `csi300`：按调仓日回看当时的沪深300成分股

注意：`csi300` 不是静态股票池，代码会按调仓日回看最近一期成分快照。

### `selection.mode`

当前支持两种选股模式：

- `factor_topk`：默认模式。每个调仓日按综合因子分数排序，再叠加 `sticky / buffer / churn_limit / event gate` 等稳定性逻辑。这一模式主要用于原始因子研究、Top-K 选股和找因子。
- `stoploss_replace`：先按综合因子分数选出初始持仓，之后每日只检查当前持仓是否较最近 `N` 个交易日最高收盘价回撤超过阈值；只有触发的股票才卖出，再从当前因子排序靠前的非持仓股票里补回。这个模式对应 [docs/strategy_plan.md](/Users/sxt/code/qlib/docs/strategy_plan.md) 里的“动态止损调仓”思路。

`stoploss_replace` 额外支持：

- `selection.stoploss_lookback_days`：回看近期最高收盘价的交易日窗口，默认 `20`
- `selection.stoploss_drawdown`：触发换仓的回撤阈值，正数，如 `0.10`
- `selection.replacement_pool_size`：候选替换股票池大小，`0` 表示不限制

建议约定：

- 因子研究、单因子扫描、原始 Top-K 回测：继续用 `factor_topk`
- 已经确定主因子后，要验证“低换手 + 跌破高点才换仓”的执行版本：用 `stoploss_replace`

### `composition`

组合策略用于把多个已有策略按固定权重混合成一个“母策略”：

- `composition.components`：成员策略与权重，至少 2 个
- `composition.cash_weight`：残余现金权重；如果不写，系统自动用 `1 - 权重和`
- 组合策略不会生成独立选股文件，选股/预检都会递归落到各成员策略
- 如果同时配置 `validity.apply_in_backtest: true`，组合层会按近期表现动态缩放总暴露，回测结果里会额外写出 `raw_return` 与 `exposure_factor`
- `python main.py backtest -s fixed/portfolio/dual_guard_18 -e qlib` 可直接回测组合策略

## 添加策略

新增策略通常只需要新增一个 YAML 文件，不需要改回测引擎或选股框架。

如果只是换因子、调权重、调 `topk`，继续沿用 `selection.mode: factor_topk` 即可。
只有当策略本身需要“持仓股跌破近期高点才换仓”这种执行逻辑时，才需要切到 `selection.mode: stoploss_replace`。

### 最小步骤

1. 在 [config/strategies](/Users/sxt/code/qlib/config/strategies) 下选择一个层级目录新建 YAML  
   例如：`fixed/balanced/<strategy_name>.yaml` 或 `experimental/safety/<strategy_name>.yaml`
2. 只写和 [config/strategy.yaml](/Users/sxt/code/qlib/config/strategy.yaml) 不同的字段
3. 运行 `python main.py select -s <layer>/<group>/<strategy_name>`
4. 运行 `python main.py backtest -s <layer>/<group>/<strategy_name> -e qlib`
5. 需要横向对比时，运行 `python main.py compare -s <layer>/<group>/<strategy_name>,top15_core_trend --no-benchmark`

### 最小模板

```yaml
name: my_strategy
description: "示例策略"

weights:
  alpha: 0.10
  risk: 0.20
  enhance: 0.70

factors:
  enhance:
    - name: ema5_dev
      expression: "$close / EMA($close, 5) - 1"
      source: qlib

selection:
  mode: factor_topk
  topk: 10
  universe: csi300
  neutralize_industry: false
  min_market_cap: 50
  exclude_st: true
  exclude_new_days: 60
  buffer: 10

rebalance:
  freq: day

position:
  model: trend
```

### 继承规则

- 单策略 YAML 会先加载 [config/strategy.yaml](/Users/sxt/code/qlib/config/strategy.yaml) 再做覆盖，所以没写的字段默认继承全局值。
- `factors` 支持两种写法：
  - 直接写 `expression`，适合临时试验单个新因子
  - 只写 `name`，引用 [core/factors.py](/Users/sxt/code/qlib/core/factors.py) 里的默认注册因子
- 分层策略的选股文件会按同样层级落到 [data/selections](/Users/sxt/code/qlib/data/selections) 下。
- `validity` 是实盘监控规则，不会直接改写历史回测收益；它用于判断“最近一段时间策略是否失效，以及建议 review / reduce / pause”。
- `selection.mode: factor_topk` 是默认研究模式；`selection.mode: stoploss_replace` 是执行模式，会在保持当前持仓的前提下，仅对触发止损条件的股票做因子池替换。

### 分层建议

- 根目录：只放当前最终晋级的 `Top5` 主策略，不再堆历史文件。
- `fixed/`：只放已经验过多轮、作为正式比较基线的策略。
- `experimental/`：只放正在验证的新思路，建议至少按主题再分一层，如 `safety / regime / validity`。
- `research/`：放历史研究、失败路线、阶段性候选，建议按主题细分，例如 `amplitude / bullbear / drawdown / mix / close_only / all_market`。

### 什么时候只改 YAML

这些场景只改 [config/strategies](/Users/sxt/code/qlib/config/strategies) 下的策略文件：

- 改因子权重
- 改 `topk / buffer / sticky / rebalance.freq`
- 改 `selection.universe / exclude_st / exclude_new_days / min_market_cap`
- 改 `selection.hard_filters / hard_filter_quantiles`
- 改 `position.model` 或仓位参数
- 改 `validity` 有效性阈值
- 改交易成本和是否启用 `block_limit_up_buy / block_limit_down_sell`

### 什么时候要改代码

这些不是“改策略参数”，而是“改系统能力”：

- 想新增一个可复用的默认因子：改 [core/factors.py](/Users/sxt/code/qlib/core/factors.py)
- 想新增一个新股票池，比如中证500：改 [core/universe.py](/Users/sxt/code/qlib/core/universe.py)
- 想新增一个新仓位模型：改 [core/position.py](/Users/sxt/code/qlib/core/position.py) 和 [core/strategy.py](/Users/sxt/code/qlib/core/strategy.py)
- 想把 `validity` 规则自动接进实盘调仓执行：改 [core/validity.py](/Users/sxt/code/qlib/core/validity.py) 和实盘执行链路
- 想改收益口径、撮合、成交约束规则：改 [modules/backtest/qlib_engine.py](/Users/sxt/code/qlib/modules/backtest/qlib_engine.py)

### 不要手改的文件

下面这些通常是产物或框架文件，不应作为“调策略”的入口：

- [data/selections](/Users/sxt/code/qlib/data/selections) 下的 CSV
- [results](/Users/sxt/code/qlib/results) 下的回测结果
- [main.py](/Users/sxt/code/qlib/main.py)
- [core/selection.py](/Users/sxt/code/qlib/core/selection.py)

## 数据要求

### 必要数据

- [data/qlib_data/cn_data](/Users/sxt/code/qlib/data/qlib_data/cn_data)：Qlib provider 数据
- [data/tushare/daily_basic.parquet](/Users/sxt/code/qlib/data/tushare/daily_basic.parquet)
- [data/tushare/balancesheet.parquet](/Users/sxt/code/qlib/data/tushare/balancesheet.parquet)
- [data/tushare/cashflow.parquet](/Users/sxt/code/qlib/data/tushare/cashflow.parquet)
- [data/tushare/income.parquet](/Users/sxt/code/qlib/data/tushare/income.parquet)
- [data/tushare/fina_indicator.parquet](/Users/sxt/code/qlib/data/tushare/fina_indicator.parquet)
- [data/tushare/index_daily.parquet](/Users/sxt/code/qlib/data/tushare/index_daily.parquet)
- [data/tushare/stock_basic.csv](/Users/sxt/code/qlib/data/tushare/stock_basic.csv)
- [data/tushare/stock_industry.csv](/Users/sxt/code/qlib/data/tushare/stock_industry.csv)

### 如果使用 `selection.universe: csi300`

还需要：

- [data/tushare/index_weight.parquet](/Users/sxt/code/qlib/data/tushare/index_weight.parquet)
- [data/tushare/namechange.parquet](/Users/sxt/code/qlib/data/tushare/namechange.parquet)（当 `exclude_st: true` 时必需）

该文件来自 Tushare `index_weight` 接口，至少需要这几个字段：

- `index_code`
- `con_code`
- `trade_date`

如果缺少这两份历史数据，策略在生成选股和回测前会直接报错，不会偷偷回退到全市场，也不会把 ST 过滤降级成 no-op。

### 下载历史指数成分和名称变更

```bash
python main.py update
```

如果只是单独排查下载问题，才需要手工调用下载器：

```bash
python modules/data/tushare_downloader.py --type index_weight --start 20160101
python modules/data/tushare_downloader.py --type namechange --start 20100101
```

### 正式预检范围

`modules/data/precheck.py` 会在正式运行前检查：

- Qlib provider：`calendars/day.txt`、`instruments/all.txt`、`factor_data.parquet`
- Tushare 核心表：`daily_basic / income / balancesheet / cashflow / fina_indicator / index_daily / stock_basic / stock_industry`
- `csi300` 历史成分：`index_weight`
- 历史 ST 过滤：`namechange`

### 价格字段审计

如果要评估 `Qlib $open/$high/$low/$close` 是否适合拿来做成交约束，可以直接运行：

```bash
python3 scripts/audit_price_fields.py --sample-size 20 --days 10
```

当前正式口径改为：

- `Qlib $open/$high/$low` 只用于审计，不作为正式成交约束输入
- 涨跌停可成交约束使用 [data/qlib_data/raw_data](/Users/sxt/code/qlib/data/qlib_data/raw_data) 下的原始日线文件
- `modules/data/updater.py` 会在日更时自动回补最近一段交易日的 `raw_data`
- 如果启用了 `block_limit_up_buy / block_limit_down_sell`，但本地缺少对应 `raw_data` 文件，回测会直接报错，不会静默降级

## 数据处理链路

### 数据流

```
Tushare API
  │
  ├─ tushare_downloader.py / data_update.py
  │   → data/tushare/*.parquet (9 个核心文件: daily_basic, adj_factor, income,
  │      balancesheet, cashflow, fina_indicator, index_daily, index_weight, namechange)
  │
  ├─ build_qlib_data.py step 2
  │   → data/qlib_data/raw_data/{inst}.parquet (5751 只股票 OHLCV)
  │
  ├─ tushare_to_qlib.py (build_adjusted_bins_batched)
  │   → data/qlib_data/cn_data/
  │       ├── calendars/day.txt           (交易日历)
  │       ├── instruments/all.txt         (股票列表 + 日期范围)
  │       ├── factor_data.parquet         (前向填充因子, ~10M行 × 42列)
  │       └── features/{inst}/{open,high,low,close,volume,amount}.day.bin
  │                                        (前复权 bin 文件, Qlib 可读)
```

### 数据质量检查与自动修复

运行 `data_quality_guard.py` 进行全量检查和自动修复（最多 5 轮循环）：

```bash
source env.sh
python scripts/data_quality_guard.py
```

脚本会自动执行以下流程：
1. **全量检查**：复用 `validate_data.py` 的 24 项检查
2. **自动诊断**：根据检查结果判断根因
3. **自动修复**：调用下载/更新/重建等修复函数
4. **循环重验**：最多 5 轮，直到全部 PASS
5. **人工介入**：如果 5 轮未通过，自动生成 `Q&A/YYYYMMDD.md` 供人工排查

如果只想做检查不做修复：

```bash
python scripts/validate_data.py
```

### 手动重建

- **全量重建**: `python scripts/build_qlib_data.py build --force`
- **只重建 Qlib 格式（复用已有下载和 raw_data）**: `python scripts/build_qlib_data.py build --skip-download --skip-raw --workers 8`
- **增量更新**: `python main.py update`

**注意：** 全量构建过程内存峰值约 8-10GB，建议在 16GB 以上机器运行。脚本已内置分批处理，每批 500-1000 只股票，避免 OOM。

## 数据流与验证

### 构建数据

**首次全量构建：**

```bash
python scripts/build_qlib_data.py build --workers 8
```

### 验证数据

```bash
python scripts/validate_data.py
```

24 项检查分为三级：

| 级别 | 含义 | 目标 |
|------|------|------|
| P0 | 致命问题（回测结果错误） | 必须全部 PASS |
| P1 | 重要问题（影响回测质量） | 不应有 FAIL |
| P2 | 数据健康指标 | 参考 |

验证报告会输出彩色终端摘要，同时保存 JSON 到 `results/` 目录。

### 内存优化说明

数据处理脚本针对 16GB 内存机器做了以下优化：

- **`save()`**：用 PyArrow 只读 instrument 列做差集，避免同时加载新旧两份全量数据
- **`build_adjusted_bins_batched()`**：每批 1000 只股票计算前复权并写 bin，之后 `gc.collect()` 释放
- **`convert()`**：分批处理 500 只股票，concat 前释放 quarterlies 和 daily

### 常见问题

**Q: 日历覆盖率 WARN（510 只股票 >70% 缺失）**

A: 属退市/新上市股票的正常现象。instruments.txt 已写入每只股票的实际日期范围，退市股票覆盖其实际存续期即可。

**Q: Adj_ratio WARN（极低值 <0.001）**

A: 源自极端复权场景（如大量拆股的历史股票），阈值已放宽至 0.001，83 条记录属正常范围。

**Q: instruments 日期全部相同**

A: 旧版代码写死日期。运行 `build_qlib_data.py build --skip-download --skip-raw` 重建即可修复。

**Q: 构建时 OOM**

A: 减少并发数或分批处理。脚本已内置分批逻辑（因子处理每批 500 只，bin 写入每批 1000 只）。如仍 OOM，可进一步降低 `batch_size` 参数。

## 结果文件

- [data/selections](/Users/sxt/code/qlib/data/selections)：策略选股快照
- [results](/Users/sxt/code/qlib/results)：回测 CSV、图表、研究记录

当前口径优先看这几份：

- [results/top15_historical_csi300_research_20260322.md](/Users/sxt/code/qlib/results/top15_historical_csi300_research_20260322.md)
- [results/tradability_constraint_compare_20260322_185830.md](/Users/sxt/code/qlib/results/tradability_constraint_compare_20260322_185830.md)
- [results/price_field_audit_20260322_183351.md](/Users/sxt/code/qlib/results/price_field_audit_20260322_183351.md)

历史阶段性研究快照：

- [results/robust_close_only_research_20260322.md](/Users/sxt/code/qlib/results/robust_close_only_research_20260322.md)
- [results/top15_core_v2_research_20260322.md](/Users/sxt/code/qlib/results/top15_core_v2_research_20260322.md)
- [results/factor_optimization_report_20260310.md](/Users/sxt/code/qlib/results/factor_optimization_report_20260310.md)

## 开发说明

### 测试

```bash
python3 -m pytest -q
```

### 说明

- Qlib 回测引擎当前使用更保守的 `close-to-close` 收益口径。
- 涨跌停成交约束当前只解决“开盘能否成交”，不会把同日收益拆成隔夜段和日内段。
- 当 `exclude_st: true` 时，正式预检要求本地存在 `namechange` 历史文件；缺失会直接失败，不会降级成 no-op。
- 仓库内曾出现过 `data/data/tushare` 的重复路径问题；当前只保留 `data/tushare` 作为有效 Tushare 数据目录。

## 当前状态

这个仓更适合“研究与验证”，不是开箱即用的实盘系统。当前已经具备：

- 多策略 YAML 配置
- 双回测引擎
- 历史沪深300股票池闭环
- 历史 ST 过滤闭环
- 基于 `raw_data` 的开盘可成交约束
- 单因子 / 多因子研究脚本
- 选股与回测结果隔离

还在持续补强的部分主要是：

- `$high/$low` 字段的独立质量复核与替代方案
- 更完整的撮合、滑点和成交冲击模型
- PyBroker 与 Qlib 的成交约束口径统一
