# Tushare 数据下载与更新说明

> 文档状态：当前有效说明。早期“下载计划”已经完成，当前请以本文件和 `README.md` 为准。

## 当前目标

这个仓当前以 Tushare 作为标准原始数据源，主要承担四类任务：

1. 提供财务、估值、行业和指数数据给因子与选股链路
2. 提供 `index_weight.parquet` 以支持历史沪深300成分股股票池
3. 提供 `namechange.parquet` 以支持历史 ST 过滤
4. 通过日更流程补齐 `raw_data`，供 Qlib 回测做开盘可成交约束

正式 Tushare 表统一放在 `data/tushare/`，不再使用 `data/data/tushare/` 之类的重复目录。

## 当前正式数据

### `data/tushare/` 必备表

- `daily_basic.parquet`
- `income.parquet`
- `balancesheet.parquet`
- `cashflow.parquet`
- `fina_indicator.parquet`
- `index_daily.parquet`
- `index_weight.parquet`
- `namechange.parquet`
- `stock_basic.csv`
- `stock_industry.csv`

### 其他正式依赖

- `data/qlib_data/cn_data/`：Qlib provider
- `data/qlib_data/raw_data/`：涨跌停开盘可成交约束用的原始日线文件

注意：

- `index_weight.parquet` 是 `selection.universe: csi300` 的前提
- `namechange.parquet` 是 `exclude_st: true` 的前提
- `raw_data` 不放在 `data/tushare/`，但由同一套日更流程维护

## 标准更新方式

优先使用统一入口：

```bash
python main.py update
```

这条命令会负责：

1. 更新核心 Tushare 表
2. 更新 `stock_basic.csv` 和 `stock_industry.csv`
3. 更新 `index_daily.parquet`
4. 更新 `index_weight.parquet`
5. 更新 `namechange.parquet`
6. 增量更新 `data/qlib_data/raw_data/`
7. 运行正式数据预检
8. 重新生成策略选股结果

如果你只是要跑正式 `select / backtest / compare`，通常不需要单独执行别的下载脚本。

## 手动下载方式

如果只想补某一类 Tushare 表，可以直接用：

```bash
python modules/data/tushare_downloader.py --type all --start 20160101
python modules/data/tushare_downloader.py --type index_weight --start 20160101
python modules/data/tushare_downloader.py --type namechange --start 20100101
```

当前 `--type` 支持：

- `all`
- `daily_basic`
- `income`
- `balancesheet`
- `cashflow`
- `fina_indicator`
- `index_daily`
- `index_weight`
- `namechange`

这个下载器只负责把表落到 `data/tushare/`。正式使用前，仍建议跑一次 `python main.py update` 或至少执行一次正式预检。

## 正式预检

`select`、`backtest`、`compare` 在正式链路里会调用 `modules/data/precheck.py`。当前会检查：

- Qlib provider 是否齐备，日期是否覆盖回测区间
- `daily_basic / income / balancesheet / cashflow / fina_indicator / index_daily`
- `stock_basic.csv / stock_industry.csv`
- 当股票池是 `csi300` 时，`index_weight.parquet` 是否存在且包含 `000300.SH`
- 当 `exclude_st: true` 时，`namechange.parquet` 是否存在

缺失核心数据时会直接失败，不会再静默回退到全市场，也不会把历史 ST 过滤降级成 no-op。

## 当前数据口径注意事项

- Qlib 的 `$open/$high/$low` 当前只用于审计，不作为正式成交约束输入
- 正式成交约束使用 `data/qlib_data/raw_data/` 下的原始日线文件
- `index_weight` 是按调仓日回看最近一期快照，不是静态成分股名单
- `namechange` 的作用是做历史 ST 区间过滤，不是只看当前股票简称

## 不再推荐的旧做法

下面这些说法已经过时，不应继续沿用：

- 使用 `scripts/download_tushare.py`、`scripts/process_tushare_data.py`、`scripts/convert_to_qlib.py` 作为官方主流程
- 把 `data/data/tushare/` 当作有效数据目录
- 缺少 `index_weight` 或 `namechange` 时仍声称结果通过了历史成分股 / 历史 ST 验证
- 直接用 Qlib 的 `$open/$high/$low` 做正式涨跌停可成交判断

## 相关文档

- `README.md`
- `results/top15_historical_csi300_research_20260322.md`
- `results/tradability_constraint_compare_20260322_185830.md`
- `results/price_field_audit_20260322_183351.md`

*最后更新：2026-03-22*
