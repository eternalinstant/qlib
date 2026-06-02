# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 运行要求

所有脚本必须用 `C:\Python312\python.exe` 跑——这是装了 `xtquant` 的解释器。运行任何脚本前，本地必须先启动 MiniQMT 客户端（国金QMT交易端模拟）。

```
C:\Python312\python.exe demo.py
C:\Python312\python.exe test_connection.py
C:\Python312\python.exe test_download.py
C:\Python312\python.exe live_trade.py --dry-run   # 只算不下单；--live 才真实下单
```

## 架构

本仓库分两层：一层是 `xtquant` 的薄封装（行情 + 交易），另一层是建在其上的实盘调仓执行器。另有一个独立的 FastAPI/qlib 研究工作台嵌在 `qlib_ui/` 里（独立子项目，见下文）。

### 封装层

对 `xtquant`（MiniQMT SDK）的薄封装：

**`qmt_data.py`** —— 封装 `xtquant.xtdata`
- 实时 tick、K 线历史、财务报表、板块/指数成分、交易日历
- 先调 `qmt_data.init()` 连接数据服务（IP/端口来自 `config.py`）
- `get_bars()` 是核心方法，`get_daily()` / `get_minutes()` 是便捷封装
- 历史数据返回 `{stock_code: DataFrame}`；tick 数据返回原始 dict

**`qmt_trade.py`** —— 封装 `xtquant.xttrader`
- 下单、撤单，以及账户/持仓/委托/成交查询
- 用模块级单例 `_trader` 和 `_account`；调 `qmt_trade.init()` 连接
- `buy()`/`sell()` 按 `price > 0` 自动选限价/最新价；市价变体（`buy_market`、`sell_market`）用交易所对应的五档即成剩撤类型
- `config.py` 里 `ACCOUNT_ID` 可留空——`get_account()` 自动检测第一个可用账户

**`config.py`** —— 连接相关常量（路径、端口、会话 ID、账户、默认复权方式）

### 实盘交易层

一个每日一次的调仓执行器，把实盘（当前为模拟）账户调整到与目标持仓一致。**信号来源是解耦的**：qlib 模型跑在*另一台*机器（一台 Mac）上，产出每天的 top-K 选股并导出成目标文件；本机只负责读这个文件并执行。本机没有装 qlib、没有模型、也没有 qlib 行情数据——**不要假设可以在本机算信号**。

数据流：

```
[Mac] qlib 预测 → export_target.py → targets/target_YYYYMMDD.json   （同步过来）
[Win] target_portfolio.load_today_target() → rebalancer.plan_orders() → live_trade.py
```

- **`target_portfolio.py`** —— 加载并校验 `targets/target_YYYYMMDD.json`（跨机契约：`{date, strategy, invested_pct, positions:[{symbol, weight}]}`）。文件 `date` ≠ 今天则拒绝（防陈旧文件）。`normalize_symbol()` 接受 `SH600641` / `600641` / `600641.SH`，统一输出 QMT 后缀格式 `600641.SH`。
- **`rebalancer.py`** —— 纯计算「大脑」，无副作用（便于离线测试）。`plan_orders()` 接收已抓好的账户/行情/合约快照，返回计划订单，套用 A 股规则：T+1 可卖量（`can_use_volume`）、100 股整手（仅清仓时才卖零股）、涨/跌停跳过、停牌/无报价跳过、限价带滑点且封顶/封底于涨跌停价、现金约束、**先卖后买排序**。
- **`live_trade.py`** —— 执行器入口。预检（交易日、交易时段、kill-switch、连接、账户）→ 出计划 → 换手率风控闸 → 限价下单（先卖后买）→ 轮询 `get_orders(cancelable_only=True)`，超时撤单，重新出计划并重试最多 `MAX_RETRIES` 次 → 写 `reports/exec_YYYYMMDD.json`。**幂等**：它是「调到目标」，所以重跑只补残差；目标已达成则零下单。
- **`live_config.py`** —— 实盘参数，与连接配置分开：`TARGET_DIR`、`REBALANCE_TIME`、`SLIPPAGE_BPS`、`ORDER_TIMEOUT_SEC`、`MAX_RETRIES`、`DRY_RUN`、`MAX_ORDER_NOTIONAL`、`MAX_TURNOVER`、`KILL_SWITCH_FILE`、`TRADING_WINDOWS`。
- **`export_target.py`** —— **仅 Mac 侧参考模板**，本机无法运行（没有 qlib）。说明如何把一次 qlib 预测转成目标文件。

`live_trade.py` 设计为由 Windows 任务计划程序每个交易日在 `REBALANCE_TIME` 拉起一次；脚本不常驻。用 `--dry-run` / `--live` 覆盖 `live_config.DRY_RUN`。

### qlib_ui/（嵌套子项目）

`qlib_ui/` 是一个独立的 FastAPI + React 量化回测工作台，**有它自己的 `CLAUDE.md`**——动它之前先读那个。它是挖掘这些实盘策略的上游研究工具。本机上只有它在 `qlib_ui/results/` 里的回测输出 CSV 可用（它驱动的 qlib 引擎在 Mac 上）。注意这些 CSV 用 `SH600641` 前缀代码格式、股票名是 GBK 编码——实盘目标契约刻意只用代码不用中文名，就是为了绕开编码乱码。

## 关键约定

- 股票代码：`XXXXXX.SZ`（深圳）或 `XXXXXX.SH`（上海）
- `dividend_type` 取值：`"none"`、`"front"`、`"back"`、`"front_ratio"`、`"back_ratio"`——默认值在 `config.py` 设
- K 线 `period` 取值：`tick`、`1m`、`5m`、`15m`、`30m`、`60m`、`1h`、`1d`、`1w`、`1mon`
- 封装层函数出错时返回 `None`/`[]`/`{}` 并向 stdout 打 `[ERROR]`，从不抛异常。实盘层相反：校验输入并在预检阶段大声中止，而不是拿坏数据去交易。
- 多实例同时运行时，`SESSION_ID` 每个进程必须唯一
- 执行器只需要实时数据（`get_ticks`、`get_instrument_detail`）加交易日历——它**不在本机算因子**，所以实盘不需要批量下载 K 线。若 `is_trading_day()` 意外返回 False，多半是交易日历没加载。
- 数据服务连上（`qmt_data.init()`）不代表交易能用：`qmt_trade.init()` 返回 -1 且查不到账户，说明 QMT 交易端没登录。预检会捕获并安全中止。

## 脚本

| 脚本 | 用途 |
|---|---|
| `live_trade.py` | 每日调仓执行器——读当天目标文件，把账户调到一致（`--dry-run`/`--live`） |
| `demo.py` | 端到端测试所有数据与交易接口 |
| `test_connection.py` | 最小连通性检查（直接调 xtquant，不走封装） |
| `test_download.py` | 下载历史数据并验证读取 |
| `export_target.py` | Mac 侧生成目标文件的参考模板（本机不可运行） |
