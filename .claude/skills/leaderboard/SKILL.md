# 策略 Leaderboard 技能

扫描所有回测结果，以中文表格展示策略排行榜，并给出中文解读。

## 参数说明

用户可在 $ARGUMENTS 中传入以下参数（均可选）：

- `--sort <字段>`：排序字段，可选 sharpe / cagr / calmar / max_dd / oos_sharpe / oos_cagr / oos_decay / cost_ratio / turnover / last_run（默认 sharpe）
- `--top <N>`：只显示前 N 条
- `--min-sharpe <值>`：最低夏普过滤，如 1.5
- `--max-dd <值>`：最大回撤下限，如 -0.10（回撤不超过 -10%）
- `--not-stale <天数>`：仅显示最近 N 天内有回测的策略
- `--grep <关键字>`：按策略名关键字过滤，如 push25 或 alpha158
- `--no-save`：只展示不保存快照
- `--format markdown`：输出 Markdown 格式

## 执行步骤

1. 解析 $ARGUMENTS，构造完整命令
2. 运行：`python3 main.py leaderboard --format markdown $ARGUMENTS`
3. 将输出的 Markdown 表格原样展示给用户
4. 在表格下方用中文给出解读，包括：
   - 列名含义说明（首次出现时）
   - 前 3 名策略点评（收益/风险/OOS表现）
   - 红色预警：OOS衰减 < 0.5 的策略（可能过拟合），用⚠️标出
   - 过期策略提示（超过30天未回测）

## 列名中文含义

| 列名 | 含义 |
|------|------|
| 策略名称 | 策略标识符 |
| 年化收益 | 全期年化收益率（复利） |
| 最大回撤 | 全期最大峰谷回撤 |
| 夏普比率 | 全期年化夏普（收益/波动） |
| 卡玛比率 | 年化收益 / 最大回撤绝对值 |
| OOS年化 | 样本外（2024-01-01后）年化收益 |
| OOS夏普 | 样本外夏普比率 |
| OOS衰减 | OOS夏普 / 全期夏普，<0.5 警示过拟合 |
| 年化换手 | 年化买卖次数（有日频CSV才有值） |
| 成本占比 | 交易成本占毛收益比例 |
| 日胜率 | 日收益为正的天数占比 |
| 调仓频率 | biweek双周 / week周 / month月 |
| 持股数 | 同时持有的股票数量 |
| 股票池 | all / csi300 / csi500 / csi800 / csi1000 |
| 最近回测 | 最后一次回测日期 |
| 是否过期 | 超过30天未回测则为"是" |

## 注意事项

- 不修改任何策略或代码，只读取已有回测结果
- 数据来源：results/analysis/all_return_series_metrics.csv（当前本地汇总快照）
- 快照保存在 results/leaderboard/ 目录
