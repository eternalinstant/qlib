# Current Strategy Summary

Updated: 2026-06-20

## 一句话

当前主线不是单一 `alpha158`，而是两类东西混在一起看。2026-06 之后仓库已经切到 `src/` layout，模型配置在 `src/strategy/configs/models/`，历史结果仍在 `results/model_signals/`。

- QVF / 现金流 / 质量因子主线
- Alpha158 小因子实验线

如果只记结论：

- 阶段一最终归档方案是主备 `60/40`：`alpha158_momentum_volume_k6_dd10_overlay` + `push25_cq10_v3_vol_norm`
- QVF/现金流/质量单策略里仍重点看 `push25_cq10_k8d2_very_tight`
- 想要更简单、更低回撤一点的是 `push25_cq7_k8d2_very_tight`
- 新增容量/股票池验证线：`qvf_alpha158_core12_top30_cap200w`、`qvf_alpha158_core12_csi1000`、`qvf_alpha158_core12_csi1000_top30_cap200w`
- Alpha158 支线里，收益最高的是 `alpha158_prune_r1_drop_rsv20_k6`
- Alpha158 支线里，回撤更小的是 `alpha158_prune_r2_drop_vsumd20_k6`

## 1. 先怎么评价策略

只记因子没意义，至少要一起看这 4 件事：

1. `Holdout / OOS` 收益
2. `Holdout / OOS` 最大回撤
3. `Holdout / OOS` 夏普
4. 因子数和可解释性

我的建议顺序是：

1. 先看 `2024-01-01` 到 `2026-04-15` 的历史 `holdout / OOS`
2. 再看全样本历史回测
3. 最后才看因子数量是不是更少、更容易记

原因很简单：因子少不等于更强，收益高也不等于更稳，必须把收益、回撤、夏普一起看。

## 2. 当前主线

### 主线表现总览

| 策略 | 因子数 | 全样本年化 | 全样本回撤 | 全样本夏普 | OOS年化 | OOS回撤 | OOS夏普 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `qvf_core_plus_fixed80_k6_overlay_fullspan_turnover_soft_pruned` | 27 | `26.02%` | `-11.69%` | `1.70` | `25.37%` | `-11.69%` | `1.74` | 大综合基线，稳，但因子很多 |
| `push25_cq10_k8d2_very_tight` | 10 | `21.85%` | `-9.27%` | `1.87` | `26.04%` | `-8.56%` | `2.09` | 当前最平衡，风险收益比最好 |
| `push25_cq7_k8d2_very_tight` | 7 | `18.89%` | `-8.77%` | `1.71` | `20.53%` | `-7.04%` | `1.79` | 更轻、更容易记，回撤也更小 |

数据来源:

- 全样本 / OOS: [`../results/model_signals/validation_runs/strategy_validation_summary.csv`](../results/model_signals/validation_runs/strategy_validation_summary.csv)
- `push25` 的三窗口拆分: [`../results/model_signals/validation_runs/three_window_eval_le10_20260424/summary_enriched.csv`](../results/model_signals/validation_runs/three_window_eval_le10_20260424/summary_enriched.csv)

### `qvf_core_plus_fixed80_k6_overlay_fullspan_turnover_soft_pruned`

定位: 主基线 / `main_qvf`

因子数: `27`

来源: [`../results/model_signals/qvf_core_plus_fixed80_k6_overlay_fullspan_turnover_soft_pruned/training_summary.json`](../results/model_signals/qvf_core_plus_fixed80_k6_overlay_fullspan_turnover_soft_pruned/training_summary.json)

因子清单:

- 价值/现金流: `book_to_market`, `roic_proxy`, `ebit_to_mv`, `ebitda_to_mv`, `ocf_to_mv`, `ocf_to_ev`, `fcff_to_mv`
- 盈利质量: `roe_fina`, `roe_dt_fina`, `roa_fina`, `net_margin`
- 流动性/杠杆: `current_ratio_fina`, `quick_ratio_fina`, `debt_to_assets_fina`
- 增长/经营: `total_revenue_inc`, `operate_profit_inc`, `n_cashflow_act`
- 资金流: `smart_ratio_5d`, `net_mf_amount_5d`, `net_mf_amount_20d`, `net_mf_vol_5d`
- 聚合因子: `rank_value_profit_core`, `rank_flow_momentum_core`, `rank_growth_quality_core`, `rank_balance_core`, `qvf_core_alpha`, `qvf_core_interaction`

评价:

- 优点: 全样本和 OOS 都不差，算是稳健基线
- 缺点: 因子太多，不适合拿来做“我到底记哪几个因子”的速记版本
- 适合: 当对照组、基线仓、研究起点

### `push25_cq10_k8d2_very_tight`

定位: 当前最像“基本面/QVF + 少量 Alpha158 技术确认”的 hybrid 主线

因子数: `10`

来源: [`../src/strategy/configs/models/push25_cq10_k8d2_very_tight.yaml`](../src/strategy/configs/models/push25_cq10_k8d2_very_tight.yaml)

因子清单:

- 基本面/QVF: `ocf_to_ev`, `fcff_to_mv`, `roe_fina`, `current_ratio_fina`, `n_cashflow_act`, `rank_value_profit_core`, `rank_balance_core`, `qvf_core_interaction`
- Alpha158 技术: `ROC20`, `CORD20`

三窗口拆分:

- valid (`2023-01-01` 到 `2023-12-31`): 年化 `-4.63%`，回撤 `-8.69%`，夏普 `-0.82`
- holdout (`2024-01-01` 到 `2026-04-15`): 年化 `25.05%`，回撤 `-5.62%`，夏普 `1.92`

评价:

- 优点: OOS 夏普最高，OOS 回撤也控制得很好
- 缺点: `2023` 单年 valid 表现偏差，说明它不是每一段都漂亮
- 适合: 作为当前主用候选

### `push25_cq7_k8d2_very_tight`

定位: `push25_cq10` 的更轻版本

因子数: `7`

来源: [`../results/model_signals/validation_runs/three_window_eval_le10_20260424/summary_enriched.csv`](../results/model_signals/validation_runs/three_window_eval_le10_20260424/summary_enriched.csv)

因子清单:

- `ocf_to_ev`
- `fcff_to_mv`
- `roe_fina`
- `current_ratio_fina`
- `rank_value_profit_core`
- `qvf_core_interaction`
- `ROC20`

三窗口拆分:

- valid (`2023-01-01` 到 `2023-12-31`): 年化 `0.98%`，回撤 `-6.99%`，夏普 `0.18`
- holdout (`2024-01-01` 到 `2026-04-15`): 年化 `20.72%`，回撤 `-6.00%`，夏普 `1.71`

评价:

- 优点: 因子更少，OOS 回撤最低，结构更容易记
- 缺点: 收益和夏普略逊于 `push25_cq10`
- 适合: 想简化因子、优先控制回撤时用

## 3. 其他 QVF 变体

这两条更像“研究分支”，不是当前最强主线。

| 策略 | 因子数 | valid年化 | valid回撤 | valid夏普 | holdout年化 | holdout回撤 | holdout夏普 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `qvf_grouped7_fixed80_k6_overlay_fullspan` | 7 | `-5.59%` | `-16.40%` | `-0.46` | `5.30%` | `-15.34%` | `0.50` | grouped 思路成立，但不够强 |
| `qvf_grouped9_fixed80_k6_overlay_fullspan` | 9 | `-6.20%` | `-16.74%` | `-0.55` | `3.49%` | `-13.35%` | `0.37` | 比 grouped7 更复杂，但没更好 |

来源: [`../results/model_signals/validation_runs/three_window_eval_le10_20260424/summary_enriched.csv`](../results/model_signals/validation_runs/three_window_eval_le10_20260424/summary_enriched.csv)

因子清单:

### `qvf_grouped7_fixed80_k6_overlay_fullspan`

- `rank_value_cashflow_core`
- `rank_profitability_quality_core`
- `rank_balance_sheet_core`
- `rank_growth_cashflow_core`
- `rank_flow_liquidity_core`
- `qvf_group_alpha`
- `qvf_group_blend`

### `qvf_grouped9_fixed80_k6_overlay_fullspan`

- `rank_value_cashflow_core`
- `rank_profitability_quality_core`
- `rank_balance_sheet_core`
- `rank_growth_cashflow_core`
- `rank_flow_liquidity_core`
- `qvf_group_alpha`
- `qvf_group_interaction`
- `qvf_group_quality_anchor`
- `qvf_group_blend`

## 4. Alpha158 实验线

### Alpha158 表现总览

| 策略 | 因子数 | 年化 | 回撤 | 夏普 | 结论 |
|---|---:|---:|---:|---:|---|
| `alpha158_momentum_volume_k6` | 5 | valid `3.48%` / holdout `35.45%` | valid `-8.61%` / holdout `-9.67%` | valid `0.35` / holdout `1.88` | 原始 5 因子版，holdout 很强 |
| `alpha158_momentum_volume_compact3_k6` | 3 | valid `-2.81%` / holdout `14.72%` | valid `-13.43%` / holdout `-12.20%` | valid `-0.13` / holdout `0.97` | 3 因子直接压缩后明显变弱 |
| `alpha158_prune_r1_drop_rsv20_k6` | 4 | `27.23%` | `-17.77%` | `1.54` | Alpha158 支线里收益最好 |
| `alpha158_prune_r2_drop_vsumd20_k6` | 3 | `25.27%` | `-15.74%` | `1.49` | Alpha158 支线里回撤更小 |

数据来源:

- `momentum_volume_k6` / `compact3_k6`: [`../results/model_signals/validation_runs/three_window_eval_le10_20260424/summary_enriched.csv`](../results/model_signals/validation_runs/three_window_eval_le10_20260424/summary_enriched.csv)
- prune 版: [`../results/model_signals/alpha158_prune_runs/alpha158_momentum_volume_greedy_prune_v1/alpha158_prune_r1_drop_rsv20_k6/backtest_summary.json`](../results/model_signals/alpha158_prune_runs/alpha158_momentum_volume_greedy_prune_v1/alpha158_prune_r1_drop_rsv20_k6/backtest_summary.json), [`../results/model_signals/alpha158_prune_runs/alpha158_momentum_volume_greedy_prune_v1/alpha158_prune_r2_drop_vsumd20_k6/backtest_summary.json`](../results/model_signals/alpha158_prune_runs/alpha158_momentum_volume_greedy_prune_v1/alpha158_prune_r2_drop_vsumd20_k6/backtest_summary.json)

### `alpha158_momentum_volume_k6`

因子数: `5`

因子清单: `ROC20`, `RSV20`, `RANK20`, `CORD20`, `VSUMD20`

评价:

- holdout 非常强
- 但这条线整体是实验支线，不是你当前最该记住的主策略

### `alpha158_prune_r1_drop_rsv20_k6`

因子数: `4`

因子清单: `ROC20`, `RANK20`, `CORD20`, `VSUMD20`

评价:

- 如果只在 Alpha158 小因子里选“收益最好”的，就是它

### `alpha158_prune_r2_drop_vsumd20_k6`

因子数: `3`

因子清单: `ROC20`, `RANK20`, `CORD20`

评价:

- 如果只在 Alpha158 小因子里选“回撤更小、更好记”的，就是它

### `alpha158_csi300`

因子数: `8`

来源: 旧因子策略配置，当前工作树未保留该 YAML；如需恢复，应放在 `src/strategy/configs/strategies/experimental/alpha158/alpha158_csi300.yaml`

因子分层:

- alpha: `LOW0`, `VWAP0`, `KLOW`, `KLEN`
- enhance: `MIN10`, `QTLD5`
- risk: `VMA60`, `CORD5`

评价:

- 更像一份“Alpha158 精选策略配置”
- 不像你现在最常回忆的那条小因子路线

## 5. 以后回看时先看哪里

- 主策略总览: [`../results/model_signals/validation_runs/strategy_validation_summary.csv`](../results/model_signals/validation_runs/strategy_validation_summary.csv)
- 三窗口因子和 holdout: [`../results/model_signals/validation_runs/three_window_eval_le10_20260424/summary_enriched.csv`](../results/model_signals/validation_runs/three_window_eval_le10_20260424/summary_enriched.csv)
- 历史收益排名: [`../results/model_signals/validation_runs/historical_return_ranking_latest.csv`](../results/model_signals/validation_runs/historical_return_ranking_latest.csv)
- Alpha158 剪枝路径: [`../results/model_signals/alpha158_prune_runs/alpha158_momentum_volume_greedy_prune_v1/accepted_steps.csv`](../results/model_signals/alpha158_prune_runs/alpha158_momentum_volume_greedy_prune_v1/accepted_steps.csv)
- 当前模型配置: [`../src/strategy/configs/models`](../src/strategy/configs/models)
- 阶段一主备组合归档: [`../results/analysis/phase1_final_main_backup_60_40`](../results/analysis/phase1_final_main_backup_60_40)

## 6. 最短记忆版

如果以后只想记一句：

`当前主线 = QVF/现金流/质量因子为主，叠加少量 Alpha158 技术因子；主用候选偏向 push25_cq10，轻量版是 push25_cq7；Alpha158 小因子线主要记 ROC20、RANK20、CORD20 这支。`
