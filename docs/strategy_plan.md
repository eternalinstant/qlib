# 旧因子策略执行模式说明

## 当前状态

基于因子组合的旧策略执行层已经支持两种模式：

- `factor_topk`：按调仓频率重新排序选 Top-K，叠加 sticky/buffer/churn_limit 等稳定性控制。
- `stoploss_replace`：先建立 Top-K 持仓，之后只在持仓股从近期高点回撤超过阈值时替换。

相关实现：

- `src/strategy/selection.py`
- `src/strategy/builder.py`
- 全局默认配置：`src/common/configs/strategy.yaml`

---

## `stoploss_replace` 动态调仓逻辑

- **触发条件**：持仓股票从近期最高点下跌 >= 10%
- **调仓动作**：卖出该股票，从因子得分排名靠前的非持仓股票中选择一只买入
- **不调仓情况**：如果没有股票跌破10%阈值，则当日不进行任何操作

```
伪代码：
for each holding_stock:
    recent_high = max(holding_stock.price, lookback=20d)
    drawdown = (holding_stock.price - recent_high) / recent_high
    if drawdown <= -10%:
        sell(holding_stock)
        buy(best_available_stock_from_factor_pool)
```

---

## 关键参数

```yaml
selection:
  mode: stoploss_replace
  topk: 15
  universe: csi300
  stoploss_lookback_days: 20
  stoploss_drawdown: 0.10
  replacement_pool_size: 30
```

`factor_topk` 和 `stoploss_replace` 也都支持旧持仓自然月衰减：

```yaml
selection:
  monthly_decay: 0.2
```

含义：旧持仓每跨一个自然月，排序分数乘以 `1 - monthly_decay`。`0` 表示关闭，取值范围 `[0, 1]`。

---

## 使用建议

1. **低换手**：只在止损触发时调仓，避免频繁交易的手续费侵蚀
2. **顺势持有**：让盈利的股票继续持有，截断亏损
3. **因子驱动**：换入的股票来自因子选股池，保证质量
4. **持仓老化**：需要提高组合更新速度时，启用 `monthly_decay`
