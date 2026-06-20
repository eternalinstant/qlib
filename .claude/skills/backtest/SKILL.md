# 回测技能

执行步骤：
1. 如需清缓存，检查 `src/engine/` 和 `results/` 下是否有明确的临时产物；不要删除策略配置或正式归档结果
2. 运行回测：`python3 main.py backtest -e qlib $ARGUMENTS`
3. 以 Markdown 表格展示：年化收益率、夏普比率、最大回撤、日胜率
4. 如可用，与沪深300基准对比
5. 不要修改任何策略或因子代码，只运行并报告

规则驱动策略使用：

```bash
python3 main.py backtest -e rule -s src/strategy/producers/configs/<strategy>.yaml
```
