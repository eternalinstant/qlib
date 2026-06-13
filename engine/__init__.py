"""引擎层（engine）——回测引擎，消费策略产出 → BacktestResult。

五层架构第 3 层（app → strategy → engine → data → common）。
引擎不在导入期依赖策略层具体实现；对外契约为 BacktestResult。
"""
