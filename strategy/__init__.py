"""策略层（strategy）——产出统一信号契约 TargetPortfolio 或逐日 on_bar 事件。

五层架构第 2 层（app → strategy → engine → data → common）。
因子/模型/PE producer 产出 TargetPortfolio；规则策略（海龟/金字塔）产出 on_bar 事件。
"""
