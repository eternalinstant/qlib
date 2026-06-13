"""交易摩擦成本计算。

公共库：纯算术，无外部依赖。原驻 modules/backtest/common.py，五层迁移下沉至此。
"""


def compute_trade_cost(
    buy_value: float,
    sell_value: float,
    buy_commission_rate: float = 0.0003,
    sell_commission_rate: float = 0.0003,
    sell_stamp_tax_rate: float = 0.001,
    slippage_bps: float = 5,
    impact_bps: float = 2,
) -> float:
    """计算单次换仓的总摩擦成本。

    买入成本 = buy_value * (买入佣金率 + 滑点)
    卖出成本 = sell_value * (卖出佣金率 + 印花税率 + 滑点)
    滑点 = (slippage_bps + impact_bps) / 10000
    """
    slippage_pct = (slippage_bps + impact_bps) / 10000
    buy_cost = buy_value * (buy_commission_rate + slippage_pct)
    sell_cost = sell_value * (sell_commission_rate + sell_stamp_tax_rate + slippage_pct)
    return buy_cost + sell_cost
