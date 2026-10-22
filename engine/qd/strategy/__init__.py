"""策略层."""
from .base import Strategy, rule
from .builtin import (
    sma_cross, dual_momentum, factor_long_only, pairs_trading, vol_target,
    bollinger_reversion, turtle_breakout, risk_parity, rotation, buy_and_hold,
    mean_reversion_basket, multi_factor_rank, equal_weight,
)

__all__ = [
    "Strategy", "rule",
    "sma_cross", "dual_momentum", "factor_long_only", "pairs_trading", "vol_target",
    "bollinger_reversion", "turtle_breakout", "risk_parity", "rotation", "buy_and_hold",
    "mean_reversion_basket", "multi_factor_rank", "equal_weight",
]
