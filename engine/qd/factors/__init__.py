"""因子层.

一个因子 = ``Factor`` 对象：有名字、分类、方向、所需字段，以及
``compute(md) -> DataFrame(dates × symbols)``。

方向约定
--------
``direction=+1`` 表示因子值越大 → 未来收益越高（做多高分组）；
``direction=-1`` 表示因子值越大 → 未来收益越低（即反向因子，如高估值）。
平台在合成与评分时会统一把因子**先按 direction 调整为正号因子**再做加权，
避免"低 PE 好"和"高动量好"混在一起时权重方向打架。
"""
from .base import Factor, FactorSpec, FACTOR_REGISTRY, get_factor, list_factors, register, FACTOR_CATEGORIES
from .pipeline import winsorize, zscore, neutralize, rank, combine, orthogonalize, standardize
from .library import *  # noqa: F401,F403 - 注册内置因子
from .library import BUILTIN_FACTORS

__all__ = [
    "Factor", "FactorSpec", "FACTOR_REGISTRY", "get_factor", "list_factors", "register",
    "FACTOR_CATEGORIES", "BUILTIN_FACTORS",
    "winsorize", "zscore", "neutralize", "rank", "combine", "orthogonalize", "standardize",
]
