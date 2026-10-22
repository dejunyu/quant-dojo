"""pytest 共享 fixture."""
import numpy as np
import pandas as pd
import pytest

# 刻意**不**把 engine/ 加进 sys.path：测试应当从「已安装的包」导入 qd。
# 之前这里有一行 sys.path.insert，它会掩盖打包配置的错误——
# 本地怎么跑都绿，CI 从干净 clone 安装后却必然失败。
from qd.data.synthetic import generate_demo_market, generate_random_walk_market
from qd.data.market import MarketData


@pytest.fixture(scope="session")
def md_small():
    """小规模合成数据（快）。"""
    return generate_demo_market(n_symbols=20, start="2019-01-01", end="2022-12-31", seed=3)


@pytest.fixture(scope="session")
def md():
    """标准合成数据。"""
    return generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)


@pytest.fixture(scope="session")
def rw():
    """零 alpha 随机游走数据（用于无前视与过拟合测试）。"""
    return generate_random_walk_market(n_symbols=20, start="2020-01-01", end="2022-12-31", seed=11)


@pytest.fixture(autouse=True)
def _isolate_factor_registry():
    """隔离因子注册表：测试里注册的自定义因子不能泄漏到别的测试。

    课程代码与 ``FactorLab.register`` 会往全局 ``FACTOR_REGISTRY`` 里加因子
    （例如 combo_equal、mom_adj、ep_orth）。如果不在用例之间清理，后续
    "因子总数 == 内置数量"之类的断言就会因为执行顺序不同而失败。
    """
    from qd.factors.base import FACTOR_REGISTRY

    snapshot = dict(FACTOR_REGISTRY)
    try:
        yield
    finally:
        FACTOR_REGISTRY.clear()
        FACTOR_REGISTRY.update(snapshot)
