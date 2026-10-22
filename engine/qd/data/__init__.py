"""数据层：行情容器、合成数据、外部数据源、本地存储."""
from .market import MarketData
from .synthetic import generate_demo_market

__all__ = ["MarketData", "generate_demo_market"]
