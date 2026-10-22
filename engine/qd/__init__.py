"""量化平台核心包 qd (Quant Dojo).

子模块
------
data      行情与基本面数据容器、合成数据、akshare 数据源、本地存储
factors   因子库与因子处理流水线
research  单因子研究实验室（IC / 分层回测 / 衰减 / 相关性）
backtest  事件驱动 + 逐日撮合的回测引擎（含成本、滑点、涨跌停、T+1）
strategy  策略基类与内置策略工厂
metrics   绩效与风险指标
optimize  参数搜索、滚动前推、组合优化
tutor     课程体系（扫描 docs 下的 Markdown 课程）
api       FastAPI 服务
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
