"""纸面交易（Paper Trading）.

用途不是"模拟赚钱"，而是**验证代码链路**：
你的策略在"每天只看到当天为止的数据"这个约束下，还能不能跑出与回测一致的
结果？很多 bug（用了未来数据、用了未复权价格、复权后权重算错）只有在这条
链路上才会暴露。

工作流程
--------
1. 从某个日期开始，逐日推进；
2. 每天只用 **截至昨天收盘** 的数据算目标权重；
3. 用**今天的开盘价**成交（与回测引擎完全相同的协议）；
4. 记录每天的持仓、成交、净值，供与回测对齐比对。

如果纸面交易的结果与回测差异很大，说明你的回测或策略里有隐藏的前视。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..backtest.broker import Broker
from ..backtest.config import BacktestConfig
from ..backtest.engine import _to_target_array
from ..data.market import MarketData

__all__ = ["PaperTrader"]


class PaperTrader:
    """纸面交易运行器."""

    def __init__(self, md: MarketData, strategy, config: BacktestConfig | None = None,
                 gate=None):
        self.md = md
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.gate = gate
        self.broker = Broker(md, self.config)
        self.log: list[dict] = []

    def run(self, start=None, end=None) -> dict:
        """逐日推进，返回运行记录。"""
        md = self.md.slice(start=start, end=end) if (start or end) else self.md
        # Broker 必须基于完整数据（价格数组），但只允许在 <= i 日成交
        broker = self.broker
        dates = md.dates
        offsets = {d: i for i, d in enumerate(self.md.dates)}
        idxs = [offsets[d] for d in dates]

        equity_rows = []
        pending: np.ndarray | None = None
        n = len(dates)
        for k, i in enumerate(idxs):
            broker.unlock_daily()
            if pending is not None and k > 0:
                broker.order_target_weights(i, pending, reason="paper")
                pending = None
            if k < n - 1:
                ctx = _PaperContext(self.md, i)
                out = self.strategy.weights(ctx) if hasattr(self.strategy, "weights") else self.strategy(ctx)
                if out is not None:
                    # 复用回测引擎的同一套转换逻辑，保证两条链路完全一致
                    pending = _to_target_array(out, self.md.symbols)
            equity_rows.append({
                "date": dates[k],
                "equity": broker.mark_to_market(i),
                "cash": broker.cash,
                "n_positions": int(np.count_nonzero(broker.qty)),
            })

        df = pd.DataFrame(equity_rows).set_index("date")
        df["equity"] = df["equity"] / df["equity"].iloc[0]
        return {
            "equity": df["equity"],
            "daily": df,
            "fills": broker.fills_frame(),
            "trades": broker.trades_frame(),
            "final_value": float(df["equity"].iloc[-1]),
        }


class _PaperContext:
    """与回测 _Context 完全一致的截断上下文（保证链路一致）。"""

    __slots__ = ("md", "i", "date", "_cache")

    def __init__(self, md: MarketData, i: int):
        self.md = md
        self.i = i
        self.date = md.dates[i]
        self._cache: dict[str, pd.DataFrame] = {}

    def _slice(self, key: str) -> pd.DataFrame:
        if key not in self._cache:
            src = self.md.panels.get(key)
            if src is None:
                src = self.md.fundamentals.get(key)
            if src is None:
                raise KeyError(f"字段 {key} 不存在")
            self._cache[key] = src.iloc[: self.i + 1]
        return self._cache[key]

    def close(self):
        return self._slice("close")

    def open(self):
        return self._slice("open")

    def high(self):
        return self._slice("high")

    def low(self):
        return self._slice("low")

    def volume(self):
        return self._slice("volume")

    def amount(self):
        return self._slice("amount") if "amount" in self.md else self.close() * self.volume()

    def returns(self):
        return self.close().pct_change()

    def field(self, name):
        return self._slice(str(name).lower())

    def fundamental(self, name):
        return self._slice(str(name).lower())

    @property
    def symbols(self):
        return list(self.md.symbols)

    def __repr__(self):
        return f"<PaperContext i={self.i} date={self.date.date()}>"
