"""策略基类.

一个策略只需要实现一个方法：

```python
class MyStrategy(Strategy):
    def weights(self, ctx) -> pd.Series:
        close = ctx.close()          # 只有截至今天的收盘价
        score = close.iloc[-1] / close.iloc[-20] - 1
        return self.top_k(score, k=10)
```

``ctx`` 由回测引擎构造，**只暴露 i 日及以前的数据**，因此你写不出前视逻辑
（想写也拿不到未来数据）。
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

__all__ = ["Strategy", "rule", "weight_from_score"]


class Strategy:
    """策略基类."""

    name: str = "strategy"
    description: str = ""

    def weights(self, ctx) -> pd.Series | None:
        """返回目标权重（index 为标的代码）。返回 None 表示本日不调仓。"""
        raise NotImplementedError

    # ---------------------------------------------------------- 工具方法
    @staticmethod
    def top_k(score: pd.Series, k: int, weight: float = 1.0) -> pd.Series:
        """横截面取前 k 名等权（或给定总权重）。"""
        s = pd.Series(score).dropna()
        out = pd.Series(0.0, index=pd.Series(score).index)
        if s.empty or k <= 0:
            return out.fillna(0.0)
        k = min(k, len(s))
        picked = s.sort_values(ascending=False).index[:k]
        out.loc[picked] = weight / k
        return out.fillna(0.0)

    @staticmethod
    def bottom_k(score: pd.Series, k: int, weight: float = 1.0) -> pd.Series:
        s = pd.Series(score).dropna()
        out = pd.Series(0.0, index=pd.Series(score).index)
        if s.empty or k <= 0:
            return out.fillna(0.0)
        k = min(k, len(s))
        picked = s.sort_values().index[:k]
        out.loc[picked] = weight / k
        return out.fillna(0.0)

    @staticmethod
    def equal_weight(symbols: Sequence[str], weight: float = 1.0) -> pd.Series:
        if not len(symbols):
            return pd.Series(dtype=float)
        return pd.Series(weight / len(symbols), index=list(symbols))

    @staticmethod
    def normalize(w: pd.Series, gross: float = 1.0) -> pd.Series:
        s = pd.Series(w).fillna(0.0)
        total = s.abs().sum()
        if total == 0:
            return s
        return s * gross / total

    def __call__(self, ctx):
        return self.weights(ctx)

    def __repr__(self) -> str:
        return f"<Strategy {self.name}>"


def long_top_weights(score, k: int, gross: float = 1.0) -> pd.Series:
    """numpy 快速版：取分数最高的 k 个标的等权。

    与 ``weight_from_score(..., mode="long_top", k=k)`` 结果一致，但走 numpy
    的 argpartition/argsort，在"每个交易日调用一次、50 个标的、1800 天"的
    场景下比 pandas 版本快一个数量级（不构造中间 DataFrame/索引）。
    """
    if isinstance(score, pd.Series):
        idx = score.index
        vals = score.to_numpy(dtype=float, copy=False)
    else:
        vals = np.asarray(score, dtype=float)
        idx = None
    out = np.zeros(vals.shape, dtype=float)
    valid = np.isfinite(vals)
    n_valid = int(valid.sum())
    if n_valid == 0 or k <= 0:
        return pd.Series(out, index=idx) if idx is not None else out
    k = min(int(k), n_valid)
    # 有效集合里取前 k
    pos = np.flatnonzero(valid)
    v = vals[pos]
    if k < len(v):
        top = np.argpartition(-v, k - 1)[:k]
    else:
        top = np.arange(len(v))
    out[pos[top]] = gross / k
    return pd.Series(out, index=idx) if idx is not None else out


def classify_score(score, ctx_syms):
    """把 Series 快速转成"与 symbols 对齐的 ndarray"，供 long_top_weights 使用。"""
    if isinstance(score, pd.Series) and score.index.equals(pd.Index(ctx_syms)):
        return score.to_numpy(dtype=float, copy=False)
    return score


def weight_from_score(score: pd.Series, mode: str = "long_top", k: int = 10,
                      gross: float = 1.0, long_only: bool = True) -> pd.Series:
    """把横截面打分转成权重.

    mode :
        ``"long_top"`` 做多前 k；``"long_short"`` 多前 k 空后 k；
        ``"demean"`` 横截面去均值后归一（多空且美元中性）。
    """
    s = pd.Series(score).dropna()
    out = pd.Series(0.0, index=pd.Series(score).index)
    if s.empty:
        return out
    if mode == "demean":
        d = s - s.mean()
        tot = d.abs().sum()
        if tot == 0:
            return out
        out.loc[d.index] = d * (gross / tot)
        if long_only:
            out = out.clip(lower=0)
            t = out.sum()
            if t > 0:
                out = out * gross / t
        return out.fillna(0.0)

    k = min(k, len(s) // 2 if mode == "long_short" else len(s))
    ordered = s.sort_values(ascending=False)
    longs = ordered.index[:k]
    out.loc[longs] = gross * (0.5 if mode == "long_short" else 1.0) / max(k, 1)
    if mode == "long_short" and not long_only:
        shorts = ordered.index[-k:]
        out.loc[shorts] = -gross * 0.5 / max(k, 1)
    return out.fillna(0.0)


def rule(fn: Callable) -> Callable:
    """把函数包装成策略（用于快速实验）。

    ```python
    @rule
    def my_rule(ctx):
        return ...
    ```
    """
    class _FnStrategy(Strategy):
        name = getattr(fn, "__name__", "rule")

        def weights(self, ctx):
            return fn(ctx)

    _FnStrategy.__name__ = getattr(fn, "__name__", "FnStrategy")
    return _FnStrategy()
