"""杂项工具函数."""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["to_datetime", "annualization_factor", "ensure_dir", "timer", "safe_div"]


def to_datetime(x) -> pd.Timestamp:
    """把各种日期表达转换为 pandas Timestamp（天级精度）。"""
    return pd.Timestamp(x).normalize()


def annualization_factor(index: pd.Index, fallback: int = 252) -> float:
    """根据索引频率推断年化因子（一年有多少个观测）。

    规则按"平均间隔天数"判断：

    * ≤1.5 天   → 日频（252）
    * ≤8 天     → 周频（52）
    * ≤35 天    → 月频（12）
    * ≤100 天   → 季频（4）
    * 其它      → 用 ``252 / 间隔天数``（覆盖每 20 日调仓这类非标准频率）

    最后一条很关键：频率采样后的收益序列（如按 20 个交易日调仓）平均间隔
    就是 20 天，若硬套"月频 = 12"会把夏普算大一个数量级。
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return float(fallback)
    delta = np.median(np.diff(index.values).astype("timedelta64[s]").astype(float))
    if delta <= 0:
        return float(fallback)
    seconds_per_year = 365.25 * 24 * 3600
    if delta <= 120:  # 分钟/秒级
        return seconds_per_year / delta
    days = delta / 86400.0
    if days <= 3.0:      # 日频（含节假日跳空）
        return 252.0
    if days <= 9.0:      # 周频
        return 52.0
    return max(1.0, 252.0 / days)


def ensure_dir(path) -> Path:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def timer(label: str = "", verbose: bool = True):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if verbose:
            print(f"[{label}] 耗时 {time.perf_counter() - t0:.3f}s")


def safe_div(a, b, fill: float = np.nan):
    """安全除法：分母为 0/NaN 时返回 fill。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.isnan(a) | np.isnan(b) | (b == 0), fill, a / b)
    return out
