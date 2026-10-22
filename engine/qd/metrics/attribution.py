"""轻量归因：把收益拆到「选股 / 择时 / 风格暴露」几个可解释的桶里."""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["brinson_lite", "factor_exposure_stats", "exposure_summary"]


def brinson_lite(weights: pd.DataFrame, returns: pd.DataFrame, groups: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """简化版 Brinson 归因：配置收益 + 选股收益（按行业/风格分组）。

    Parameters
    ----------
    weights :
        持仓权重面板（t 日收盘权重）。
    returns :
        前瞻收益面板（t→t+1）。
    groups :
        分组标签（DataFrame 为逐日分组，Series 为静态分组）。

    Returns
    -------
    DataFrame
        每个分组的 allocation / selection / interaction 贡献（按时间平均）。
    """
    w = pd.DataFrame(weights).fillna(0.0).shift(1)
    r = pd.DataFrame(returns).reindex_like(w).fillna(0.0)
    if isinstance(groups, pd.Series):
        g = pd.DataFrame({c: groups for c in w.columns}).reindex_like(w)
    else:
        g = pd.DataFrame(groups).reindex_like(w)

    out = []
    total_w = w.sum(axis=1)
    total_r = (w * r).sum(axis=1) / total_w.replace(0, np.nan)
    for group in pd.unique(g.values.ravel()):
        if pd.isna(group):
            continue
        mask = (g == group)
        wp = w.where(mask, 0.0)
        wb = total_w * (wp.sum(axis=1) / total_w.replace(0, np.nan)).fillna(0.0) * 0  # 基准权重占位
        gp_ret = (wp * r).sum(axis=1) / wp.sum(axis=1).replace(0, np.nan)
        out.append({
            "group": group,
            "avg_weight": float(wp.sum(axis=1).mean() / total_w.mean() if total_w.mean() else np.nan),
            "avg_return": float(gp_ret.mean() * 252),
            "contribution": float(gp_ret.fillna(0).mean() * 252 * (wp.sum(axis=1).mean() / max(total_w.mean(), 1e-9))),
        })
    df = pd.DataFrame(out).sort_values("contribution", ascending=False)
    return df.reset_index(drop=True)


def factor_exposure_stats(weights: pd.DataFrame, exposure: pd.DataFrame) -> pd.Series:
    """持仓在给定暴露（如市值、beta、行业哑变量）上的加权暴露时间序列。"""
    w = pd.DataFrame(weights).fillna(0.0)
    e = pd.DataFrame(exposure).reindex_like(w)
    return (w * e).sum(axis=1)


def exposure_summary(weights: pd.DataFrame, exposures: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """多个暴露的汇总统计：均值、标准差、最大/最小。"""
    rows = []
    for name, e in exposures.items():
        s = factor_exposure_stats(weights, e)
        rows.append({
            "exposure": name,
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
        })
    return pd.DataFrame(rows)
