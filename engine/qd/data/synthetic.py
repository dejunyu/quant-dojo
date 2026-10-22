"""合成行情生成器（教学 / 离线演示 / 单元测试用）.

为什么需要合成数据
------------------
1. 平台必须能**离线**教学：没网、没 token 也能跑完整流程；
2. 研究方法的正确性（IC、分层回测、回测引擎是否无前视）在有已知真值的数据
   上验证最可靠；
3. 合成数据里我们**刻意植入**了弱动量、短期反转、价值、低波、质量等效应，
   并有可控的随机噪声，用来演示「因子有效但不神奇」的真实体验。

生成器保证：价格为正、有停牌（volume=0）、有涨跌停迹象、行业与市值分层，
但**不保证**任何真实市场的统计特性——它只用于学习和流程验证。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..utils.calendar import trading_days
from .market import MarketData

__all__ = ["generate_demo_market", "generate_random_walk_market"]

_INDUSTRIES = [
    "银行", "食品饮料", "医药生物", "电子", "电力设备", "计算机",
    "房地产", "有色金属", "汽车", "机械设备", "国防军工", "公用事业",
]


def generate_demo_market(
    n_symbols: int = 50,
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    seed: int = 7,
    n_industries: int | None = None,
    annual_vol: float = 0.28,
    annual_drift: float = 0.20,
) -> MarketData:
    """生成一份带弱因子效应的合成 A 股市场数据.

    Parameters
    ----------
    n_symbols :
        标的数量。
    start, end :
        区间（周一至周五作为交易日）。
    seed :
        随机种子，固定种子可复现。
    annual_vol, annual_drift :
        标的年化波动率与漂移。

    Returns
    -------
    MarketData
        含 open/high/low/close/volume/amount 与 pe/pb/roe/mktcap/industry 基本面。
    """
    rng = np.random.default_rng(seed)
    dates = trading_days(start, end, market="A")
    n_dates = len(dates)
    if n_dates < 60:
        raise ValueError("区间太短，至少需要 60 个交易日")

    k_ind = int(n_industries or min(len(_INDUSTRIES), max(4, n_symbols // 5)))
    symbols = [f"{600000 + i * 7 // 5:06d}" for i in range(n_symbols)]
    industry = pd.Series(rng.choice(_INDUSTRIES[:k_ind], size=n_symbols), index=symbols)

    # ---------- 静态属性 ----------
    log_mktcap = rng.normal(23.0, 1.1, size=n_symbols)  # 市值跨度约 2 个数量级
    mktcap0 = np.exp(log_mktcap) * 1e4
    quality_z = rng.normal(0, 1, size=n_symbols)          # 质量（ROE）潜变量
    value_z = rng.normal(0, 1, size=n_symbols)            # 价值（低估值）潜变量

    # ---------- 共同市场因子 ----------
    dt = 1.0 / 252
    market_shock = rng.normal(0, 1, size=n_dates) * np.sqrt(dt)
    # 市场状态切换：牛/熊/震荡，制造真实的波动聚集
    regime = np.zeros(n_dates)
    state = 0
    for t in range(n_dates):
        if rng.random() < 0.004:
            state = rng.integers(-1, 2)
        regime[t] = state
    market_ret = annual_drift * dt + annual_vol * 0.55 * market_shock + regime * 0.0004

    # ---------- 行业因子 ----------
    ind_idx = {name: i for i, name in enumerate(_INDUSTRIES[:k_ind])}
    ind_code = np.array([ind_idx[i] for i in industry.values])
    ind_shock = rng.normal(0, 1, size=(n_dates, k_ind)) * np.sqrt(dt)
    ind_ret = 0.22 * annual_vol * ind_shock[:, ind_code]

    # ---------- 个股特质收益 ----------
    idio = rng.normal(0, 1, size=(n_dates, n_symbols)) * np.sqrt(dt)
    idio_ret = annual_vol * 0.75 * idio

    beta = rng.uniform(0.7, 1.4, size=n_symbols)
    # 低波效应：特质波动高的股票长期收益偏低（低波动异象）
    idio_vol = np.std(idio, axis=0) * np.sqrt(252)
    low_vol_prem = -0.06 * (idio_vol / idio_vol.mean() - 1.0) * dt

    total_ret = (market_ret[:, None] * beta[None, :]) + ind_ret + idio_ret
    total_ret = total_ret + low_vol_prem[None, :]

    # ---------- 植入可预测结构（弱 alpha）----------
    # 1) 中期动量：过去 60 日收益对下期有微弱正向预测
    # 2) 短期反转：过去 5 日收益对下期有微弱负向预测
    # 3) 价值 / 质量：与潜变量成正比
    mom_signal = np.zeros((n_dates, n_symbols))
    rev_signal = np.zeros((n_dates, n_symbols))
    for t in range(n_dates):
        if t >= 60:
            past60 = total_ret[t - 60 : t].sum(axis=0)
            mom_signal[t] = _zscore(past60)
        if t >= 5:
            rev_signal[t] = _zscore(total_ret[t - 5 : t].sum(axis=0))
    alpha = (0.00035 * mom_signal) - (0.00055 * rev_signal)
    alpha += (0.00018 * value_z[None, :] + 0.00015 * quality_z[None, :]) * dt * 252
    # alpha 只影响未来（在构造时已天然对齐到 t+1 的收益上）
    total_ret = total_ret + np.vstack([alpha[1:], np.zeros(n_symbols)])

    # ---------- 价格路径 ----------
    px0 = np.clip(np.exp(rng.normal(3.0, 0.7, size=n_symbols)), 3.0, 300.0)
    close = px0[None, :] * np.exp(np.cumsum(total_ret - 0.5 * (total_ret ** 2).mean(), axis=0))
    # 用累计收益直接生成价格，避免漂移失真
    close = px0[None, :] * np.cumprod(1.0 + total_ret, axis=0)

    # ---------- 交易摩擦：停牌、涨跌停 ----------
    tradable = rng.random((n_dates, n_symbols)) > 0.004
    limit_up = rng.random((n_dates, n_symbols)) < 0.006
    limit_down = rng.random((n_dates, n_symbols)) < 0.005

    prev_close = np.vstack([np.full(n_symbols, np.nan), close[:-1]])
    close = np.where(tradable, close, prev_close)
    close = np.nan_to_num(close, nan=px0[0])
    close = np.where(limit_up, prev_close * 1.1, close)
    close = np.where(limit_down, prev_close * 0.9, close)
    close = np.maximum(close, 0.5)
    close = np.nan_to_num(close, nan=px0[0])

    intraday_vol = np.abs(rng.normal(0, 1, size=(n_dates, n_symbols))) * 0.008 + 0.002
    open_ = close * (1.0 + rng.normal(0, 1, size=(n_dates, n_symbols)) * 0.004)
    high = np.maximum(open_, close) * (1.0 + intraday_vol)
    low = np.minimum(open_, close) * (1.0 - intraday_vol)
    low = np.maximum(low, 0.01)

    base_volume = np.exp(rng.normal(15.0, 1.0, size=n_symbols))
    volume = base_volume[None, :] * np.exp(rng.normal(0, 0.45, size=(n_dates, n_symbols)))
    volume = volume * (1.0 + 0.6 * (limit_up | limit_down))  # 涨停放量
    volume = np.where(tradable, volume, 0.0)

    amount = close * volume
    vwap = (close * (1.0 + rng.normal(0, 0.0015, size=(n_dates, n_symbols)))).clip(low, high)

    # ---------- 基本面（缓慢变化 + 随价格更新估值）----------
    eps = np.exp(rng.normal(0.6, 0.5, size=n_symbols)) * (1.0 + 0.10 * quality_z)
    roe = np.clip(0.08 + 0.05 * quality_z + rng.normal(0, 0.02, size=(n_dates, n_symbols)), -0.3, 0.6)
    n_shares = (mktcap0 / px0)
    mktcap = close * n_shares[None, :]
    pe = mktcap / (eps[None, :] * n_shares[None, :] * (1.0 + 0.02 * np.arange(n_dates)[:, None] / 252))
    pe = np.clip(pe, 3.0, 400.0)
    pb = np.clip(pe / np.clip(roe * 100, 1.0, None) * 12.0, 0.3, 30.0)
    ps = np.clip(pe * rng.uniform(0.05, 0.25, size=n_symbols)[None, :], 0.2, 60.0)

    def panel(mat: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(mat, index=dates, columns=symbols)

    md = MarketData(
        panels={
            "open": panel(open_),
            "high": panel(high),
            "low": panel(low),
            "close": panel(close),
            "volume": panel(volume),
            "amount": panel(amount),
            "vwap": panel(vwap),
        },
        fundamentals={
            "pe": panel(pe),
            "pb": panel(pb),
            "ps": panel(ps),
            "roe": panel(roe),
            "mktcap": panel(mktcap),
            "float_mktcap": panel(mktcap * rng.uniform(0.4, 0.95, size=n_symbols)[None, :]),
        },
        name=f"synthetic-{n_symbols}",
        meta={"seed": seed, "generator": "generate_demo_market", "synthetic": True},
    )
    # industry 是字符串，单独用 DataFrame 存（不作为浮点 panel）
    md.fundamentals["industry"] = pd.DataFrame(
        np.tile(industry.values, (n_dates, 1)), index=dates, columns=symbols
    )
    return md


def generate_random_walk_market(
    n_symbols: int = 30,
    start: str = "2020-01-01",
    end: str = "2022-12-31",
    seed: int = 0,
) -> MarketData:
    """生成**纯随机游走**数据（无任何可预测结构）。

    用途：验证「过拟合检测器」。在一个真 alpha 为零的数据集上做参数搜索，
    最优样本内 Sharpe 一定 > 0，但它应该是虚假的——这是每个量化研究者都
    必须亲手做一次的实验。
    """
    rng = np.random.default_rng(seed)
    dates = trading_days(start, end, market="A")
    n = len(dates)
    ret = rng.normal(0, 0.016, size=(n, n_symbols))
    symbols = [f"RW{i:03d}" for i in range(n_symbols)]
    close = 100.0 * np.cumprod(1.0 + ret, axis=0)

    # 构造合理的 OHLCV，让撮合能真实发生（否则零成交量会导致永远无法成交）
    intraday = np.abs(rng.normal(0, 0.006, size=(n, n_symbols))) + 0.002
    open_ = np.vstack([close[0] * (1 - ret[0]), close[:-1] * (1 + ret[1:])])
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)
    volume = np.exp(rng.normal(14.0, 0.6, size=(n, n_symbols)))

    def panel(mat):
        return pd.DataFrame(mat, index=dates, columns=symbols)

    md = MarketData(
        panels={
            "open": panel(open_), "high": panel(high), "low": panel(low),
            "close": panel(close), "volume": panel(volume),
            "amount": panel(close * volume), "vwap": panel(close),
        },
        name=f"randomwalk-{n_symbols}",
        meta={"synthetic": True, "generator": "generate_random_walk_market", "zero_alpha": True},
    )
    return md


def _zscore(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mu, sd = np.nanmean(arr), np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(arr)
    return np.clip((arr - mu) / sd, -4, 4)
