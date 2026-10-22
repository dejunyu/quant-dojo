"""无前视偏差测试 —— 平台最重要的正确性保证.

这组测试验证一个不变量：**任何策略都无法利用未来信息**。

验证方法（比"人工检查代码"可靠得多）
-----------------------------------
把数据在某个切点之后的**未来收益完全随机重排**。如果策略没有前视，那么它
在"原数据"和"未来被打乱的数据"上，**切点之前**的净值曲线必须完全一致
（因为切点前的决策不可能用到切点后的数据）。

如果策略偷看了未来，切点前的曲线就会发生变化——这是数学上的必然。
"""
import numpy as np
import pandas as pd
import pytest

from qd.backtest import Backtest, BacktestConfig
from qd.data.market import MarketData
from qd.data.synthetic import generate_demo_market
from qd.strategy.builtin import (
    sma_cross, dual_momentum, bollinger_reversion, risk_parity,
    mean_reversion_basket, turtle_breakout, rotation, equal_weight,
)


def _shuffle_future(md: MarketData, cut: int, seed: int = 0) -> MarketData:
    """把 cut 之后的收益序列随机重排，价格路径随之改变。

    保留 cut 之前的一切不变，只改变"未来"。
    """
    rng = np.random.default_rng(seed)
    px = md.close().copy()
    ret = px.pct_change()

    # 对 cut 之后的每日**横截面收益**做随机置换（打乱哪只股票涨）
    for i in range(cut, len(ret)):
        row = np.array(ret.iloc[i].to_numpy(), dtype=float, copy=True)
        rng.shuffle(row)
        ret.iloc[i] = row

    # 用打乱后的收益重建价格（从 cut-1 的收盘价出发）
    base = px.iloc[cut - 1].to_numpy()
    new = np.empty_like(px.to_numpy())
    new[:cut] = px.to_numpy()[:cut]
    cur = base.copy()
    for i in range(cut, len(ret)):
        cur = cur * (1.0 + ret.iloc[i].to_numpy())
        new[i] = cur
        # 同步调整 open/high/low，保持 OHLC 合理性
    new_px = pd.DataFrame(new, index=px.index, columns=px.columns)

    # 用同样的比例缩放 open/high/low
    scale = new_px / px
    panels = {}
    for field, df in md.panels.items():
        panels[field] = df * scale if field != "volume" else df
    funds = dict(md.fundamentals)
    return MarketData(panels, funds, name=md.name + "-shuffled")


@pytest.mark.parametrize("factory", [
    lambda: dual_momentum(lookback=60, top_k=5),
    lambda: sma_cross(fast=10, slow=30),
    lambda: bollinger_reversion(window=20, top_k=5),
    lambda: mean_reversion_basket(lookback=5, top_k=5),
    lambda: risk_parity(lookback=60),
    lambda: rotation(top_k=3, lookback=60),
    lambda: turtle_breakout(entry_window=30, n=5),
    lambda: equal_weight(),
])
def test_no_lookahead(factory, md_small):
    """切点之前的净值曲线，在"未来被打乱"后必须完全不变。"""
    cut = len(md_small.dates) // 2
    shuffled = _shuffle_future(md_small, cut, seed=42)

    cfg = BacktestConfig(initial_cash=1_000_000)
    r1 = Backtest(md_small, factory(), cfg).run()
    r2 = Backtest(shuffled, factory(), cfg).run()

    e1 = r1.equity.iloc[:cut]
    e2 = r2.equity.iloc[:cut]
    np.testing.assert_allclose(
        e1.to_numpy(), e2.to_numpy(), rtol=1e-9, atol=1e-9,
        err_msg="策略使用了未来信息（切点前的净值被未来的数据影响）",
    )


def test_no_lookahead_trades_identical_before_cut(md_small):
    """切点之前的成交记录也必须完全一致。"""
    cut = len(md_small.dates) // 2
    shuffled = _shuffle_future(md_small, cut, seed=7)
    cfg = BacktestConfig()
    r1 = Backtest(md_small, dual_momentum(lookback=60, top_k=5), cfg).run()
    r2 = Backtest(shuffled, dual_momentum(lookback=60, top_k=5), cfg).run()

    f1 = r1.fills
    f2 = r2.fills
    f1 = f1[f1.index < md_small.dates[cut]]
    f2 = f2[f2.index < md_small.dates[cut]]
    assert len(f1) == len(f2), f"成交流不同：{len(f1)} vs {len(f2)}"
    if len(f1):
        assert (f1["symbol"].to_numpy() == f2["symbol"].to_numpy()).all()
        assert (f1["side"].to_numpy() == f2["side"].to_numpy()).all()


def test_zero_alpha_data_gives_zero_alpha(rw):
    """在零 alpha 数据上，任何策略的样本内夏普都不应显著为正（统计意义上）。

    我们检查：多个策略在随机游走数据上的夏普，其均值应接近 0，而不是显著
    大于 0.5。
    """
    from qd.metrics.perf import sharpe

    strats = [
        dual_momentum(lookback=60, top_k=5),
        sma_cross(fast=10, slow=30),
        bollinger_reversion(window=20, top_k=5),
        risk_parity(lookback=60),
        mean_reversion_basket(lookback=5, top_k=5),
    ]
    sharpes = []
    for s in strats:
        r = Backtest(rw, s, BacktestConfig()).run()
        v = sharpe(r.returns)
        if np.isfinite(v):
            sharpes.append(v)
    assert len(sharpes) >= 3
    mean_sr = float(np.mean(sharpes))
    # 零 alpha + 交易成本 → 期望夏普应 <= 0.5（且通常 < 0）
    assert mean_sr < 0.5, f"零 alpha 数据上平均夏普 {mean_sr:.2f}，疑似存在前视偏差"


def test_execution_lag_is_real(md_small):
    """信号在 t 日收盘生成，必须在 t+1 日才成交（不能当日成交）。"""
    from qd.backtest.engine import _Context

    strat = dual_momentum(lookback=60, top_k=3)
    res = Backtest(md_small, strat, BacktestConfig()).run()
    fills = res.fills
    if fills.empty:
        pytest.skip("没有成交")

    # 用「当日收盘价」与「当日开盘价」对比：next_open 模式下成交价应等于开盘价
    openpx = md_small.open()
    first = fills.iloc[0]
    d = first.name
    sym = first["symbol"]
    idx = md_small.dates.get_loc(d)
    expected = float(openpx.iloc[idx][sym])
    assert abs(first["ref_price"] - expected) < 1e-6, (
        f"成交参考价应为开盘价 {expected}，实际 {first['ref_price']}"
    )


def test_factor_compute_has_no_lookahead(md_small):
    """因子在 t 日的值，不能依赖于 t 日之后的价格。"""
    from qd.factors import get_factor

    cut = len(md_small.dates) // 2
    shuffled = _shuffle_future(md_small, cut, seed=99)
    for name in ("mom_12_1", "mom_3m", "rev_5d", "vol_60d", "rsi_14", "ma_ratio_20_60"):
        f = get_factor(name)
        a = f.compute(md_small).iloc[:cut]
        b = f.compute(shuffled).iloc[:cut]
        np.testing.assert_allclose(
            a.to_numpy(), b.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True,
            err_msg=f"因子 {name} 使用了未来数据",
        )
