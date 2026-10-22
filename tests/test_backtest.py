"""回测引擎与绩效指标测试."""
import numpy as np
import pandas as pd
import pytest

from qd.backtest import Backtest, BacktestConfig, CostModel
from qd.backtest.broker import Broker
from qd.metrics.perf import (
    sharpe, sortino, max_drawdown, annual_return, calmar, perf_report,
    value_at_risk, conditional_var, drawdown_table, probabilistic_sharpe,
    deflated_sharpe, monthly_returns_table,
)
from qd.strategy.builtin import (
    buy_and_hold, equal_weight, dual_momentum, sma_cross, risk_parity,
)


# ------------------------------------------------------------------ 绩效
def test_sharpe_on_known_series():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 2520))
    s = sharpe(r, rf=0.0)
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert abs(s - expected) < 1e-9
    assert -2 < s < 5


def test_sharpe_zero_vol():
    r = pd.Series([0.01] * 100)
    assert np.isnan(sharpe(r))


def test_max_drawdown():
    eq = pd.Series([1.0, 1.1, 1.2, 0.9, 1.0, 0.8, 1.5])
    assert abs(max_drawdown(eq) - (0.8 / 1.2 - 1)) < 1e-12
    # 单调上涨 -> 0 回撤
    assert max_drawdown(pd.Series([1.0, 1.1, 1.2])) == 0.0


def test_drawdown_table():
    eq = pd.Series([1.0, 1.2, 0.9, 1.0, 1.3, 1.0, 1.4],
                   index=pd.date_range("2024-01-01", periods=7))
    tbl = drawdown_table(eq, top=3)
    assert len(tbl) >= 1
    assert tbl.iloc[0]["depth"] < 0
    assert "recovery_days" in tbl.columns


def test_annual_return_compounding():
    # 每天涨 0.1%，252 天后应约为 1.001^252 - 1
    r = pd.Series([0.001] * 252)
    eq = (1 + r).cumprod()
    ar = annual_return(eq)
    assert abs(ar - (1.001 ** 252 - 1)) < 1e-9


def test_var_cvar():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 2000))
    v_hist = value_at_risk(r, 0.05, "historical")
    v_gauss = value_at_risk(r, 0.05, "gaussian")
    c = conditional_var(r, 0.05)
    assert v_hist > 0
    assert abs(v_hist - v_gauss) < 0.005   # 正态数据下两种方法应接近
    assert c >= v_hist - 1e-9             # CVaR 必然不轻于 VaR


def test_perf_report_keys(md_small):
    res = Backtest(md_small, equal_weight(), BacktestConfig()).run()
    rep = perf_report(res.equity, rf=0.02)
    for k in ("sharpe", "annual_return", "annual_vol", "max_drawdown",
              "calmar", "win_rate", "var_95", "probabilistic_sharpe"):
        assert k in rep, f"报告缺少 {k}"


def test_psr_dsr_bounds(md_small):
    res = Backtest(md_small, dual_momentum(120, 10), BacktestConfig()).run()
    r = res.returns
    psr = probabilistic_sharpe(r)
    dsr1 = deflated_sharpe(r, n_trials=1)
    dsr100 = deflated_sharpe(r, n_trials=100)
    assert 0.0 <= psr <= 1.0
    assert 0.0 <= dsr1 <= 1.0
    # 试验次数越多，DSR 必须越低（这是 DSR 的核心性质）
    assert dsr100 <= dsr1 + 1e-9, "试验次数增加后 DSR 反而上升，公式有问题"


def test_monthly_returns_table(md_small):
    res = Backtest(md_small, equal_weight(), BacktestConfig()).run()
    tbl = monthly_returns_table(res.equity)
    assert not tbl.empty
    assert tbl.shape[1] >= 12


# ------------------------------------------------------------------ 撮合
def test_cost_model_charges_commission_and_tax():
    cost = CostModel(commission_rate=0.0003, stamp_tax=0.001, min_commission=5.0)
    notional = 100_000.0
    assert abs(cost.commission(notional) - 30.0) < 1e-9
    assert abs(cost.commission(1000.0) - 5.0) < 1e-9    # 最低佣金
    assert cost.tax(notional, "sell") == 100.0
    assert cost.tax(notional, "buy") == 0.0             # 买入无印花税


def test_slippage_direction():
    cost = CostModel(slippage_bps=10.0)
    assert cost.slippage(100.0, "buy") > 0     # 买入价更高
    assert cost.slippage(100.0, "sell") < 0    # 卖出价更低


def test_impact_cost_increases_with_size():
    cost = CostModel(impact_coeff=0.1)
    small = cost.impact(100.0, 1000, adv=1_000_000)
    large = cost.impact(100.0, 100_000, adv=1_000_000)
    assert large > small >= 0


def test_t_plus_1_blocks_same_day_sell(md_small):
    """T+1：当日买入的股票当日不能卖出。"""
    cfg = BacktestConfig(t_plus_1=True, allow_short=False)
    b = Broker(md_small, cfg)
    i = 200
    w = pd.Series(0.0, index=md_small.symbols)
    w.iloc[0] = 0.5
    b.order_target_weights(i, w.to_numpy(), reason="test")
    assert b.qty[0] > 0, "买入未成交"
    assert b.locked[0] == b.qty[0], "T+1 未冻结份额"

    # 同日尝试清仓，应无法卖出
    cash_before = b.cash
    zero = np.zeros(len(md_small.symbols))
    b.order_target_weights(i, zero, reason="test-sell")
    assert b.qty[0] > 0, "T+1 下当日买入的股票被卖出了"


def test_lot_size_rounding(md_small):
    cfg = BacktestConfig(lot_size=100)
    b = Broker(md_small, cfg)
    w = pd.Series(0.0, index=md_small.symbols)
    w.iloc[0] = 0.3
    b.order_target_weights(200, w.to_numpy())
    assert b.qty[0] % 100 == 0, f"股数 {b.qty[0]} 不是 100 的整数倍"


def test_cannot_buy_limit_up(md_small):
    """涨停不能买入。"""
    cfg = BacktestConfig(limit_up_buy=False)
    b = Broker(md_small, cfg)
    up, _ = b.limit_masks(300)
    if not up.any():
        pytest.skip("该日没有涨停股")
    j = int(np.argmax(up))
    w = np.zeros(len(md_small.symbols))
    w[j] = 0.5
    b.order_target_weights(300, w)
    assert b.qty[j] == 0, "涨停股被买入了"


def test_volume_participation_cap(md_small):
    """单日成交不能超过成交量参与率上限。"""
    cfg = BacktestConfig(max_participation=0.01, lot_size=1)
    b = Broker(md_small, cfg)
    i = 400
    w = np.zeros(len(md_small.symbols))
    w[0] = 1.0    # 要求满仓单只股票
    b.order_target_weights(i, w)
    vol = float(md_small.volume().iloc[i].iloc[0])
    assert b.qty[0] <= vol * 0.01 + 1, f"成交量 {b.qty[0]} 超过上限 {vol*0.01}"


def test_cash_never_negative(md_small):
    cfg = BacktestConfig()
    res = Backtest(md_small, dual_momentum(60, 5), cfg).run()
    assert (res.daily_stats["cash"] >= -1e-6).all(), "现金为负，账户穿仓了"


def test_gross_exposure_within_limit(md_small):
    cfg = BacktestConfig(gross_leverage=1.0)
    res = Backtest(md_small, dual_momentum(60, 5), cfg).run()
    assert res.daily_stats["gross"].max() <= 1.05, "总敞口超过上限"


def test_equity_starts_at_one(md_small):
    res = Backtest(md_small, equal_weight(), BacktestConfig()).run()
    assert abs(res.equity.iloc[0] - 1.0) < 1e-12


def test_cost_reduces_return(md_small):
    """成本越高性能越差 —— 验证成本模型真的在起作用。"""
    cheap = BacktestConfig(cost=CostModel(commission_rate=0.0, stamp_tax=0.0, slippage_bps=0.0, impact_coeff=0.0))
    costly = BacktestConfig(cost=CostModel(commission_rate=0.003, stamp_tax=0.002, slippage_bps=30.0, impact_coeff=0.5))
    r1 = Backtest(md_small, dual_momentum(60, 5), cheap).run()
    r2 = Backtest(md_small, dual_momentum(60, 5), costly).run()
    assert r2.equity.iloc[-1] <= r1.equity.iloc[-1], "高成本版本的收益反而更高，成本模型无效"


def test_stop_loss_triggers(md_small):
    cfg = BacktestConfig(stop_loss=0.02)   # 极紧的止损必然触发
    res = Backtest(md_small, dual_momentum(60, 5), cfg).run()
    fills = res.fills
    if not fills.empty:
        assert "风控平仓" in set(fills["reason"]), "止损从未触发"


def test_max_drawdown_halt(md_small):
    cfg = BacktestConfig(max_drawdown_stop=0.05)
    res = Backtest(md_small, dual_momentum(60, 5), cfg).run()
    reasons = set(res.fills["reason"]) if not res.fills.empty else set()
    assert "回撤熔断清仓" in reasons, "回撤熔断从未触发"


def test_backtest_result_api(md_small):
    res = Backtest(md_small, dual_momentum(120, 10), BacktestConfig()).run()
    assert isinstance(res.metrics, dict)
    assert isinstance(res.trades, pd.DataFrame)
    assert isinstance(res.fills, pd.DataFrame)
    assert isinstance(res.monthly_returns(), pd.DataFrame)
    s = res.summary()
    assert "夏普" in s
    d = res.to_dict()
    assert "metrics" in d and "trade_stats" in d


def test_rebalance_frequency_reduces_turnover(md_small):
    """调仓频率越低，换手越少、交易越少。"""
    results = {}
    for freq in ("daily", "weekly", "monthly", "quarterly"):
        cfg = BacktestConfig(rebalance=freq)
        r = Backtest(md_small, dual_momentum(60, 5), cfg).run()
        turnover = float(r.weights.diff().abs().sum(axis=1).mean())
        results[freq] = (turnover, r.trade_stats().get("n_trades", 0))
    assert results["quarterly"][0] < results["monthly"][0] <= results["weekly"][0] <= results["daily"][0], (
        f"换手率未随调仓频率降低：{results}"
    )
    assert results["quarterly"][1] < results["daily"][1]


def test_rebalance_frequency_is_validated(md_small):
    with pytest.raises(ValueError):
        Backtest(md_small, dual_momentum(60, 5), BacktestConfig(rebalance="每两天")).run()


def test_benchmark_aligned(md_small):
    cfg = BacktestConfig(benchmark=md_small.symbols[0])
    res = Backtest(md_small, equal_weight(), cfg).run()
    assert res.benchmark is not None
    assert len(res.benchmark) == len(res.equity)
    assert "beta" in res.metrics and "alpha" in res.metrics
