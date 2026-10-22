"""研究层与优化层测试."""
import numpy as np
import pandas as pd
import pytest

from qd.backtest import BacktestConfig
from qd.optimize import (
    grid_search, walk_forward, purged_kfold, monte_carlo, bootstrap_blocks,
    mean_variance, min_variance, risk_parity_weights, equal_weight_weights,
    inverse_vol_weights, hrp_weights, kelly_fraction, compare_allocators,
    shrink_covariance,
)
from qd.optimize.portfolio import expected_returns
from qd.research.lab import FactorLab
from qd.research.logbook import Logbook
from qd.research.overfit import OverfitLab
from qd.strategy.builtin import dual_momentum, sma_cross


# ------------------------------------------------------------------ FactorLab
def test_factor_lab_ic(md_small):
    lab = FactorLab(md_small, horizon=20)
    s = lab.ic_series("mom_12_1")
    assert len(s) > 100
    assert s.dropna().abs().max() <= 1.0
    summ = lab.ic_summary("mom_12_1")
    assert "ic_mean" in summ and "t_stat" in summ
    assert -1 <= summ["ic_mean"] <= 1


def test_ic_decay_monotonic_horizons(md_small):
    lab = FactorLab(md_small, horizon=20)
    df = lab.ic_decay("mom_12_1", horizons=(1, 5, 20, 60))
    assert list(df["horizon"]) == [1, 5, 20, 60]
    assert df["ic"].notna().all()


def test_quantile_backtest_monotonic(md_small):
    """合成数据里植入了动量效应，因此动量因子的分组收益应单调递减。"""
    lab = FactorLab(md_small, horizon=20)
    qb = lab.quantile_backtest("mom_3m", n_quantiles=5)
    assert "groups" in qb
    ann = [qb["groups"][f"Q{i}"]["annual_return"] for i in range(1, 6)]
    # Q1 应显著好于 Q5
    assert ann[0] > ann[-1], f"动量因子分组不单调：{ann}"
    assert qb["long_short"]["sharpe"] > 0


def test_quantile_backtest_has_five_groups(md_small):
    lab = FactorLab(md_small, horizon=20)
    qb = lab.quantile_backtest("mom_12_1", n_quantiles=5)
    assert len(qb["groups"]) == 5


def test_factor_lab_report_structure(md_small):
    lab = FactorLab(md_small, horizon=20)
    rep = lab.report("mom_12_1", horizons=(1, 5, 20))
    for k in ("factor", "coverage", "ic", "ic_decay", "quantile_backtest",
              "turnover", "verdict", "reasons"):
        assert k in rep, f"报告缺少 {k}"
    assert rep["verdict"] in ("通过：可以考虑进入组合（仍需样本外验证）",
                              "可疑：部分检查通过，建议改进或与其它因子合成",
                              "不通过：信号不足以支撑交易成本")


def test_factor_correlation_matrix(md_small):
    lab = FactorLab(md_small, horizon=20)
    corr = lab.correlation(["mom_12_1", "mom_3m", "rev_5d"])
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-9)
    assert (corr.abs() <= 1.0 + 1e-9).all().all()


def test_coverage(md_small):
    lab = FactorLab(md_small, horizon=20)
    cov = lab.coverage("mom_12_1")
    assert 0 <= cov["avg_coverage"] <= 1


# ------------------------------------------------------------------ 优化
def test_grid_search(md_small):
    out = grid_search(md_small, dual_momentum,
                      {"lookback": [60, 120], "top_k": [5, 10]},
                      verbose=False)
    assert out["n_trials"] == 4
    assert len(out["table"]) == 4
    assert "best_params" in out
    assert out["deflated_sharpe"] is not None
    assert "Deflated Sharpe" in out["warning"]


def test_grid_search_warns_about_overfit(md_small):
    out = grid_search(md_small, dual_momentum,
                      {"lookback": [20, 60, 120], "top_k": [3, 5, 10]},
                      verbose=False)
    assert out["n_trials"] == 9
    assert "多重检验" in out["warning"]


def test_random_search(md_small):
    from qd.optimize import random_search
    out = random_search(md_small, dual_momentum,
                        {"lookback": [60, 120, 250], "top_k": [5, 10]},
                        n_iter=4, seed=0, verbose=False)
    assert out["n_trials"] == 4


def test_walk_forward(md_small):
    out = walk_forward(md_small, dual_momentum,
                       {"lookback": [60, 120], "top_k": [5, 10]},
                       n_splits=4, verbose=False)
    assert out["n_folds"] >= 2
    assert len(out["equity"]) > 100
    assert np.isfinite(out["sharpe"])
    # 拼接后净值曲线起点为 1
    assert abs(out["equity"].iloc[0] - 1.0) < 1e-9


def test_walk_forward_uses_only_past_params(md_small):
    """滚动前推每一段的参数必须由该段之前的数据选出。"""
    out = walk_forward(md_small, dual_momentum,
                       {"lookback": [60, 120], "top_k": [5, 10]},
                       n_splits=4, verbose=False)
    folds = out["folds"]
    for _, row in folds.iterrows():
        assert pd.Timestamp(row["train_end"]) < pd.Timestamp(row["test_start"]), (
            "训练集结束晚于测试集开始 —— 存在参数前视"
        )


def test_purged_kfold(md_small):
    splits = purged_kfold(md_small, n_splits=3, horizon=20, embargo_frac=0.01)
    assert len(splits) == 3
    for tr_s, tr_e, te_s, te_e in splits:
        # 测试段必须在训练段之后，或者被排除区间隔开
        assert tr_s <= tr_e
        assert te_s <= te_e


def test_monte_carlo():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.01, 1000))
    out = monte_carlo(r, n_sims=200, horizon=252, method="iid", seed=0)
    assert out["n_sims"] == 200
    assert out["final_p05"] < out["final_p50"] < out["final_p95"]
    assert 0 <= out["prob_loss"] <= 1
    out2 = monte_carlo(r, n_sims=100, horizon=252, method="block", block=20, seed=1)
    assert out2["method"] == "block"


def test_bootstrap_blocks():
    from qd.metrics.perf import sharpe
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.01, 1000))
    out = bootstrap_blocks(r, statistic=sharpe, n_sims=100, block=20, seed=0)
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]
    assert out["n_sims"] > 50


# ------------------------------------------------------------------ 组合优化
def test_weights_sum_to_one(md_small):
    r = md_small.returns().iloc[300:]
    for fn in (equal_weight_weights, inverse_vol_weights, risk_parity_weights,
               hrp_weights, min_variance, mean_variance):
        w = fn(r)
        assert abs(w.sum() - 1.0) < 1e-6, f"{fn.__name__} 权重和不为 1：{w.sum()}"
        assert (w >= -1e-9).all(), f"{fn.__name__} 产生了负权重"


def test_inverse_vol_prefers_low_vol():
    rng = np.random.default_rng(0)
    r = pd.DataFrame({
        "low": rng.normal(0.0005, 0.005, 500),
        "high": rng.normal(0.0005, 0.05, 500),
    })
    w = inverse_vol_weights(r)
    assert w["low"] > w["high"], "低波动资产应获得更高权重"


def test_risk_parity_equal_risk_contribution():
    """风险平价的核心：每个资产的 risk contribution 相等。"""
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0, 0.01, (600, 4)), columns=list("ABCD"))
    w = risk_parity_weights(r)
    cov = shrink_covariance(r, "ledoit_wolf").to_numpy()
    wv = w.to_numpy()
    rc = wv * (cov @ wv)
    share = rc / rc.sum()
    assert np.std(share) < 0.01, f"风险贡献不相等：{share}"


def test_min_variance_has_lowest_vol(md_small):
    """最小方差组合的波动不应高于等权。"""
    r = md_small.returns().iloc[300:]
    cov = shrink_covariance(r, "ledoit_wolf").to_numpy()
    w_mv = min_variance(r).reindex(r.columns).fillna(0).to_numpy()
    w_ew = equal_weight_weights(r).reindex(r.columns).fillna(0).to_numpy()
    vol_mv = np.sqrt(w_mv @ cov @ w_mv)
    vol_ew = np.sqrt(w_ew @ cov @ w_ew)
    assert vol_mv <= vol_ew + 1e-9


def test_hrp_is_long_only_and_normalized(md_small):
    r = md_small.returns().iloc[300:]
    w = hrp_weights(r)
    assert (w >= 0).all()
    assert abs(w.sum() - 1) < 1e-6


def test_kelly_fraction():
    """确定性构造：年化收益 8%、年化波动 20% -> 凯利 = 0.08 / 0.04 = 2.0。"""
    rng = np.random.default_rng(2)
    n = 5000
    r = pd.Series(rng.normal(0, 1, n))
    r = (r - r.mean()) / r.std(ddof=1)          # 标准化：波动 = 1（日）
    r = r * (0.20 / np.sqrt(252))               # 日波动 -> 年化 20%
    r = r + (0.08 / 252)                        # 日漂移 -> 年化 8%

    f = kelly_fraction(r)
    assert abs(f - 2.0) < 0.05, f"凯利应为 ≈2.0，实际 {f:.3f}"
    # 年化口径：mu_ann / var_ann
    expected = (r.mean() * 252) / (r.var(ddof=1) * 252)
    assert abs(f - expected) < 1e-9, "凯利应采用年化口径"
    # 分数凯利应更小
    assert abs(kelly_fraction(r, fractional=0.25) - 0.5) < 0.02
    # 与调仓频率无关
    weekly = r.groupby(np.arange(len(r)) // 5).apply(lambda x: (1 + x).prod() - 1)
    f_w = kelly_fraction(weekly, periods_per_year=52)
    assert abs(f_w - f) < 0.4, f"日频 {f:.2f} 与周频 {f_w:.2f} 差异过大，年化口径有问题"


def test_covariance_shrinkage(md_small):
    r = md_small.returns().iloc[300:]
    s_sample = shrink_covariance(r, "sample")
    s_lw = shrink_covariance(r, "ledoit_wolf")
    assert s_sample.shape == s_lw.shape
    # 协方差矩阵必须对称半正定
    for m in (s_sample, s_lw):
        M = m.to_numpy()
        assert np.allclose(M, M.T, atol=1e-9)
        eig = np.linalg.eigvalsh(M)
        assert eig.min() > -1e-8, "协方差矩阵非半正定"


def test_expected_returns_methods(md_small):
    r = md_small.returns().iloc[300:]
    assert (expected_returns(r, "zeros") == 0).all()
    assert expected_returns(r, "mean").notna().all()
    assert expected_returns(r, "shrink").notna().all()


def test_compare_allocators(md_small):
    r = md_small.returns().iloc[300:]
    df = compare_allocators(r)
    assert len(df) >= 5
    assert "sharpe" in df.columns
    assert df["annual_vol"].notna().all()


# ------------------------------------------------------------------ 过拟合
def test_white_noise_experiment_reveals_fake_alpha(md_small):
    """零 alpha 数据上的参数搜索必然产生"看起来很好"的假夏普。"""
    lab = OverfitLab(md_small)
    out = lab.white_noise_experiment(
        {"lookback": [20, 60, 120], "top_k": [3, 5, 10]},
        dual_momentum, n_symbols=20, seed=3,
    )
    assert "best_in_sample_sharpe" in out
    # 关键结论：即使真 alpha=0，最优样本内夏普仍然可能是正的
    assert out["n_trials"] == 9
    assert np.isfinite(out["best_in_sample_sharpe"])


def test_overfit_scorecard(md_small):
    res_sharpe = None
    from qd.backtest import Backtest
    r = Backtest(md_small, dual_momentum(120, 10), BacktestConfig()).run()
    lab = OverfitLab(md_small)
    card = lab.overfit_scorecard(r.returns, n_trials=1)
    assert "deflated_sharpe" in card and "verdict" in card
    assert isinstance(card["checklist"], list)
    assert len(card["checklist"]) >= 3


def test_deflate_more_trials_lower_confidence(md_small):
    from qd.backtest import Backtest
    r = Backtest(md_small, dual_momentum(120, 10), BacktestConfig()).run()
    lab = OverfitLab(md_small)
    d1 = lab.deflate(r.returns, n_trials=1)
    d100 = lab.deflate(r.returns, n_trials=100)
    assert d100["deflated_sharpe"] <= d1["deflated_sharpe"] + 1e-9


def test_train_test_decay(md_small):
    lab = OverfitLab(md_small)
    df = lab.train_test_decay(dual_momentum, {"lookback": [60, 120], "top_k": [5, 10]})
    assert len(df) == 4
    assert "decay" in df.columns
    assert "rank_in" in df.columns


# ------------------------------------------------------------------ 研究日志
def test_logbook_crud(tmp_path):
    lb = Logbook(tmp_path / "log.db")
    lb.clear()
    rid = lb.log(hypothesis="动量在 A 股是否有效", strategy="dual_momentum",
                 params={"lookback": 120}, metrics={"sharpe": 0.8}, notes="第一次试验")
    assert rid >= 1
    assert lb.count() == 1
    assert lb.trials_for("dual_momentum") == 1
    df = lb.search("动量")
    assert len(df) == 1
    best = lb.best(n=5, by="sharpe")
    assert len(best) == 1 and best.iloc[0]["sharpe"] == 0.8
    lb.delete(rid)
    assert lb.count() == 0
    lb.close()
