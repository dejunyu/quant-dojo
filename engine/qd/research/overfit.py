"""过拟合实验室（OverfitLab）.

这是平台里最"反直觉"也最重要的一个模块。它用**实验**而不是说教来回答
一个问题：*我的策略是真的，还是我搜出来的？*

三个实验
--------
1. **白噪声实验**：在纯随机游走数据上跑参数搜索，看最优样本内夏普能有多高。
   答案通常是 1.5~3.0——比大多数人真策略的夏普还高。这一步让研究者彻底
   放弃"夏普 > 2 就是好策略"的幻觉。
2. **多重检验惩罚**：给定试验次数 N，用 Deflated Sharpe Ratio 重新评估。
3. **样本内外衰减**：同一组参数在 train/test 上的表现落差，落差本身就是
   过拟合程度的度量。
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..backtest import Backtest, BacktestConfig
from ..data.market import MarketData
from ..data.synthetic import generate_random_walk_market
from ..metrics.perf import sharpe, deflated_sharpe, probabilistic_sharpe

__all__ = ["OverfitLab"]


class OverfitLab:
    """过拟合诊断实验室."""

    def __init__(self, md: MarketData, config: BacktestConfig | None = None):
        self.md = md
        self.config = config or BacktestConfig()

    # ------------------------------------------------------------ 实验 1
    def white_noise_experiment(
        self,
        param_grid: dict[str, Sequence],
        strategy_factory: Callable[..., object],
        n_symbols: int = 30,
        start: str = "2020-01-01",
        end: str = "2022-12-31",
        seed: int = 0,
    ) -> dict:
        """在**零 alpha** 的随机游走数据上做参数搜索。

        Parameters
        ----------
        param_grid :
            参数名 -> 候选值列表。
        strategy_factory :
            工厂函数 ``**params -> Strategy``。

        Returns
        -------
        dict
            含所有试验的夏普分布、最优夏普、以及对应的 Deflated Sharpe。
        """
        from ..optimize.grid import _expand_grid

        rw = generate_random_walk_market(n_symbols=n_symbols, start=start, end=end, seed=seed)
        combos = _expand_grid(param_grid)
        sharpes = []
        for combo in combos:
            strat = strategy_factory(**combo)
            res = Backtest(rw, strat, self.config).run()
            s = sharpe(res.returns, rf=0.0)
            sharpes.append(float(s) if np.isfinite(s) else np.nan)

        arr = np.array([s for s in sharpes if np.isfinite(s)])
        if not len(arr):
            return {"error": "所有试验都失败了"}
        best_idx = int(np.nanargmax(sharpes))
        best = float(arr.max())
        best_combo = combos[best_idx]

        # 用搜索空间内的**最优**夏普（即使它是负的，也要展示"搜索能抬多高"）
        best = float(arr.max())
        worst = float(arr.min())
        mean = float(arr.mean())
        # 搜索带来的"抬升幅度"：最优值与均值的差，这就是纯粹的过拟合偏误
        lift = best - mean

        return {
            "data": "纯随机游走（真 alpha = 0）",
            "n_trials": len(combos),
            "best_in_sample_sharpe": round(best, 3),
            "worst_in_sample_sharpe": round(worst, 3),
            "best_params": best_combo,
            "sharpe_mean": round(mean, 3),
            "sharpe_std": round(float(arr.std(ddof=1)), 3),
            "sharpe_p95": round(float(np.percentile(arr, 95)), 3),
            "search_lift": round(float(lift), 3),
            "deflated_sharpe_of_best": round(
                float(deflated_sharpe(_make_returns_from_sharpe(best, len(rw.dates)), len(combos))), 4
            ),
            "lesson": (
                f"真 alpha 为 0 的数据上，搜索 {len(combos)} 组参数后：夏普均值 {mean:.2f}，"
                f"最差 {worst:.2f}，最优却被抬到 {best:.2f}——搜索凭空制造了 {lift:.2f} 的"
                f"「假 alpha」。这就是多重检验偏误：你试得越多，最好的那个越不可信。"
                f"Deflated Sharpe 会把它打回原形。"
            ),
        }

    # ------------------------------------------------------------ 实验 2
    def deflate(self, returns, n_trials: int) -> dict:
        """给定试验次数，给出多重检验校正后的可信度。"""
        s = sharpe(returns)
        dsr = deflated_sharpe(returns, n_trials)
        psr = probabilistic_sharpe(returns)
        return {
            "sharpe": round(float(s), 3),
            "n_trials": int(n_trials),
            "deflated_sharpe": round(float(dsr), 4),
            "probabilistic_sharpe": round(float(psr), 4),
            "verdict": _dsr_verdict(dsr, n_trials),
        }

    # ------------------------------------------------------------ 实验 3
    def train_test_decay(
        self,
        strategy_factory: Callable[..., object],
        param_grid: dict[str, Sequence],
        split: float = 0.7,
        metric: str = "sharpe",
    ) -> pd.DataFrame:
        """样本内最优参数在样本外的表现落差."""
        from ..optimize.grid import _expand_grid

        md = self.md
        n = len(md.dates)
        cut = int(n * split)
        train_md = md.slice(end=md.dates[cut - 1])
        test_md = md.slice(start=md.dates[cut])

        rows = []
        for combo in _expand_grid(param_grid):
            strat = strategy_factory(**combo)
            r_in = Backtest(train_md, strat, self.config).run()
            r_out = Backtest(test_md, strat, self.config).run()
            rows.append({
                **combo,
                f"train_{metric}": float(_metric(r_in, metric)),
                f"test_{metric}": float(_metric(r_out, metric)),
            })
        df = pd.DataFrame(rows)
        in_col, out_col = f"train_{metric}", f"test_{metric}"
        df["decay"] = df[out_col] - df[in_col]
        df["decay_ratio"] = np.where(
            np.abs(df[in_col]) > 1e-9, df[out_col] / df[in_col], np.nan
        )
        df = df.sort_values(in_col, ascending=False).reset_index(drop=True)
        df["rank_in"] = np.arange(1, len(df) + 1)
        return df

    def overfit_scorecard(self, returns, n_trials: int) -> dict:
        """一张卡的过拟合体检报告."""
        s = sharpe(returns)
        dsr = deflated_sharpe(returns, n_trials)
        psr = probabilistic_sharpe(returns)
        n = len(returns)
        # 最小可检测夏普的跟踪记录长度（Bailey 的 Minimum Track Record Length）
        min_trl = _min_track_record_length(max(s, 1e-6))
        return {
            "sharpe": round(float(s), 3),
            "probabilistic_sharpe": round(float(psr), 4),
            "deflated_sharpe": round(float(dsr), 4),
            "n_trials": int(n_trials),
            "n_observations": int(n),
            "min_track_record_length": int(min_trl),
            "track_record_sufficient": bool(n >= min_trl),
            "verdict": _dsr_verdict(dsr, n_trials),
            "checklist": [
                "参数数量是否远小于样本量？（每加一个参数都要更多数据）",
                "是否用了净化 K 折或滚动前推？",
                "样本外表现是否与样本内同量级？",
                "策略是否有经济学解释，还是纯数据挖掘？",
                "换手率与成本是否在回测中完整计入？",
            ],
        }


def _metric(res, metric: str) -> float:
    m = res.metrics
    key = {"sharpe": "sharpe", "return": "annual_return", "calmar": "calmar", "drawdown": "max_drawdown"}[metric]
    v = m.get(key, np.nan)
    return float(v) if v is not None and np.isfinite(v) else -9.99


def _make_returns_from_sharpe(target_sharpe: float, n: int, seed: int = 0) -> pd.Series:
    """构造一个具有指定夏普的收益序列（仅用于 DSR 演示）。"""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, size=n)
    r = r - r.mean() + target_sharpe * 0.01 / np.sqrt(252)
    return pd.Series(r)


def _min_track_record_length(sharpe_target: float, sharpe_std: float = 0.5) -> float:
    """Minimum Track Record Length（Bailey & López de Prado, 2012）.

    $\\mathrm{MinTRL} = 1 + \\left(1 - \\gamma_3 \\hat{SR} + \\frac{\\gamma_4-1}{4}\\hat{SR}^2\\right)\\left(\\frac{Z_\\alpha}{\\hat{SR}}\\right)^2$
    """
    from scipy import stats as sps

    if sharpe_target <= 1e-9:
        return float("inf")
    z = sps.norm.ppf(0.95)
    return float(1 + (z / sharpe_target) ** 2)


def _dsr_verdict(dsr: float, n_trials: int) -> str:
    if not np.isfinite(dsr):
        return "数据不足，无法判断"
    if dsr > 0.95:
        return f"强：{n_trials} 次试验后仍显著（DSR={dsr:.3f}）"
    if dsr > 0.8:
        return f"中等：{n_trials} 次试验后勉强显著（DSR={dsr:.3f}），建议再压缩参数空间"
    if dsr > 0.5:
        return f"弱：{n_trials} 次试验后不太可信（DSR={dsr:.3f}），高度怀疑过拟合"
    return f"失败：{n_trials} 次试验后不显著（DSR={dsr:.3f}），几乎可以确定为过拟合"
