"""绩效指标全家桶.

约定
----
* ``equity`` 是净值曲线（Series，起点 1.0 或任意正数）；
* 收益序列按 ``equity.pct_change()`` 得出；
* 年化因子由索引频率自动推断（日频 252）。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..utils.misc import annualization_factor, safe_div

__all__ = [
    "to_returns", "annual_return", "annual_vol", "cumulative_return", "sharpe",
    "sortino", "calmar", "max_drawdown", "drawdown_series", "drawdown_table",
    "win_rate", "profit_loss_ratio", "value_at_risk", "conditional_var",
    "ulcer_index", "omega_ratio", "information_ratio", "tracking_error",
    "beta_alpha", "skew_kurt", "deflated_sharpe", "probabilistic_sharpe",
    "roll_sharpe", "turnover_stats", "perf_report", "monthly_returns_table",
]


def to_returns(equity) -> pd.Series:
    """净值 -> 收益率序列。"""
    s = pd.Series(equity).astype(float).dropna()
    if s.iloc[0] <= 0:
        raise ValueError("净值必须为正")
    r = s.pct_change()
    return r.iloc[1:] if r.iloc[0] != r.iloc[0] else r.fillna(0.0)


def cumulative_return(equity) -> float:
    s = pd.Series(equity).astype(float).dropna()
    return float(s.iloc[-1] / s.iloc[0] - 1.0)


def annual_return(equity, periods_per_year: float | None = None) -> float:
    """几何年化收益（CAGR），比算术平均更贴近真实体验。"""
    s = pd.Series(equity).astype(float).dropna()
    n = len(s)
    if n < 2:
        return float("nan")
    start = s.iloc[0]
    if not np.isfinite(start) or start <= 0:
        return float("nan")
    ppy = periods_per_year or annualization_factor(s.index)
    years = (n - 1) / ppy
    if years <= 0:
        return float("nan")
    total = s.iloc[-1] / start
    if not np.isfinite(total):
        return float("nan")
    if total <= 0:
        return -1.0
    return float(total ** (1 / years) - 1.0)


def annual_vol(returns, periods_per_year: float | None = None) -> float:
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return float("nan")
    ppy = periods_per_year or annualization_factor(r.index)
    return float(r.std(ddof=1) * np.sqrt(ppy))


def _as_returns(r: pd.Series) -> pd.Series:
    """把输入统一成收益率序列。

    用户经常把**净值曲线**直接传给 ``sharpe``/``sortino``（``sharpe(nav_ls)``
    是非常自然的写法）。这里做一次启发式识别：收益序列的绝对值通常都在 50%
    以内，而净值会随复利涨到 1.5、3、10 甚至更大。因此只要出现超过 0.6 的
    值（或负到 -0.6 以下），就当成净值，先取 ``pct_change``。
    """
    s = pd.Series(r).astype(float).dropna()
    if len(s) < 3:
        return s
    looks_like_nav = bool((s.abs() > 0.6).any())
    if looks_like_nav:
        return s.pct_change().dropna()
    return s


def sharpe(returns, rf: float = 0.0, periods_per_year: float | None = None) -> float:
    """年化夏普比率.

    $\\mathrm{Sharpe}=\\frac{\\mathbb{E}[r-r_f]\\cdot N}{\\sigma(r)\\sqrt{N}}$.
    ``rf`` 为**年化**无风险利率。传净值曲线也能用（会自动转成收益）。
    """
    r = _as_returns(pd.Series(returns))
    if len(r) < 3:
        return float("nan")
    ppy = periods_per_year or annualization_factor(r.index)
    rf_per = rf / ppy
    excess = r - rf_per
    sd = excess.std(ddof=1)
    # 波动率为 0 时夏普在数学上无定义（会出现 ±inf）；真实数据里这只说明
    # 样本太少或价格被冻结，返回 NaN 比返回一个荒谬的巨大数字诚实。
    if sd <= 1e-12 or not np.isfinite(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(ppy))


def sortino(returns, rf: float = 0.0, periods_per_year: float | None = None) -> float:
    """索提诺比率：只用下行波动做分母，更适合非正态的策略。

    当序列没有任何负收益时（例如单调上涨的净值），下行波动为 0，
    比率在数学上无定义；这里返回 ``inf`` 的替代——一个很大的有限数，
    避免下游比较时出现 NaN 传播。
    """
    r = _as_returns(pd.Series(returns))
    if len(r) < 3:
        return float("nan")
    ppy = periods_per_year or annualization_factor(r.index)
    excess = r - rf / ppy
    downside = np.sqrt((excess.clip(upper=0) ** 2).mean())
    if downside <= 1e-12:
        return float("inf") if excess.mean() > 0 else float("nan")
    return float(excess.mean() / downside * np.sqrt(ppy))


def drawdown_series(equity) -> pd.Series:
    s = pd.Series(equity).astype(float).dropna()
    peak = s.cummax()
    return s / peak - 1.0


def max_drawdown(equity) -> float:
    dd = drawdown_series(equity)
    return float(dd.min()) if len(dd) else float("nan")


def drawdown_table(equity, top: int = 5) -> pd.DataFrame:
    """前 N 大回撤：起止时间、深度、恢复天数。"""
    s = pd.Series(equity).astype(float).dropna()
    dd = s / s.cummax() - 1.0
    out = []
    in_dd = False
    start = None
    trough = None
    for i, (date, val) in enumerate(dd.items()):
        if not in_dd and val < 0:
            in_dd, start, trough = True, date, date
        elif in_dd:
            if val < dd.loc[trough]:
                trough = date
            if val >= 0:
                out.append({
                    "start": start, "trough": trough, "end": date,
                    "depth": float(dd.loc[trough]),
                    "length_days": int((date - start).days),
                    "recovery_days": int((date - trough).days),
                })
                in_dd = False
    if in_dd:
        out.append({
            "start": start, "trough": trough, "end": pd.NaT,
            "depth": float(dd.loc[trough]),
            "length_days": int((dd.index[-1] - start).days),
            "recovery_days": np.nan,
        })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values("depth").head(top).reset_index(drop=True)


def calmar(equity, periods_per_year: float | None = None) -> float:
    """Calmar = 年化收益 / 最大回撤绝对值。"""
    ar = annual_return(equity, periods_per_year)
    mdd = abs(max_drawdown(equity))
    return float(ar / mdd) if mdd > 0 else float("nan")


def win_rate(returns) -> float:
    r = pd.Series(returns).astype(float).dropna()
    if not len(r):
        return float("nan")
    return float((r > 0).sum() / (r != 0).sum()) if (r != 0).any() else float("nan")


def profit_loss_ratio(returns) -> float:
    """盈亏比 = 平均盈利 / 平均亏损。"""
    r = pd.Series(returns).astype(float).dropna()
    win = r[r > 0]
    loss = r[r < 0]
    if not len(win) or not len(loss):
        return float("nan")
    return float(win.mean() / abs(loss.mean()))


def value_at_risk(returns, alpha: float = 0.05, method: str = "historical") -> float:
    """VaR（正数表示损失幅度）。

    method : ``historical`` 历史分位；``gaussian``（别名 ``parametric`` /
    ``normal``）正态假设；``cornish_fisher`` 修正偏度峰度（更贴近厚尾现实）。
    """
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 5:
        return float("nan")
    # 常见别名：参数法在不同教材里有不同叫法
    method = {"parametric": "gaussian", "normal": "gaussian", "analytic": "gaussian"}.get(
        str(method).lower(), str(method).lower()
    )
    if method == "historical":
        return float(-np.percentile(r, alpha * 100))
    mu, sd = r.mean(), r.std(ddof=1)
    if method == "gaussian":
        z = sps.norm.ppf(alpha)
        return float(-(mu + z * sd))
    if method == "cornish_fisher":
        z = sps.norm.ppf(alpha)
        s, k = sps.skew(r), sps.kurtosis(r)
        zc = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
        return float(-(mu + zc * sd))
    raise ValueError("method 必须是 historical / gaussian / cornish_fisher")


def conditional_var(returns, alpha: float = 0.05, method: str = "historical") -> float:
    """CVaR / Expected Shortfall：尾部平均损失。"""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 5:
        return float("nan")
    var = value_at_risk(r, alpha, method)
    tail = r[r <= -var]
    return float(-tail.mean()) if len(tail) else float(var)

def ulcer_index(equity) -> float:
    """Ulcer Index：对回撤深度与持续时间同时惩罚。"""
    dd = drawdown_series(equity) * 100
    return float(np.sqrt((dd ** 2).mean()))


def omega_ratio(returns, threshold: float = 0.0) -> float:
    r = pd.Series(returns).astype(float).dropna()
    up = (r - threshold).clip(lower=0).sum()
    down = (threshold - r).clip(lower=0).sum()
    return float(up / down) if down > 0 else float("nan")


def tracking_error(returns, benchmark) -> float:
    r = pd.Series(returns).astype(float)
    b = pd.Series(benchmark).astype(float).reindex_like(r)
    diff = (r - b).dropna()
    ppy = annualization_factor(diff.index)
    return float(diff.std(ddof=1) * np.sqrt(ppy))


def information_ratio(returns, benchmark) -> float:
    r = pd.Series(returns).astype(float)
    b = pd.Series(benchmark).astype(float).reindex_like(r)
    diff = (r - b).dropna()
    te = diff.std(ddof=1)
    if te == 0:
        return float("nan")
    ppy = annualization_factor(diff.index)
    return float(diff.mean() / te * np.sqrt(ppy))


def beta_alpha(returns, benchmark, rf: float = 0.0) -> tuple[float, float]:
    """返回 ``(beta, 年化 alpha)``。"""
    r = pd.Series(returns).astype(float)
    b = pd.Series(benchmark).astype(float).reindex_like(r)
    df = pd.concat([r, b], axis=1).dropna()
    if len(df) < 5:
        return float("nan"), float("nan")
    x = df.iloc[:, 1].values
    y = df.iloc[:, 0].values
    xv = np.column_stack([np.ones(len(x)), x])
    beta, intercept = np.linalg.lstsq(xv, y, rcond=None)[0]
    ppy = annualization_factor(df.index)
    return float(beta), float(intercept * ppy)


def skew_kurt(returns) -> tuple[float, float]:
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 5:
        return float("nan"), float("nan")
    return float(sps.skew(r)), float(sps.kurtosis(r))


def probabilistic_sharpe(
    returns,
    benchmark_sharpe: float = 0.0,
    rf: float = 0.0,
    n_obs: int | None = None,
    skew: float | None = None,
    kurt: float | None = None,
    sr_benchmark: float | None = None,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012).

    PSR = 观测夏普显著高于 ``benchmark_sharpe`` 的概率（考虑偏度峰度修正）。
    值 > 0.95 才算"统计上可信"。

    Parameters
    ----------
    benchmark_sharpe :
        比较基准（默认 0）。
    n_obs, skew, kurt :
        若已提前算好这些统计量，可直接传入，避免重复计算。
    """
    if sr_benchmark is not None:
        benchmark_sharpe = float(sr_benchmark)
    r = pd.Series(returns).astype(float).dropna()
    n = int(n_obs) if n_obs is not None else len(r)
    if n < 20:
        return float("nan")
    ppy = annualization_factor(r.index)
    sr = sharpe(r, rf, ppy)
    if not np.isfinite(sr):
        return float("nan")
    s, k = _skew_kurt_of(r, skew, kurt)
    denom = np.sqrt(max(1e-12, 1 - s * sr + (k - 1) / 4 * sr ** 2))
    z = (sr - benchmark_sharpe) * np.sqrt(n - 1) / denom
    return float(sps.norm.cdf(z))


def deflated_sharpe(
    returns,
    n_trials: int,
    rf: float = 0.0,
    expected_sr: float = 0.0,
    sharpe_std: float | None = None,
    n_obs: int | None = None,
    skew: float | None = None,
    kurt: float | None = None,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    在做了 ``n_trials`` 次试验之后，观测夏普仍然显著的概率。它把"多重检验"
    的惩罚显式写进公式，是防止参数搜索过拟合最有用的单一数字。

    Parameters
    ----------
    n_trials :
        参数搜索实际尝试过的组合数（**必须诚实填写**，填 1 等于自欺）。
    sharpe_std :
        所有试验夏普的标准差；``None`` 时按经验公式估计。
    n_obs, skew, kurt :
        若已提前算好这些统计量，可直接传入，避免重复计算。
    """
    r = pd.Series(returns).astype(float).dropna()
    n = int(n_obs) if n_obs is not None else len(r)
    if n < 20:
        return float("nan")
    ppy = annualization_factor(r.index)
    sr = sharpe(r, rf, ppy)
    if not np.isfinite(sr):
        return float("nan")
    s, k = _skew_kurt_of(r, skew, kurt)
    if sharpe_std is None:
        # 多次试验的夏普标准差经验估计（Bailey et al.）
        euler = 0.5772156649
        n_t = max(int(n_trials), 2)
        sharpe_std = np.sqrt(max(1e-12, (1 - euler) / n + euler * (n_t - 1) / n_t / n))
        sharpe_std = float(sharpe_std * np.sqrt(ppy))
    sr0 = expected_sr + sharpe_std * (
        (1 - euler_gamma_const()) * sps.norm.ppf(1 - 1.0 / max(n_trials, 2))
        + euler_gamma_const() * sps.norm.ppf(1 - 1.0 / (max(n_trials, 2) * np.e))
    )
    denom = np.sqrt(max(1e-12, 1 - s * sr + (k - 1) / 4 * sr ** 2))
    z = (sr - sr0) * np.sqrt(n - 1) / denom
    return float(sps.norm.cdf(z))


def _skew_kurt_of(r: pd.Series, skew: float | None, kurt: float | None) -> tuple[float, float]:
    """取出偏度/峰度；未显式提供时才从数据计算。"""
    if skew is None or kurt is None:
        s, k = skew_kurt(r)
        skew = s if skew is None else float(skew)
        kurt = k if kurt is None else float(kurt)
    return float(skew), float(kurt)


def euler_gamma_const() -> float:
    return 0.5772156649


def roll_sharpe(returns, window: int = 252, rf: float = 0.0) -> pd.Series:
    """滚动夏普（看 alpha 是否稳定，比单一数字诚实得多）。"""
    r = pd.Series(returns).astype(float)
    ppy = annualization_factor(r.index)
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std(ddof=1)
    return (mu - rf / ppy) / sd.replace(0, np.nan) * np.sqrt(ppy)


def turnover_stats(weights: pd.DataFrame) -> dict:
    """换手率统计（weights 为 dates × symbols 的权重）。"""
    w = pd.DataFrame(weights).fillna(0.0)
    delta = w.diff().abs().sum(axis=1)
    return {
        "avg_daily_turnover": float(delta.mean()),
        "annual_turnover": float(delta.mean() * 252),
        "median_daily_turnover": float(delta.median()),
        "max_daily_turnover": float(delta.max()),
    }


def monthly_returns_table(equity) -> pd.DataFrame:
    """年 × 月 收益透视表（最后一列是每年的全年收益）。"""
    s = pd.Series(equity).astype(float).dropna()
    if len(s) < 2:
        return pd.DataFrame()
    m = s.resample("ME").last()
    mr = m.pct_change().dropna()
    if mr.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"year": mr.index.year, "month": mr.index.month, "ret": mr.values})
    piv = df.pivot(index="year", columns="month", values="ret")
    piv.columns = [f"{c}M" for c in piv.columns]

    # 年度收益：用每年最后一个月末净值相对上一年最后一个月的净值
    yearly = m.groupby(m.index.year).last()
    yr_ret = yearly.pct_change()
    piv["全年"] = yr_ret.reindex(piv.index).to_numpy()
    return piv


class PerfReport(dict):
    """绩效报告：dict 子类，支持属性访问与快速格式化。

    嵌套字段（如 ``turnover``、``top_drawdowns``）保留在字典里，但提供
    :meth:`flat` 只取标量部分——``{k: round(v, 4) for k, v in rep.flat().items()}``
    是很常见的打印写法。
    """

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def flat(self) -> dict:
        """只保留数值标量字段（去掉字符串、列表、嵌套字典）。"""
        out = {}
        for k, v in self.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                out[k] = v
        return out

    def __repr__(self) -> str:
        n = len(self.flat())
        return f"PerfReport(sharpe={self.get('sharpe')}, annual={self.get('annual_return')}, n_fields={n})"


def perf_report(equity, benchmark=None, rf: float = 0.0, weights=None, n_trials: int = 1) -> PerfReport:
    """一次性产出完整绩效报告.

    Parameters
    ----------
    equity :
        净值曲线（Series 或 array）。
    benchmark :
        基准净值曲线，用于算 alpha/beta/IR/超额。
    rf :
        年化无风险利率。
    weights :
        持仓权重面板，给定时计算换手率。
    n_trials :
        该结果经历过多少次参数试验，用于 Deflated Sharpe。
    """
    s = pd.Series(equity).astype(float).dropna()
    r = to_returns(s)
    ppy = annualization_factor(s.index)
    years = (len(s) - 1) / ppy

    rep = {
        "start": str(s.index[0].date()) if isinstance(s.index, pd.DatetimeIndex) else str(s.index[0]),
        "end": str(s.index[-1].date()) if isinstance(s.index, pd.DatetimeIndex) else str(s.index[-1]),
        "n_periods": int(len(r)),
        "years": round(float(years), 2),
        "cumulative_return": round(cumulative_return(s), 6),
        "annual_return": round(annual_return(s, ppy), 6),
        "annual_vol": round(annual_vol(r, ppy), 6),
        "sharpe": round(sharpe(r, rf, ppy), 4),
        "sortino": round(sortino(r, rf, ppy), 4),
        "max_drawdown": round(max_drawdown(s), 6),
        "calmar": round(_n(calmar(s, ppy)), 4),
        "ulcer_index": round(_n(ulcer_index(s)), 4),
        "omega_ratio": round(_n(omega_ratio(r)), 4),
        "win_rate": round(_n(win_rate(r)), 4),
        "profit_loss_ratio": round(_n(profit_loss_ratio(r)), 4),
        "skew": round(_n(skew_kurt(r)[0]), 4),
        "kurtosis": round(_n(skew_kurt(r)[1]), 4),
        "var_95": round(_n(value_at_risk(r, 0.05)), 6),
        "cvar_95": round(_n(conditional_var(r, 0.05)), 6),
        "var_95_cornish_fisher": round(_n(value_at_risk(r, 0.05, "cornish_fisher")), 6),
        "probabilistic_sharpe": round(_n(probabilistic_sharpe(r, 0.0, rf)), 4),
        "deflated_sharpe": round(_n(deflated_sharpe(r, n_trials, rf)), 4),
        "n_trials": int(n_trials),
    }

    if benchmark is not None:
        b = pd.Series(benchmark).astype(float).reindex_like(s).dropna()
        if len(b) > 5:
            br = to_returns(b)
            beta, alpha = beta_alpha(r, br, rf)
            rep.update({
                "benchmark_return": round(cumulative_return(b), 6),
                "benchmark_annual_return": round(annual_return(b, ppy), 6),
                "excess_annual_return": round(annual_return(s, ppy) - annual_return(b, ppy), 6),
                "beta": round(_n(beta), 4),
                "alpha": round(_n(alpha), 6),
                "tracking_error": round(_n(tracking_error(r, br)), 6),
                "information_ratio": round(_n(information_ratio(r, br)), 4),
            })

    if weights is not None:
        rep["turnover"] = {k: round(v, 4) for k, v in turnover_stats(weights).items()}

    rep["top_drawdowns"] = (
        drawdown_table(s, top=3).to_dict("records") if len(s) > 30 else []
    )
    return PerfReport(rep)


def _n(x) -> float:
    return float(x) if x is not None and np.isfinite(x) else float("nan")
