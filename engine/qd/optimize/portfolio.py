"""组合优化：均值方差、风险平价、层次风险平价、凯利、协方差收缩.

为什么需要这么多方法
--------------------
均值方差优化在教科书里完美，在实践中脆弱：**预期收益的估计误差会被优化器
放大**（Best & Grauer 的著名结论：均值方差的最优权重对预期收益假设极其敏感，
而对协方差敏感得多）。所以实践中更常用的是不依赖预期收益的方法：

* 等权 —— 简单到没有估计误差，经常打赢复杂模型（DeMiguel et al., 2009）；
* 逆波动 / 风险平价 —— 只依赖协方差，稳健；
* HRP —— 用层次聚类先分块再分配，避免矩阵求逆的数值问题；
* 最小方差 / 最大分散化 —— 只用协方差。

本模块把全部方法都实现，并让用户可以**对比它们的样本外表现**——这才是
正确的学习方式：不是背结论，而是亲手看到复杂方法什么时候输给等权。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

__all__ = [
    "expected_returns", "shrink_covariance", "mean_variance", "min_variance",
    "max_diversification", "risk_parity_weights", "inverse_vol_weights",
    "equal_weight_weights", "hrp_weights", "kelly_fraction", "compare_allocators",
]


def expected_returns(returns, method: str = "mean", periods_per_year: float = 252.0) -> pd.Series:
    """预期收益向量.

    method :
        ``"mean"`` 历史均值年化（噪声最大）；
        ``"ewm"`` 指数加权（对新信息更敏感）；
        ``"zeros"`` 全零（用于最小方差/风险平价，避免估计误差）；
        ``"shrink"`` 向横截面均值收缩（James-Stein 式）。
    """
    r = pd.DataFrame(returns).dropna(how="all")
    if method == "zeros":
        return pd.Series(0.0, index=r.columns)
    if method == "mean":
        return r.mean() * periods_per_year
    if method == "ewm":
        return r.ewm(halflife=126).mean().iloc[-1] * periods_per_year
    if method == "shrink":
        mu = r.mean() * periods_per_year
        grand = mu.mean()
        # 收缩强度随横截面离散度与估计误差而定
        sd = mu.std(ddof=1)
        n = len(r)
        lam = np.clip(sd ** 2 / (sd ** 2 + (periods_per_year / max(n, 1)) * (r.std(ddof=1) ** 2).mean()), 0, 1)
        return grand + lam * (mu - grand)
    raise ValueError("method 必须是 mean / ewm / zeros / shrink")


def shrink_covariance(returns, method: str = "ledoit_wolf", periods_per_year: float = 252.0) -> pd.DataFrame:
    """协方差收缩估计.

    method :
        ``"sample"`` 样本协方差（无偏但噪声大）；
        ``"ledoit_wolf"`` 向"常数相关矩阵"收缩（Ledoit & Wolf 2003/2004）；
        ``"identity"`` 向单位阵收缩（最保守）。
    """
    r = pd.DataFrame(returns).dropna()
    cols = r.columns
    S = np.cov(r.to_numpy(dtype=float), rowvar=False, ddof=1) * periods_per_year
    if method == "sample":
        return pd.DataFrame(S, index=cols, columns=cols)
    n = S.shape[0]
    if method == "identity":
        target = np.eye(n) * np.trace(S) / n
    else:
        # 常数相关目标：方差用样本方差，相关系数统一为平均相关系数
        sd = np.sqrt(np.diag(S))
        corr = S / np.outer(sd, sd)
        off = corr[~np.eye(n, dtype=bool)]
        rbar = float(off.mean()) if len(off) else 0.0
        target_corr = np.full((n, n), rbar)
        np.fill_diagonal(target_corr, 1.0)
        target = target_corr * np.outer(sd, sd)

    # Ledoit-Wolf 最优收缩强度（简化版：用样本方差的渐近估计）
    T = len(r)
    diff = S - target
    if method == "ledoit_wolf" and T > n:
        var_S = _asymptotic_var(r.to_numpy(dtype=float))
        delta = float(np.sum(diff ** 2))
        if delta > 0:
            kappa = float(np.clip(var_S / delta, 0.0, 1.0))
        else:
            kappa = 0.0
    else:
        kappa = 0.5
    shrunk = kappa * target + (1 - kappa) * S
    return pd.DataFrame(shrunk, index=cols, columns=cols)


def _asymptotic_var(X: np.ndarray) -> float:
    """样本协方差矩阵总方差的渐近估计（Ledoit-Wolf 的 pi 项简化）。"""
    T, n = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / (T - 1)
    # 逐元素的方差估计
    Y = Xc ** 2
    pi_hat = (Y.T @ Y) / T - S ** 2
    return float(pi_hat.sum())


def _solve_quadratic(Sigma: np.ndarray, mu: np.ndarray | None, risk_aversion: float = 1.0,
                     long_only: bool = True, max_weight: float = 1.0,
                     target_vol: float | None = None):
    """带约束的二次规划（用 scipy 的 SLSQP）。"""
    from scipy.optimize import minimize

    n = len(Sigma)
    x0 = np.full(n, 1.0 / n)
    mu = np.zeros(n) if mu is None else np.asarray(mu, dtype=float)

    def obj(w):
        var = float(w @ Sigma @ w)
        if target_vol is not None:
            return (np.sqrt(max(var, 1e-12)) - target_vol) ** 2 - 1e-3 * float(mu @ w)
        return -float(mu @ w) + 0.5 * risk_aversion * var

    def grad(w):
        if target_vol is not None:
            var = max(float(w @ Sigma @ w), 1e-12)
            return 2 * (np.sqrt(var) - target_vol) * (Sigma @ w) / np.sqrt(var) - 1e-3 * mu
        return -mu + risk_aversion * (Sigma @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
             "jac": lambda w: np.ones(n)}]
    bounds = [(0.0, max_weight)] * n if long_only else [(-max_weight, max_weight)] * n

    res = minimize(obj, x0, jac=grad, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0.0 if long_only else -max_weight, max_weight)
    tot = w.sum()
    if abs(tot) > 1e-9:
        w = w / tot
    return w


def mean_variance(
    returns=None,
    mu=None,
    Sigma=None,
    risk_aversion: float = 1.0,
    long_only: bool = True,
    max_weight: float = 1.0,
    periods_per_year: float = 252.0,
    cov_method: str = "ledoit_wolf",
    mu_method: str = "shrink",
    allow_short: bool | None = None,
) -> pd.Series:
    """均值方差优化：$\\max_w \\; w^\\top\\mu - \\frac{\\lambda}{2} w^\\top\\Sigma w$。

    ``risk_aversion`` 越大越保守（趋向最小方差）。
    ``allow_short=False`` 是 ``long_only=True`` 的等价写法。

    支持两种调用形式：
    * ``mean_variance(returns_df)`` —— 从收益数据估计 $\\mu$ 与 $\\Sigma$；
    * ``mean_variance(mu_vector, Sigma_matrix)`` —— 直接传入已算好的量
      （课程与实验中常用，便于自己控制估计方式）。
    """
    if allow_short is not None:
        long_only = not allow_short

    # ---- 形式二：位置参数写成 mean_variance(mu, Sigma) ----
    # 这里第一个位置参数名是 returns，第二个是 mu；调用方按 (mu, Sigma) 传入时
    # Sigma 落在 mu 上，需要还原过来。
    if Sigma is None and mu is not None and returns is not None:
        arr0 = np.asarray(returns)
        arr1 = np.asarray(mu, dtype=float)
        if arr0.ndim == 1 and arr1.ndim == 2 and arr1.shape[0] == arr1.shape[1]:
            returns, mu, Sigma = None, arr0, arr1

    # ---- 形式二：直接给 mu 与 Sigma ----
    if mu is not None and Sigma is not None:
        mu_arr = np.asarray(mu, dtype=float).ravel()
        S = np.asarray(Sigma, dtype=float)
        if S.shape[0] != S.shape[1]:
            raise ValueError("Sigma 必须是方阵")
        if mu_arr.shape[0] != S.shape[0]:
            raise ValueError("mu 与 Sigma 维度不一致")
        # 简单地在内部补一个默认 index（调用方通常不需要）
        idx = pd.Index([f"a{i}" for i in range(len(mu_arr))])
        w = _solve_quadratic(S, mu_arr, risk_aversion, long_only, max_weight)
        return pd.Series(w, index=idx)

    # ---- 形式一：从收益估计 ----
    r = pd.DataFrame(returns).dropna()
    cols = r.columns
    mu_v = expected_returns(r, mu_method, periods_per_year).reindex(cols).fillna(0.0).to_numpy()
    Sigma_v = shrink_covariance(r, cov_method, periods_per_year).to_numpy()
    w = _solve_quadratic(Sigma_v, mu_v, risk_aversion, long_only, max_weight)
    return pd.Series(w, index=cols)


def min_variance(returns, long_only: bool = True, max_weight: float = 1.0,
                 cov_method: str = "ledoit_wolf", periods_per_year: float = 252.0) -> pd.Series:
    """全局最小方差组合（不依赖预期收益）。"""
    r = pd.DataFrame(returns).dropna()
    cols = r.columns
    Sigma = shrink_covariance(r, cov_method, periods_per_year).to_numpy()
    w = _solve_quadratic(Sigma, None, 1.0, long_only, max_weight)
    return pd.Series(w, index=cols)


def max_diversification(returns, long_only: bool = True, cov_method: str = "ledoit_wolf",
                        periods_per_year: float = 252.0) -> pd.Series:
    """最大分散化：最大化 $\\frac{w^\\top\\sigma}{\\sqrt{w^\\top\\Sigma w}}$。"""
    from scipy.optimize import minimize

    r = pd.DataFrame(returns).dropna()
    cols = r.columns
    Sigma = shrink_covariance(r, cov_method, periods_per_year).to_numpy()
    sd = np.sqrt(np.diag(Sigma))
    n = len(sd)

    def neg_dr(w):
        var = max(float(w @ Sigma @ w), 1e-12)
        return -(float(sd @ w) / np.sqrt(var))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    res = minimize(neg_dr, np.full(n, 1 / n), method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0.0 if long_only else -1.0, 1.0)
    return pd.Series(w / w.sum() if w.sum() else w, index=cols)


def inverse_vol_weights(returns) -> pd.Series:
    """逆波动率加权：$w_i \\propto 1/\\sigma_i$（风险平价的一阶近似）。"""
    r = pd.DataFrame(returns).dropna()
    vol = r.std(ddof=1).replace(0, np.nan)
    inv = 1.0 / vol
    return (inv / inv.sum()).fillna(0.0)


def risk_parity_weights(returns, cov_method: str = "ledoit_wolf",
                        periods_per_year: float = 252.0, max_iter: int = 500) -> pd.Series:
    """等风险贡献（ERC）权重：每个资产对组合风险的贡献相等。

    $\\mathrm{RC}_i = w_i(\\Sigma w)_i / \\sqrt{w^\\top\\Sigma w}$，
    求解 $\\mathrm{RC}_i = \\mathrm{RC}_j,\\ \\forall i,j$。
    用循环坐标下降法求解，数值稳定且天然给出正权重。
    """
    r = pd.DataFrame(returns).dropna()
    cols = r.columns
    Sigma = shrink_covariance(r, cov_method, periods_per_year).to_numpy()
    n = len(Sigma)
    w = np.full(n, 1.0 / n)
    scale = np.sqrt(np.diag(Sigma))
    scale[scale <= 0] = 1e-12
    for _ in range(max_iter):
        Sw = Sigma @ w
        var = max(float(w @ Sw), 1e-12)
        rc = w * Sw
        target = var / n
        grad = (rc - target) / scale          # 按波动缩放步长，收敛更稳
        w = np.clip(w - 0.5 * grad, 1e-8, None)
        w = w / w.sum()
        if np.max(np.abs(rc / rc.sum() - 1.0 / n)) < 1e-12:
            break
    return pd.Series(w / w.sum(), index=cols)


def equal_weight_weights(returns) -> pd.Series:
    """等权：$w_i = 1/N$。最简单的基准，也是最强的对手。"""
    cols = pd.DataFrame(returns).columns
    return pd.Series(1.0 / len(cols), index=cols)


def hrp_weights(returns, cov_method: str = "sample", periods_per_year: float = 252.0) -> pd.Series:
    """层次风险平价（Hierarchical Risk Parity, López de Prado 2016）.

    三步：层次聚类 → 矩阵准对角化 → 递归二分分配权重。
    优点：不需要矩阵求逆，对协方差估计误差远不如均值方差敏感。
    """
    r = pd.DataFrame(returns).dropna()
    cols = list(r.columns)
    n = len(cols)
    if n < 2:
        return pd.Series(1.0, index=cols)
    cov = shrink_covariance(r, cov_method, periods_per_year).to_numpy()
    corr = _cov2corr(cov)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="single")
    # 按聚类顺序重排
    order = _quasi_diag(Z, n)
    sorted_cols = [cols[i] for i in order]
    cov_sorted = cov[np.ix_(order, order)]
    w = _recursive_bisection(cov_sorted)
    out = pd.Series(0.0, index=cols)
    for name, wi in zip(sorted_cols, w):
        out[name] = wi
    return out / out.sum()


def _cov2corr(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(cov))
    sd[sd == 0] = 1e-12
    return cov / np.outer(sd, sd)


def _quasi_diag(Z, n: int) -> list[int]:
    """按层次聚类树恢复准对角顺序。"""
    order = []
    clusters = [i for i in range(n)]
    # 自顶向下遍历树
    def recurse(idx):
        if idx < n:
            order.append(idx)
            return
        left = int(Z[idx - n, 0])
        right = int(Z[idx - n, 1])
        recurse(left)
        recurse(right)

    recurse(len(Z) + n - 1)
    return order


def _recursive_bisection(cov: np.ndarray) -> np.ndarray:
    w = np.ones(cov.shape[0])
    items = [list(range(cov.shape[0]))]
    while items:
        new_items = []
        for group in items:
            if len(group) == 1:
                continue
            half = len(group) // 2
            a, b = group[:half], group[half:]
            va = _cluster_var(cov, a)
            vb = _cluster_var(cov, b)
            alpha = 1 - va / (va + vb) if (va + vb) > 0 else 0.5
            for i in a:
                w[i] *= alpha
            for i in b:
                w[i] *= (1 - alpha)
            new_items.append(a)
            new_items.append(b)
        items = new_items
    return w / w.sum()


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    w = inverse_vol_from_cov(sub)
    return float(w @ sub @ w)


def inverse_vol_from_cov(sub: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(sub))
    sd[sd == 0] = 1e-12
    inv = 1.0 / sd
    return inv / inv.sum()


def kelly_fraction(returns, leverage: float = 1.0, fractional: float = 1.0,
                   periods_per_year: float = 252.0, clip: float = 3.0) -> float:
    """凯利公式的最优仓位.

    连续型凯利：$f^* = \\frac{\\mu}{\\sigma^2}$。这里 $\\mu,\\sigma$ 都按
    ``periods_per_year`` **年化**后再相除，因此结果与调仓频率无关（日频
    收益的 $\\mu/\\sigma^2$ 会比年化版本大 252 倍，那是常见错误）。

    ``fractional`` < 1 即"分数凯利"，实践中几乎总是必要的——因为完整凯利
    对 $\\mu$ 的估计误差极度敏感，稍微高估就会让仓位爆掉。
    """
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 5:
        return float("nan")
    mu = float(r.mean()) * periods_per_year
    var = float(r.var(ddof=1)) * periods_per_year
    if var <= 0:
        return float("nan")
    f = mu / var
    return float(np.clip(f * fractional * leverage, -clip, clip))


def compare_allocators(returns, methods: Sequence[str] | None = None) -> pd.DataFrame:
    """对比各种资产配置方法的样本内特征（教学用途）。

    注意：这只是**权重特征**对比，不是样本外表现对比。要比较真实表现请用
    ``qd.optimize.walk_forward`` 或自己切样本外。
    """
    r = pd.DataFrame(returns).dropna()
    methods = list(methods or ["equal_weight", "inverse_vol", "risk_parity", "hrp",
                               "min_variance", "max_diversification", "mean_variance"])
    Sigma = shrink_covariance(r, "ledoit_wolf").to_numpy()
    mu = expected_returns(r, "mean").to_numpy()
    sd = np.sqrt(np.diag(Sigma))

    rows = []
    for m in methods:
        fn = {
            "equal_weight": equal_weight_weights,
            "inverse_vol": inverse_vol_weights,
            "risk_parity": risk_parity_weights,
            "hrp": hrp_weights,
            "min_variance": min_variance,
            "max_diversification": max_diversification,
            "mean_variance": mean_variance,
        }[m]
        w = fn(r).reindex(r.columns).fillna(0.0).to_numpy()
        var = float(w @ Sigma @ w)
        vol = np.sqrt(max(var, 0))
        ret = float(w @ mu)
        rc = w * (Sigma @ w)
        rc_share = rc / rc.sum() if abs(rc.sum()) > 1e-12 else rc
        rows.append({
            "method": m,
            "expected_annual_return": ret,
            "annual_vol": vol,
            "sharpe": ret / vol if vol > 1e-9 else np.nan,
            "max_weight": float(w.max()),
            "min_weight": float(w.min()),
            "effective_n": float(1.0 / np.sum(w ** 2)),
            "risk_concentration": float(np.std(rc_share) * len(rc_share)),
        })
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
