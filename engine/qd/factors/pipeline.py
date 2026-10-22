"""因子处理流水线：去极值、标准化、中性化、排名、合成、正交化.

顺序约定（重要）
----------------
标准流程是：**去极值 → 中性化 → 标准化**，而不是反过来。
先中性化再去极值，会让回归残差里的极端值重新污染回归；先标准化再中性化，
中性化后量纲又变了。本模块的每个函数都是纯函数（不改输入），可以任意组合，
但 ``combine`` 与 UI 的默认流程严格遵循上述顺序。
"""
from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "winsorize", "zscore", "standardize", "neutralize", "rank", "combine",
    "orthogonalize", "demean", "fill_missing", "to_weights", "factor_returns",
]


def _ensure_df(x) -> pd.DataFrame:
    """转成 DataFrame 副本。

    * ``Series`` 输入被视为**单个横截面**（一行），索引即标的，``name`` 即日期；
    * 普通索引不会被强制转成日期——因子面板在真实使用中索引是日期，但单元测试
      与 ``combine`` 之类的组合运算也可能传入任意索引。
    """
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, pd.Series):
        return x.to_frame().T
    return pd.DataFrame(x).copy()


def fill_missing(x, method: str = "cross_sectional_mean", value: float = 0.0) -> pd.DataFrame:
    """缺失值填充.

    method :
        ``"cross_sectional_mean"`` 用当日横截面均值（推荐：不引入时序信息）；
        ``"value"`` 用常数；
        ``"ffill"`` 前向填充（注意可能引入停牌股僵尸值）。
    """
    df = _ensure_df(x)
    if method == "cross_sectional_mean":
        return df.where(df.notna(), df.mean(axis=1), axis=0)
    if method == "value":
        return df.fillna(value)
    if method == "ffill":
        return df.ffill()
    raise ValueError("method 必须是 cross_sectional_mean / value / ffill")


def winsorize(
    x,
    method: str = "mad",
    n: float = 5.0,
    lower: float | None = None,
    upper: float | None = None,
    limits: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """横截面去极值（缩尾）。

    Parameters
    ----------
    method :
        ``"mad"`` 中位数 ± n×1.4826×MAD（对厚尾最稳健，推荐）；
        ``"std"`` 均值 ± n×标准差；
        ``"quantile"`` 分位数缩尾，此时 ``n`` 为小数比例（如 0.01 表示 1%/99%）。
    limits :
        分位数边界。``0.01`` 表示双侧各截 1%；``(0.01, 0.99)`` 表示显式上下界。
        这是 `quantile` 方法最常用的写法。
    """
    df = _ensure_df(x)

    if limits is not None:
        # 支持 limits=0.01（双侧 1%）与 limits=(0.01, 0.99)（显式上下界）
        if isinstance(limits, (int, float)):
            lo_q, hi_q = float(limits), 1.0 - float(limits)
        else:
            lo_q, hi_q = float(limits[0]), float(limits[1])
        lo = df.quantile(lo_q, axis=1)
        hi = df.quantile(hi_q, axis=1)
        return df.clip(lower=lo, upper=hi, axis=0)

    if method == "quantile":
        lo = df.quantile(n, axis=1)
        hi = df.quantile(1 - n, axis=1)
        return df.clip(lower=lo, upper=hi, axis=0)

    med = df.median(axis=1)
    if method == "mad":
        mad = (df.sub(med, axis=0)).abs().median(axis=1) * 1.4826
        lo, hi = med - n * mad, med + n * mad
    elif method == "std":
        sd = df.std(axis=1)
        lo, hi = med - n * sd, med + n * sd
    else:
        raise ValueError("method 必须是 mad / std / quantile")
    if lower is not None:
        lo = lo.clip(lower=lower)
    if upper is not None:
        hi = hi.clip(upper=upper)
    return df.clip(lower=lo, upper=hi, axis=0)


def demean(x) -> pd.DataFrame:
    """横截面去均值（多空组合的隐含操作）。"""
    df = _ensure_df(x)
    return df.sub(df.mean(axis=1), axis=0)


def zscore(x, clip: float | None = 3.0) -> pd.DataFrame:
    """横截面 z-score 标准化：$\\tilde{x}=(x-\\mu_t)/\\sigma_t$。"""
    df = _ensure_df(x)
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    out = df.sub(mu, axis=0).div(sd, axis=0)
    if clip is not None:
        out = out.clip(-clip, clip)
    return out


def standardize(x, clip: float | None = 3.0) -> pd.DataFrame:
    """``zscore`` 的别名（先去极值再标准化才是完整流程）。"""
    return zscore(x, clip=clip)


def rank(x, pct: bool = True, ascending: bool = True) -> pd.DataFrame:
    """横截面排名。``pct=True`` 时映射到 [0, 1]，天然抗极值。"""
    df = _ensure_df(x)
    return df.rank(axis=1, pct=pct, ascending=ascending)


# 行业哑变量缓存：(id(arr), shape) -> (arr_ref, [dummy_arrays])
_DUMMY_CACHE: dict = {}


def neutralize(
    x,
    md=None,
    exposures: Sequence | None = None,
    industry: bool = True,
    size: bool = True,
    by=None,
    group=None,
) -> pd.DataFrame:
    """横截面中性化：对暴露变量做 OLS，取残差。

    参数命名兼容几种常见写法：

    * ``neutralize(x, md)`` —— 用 ``md`` 里的 mktcap + industry 中性化；
    * ``neutralize(x, md, exposures=["mktcap", "industry"])`` —— 自定义暴露字段；
    * ``neutralize(x, md, by=lncap_panel)`` —— 直接给连续暴露面板（如对数市值）；
    * ``neutralize(x, md, by=lncap, group=industries)`` —— 连续暴露 + 分组标签
      （``group`` 可以是 DataFrame 或 Series，列名/索引与因子面板对齐）。

    实现说明
    --------
    对每个交易日独立做一次**横截面** OLS（N 只股票是观测，K 个暴露是特征）。
    全部日期的回归用矩阵运算一次解出，因此 1800 天 × 50 标的只需几十毫秒。
    注意 beta 的形状是 (T, K) 而不是 (N, K)——这是最容易写反的地方。
    """
    df = _ensure_df(x)

    # ---- 归一化暴露来源 ----
    cont_blocks: list[np.ndarray] = []
    group_series = None

    if md is not None:
        if exposures is None:
            exposures = []
            if size:
                exposures.append("mktcap")
            if industry:
                exposures.append("industry")
        ex = exposures
        ln_cap = np.log(md.fundamentals["mktcap"]) if "mktcap" in md else None
        ind = md.fundamentals.get("industry") if "industry" in md else None

        if any(e in ("mktcap", "size") for e in ex) and ln_cap is not None:
            cont_blocks.append(ln_cap.reindex(index=df.index, columns=df.columns).to_numpy(dtype=float))
        if "industry" in ex and ind is not None:
            group_series = ind.reindex(index=df.index, columns=df.columns)
        for e in ex:
            if e in ("mktcap", "size", "industry"):
                continue
            if e in md:
                cont_blocks.append(
                    md.field(e).reindex(index=df.index, columns=df.columns).to_numpy(dtype=float)
                )

    # ---- by= 的两种含义 ----
    #   by=["mktcap","industry"]  -> 字段名列表（等价于 exposures）
    #   by=<DataFrame/Series>      -> 直接传入的暴露面板
    by_is_names = (
        isinstance(by, (list, tuple)) and len(by) > 0 and all(isinstance(v, str) for v in by)
    )
    if by_is_names and md is None:
        raise ValueError(
            f"neutralize(x, by={by!r}) 需要传入 md 才能取得这些字段："
            f"neutralize(x, md, by={by!r})"
        )
    if by_is_names and md is not None:
        exposures = list(by)
        ln_cap_b = np.log(md.fundamentals["mktcap"]) if "mktcap" in md else None
        ind_b = md.fundamentals.get("industry") if "industry" in md else None
        if any(e in ("mktcap", "size") for e in exposures) and ln_cap_b is not None:
            cont_blocks.append(ln_cap_b.reindex(index=df.index, columns=df.columns).to_numpy(dtype=float))
        if "industry" in exposures and ind_b is not None:
            group_series = ind_b.reindex(index=df.index, columns=df.columns)
        for e in exposures:
            if e in ("mktcap", "size", "industry"):
                continue
            if e in md:
                cont_blocks.append(md.field(e).reindex(index=df.index, columns=df.columns).to_numpy(dtype=float))
        by = None  # 已处理

    if by is not None:
        if isinstance(by, pd.DataFrame):
            cont_blocks.append(by.reindex(index=df.index, columns=df.columns).to_numpy(dtype=float))
        elif isinstance(by, pd.Series):
            arr = np.tile(by.reindex(df.columns).to_numpy(dtype=float), (len(df), 1))
            cont_blocks.append(arr)
        else:
            cont_blocks.append(np.asarray(by, dtype=float))

    if group is not None:
        g = group
        if isinstance(g, pd.Series):
            g = pd.DataFrame({c: g for c in df.columns}, index=df.index)
        elif isinstance(g, pd.DataFrame):
            g = g.reindex(index=df.index, columns=df.columns)
        else:
            g = pd.DataFrame(np.asarray(g), index=df.index, columns=df.columns)
        group_series = g

    if not cont_blocks and group_series is None:
        return df

    # ---- 预构造设计矩阵，形状 (T, N, K)：每个标的 K 个暴露 ----
    cols = list(df.columns)
    if group_series is not None:
        arr = group_series.to_numpy(dtype=object)
        # 行业哑变量按"行业标签"缓存：同一份数据反复中性化 48 个因子时，
        # 只有第一次需要真正构造 dummies。
        dummy_key = (id(arr), arr.shape)
        cached = _DUMMY_CACHE.get(dummy_key)
        if cached is None or cached[0] is not arr:
            levels = sorted({v for v in arr.ravel() if isinstance(v, str)})
            dummies = [(arr == lev).astype(float) for lev in levels[1:]]  # drop_first
            cached = (arr, dummies)
            _DUMMY_CACHE.clear()
            _DUMMY_CACHE[dummy_key] = cached
        blocks = list(cont_blocks) + list(cached[1])
    else:
        blocks = list(cont_blocks)

    if not blocks:
        return df

    X = np.stack(blocks, axis=2) if len(blocks) > 1 else blocks[0][:, :, None]
    Y = df.to_numpy(dtype=float)
    valid = np.isfinite(Y) & np.isfinite(X).all(axis=2)

    # 加截距项：X_aug = [1, X]
    ones = np.ones((*X.shape[:2], 1))
    Xa = np.concatenate([ones, X], axis=2)                      # (T, N, K+1)
    Xa = np.where(valid[:, :, None], Xa, 0.0)

    Yv = np.where(valid, Y, 0.0)
    XtX = np.einsum("tnk,tnj->tkj", Xa, Xa)          # (T, K, K)
    Xty = np.einsum("tnk,tn->tk", Xa, Yv)            # (T, K)
    K = Xa.shape[2]
    ridge = np.eye(K) * 1e-10
    # 用 Cholesky 批量求解（比通用 solve 快 2~3 倍，数值上等价）：
    #   XtX β = Xty  =>  L Lᵀ β = Xty  =>  先解 L z = Xty，再解 Lᵀ β = z
    try:
        L = np.linalg.cholesky(XtX + ridge[None])
        beta = np.linalg.solve(np.swapaxes(L, 1, 2), np.linalg.solve(L, Xty[:, :, None]))[:, :, 0]
    except np.linalg.LinAlgError:
        # 极端情况下（截面样本过少导致矩阵奇异）退回通用最小二乘
        beta = np.linalg.lstsq(XtX + ridge[None], Xty[:, :, None], rcond=None)[0][:, :, 0]
    resid = np.where(valid, Yv - np.einsum("tnk,tk->tn", Xa, beta), np.nan)
    return pd.DataFrame(resid, index=df.index, columns=df.columns, copy=False)


def orthogonalize(x, against=None, md=None, method: str = "regression") -> pd.DataFrame:
    """把因子 ``x`` 对一组因子 ``against`` 正交化（施密特正交的回归实现）。

    两种调用方式：

    1. **面板形式**（纯函数，最常用）::

           orth = orthogonalize(ep_panel, [bp_panel])

    2. **因子名列表形式**：按顺序做施密特正交，并**自动注册**
       ``<name>_orth`` 因子，方便直接放进 ``lab.ic("ep_orth")`` 之类的分析::

           orthogonalize(["bp", "ep"], md=md)
           lab.ic("ep_orth")

       列表第一项保留原貌，后续每一项都剔除了前面所有因子的线性解释部分。
    """
    # ---- 形式 2：因子名列表 ----
    if isinstance(x, (list, tuple)) and all(isinstance(v, str) for v in x):
        names = list(x)
        if md is None:
            raise TypeError(
                "orthogonalize(['bp', 'ep']) 需要传入 md："
                "orthogonalize(['bp', 'ep'], md=md)"
            )
        from .base import register, FACTOR_REGISTRY, get_factor
        from .base import FactorSpec

        completed: list[pd.DataFrame] = []
        results: dict[str, pd.DataFrame] = {}
        for k, name in enumerate(names):
            base = get_factor(name)
            raw = base.direction * base.compute(md)
            cur = raw if (k == 0 or not completed) else orthogonalize(raw, completed)
            completed.append(cur)

            if k > 0:
                orth_name = f"{base.name}_orth"
                # 把结果直接注册成一个正常因子：compute 时返回缓存的面板
                def _make(_panel=cur, _n=orth_name):
                    def _fn(_md, **_kw):
                        return _panel.reindex(index=_md.dates, columns=_md.symbols)
                    _fn.__name__ = _n
                    return _fn

                spec = FactorSpec(
                    name=orth_name, category=base.category, direction=1,
                    description=f"{base.description}（已对 {', '.join(names[:k])} 正交化）",
                    formula=base.formula,
                    required_fields=tuple(base.spec.required_fields),
                    params={"_orthogonalized_against": tuple(names[:k])},
                )
                register(spec, _make())
                results[orth_name] = cur
        return completed[-1] if completed else pd.DataFrame()

    # ---- 形式 1：面板形式 ----
    df = _ensure_df(x)
    if against is None:
        raise TypeError(
            "orthogonalize 需要正交基准：orthogonalize(x, [other_panel]) "
            "或 orthogonalize(['bp','ep'], md=md)"
        )
    if isinstance(against, (pd.DataFrame, pd.Series)):
        against = [against]
    others = [_ensure_df(a) for a in against]
    out = df.copy()
    for date in df.index:
        y = df.loc[date]
        Z = pd.concat([o.loc[date] for o in others], axis=1)
        mask = y.notna() & Z.notna().all(axis=1)
        if mask.sum() < max(10, Z.shape[1] + 2):
            continue
        Zv = np.column_stack([np.ones(mask.sum()), Z.loc[mask].values])
        yv = y.loc[mask].values.astype(float)
        try:
            beta, *_ = np.linalg.lstsq(Zv, yv, rcond=None)
            out.loc[date, y.loc[mask].index] = yv - Zv @ beta
        except np.linalg.LinAlgError:
            continue
    return out


def combine(
    factors: Mapping[str, pd.DataFrame] | list[pd.DataFrame],
    weights: Mapping[str, float] | Sequence[float] | None = None,
    method: str = "zscore",
) -> pd.DataFrame:
    """把多个因子合成为一个综合得分.

    Parameters
    ----------
    factors :
        因子面板字典或列表。
    weights :
        权重；``None`` 表示等权。
    method :
        ``"zscore"`` 各自标准化后等权/加权（推荐）；
        ``"mean"`` / ``"equal"`` 别名，与 ``"zscore"`` 等价；
        ``"ic"`` IC 加权合成（``weights`` 应传入各因子的 IC，负 IC 会被自动取负号）；
        ``"rank"`` 各自横截面排名后加权；
        ``"raw"`` 直接加权（量纲不同时不要用）。

    实现说明
    --------
    全部走 numpy 的横截面运算，不逐日构造 pandas 对象——回测里每个调仓日都
    会调用它，pandas 的那点开销会被放大一千倍。
    """
    if isinstance(factors, Mapping):
        names = list(factors)
        panels = [factors[n] for n in names]
    else:
        panels = list(factors)
        names = [f"f{i}" for i in range(len(panels))]

    if weights is None:
        w = [1.0 / len(panels)] * len(panels)
    elif isinstance(weights, Mapping):
        w = [float(weights.get(n, 0.0)) for n in names]
    else:
        w = [float(v) for v in weights]
    if len(w) != len(panels):
        raise ValueError("权重数量与因子数量不一致")
    total = sum(abs(v) for v in w)
    if total == 0:
        raise ValueError("权重全为 0")
    w = np.array([v / total for v in w], dtype=float)

    if method not in ("zscore", "mean", "equal", "ic", "rank", "raw"):
        raise ValueError("method 必须是 zscore / mean / equal / ic / rank / raw")

    # 参考网格：优先用第一个 DataFrame 面板；若全是 Series（单截面），
    # 就把它们拼成一个单行网格。
    ref_df = next((p for p in panels if isinstance(p, pd.DataFrame)), None)
    if ref_df is None:
        ref_df = pd.DataFrame([pd.Series(p) for p in panels])
        panels = [pd.Series(p) for p in panels]
    ref = ref_df
    aligned = [np.asarray(_align_to(p, ref), dtype=float) for p in panels]
    if method == "rank":
        # 排名：pandas 的 rank 是 NaN 安全的；结果归一化到 [0,1] 便于合成
        acc = None
        for arr, wi in zip(aligned, w):
            v = pd.DataFrame(arr).rank(axis=1, pct=True).to_numpy(dtype=float)
            acc = (v * wi) if acc is None else acc + v * wi
        return pd.DataFrame(acc, index=ref.index, columns=ref.columns)

    if method == "raw":
        acc = None
        for arr, wi in zip(aligned, w):
            acc = (arr * wi) if acc is None else acc + arr * wi
        return pd.DataFrame(acc, index=ref.index, columns=ref.columns)

    # z-score：逐行去均值除标准差
    acc = np.zeros_like(aligned[0], dtype=float)
    covered = np.zeros_like(aligned[0], dtype=bool)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        # 因子预热期整行为 NaN（np.nanmean 会对空切片告警），这里静音即可
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for arr, wi in zip(aligned, w):
            mu = np.nanmean(arr, axis=1, keepdims=True)
            sd = np.nanstd(arr, axis=1, ddof=1, keepdims=True)
            sd = np.where((sd == 0) | ~np.isfinite(sd), np.nan, sd)
            z = (arr - mu) / sd
            z = np.where(np.isfinite(z), z, np.nan)
            ok = np.isfinite(z)
            acc += np.where(ok, z * wi, 0.0)
            covered |= ok
    acc = np.where(covered, acc, np.nan)
    return pd.DataFrame(acc, index=ref.index, columns=ref.columns)


def _align_to(panel, ref):
    """把面板对齐到参考面板的行列，返回 numpy 数组。

    Series 视为**单个横截面**：如果 ref 是单行网格，把它当一行；如果 ref 是多行
    网格且行数相同，也按行对齐。列始终按 ref 的列对齐。
    """
    if isinstance(panel, pd.DataFrame):
        p = panel
    elif isinstance(panel, pd.Series):
        p = panel.to_frame().T
        if len(ref) > 1 and len(p) == 1:
            # 单截面放进多行网格：广播到每一行（调用方通常不会这样混用）
            p = pd.DataFrame(np.repeat(p.to_numpy(), len(ref), axis=0),
                             index=ref.index, columns=p.columns)
    else:
        p = pd.DataFrame(panel)
    if not p.index.equals(ref.index) or not p.columns.equals(ref.columns):
        p = p.reindex(index=ref.index, columns=ref.columns)
    return p.to_numpy(dtype=float)


def to_weights(score: pd.DataFrame, long_top: float | None = None, short_bottom: float | None = None, gross: float = 1.0) -> pd.DataFrame:
    """把因子得分转成组合权重（美元中性可选）。

    默认做**横截面去均值后的多头排序权重**：先排序，取前 ``long_top`` 比例
    做多，若给了 ``short_bottom`` 则同时做空后 ``short_bottom`` 比例，总敞口
    归一到 ``gross``。
    """
    s = _ensure_df(score)
    if long_top is None and short_bottom is None:
        w = demean(s)
        w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0) * gross
        return w.fillna(0.0)

    lt = long_top if long_top is not None else 0.2
    sb = short_bottom if short_bottom is not None else 0.0
    n_sym = s.shape[1]
    n_long = max(1, int(round(lt * n_sym)))
    n_short = max(0, int(round(sb * n_sym)))
    long_w = gross * (0.5 if n_short else 1.0) / n_long
    short_w = -gross * 0.5 / n_short if n_short else 0.0

    # 向量化：把名次映射成权重矩阵，避免逐日 for 循环
    # 降序名次（1 = 最大），NaN 保留为 NaN
    rk = s.rank(axis=1, ascending=False, method="first")
    W = np.where(rk <= n_long, long_w, 0.0)
    if n_short:
        W = np.where(rk > (s.notna().sum(axis=1).to_numpy()[:, None] - n_short), short_w, W)
    out = pd.DataFrame(W, index=s.index, columns=s.columns)
    out = out.where(s.notna(), 0.0)
    return out


def factor_returns(weights: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    """由持仓权重与前瞻收益计算因子日收益序列（组合收益）。

    约定：``weights`` 是 t 日收盘形成的权重，``fwd_returns`` 是 t→t+1 的收益，
    因此这里用 ``weights.shift(1) * returns``，即**明天赚的是今天选出来的**。
    """
    w = _ensure_df(weights).shift(1)
    r = _ensure_df(fwd_returns).reindex_like(w).fillna(0.0)
    return (w * r).sum(axis=1)
