"""内置策略工厂.

每个函数返回一个 :class:`Strategy` 实例（而不是类），这样用户可以
``sma_cross(fast=10, slow=30)`` 直接拿到可调参数的策略。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..factors.pipeline import winsorize, zscore, rank, combine
from .base import Strategy, weight_from_score, long_top_weights

__all__ = [
    "sma_cross", "dual_momentum", "factor_long_only", "pairs_trading", "vol_target",
    "bollinger_reversion", "turtle_breakout", "risk_parity", "rotation", "buy_and_hold",
    "mean_reversion_basket", "multi_factor_rank", "equal_weight", "BUILTIN_STRATEGIES",
]


# ------------------------------------------------------------------ 趋势
def sma_cross(fast: int = 20, slow: int = 60, weight: float = 1.0) -> Strategy:
    """双均线择时：快线上穿慢线满仓，下穿空仓（对全组合统一开关）。

    说明：这是一个**择时**策略，对每只标的独立判断，符合快线>慢线则持有。
    """
    class _S(Strategy):
        name = f"sma_cross({fast},{slow})"
        description = f"{fast}/{slow} 日均线金叉买入、死叉卖出"

        def weights(self, ctx) -> pd.Series:
            i = ctx.i
            if i + 1 < slow + 2:
                return None
            # 快线：用滚动均线产品缓存的整块面板，O(1) 取第 i 行，
            # 而不是每天把最后 fast 行切片再求均值（那是 O(fast) 的 pandas 开销）。
            f = _rolling_mean_row(ctx, fast, i)
            s = _rolling_mean_row(ctx, slow, i)
            hold = f > s
            n = int(hold.sum())
            out = np.zeros(len(ctx.symbols), dtype=float)
            if n == 0:
                return out
            out[hold] = weight / n
            return out

    return _S()


def _returns_array(md) -> np.ndarray:
    """日收益率面板的 numpy 视图（缓存）。"""
    cache = getattr(md, "_ret_arr", None)
    if cache is None:
        cache = md.close().pct_change().to_numpy(dtype=float)
        try:
            md._ret_arr = cache  # noqa: SLF001
        except AttributeError:  # pragma: no cover
            pass
    return cache


def _close_array(md) -> np.ndarray:
    """收盘价面板的 numpy 视图（缓存，避免每天取 DataFrame）。"""
    cache = getattr(md, "_close_arr", None)
    if cache is None:
        cache = md.close().to_numpy(dtype=float)
        try:
            md._close_arr = cache  # noqa: SLF001
        except AttributeError:  # pragma: no cover
            pass
    return cache


def _rolling_mean_row(ctx, window: int, i: int) -> np.ndarray:
    """取第 i 日的滚动均值横截面（带缓存）。

    与"每天切片求均值"等价，但在整块面板上 rolling 一次，之后 O(1) 取行。
    """
    md = ctx.md
    cache = getattr(md, "_rolling_cache", None)
    if cache is None:
        cache = {}
        try:
            md._rolling_cache = cache  # noqa: SLF001
        except AttributeError:  # pragma: no cover
            px = ctx.close().iloc[: i + 1]
            return px.iloc[-window:].mean().to_numpy()
    key = ("ma", window, id(md.panels["close"]))
    arr = cache.get(key)
    if arr is None:
        arr = md.close().rolling(window, min_periods=window).mean().to_numpy(dtype=float)
        cache[key] = arr
    return arr[i]


def _norm_kwargs(kw: dict, context: str) -> dict:
    """把常见同义参数名映射为标准名；无法识别时报出清晰的错误。

    策略工厂的参数命名在不同教材/框架里差异很大（``entry`` vs
    ``entry_window``、``factor_name`` vs ``factors``、``lookback`` vs
    ``window``……）。与其让用户逐个踩坑，不如在这里统一收口。
    """
    aliases = {
        "entry": "entry_window", "entry_n": "entry_window",
        "exit": "exit_window", "exit_n": "exit_window",
        "factor_name": "factors", "factor": "factors", "names": "factors",
        "top_quantile": "top_pct", "quantile": "top_pct", "pct": "top_pct",        "lookback": "window", "n": "window", "period": "window",
        "up_z": "entry_z", "threshold": "entry_z",
        "close_z": "exit_z", "take_profit_z": "exit_z",
        "k": "top_k", "top": "top_k", "n_stocks": "top_k",
        "fast_ma": "fast", "slow_ma": "slow",
        "lb": "lookback_", "periods": "lookback_",
        "nstd": "n_std", "num_std": "n_std", "std": "n_std",
        "target": "target_vol", "vol_target": "target_vol",
    }
    out = {}
    for key, val in kw.items():
        std = aliases.get(key, key)
        # lookback 在部分策略里是合法标准名，交给签名自己决定
        if std == "lookback_":
            std = "lookback"
        out[std] = val
    return out


def _accepts(fn_name: str, **kw):
    """按签名过滤参数，返回 (可接受参数, 被丢弃的键)。"""
    import inspect

    fn = globals()[fn_name]
    params = inspect.signature(fn).parameters
    has_varkw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    if has_varkw:
        return kw, []
    ok = {k: v for k, v in kw.items() if k in params}
    dropped = [k for k in kw if k not in params]
    return ok, dropped


def turtle_breakout(entry_window: int = 55, exit_window: int = 20, n: int = 10, **kwargs) -> Strategy:
    """海龟突破：价格创 N 日新高买入，跌破 M 日新低卖出。

    参数别名：``entry`` → ``entry_window``，``exit`` → ``exit_window``。
    """
    if kwargs:
        kw = _norm_kwargs(kwargs, "turtle_breakout")
        entry_window = kw.pop("entry_window", entry_window)
        exit_window = kw.pop("exit_window", exit_window)
        n = kw.pop("n", n)
    class _S(Strategy):
        name = f"turtle({entry_window},{exit_window})"
        description = "海龟唐奇安通道突破"

        def weights(self, ctx) -> pd.Series:
            i = ctx.i
            if i + 1 < entry_window + 2:
                return None
            arr = _close_array(ctx.md)
            last = arr[i]
            win = arr[i - entry_window : i]          # 不含当日，避免自我比较
            with np.errstate(invalid="ignore", divide="ignore"):
                hi = np.nanmax(win, axis=0)
                # 只买创出 N 日新高的标的，按突破幅度排序取前 n 只
                score = np.where(last > hi, last / hi, np.nan)
            return long_top_weights(score, n)

    return _S()


# ------------------------------------------------------------------ 动量
def dual_momentum(lookback: int = 120, top_k: int = 5, skip_recent: int = 0) -> Strategy:
    """截面动量：选过去 lookback 日收益最高的 top_k 只持有。"""
    class _S(Strategy):
        name = f"dual_momentum({lookback},k={top_k})"
        description = "横截面动量选股"

        def weights(self, ctx) -> pd.Series:
            i = ctx.i
            if i + 1 < lookback + 2:
                return None
            arr = _close_array(ctx.md)
            end = i - int(skip_recent)
            start = end - lookback
            with np.errstate(invalid="ignore", divide="ignore"):
                mom = arr[end] / arr[start] - 1.0
            return long_top_weights(mom, top_k)

    return _S()


def factor_long_only(factors=("mom_12_1", "bp"), top_k: int = 20,
                     weights: dict | None = None, rebalance: str = "daily",
                     winsor: bool = True, neutral: bool = False,
                     top_pct: float | None = None, **kwargs) -> Strategy:
    """多因子选股（多头）：因子标准化加权后取前 k。

    参数别名：``factor_name``/``factor`` → ``factors``；``top_quantile``/
    ``quantile`` → ``top_pct``（按比例选股，会换算成 top_k）。
    """
    if kwargs:
        kw = _norm_kwargs(kwargs, "factor_long_only")
        factors = kw.pop("factors", factors)
        top_k = kw.pop("top_k", top_k)
        top_pct = kw.pop("top_pct", top_pct)
        weights = kw.pop("weights", weights)
        rebalance = kw.pop("rebalance", rebalance)
        winsor = kw.pop("winsor", winsor)
        neutral = kw.pop("neutral", neutral)
    if isinstance(factors, str):
        factors = [factors]
    from ..factors import get_factor

    resolved = [get_factor(f) if isinstance(f, str) else f for f in factors]
    wmap = weights or {f.name: 1.0 for f in resolved}

    class _S(Strategy):
        name = f"factor_long_only({','.join(f.name for f in resolved)})"
        description = "多因子横截面选股，等权/自定义权重持有前 top_k"

        def weights(self, ctx) -> pd.Series:
            # 因子在 t 日的值只依赖 t 日及之前的数据（有专门测试验证），
            # 因此在整个面板上算一次、按日取行，与"每天只用历史重算"等价，
            # 但快两个数量级。
            k_default = top_k
            k = k_default
            valid_est = 0
            with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
                warnings.simplefilter("ignore", category=RuntimeWarning)
                total = np.zeros(len(ctx.symbols), dtype=float)
                seen = np.zeros(len(ctx.symbols), dtype=bool)
                for f in resolved:
                    v = np.asarray(_factor_panel_at(ctx.md, f, ctx.i), dtype=float)
                    if winsor:
                        v = _winsor_np(v)
                    mu = np.nanmean(v)
                    sd = np.nanstd(v, ddof=1)
                    if not np.isfinite(sd) or sd == 0:
                        continue
                    z = (v - mu) / sd
                    ok = np.isfinite(z)
                    total += np.where(ok, z * wmap.get(f.name, 1.0), 0.0)
                    seen |= ok
            score = np.where(seen, total, np.nan)
            if top_pct:
                n_valid = int(np.isfinite(score).sum())
                k = max(1, int(round(n_valid * float(top_pct))))
            return long_top_weights(score, k)

    return _S()


def multi_factor_rank(factors=("mom_12_1", "ep"), weights: dict | None = None,
                      top_k: int = 20) -> Strategy:
    """排名法多因子：先各自排名再加权，抗极值更强。"""
    from ..factors import get_factor

    resolved = [get_factor(f) if isinstance(f, str) else f for f in factors]
    wmap = weights or {f.name: 1.0 for f in resolved}

    class _S(Strategy):
        name = f"multi_factor_rank({','.join(f.name for f in resolved)})"
        description = "排名合成多因子选股"

        def weights(self, ctx) -> pd.Series:
            # 精确复刻 combine(method="rank") 的语义：
            #   每个因子先做横截面排名（pct，名次/有效数），再 z-score，再加权求和。
            # 用 numpy 实现同样的计算，省掉每天构造 DataFrame 的开销。
            n = len(ctx.symbols)
            total = np.zeros(n, dtype=float)
            seen = np.zeros(n, dtype=bool)
            wsum = sum(abs(wmap.get(f.name, 1.0)) for f in resolved) or 1.0
            for f in resolved:
                v = np.asarray(_factor_panel_at(ctx.md, f, ctx.i), dtype=float)
                ok = np.isfinite(v)
                nok = int(ok.sum())
                if nok < 2:
                    continue
                # 只对有效值排序，NaN 不参与（与 pandas rank 完全一致）。
                # combine(method="rank") 的定义是：排名(pct) 直接加权，不再做 z-score。
                order = np.empty(n, dtype=float)
                order[ok] = np.argsort(np.argsort(v[ok])) + 1
                total[ok] += (order[ok] / nok) * (wmap.get(f.name, 1.0) / wsum)
                seen |= ok
            score = np.where(seen, total, np.nan)
            return long_top_weights(score, top_k)

    return _S()


# ------------------------------------------------------------------ 均值回归
def bollinger_reversion(window: int = 20, n_std: float = 2.0, top_k: int = 10) -> Strategy:
    """布林带均值回归：跌破下轨买入（反向因子）。"""
    class _S(Strategy):
        name = f"bollinger({window},{n_std})"
        description = "布林带超卖买入"

        def weights(self, ctx) -> pd.Series:
            i = ctx.i
            if i + 1 < window + 2:
                return None
            arr = _close_array(ctx.md)
            seg = arr[i - window + 1 : i + 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                ma = np.nanmean(seg, axis=0)
                sd = np.nanstd(seg, axis=0, ddof=1)
                z = (arr[i] - ma) / np.where(sd > 0, sd, np.nan)
            # 只在有效标的中取 z 最小（最超卖）的 top_k
            return long_top_weights(-z, top_k)

    return _S()


def mean_reversion_basket(lookback: int = 5, top_k: int = 10) -> Strategy:
    """短期反转：买过去几日跌得最多的。"""
    class _S(Strategy):
        name = f"rev({lookback})"
        description = "短期反转"

        def weights(self, ctx) -> pd.Series:
            i = ctx.i
            if i + 1 < lookback + 2:
                return None
            arr = _close_array(ctx.md)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = arr[i] / arr[i - lookback] - 1.0
            return long_top_weights(-r, top_k)

    return _S()


def pairs_trading(a: str | None = None, b: str | None = None, window: int = 60,
                   entry_z: float = 2.0, exit_z: float = 0.5, gross: float = 1.0,
                   **kwargs) -> Strategy:
    """配对交易：价差 z-score 超过阈值做空强势、做多弱势。

    参数别名：``lookback``/``period``/``n`` → ``window``。

    不指定 ``a``/``b`` 时，会在运行时自动挑出**相关性最高的一对**做配对，
    这让它在任意数据集上都能跑起来（课程示例常用这种写法）。
    """
    if kwargs:
        kw = _norm_kwargs(kwargs, "pairs_trading")
        window = kw.pop("window", window)
        entry_z = kw.pop("entry_z", entry_z)
        exit_z = kw.pop("exit_z", exit_z)
        gross = kw.pop("gross", gross)
        a = kw.pop("a", a)
        b = kw.pop("b", b)

    class _S(Strategy):
        name = f"pairs({a},{b})" if a and b else "pairs(auto)"
        description = "协整配对交易（价差 z-score）"
        _choice = None

        def _pick(self, ctx):
            """自动选择相关性最高的一对标的（只用历史数据）。"""
            if a and b:
                return a, b
            if self._choice is not None and self._choice[0] in ctx.symbols:
                return self._choice
            px = ctx.close()
            if len(px) < window + 2 or px.shape[1] < 2:
                return None
            rets = px.iloc[-window * 2:].pct_change().dropna()
            corr = rets.corr().abs()
            # .values 可能是只读视图，必须显式复制后再改对角线
            arr = np.array(corr.values, dtype=float, copy=True)
            np.fill_diagonal(arr, 0.0)
            corr = pd.DataFrame(arr, index=corr.index, columns=corr.columns)
            stacked = corr.stack()
            if stacked.empty:
                return None
            i, j = stacked.idxmax()
            self._choice = (i, j)
            return i, j

        def weights(self, ctx) -> pd.Series:
            pair = self._pick(ctx)
            if pair is None:
                return None
            sa, sb = pair
            px = ctx.close()
            if len(px) < window + 2 or sa not in px.columns or sb not in px.columns:
                return None
            spread = np.log(px[sa]) - np.log(px[sb])
            s = spread.iloc[-window:]
            z = (spread.iloc[-1] - s.mean()) / s.std(ddof=1)
            out = pd.Series(0.0, index=ctx.symbols)
            if abs(z) > entry_z:
                out[sa] = -gross * 0.5 * np.sign(z)
                out[sb] = gross * 0.5 * np.sign(z)
            return out

    return _S()


# ------------------------------------------------------------------ 组合
def risk_parity(lookback: int = 120) -> Strategy:
    """风险平价：权重 ∝ 1/波动率。"""
    class _S(Strategy):
        name = f"risk_parity({lookback})"
        description = "逆波动率加权（等风险贡献的简化实现）"

        def weights(self, ctx) -> pd.Series:
            r = ctx.returns()
            if len(r) < lookback + 2:
                return None
            vol = r.iloc[-lookback:].std(ddof=1).replace(0, np.nan)
            inv = 1.0 / vol
            inv = inv.dropna()
            if inv.empty:
                return None
            return (inv / inv.sum()).reindex(ctx.symbols).fillna(0.0)

    return _S()


def vol_target(target_vol: float = 0.15, lookback: int = 60, base: Strategy | None = None) -> Strategy:
    """波动率目标：按组合已实现波动动态调整总敞口。"""
    inner = base or risk_parity(lookback)

    class _S(Strategy):
        name = f"vol_target({target_vol:.0%})"
        description = "在基础策略上做波动率目标缩放"

        def weights(self, ctx) -> pd.Series:
            i = ctx.i
            w = inner.weights(ctx)
            if w is None:
                return None
            if i + 1 < lookback + 2:
                return w
            arr = _returns_array(ctx.md)
            wv = np.asarray(w, dtype=float)
            seg = arr[i - lookback + 1 : i + 1]
            port = np.nansum(seg * wv, axis=1)
            rv = float(np.nanstd(port, ddof=1) * np.sqrt(252))
            if not np.isfinite(rv) or rv <= 1e-6:
                return w
            scale = float(np.clip(target_vol / rv, 0.1, 2.0))
            return w * scale

    return _S()


def rotation(universe: list[str] | None = None, top_k: int = 3, lookback: int = 60) -> Strategy:
    """资产轮动：在给定池子里按动量选前 k。"""
    class _S(Strategy):
        name = f"rotation(k={top_k})"
        description = "资产轮动（动量排序）"

        def weights(self, ctx) -> pd.Series:
            i = ctx.i
            syms = ctx.symbols
            if i + 1 < lookback + 2:
                return None
            arr = _close_array(ctx.md)
            pos = np.array([syms.index(s) for s in (universe or syms) if s in syms])
            if len(pos) == 0:
                return None
            with np.errstate(invalid="ignore", divide="ignore"):
                mom = arr[i, pos] / arr[i - lookback, pos] - 1.0
            w = np.asarray(long_top_weights(mom, top_k))
            out = np.zeros(len(syms))
            out[pos] = w
            return out

    return _S()


def equal_weight(universe: list[str] | None = None) -> Strategy:
    """等权持有（最常用的基准）。"""
    class _S(Strategy):
        name = "equal_weight"
        description = "等权持有全部标的"

        def weights(self, ctx) -> pd.Series:
            pool = [s for s in (universe or ctx.symbols)]
            return pd.Series(1.0 / len(pool), index=pool).reindex(ctx.symbols).fillna(0.0)

    return _S()


def buy_and_hold(symbol: str | None = None) -> Strategy:
    """买入持有。``symbol=None`` 时等权买入全部。"""
    class _S(Strategy):
        name = "buy_and_hold"
        description = "买入并持有"

        def __init__(self):
            self._done = False

        def weights(self, ctx) -> pd.Series | None:
            if self._done:
                return None
            self._done = True
            if symbol is not None:
                out = pd.Series(0.0, index=ctx.symbols)
                if symbol in out.index:
                    out[symbol] = 1.0
                return out
            return pd.Series(1.0 / len(ctx.symbols), index=ctx.symbols)

    return _S()


# ------------------------------------------------------------------ 辅助
def _factor_panel_at(md, factor, i: int) -> pd.Series:
    """取因子在第 i 日的横截面（带缓存）。

    性能说明：因子 `compute` 需要完整 MarketData。早期实现每个交易日都把
    "截至今天"的数据重新组装成 MarketData 再算一遍因子，导致 1800 天里
    重复计算了 1799 次相同的历史。正确做法是**在完整数据上算一次因子面板**，
    然后只取第 i 行——因为因子本身在 t 日的值只依赖 t 日及之前的数据
    （这一点由 `test_no_lookahead.py` 单独验证），所以 "先算全量再取第 i 行"
    与 "每天只用历史算" 在数学上完全等价，但快两个数量级。

    安全性：缓存键里带上数据集的日期长度与首尾日期，避免对 slice 后的不同
    数据集误用同一份缓存（那会造成真正的错位）。
    """
    sig = (len(md.dates), md.dates[0].value, md.dates[-1].value) if len(md.dates) else (0, 0, 0)
    cache = getattr(md, "_factor_cache", None)
    if cache is None:
        cache = {}
        try:
            md._factor_cache = cache  # noqa: SLF001
        except AttributeError:  # pragma: no cover
            return factor.direction * factor.compute(md).iloc[i]
    key = (factor.name, sig)
    if key not in cache:
        cache[key] = factor.direction * factor.compute(md)
    if i >= len(cache[key]):
        # 理论上不会发生（i 是相对当前 md 的），保险起见退化为即时计算
        return factor.direction * factor.compute(md).iloc[i]
    return cache[key].iloc[i]


def _truncate_md(ctx):
    """把 ctx 里截至当日的数据还原成 MarketData（供因子计算使用）。

    这是**策略里直接写因子计算**时的正确做法，但每天调用会重建整个数据集，
    因此只应在确实需要完整 MarketData 的场合使用；能用
    :func:`_factor_panel_at` 时优先用它。
    """
    from ..data.market import MarketData

    i = ctx.i
    md = ctx.md
    panels = {k: v.iloc[: i + 1] for k, v in md.panels.items()}
    funds = {k: v.iloc[: i + 1] for k, v in md.fundamentals.items()}
    return MarketData(panels, funds, name=md.name, meta=md.meta)


def _winsor_np(v: np.ndarray, n: float = 5.0) -> np.ndarray:
    """numpy 版横截面去极值（中位数 ± n×1.4826×MAD）。

    与 `_winsor_row` 等价；策略每天要跑一次，避免 pandas 开销。
    """
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(v)
        mad = np.nanmedian(np.abs(v - med)) * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return v
    return np.clip(v, med - n * mad, med + n * mad)


def _winsor_row(s: pd.Series, n: float = 5.0) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return s
    return s.clip(med - n * mad, med + n * mad)


BUILTIN_STRATEGIES = [
    "sma_cross", "dual_momentum", "factor_long_only", "multi_factor_rank",
    "turtle_breakout", "bollinger_reversion", "mean_reversion_basket",
    "pairs_trading", "risk_parity", "vol_target", "rotation", "equal_weight",
    "buy_and_hold",
]
