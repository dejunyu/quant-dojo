"""单因子研究实验室（FactorLab）.

一个因子从"想法"到"可用"要过五关，本模块把每一关都做成一个可调用的检查：

1. **覆盖率**：因子在横截面上有多少标的非缺失？< 50% 基本不可用。
2. **IC**：因子值与未来收益的横截面相关性。
3. **ICIR / t 检验**：IC 序列是否稳定为正？
4. **分层回测**：单调性如何？多空组合能否赚钱？
5. **衰减 / 换手 / 成本**：多久失效？扣掉成本还剩多少？

只有五关都过，因子才值得进入组合。``lab.report(name)`` 一次给出全部结论。
"""
from __future__ import annotations

from typing import Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..factors.pipeline import winsorize, zscore, neutralize, rank, to_weights, factor_returns
from ..factors.base import get_factor
from ..metrics.perf import perf_report, sharpe, max_drawdown, annual_return

__all__ = ["FactorLab"]


class ICSeries(pd.Series):
    """IC 时间序列，但**可以当标量用**。

    课程与实践中两种写法都很常见：

    * ``ic.groupby(ic.index.year).mean()`` —— 当序列用；
    * ``print(f"IC={lab.ic(name):.4f}")``、``ic * 2`` —— 当均值用。

    二者在意图上都清楚，强迫用户区分只是一种没必要的摩擦。所以这里对
    格式化、算术、与数字比较都返回**均值**，其余操作仍是 Series 语义。
    """

    @property
    def _v(self) -> float:
        return float(pd.Series(self).mean()) if len(self) else float("nan")

    def __format__(self, spec: str) -> str:
        return format(self._v, spec) if spec else str(self._v)

    def __float__(self) -> float:
        return self._v

    def __round__(self, ndigits: int = 0):  # type: ignore[override]
        return round(self._v, ndigits)

    def __eq__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return bool(self._v == other)
        return pd.Series.__eq__(self, other)

    def __hash__(self) -> int:  # 与 __eq__ 配套
        return object.__hash__(self)

    def __repr__(self) -> str:
        try:
            return f"ICSeries(mean={self._v:.4f}, n={len(self)})"
        except Exception:  # noqa: BLE001
            return "ICSeries()"


class ICComparison(pd.DataFrame):
    """IC 衰减表：既像 DataFrame（``["ic"]`` 取列）又像 Series（横向取统计量）。

    两种用法都很自然：

    ```python
    decay = lab.ic_decay("mom_12_1")
    decay["ic"]                 # DataFrame 风格：取 IC 列
    decay.max(); decay.idxmax() # Series 风格：IC 的最大值与其 horizon
    decay.items()               # 遍历 (horizon, IC)
    ```

    为避免歧义：``decay["ic"]`` 这类**列名**访问走 DataFrame 语义，
    其余 Series 属性与运算默认作用于 ``ic`` 列。
    """

    _metric = "ic"

    @property
    def _as_series(self) -> pd.Series:
        s = self[self._metric] if self._metric in self.columns else self.iloc[:, -1]
        return pd.Series(s.to_numpy(), index=pd.Index(self["horizon"].to_numpy(), name="horizon"))

    def __getattr__(self, item):
        # 只有在 DataFrame 自己没有这个属性时才退回 Series 语义
        if item.startswith("_") or item in ("horizon", "ic", "icir", "t_stat", "pos_ratio"):
            raise AttributeError(item)
        if hasattr(pd.DataFrame, item):
            return super().__getattr__(item)
        try:
            s = self._as_series
        except Exception as exc:  # noqa: BLE001
            raise AttributeError(item) from exc
        return getattr(s, item)

    def items(self):  # type: ignore[override]
        return self._as_series.items()

    def best_horizon(self) -> int:
        """IC 最大的那个预测周期（衰减曲线的峰值位置）。"""
        s = self._as_series
        return int(s.idxmax())

    def peak_ic(self) -> float:
        """IC 的峰值。"""
        return float(self._as_series.max())

    def half_life(self) -> int | None:
        """IC 衰减到峰值一半所需的周期（线性插值）。

        直观含义："这个信号还能活多久"。半衰期越短，调仓就要越频繁，
        而越频繁就越容易被交易成本吃掉。峰值若出现在最大周期上（IC 还在
        上升），说明观测窗口不够长，返回 ``None``。
        """
        s = self._as_series.dropna()
        if s.empty or s.max() <= 0:
            return None
        target = s.max() / 2
        peak_pos = int(np.argmax(s.to_numpy()))
        xs = [int(x) for x in s.index]
        # 只在峰值之后找（衰减段）
        for i in range(peak_pos + 1, len(xs)):
            if s.iloc[i] <= target:
                x0, x1 = xs[i - 1], xs[i]
                y0, y1 = float(s.iloc[i - 1]), float(s.iloc[i])
                if y0 == y1:
                    return x1
                return int(round(x0 + (target - y0) * (x1 - x0) / (y1 - y0)))
        return None

    def to_dict(self, orient="dict", *args, **kwargs):  # type: ignore[override]
        # 保留 DataFrame 的 records 用法；默认（dict）改为 horizon -> ic
        if orient == "rows":
            orient = "records"
        if orient != "dict":
            return super().to_dict(orient=orient, *args, **kwargs)
        return self._as_series.to_dict()


def _to_ic_comparison(df: pd.DataFrame) -> ICComparison:
    return ICComparison(df.reset_index(drop=True))


class FactorLab:
    """单因子研究实验室.

    Parameters
    ----------
    md :
        行情数据（含所需基本面字段）。
    horizon :
        默认预测周期（交易日）。
    fwd_returns :
        自定义前瞻收益面板；``None`` 时按 horizon 自动算。
    """

    def __init__(self, md, horizon: int = 20, fwd_returns: pd.DataFrame | None = None,
                 _shared: dict | None = None):
        self.md = md
        self.horizon = int(horizon)
        # 跨 horizon 共享：原始因子面板与后期收益面板都很贵，不同 horizon 的
        # FactorLab 只是切片方式不同，没必要重算。
        self._shared = _shared if _shared is not None else {}
        if fwd_returns is None:
            px = md.close()
            self.fwd = px.shift(-self.horizon) / px - 1.0
        else:
            self.fwd = pd.DataFrame(fwd_returns).reindex(index=md.dates, columns=md.symbols)
        self._cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------ 自定义因子
    def register(self, name: str, panel: pd.DataFrame, category: str = "未分类",
                 direction: int = 1, description: str = "") -> pd.DataFrame:
        """把一份自己算出来的因子面板注册进因子库，之后就能用名字分析了。

        这是做自定义因子研究最短的路径：

        ```python
        raw = get_factor("bp").compute(md)
        lab.register("bp_rank", raw.rank(axis=1, pct=True), description="BP 的截面排名")
        lab.ic("bp_rank").mean()          # 和内置因子完全一样地分析
        lab.quantile_returns("bp_rank")
        ```

        同名因子会被覆盖更新（方便“改一版再试”）。
        """
        from ..factors.base import register as _register, FactorSpec

        panel = pd.DataFrame(panel).reindex(index=self.md.dates, columns=self.md.symbols)
        key = str(name)
        self._cache[key] = panel

        def _fn(_md, **_kw):
            return panel.reindex(index=_md.dates, columns=_md.symbols)

        _fn.__name__ = key
        spec = FactorSpec(
            name=key, category=category, direction=direction,
            description=description or f"用户注册因子 {key}",
            required_fields=("close",),
            params={"_custom": True},
        )
        _register(spec, _fn)
        return panel

    # ------------------------------------------------------------ 因子值
    def raw(self, name: str) -> pd.DataFrame:
        """原始因子面板（跨实例共享缓存）。"""
        key = ("raw", str(name))
        if key not in self._shared:
            self._shared[key] = get_factor(name).compute(self.md)
        return self._shared[key]

    def processed(self, name: str, winsor: bool = True, neutral: bool = True,
                  standardize: bool = True) -> pd.DataFrame:
        """标准处理后的因子面板：去极值 → 中性化 → 标准化（带缓存）。"""
        key = (str(name), winsor, neutral, standardize)
        cache = self.__dict__.setdefault("_proc_cache", {})
        if key in cache:
            return cache[key]
        out = self._processed_uncached(name, winsor, neutral, standardize)
        cache[key] = out
        return out

    def _processed_uncached(self, name: str, winsor: bool, neutral: bool,
                            standardize: bool) -> pd.DataFrame:
        f = get_factor(name)
        x = f.direction * self.raw(name)
        if winsor:
            x = winsorize(x)
        if neutral:
            x = neutralize(x, self.md)
        if standardize:
            x = zscore(x)
        return x

    # ------------------------------------------------------------ 覆盖率
    def coverage(self, name: str) -> dict:
        x = self.raw(name)
        valid = x.notna().sum(axis=1)
        n = x.shape[1]
        return {
            "factor": name,
            "avg_coverage": float(valid.mean() / n),
            "min_coverage": float(valid.min() / n),
            "last_coverage": float(valid.iloc[-1] / n),
            "n_days": int(len(x)),
        }

    # ------------------------------------------------------------ IC
    def ic(self, name: str, horizon: int | None = None, method: str = "spearman") -> pd.Series:
        """IC 时间序列（每个交易日一个横截面相关系数）。

        返回的是 ``Series``，因此可以直接做 ``.mean()``、``.std()``、
        ``.groupby(index.year).mean()`` 之类的分析。只想要平均值时用
        :meth:`mean_ic`。
        """
        return self.ic_series(name, horizon, method)

    def mean_ic(self, name: str, horizon: int | None = None, method: str = "spearman") -> float:
        """平均 IC（标量）。"""
        s = self.ic_series(name, horizon, method)
        return float(s.mean())

    def _ic_from_panel(self, x: pd.DataFrame, name: str, h: int,
                       method: str = "spearman") -> "ICSeries":
        """用已处理好的因子面板计算某个 horizon 的 IC 序列。

        与 :meth:`ic_series` 的区别是：因子面板由调用方传入（已预处理、已缓存），
        因此扫多个 horizon 时不必重复做去极值/中性化。
        """
        px = self.md.close()
        fwd = (px.shift(-h) / px - 1.0).reindex(index=x.index, columns=x.columns)
        both = x.notna() & fwd.notna()
        enough = both.sum(axis=1) >= 5
        mask = both & enough.to_numpy()[:, None]
        xm = x.where(mask)
        fm = fwd.where(mask)
        if method == "spearman":
            xm = xm.rank(axis=1)
            fm = fm.rank(axis=1)
        elif method != "pearson":
            raise ValueError("method 必须是 spearman / pearson")
        out = _rowwise_corr(xm.to_numpy(dtype=float), fm.to_numpy(dtype=float))
        s = pd.Series(out, index=x.index, name=f"IC_{name}_{h}").dropna()
        return ICSeries(s)

    def ic_series(self, name: str, horizon: int | None = None, method: str = "spearman") -> pd.Series:
        """IC 时间序列。

        实现说明：先对因子与前瞻收益做**横截面排名**（spearman 等价于对
        排名做 pearson），再用矩阵运算一次性算出所有日期的相关系数，避免
        逐日循环。这样 1800 天 × 50 标的的 IC 序列在毫秒级完成。
        """
        h = int(horizon or self.horizon)
        x = self.processed(name)          # 带缓存，多个 horizon 只算一次
        if h != self.horizon:
            return self._ic_from_panel(x, name, h, method)
        fwd = self.fwd.reindex(index=x.index, columns=x.columns)

        both = x.notna() & fwd.notna()
        # 太少的横截面样本不参与计算
        enough = both.sum(axis=1) >= 5
        xm = x.where(both & enough.to_numpy()[:, None])
        fm = fwd.where(both & enough.to_numpy()[:, None])

        if method == "spearman":
            xm = xm.rank(axis=1)
            fm = fm.rank(axis=1)
        elif method != "pearson":
            raise ValueError("method 必须是 spearman / pearson")

        out = _rowwise_corr(xm.to_numpy(dtype=float), fm.to_numpy(dtype=float))
        s = pd.Series(out, index=x.index, name=f"IC_{name}").dropna()
        return ICSeries(s)

    def icir(self, name: str, horizon: int | None = None) -> float:
        """ICIR = mean(IC) / std(IC)，IC 的稳定性。"""
        s = self.ic_series(name, horizon)
        sd = s.std(ddof=1)
        return float(s.mean() / sd) if sd and np.isfinite(sd) else float("nan")

    def ic_tstat(self, name: str, horizon: int | None = None) -> float:
        """IC 序列的 t 统计量：$t = \\mathrm{ICIR}\\sqrt{n-1}$。

        经验阈值：|t| > 2 才算显著（但样本大时几乎总会显著，所以更要看 IC 本身）。
        """
        s = self.ic_series(name, horizon).dropna()
        n = len(s)
        icir = s.mean() / s.std(ddof=1) if s.std(ddof=1) else np.nan
        return float(icir * np.sqrt(n - 1)) if n > 2 and np.isfinite(icir) else float("nan")

    def ic_decay(self, name: str, horizons: Sequence[int] = (1, 5, 10, 20, 60, 120)) -> pd.DataFrame:
        """IC 衰减曲线：不同预测周期下的 IC，用来选调仓周期。"""
        rows = []
        for h in horizons:
            s = self.ic_series(name, horizon=int(h)).dropna()
            rows.append({
                "horizon": int(h),
                "ic": float(s.mean()) if len(s) else np.nan,
                "icir": float(s.mean() / s.std(ddof=1)) if len(s) > 2 and s.std(ddof=1) else np.nan,
                "t_stat": float(s.mean() / s.std(ddof=1) * np.sqrt(len(s) - 1)) if len(s) > 2 and s.std(ddof=1) else np.nan,
                "pos_ratio": float((s > 0).mean()) if len(s) else np.nan,
            })
        return _to_ic_comparison(pd.DataFrame(rows))

    def ic_summary(self, name: str) -> dict:
        s = self.ic_series(name).dropna()
        if s.empty:
            return {"factor": name, "error": "没有足够的 IC 样本"}
        return {
            "factor": name,
            "horizon": self.horizon,
            "ic_mean": float(s.mean()),
            "ic_std": float(s.std(ddof=1)),
            "icir": float(s.mean() / s.std(ddof=1)) if s.std(ddof=1) else np.nan,
            "t_stat": float(s.mean() / s.std(ddof=1) * np.sqrt(len(s) - 1)) if len(s) > 2 else np.nan,
            "ic_positive_ratio": float((s > 0).mean()),
            "ic_min": float(s.min()),
            "ic_max": float(s.max()),
            "n_periods": int(len(s)),
        }

    # ------------------------------------------------------------ 分层回测
    def quantile_returns(self, name: str, n_quantiles: int = 5,
                         horizon: int | None = None, stride: int | None = None) -> pd.DataFrame:
        """各分组的**周期收益时间序列表**（列为 ``Q1..Qn``）。

        重要：默认返回的收益是**重叠**的（第 t 天与 t+1 天都覆盖了未来
        ``horizon`` 天），直接 ``(1 + ret).cumprod()`` 会把同一段行情重复算
        ``horizon`` 次，结果严重失真。因此默认按 ``stride=horizon`` **采样**，
        只保留互不重叠的调仓日，这样的收益序列才能直接累乘成净值：

        ```python
        ret = lab.quantile_returns("mom_12_1")      # 已按 20 日步长采样
        nav = (1 + ret["Q1"]).cumprod()             # 可以直接累乘
        ```

        想要完整的逐日序列（做 IC 分析、画图）时传 ``stride=1``。
        """
        h = int(horizon or self.horizon)
        data = self._quantile_returns_impl(name, n_quantiles, horizon)
        if data.empty:
            return data
        step = int(stride) if stride is not None else max(1, h)
        if step > 1:
            data = data.iloc[::step]
        return data

    def _quantile_returns_impl(self, name: str, n_quantiles: int, horizon: int | None) -> pd.DataFrame:
        h = int(horizon or self.horizon)
        lab = self if h == self.horizon else FactorLab(self.md, horizon=h, _shared=self._shared)
        x = lab.processed(name)
        fwd = lab.fwd.reindex(index=x.index, columns=x.columns)

        valid = x.notna() & fwd.notna()
        cnt = valid.sum(axis=1)
        keep = cnt >= n_quantiles * 2
        if not keep.any():
            return pd.DataFrame()

        xv = x.where(valid)
        fv = fwd.where(valid)
        rk = xv.rank(axis=1, ascending=False, method="first")
        maxr = cnt.where(keep)
        qid = np.ceil(rk.to_numpy() / (maxr.to_numpy()[:, None] / n_quantiles))
        qid = np.clip(qid, 1, n_quantiles)
        qid = np.where(np.isfinite(rk.to_numpy()), qid, np.nan)

        out = {}
        fv_arr = fv.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for q in range(1, n_quantiles + 1):
                masked = np.where(qid == q, fv_arr, np.nan)
                out[f"Q{q}"] = np.nanmean(masked, axis=1)
        df = pd.DataFrame(out, index=x.index)
        return df.loc[keep.to_numpy()].dropna(how="all")

    def quantile_backtest(self, name: str, n_quantiles: int = 5,
                          horizon: int | None = None, long_top: bool = True) -> dict:
        """分层回测：把因子排序分为 n 组，比较各组未来收益。

        返回每组的年化收益、夏普，以及多空（Q1 − Qn）表现与单调性。
        想要**原始时间序列**自己做分析时用 :meth:`quantile_returns`。
        """
        h = int(horizon or self.horizon)
        # 用**非重叠**收益：重叠窗口会把同一段行情重复计 h 次，年化严重失真
        ret_df = self._quantile_returns_impl(name, n_quantiles, horizon).iloc[:: max(1, h)]
        if ret_df.empty:
            return {"factor": name, "error": "分层回测样本不足"}
        grp_ret = {q: ret_df[f"Q{q}"] for q in range(1, n_quantiles + 1)}
        idx = ret_df.index
        # 重叠收益：按 horizon 调整年化（重叠窗口下收益序列自相关，这里对
        # 年化只做频率换算，不做 Newey-West 修正，报告里给出提示）
        ppy = 252.0 / h
        groups = {}
        for q in range(1, n_quantiles + 1):
            s = pd.Series(grp_ret[q], index=idx)
            nav = (1.0 + s).cumprod()
            groups[f"Q{q}"] = {
                "annual_return": float(annual_return(nav, ppy)),
                "sharpe": float(sharpe(s, 0.0, ppy)),
                "avg_period_return": float(s.mean()),
                "max_drawdown": float(max_drawdown(nav)),
            }

        # 多空组合
        ls = pd.Series(np.array(grp_ret[1]) - np.array(grp_ret[n_quantiles]), index=idx)
        ls_nav = (1.0 + ls).cumprod()
        ann = np.array([groups[f"Q{q}"]["annual_return"] for q in range(1, n_quantiles + 1)])
        mono = _monotonicity(ann)

        return {
            "factor": name,
            "horizon": h,
            "n_quantiles": n_quantiles,
            "groups": groups,
            "long_short": {
                "annual_return": float(annual_return(ls_nav, ppy)),
                "sharpe": float(sharpe(ls, 0.0, ppy)),
                "max_drawdown": float(max_drawdown(ls_nav)),
                "avg_period_return": float(ls.mean()),
                "win_rate": float((ls > 0).mean()),
            },
            "monotonicity": float(mono),
            "note": "重叠窗口收益存在自相关，夏普与 t 值偏乐观；请用非重叠窗口复核。",
        }

    # ------------------------------------------------------------ 换手 / 成本
    def turnover(self, name: str, top_k: int | None = None) -> dict:
        """因子的换手率：相邻两期持仓名单/权重的变动程度。"""
        x = self.processed(name)
        w = to_weights(x, long_top=0.2, short_bottom=0.0)
        delta = w.diff().abs().sum(axis=1)
        if top_k:
            w2 = to_weights(x, long_top=top_k / x.shape[1], short_bottom=0.0)
            delta = w2.diff().abs().sum(axis=1)
        return {
            "avg_daily_turnover": float(delta.mean()),
            "annual_turnover": float(delta.mean() * 252),
            "median_daily_turnover": float(delta.median()),
        }

    def net_ic_after_cost(self, name: str, cost_bps: float = 30.0) -> dict:
        """扣成本后的净 IC 估算（经验：成本按换手率的固定 bp 计）。"""
        base = self.ic_summary(name)
        tn = self.turnover(name)
        cost_annual = tn["annual_turnover"] * cost_bps / 1e4
        gross = base.get("ic_mean", np.nan)
        # 粗略换算：IC → 预期超额收益（每单位换手成本直接从多头端扣）
        return {
            "factor": name,
            "gross_ic": gross,
            "annual_turnover": tn["annual_turnover"],
            "cost_annual_est": float(cost_annual),
            "breakeven_note": f"若年化换手 {tn['annual_turnover']:.1f}×，成本约 {cost_annual:.2%}/年，需 gross alpha 高于此才有利可图",
        }

    # ------------------------------------------------------------ 相关性
    def correlation(self, names: Sequence[str], method: str = "spearman") -> pd.DataFrame:
        """因子间相关性矩阵（对因子值的时间序列均值相关）。"""
        cols = {}
        for n in names:
            x = self.processed(n)
            # 用每日横截面 rank 的时间均值代表因子
            cols[n] = rank(x).mean(axis=0)
        df = pd.DataFrame(cols)
        return df.corr(method=method)

    # ------------------------------------------------------------ 总报告
    def report(self, name: str, horizons: Sequence[int] = (1, 5, 20, 60)) -> "FactorReport":
        """一次跑完所有检查，返回完整单因子报告。

        返回值既是字典（``rep["ic"]``、``rep["verdict"]``），也可以直接取
        常用标量（``rep.ic_mean``、``rep["ic_mean"]``、``rep.icir``），
        以及属性风格访问（``rep.quantile_backtest``）。
        """
        cov = self.coverage(name)
        summary = self.ic_summary(name)
        decay = self.ic_decay(name, horizons)
        qb = self.quantile_backtest(name, n_quantiles=5)
        tn = self.turnover(name)
        f = get_factor(name)

        verdict, reasons = _verdict(summary, qb, cov)

        data = {
            "factor": name,
            "category": f.category,
            "direction": f.direction,
            "description": f.description,
            "formula": f.formula,
            "coverage": cov,
            "ic": summary,
            "ic_decay": decay,
            "quantile_backtest": qb,
            "turnover": tn,
            "verdict": verdict,
            "reasons": reasons,
            # 把常用的标量平铺到顶层，便于直接取用
            "ic_mean": summary.get("ic_mean"),
            "icir": summary.get("icir"),
            "t_stat": summary.get("t_stat"),
        }
        return FactorReport(data)


class FactorReport(dict):
    """单因子报告：字典与属性的双重访问。"""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def to_json(self) -> str:
        """序列化成 JSON（把 DataFrame 转成 records）。"""
        import json

        def conv(v):
            if isinstance(v, pd.DataFrame):
                return v.to_dict("records")
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [conv(x) for x in v]
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating, float)):
                return float(v) if np.isfinite(v) else None
            return v

        return json.dumps(conv(self), ensure_ascii=False, indent=2, default=str)

    def __repr__(self) -> str:
        return (
            f"FactorReport({self.get('factor')!r}, IC={self.get('ic_mean'):+.4f}, "
            f"verdict={str(self.get('verdict'))[:6]}…)"
        )


def _rowwise_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """逐行 Pearson 相关（自动处理 NaN 与零方差行）。

    对每一行做去均值、按共同有效样本求内积，等价于逐行 corrcoef，但快
    一到两个数量级。
    """
    mask = np.isfinite(A) & np.isfinite(B)
    n = mask.sum(axis=1)
    out = np.full(A.shape[0], np.nan)

    A2 = np.where(mask, A, 0.0)
    B2 = np.where(mask, B, 0.0)

    sum_a = A2.sum(axis=1)
    sum_b = B2.sum(axis=1)
    mean_a = np.where(n > 0, sum_a / np.maximum(n, 1), 0.0)
    mean_b = np.where(n > 0, sum_b / np.maximum(n, 1), 0.0)

    da = np.where(mask, A - mean_a[:, None], 0.0)
    db = np.where(mask, B - mean_b[:, None], 0.0)

    cov = (da * db).sum(axis=1)
    va = (da * da).sum(axis=1)
    vb = (db * db).sum(axis=1)
    denom = np.sqrt(va * vb)
    ok = (n >= 5) & (denom > 0)
    out[ok] = cov[ok] / denom[ok]
    return np.clip(out, -1.0, 1.0)


def _monotonicity(arr: np.ndarray) -> float:
    """单调性得分：1 = 完美单调递减（Q1 最高），0 = 完全乱序，-1 = 反向。"""
    n = len(arr)
    if n < 2:
        return float("nan")
    order = np.argsort(np.argsort(-arr))  # 0 = 最大
    ideal = np.arange(n)
    # Spearman 相关：实际排名 vs 理想排名（ideal 升序表示收益递减）
    if np.std(order) == 0:
        return 0.0
    corr = np.corrcoef(order, -ideal)[0, 1]
    return float(corr)


def _verdict(ic: dict, qb: dict, cov: dict) -> tuple[str, list[str]]:
    """给出因子"能不能用"的机械判定（教学用途，不能代替人的判断）。"""
    reasons: list[str] = []
    score = 0
    icm = ic.get("ic_mean", np.nan)
    t = ic.get("t_stat", np.nan)
    covg = cov.get("avg_coverage", 0)
    ls_sharpe = (qb.get("long_short") or {}).get("sharpe", np.nan)
    mono = qb.get("monotonicity", np.nan)

    if np.isfinite(covg):
        if covg < 0.5:
            reasons.append(f"覆盖率仅 {covg:.0%}，样本太少，结论不可靠")
        else:
            score += 1
    if np.isfinite(icm):
        if abs(icm) < 0.02:
            reasons.append(f"|IC|={abs(icm):.3f} < 0.02，信号太弱（经验阈值）")
        else:
            score += 1
            reasons.append(f"|IC|={abs(icm):.3f}，达到可用水平")
    if np.isfinite(t):
        if abs(t) < 2:
            reasons.append(f"IC 的 t={t:.2f}，|t|<2 不显著")
        else:
            score += 1
    if np.isfinite(ls_sharpe):
        if ls_sharpe < 0.5:
            reasons.append(f"多空夏普 {ls_sharpe:.2f} < 0.5，扣除成本后大概率不赚钱")
        else:
            score += 1
    if np.isfinite(mono):
        if mono > 0.6:
            score += 1
            reasons.append(f"分组单调性 {mono:.2f}，收益随因子值单调变化，可信度高")
        elif mono < 0.2:
            reasons.append(f"分组单调性 {mono:.2f} 差，可能是少数极端股票驱动的假象")

    if score >= 4:
        verdict = "通过：可以考虑进入组合（仍需样本外验证）"
    elif score >= 2:
        verdict = "可疑：部分检查通过，建议改进或与其它因子合成"
    else:
        verdict = "不通过：信号不足以支撑交易成本"
    return verdict, reasons
