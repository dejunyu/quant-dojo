"""样本外验证：滚动前推、净化 K 折、蒙特卡洛、区块自助法.

金融数据不能随机切分，原因有两个，本模块的两个核心函数分别对应它们：

1. **时序依赖**：相邻样本不独立 → 用 **滚动前推**（walk_forward）保持时序；
2. **标签重叠**：``horizon=20`` 的预测样本，第 t 天与第 t+1 天的标签共用
   未来 19 天的数据 → 用 **净化（purging）+ 禁运（embargo）** 切断泄漏。
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..backtest import Backtest, BacktestConfig
from ..data.market import MarketData
from ..metrics.perf import sharpe, annual_return, max_drawdown

__all__ = ["walk_forward", "purged_kfold", "purged_kfold_indices", "monte_carlo",
           "bootstrap_blocks", "anchored_walk_forward"]


def _slice_md(md: MarketData, start=None, end=None) -> MarketData:
    return md.slice(start=start, end=end)


def walk_forward(
    md_or_factory,
    factory_or_md=None,
    param_grid: dict[str, Sequence] | None = None,
    n_splits: int = 5,
    train_ratio: float = 0.6,
    config: BacktestConfig | None = None,
    metric: str = "sharpe",
    anchored: bool = False,
    verbose: bool = True,
    train_window: int | None = None,
    test_window: int | None = None,
    **_ignored,
) -> dict:
    """滚动前推（Walk-Forward）验证.

    把时间轴切成 ``n_splits`` 段，每段用**之前**的数据选参数，然后用选出的
    参数在**随后的样本外段**回测。最后把所有样本外段拼接成一条**无参数前视**
    的净值曲线。

    两种调用形式：

    * ``walk_forward(md, factory, param_grid)`` —— 以数据为主语（推荐）；
    * ``walk_forward(factory, md, train_window=504, test_window=126)`` ——
      以策略为主语，适合课程里"按窗口滚动"的直觉写法。

    Parameters
    ----------
    train_window, test_window :
        训练窗口（交易日数）与测试窗口。给出 ``test_window`` 时，分段方式
        改为固定窗口滚动，``n_splits`` 会被忽略。
    anchored :
        True 时训练集起点固定（扩展窗口），False 时滚动窗口。
    """
    # ---- 兼容 (factory, md) 的参数顺序 ----
    if factory_or_md is not None and callable(md_or_factory) and hasattr(factory_or_md, "dates"):
        md, factory = factory_or_md, md_or_factory
    elif hasattr(md_or_factory, "dates"):
        md, factory = md_or_factory, factory_or_md
    else:
        raise TypeError(
            "walk_forward 需要 (md, strategy_factory) 或 (strategy_factory, md)；"
            f"收到 {type(md_or_factory).__name__} 与 {type(factory_or_md).__name__}"
        )
    if not callable(factory):
        raise TypeError(f"第二参数必须是策略工厂（可调用对象），收到 {type(factory).__name__}")

    # 也接受"已经建好的策略实例"：此时没有参数可搜索，直接用它滚动回测。
    # （课程里 walk_forward(strategy, md, train_window=..) 是很自然的写法）
    if hasattr(factory, "weights") or hasattr(factory, "name") and not callable(getattr(factory, "__init__", None)):
        instance = factory
        param_grid = None

        def factory(**_):  # type: ignore[misc] # noqa: F811
            return instance

    from .grid import search_combos, _expand_grid

    cfg = config or BacktestConfig()
    dates = md.dates
    n = len(dates)
    if n < 100:
        raise ValueError(f"样本太短（{n} 天），滚动前推至少需要 100 个交易日")

    # ---- 计算分段 ----
    segments: list[tuple[int, int]] = []   # (test_start_i, test_end_i)
    if test_window:
        tw = int(test_window)
        start = int(train_window or tw * 4)
        k = start
        while k < n - 5:
            segments.append((k, min(k + tw - 1, n - 1)))
            k += tw
    else:
        if n_splits < 2:
            raise ValueError("n_splits 至少为 2")
        seg = n // int(n_splits)
        for k in range(1, int(n_splits)):
            segments.append((k * seg, min((k + 1) * seg - 1, n - 1)))

    rows = []
    oos_returns: list[pd.Series] = []
    oos_equities: list[pd.Series] = []

    for fold, (test_start_i, test_end_i) in enumerate(segments, 1):
        if test_start_i >= n - 5:
            break
        test_start, test_end = dates[test_start_i], dates[test_end_i]
        train_end = dates[test_start_i - 1]
        if anchored:
            train_start = dates[0]
        elif train_window:
            train_start = dates[max(0, test_start_i - int(train_window))]
        else:
            train_start_i = max(0, test_start_i - int((test_start_i) / max(train_ratio, 1e-6)))
            train_start = dates[train_start_i]

        train_md = _slice_md(md, start=train_start, end=train_end)
        test_md = _slice_md(md, start=test_start, end=test_end)

        # --- 训练段内选参（再切一部分做内部验证）---
        if param_grid:
            inner_cut = int(len(train_md.dates) * train_ratio)
            if inner_cut > 60:
                sel_md = train_md.slice(end=train_md.dates[inner_cut - 1])
            else:
                sel_md = train_md
            search = search_combos(sel_md, factory, _expand_grid(param_grid),
                                   config=cfg, metric=metric, verbose=False)
            best = search["best_params"]
        else:
            best = {}

        strat = factory(**best)
        res = Backtest(test_md, strat, cfg).run()
        rows.append({
            "fold": fold,
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "params": {kk: (vv.item() if hasattr(vv, "item") else vv) for kk, vv in best.items()},
            "sharpe": res.metrics.get("sharpe", np.nan),
            "annual_return": res.metrics.get("annual_return", np.nan),
            "max_drawdown": res.metrics.get("max_drawdown", np.nan),
        })
        oos_returns.append(res.returns)
        if len(oos_equities) == 0:
            oos_equities.append(res.equity)
        else:
            base = oos_equities[-1].iloc[-1]
            oos_equities.append(res.equity * base)
        if verbose:
            print(f"  fold {fold}: {test_start.date()} ~ {test_end.date()}  "
                  f"sharpe={res.metrics.get('sharpe', float('nan')):.2f}  params={best}")

    if not rows:
        raise ValueError("没有可用的样本外区间，请增大数据或减少 test_window")

    folds = pd.DataFrame(rows)
    combined = pd.concat(oos_equities)
    # 去掉拼接处的重复日期
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    equity = combined / combined.iloc[0]
    all_ret = pd.concat(oos_returns).sort_index()

    return {
        "folds": folds,
        "equity": equity,
        "returns": all_ret,
        "sharpe": float(sharpe(all_ret)),
        "annual_return": float(annual_return(equity)),
        "max_drawdown": float(max_drawdown(equity)),
        "n_folds": len(folds),
        "params": folds["params"].tolist() if "params" in folds else [],
        # 各折的样本外夏普（便于快速看"是不是每段都还行"）
        "scores": folds["sharpe"].tolist() if "sharpe" in folds else [],
        "note": "这条曲线每一段用的都是该段之前选出的参数，因此不含参数前视。",
    }


def anchored_walk_forward(md, factory, param_grid=None, n_splits=5, **kw) -> dict:
    """扩展窗口版滚动前推（训练集起点固定）。"""
    return walk_forward(md, factory, param_grid, n_splits=n_splits, anchored=True, **kw)


def purged_kfold(
    md,
    n_splits: int = 5,
    horizon: int = 20,
    embargo_frac: float = 0.01,
    embargo: int | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """净化 K 折（Purged K-Fold with Embargo）.

    Parameters
    ----------
    md :
        MarketData，或一个可切片的对象；也可以直接传**观测数**（整数），
        此时返回纯索引区间的折。
    horizon :
        预测周期。净化会剔除测试集之前 horizon 个样本。
    embargo_frac :
        禁运比例（占样本总数）。
    embargo :
        禁运的**绝对天数**；给出时优先于 ``embargo_frac``。

    Returns
    -------
    每个折是 ``(train_start, train_end, test_start, test_end)`` 四个日期。

    另外提供了 :func:`purged_kfold_indices`，返回 ``(train_idx, test_idx)``
    的整数索引列表——做机器学习交叉验证时这种形式更直接。

    注意：本函数只给出折的边界，真正的"净化"（从训练集里删除与测试期标签
    重叠的样本）需要在你的训练管线里按 horizon 自行完成——这里提供的是
    最小可用的边界划分。
    """
    # 允许传入观测数（课程示例里常见 purged_kfold(T, ...) 的写法）
    if isinstance(md, (int, np.integer)):
        n_total = int(md)
        dates = pd.date_range("2020-01-01", periods=n_total, freq="B")
    else:
        dates = pd.DatetimeIndex(md.dates) if hasattr(md, "dates") else pd.DatetimeIndex(md)
        n_total = len(dates)

    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError("n_splits 至少为 2")

    embargo_days = int(embargo) if embargo is not None else int(n_total * embargo_frac)
    fold_size = n_total // n_splits

    out = []
    for k in range(n_splits):
        test_start_i = k * fold_size
        test_end_i = min((k + 1) * fold_size - 1, n_total - 1)
        test_start, test_end = dates[test_start_i], dates[test_end_i]

        # 净化：训练集要排除测试段之前 horizon 个样本
        purge_start_i = max(0, test_start_i - horizon)
        # 禁运：训练集还要排除测试段之后 embargo_days 个样本
        embargo_end_i = min(n_total - 1, test_end_i + embargo_days)

        before = (0, purge_start_i - 1) if purge_start_i > 0 else None
        after = (embargo_end_i + 1, n_total - 1) if embargo_end_i + 1 < n_total else None

        if before and after:
            train_start, train_end = dates[before[0]], dates[after[1]]
        elif before:
            train_start, train_end = dates[before[0]], dates[before[1]]
        elif after:
            train_start, train_end = dates[after[0]], dates[after[1]]
        else:
            continue
        out.append((train_start, train_end, test_start, test_end))
    return out


def purged_kfold_indices(
    n_obs: int,
    n_splits: int = 5,
    horizon: int = 20,
    embargo: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """净化 K 折的**索引**版本（机器学习交叉验证用）。

    返回 ``[(train_idx, test_idx), ...]``，其中：

    * 测试折是连续的时序段（不随机打乱）；
    * 训练集剔除了测试段之前 ``horizon`` 个样本（净化，因为标签会看未来）；
    * 训练集还剔除了测试段之后 ``embargo`` 个样本（禁运，切断序列相关泄漏）。

    ```python
    for tr, te in purged_kfold_indices(len(X), n_splits=5, horizon=20, embargo=5):
        model.fit(X[tr], y[tr]); pred[te] = model.predict(X[te])
    ```
    """
    n_obs = int(n_obs)
    if n_splits < 2:
        raise ValueError("n_splits 至少为 2")
    if horizon < 0 or embargo < 0:
        raise ValueError("horizon 与 embargo 不能为负")
    fold = n_obs // n_splits
    out = []
    for k in range(n_splits):
        te_start = k * fold
        te_end = min((k + 1) * fold, n_obs)          # 左闭右开
        test_idx = np.arange(te_start, te_end)

        # 净化：训练集排除 [te_start - horizon, te_end + embargo)
        lo = max(0, te_start - horizon)
        hi = min(n_obs, te_end + embargo)
        train_idx = np.concatenate([np.arange(0, lo), np.arange(hi, n_obs)])
        out.append((train_idx, test_idx))
    return out


def monte_carlo(
    returns,
    n_sims: int = 1000,
    horizon: int = 252,
    method: str = "iid",
    block: int = 20,
    seed: int = 0,
) -> dict:
    """蒙特卡洛模拟净值路径（用于置信区间与风险估计）。

    method : ``"iid"`` 独立抽样；``"block"`` 区块自助法（保留局部自相关，
    更贴近真实金融序列）。
    """
    r = pd.Series(returns).astype(float).dropna().to_numpy()
    if len(r) < 10:
        return {"error": "样本太少"}
    rng = np.random.default_rng(seed)
    finals = np.empty(n_sims)
    max_dds = np.empty(n_sims)

    for s in range(n_sims):
        if method == "iid":
            draws = rng.choice(r, size=horizon, replace=True)
        else:
            draws = _block_sample(rng, r, horizon, block)
        nav = np.cumprod(1.0 + draws)
        finals[s] = nav[-1]
        peak = np.maximum.accumulate(nav)
        max_dds[s] = (nav / peak - 1.0).min()
        if s == 0:
            sample_path = nav

    return {
        "n_sims": int(n_sims),
        "horizon": int(horizon),
        "method": method,
        "final_p05": float(np.percentile(finals, 5)),
        "final_p50": float(np.percentile(finals, 50)),
        "final_p95": float(np.percentile(finals, 95)),
        "final_mean": float(finals.mean()),
        "prob_loss": float((finals < 1.0).mean()),
        "maxdd_p05": float(np.percentile(max_dds, 5)),
        "maxdd_median": float(np.percentile(max_dds, 50)),
        # 全部模拟的终值与最大回撤（做自定义分位数分析时直接用）
        "finals": finals,
        "max_drawdowns": max_dds,
        "sample_path": sample_path.tolist()[: min(horizon, 500)],
    }


def bootstrap_blocks(returns, statistic: Callable = sharpe, n_sims: int = 500,
                     block: int = 20, seed: int = 0, alpha: float = 0.05) -> dict:
    """区块自助法：估计统计量的置信区间（比正态假设稳健）。"""
    r = pd.Series(returns).astype(float).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_sims):
        sample = _block_sample(rng, r, len(r), block)
        try:
            vals.append(float(statistic(pd.Series(sample))))
        except Exception:  # noqa: BLE001
            continue
    arr = np.array([v for v in vals if np.isfinite(v)])
    if not len(arr):
        return {"error": "统计量计算失败"}
    return {
        "statistic": getattr(statistic, "__name__", "stat"),
        "point_estimate": float(statistic(pd.Series(r))),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "ci_low": float(np.percentile(arr, alpha * 100)),
        "ci_high": float(np.percentile(arr, (1 - alpha) * 100)),
        "n_sims": len(arr),
    }


def _block_sample(rng, arr: np.ndarray, size: int, block: int) -> np.ndarray:
    n = len(arr)
    block = max(1, min(int(block), n))
    n_blocks = int(np.ceil(size / block))
    starts = rng.integers(0, n - block, size=n_blocks)
    out = np.concatenate([arr[s : s + block] for s in starts])
    return out[:size]
