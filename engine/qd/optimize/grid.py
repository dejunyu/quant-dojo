"""参数搜索：网格与随机搜索.

**使用前的警告**（这不是客套话）
--------------------------------
在金融数据上做参数搜索，样本内最优结果的期望偏误约为

$$\\mathbb{E}[\\max] \\approx \\sigma_{SR}\\left((1-\\gamma)\\Phi^{-1}\\left(1-\\frac1N\\right)+\\gamma\\Phi^{-1}\\left(1-\\frac{1}{Ne}\\right)\\right)$$

也就是说：**搜索次数越多，最优结果越不可信**。本模块因此强制要求传入
``n_trials`` 用于 Deflated Sharpe，并在结果里显式给出过拟合提示。
"""
from __future__ import annotations

import itertools
import time
import warnings
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..backtest import Backtest, BacktestConfig
from ..data.market import MarketData
from ..metrics.perf import deflated_sharpe, sharpe

__all__ = ["grid_search", "random_search", "_expand_grid"]


def _expand_grid(param_grid: dict[str, Sequence]) -> list[dict]:
    """展开参数网格为组合列表。"""
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    values = [list(param_grid[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _run_combos(
    runner: Callable[[dict], dict],
    combos: Sequence[dict],
    verbose: bool = True,
    label: str = "search",
) -> list[dict]:
    """依次调用 ``runner(combo) -> dict``，收集结果。

    ``runner`` 可以是一个普通函数（自行组织回测与指标），这是最灵活的用法：
    课程与自定义研究里通常需要同时扫多个维度（策略参数 + 成本参数）。
    """
    t0 = time.time()
    rows = []
    n = len(combos)
    for i, combo in enumerate(combos, 1):
        try:
            # runner 既可以接受一个 combo 字典（run(combo)），也可以接受
            # 展开的关键字（run(delta_rel=..., slippage_bps=...)）。按签名自动判断。
            out = _call_runner(runner, combo)
            row = dict(combo)
            if isinstance(out, dict):
                row.update(out)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - 单组失败不应终止整轮搜索
            rows.append({**combo, "error": str(exc)[:160]})
        if verbose and n and i % max(1, n // 10) == 0:
            print(f"  {label} {i}/{n}  ({time.time() - t0:.1f}s)")
    return rows


class GridResult(dict):
    """网格搜索结果：既当字典用（``out["table"]``），也可以直接迭代。

    ```python
    out = grid_search(run_fn, grid)
    for row in out:            # 直接遍历结果表（按 metric 降序）
        print(row["sharpe"])
    ```
    """

    def __iter__(self):
        table = self.get("table")
        if isinstance(table, pd.DataFrame):
            yield from table.to_dict("records")
        else:
            yield from ()

    def __len__(self) -> int:
        table = self.get("table")
        return len(table) if isinstance(table, pd.DataFrame) else dict.__len__(self)

    def __getitem__(self, key):
        # 便捷别名：out["params"] / out["score"] 等价于 best_params / best_metric
        if key == "params":
            key = "best_params"
        elif key == "score":
            key = "best_metric"
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key) -> bool:
        if key in ("params", "score"):
            return True
        return dict.__contains__(self, key)


def _call_runner(runner, combo: dict):
    """按签名决定是调用 ``runner(combo)`` 还是 ``runner(**combo)``。"""
    import inspect

    try:
        sig = inspect.signature(runner)
    except (TypeError, ValueError):
        return runner(dict(combo))
    params = list(sig.parameters.values())
    has_varkw = any(p.kind == p.VAR_KEYWORD for p in params)
    if has_varkw:
        return runner(**combo)
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    # 只有一个位置参数且名字像 "combo/params/kwargs" -> 传字典
    if len(positional) == 1 and positional[0].name in ("combo", "params", "kw", "config", "c", "p"):
        return runner(dict(combo))
    # 参数名与 combo 的键能对上 -> 展开成关键字
    names = {p.name for p in positional}
    if names and names.intersection(combo.keys()):
        return runner(**{k: v for k, v in combo.items() if k in names or has_varkw})
    return runner(dict(combo))


def _rows_to_table(rows: list[dict], metric: str, n_trials: int, elapsed: float) -> dict:
    """把结果列表整理成带排序、DSR 与过拟合警告的字典。"""
    df = pd.DataFrame(rows)
    if df.empty:
        return {"table": df, "n_trials": n_trials, "warning": "没有成功的结果"}
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    return {
        "table": df,
        "n_trials": n_trials,
        "best": df.iloc[0].to_dict(),
        "elapsed_sec": round(time.time() - 0, 2),
        "warning": _overfit_warning(n_trials, float(df.iloc[0].get(metric, np.nan)) if metric in df else np.nan, np.nan),
        "elapsed_sec": round(elapsed, 2),
    }


def _evaluate(md, factory, config, combo, metric) -> dict:
    strat = factory(**combo)
    cfg = config
    res = Backtest(md, strat, cfg).run()
    m = res.metrics
    row = dict(combo)
    row.update({
        "sharpe": m.get("sharpe", np.nan),
        "annual_return": m.get("annual_return", np.nan),
        "max_drawdown": m.get("max_drawdown", np.nan),
        "calmar": m.get("calmar", np.nan),
        "sortino": m.get("sortino", np.nan),
        "annual_vol": m.get("annual_vol", np.nan),
        "n_trades": res.trade_stats().get("n_trades", 0),
        "result": res,
    })
    return row


def grid_search(
    md_or_runner,
    factory_or_grid,
    param_grid=None,
    config: BacktestConfig | None = None,
    metric: str = "sharpe",
    top: int = 10,
    verbose: bool = True,
    **_ignored,
):
    """参数网格搜索，支持两种调用形式。

    **形式一（策略搜索，最常用）**：::

        out = grid_search(md, strategy_factory,
                          {"lookback": [60, 120], "top_k": [5, 10]})
        out["table"], out["best_params"], out["deflated_sharpe"]

    **形式二（自定义评测函数）**：::

        rows = grid_search(run_fn, {"rebalance": ["1M", "3M"], "slippage_bps": [5, 20]})
        # run_fn(combo) -> dict，返回一个可排序的 DataFrame（rows）

    形式二适合同时扫**多个维度**（策略参数 + 成本参数 + 数据子集），
    或需要自定义指标的场合。
    """
    # ---- 形式二：第一个参数是可调用对象（自定义评测函数）----
    if callable(md_or_runner) and not hasattr(md_or_runner, "symbols"):
        runner = md_or_runner
        grid = factory_or_grid if isinstance(factory_or_grid, dict) else (param_grid or {})
        combos = _expand_grid(grid)
        t0 = time.time()
        rows = _run_combos(runner, combos, verbose=verbose, label="grid")
        # 结果列 = 评测函数返回的键（而非参数键）
        run_keys = set(rows[0].keys()) - set(combos[0].keys()) if rows and combos else set()
        run_keys |= {"error"}
        out = GridResult(_rows_to_table(rows, metric, len(combos), time.time() - t0))
        out["best_params"] = _first_param_combo(out["best"], run_keys=run_keys) if out.get("best") else {}
        return out

    # ---- 形式一：MarketData + 策略工厂 ----
    md = md_or_runner
    factory = factory_or_grid
    grid = param_grid or {}
    combos = _expand_grid(grid)
    out = GridResult(search_combos(md, factory, combos, config=config, metric=metric, verbose=verbose))
    if grid:
        out["best_params"] = {k: out["best_params"].get(k) for k in grid}
    return out


def random_search(
    md: MarketData,
    factory: Callable[..., object],
    param_space: dict[str, Sequence],
    n_iter: int = 50,
    config: BacktestConfig | None = None,
    metric: str = "sharpe",
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """随机搜索：维度高时比网格搜索高效得多（Bergstra & Bengio, 2012）。

    ``param_space`` 与网格搜索同构（参数名 -> 候选值列表），但只随机采样
    ``n_iter`` 组，而不是穷举全部组合。
    """
    rng = np.random.default_rng(seed)
    combos = []
    for _ in range(int(n_iter)):
        combos.append({k: rng.choice(list(v)) for k, v in param_space.items()})
    return search_combos(md, factory, combos, config=config, metric=metric, verbose=verbose)


def search_combos(
    md: MarketData,
    factory: Callable[..., object],
    combos: Sequence[dict],
    config: BacktestConfig | None = None,
    metric: str = "sharpe",
    verbose: bool = True,
) -> dict:
    """对一组给定的参数组合逐个回测（grid_search 与 random_search 的公共内核）。"""
    cfg = config or BacktestConfig()
    t0 = time.time()
    rows = []
    n = len(combos)
    for i, combo in enumerate(combos, 1):
        try:
            rows.append(_evaluate(md, factory, cfg, combo, metric))
        except Exception as exc:  # noqa: BLE001
            rows.append({**combo, "sharpe": np.nan, "annual_return": np.nan,
                         "max_drawdown": np.nan, "calmar": np.nan, "sortino": np.nan,
                         "annual_vol": np.nan, "n_trades": 0, "error": str(exc)[:120]})
        if verbose and n and i % max(1, n // 10) == 0:
            print(f"  search {i}/{n}  ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    best_row = df.iloc[0]
    best_result = best_row.get("result")
    dsr = deflated_sharpe(best_result.returns, n_trials=n) if best_result is not None else np.nan
    return {
        "table": df.drop(columns=["result"], errors="ignore"),
        "best_params": {k: v for k, v in best_row.items() if k not in _RESULT_KEYS},
        "best_result": best_result,
        "best_metric": float(best_row[metric]) if np.isfinite(best_row[metric]) else np.nan,
        "n_trials": n,
        "deflated_sharpe": float(dsr) if np.isfinite(dsr) else None,
        "elapsed_sec": round(time.time() - t0, 2),
        "warning": _overfit_warning(n, best_row.get(metric, np.nan), dsr),
    }


_RESULT_KEYS = {
    "sharpe", "annual_return", "max_drawdown", "calmar", "sortino", "annual_vol",
    "n_trades", "result", "error", "rank",
}


def _first_param_combo(row: dict, run_keys: set | None = None) -> dict:
    """从结果行里剔除"评测函数输出的列"，返回纯参数。"""
    excl = set(_RESULT_KEYS) | set(run_keys or ()) | {"rank", "best"}
    return {k: v for k, v in row.items() if k not in excl}


def _overfit_warning(n_trials: int, best: float, dsr: float) -> str:
    if n_trials >= 20:
        sev = "严重"
    elif n_trials >= 5:
        sev = "中等"
    else:
        sev = "轻微"
    dsr_txt = f"{dsr:.3f}" if np.isfinite(dsr) else "n/a"
    return (
        f"你在 {n_trials} 组参数中选出了最优（{best if np.isfinite(best) else 'n/a'}），"
        f"多重检验偏误风险{sev}；Deflated Sharpe = {dsr_txt}。"
        f"请务必做样本外/滚动前推验证，不要直接用这个数字对外汇报。"
    )
