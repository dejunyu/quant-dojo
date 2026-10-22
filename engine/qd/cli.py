"""命令行入口 ``qd``.

```bash
qd serve                      # 启动 Web 平台
qd factors                    # 列出全部因子
qd factor mom_12_1 --report   # 单因子完整研究报告
qd backtest dual_momentum     # 跑一个策略
qd grid dual_momentum         # 参数搜索 + 过拟合警告
qd wf dual_momentum           # 滚动前推
qd overfit                    # 白噪声过拟合实验
qd lessons                    # 列出课程
qd lesson 05-05               # 读一篇课程
qd demo                       # 一键跑通全流程演示
```
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import Backtest, BacktestConfig
from .data.synthetic import generate_demo_market
from .factors import get_factor, list_factors
from .optimize import grid_search, walk_forward
from .research.lab import FactorLab
from .research.overfit import OverfitLab
from .strategy import builtin as S
from .tutor import load_curriculum

__all__ = ["main"]

_FACTORIES = {
    "buy_and_hold": S.buy_and_hold, "equal_weight": S.equal_weight,
    "sma_cross": S.sma_cross, "dual_momentum": S.dual_momentum,
    "turtle_breakout": S.turtle_breakout, "bollinger_reversion": S.bollinger_reversion,
    "mean_reversion_basket": S.mean_reversion_basket, "risk_parity": S.risk_parity,
    "vol_target": S.vol_target, "rotation": S.rotation,
    "factor_long_only": S.factor_long_only, "multi_factor_rank": S.multi_factor_rank,
}

_DEFAULTS = {
    "dual_momentum": {"lookback": 120, "top_k": 10},
    "sma_cross": {"fast": 20, "slow": 60},
    "risk_parity": {"lookback": 120},
    "bollinger_reversion": {"window": 20, "top_k": 10},
    "turtle_breakout": {"entry_window": 55, "exit_window": 20},
    "mean_reversion_basket": {"lookback": 5, "top_k": 10},
    "rotation": {"top_k": 3, "lookback": 60},
    "vol_target": {"target_vol": 0.15},
    "factor_long_only": {"factors": ("mom_12_1", "bp"), "top_k": 15},
    "multi_factor_rank": {"factors": ("mom_12_1", "ep"), "top_k": 15},
}


def _md(n_symbols: int = 50, start: str = "2018-01-01", end: str = "2024-12-31"):
    return generate_demo_market(n_symbols=n_symbols, start=start, end=end, seed=7)


def _print(title: str = "", char: str = "=") -> None:
    if title:
        print(f"\n{char * 72}\n{title}\n{char * 72}")


# ------------------------------------------------------------------ 子命令
def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("qd.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_factors(args) -> int:
    fs = list_factors()
    if args.category:
        fs = [f for f in fs if f["category"] == args.category]
    _print(f"因子库：{len(fs)} 个因子")
    by_cat: dict[str, list] = {}
    for f in fs:
        by_cat.setdefault(f["category"], []).append(f)
    for cat in sorted(by_cat):
        print(f"\n【{cat}】")
        for f in by_cat[cat]:
            d = "↑" if f["direction"] > 0 else "↓"
            print(f"  {d} {f['name']:22s} {f['description'][:44]}")
    return 0


def cmd_factor(args) -> int:
    md = _md()
    lab = FactorLab(md, horizon=args.horizon)
    if args.report:
        rep = lab.report(args.name, horizons=(1, 5, 20, 60))
        ic = rep["ic"]
        qb = rep["quantile_backtest"]
        _print(f"因子报告：{rep['factor']}  [{rep['category']}]  方向 {'正向' if rep['direction']>0 else '反向'}")
        print(f"说明：{rep['description']}")
        if rep["formula"]:
            print(f"公式：{rep['formula']}")
        print(f"\n覆盖率：平均 {rep['coverage']['avg_coverage']:.0%}  最低 {rep['coverage']['min_coverage']:.0%}")
        print(f"IC 均值 {ic['ic_mean']:+.4f}   ICIR {ic['icir']:+.3f}   t {ic['t_stat']:+.2f}   IC>0 占比 {ic['ic_positive_ratio']:.0%}")
        print("\nIC 衰减：")
        for row in rep["ic_decay"]:
            print(f"  {row['horizon']:4d} 日:  IC {row['ic']:+.4f}  ICIR {row['icir']:+.3f}  t {row['t_stat']:+.2f}")
        if "groups" in qb:
            print("\n分层回测（5 组年化）：")
            for q, v in qb["groups"].items():
                bar = "█" * max(0, int(v["annual_return"] * 60))
                print(f"  {q}: {v['annual_return']:+.2%}  夏普 {v['sharpe']:+.2f}  {bar}")
            ls = qb["long_short"]
            print(f"\n多空：年化 {ls['annual_return']:+.2%}  夏普 {ls['sharpe']:+.2f}  最大回撤 {ls['max_drawdown']:.2%}")
            print(f"单调性：{qb['monotonicity']:+.2f}（1=完美单调，-1=完全反向）")
        tn = rep["turnover"]
        print(f"\n换手：日均 {tn['avg_daily_turnover']:.2%}  年化 {tn['annual_turnover']:.1f}×")
        _print("结论", "-")
        print(f"{rep['verdict']}")
        for r in rep["reasons"]:
            print(f"  · {r}")
    else:
        s = lab.ic_series(args.name)
        print(f"{args.name}: IC 均值 {s.mean():+.4f}  ICIR {s.mean()/s.std(ddof=1):+.3f}")
    return 0


def cmd_backtest(args) -> int:
    md = _md(args.n_symbols)
    factory = _FACTORIES.get(args.strategy)
    if factory is None:
        print(f"未知策略 {args.strategy}；可选：{sorted(_FACTORIES)}")
        return 1
    params = _DEFAULTS.get(args.strategy, {})
    if args.params:
        params.update(json.loads(args.params))
    cfg = BacktestConfig(
        initial_cash=args.cash, benchmark=args.benchmark,
        min_trade_value=args.min_trade_value,
    )
    t0 = time.time()
    res = Backtest(md, factory(**params), cfg).run()
    _print(f"回测：{args.strategy}  {params}  ({time.time()-t0:.1f}s)")
    print(res.summary())
    if args.trades:
        _print("最近交易", "-")
        print(res.trades.tail(args.trades).to_string(index=False))
    return 0


def cmd_grid(args) -> int:
    md = _md(args.n_symbols)
    factory = _FACTORIES.get(args.strategy)
    if factory is None:
        print(f"未知策略 {args.strategy}"); return 1
    grid = json.loads(args.grid) if args.grid else _default_grid(args.strategy)
    out = grid_search(md, factory, grid, metric=args.metric, verbose=True)
    _print(f"参数搜索：{args.strategy}  {out['n_trials']} 组  ({out['elapsed_sec']}s)")
    tbl = out["table"]
    cols = list(grid) + [args.metric, "annual_return", "max_drawdown", "n_trades"]
    cols = [c for c in cols if c in tbl.columns]
    print(tbl[cols].head(args.top).to_string(index=False))
    print(f"\n最优参数：{out['best_params']}   {args.metric}={out['best_metric']:.3f}")
    print(f"Deflated Sharpe：{out['deflated_sharpe']}")
    print(f"\n⚠️  {out['warning']}")
    return 0


def cmd_wf(args) -> int:
    md = _md(args.n_symbols)
    factory = _FACTORIES.get(args.strategy)
    if factory is None:
        print(f"未知策略 {args.strategy}"); return 1
    grid = json.loads(args.grid) if args.grid else _default_grid(args.strategy)
    out = walk_forward(md, factory, param_grid=grid, n_splits=args.splits, verbose=True)
    _print(f"滚动前推：{args.strategy}  {out['n_folds']} 段")
    print(out["folds"].to_string(index=False))
    print(f"\n拼接后样本外：夏普 {out['sharpe']:.3f}  年化 {out['annual_return']:.2%}  最大回撤 {out['max_drawdown']:.2%}")
    print(out["note"])
    return 0


def cmd_overfit(args) -> int:
    md = _md(args.n_symbols)
    factory = _FACTORIES.get(args.strategy)
    if factory is None:
        print(f"未知策略 {args.strategy}"); return 1
    grid = json.loads(args.grid) if args.grid else _default_grid(args.strategy)
    lab = OverfitLab(md)
    out = lab.white_noise_experiment(grid, factory, seed=args.seed)
    _print("白噪声过拟合实验")
    for k, v in out.items():
        if k != "best_params":
            print(f"  {k}: {v}")
    print(f"  best_params: {out.get('best_params')}")
    return 0


def cmd_lessons(args) -> int:
    c = load_curriculum()
    st = c.stats()
    _print(f"课程体系：{st['n_lessons']} 篇 / {st['n_categories']} 章 / {st['total_words']:,} 字（约 {st['total_reading_minutes']} 分钟）")
    for cat in c.categories():
        print(f"\n【{cat['id']}】{cat['n_lessons']} 篇  {cat['word_count']:,} 字")
        if args.detail:
            for l in cat["lessons"]:
                print(f"  {l['id']}  {l['title']}  ({l['level']}, {l['reading_minutes']} 分钟)")
    return 0


def cmd_lesson(args) -> int:
    c = load_curriculum()
    lesson = c.get(args.id)
    if not lesson:
        print(f"课程 {args.id} 不存在"); return 1
    if args.body:
        print(lesson.body)
    else:
        print(f"{lesson.id}  {lesson.title}  [{lesson.category}/{lesson.level}]")
        print(f"字数 {lesson.word_count}  阅读约 {lesson.reading_minutes} 分钟")
        print(f"\n学习目标：")
        for o in lesson.objectives:
            print(f"  · {o}")
        print(f"\n核心要点：")
        for k in lesson.key_points:
            print(f"  · {k}")
        if lesson.exercises:
            print(f"\n练习：")
            for e in lesson.exercises:
                print(f"  · {e}")
        if lesson.prereq:
            print(f"\n先修：{', '.join(lesson.prereq)}")
    return 0


def cmd_search(args) -> int:
    c = load_curriculum()
    for r in c.search(args.query, limit=args.limit):
        print(f"{r['id']}  {r['title']}  [{r['category']}]")
    return 0


def cmd_demo(args) -> int:
    """一键跑通全流程，用于验证安装是否正确。"""
    _print("Quant Dojo 全流程演示", "#")
    t0 = time.time()
    print("[1/7] 生成合成数据 ...")
    md = _md(n_symbols=args.n_symbols)
    print(f"      {md}")

    print("[2/7] 加载课程体系 ...")
    c = load_curriculum()
    print(f"      {len(c)} 篇课程，{c.stats()['total_words']:,} 字")

    print("[3/7] 因子库 ...")
    print(f"      {len(list_factors())} 个因子")

    print("[4/7] 单因子检验 mom_12_1 ...")
    lab = FactorLab(md, horizon=20)
    rep = lab.report("mom_12_1", horizons=(1, 5, 20))
    print(f"      IC {rep['ic']['ic_mean']:+.4f}  t {rep['ic']['t_stat']:+.2f}  "
          f"多空夏普 {rep['quantile_backtest']['long_short']['sharpe']:+.2f}")
    print(f"      {rep['verdict']}")

    print("[5/7] 回测 dual_momentum（对比不同调仓频率）...")
    for freq, label in [("daily", "日频"), ("monthly", "月频")]:
        res = Backtest(md, S.dual_momentum(lookback=120, top_k=10),
                       BacktestConfig(rebalance=freq)).run()
        print(f"      {label}：夏普 {res.metrics['sharpe']:+.2f}  年化 {res.metrics['annual_return']:+.2%}  "
              f"最大回撤 {res.metrics['max_drawdown']:.2%}  成交 {len(res.fills)} 笔")
    print("      ↑ 同一策略、同一数据，只因调仓频率不同，日频被交易成本吃成负收益。")

    print("[6/7] 滚动前推 ...")
    wf = walk_forward(md, S.dual_momentum, {"lookback": [60, 120], "top_k": [5, 10]},
                      n_splits=4, verbose=False)
    print(f"      样本外夏普 {wf['sharpe']:.2f}（{wf['n_folds']} 段）")

    print("[7/7] 白噪声过拟合实验 ...")
    of = OverfitLab(md).white_noise_experiment(
        {"lookback": [20, 60, 120], "top_k": [3, 5, 10]}, S.dual_momentum, seed=0)
    print(f"      零 alpha 数据上：夏普均值 {of['sharpe_mean']:.2f}，最优被搜索抬到 "
          f"{of['best_in_sample_sharpe']:.2f}（凭空多出 {of['search_lift']:.2f}）")
    print(f"      {of['lesson']}")

    _print(f"全部通过，总耗时 {time.time()-t0:.1f}s", "#")
    print("\n启动 Web 平台：qd serve   然后打开 http://127.0.0.1:8000")
    return 0


def _default_grid(strategy: str) -> dict:
    return {
        "dual_momentum": {"lookback": [60, 120, 250], "top_k": [5, 10, 20]},
        "sma_cross": {"fast": [5, 10, 20], "slow": [30, 60, 120]},
        "bollinger_reversion": {"window": [10, 20, 40], "top_k": [5, 10]},
        "mean_reversion_basket": {"lookback": [3, 5, 10], "top_k": [5, 10]},
        "risk_parity": {"lookback": [60, 120, 250]},
        "turtle_breakout": {"entry_window": [20, 55, 90], "n": [5, 10]},
        "rotation": {"lookback": [20, 60, 120], "top_k": [2, 3, 5]},
    }.get(strategy, {"lookback": [60, 120], "top_k": [5, 10]})


# ------------------------------------------------------------------ 入口
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qd", description="量化投资学习与实战平台")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("serve", help="启动 Web 平台")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--reload", action="store_true")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("factors", help="列出因子库")
    sp.add_argument("--category", default=None)
    sp.set_defaults(func=cmd_factors)

    sp = sub.add_parser("factor", help="单因子研究")
    sp.add_argument("name")
    sp.add_argument("--report", action="store_true")
    sp.add_argument("--horizon", type=int, default=20)
    sp.set_defaults(func=cmd_factor)

    sp = sub.add_parser("backtest", help="跑回测")
    sp.add_argument("strategy")
    sp.add_argument("--params", default=None, help='JSON，如 \'{"lookback":60}\'')
    sp.add_argument("--cash", type=float, default=1_000_000)
    sp.add_argument("--benchmark", default=None)
    sp.add_argument("--n-symbols", type=int, default=50)
    sp.add_argument("--min-trade-value", type=float, default=0.005)
    sp.add_argument("--trades", type=int, default=0)
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("grid", help="参数网格搜索")
    sp.add_argument("strategy")
    sp.add_argument("--grid", default=None, help='JSON，如 \'{"lookback":[60,120]}\'')
    sp.add_argument("--metric", default="sharpe")
    sp.add_argument("--top", type=int, default=10)
    sp.add_argument("--n-symbols", type=int, default=50)
    sp.set_defaults(func=cmd_grid)

    sp = sub.add_parser("wf", help="滚动前推验证")
    sp.add_argument("strategy")
    sp.add_argument("--grid", default=None)
    sp.add_argument("--splits", type=int, default=5)
    sp.add_argument("--n-symbols", type=int, default=50)
    sp.set_defaults(func=cmd_wf)

    sp = sub.add_parser("overfit", help="白噪声过拟合实验")
    sp.add_argument("strategy")
    sp.add_argument("--grid", default=None)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--n-symbols", type=int, default=50)
    sp.set_defaults(func=cmd_overfit)

    sp = sub.add_parser("lessons", help="列出课程")
    sp.add_argument("--detail", action="store_true")
    sp.set_defaults(func=cmd_lessons)

    sp = sub.add_parser("lesson", help="读一篇课程")
    sp.add_argument("id")
    sp.add_argument("--body", action="store_true")
    sp.set_defaults(func=cmd_lesson)

    sp = sub.add_parser("search", help="搜索课程")
    sp.add_argument("query")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("demo", help="一键全流程演示")
    sp.add_argument("--n-symbols", type=int, default=50)
    sp.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
