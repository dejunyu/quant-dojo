"""生成博客文章与 README 用的调仓频率对比图."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from qd.backtest import Backtest, BacktestConfig
from qd.data.synthetic import generate_demo_market
from qd.strategy.builtin import dual_momentum

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"


def main() -> None:
    md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
    strat = dual_momentum(lookback=120, top_k=10)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})

    # 左图：净值曲线对比
    ax = axes[0]
    colors = {"daily": "#f85149", "monthly": "#3fb950"}
    for freq, label in (("daily", "日频调仓"), ("monthly", "月频调仓")):
        r = Backtest(md, strat, BacktestConfig(rebalance=freq)).run()
        eq = r.equity
        ax.plot(eq.index, eq.values, label=f"{label}（年化 {r.metrics['annual_return']:+.1%}，"
                                           f"成交 {len(r.fills):,} 笔）",
                color=colors[freq], linewidth=1.4)
    ax.axhline(1.0, color="#9aa7b4", linewidth=0.8, linestyle="--")
    ax.set_title("同一策略，只改调仓频率", fontsize=12)
    ax.set_ylabel("净值")
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(alpha=0.25)

    # 右图：收益 vs 成本的瀑布式对比
    ax = axes[1]
    stats = {}
    for freq in ("daily", "monthly"):
        r = Backtest(md, strat, BacktestConfig(rebalance=freq)).run()
        cb = r.cost_breakdown()
        stats[freq] = {
            "cum": r.metrics["cumulative_return"],
            "cost": cb["total_cost"] / 1_000_000,   # 占初始资金比例
            "n": len(r.fills),
        }
    labels = ["累计收益", "交易成本", "成交笔数(百)"]
    daily = [stats["daily"]["cum"], stats["daily"]["cost"], stats["daily"]["n"] / 100]
    monthly = [stats["monthly"]["cum"], stats["monthly"]["cost"], stats["monthly"]["n"] / 100]
    x = range(len(labels))
    w = 0.36
    ax.bar([i - w / 2 for i in x], daily, w, label="日频", color="#f85149")
    ax.bar([i + w / 2 for i in x], monthly, w, label="月频", color="#3fb950")
    for i, v in enumerate(daily):
        ax.text(i - w / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(monthly):
        ax.text(i + w / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("收益与成本对比（初始资金 100 万）", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "rebalance-frequency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"已生成 {out}")
    for f, s in stats.items():
        print(f"  {f}: cum={s['cum']:+.2%} cost={s['cost']:.2%} fills={s['n']}")


if __name__ == "__main__":
    main()
