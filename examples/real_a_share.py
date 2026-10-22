"""真实 A 股数据端到端示例.

用法：
    pip install akshare
    python examples/real_a_share.py

需要联网。数据会缓存到 storage/cache/（已在 .gitignore 中排除）。

诚实声明（写给每一位想直接抄收益数字的人）：
    本示例的样本只有 10 只 2026 年知名的 A 股——这是幸存者偏差样本；
    横截面太小，IC 与分层结果的噪声极大；没有做行业/市值中性化；
    样本期 2021-2024 没有覆盖极端行情。所以下面的所有数字都只说明
    「流程通了」，不构成任何投资建议，也不代表策略能赚钱。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 让示例在未 pip install 的情况下也能直接跑（engine/ 在仓库根）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

# 国内数据源（东方财富等）不需要也不应该走代理。
# macOS 的「系统代理」会被 requests 自动读取，代理软件一断，
# akshare 就会报 ProxyError——这是国内用户最常见的坑。
# 这里显式清空代理环境，强制直连国内源。
import os

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
           "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"

import pandas as pd

from qd.backtest import Backtest, BacktestConfig
from qd.data.market import MarketData
from qd.metrics.perf import sharpe
from qd.optimize import walk_forward
from qd.research.lab import FactorLab
from qd.strategy.builtin import dual_momentum

# ------------------------------------------------------------------ 配置
SYMBOLS = {
    "600519": "贵州茅台", "000858": "五粮液", "601318": "中国平安",
    "600036": "招商银行", "300750": "宁德时代", "600276": "恒瑞医药",
    "000002": "万科A", "603288": "海天味业", "600900": "长江电力",
    "600887": "伊利股份",
}
START, END = "2021-01-01", "2024-12-31"
CACHE_DIR = Path(__file__).resolve().parents[1] / "storage" / "cache"


def fetch_data() -> MarketData:
    """拉取 10 只真实 A 股的前复权日线并组装成 MarketData。"""
    from qd.data.akshare_feed import AkshareFeed

    feed = AkshareFeed(cache_dir=CACHE_DIR, adjust="qfq", cache_days=30,
                       pause=1.5, retries=4)
    print("==================================================================")
    print("第 1 步：拉取真实数据（akshare, 前复权）")
    print("==================================================================")
    t0 = time.time()
    md = feed.load(list(SYMBOLS), START, END, min_days=200)
    md.meta["display_names"] = SYMBOLS
    print(f"✓ 拉取完成：{md.n_symbols} 只标的，{md.n_dates} 个交易日，"
          f"耗时 {time.time() - t0:.1f}s")
    if md.meta.get("failed"):
        print(f"⚠ 下载失败的标的（已剔除）：{[s for s, _ in md.meta['failed']]}")
    return md


def factor_ic(lab: FactorLab, names: list[str]) -> None:
    print("\n" + "=" * 66)
    print("第 2 步：因子 IC 检验（真实数据，horizon=20 交易日）")
    print("=" * 66)
    print(f"{'因子':<10} {'IC均值':>8} {'ICIR':>8} {'t统计':>8} {'IC>0占比':>9}   判定")
    for name in names:
        rep = lab.report(name, horizons=(20,))
        ic = rep["ic"]
        verdict = rep["verdict"].split("：")[0]
        print(f"{name:<10} {ic['ic_mean']:>+8.4f} {ic['icir']:>+8.2f} "
              f"{ic['t_stat']:>+8.2f} {ic['ic_positive_ratio']:>9.1%}   {verdict}")


def out_of_sample(md: MarketData) -> dict:
    """前 70% 选参、后 30% 检验——最朴素的样本外设计。"""
    print("\n" + "=" * 66)
    print("第 3 步：样本外回测（前 70% 选参，后 30% 检验）")
    print("=" * 66)
    wf = walk_forward(
        md, dual_momentum,
        param_grid={"lookback": [60, 120, 250], "top_k": [3, 5]},
        n_splits=3,          # 3 段滚动：每段的参数都来自它之前的数据
        train_ratio=0.5,
        verbose=False,
    )
    folds = wf["folds"]
    print(folds[["test_start", "test_end", "params", "sharpe"]].to_string(index=False))
    print(f"\n样本外拼接：夏普 {wf['sharpe']:+.3f}  年化 {wf['annual_return']:+.2%}  "
          f"最大回撤 {wf['max_drawdown']:.2%}")
    print(f"（每一段用的都是该段之前选出的参数，因此不含参数前视）")
    return wf


def rebalance_cost(md: MarketData, best_params: dict) -> None:
    print("\n" + "=" * 66)
    print("第 4 步：调仓频率的成本影响（同一样本外区间）")
    print("=" * 66)
    for freq, label in [("daily", "daily  "), ("monthly", "monthly")]:
        res = Backtest(md, dual_momentum(**best_params),
                       BacktestConfig(rebalance=freq)).run()
        m = res.metrics
        print(f"{label}  年化 {m['annual_return']:+7.2%}   "
              f"夏普 {m['sharpe']:+6.2f}   成交 {len(res.fills):5d} 笔   "
              f"成本 {res.cost_breakdown().get('total_cost', 0):,.0f} 元")


def main() -> None:
    md = fetch_data()
    lab = FactorLab(md, horizon=20)

    # 第 2 步：因子检验。挑 3 个逻辑各异的因子：动量/反转/低波
    # （腾讯源的日线不带 pe/pb 等估值字段，所以这里不跑 ep/bp——
    #   想跑估值因子请用 eastmoney 源或自行拼接财务数据）
    factor_ic(lab, ["mom_12_1", "rev_20d", "vol_60d"])

    # 第 3 步：样本外
    wf = out_of_sample(md)

    # 第 4 步：成本影响（用样本外第一段选出的参数）
    best = wf["folds"].iloc[0]["params"] if len(wf["folds"]) else {"lookback": 120, "top_k": 3}
    rebalance_cost(md, best)

    # 收尾：把局限再说一遍（这不是客套，是研究纪律）
    print("\n" + "=" * 66)
    print("⚠ 重要局限（详见 examples/README.md）")
    print("=" * 66)
    print("""
1. 幸存者偏差：这 10 只是 2026 年的知名公司，2021 年未必会选它们，
   期间退市/暴雷的标的不在样本里 → 收益被系统性高估。
2. 横截面太小：10 只股票撑不起因子统计，正经研究需要全市场几千只。
3. 无行业/市值中性化：因子「有效」可能只是押对了行业。
4. 样本期固定：未覆盖 2015/2018 级别极端行情，回撤被低估。

结论：本示例的价值在于流程——换数据、换因子、换策略，检验流程不变。
数字本身不构成任何投资建议。""")


if __name__ == "__main__":
    main()
