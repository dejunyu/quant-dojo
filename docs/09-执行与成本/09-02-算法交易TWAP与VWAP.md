<!--
id: 09-02
title: 算法交易TWAP与VWAP
category: 09-执行与成本
level: 进阶
order: 2
prereq: [09-01]
tags: [算法交易, TWAP, VWAP, 执行]
api: [qd.data.MarketData, qd.metrics.perf.perf_report]
-->

# 算法交易TWAP与VWAP

## 学完这一篇，你能做到

- 掌握 TWAP/VWAP/POV/IS 四类算法的目标函数与差异
- 能计算跟踪误差并理解它在算法选择中的作用
- 能按订单规模、流动性与紧迫性选择合适算法

## 本篇要点

- TWAP 按时间均匀切片，VWAP 按成交量分布加权
- POV 限定参与率 $\le\rho$，自适应于成交量，冲击可控但时间不确定
- IS（实现落差）以决策价 $P_{dec}$ 为基准，最小化总成本，兼顾机会成本

当单笔订单占标的日成交量的比例超过约 1%~5% 时，一次性下单会造成明显冲击。算法交易（algorithmic execution）把大单拆成许多小单，在时间与成交量上分散，以换取更低的冲击成本。本节讲四类最核心的算法及其选择逻辑。

## 一、从"你赚的钱，有一半消失在执行里"说起

1988 年，**André Perold** 在《组合管理杂志》发表了一篇论文，提出了一个后来成为行业标准的指标：**实现落差**（Implementation Shortfall，简称 IS）。

他问了一个非常朴素、但以前没人认真量化的问题：

> **"我的策略说'买这只股票'，到我真正买完，中间到底损失了多少？"**

这个问题的难点在于：**你得先定义"原本应该成交在什么价"**。

Perold 的答案是：**用"决策那一刻的价格"作为基准**（也就是你脑子里冒出"买"这个念头时的价格）。然后，从那一刻到你实际成交，所有的差距都算成本——包括：

| 损失 | 是什么 |
| --- | --- |
| **延迟成本** | 从"决定"到"开始执行"，价格已经动了 |
| **执行成本** | 执行过程中，你的单子把价格推走了 |
| **未成交的机会成本** | 没买到的部分，错过了收益 |

> **这个框架的厉害之处，是它把"我看不见的损失"变成了"可以归因的三个数"。**
>
> 因为在它之前，人们只算"佣金和税费"——那是**显性成本**，看得见。
> 而 Perold 指出：**真正贵的是那些看不见的部分**（03-04 讲过，隐性成本才是决定生死的）。

**顺便说个 VWAP 的起源**：据 Abel Noser 的总裁回忆，**1984 年**出现了第一笔"以 VWAP 为基准"的执行——那是纽约一家代理经纪商为福特汽车养老金做的，**当时还是纯人工拆单，没有任何算法**。1988 年，Berkowitz、Logue 与 Noser 在《金融学杂志》上把 VWAP 正式用作衡量 NYSE 交易成本的指标。

（⚠️ 这条来自**单一行业来源**（公司高管口述），引用时宜写"据 Abel Noser 总裁回忆"，别写成学界公认。）

> 这也正是本章的出发点：
> **执行不是一个"下单"动作，它是一段有成本、可度量、可优化的过程。**
> 而 TWAP / VWAP 这些算法，就是在"怎么把这段过程的成本压到最低"这个问题上的几种答案。

---

## 二、统一的语言：基准与成本

执行算法的效果必须相对某个**基准价**评价。常用基准：

- **决策价 $P_{dec}$**：下单瞬间（信号产生时）的中间价。用它定义的成本叫**实现落差（Implementation Shortfall, IS）**，由 Perold（1988）提出，是最完整的成本定义。
- **到达价 $P_{arr}$**：算法开始执行时的价格。IS 可分为：

$$\text{IS} = \underbrace{(P_{arr}-P_{dec})}_{\text{延迟成本}} + \underbrace{(\bar P_{exec}-P_{arr})}_{\text{执行成本}} + \underbrace{\text{未成交的机会成本}}_{}$$

其中 $\bar P_{exec}$ 是实际成交均价。注意符号按买入方向定义，$\text{IS}>0$ 表示成本为正。

- **VWAP 基准**：当日成交量加权均价。用它评价的算法叫 VWAP 算法。
- **收盘价 / 开盘价**：被动型、指数跟踪型策略常用。

**跟踪误差（tracking error）**定义算法执行均价与基准价之差的波动：

$$TE = \text{std}\big(\bar P_{exec} - P_{benchmark}\big)$$

跟踪误差越小，执行结果越可预测，但通常意味着算法越"刚性"、冲击越大。这是执行的核心权衡。

## 三、四类算法

### 1. TWAP（Time Weighted Average Price）

把总量 $Q$ 在 $[0,T]$ 内切成 $N$ 等份，每份 $Q/N$，每隔 $\Delta t=T/N$ 下一单。

$$\bar P_{TWAP} = \frac{1}{N}\sum_{k=1}^{N} P_{t_k}$$

- 优点：实现最简单、行为最可预测、不依赖成交量预测。
- 缺点：**无视成交量的日内分布**。实际市场的成交量呈 U 型（开盘与收盘高、午间低），在清淡时段用同样的量会造成不必要的冲击；在活跃时段则执行不足。
- 适用：流动性均匀、无可靠日内量分布、或需要行为隐蔽（TWAP 比 VWAP 更难被识别）。

### 2. VWAP（Volume Weighted Average Price）

按预测的日内成交量分布 $v(t)$ 分配：

$$q(t) = Q\cdot\frac{v(t)}{\int_0^T v(u)du},\qquad P_{VWAP}=\frac{\sum_t P_t V_t}{\sum_t V_t}$$

- 优点：顺应流动性，在量大的时段多成交，冲击最小；基准即 VWAP，跟踪误差天然小。
- 缺点：**依赖成交量预测的准确性**。用历史平均分布会在异常日（放量/缩量）显著偏离；且 VWAP 算法的行为模式容易被识别（"VWAP 跟随者"会被抢跑）。
- 适用：大单、流动性充足、对 VWAP 基准考核的场景。

### 3. POV（Percentage of Volume）/ 参与率算法

维持参与率 $\rho$（如 10%）：每个时间片的成交量 $q_t=\rho\cdot V_t$，随实际成交量自适应。

- 优点：天然适应放量/缩量；参与率上限直接控制冲击（见 [冲击成本模型](09-03-冲击成本模型.md)）。
- 缺点：**完成时间不确定**。若当日缩量，订单可能做不完，产生机会成本。
- 适用：不急于完成、希望严格控冲击的长期调仓。

### 4. IS / Implementation Shortfall 算法

以最小化 IS 总成本为目标，在**冲击成本**与**时机风险（timing risk）**之间求最优：

$$\min_{q(\cdot)}\ \underbrace{\eta\int_0^T \frac{q(t)^2}{V(t)}dt}_{\text{冲击成本}} + \underbrace{\lambda\,\sigma\sqrt{\int_0^T q_{remain}(t)^2 dt}}_{\text{时机风险}}$$

这本质上是 Almgren-Chriss 最优执行框架的连续形式。$\lambda$ 越大（越厌恶风险），算法越激进（早日完成）；$\lambda$ 越小，越被动（慢慢做）。

- 优点：唯一把机会成本纳入目标的算法，理论上最优。
- 缺点：依赖波动率与冲击系数估计；参数敏感。
- 适用：有明确 alpha 衰减预期的订单（信号会过期就必须快做）。

## 四、什么时候用哪个

| 维度 | TWAP | VWAP | POV | IS |
| --- | --- | --- | --- | --- |
| 需要的信息 | 时间 | 日内量分布 | 实时成交量 | 波动率、冲击系数 |
| 完成时间 | 确定 | 确定 | **不确定** | 确定（可设） |
| 冲击控制 | 一般 | 好 | 最好（有上限） | 可调 |
| 跟踪误差（对 VWAP） | 中高 | 最低 | 中 | 中 |
| 机会成本 | 中 | 中 | 高 | 最低 |
| 隐蔽性 | 好 | 差 | 中 | 好 |

选择规则：

1. **订单占 ADV（日均成交量）< 1%**：不需要算法，直接下（或用简单 [限价单](09-01-订单类型.md)）。
2. **1%~10% ADV**：VWAP 或 TWAP，半天到一天完成。
3. **> 10% ADV**：POV（参与率 10%~15%）+ 跨日执行，或拆到多个交易日。
4. **信号衰减快**（如日内动量）：IS 算法，前置成交量。
5. **无 alpha、纯被动调仓**（如 [再平衡](../08-组合与风险/08-08-再平衡与调仓节奏.md)）：POV，最省冲击。

## 五、代码示例

```python
import numpy as np
import pandas as pd
from qd.data.synthetic import generate_demo_market
from qd.data import MarketData

md = generate_demo_market(n_symbols=10, start="2018-01-01", end="2024-12-31", seed=7)
close, vol = md.close(), md.volume()

sym = md.symbols[0]
v = vol[sym].dropna()
# 1) 由历史数据估计日内 U 型成交量分布（此处以"每月第几个交易日"作为演示切片维度，
#    实盘中应换成分钟级日内分布）
intraday_profile = v.groupby(v.index.dayofweek).mean()
intraday_profile = intraday_profile / intraday_profile.sum()
print("成交量分布:\n", intraday_profile.round(3))

# 2) TWAP 切片：等份
Q = 100_000
n_slice = 8
twap_slices = np.repeat(Q / n_slice, n_slice)

# 3) VWAP 切片：按成交量分布加权
p = intraday_profile.reindex(range(5)).fillna(0.2).values
p = p / p.sum()
vwap_slices = Q * np.repeat(p, 1)[:n_slice] if len(p) >= n_slice else Q * np.resize(p, n_slice)
vwap_slices = vwap_slices / vwap_slices.sum() * Q
print("TWAP 切片:", twap_slices.round(0))
print("VWAP 切片:", vwap_slices.round(0))

# 4) POV：给定参与率，按实时成交量生成子单
def pov_child_orders(market_volume, rho=0.10, cap=0.15):
    """参与率算法：每片成交量 = rho * 市场成交量，单片不超过市场量的 cap。"""
    return np.minimum(rho * market_volume, cap * market_volume)

# 5) 跟踪误差：算法均价 vs VWAP 基准
vwap_bench = float((close[sym] * vol[sym]).dropna().sum() / vol[sym].dropna().sum())
twap_bench = float(close[sym].dropna().mean())
print("VWAP 基准价 %.2f  TWAP 基准价 %.2f  跟踪误差 %.3f"
      % (vwap_bench, twap_bench, abs(twap_bench - vwap_bench) / vwap_bench * 1e4), "bps")
```

## 常见坑

- **用历史平均量分布预测当日**：个股放量/缩量的方差极大，VWAP 预测误差是主要误差源。应使用同行业、同市值、同日的实时修正（intraday volume curve update）。
- **把 VWAP 达成率当绩效指标**：跑赢 VWAP 可能是因为你在下跌日少买了，属于运气。要用多日平均、分市场状态统计。
- **忽略收盘集合竞价**：A股收盘前 3 分钟（创业板/科创板为最后 3 分钟集合竞价）无法连续交易，算法需提前结束或专门处理。
- **参与率过高**：$\rho>25\%$ 时冲击成本非线性上升，且极易被识别。
- **算法与信号周期不匹配**：用一整天执行一个 30 分钟就衰减的信号，等于白做。
- **回测里不建模切片**：见 [实盘延迟与系统](09-06-实盘延迟与系统.md)，回测必须假设次日或延后成交。

## 延伸阅读

- Robert Almgren & Neil Chriss (2001), *Optimal Execution of Portfolio Transactions*, Journal of Risk
- Robert Almgren (2003), *Optimal Execution with Nonlinear Impact Functions and Trading-Enhanced Risk*, Applied Mathematical Finance
- André Perold (1988), *The Implementation Shortfall: Paper versus Reality*, Journal of Portfolio Management
- Robert Kissell & Morton Glantz (2003), *Optimal Trading Strategies*, AMACOM
- Robert Kissell (2013), *The Science of Algorithmic Trading and Portfolio Management*, Academic Press

## 动手练习

- 用日内成交量数据画出 U 型分布，构造 VWAP 切片表并与 TWAP 对比
- 计算不同算法相对 VWAP 基准的跟踪误差，看参与率如何影响它
