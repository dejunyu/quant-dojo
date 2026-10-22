---
id: 08-06
title: 风险指标VaR与CVaR
category: 08-组合与风险
level: 进阶
order: 6
prereq: []
tags: [风险指标, VaR, 尾部风险]
objectives:
  - 准确写出 VaR 与 CVaR 的定义并说明二者关系
  - 能用历史模拟、参数法、蒙特卡洛三种方法计算 VaR
  - 知道 VaR 不满足次可加性，以及为什么要改用 CVaR
key_points:
  - $VaR_\alpha=\inf\{l:P(L>l)\le 1-\alpha\}$，只给出阈值不给尾部深度
  - $CVaR_\alpha=\mathbb{E}[L\mid L\ge VaR_\alpha]$，是**一致风险度量**
  - VaR 不满足次可加性：$VaR(A+B)$ 可能大于 $VaR(A)+VaR(B)$
exercises:
  - 用同一组合分别以历史法/正态法/t 分布法算 99% VaR，比较差异
  - 构造两个厚尾分布的组合，验证 VaR 违反次可加性而 CVaR 不违反
api:
  - qd.metrics.perf.value_at_risk
  - qd.metrics.perf.conditional_var
---

# 风险指标VaR与CVaR

VaR（Value at Risk，在险价值）是过去三十年金融机构最常用的风险数字，也是被批评最多的。理解它的定义、算法与缺陷，是风险管理入门的必修课。

## 一、定义

给定置信水平 $\alpha$（如 0.99）与持有期 $h$（如 1 天），令 $L$ 为该期间的**损失**（正数为亏）。VaR 是这样一个阈值：

$$VaR_\alpha = \inf\big\{l:\ P(L>l)\le 1-\alpha\big\}$$

等价写法：$VaR_\alpha = F_L^{-1}(\alpha)$，即损失分布的 $\alpha$ 分位数。它的通俗表述是"在 $\alpha$ 的日子里，损失不会超过 $VaR_\alpha$"。

**CVaR**（Conditional VaR，又称 Expected Shortfall、Expected Tail Loss）：

$$CVaR_\alpha = \mathbb{E}\big[L \mid L \ge VaR_\alpha\big]$$

即"一旦真的突破 VaR，平均会亏多少"。

二者关系：$CVaR_\alpha = \frac{1}{1-\alpha}\int_\alpha^1 VaR_u\,du$，所以 CVaR 是 VaR 在尾部区间上的平均，恒有 $CVaR_\alpha \ge VaR_\alpha$。

举例：某组合 99% 单日 VaR = 200 万，99% CVaR = 450 万。含义是：每 100 天里大约有 1 天亏损超过 200 万，而那些"坏日子"的平均亏损是 450 万。**只看 VaR 会严重低估风险**，这正是 2008 年后监管（巴塞尔 III FRTB）把内部模型从 VaR 换成 ES 的原因。

## 二、三种计算方法

### 1. 历史模拟法

直接用过去 $T$ 天的实际损益序列，取经验分位数：

$$VaR_\alpha = -\text{quantile}_\alpha\big(\{r_t\}_{t=1}^{T}\big) \times V$$

- 优点：无需分布假设，自动保留厚尾、偏态与相关性结构。
- 缺点：完全依赖历史样本，**无法预测从未出现过的事件**；样本尾部点极少（99% VaR 在 250 天样本里只用到了第 2.5 个最坏观测）；窗口选择主观。

### 2. 参数法（方差-协方差法）

假设收益正态，$\sigma$ 为组合日波动率：

$$VaR_\alpha = V\cdot\big(\mu_h - z_{1-\alpha}\,\sigma_h\big),\qquad CVaR_\alpha=V\cdot\Big(\mu_h-\sigma_h\frac{\phi(z_{1-\alpha})}{1-\alpha}\Big)$$

其中 $\phi$ 是标准正态密度，正态下尾部条件期望有闭式解。1 天 99% 的 $z_{0.01}=-2.326$，$\phi(z_{0.01})/0.01=0.2666$。

- 优点：计算快，可用于大规模组合与实时风控。
- 缺点：**正态假设严重低估尾部**。真实金融收益尖峰厚尾，极端事件的频率远高于正态预测。改进方式是用 Student-$t$（自由度 3~5）或 Cornish-Fisher 展开修正偏度与峰度。

### 3. 蒙特卡洛法

指定收益的生成过程（如多元 $t$ 分布、GARCH、Copula 或因子模型），模拟 $M$ 条路径得到组合损益分布，取分位数。

- 优点：可以处理非线性资产（期权、可转债）、路径依赖，并可以叠加任意分布假设。
- 缺点：计算慢、模型风险大（结果取决于你假设的生成过程）、需要固定随机种子保证可复现。

| 方法 | 分布假设 | 非线性资产 | 计算成本 | 主要风险 |
| --- | --- | --- | --- | --- |
| 历史模拟 | 无 | 差（需重估） | 低 | 样本外事件 |
| 参数法 | 正态/t | 差 | 极低 | 厚尾低估 |
| 蒙特卡洛 | 自定义 | 好 | 高 | 模型设定错误 |

## 三、VaR 的根本缺陷：不是一致风险度量

Artzner、Delbaen、Eber 与 Heath（1999）提出**一致风险度量（coherent risk measure）**应满足四条公理：单调性、次可加性、正齐次性、平移不变性。

VaR **不满足次可加性**：

$$VaR(A+B) \not\le VaR(A)+VaR(B)$$

这意味着：分散化在 VaR 的度量下可能"增加风险"，甚至出现"两个子柜台的 VaR 之和小于总组合 VaR"的荒谬结论，从而诱发监管套利。CVaR 满足全部四条公理，是一致风险度量，且是凸的，便于放进优化问题（Rockafellar & Uryasev 2000 给出了用线性规划求解 CVaR 最小组合的经典方法）。

其他局限：

- **只给阈值不给深度**：VaR 完全不区分"刚好突破"与"亏光"。
- **对尾部估计极不稳定**：$\alpha$ 越高，用到的样本越少，估计方差越大。99.9% 的 VaR 在实务上几乎无法从历史数据可靠估计。
- **不度量流动性风险**：危机时买卖价差扩大、无法成交，VaR 假设可以按市价平仓。
- **不度量内生流动性/反馈**：多家机构同时止损会造成螺旋，VaR 假设头寸不影响价格。
- **回测 VaR 本身很难**：突破事件稀少，Kupiec 检验（1995）统计量功效很低。

## 四、代码示例

```python
import numpy as np
from qd.data.synthetic import generate_demo_market
from qd.metrics.perf import value_at_risk, conditional_var, max_drawdown, ulcer_index

md = generate_demo_market(n_symbols=40, start="2018-01-01", end="2024-12-31", seed=7)
rets = md.returns().dropna()
port = rets.mean(axis=1).values           # 等权组合日收益（近似）

for alpha in (0.95, 0.99):
    var_h = value_at_risk(port, alpha=alpha, method="historical")
    cvar_h = conditional_var(port, alpha=alpha, method="historical")
    var_n = value_at_risk(port, alpha=alpha, method="parametric")
    cvar_n = conditional_var(port, alpha=alpha, method="parametric")
    print("alpha=%.2f  历史法 VaR=%.3f%% CVaR=%.3f%% | 正态法 VaR=%.3f%% CVaR=%.3f%%"
          % (alpha, var_h*100, cvar_h*100, var_n*100, cvar_n*100))

# 尾部厚度自检：正态下 99% CVaR/VaR 应约为 1.15，实际偏大即为厚尾
print("尾部比 CVaR/VaR =", round(conditional_var(port, 0.99) / value_at_risk(port, 0.99), 2))
print("最大回撤 %.2f%%  Ulcer %.2f" % (max_drawdown(port)*100, ulcer_index(port)))
```

## 常见坑

- **把历史 VaR 当预测**：它是"过去分布的描述"，不是前瞻风险。应配合压力测试与情景分析。
- **只看 VaR 不看 CVaR**：尤其在含期权、可转债、或小概率大额亏损策略上。
- **忽略波动率聚集**：用全样本恒定 $\sigma$ 会低估危机期风险。建议用 EWMA 或 GARCH 得到条件波动率再算 VaR。
- **用日 VaR 直接乘 $\sqrt{h}$**：平方根法则只在收益独立同分布时成立；存在自相关（波动率聚集）时会低估多日风险。
- **回测突破次数说不清**：样本内 250 天、99% VaR，期望突破次数只有 2.5 次，多一次少一次都在噪声范围内。要用足够长样本与 [净化切分](../10-机器学习/10-03-金融数据的交叉验证.md) 思维。

## 延伸阅读

- Philippe Artzner, Freddy Delbaen, Jean-Marc Eber & David Heath (1999), *Coherent Measures of Risk*, Mathematical Finance
- R. Tyrrell Rockafellar & Stanislav Uryasev (2000), *Optimization of Conditional Value-at-Risk*, Journal of Risk
- R. Tyrrell Rockafellar & Stanislav Uryasev (2002), *Conditional Value-at-Risk for General Loss Distributions*, Journal of Banking & Finance
- Paul Kupiec (1995), *Techniques for Verifying the Accuracy of Risk Measurement Models*, Journal of Derivatives
- Philippe Jorion (2006), *Value at Risk: The New Benchmark for Managing Financial Risk*, 3rd ed., McGraw-Hill
