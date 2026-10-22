---
id: 05-05
title: 单因子检验IC与ICIR
category: 05-因子
level: 进阶
order: 5
prereq: [05-03]
tags: [因子, IC, ICIR, 显著性检验]
objectives:
  - 能计算 Pearson IC 与 RankIC 并说明二者差异
  - 能用 ICIR 与 t 统计量判断单因子是否统计显著
  - 能说明 IC>0.02 这类经验阈值的适用条件与局限
key_points:
  - IC = 第 t 期因子暴露与 t+1 期收益的横截面相关系数
  - ICIR = mean(IC)/std(IC)，t 统计量 $t = \mathrm{ICIR}\sqrt{n-1}$
  - RankIC 更稳健，Pearson IC 对极端值敏感
  - IC>0.02、ICIR>0.3 是经验值，依赖市场、样本期与调仓频率
exercises:
  - 用 np.corrcoef 手算 mom_12_1 的 IC 序列，与 lab.ic_series 对账
  - 对同一因子分别算 horizon=5/20/60 的 IC，观察衰减方向
api:
  - qd.research.lab.FactorLab
  - qd.research.lab.FactorLab.ic
  - qd.research.lab.FactorLab.ic_series
  - qd.research.lab.FactorLab.report
  - qd.data.synthetic.generate_demo_market
---

# 单因子检验IC与ICIR

一个因子算出来之后，第一个要回答的问题是：**它到底有没有预测能力？** IC（Information Coefficient，信息系数）就是回答这个问题最基础的工具。

## 一、IC 的定义

在第 $t$ 期，把所有股票的因子暴露 $x_{i,t}$ 与它们在未来 $h$ 期的收益 $r_{i,t \to t+h}$ 放在一起，计算横截面相关系数，就得到当期 IC：

$$\mathrm{IC}_t = \mathrm{corr}\Big(\{x_{i,t}\}_{i=1}^{N},\ \{r_{i,t\to t+h}\}_{i=1}^{N}\Big)$$

得到的 $\mathrm{IC}_t$ 序列随时间变化，其最常用的两个统计量是：

$$\overline{\mathrm{IC}} = \frac{1}{n}\sum_{t=1}^{n}\mathrm{IC}_t,\qquad \sigma_{\mathrm{IC}} = \sqrt{\frac{1}{n-1}\sum_{t=1}^{n}(\mathrm{IC}_t - \overline{\mathrm{IC}})^2}$$

直觉上，$\overline{\mathrm{IC}} = 0.05$ 意味着因子暴露能"解释"未来收益横截面约 5% 的相关结构。它很小，但由于横截面样本极多（几百到几千只股票 × 几百个时间点），即使微弱也可以高度显著。

### 关键：收益必须向前对齐

计算 IC 时最容易出错的一步是对齐。第 $t$ 期的因子暴露只能对应 $t$ 期之后实现的收益。若用 $t$ 期因子对 $t$ 期当天收益求相关，看起来 IC 会异常之高（尤其对反转因子），但这是纯粹的自相关假象，属于典型的[前视偏差](../06-回测/06-03-前视偏差的十种来源.md)。

## 二、Pearson IC 还是 RankIC

| 维度 | Pearson IC | Spearman RankIC |
| --- | --- | --- |
| 定义 | 原始值线性相关 | 排名之间的相关（即 Pearson 作用于秩） |
| 对极端值 | 敏感，一个离群点可大幅改变 | 免疫 |
| 捕捉关系 | 仅线性 | 任意单调关系 |
| 典型量级 | 较小 | 通常比 Pearson 大 0.01~0.03 |
| 适用场景 | 因子已去极值、分布对称 | 厚尾、偏态的基本面因子 |

**实践建议：以 RankIC 为主，Pearson IC 为辅。** 若两者差异巨大，说明因子的预测能力集中在少数极端样本上，应回到[去极值](../05-因子/05-03-去极值与标准化.md)与[分层回测](../05-因子/05-06-分层回测.md)确认是否为个例驱动。

## 三、ICIR 与 t 检验

只有 IC 均值还不够——一个 IC 均值为 0.03 但每期在 ±0.10 之间剧烈波动的因子，实盘体验会很差。因此引入信息比率：

$$\mathrm{ICIR} = \frac{\overline{\mathrm{IC}}}{\sigma_{\mathrm{IC}}}$$

ICIR 衡量的是每单位 IC 波动能换来多少 IC 均值，即"稳定性"。在 $\{\mathrm{IC}_t\}$ 近似独立同分布的假设下，检验 $\overline{\mathrm{IC}} = 0$ 的 t 统计量为：

$$t = \frac{\overline{\mathrm{IC}}}{\sigma_{\mathrm{IC}} / \sqrt{n}} = \mathrm{ICIR}\sqrt{n}$$

（教材中常见的写法 $t = \mathrm{ICIR}\sqrt{n-1}$ 与之等价，差别只是用 $n-1$ 还是 $n$ 做自由度修正。）经验上 $|t| > 2$ 对应 5% 显著性水平。

### 自相关问题

$\{\mathrm{IC}_t\}$ 往往存在正自相关——因为重叠的持有期（horizon=20 时相邻期的收益窗口重叠 19 天）会产生机械性自相关，宏观状态也会让因子连续几期同时有效或失效。此时 $\sqrt{n}$ 放大 t 值是不合法的，有效样本量远小于 $n$，标准做法是：

- 使用 **Newey-West** 修正的标准误（滞后阶数取约 $h$ 期）；
- 或使用**非重叠期**采样（每 20 期取一个 IC），牺牲样本量换取独立性。

不加修正直接看 t 值，是最常见的"因子显著"假象来源之一。

## 四、经验阈值及其局限

业界流传的门槛大致是：

| 指标 | 经验阈值 | 含义 |
| --- | --- | --- |
| $\vert\overline{\mathrm{IC}}\vert$ | > 0.02 | 具备初步可用性 |
| $\vert\overline{\mathrm{RankIC}}\vert$ | > 0.03 | 同上（RankIC 通常更大） |
| $\vert\mathrm{ICIR}\vert$ | > 0.3 | 稳定性尚可 |
| IC 为正的比例 | > 60% | 方向一致性 |
| $ \vert t \vert$ | > 2 | 统计显著 |

**这些阈值必须谨慎使用**，原因有四：

1. **依赖调仓频率**：日频因子的 IC 天然比月频小（单日收益噪声占比高），用同一阈值横评不公平。
2. **依赖横截面样本**：小盘股池的 IC 通常高于大盘股池，因为前者定价效率低。
3. **依赖市场与样本期**：A 股、美股、新兴市场的有效因子与量级差异明显，牛市与熊市也不同。
4. **不可跨因子直接比较**：一个覆盖全市场的因子 IC=0.03，与一个只在沪深 300 内 IC=0.03 的因子，难度完全不同。

更合理的做法是**用同一市场、同一股票池、同一频率下的全部候选因子做横向排名**，而不是套用一个绝对阈值。

## 五、IC 为正但分组不单调

IC 只看相关性，不看形态。可能出现：IC 均值为正且显著，但[分层回测](../05-因子/05-06-分层回测.md)的 5 组收益是 0.5%、0.4%、0.3%、0.5%、0.9%——不单调。常见成因：

- **非线性**：因子与收益是 U 型关系，线性相关系数无法刻画；
- **尾部驱动**：只有第 1 组（或第 5 组）显著异常，中间的区分度为零，此时多空收益来自一端；
- **少数股票主导**：某一期某只重组股暴涨，单独拉高了 IC。

因此 **IC 与分层回测必须同时做**：IC 回答"有没有关系"，分层回测回答"关系长什么样、能不能靠分组赚到"。

## 六、代码：手算 IC 并与平台对账

```python
import numpy as np
import pandas as pd
from qd.data.synthetic import generate_demo_market
from qd.factors import get_factor
from qd.research.lab import FactorLab

md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
lab = FactorLab(md)

name = "mom_12_1"
expo = get_factor(name).compute(md)          # dates × symbols
fwd = md.close().pct_change(20).shift(-20)   # 未来 20 日收益，注意 shift 方向

# 手工计算 Pearson IC 序列
ics = []
for d in expo.index[:-20]:
    x = expo.loc[d]
    y = fwd.loc[d]
    mask = x.notna() & y.notna()
    if mask.sum() >= 20:
        ics.append(np.corrcoef(x[mask], y[mask])[0, 1])
ics = pd.Series(ics)

ic_mean = ics.mean()
ic_std = ics.std()
icir = ic_mean / ic_std
t_stat = icir * np.sqrt(len(ics))
print(f"IC均值={ic_mean:.4f}  IC标准差={ic_std:.4f}  ICIR={icir:.3f}  t={t_stat:.2f}")
print("IC 为正的比例:", (ics > 0).mean())

# 与平台结果对账
print("lab.ic        :", lab.ic(name, horizon=20))
print("lab.ic_series :", lab.ic_series(name).tail())
print("lab.report    :", lab.report(name))
```

手工复现一遍 IC 是理解定义的最好方式，也是排查"平台 IC 为何与自己算的不同"的标准手段——绝大多数差异来自对齐方式（`shift(-20)` 的方向）与有效样本数的阈值。

## 常见坑

1. **对齐方向写反**：`shift(20)` 与 `shift(-20)` 的差别就是"用过去收益"和"用未来收益"，前者会得到虚假的高 IC。
2. **持有期重叠未修正**：horizon 大于调仓间隔时，IC 序列自相关，t 值被高估，必须用 Newey-West 或非重叠采样。
3. **样本量阈值过低**：每期只有十几只有效样本时算出的 IC 噪声极大，应设最低样本数（如 ≥ 20~30）。
4. **只看均值不看时间序列**：IC 均值可能被个别极端期主导，应同时看 IC 序列图、正比例与分年 IC。
5. **幸存者偏差**：只用当前仍在市的成分股算历史 IC，会系统性高估因子表现。
6. **用合成数据的结论外推**：`generate_demo_market` 的因子效应由构造决定，只用于验证代码正确性。

## 延伸阅读

- Grinold, R. C. & Kahn, R. N., Active Portfolio Management
- Jegadeesh, N. & Titman, S. (1993), Returns to Buying Winners and Selling Losers
- Fama, E. F. & MacBeth, J. D. (1973), Risk, Return, and Equilibrium: Empirical Tests
