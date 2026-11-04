<!--
id: 02-07
title: pandas与numpy量化技巧
category: 02-数据与工具
level: 中级
order: 7
prereq: [02-01]
tags: [工具, pandas, 性能]
api: [qd.data.MarketData, qd.data.synthetic.generate_demo_market, qd.utils.calendar.trading_days]
-->

# pandas与numpy量化技巧

## 学完这一篇，你能做到

- 说清 `dates × symbols`（行=日期、列=股票）这个约定为什么是全平台的地基
- 理解"向量化"是什么，知道它为什么能比循环快几十倍
- 分清 `axis=0` / `axis=1`，不再把横截面运算写错
- 认出几个最坑人的对齐陷阱与性能陷阱

## 本篇要点

- 全平台统一约定：**index 是日期，columns 是股票**，坚持它，因子之间才能直接做算术
- `shift(1)` 是防止前视偏差最常用的操作；`shift(-1)` 只能出现在标签里
- **向量化 = 把 for 循环写成一整块表达式**，同一个计算可以快几十倍
- 对 `dates × symbols` 面板，**横截面运算看"每一行内部"**，用 `axis=1`

---

## 一、从一个人嫌工具太难用说起

2008 年前后，一个叫 **Wes McKinney** 的年轻人在对冲基金 AQR 做量化研究。他每天要处理大量金融时间序列：不同频率、不同对齐方式、缺失值、分组统计。

当时 Python 里能用的工具都不趁手：R 有 dataframe 但生态封闭，Python 只有偏底层的 NumPy 数组，**没有"带标签的表格"这种东西**。于是他干脆自己写了一个库，2009 年开源，取名 **pandas**（来自 panel data，面板数据）。

后来他成了 pandas 的"终身仁慈独裁者"（BDFL），又创建了 Apache Arrow 和 Ibis，写了《Python for Data Analysis》。今天几乎所有做量化、做数据分析的 Python 程序员，第一行都是 `import pandas as pd`。（[Wikipedia: Wes McKinney](https://en.wikipedia.org/wiki/Wes_McKinney)）

> **这个故事里藏着一个很实在的道理：**
> pandas 不是为了"语法优雅"诞生的，是为了解决**金融数据又脏又乱、还要算得快**这个具体痛点。
> 所以学它的正确姿势不是背 API，而是理解**它替你解决了什么麻烦**——
> 尤其是一个叫"**对齐**"的麻烦。

### 全平台的地基：一个必须背下来的约定

> **所有面板数据：行（index）是日期，列（columns）是股票代码，即 `dates × symbols`。**

`md.close()`、`md.volume()`、`md.returns()`、任何因子的计算结果，全都遵守这一条。

为什么值得这么较真？因为**只要形状统一，两张表就能直接相加减**，pandas 会按行列标签**自动对齐**：

```python
from qd.data.synthetic import generate_demo_market

md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
close, vol, ret = md.close(), md.volume(), md.returns()
print(close.index[:2], close.columns[:3], close.shape)
```

实跑输出：`(1827, 50)` —— 1827 个交易日、50 只股票。

> **"自动对齐"是 pandas 最值钱、也最容易被忽视的能力。**
> 你不用自己写循环去凑"这两张表的第 3 行是不是同一天"，
> 它按标签对，**对不上就给 NaN，而不是悄悄错位**。
> 在量化里，"悄悄错位"几乎必然等于前视偏差（02-04）——所以这个特性是安全网，不是便利功能。

---

## 二、向量化：把循环写成表达式（单独讲透）

先看一段最能说明问题的对比。同样是算"20 日动量"：

```python
import time
import numpy as np
import pandas as pd
from qd.data.synthetic import generate_demo_market

md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
close = md.close()

# 写法 A：老老实实两层 for 循环
t0 = time.perf_counter()
vals = close.values.copy()
out = np.full_like(vals, np.nan)
for d in range(21, vals.shape[0]):
    for s in range(vals.shape[1]):
        out[d, s] = vals[d-1, s] / vals[d-21, s] - 1
t_loop = time.perf_counter() - t0

# 写法 B：向量化，一行表达式
t0 = time.perf_counter()
mom = close.shift(1) / close.shift(21) - 1
t_vec = time.perf_counter() - t0

print(f"循环: {t_loop:.3f}s  向量化: {t_vec:.4f}s  快 {t_loop/max(t_vec,1e-9):.0f} 倍")
```

我实跑的结果：

```
循环: 0.058s  向量化: 0.0010s  快 58 倍
```

**同一个结果，差了近 60 倍。** 而且这还只是 50 只股票 × 7 年；真实全市场是几千只、十几年，差距会进一步放大。

> **为什么快？** 因为循环写法是在 **Python 解释器里一格一格地算**，每格都有类型检查、对象创建等开销；
> 向量化写法是把整块数据**一次性交给底层编译好的 C/NumPy 代码**去算。
> 类比：**循环是你自己一粒一粒数米，向量化是把整袋米倒进电子秤。**

用复杂度可以写得更清楚。设面板有 $T$ 个交易日、$N$ 只股票，一次逐格循环的计算量是：

$$\text{循环} = O(T \cdot N) \quad\text{次 Python 解释器调用}$$

而向量化**计算量还是 $O(T \cdot N)$**，但解释器只被调用**一次**：

$$\text{向量化} = O(T \cdot N) \ \text{次 C 运算} + 1 \ \text{次 Python 调用}$$

逐个拆：$T$、$N$ 就是 02-01 说的"行是日期、列是股票"；$O(\cdot)$ 读作"大 O"，表示"随着数据量增大，工作量按什么规模涨"。

**所以快不是因为算得少，是因为"起跑"的次数少。** 这也解释了为什么数据越大、优势越明显——$T \times N$ 越大，那"每次起步的固定开销"被摊得越薄，而我实跑的 58 倍只是 50 只股票 × 7 年的结果。

所以量化的基本功之一，就是**看到 `for` 循环先停一下，问能不能改写成一整块表达式**。

### 三个最常用的向量化零件

| 写法 | 干什么 | 白话 |
| --- | --- | --- |
| `close.shift(1)` | 取 1 行之前的值 | "用昨天" |
| `close.rolling(20).mean()` | 20 日滚动均值 | 20 日均线 |
| `ret.rank(axis=1, pct=True)` | 每行内部排名转百分比 | 今天谁排前 20% |

```python
import pandas as pd
from qd.data.synthetic import generate_demo_market

md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
close, vol, ret = md.close(), md.volume(), md.returns()

# 20 日动量（不含最近 1 日）
mom_20_1 = close.shift(1) / close.shift(21) - 1
# 20 日已实现波动率（年化）
rv_20 = ret.rolling(20).std() * (252 ** 0.5)
# 20 日均线偏离度
dev = close / close.rolling(20).mean() - 1
# 流动性过滤：20 日平均成交额 > 1 亿
liquid = (close * vol).rolling(20).mean() > 1e8
```

> ⚠️ **`shift(1)` 值得单独记一笔**：它和 02-04 直接相关。
> 信号里出现 `shift(1)`，意味着"t 日用的是 t-1 日的信息"，**这是安全的**；
> 反过来 **`shift(-1)` 只能出现在标签里，绝不能进特征**——那是明晃晃的前视偏差。
> 一个字符的方向，就是"合规"和"作弊"的分界。

---

## 三、`axis` 到底该填几（最常见的踩坑点）

这是 pandas 里最反直觉、也最容易写错的一处，**单独讲透**。

对 `dates × symbols` 面板：

| 你想干什么 | 沿哪个方向 | 用 |
| --- | --- | --- |
| 算**某一天**所有股票的均值/排名/标准化 | 每一行内部（横向，跨列） | `axis=1` |
| 算**某只股票**跨时间的均值/滚动 | 每一列内部（纵向，跨行） | `axis=0`（默认） |

横截面标准化（每个交易日内部比较所有股票）是量化里最典型的动作：

```python
import pandas as pd
from qd.data.synthetic import generate_demo_market

md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
close, vol, ret = md.close(), md.volume(), md.returns()

# 每个交易日内部：减去当日均值、除以当日标准差
mom = close.shift(1) / close.shift(21) - 1
z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1), axis=0)

# 横截面排名，映射到 [-0.5, 0.5]
rank = mom.rank(axis=1, pct=True) - 0.5

# MAD 法去极值（比固定分位数更稳健）
med = mom.median(axis=1)
mad = mom.sub(med, axis=0).abs().median(axis=1)
wins = mom.clip(lower=med.sub(5 * mad, axis=0), upper=med.add(5 * mad, axis=0), axis=0)
```

**注意那个反直觉的组合**：`mean(axis=1)` 是在"每行内部"求均值，但 `sub(..., axis=0)` 里的 `axis=0` 说的是"**按行标签把这一串均值广播回去**"。两者不矛盾，只是 pandas 的广播方向写法。

> **别靠背，靠验。** 写完标准化，随手抽一行检查：
> ```
> print(round(z.iloc[100].mean(), 8))   # 应约为 0
> print(round(z.iloc[100].std(), 4))    # 应约为 1
> ```
> 我实跑得到 `0.0`。**一行验证，能挡掉 90% 的 axis 错误。**

> 补两个常识：
> - **横截面（cross-section）**："同一时刻、所有标的"的那一刀。01-01 讲因子时反复出现，说的就是这个。
> - **MAD（中位数绝对偏差）**：用中位数而不是均值来定"多远算异常"，比固定分位数更抗极端值——这正是"稳健统计"的思路。

---

## 四、groupby 与 resample（一句话说清）

| 动作 | 干什么 | 典型用法 |
| --- | --- | --- |
| `groupby` | 按某个标签分组后分别算 | 按行业分组做中性化、按月份算 IC 均值 |
| `resample` | 改变时间频率 | 日频转周频/月频 |

```python
import pandas as pd
from qd.data.synthetic import generate_demo_market

md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
close, vol, ret = md.close(), md.volume(), md.returns()

mom = close.shift(1) / close.shift(21) - 1

# 按年看动量因子的均值（01-02 里见过这个动作）
yearly = mom.groupby(mom.index.year).mean()
# 日频收益转成月频
monthly_ret = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
```

> **groupby 的心智模型**："**先按标签把桌子拆成几堆，每堆各算各的，再拼回来**"。
> 按行业分组做中性化（05 章会讲），本质就是"每个行业内部各自标准化"，避免把"银行股整体便宜"误判成选股信号。

---

## 五、对齐与索引陷阱（最坑人的一类）

pandas 自动对齐是安全网，但有两个前提不满足时，它会**静默地给你错误结果**：

| 陷阱 | 会发生什么 | 怎么防 |
| --- | --- | --- |
| 索引有重复 | 对齐时产生笛卡尔积，行数暴涨 | 先 `index.is_unique` 检查 |
| 索引未排序 | 部分操作结果错位 | `sort_index()` |
| 混用 `date` 与 `Timestamp` | 比较时静默失败，筛出空集 | 统一 `.normalize()` 或转 datetime64 |
| 用位置 `iloc` 拼不同表 | 绕过标签对齐，悄悄错位 | **永远用标签对齐，别用位置硬拼** |

> **最后一条最要命**：`iloc[i]` 拿的是"第 i 行"，不管那天是哪天。
> 两张日期不同的表用位置硬拼，等于把 3 月的数据当成 4 月的用——
> **这正是 02-04 前视偏差的一个隐蔽来源**，而且不报任何错。

---

## 六、性能与工程清单

- **别用 `iterrows()` / `itertuples()` 逐行处理面板**，那是把向量化好不容易省下的时间又还回去了。
- **别在循环里 `pd.concat` 追加行**，先攒进 list 最后拼一次。
- **宽表按年分区**存 Parquet（见 [02-06](./02-06-数据存储与工程.md)），别把 10 年全市场塞进一个 CSV。
- **`rolling` 的 `min_periods` 别设太小**，窗口未满时用几个样本估标准差会造出早期极端值。
- **写完后抽样验证**：某一行均值/标准差、某段时间范围、NaN 数量。

---

## 七、把这一篇串起来

回到 Wes McKinney。他当年不是想造一个"更好的语法"，而是被金融数据里那些**对不齐、缺一块、频率还不一样**的麻烦逼得自己动手。

所以 pandas 真正卖给我们的，是两样东西：

> 1. **自动对齐** —— 让"错位"变成 NaN，而不是变成一条漂亮的假曲线（02-04）；
> 2. **向量化** —— 让"检验 100 个想法"从等一整天变成等一分钟，从而**愿意多跑几次**（01-03）。

> 这两样恰好接上了前面两篇：
> **跑得快，你才肯做样本外检验；对得齐，你才不会偷偷看未来。**
> 工具从来不是中性的——**它决定了你愿意做多少诚实的检查。**

> 一句话收尾：
> **`shift(1)` 和 `shift(-1)` 只差一个字符，中间隔着整个行业的信誉。**
> 把工具用对，本身就是风控。

---

## 动手练习

- 跑一遍向量化对比，把 `n_symbols` 从 50 改成 200，看"快多少倍"怎么变。**先猜，再跑。**
- 写一个横截面标准化，**必须**用 `z.iloc[100].mean()` 和 `.std()` 验证均值为 0、标准差为 1。
- 在自己的因子代码里搜一遍 `iterrows`、`shift(-`、`iloc[i+`，逐个确认没有踩本篇和 02-04 的坑。

## 延伸阅读

- Wes McKinney，《Python for Data Analysis》——pandas 作者本人写的入门与实战。
- pandas 官方文档：Indexing and Selecting Data、Group by、Window functions——`axis` 与广播规则的权威说明。
- [02-04 前视偏差](./02-04-前视偏差.md)、[02-06 数据存储与工程](./02-06-数据存储与工程.md)——本篇两处伏笔的落点。
