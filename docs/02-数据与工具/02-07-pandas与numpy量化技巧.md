---
id: 02-07
title: pandas与numpy量化技巧
category: 02-数据与工具
level: 中级
order: 7
prereq: ["02-01", "02-04"]
tags: [工具, pandas, 性能]
objectives:
  - 养成"面板即 DataFrame(dates × symbols)"的统一心智模型
  - 掌握向量化、rolling、groupby/resample、shift 对齐的正确用法
  - 能识别并规避 apply/循环/链式赋值等性能陷阱
key_points:
  - 统一约定：行=日期升序，列=标的；所有信号都是这个形状
  - 90% 的因子可以用 rolling + 算术运算符写完，不需要循环
  - 性能陷阱：`df.apply`、`iterrows`、逐列 concat、链式赋值
exercises:
  - 用纯向量化方式实现 20 日动量、20 日波动率、20 日成交额均值三个因子并计算它们的截面相关性
  - 把一段 for 循环实现的因子改成向量化版本，测量加速比
api:
  - qd.data.MarketData
  - qd.data.synthetic.generate_demo_market
  - qd.factors.pipeline.winsorize
  - qd.factors.pipeline.zscore
---

# pandas与numpy量化技巧

## 一、统一心智模型

在整个平台里，请始终坚持一个约定：

> **所有面板数据的 index 是日期（升序、无重复），columns 是标的代码，即 `dates × symbols`。**

`md.close()`、`md.open()`、`md.volume()`、`md.returns()` 以及任何 `f.compute(md)` 的返回值都遵守这个约定。坚持统一形状带来的直接好处是：**因子之间可以直接做算术运算**，因为索引会自动对齐。

```python
from qd.data.synthetic import generate_demo_market

md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
close, vol, ret = md.close(), md.volume(), md.returns()
print(close.index[:2], close.columns[:3], close.shape)
```

## 二、向量化：把循环写成表达式

### 基础：滚动窗口

```python
from qd.data.synthetic import generate_demo_market
md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
close, vol, ret = md.close(), md.volume(), md.returns()

# 20 日动量（不含最近 1 日，即经典的 mom_12_1 思路的短周期版本）
mom_20_1 = close.shift(1) / close.shift(21) - 1

# 20 日已实现波动率（年化）
rv_20 = ret.rolling(20).std() * (252 ** 0.5)

# 20 日均线与价格偏离度
ma20 = close.rolling(20).mean()
dev = close / ma20 - 1

# 成交额：用滚动均值做流动性过滤
adv_20 = (close * vol).rolling(20).mean()
liquid = adv_20 > 1e8                      # 布尔面板，可直接作掩码
```

关键点：

- `shift(k)` 表示"取 k 行之前的值"。**`shift(1)` 是防止前视偏差最常用的操作**：它保证 $t$ 行的值来自 $t-1$ 日。
- 任何形如 `X.shift(1)` 出现在信号里，都意味着"信号在 t 日使用 t-1 日信息"，这是安全的。
- 反向的 `shift(-1)` 只应出现在**标签**里，绝不能进入特征。详见 [02-04-前视偏差](./02-04-前视偏差.md)。

### 截面运算（axis 的选择）

```python
from qd.data.synthetic import generate_demo_market
md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
close, vol, ret = md.close(), md.volume(), md.returns()

# 前置：计算本节要用的基础面板
mom_20_1 = close.shift(1) / close.shift(21) - 1
rv_20 = ret.rolling(20).std() * (252 ** 0.5)
adv_20 = (close * vol).rolling(20).mean()
liquid = adv_20 > 1e8

# 横截面标准化：每个交易日内部做（axis=1 表示沿列方向，即对每行的所有股票）
z = (mom_20_1.sub(mom_20_1.mean(axis=1), axis=0)).div(mom_20_1.std(axis=1), axis=0)

# 横截面排名并映射到 [-0.5, 0.5]
rank = mom_20_1.rank(axis=1, pct=True) - 0.5

# 分位数去极值（MAD 法，比固定分位数更稳健）
med = mom_20_1.median(axis=1)
mad = (mom_20_1.sub(med, axis=0)).abs().median(axis=1)
cap = med.add(5 * mad, axis=0)
wins = mom_20_1.clip(lower=med.sub(5 * mad, axis=0), upper=cap, axis=0)
```

容易搞错的是 `axis`：对 `dates × symbols` 面板，**横截面运算用 `axis=1`**，而 `sub/div` 这类带对齐的方法需要配合 `axis=0` 参数才能按行广播。这个反直觉的组合是 pandas 最常见的踩坑点，写完后一定要抽样验证一行的均值为 0、标准差为 1。

## 三、groupby 与 resample

```python
from qd.data.synthetic import generate_demo_market
md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
close, vol, ret = md.close(), md.volume(), md.returns()

from qd.data.synthetic import generate_demo_market
md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
close, vol, ret = md.close(), md.volume(), md.returns()
mom_20_1 = close.shift(1) / close.shift(21) - 1
rv_20 = ret.rolling(20).std() * (252 ** 0.5)
adv_20 = (close * vol).rolling(20).mean()
liquid = adv_20 > 1e8

# 日频 → 月频：取每月最后一个交易日
monthly_close = close.resample("ME").last()

# 月频收益：月内复利
monthly_ret = (1 + ret).resample("ME").prod() - 1

# 按行业分组做中性化（示意：industry 是一个 symbol → 行业的 Series）
industry = pd.Series(pd.Index(md.symbols).astype(str).str[:3], index=md.symbols)  # 示意：用代码前缀代行业
def demean_by_group(df, groups):
    """按列分组做横截面去均值。

    注意：pandas 3.0 起 `DataFrame.groupby(..., axis=1)` 已被移除，
    正确写法是先转置再按行分组。
    """
    return df.T.groupby(groups).transform(lambda x: x.sub(x.mean())).T

neutral_mom = demean_by_group(mom_20_1, industry)

# 按年统计 IC（IC 是一个 Series，index 为日期）
ic = mom_20_1.shift(1).corrwith(ret, axis=1)
print("IC 年均值:\n", ic.groupby(ic.index.year).mean().round(4))
print("ICIR:", round(ic.mean() / ic.std(), 3))
```

`corrwith(..., axis=1)` 是计算横截面 IC 最简洁的方式：它逐行计算两个面板的相关系数，返回一个以日期为索引的 Series。

## 四、对齐与索引陷阱

```python
import numpy as np
import pandas as pd

from qd.data.synthetic import generate_demo_market
md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
close, vol, ret = md.close(), md.volume(), md.returns()
# 前置：计算本节要用的基础面板
mom_20_1 = close.shift(1) / close.shift(21) - 1
rv_20 = ret.rolling(20).std() * (252 ** 0.5)
adv_20 = (close * vol).rolling(20).mean()
liquid = adv_20 > 1e8

# ❌ 危险：两个面板的日期/标的不一致时会静默产生 NaN
a = close.iloc[:-5, :10]
b = close.iloc[5:, :20]
print((a + b).isna().mean().mean())     # 大量 NaN，而且没有报错

# ✅ 显式对齐后再运算
b2, a2 = b.align(a, join="inner")
print((a2 + b2).isna().mean().mean())   # 0

# 布尔掩码与浮点面板相乘：用 where 更清晰
signal = mom_20_1.where(liquid)          # 不满足流动性的位置设为 NaN
```

## 五、性能陷阱与优化

| 陷阱 | 症状 | 替代方案 |
| --- | --- | --- |
| `df.apply(func, axis=1)` 逐行 | 行数多时极慢 | 改成向量化表达式，或先 `stack()` 再 `groupby` |
| `for sym in columns: ...` | 慢且易写错索引 | 用面板级运算 |
| 逐列 `pd.concat` 累积 | $O(n^2)$ 内存拷贝 | 收集到 list 后一次 concat |
| 链式赋值 `df[a][b] = c` | 可能不生效（视图/副本） | 用 `.loc[a, b] = c` |
| `df.append`（旧 API） | 每次拷贝全表 | 收集后 concat |
| 用 object dtype 存数字 | 内存 ×5、速度 ×10 慢 | 显式 `astype("float64")`，用 `np.nan` 而非 `None` |
| 频繁 `df.T` | 拷贝 | 尽量保持 `dates × symbols` 不转置 |

一个典型的加速例子——把"逐只股票算滚动回归 beta"改成批量：

```python
from qd.data.synthetic import generate_demo_market
md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
close, vol, ret = md.close(), md.volume(), md.returns()

# 慢：循环 50 只股票
betas = {}
for sym in close.columns:
    betas[sym] = ret[sym].rolling(250).cov(ret.mean(axis=1)) / ret.mean(axis=1).rolling(250).var()
beta_slow = pd.DataFrame(betas)

# 快：用 numpy 直接算（cov = E[xy] - E[x]E[y]），一次性得到面板
mkt = ret.mean(axis=1)
n = 250
xy = ret.mul(mkt, axis=0).rolling(n).mean()      # E[xy]
mx = mkt.rolling(n).mean()                       # E[x]
my = ret.rolling(n).mean()                       # E[y]
cov = xy.sub(mx, axis=0) - my.mul(mx - mx, axis=0)   # E[xy] - E[x]E[y] 的向量化写法
var = (mkt ** 2).rolling(n).mean() - mx ** 2          # E[x²] - E[x]²
var = (mkt ** 2).rolling(n).mean() - mx ** 2
beta_fast = cov.div(var.to_frame("k")["k"], axis=0)
print(np.allclose(beta_slow.dropna().values, beta_fast.dropna().reindex_like(beta_slow.dropna()).values))
```

## 六、实用片段速查

```python
from qd.data.synthetic import generate_demo_market
md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
close, vol, ret = md.close(), md.volume(), md.returns()
mom_20_1 = close.shift(1) / close.shift(21) - 1
rv_20 = ret.rolling(20).std() * (252 ** 0.5)
adv_20 = (close * vol).rolling(20).mean()
liquid = adv_20 > 1e8
dev = close / close.rolling(20).mean() - 1

# 1) 收益 → 净值
equity = (1 + ret.mean(axis=1)).cumprod()

# 2) 最大回撤
dd = equity / equity.cummax() - 1
mdd = dd.min()

# 3) 换手率（权重面板 → 每日双边换手）
def turnover(w):
    return (w.diff().abs()).sum(axis=1) / 2

# 4) 前 N 名等权多头
def top_n(signal, n=20):
    return signal.rank(axis=1, ascending=False).le(n).astype(float).div(
        signal.rank(axis=1, ascending=False).le(n).sum(axis=1), axis=0)

# 5) 未来 N 日收益（仅用于标签/评价，绝不进特征）
fwd = close.shift(-5) / close - 1

# 6) 前向填充并限制步长（停牌处理）
price_ffill = close.ffill(limit=5)

# 7) 面板相关系数矩阵（因子间冗余检查）
panel = pd.concat({"mom": mom_20_1, "vol": -rv_20, "dev": -dev}, axis=1)
corr = panel.T.groupby(level=0).mean().T.corr()   # 汇总每个字段的横截面均值后算相关
print(corr.round(3))
```

## 七、与平台 pipeline 的分工

平台提供了 `winsorize / zscore / neutralize / rank / combine`，它们对上述操作做了规范封装。**建议：学习与原型阶段自己写一遍（本章的价值），正式研究时调用 pipeline**（更少出错、口径统一）。

```python
from qd.data.synthetic import generate_demo_market
from qd.factors.pipeline import winsorize, zscore, rank, combine

md = generate_demo_market(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7)
close, vol, ret = md.close(), md.volume(), md.returns()

# 前置：计算本节要用的基础面板
mom_20_1 = close.shift(1) / close.shift(21) - 1
rv_20 = ret.rolling(20).std() * (252 ** 0.5)
adv_20 = (close * vol).rolling(20).mean()
liquid = adv_20 > 1e8

mom_c = zscore(winsorize(mom_20_1, limits=0.01))
vol_c = zscore(winsorize(-rv_20, limits=0.01))
combo = combine([mom_c, vol_c], weights=[0.6, 0.4])
print(combo.iloc[-1].describe().round(3))
```

## 常见坑

- **忘记 `shift(1)`**。信号用当日收盘价算、又用当日收盘价成交，是出现频率第一的错误。
- **`axis` 用反**。横截面标准化写成了时间序列标准化，因子的量纲会完全错乱，而且不会报错。
- **用 `fillna(0)` 填充因子**。0 在标准化后不是"中性"而是"略低于均值"，会让一批缺失值被当成有效信号。正确做法是保留 NaN 并在选股时排除。
- **`rolling` 的 `min_periods` 默认等于窗口长度**，导致样本前 N 行全是 NaN；研究时通常要显式设置，但也要警惕太小造成的噪声。
- **链式赋值不生效**。`df[df > 0]["a"] = 1` 可能静默失败，应改用 `.loc`。

## 延伸阅读

- Wes McKinney，《利用 Python 进行数据分析》（*Python for Data Analysis*）——pandas 作者本人所著，是理解索引、对齐与 groupby 的最佳参考。
- pandas 官方文档的 "User Guide → Window / Groupby / Resampling" 章节——行为细节与边界条件的权威说明。
- NumPy 官方文档关于广播（broadcasting）的说明——理解向量化的底层规则。
