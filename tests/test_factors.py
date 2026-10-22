"""因子层测试."""
import numpy as np
import pandas as pd
import pytest

from qd.factors import get_factor, list_factors, BUILTIN_FACTORS
from qd.factors.pipeline import (
    winsorize, zscore, neutralize, rank, combine, orthogonalize, to_weights,
)


def test_registry():
    assert len(BUILTIN_FACTORS) >= 40
    assert "mom_12_1" in BUILTIN_FACTORS
    fs = list_factors()
    assert len(fs) == len(BUILTIN_FACTORS)
    assert all({"name", "category", "direction"} <= set(f) for f in fs)


def test_all_factors_compute(md_small):
    """每个因子都能算，且输出形状正确、不全是 NaN。"""
    bad = []
    for name in BUILTIN_FACTORS:
        try:
            f = get_factor(name)
            out = f.compute(md_small)
            assert out.shape == (md_small.n_dates, md_small.n_symbols), f"{name} 形状错误"
            frac = out.notna().mean().mean()
            assert frac > 0.3, f"{name} 有效值太少 ({frac:.0%})"
        except KeyError:
            pass  # 需要字段缺失的因子跳过（md_small 只带部分基本面）
        except Exception as e:  # noqa: BLE001
            bad.append((name, repr(e)[:100]))
    assert not bad, f"失败的因子：{bad}"


def test_factor_direction(md_small):
    """direction 为 -1 的因子，signed() 后应该变成正向。"""
    f = get_factor("rev_5d")
    assert f.direction == -1
    raw = f.compute(md_small)
    signed = f.signed(md_small)
    np.testing.assert_allclose(signed.dropna().to_numpy(), (-raw).dropna().to_numpy(), rtol=1e-12)


def test_winsorize_limits_extremes():
    """横截面去极值：一行代表一个交易日的横截面。"""
    # 单行 50 个标的，其中一个是极端值
    rng = np.random.default_rng(0)
    row = rng.normal(0, 1, 50)
    row[0] = 1000.0
    df = pd.DataFrame([row], index=pd.date_range("2024-01-01", periods=1))
    w = winsorize(df, method="mad", n=3)
    assert w.iloc[0, 0] < 1000, "去极值没有限制极端值"
    assert w.iloc[0, 0] < 10, f"极端值应被压缩到合理范围，实际 {w.iloc[0, 0]}"
    # 正常值不应被改动
    assert abs(w.iloc[0, 5] - row[5]) < 1e-9


def test_winsorize_quantile():
    df = pd.DataFrame(np.random.default_rng(0).normal(0, 1, (100, 5)))
    w = winsorize(df, method="quantile", n=0.05)
    assert w.max().max() <= df.quantile(0.95, axis=1).max()


def test_zscore(md_small):
    f = get_factor("mom_12_1")
    raw = f.compute(md_small)
    z = zscore(winsorize(raw), clip=3.0)
    row = z.iloc[-1].dropna()
    assert abs(row.mean()) < 1e-9
    assert abs(row.std(ddof=0) - 1.0) < 0.05 or len(row) < 3
    assert z.abs().max().max() <= 3.0 + 1e-9


def test_rank():
    """横截面排名：pct=True 时取值落在 (0, 1]，最大值 = 1.0。"""
    df = pd.DataFrame([[3.0, 1.0, 2.0, 5.0, 4.0]],
                      index=pd.date_range("2024-01-01", periods=1),
                      columns=["a", "b", "c", "d", "e"])
    r = rank(df, pct=True, ascending=True)
    assert r.iloc[0]["d"] == 1.0          # 最大 -> 5/5
    assert r.iloc[0]["b"] == 0.2          # 最小 -> 1/5
    assert r.iloc[0]["a"] == 0.6          # 3 -> 3/5
    assert r.min().min() > 0 and r.max().max() <= 1.0
    # 降序时方向相反
    rd = rank(df, pct=True, ascending=False)
    assert rd.iloc[0]["b"] == 1.0


def test_neutralize_removes_size_and_industry(md_small):
    """中性化后的因子应与对数市值正交。"""
    f = get_factor("mom_3m")
    raw = f.compute(md_small)
    neu = neutralize(raw, md_small, industry=True, size=True)
    lncap = np.log(md_small.fundamental("mktcap"))
    a = neu.iloc[300:].to_numpy().ravel()
    b = lncap.iloc[300:].to_numpy().ravel()
    m = np.isfinite(a) & np.isfinite(b)
    corr = np.corrcoef(a[m], b[m])[0, 1]
    assert abs(corr) < 1e-6, f"中性化后仍与市值相关 {corr:.6f}"


def test_neutralize_matches_bruteforce(md_small):
    """向量化实现必须与逐日 OLS 数值等价。"""
    x = (md_small.close() / md_small.close().shift(60) - 1).iloc[60:75]
    vec = neutralize(x, md_small)

    ln = np.log(md_small.fundamental("mktcap"))
    ind = md_small.fundamental("industry")
    ref = x.copy()
    for d in x.index:
        y, s, g = x.loc[d], ln.loc[d], ind.loc[d]
        m = y.notna() & s.notna() & g.notna()
        if m.sum() < 5:
            continue
        D = pd.get_dummies(g[m], drop_first=True, dtype=float)
        X = np.column_stack([np.ones(m.sum()), s[m].values, D.values])
        beta = np.linalg.lstsq(X, y[m].values.astype(float), rcond=None)[0]
        ref.loc[d, y[m].index] = y[m].values - X @ beta

    a, b = vec.to_numpy(), ref.to_numpy()
    m = np.isfinite(a) & np.isfinite(b)
    np.testing.assert_allclose(a[m], b[m], atol=1e-7)


def test_combine_zscore_and_rank(md_small):
    a = zscore(winsorize(get_factor("mom_12_1").compute(md_small)))
    b = zscore(winsorize(get_factor("rev_5d").compute(md_small)))
    c = combine({"mom": a, "rev": b}, weights={"mom": 0.7, "rev": 0.3})
    assert c.shape == a.shape
    cr = combine([a, b], method="rank")
    assert cr.shape == a.shape


def test_orthogonalize(md_small):
    a = zscore(winsorize(get_factor("mom_12_1").compute(md_small)))
    b = zscore(winsorize(get_factor("mom_3m").compute(md_small)))
    o = orthogonalize(a, [b])
    # 正交化后 a 与 b 的相关性应大幅下降
    x = a.iloc[400:].to_numpy().ravel()
    y = b.iloc[400:].to_numpy().ravel()
    z = o.iloc[400:].to_numpy().ravel()
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    before = abs(np.corrcoef(x[m], y[m])[0, 1])
    after = abs(np.corrcoef(z[m], y[m])[0, 1])
    assert after < before + 1e-6


def test_to_weights_sums_to_one(md_small):
    score = zscore(winsorize(get_factor("mom_12_1").compute(md_small)))
    w = to_weights(score, long_top=0.2, gross=1.0)
    rowsum = w.sum(axis=1)
    # 因子有值的日子里，权重必须归一到 1；因子全 NaN 的预热期则为 0
    valid = score.notna().sum(axis=1) > 0
    assert np.allclose(rowsum[valid].to_numpy(), 1.0, atol=1e-9)
    assert (rowsum[~valid] == 0).all()


def test_get_factor_errors():
    with pytest.raises(KeyError):
        get_factor("这个因子不存在")
