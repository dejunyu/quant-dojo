"""内置因子库.

分类：动量 / 反转 / 价值 / 成长 / 质量 / 波动 / 流动性 / 情绪 / 技术 / 另类

所有因子都是**纯横截面**的：对每个交易日，用**截至当日收盘为止**的数据算
一个值，用来预测**未来**收益。因子函数内部只允许使用 ``shift`` 之后的过去
数据——这是「无前视」的硬约束。平台的单元测试会用打乱未来数据的方式验证
这一点（见 ``tests/test_no_lookahead.py``）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import factor_from_func

__all__ = ["BUILTIN_FACTORS"]

# ---------------------------------------------------------------- 动量 / 反转


@factor_from_func(
    name="mom_1m", category="动量", direction=1,
    description="过去 20 个交易日收益（短期动量）",
    formula=r"r_{t-20 \to t}",
    required_fields=("close",),
)
def mom_1m(md) -> pd.DataFrame:
    return md.close() / md.close().shift(20) - 1.0


@factor_from_func(
    name="mom_3m", category="动量", direction=1,
    description="过去 60 个交易日收益",
    formula=r"r_{t-60 \to t}",
)
def mom_3m(md) -> pd.DataFrame:
    return md.close() / md.close().shift(60) - 1.0


@factor_from_func(
    name="mom_6m", category="动量", direction=1,
    description="过去 120 个交易日收益",
    formula=r"r_{t-120 \to t}",
)
def mom_6m(md) -> pd.DataFrame:
    return md.close() / md.close().shift(120) - 1.0


@factor_from_func(
    name="mom_12_1", category="动量", direction=1,
    description="12 个月动量剔除最近 1 个月（经典学术动量，规避短期反转）",
    formula=r"\frac{P_{t-250}}{P_{t-21}} - 1",
)
def mom_12_1(md) -> pd.DataFrame:
    return md.close().shift(21) / md.close().shift(250) - 1.0


@factor_from_func(
    name="rev_5d", category="反转", direction=-1,
    description="过去 5 日收益（短期反转：涨得多→未来弱）",
    formula=r"r_{t-5 \to t}",
)
def rev_5d(md) -> pd.DataFrame:
    return md.close() / md.close().shift(5) - 1.0


@factor_from_func(
    name="rev_20d", category="反转", direction=-1,
    description="过去 20 日收益（中期反转：一个月涨得多→下个月弱）",
    formula=r"r_{t-20 \to t}",
)
def rev_20d(md) -> pd.DataFrame:
    return md.close() / md.close().shift(20) - 1.0


@factor_from_func(
    name="rev_1d", category="反转", direction=-1,
    description="昨日收益（极短期反转）",
    formula=r"r_{t-1 \to t}",
)
def rev_1d(md) -> pd.DataFrame:
    return md.close() / md.close().shift(1) - 1.0


@factor_from_func(
    name="mom_rev_combo", category="反转", direction=-1,
    description="短期反转 + 中期动量的组合（1 月反转 × 12-1 动量）",
    formula=r"-\mathrm{rev}_{1m} \times \mathrm{sign}(\mathrm{mom}_{12-1})",
)
def mom_rev_combo(md) -> pd.DataFrame:
    r1 = md.close() / md.close().shift(21) - 1.0
    m = md.close().shift(21) / md.close().shift(250) - 1.0
    return -r1 * np.sign(m)


@factor_from_func(
    name="mom_vol_adj", category="动量", direction=1,
    description="波动率调整动量：用已实现波动缩放动量，降低高波动股的权重",
    formula=r"\frac{r_{t-250\to t-21}}{\sigma_{250}}",
)
def mom_vol_adj(md) -> pd.DataFrame:
    ret = md.returns()
    mom = md.close().shift(21) / md.close().shift(250) - 1.0
    vol = ret.rolling(250, min_periods=120).std()
    return mom / vol.replace(0, np.nan)


@factor_from_func(
    name="mom_consistency", category="动量", direction=1,
    description="动量一致性：上涨天数占比（趋势平稳度）",
    formula=r"\frac{1}{250}\sum_{i=t-249}^{t}\mathbf{1}[r_i>0]",
)
def mom_consistency(md) -> pd.DataFrame:
    ret = md.returns()
    return (ret > 0).rolling(250, min_periods=120).mean()


@factor_from_func(
    name="dist_to_high_252", category="动量", direction=1,
    description="当前价 / 过去一年最高价（越接近新高越强）",
    formula=r"\frac{P_t}{\max_{i\le 252} P_{t-i}}",
)
def dist_to_high_252(md) -> pd.DataFrame:
    px = md.close()
    return px / px.rolling(252, min_periods=120).max()


@factor_from_func(
    name="dist_to_low_252", category="反转", direction=-1,
    description="当前价 / 过去一年最低价（越高越远离底部，反向）",
    formula=r"\frac{P_t}{\min_{i\le 252} P_{t-i}}",
)
def dist_to_low_252(md) -> pd.DataFrame:
    px = md.close()
    return px / px.rolling(252, min_periods=120).min()


# ---------------------------------------------------------------- 价值


@factor_from_func(
    name="bp", category="价值", direction=1,
    description="账面市值比（B/P = 1/PB），经典价值因子",
    formula=r"\frac{1}{P/B}",
    required_fields=("close", "pb"),
)
def bp(md) -> pd.DataFrame:
    return 1.0 / md.fundamental("pb").replace(0, np.nan)


@factor_from_func(
    name="ep", category="价值", direction=1,
    description="盈利收益率（E/P = 1/PE）",
    formula=r"\frac{1}{P/E}",
    required_fields=("close", "pe"),
)
def ep(md) -> pd.DataFrame:
    return 1.0 / md.fundamental("pe").replace(0, np.nan)


@factor_from_func(
    name="sp", category="价值", direction=1,
    description="销售市值比（S/P = 1/PS），对亏损公司更稳健",
    formula=r"\frac{1}{P/S}",
    required_fields=("close", "ps"),
)
def sp(md) -> pd.DataFrame:
    return 1.0 / md.fundamental("ps").replace(0, np.nan)


@factor_from_func(
    name="ln_mktcap", category="价值", direction=-1,
    description="对数市值（小市值因子；A 股长期显著但容量有限）",
    formula=r"-\ln(\mathrm{mktcap})",
    required_fields=("close", "mktcap"),
)
def ln_mktcap(md) -> pd.DataFrame:
    return np.log(md.fundamental("mktcap").replace(0, np.nan))


@factor_from_func(
    name="value_composite", category="价值", direction=1,
    description="价值复合：E/P、B/P、S/P 的横截面排名均值",
    formula=r"\overline{\mathrm{rank}}\{E/P,\;B/P,\;S/P\}",
    required_fields=("close", "pe", "pb", "ps"),
)
def value_composite(md) -> pd.DataFrame:
    parts = [1.0 / md.fundamental(k).replace(0, np.nan) for k in ("pe", "pb", "ps")]
    ranks = [p.rank(axis=1, pct=True) for p in parts]
    return sum(ranks) / len(ranks)


# ---------------------------------------------------------------- 成长 / 质量


@factor_from_func(
    name="roe", category="质量", direction=1,
    description="净资产收益率（质量因子）",
    formula=r"\mathrm{ROE}",
    required_fields=("close", "roe"),
)
def roe(md) -> pd.DataFrame:
    return md.fundamental("roe")


@factor_from_func(
    name="roe_stability", category="质量", direction=1,
    description="ROE 稳定性：过去 250 日 ROE 标准差的相反数",
    formula=r"-\sigma_{250}(\mathrm{ROE})",
    required_fields=("close", "roe"),
)
def roe_stability(md) -> pd.DataFrame:
    r = md.fundamental("roe")
    return -r.rolling(250, min_periods=120).std()


@factor_from_func(
    name="roe_trend", category="成长", direction=1,
    description="ROE 趋势：近 60 日均值 − 前 60 日均值",
    formula=r"\overline{\mathrm{ROE}}_{60} - \overline{\mathrm{ROE}}_{t-120\to t-60}",
    required_fields=("close", "roe"),
)
def roe_trend(md) -> pd.DataFrame:
    r = md.fundamental("roe")
    return r.rolling(60, min_periods=30).mean() - r.rolling(120, min_periods=60).mean().shift(60)


@factor_from_func(
    name="earnings_growth_proxy", category="成长", direction=1,
    description="盈利增长代理：由 E/P 的历史变化反推的盈利增速（合成/财报数据均可用）",
    formula=r"\frac{E_t/E_{t-250} - 1}{1}",
    required_fields=("close", "pe"),
)
def earnings_growth_proxy(md) -> pd.DataFrame:
    # E = mktcap / PE  (mktcap 若不存在则用 close 代替，量纲不影响横截面排序)
    cap = md.fundamental("mktcap") if "mktcap" in md else md.close()
    e = cap / md.fundamental("pe").replace(0, np.nan)
    return e / e.shift(250) - 1.0


# ---------------------------------------------------------------- 波动


@factor_from_func(
    name="vol_20d", category="波动", direction=-1,
    description="20 日已实现波动率（低波动异象：波动越高未来收益越低）",
    formula=r"-\sigma_{20}",
)
def vol_20d(md) -> pd.DataFrame:
    return md.returns().rolling(20, min_periods=15).std()


@factor_from_func(
    name="vol_60d", category="波动", direction=-1,
    description="60 日已实现波动率",
    formula=r"-\sigma_{60}",
)
def vol_60d(md) -> pd.DataFrame:
    return md.returns().rolling(60, min_periods=40).std()


@factor_from_func(
    name="vol_250d", category="波动", direction=-1,
    description="250 日已实现波动率",
    formula=r"-\sigma_{250}",
)
def vol_250d(md) -> pd.DataFrame:
    return md.returns().rolling(250, min_periods=120).std()


@factor_from_func(
    name="idiosyncratic_vol", category="波动", direction=-1,
    description="特质波动率：对市值加权市场收益回归后的残差波动",
    formula=r"-\sigma(\varepsilon_i),\quad r_i=\alpha_i+\beta_i r_m+\varepsilon_i",
)
def idiosyncratic_vol(md) -> pd.DataFrame:
    ret = md.returns()
    cap = md.fundamental("mktcap") if "mktcap" in md else pd.DataFrame(1.0, index=ret.index, columns=ret.columns)
    w = cap.div(cap.sum(axis=1), axis=0)
    mkt = (ret * w).sum(axis=1)
    out = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    window = 120
    for i in range(window, len(ret)):
        y = ret.iloc[i - window : i]
        x = mkt.iloc[i - window : i].values
        xv = np.column_stack([np.ones(len(x)), x])
        b, *_ = np.linalg.lstsq(xv, y.values, rcond=None)
        resid = y.values - xv @ b
        out.iloc[i] = resid.std(axis=0)
    return out


@factor_from_func(
    name="downside_vol", category="波动", direction=-1,
    description="下行波动率（只统计负收益的波动，更贴近真实风险厌恶）",
    formula=r"-\sqrt{\frac{1}{n}\sum \min(r_i,0)^2}",
)
def downside_vol(md) -> pd.DataFrame:
    ret = md.returns()
    neg = ret.clip(upper=0)
    return -np.sqrt((neg ** 2).rolling(60, min_periods=40).mean())


@factor_from_func(
    name="beta_250d", category="波动", direction=-1,
    description="市场 beta（对市值加权市场收益回归）",
    formula=r"-\beta_i",
)
def beta_250d(md) -> pd.DataFrame:
    ret = md.returns()
    cap = md.fundamental("mktcap") if "mktcap" in md else pd.DataFrame(1.0, index=ret.index, columns=ret.columns)
    w = cap.div(cap.sum(axis=1), axis=0)
    mkt = (ret * w).sum(axis=1)
    cov = ret.rolling(250, min_periods=120).cov(mkt)
    var = mkt.rolling(250, min_periods=120).var()
    return cov.div(var, axis=0)


@factor_from_func(
    name="max_drawdown_250d", category="波动", direction=-1,
    description="过去一年最大回撤（越大越危险）",
    formula=r"-\mathrm{MDD}_{250}",
)
def max_drawdown_250d(md) -> pd.DataFrame:
    px = md.close()
    roll_max = px.rolling(250, min_periods=120).max()
    dd = px / roll_max - 1.0
    return -dd.rolling(250, min_periods=120).min()


# ---------------------------------------------------------------- 流动性 / 情绪


@factor_from_func(
    name="turnover_20d", category="流动性", direction=-1,
    description="20 日平均换手代理（成交量/流通市值），高换手常伴随投机",
    formula=r"-\overline{\mathrm{turnover}}_{20}",
    required_fields=("close", "volume"),
)
def turnover_20d(md) -> pd.DataFrame:
    vol = md.volume()
    cap = md.fundamental("float_mktcap") if "float_mktcap" in md else md.close()
    to = vol / cap.replace(0, np.nan)
    return to.rolling(20, min_periods=10).mean()


@factor_from_func(
    name="amihud_illiquidity", category="流动性", direction=1,
    description="Amihud 非流动性：单位成交额引起的价格变动（越大越难交易，要求溢价）",
    formula=r"\frac{1}{n}\sum \frac{|r_i|}{\mathrm{amount}_i}",
    required_fields=("close", "volume"),
)
def amihud_illiquidity(md) -> pd.DataFrame:
    ret = md.returns().abs()
    amt = md.amount().replace(0, np.nan)
    return (ret / amt).rolling(20, min_periods=10).mean()


@factor_from_func(
    name="dollar_volume_20d", category="流动性", direction=-1,
    description="20 日日均成交额（取对数；小成交额有小盘溢价与流动性风险）",
    formula=r"-\ln(\overline{\mathrm{amount}}_{20})",
    required_fields=("close", "volume"),
)
def dollar_volume_20d(md) -> pd.DataFrame:
    return np.log(md.amount().rolling(20, min_periods=10).mean().replace(0, np.nan))


@factor_from_func(
    name="volume_trend", category="情绪", direction=1,
    description="量能趋势：近 5 日成交量 / 前 60 日成交量",
    formula=r"\frac{\overline{V}_5}{\overline{V}_{60}}",
    required_fields=("close", "volume"),
)
def volume_trend(md) -> pd.DataFrame:
    v = md.volume()
    return v.rolling(5, min_periods=3).mean() / v.rolling(60, min_periods=40).mean().replace(0, np.nan)


@factor_from_func(
    name="volume_surge", category="情绪", direction=1,
    description="放量异动：当日成交量相对 60 日均量的倍数（对数）",
    formula=r"\ln\frac{V_t}{\overline{V}_{60}}",
    required_fields=("close", "volume"),
)
def volume_surge(md) -> pd.DataFrame:
    v = md.volume().replace(0, np.nan)
    return np.log(v / v.rolling(60, min_periods=40).mean())


@factor_from_func(
    name="skewness_60d", category="情绪", direction=-1,
    description="60 日收益偏度（彩票型股票偏好：高偏度股票未来收益低）",
    formula=r"-\mathrm{skew}_{60}(r)",
)
def skewness_60d(md) -> pd.DataFrame:
    return md.returns().rolling(60, min_periods=40).skew()


# ---------------------------------------------------------------- 技术


@factor_from_func(
    name="rsi_14", category="技术", direction=-1,
    description="14 日 RSI（超买→未来弱，取反号做因子）",
    formula=r"-\mathrm{RSI}_{14}",
)
def rsi_14(md) -> pd.DataFrame:
    return _rsi(md.close(), 14)


@factor_from_func(
    name="rsi_2", category="技术", direction=-1,
    description="2 日 RSI（极短期均值回归常用）",
    formula=r"-\mathrm{RSI}_{2}",
)
def rsi_2(md) -> pd.DataFrame:
    return _rsi(md.close(), 2)


@factor_from_func(
    name="ma_ratio_20_60", category="技术", direction=1,
    description="20 日均线 / 60 日均线（趋势强度）",
    formula=r"\frac{\mathrm{MA}_{20}}{\mathrm{MA}_{60}}",
)
def ma_ratio_20_60(md) -> pd.DataFrame:
    px = md.close()
    return px.rolling(20, min_periods=10).mean() / px.rolling(60, min_periods=30).mean().replace(0, np.nan)


@factor_from_func(
    name="price_vs_ma20", category="技术", direction=1,
    description="价格 / 20 日均线 − 1（偏离度，配合反转或动量用）",
    formula=r"\frac{P_t}{\mathrm{MA}_{20}}-1",
)
def price_vs_ma20(md) -> pd.DataFrame:
    px = md.close()
    return px / px.rolling(20, min_periods=10).mean() - 1.0


@factor_from_func(
    name="bollinger_pctb", category="技术", direction=-1,
    description="布林带 %B：价格在上轨与下轨间的相对位置（1=触上轨）",
    formula=r"\frac{P - \mathrm{MA}+2\sigma}{4\sigma}",
)
def bollinger_pctb(md) -> pd.DataFrame:
    px = md.close()
    ma = px.rolling(20, min_periods=10).mean()
    sd = px.rolling(20, min_periods=10).std()
    return (px - (ma - 2 * sd)) / (4 * sd).replace(0, np.nan)


@factor_from_func(
    name="atr_ratio", category="技术", direction=-1,
    description="ATR / 价格（波动幅度占比）",
    formula=r"-\frac{\mathrm{ATR}_{14}}{P}",
    required_fields=("open", "high", "low", "close"),
)
def atr_ratio(md) -> pd.DataFrame:
    high, low, close = md.high(), md.low(), md.close()
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ]).groupby(level=0).max()
    atr = tr.rolling(14, min_periods=7).mean()
    return atr / close.replace(0, np.nan)


@factor_from_func(
    name="macd_hist", category="技术", direction=1,
    description="MACD 柱（12,26,9），趋势动量",
    formula=r"\mathrm{EMA}_{12}-\mathrm{EMA}_{26}-\mathrm{EMA}_9(\cdot)",
)
def macd_hist(md) -> pd.DataFrame:
    px = md.close()
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


@factor_from_func(
    name="breakout_20d", category="技术", direction=1,
    description="20 日突破：价格 / 20 日最高价",
    formula=r"\frac{P_t}{\max_{20}P}",
)
def breakout_20d(md) -> pd.DataFrame:
    px = md.close()
    return px / px.rolling(20, min_periods=10).max()


@factor_from_func(
    name="gap_open", category="技术", direction=-1,
    description="跳空缺口：开盘价 / 昨收 − 1（隔夜情绪）",
    formula=r"-\left(\frac{O_t}{P_{t-1}}-1\right)",
    required_fields=("open", "close"),
)
def gap_open(md) -> pd.DataFrame:
    return md.open() / md.close().shift(1) - 1.0


@factor_from_func(
    name="intraday_reversal", category="技术", direction=-1,
    description="日内反转：收盘价 / 开盘价 − 1",
    formula=r"-\left(\frac{C_t}{O_t}-1\right)",
    required_fields=("open", "close"),
)
def intraday_reversal(md) -> pd.DataFrame:
    return md.close() / md.open().replace(0, np.nan) - 1.0


@factor_from_func(
    name="vwap_deviation", category="技术", direction=-1,
    description="收盘价相对 VWAP 的偏离（机构行为痕迹）",
    formula=r"-\left(\frac{C_t}{\mathrm{VWAP}_t}-1\right)",
    required_fields=("close", "volume"),
)
def vwap_deviation(md) -> pd.DataFrame:
    return md.close() / md.vwap().replace(0, np.nan) - 1.0


# ---------------------------------------------------------------- 另类 / 复合


@factor_from_func(
    name="quality_composite", category="质量", direction=1,
    description="质量复合：ROE 与 ROE 稳定性的排名均值",
    formula=r"\overline{\mathrm{rank}}\{\mathrm{ROE},\;-\sigma(\mathrm{ROE})\}",
    required_fields=("close", "roe"),
)
def quality_composite(md) -> pd.DataFrame:
    r = md.fundamental("roe")
    stab = -r.rolling(250, min_periods=120).std()
    return (r.rank(axis=1, pct=True) + stab.rank(axis=1, pct=True)) / 2


@factor_from_func(
    name="low_risk_composite", category="波动", direction=1,
    description="低风险复合：低波动 + 低 beta + 低回撤 的排名均值",
    formula=r"\overline{\mathrm{rank}}\{-\sigma,\;-\beta,\;-\mathrm{MDD}\}",
)
def low_risk_composite(md) -> pd.DataFrame:
    v = vol_60d(md)
    b = beta_250d(md)
    px = md.close()
    dd = px / px.rolling(250, min_periods=120).max() - 1.0
    mdd = -dd.rolling(250, min_periods=120).min()
    return (-v).rank(axis=1, pct=True) / 3 + (-b).rank(axis=1, pct=True) / 3 + (-mdd).rank(axis=1, pct=True) / 3


@factor_from_func(
    name="smart_money", category="另类", direction=1,
    description="聪明钱代理：放量上涨 / 缩量下跌的强度（量与价的交互）",
    formula=r"\mathrm{rank}(\Delta\ln V)\cdot \mathrm{sign}(r)",
    required_fields=("close", "volume"),
)
def smart_money(md) -> pd.DataFrame:
    v = np.log(md.volume().replace(0, np.nan).rolling(5, min_periods=3).mean())
    dv = v.diff(5)
    r = md.close() / md.close().shift(5) - 1.0
    return dv.rank(axis=1, pct=True) * np.sign(r)


@factor_from_func(
    name="residual_momentum", category="动量", direction=1,
    description="残差动量：剔除市场因子后的动量（更纯粹的个股趋势）",
    formula=r"\mathrm{mom}_{12-1} - \hat\beta\, \mathrm{mom}^{mkt}_{12-1}",
)
def residual_momentum(md) -> pd.DataFrame:
    ret = md.returns()
    cap = md.fundamental("mktcap") if "mktcap" in md else pd.DataFrame(1.0, index=ret.index, columns=ret.columns)
    w = cap.div(cap.sum(axis=1), axis=0)
    mkt_ret = (ret * w).sum(axis=1)
    mom = md.close().shift(21) / md.close().shift(250) - 1.0
    mkt_mom = mkt_ret.shift(1).rolling(229, min_periods=120).sum()
    beta = ret.rolling(250, min_periods=120).cov(mkt_ret).div(
        mkt_ret.rolling(250, min_periods=120).var(), axis=0
    )
    return mom - beta * mkt_mom.values[:, None]


def _rsi(px: pd.DataFrame, window: int) -> pd.DataFrame:
    delta = px.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


from .base import FACTOR_REGISTRY  # noqa: E402

BUILTIN_FACTORS = sorted(FACTOR_REGISTRY)
