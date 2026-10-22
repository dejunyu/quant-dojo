"""交易日历工具.

说明
----
一个「最优秀」的量化平台必须对交易日历诚实：

* 本模块默认使用 **周一至周五** 作为交易日，并允许传入节假日集合；
* 当安装了 ``akshare`` 且显式指定 ``use_akshare=True`` 时，会用上交所真实
  交易日历（需要联网）覆盖默认结果；
* 因此内置的 ``trading_days`` 只用于合成数据与教学场景，**实盘/研究请以
  交易所或数据源的真实日历为准**。
"""
from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from typing import Iterable, Sequence

import pandas as pd

__all__ = ["trading_days", "shift_date", "is_trading_day"]

_WEEKEND = {5, 6}  # 周六、周日


def _coerce(day) -> pd.Timestamp:
    if isinstance(day, pd.Timestamp):
        return day.normalize()
    return pd.Timestamp(day).normalize()


def trading_days(
    start,
    end,
    market: str = "A",
    holidays: Iterable | None = None,
    use_akshare: bool = False,
) -> pd.DatetimeIndex:
    """返回 [start, end] 区间内的交易日序列.

    Parameters
    ----------
    start, end :
        起止日期（含两端）。
    market :
        ``"A"`` 沪深（周一至周五）；``"US"``/``"HK"`` 同样按周一至周五处理，
        但可通过 ``holidays`` 剔除各自假期；``"CRYPTO"`` 返回 7×24 的自然日。
    holidays :
        额外剔除的日期集合。
    use_akshare :
        为 True 且 market 为 ``"A"`` 时，尝试用 akshare 获取上交所真实日历。
    """
    s, e = _coerce(start), _coerce(end)
    if s > e:
        raise ValueError(f"起始日期 {s.date()} 晚于结束日期 {e.date()}")

    if market.upper() == "CRYPTO":
        return pd.date_range(s, e, freq="D", name="date")

    days = pd.date_range(s, e, freq="D")
    days = days[[d.weekday() not in _WEEKEND for d in days]]

    if market.upper() == "A" and use_akshare:
        real = _akshare_calendar()
        if real is not None:
            days = real[(real >= s) & (real <= e)]

    if holidays:
        drop = {_coerce(h) for h in holidays}
        days = days[[d not in drop for d in days]]

    return pd.DatetimeIndex(days, name="date")


@lru_cache(maxsize=1)
def _akshare_calendar() -> pd.DatetimeIndex | None:
    try:
        import akshare as ak  # type: ignore

        df = ak.tool_trade_date_hist_sina()
        return pd.DatetimeIndex(pd.to_datetime(df["trade_date"]), name="date")
    except Exception:  # noqa: BLE001 - 网络不可用等任何错误都退回内置日历
        return None


def is_trading_day(day, market: str = "A") -> bool:
    """判断某日是否为交易日（仅按周一至周五粗略判断）。"""
    return _coerce(day).weekday() not in _WEEKEND or market.upper() == "CRYPTO"


def shift_date(anchor, n: int, market: str = "A", holidays: Iterable | None = None) -> pd.Timestamp:
    """以交易日为单位做日期位移，``n`` 可为负。"""
    anchor = _coerce(anchor)
    step = 1 if n >= 0 else -1
    remaining = abs(n)
    cur = anchor
    while remaining > 0:
        cur = cur + pd.Timedelta(days=step)
        if is_trading_day(cur, market):
            if holidays and _coerce(cur) in {_coerce(h) for h in holidays}:
                continue
            remaining -= 1
    return cur
