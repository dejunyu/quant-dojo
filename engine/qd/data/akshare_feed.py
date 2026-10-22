"""A 股数据源（akshare）.

设计原则
--------
1. **联网是可选的**：本模块只在显式调用时才联网，import 时不产生任何请求。
   平台的全部教学与回测流程都能用 ``qd.data.synthetic`` 离线跑通。
2. **失败要说人话**：网络不通 / 接口变更 / 代码不存在时给出明确的中文错误，
   而不是抛一堆 traceback。
3. **复权要说清楚**：默认取**前复权**（qfq），因为绝大多数横截面因子研究
   都应该用前复权价格；做事件研究或需要真实成交价时用不复权。
4. **缓存**：下载过的数据落在本地 parquet，下次直接读，避免反复请求被封。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .market import MarketData

__all__ = ["AkshareFeed", "load_a_shares", "INDEX_SYMBOLS"]

# 常用指数与 ETF 代码（akshare 的 spot 接口支持这些代码）
INDEX_SYMBOLS = {
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "000016": "上证50",
    "399006": "创业板指",
    "000001": "上证指数",
    "399001": "深证成指",
}


class AkshareFeed:
    """A 股日线数据下载器（基于 akshare）。

    Parameters
    ----------
    cache_dir :
        本地缓存目录；``None`` 时禁用缓存。
    adjust :
        ``"qfq"`` 前复权（默认）、``"hfq"`` 后复权、``""`` 不复权。
    """

    def __init__(self, cache_dir=None, adjust: str = "qfq", cache_days: int = 1,
                 pause: float = 1.0, retries: int = 3):
        """
        pause :
            每次请求之间的间隔（秒）。akshare 走的是公开网页接口，请求太快
            会被限流——批量下载时 1 秒的间隔能省掉大量失败的重试。
        retries :
            单个标的失败后的重试次数（指数退避）。
        """
        self.adjust = adjust
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.cache_days = int(cache_days)
        self.pause = float(pause)
        self.retries = int(retries)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 单标的
    def fetch_one(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """下载单只股票/指数的日线数据（带缓存）。"""
        key = f"{symbol}_{start}_{end}_{self.adjust or 'raw'}"
        cache_file = self.cache_dir / f"{key}.parquet" if self.cache_dir else None
        if cache_file and cache_file.exists():
            age_days = (time.time() - cache_file.stat().st_mtime) / 86400
            if age_days < self.cache_days:
                return pd.read_parquet(cache_file)

        try:
            import akshare as ak  # noqa: PLC0415 - 延迟导入，保证离线可用
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "未安装 akshare。A 股真实数据需要它：pip install akshare；\n"
                "不想联网也可以用合成数据：from qd.data.synthetic import generate_demo_market"
            ) from exc

        s = pd.Timestamp(start).strftime("%Y%m%d")
        e = pd.Timestamp(end).strftime("%Y%m%d")
        # 多数据源回退：eastmoney（默认）失败时自动换腾讯源。
        # 实测在国内网络下，eastmoney 的 kline 接口经常被断连，而腾讯源稳定。
        sources = [
            ("eastmoney", lambda: ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=s, end_date=e, adjust=self.adjust)),
            ("tencent", lambda: ak.stock_zh_a_hist_tx(
                symbol=("sh" if symbol.startswith(("6", "9")) else "sz") + symbol,
                start_date=s.replace("-", ""), end_date=e.replace("-", ""),
                adjust=self.adjust if self.adjust in ("qfq", "hfq", "") else "qfq")),
        ]

        last_exc: Exception | None = None
        df = None
        used_source = None
        for source_name, fetch in sources:
            for attempt in range(1, self.retries + 1):
                try:
                    df = fetch()
                    used_source = source_name
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt < self.retries:
                        # 指数退避：被限流时硬重试只会更糟
                        time.sleep(self.pause * (2 ** attempt))
            if df is not None and len(df):
                break
        if df is None or not len(df):
            raise RuntimeError(
                f"下载 {symbol} 失败（eastmoney 与 tencent 两个源共重试 "
                f"{self.retries * 2} 次）：{last_exc}\n"
                f"可能原因：\n"
                f"  1. 网络不通或被数据源限流（可调大 AkshareFeed(pause=...)）；\n"
                f"  2. akshare 接口变更；\n"
                f"  3. 代码不存在（A 股六位数字，如 600519）。\n"
                f"可以先试合成数据：generate_demo_market()"
            ) from last_exc

        if df is None or df.empty:
            raise ValueError(f"{symbol} 在 {start}~{end} 没有数据")

        rename = {
            "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
            "最低": "low", "成交量": "volume", "成交额": "amount",
            "振幅": "amplitude", "涨跌幅": "pct_change",
            "涨跌额": "change", "换手率": "turnover_rate",
            # 腾讯源（stock_zh_a_hist_tx）的列名
            "date": "date", "open": "open", "close": "close", "high": "high",
            "low": "low", "volume": "volume", "turnover": "turnover_rate",
            "amount": "amount",
        }
        df = df.rename(columns=rename)
        if "date" not in df.columns:
            raise RuntimeError(f"akshare 返回字段异常：{list(df.columns)}，接口可能已变更")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        keep = [c for c in ("open", "high", "low", "close", "volume", "amount", "turnover_rate") if c in df.columns]
        out = df[keep].astype(float)

        if cache_file:
            out.to_parquet(cache_file)
        out.attrs["source"] = used_source
        return out

    # ------------------------------------------------------------ 多标的
    def load(self, symbols, start: str, end: str, min_days: int = 60,
             drop_incomplete: bool = True) -> MarketData:
        """下载多只标的并组装成 MarketData.

        Parameters
        ----------
        symbols :
            代码列表（A 股六位数字，或 INDEX_SYMBOLS 里的指数代码）。
        min_days :
            少于该天数的标的会被剔除（上市不足 / 长期停牌）。
        """
        symbols = [str(s) for s in symbols]
        frames, failed = {}, []
        for k, sym in enumerate(symbols):
            if self.pause > 0 and k > 0:
                time.sleep(self.pause)
            try:
                df = self.fetch_one(sym, start, end)
                if len(df) >= (min_days if drop_incomplete else 1):
                    frames[sym] = df
                elif not drop_incomplete:
                    frames[sym] = df
            except Exception as exc:  # noqa: BLE001
                failed.append((sym, str(exc)[:80]))

        if not frames:
            msg = "所有标的都下载失败"
            if failed:
                msg += "：" + "; ".join(f"{s} -> {e}" for s, e in failed[:5])
            raise RuntimeError(msg)

        md = MarketData.from_dataframes(frames, name=f"akshare-{len(frames)}")
        md.meta.update({
            "source": "akshare", "adjust": self.adjust,
            "requested": len(symbols), "loaded": len(frames),
            "failed": failed,
        })
        return md

    # ------------------------------------------------------------ 指数成分
    def index_components(self, index: str = "000300") -> list[str]:
        """获取指数成分股代码列表（沪深300 / 中证500 等）。"""
        try:
            import akshare as ak  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError("需要 akshare：pip install akshare") from exc
        try:
            df = ak.index_stock_cons(symbol=index)
            col = "品种代码" if "品种代码" in df.columns else df.columns[1]
            return [str(c).zfill(6) for c in df[col].tolist()]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"获取 {index} 成分股失败：{exc}") from exc

    def all_a_symbols(self) -> pd.DataFrame:
        """A 股全部股票代码与名称快照。"""
        try:
            import akshare as ak  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError("需要 akshare：pip install akshare") from exc
        return ak.stock_zh_a_spot_em()[["代码", "名称"]]


def load_a_shares(symbols, start="2020-01-01", end="2024-12-31",
                  adjust="qfq", cache_dir=None) -> MarketData:
    """一行下载 A 股数据的便捷函数。"""
    return AkshareFeed(cache_dir=cache_dir, adjust=adjust).load(symbols, start, end)
