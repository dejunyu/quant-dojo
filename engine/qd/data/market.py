"""行情数据容器.

设计要点
--------
``MarketData`` 用「面板（panel）」而不是「多列 DataFrame 」组织数据：
每个字段（open/high/low/close/volume/...）是一个 ``dates × symbols`` 的
DataFrame。这样做的收益是：

1. 因子计算天然是面板运算，可以整块向量化，不需要 groupby；
2. 缺列（例如只有收盘价）不会污染其它字段；
3. 基本面字段（pe/pb/roe/mktcap）与量价字段用同一套存取接口。

所有价格在构造时都会做基础校验（数值型、索引升序、不重复），但**不会**
替用户猜测复权方式——复权必须由数据源负责（见 :mod:`qd.data.akshare_feed`）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..utils.misc import to_datetime

__all__ = ["MarketData"]

_PRICE_FIELDS = ("open", "high", "low", "close", "vwap")
_REQUIRED = ("open", "high", "low", "close", "volume")

_COLUMN_ALIASES = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "vol": "volume",
    "vwap": "vwap",
    "amount": "amount",
    "turnover": "amount",
    "adj_factor": "adj_factor",
    "pe": "pe",
    "pb": "pb",
    "ps": "ps",
    "roe": "roe",
    "mktcap": "mktcap",
    "total_mv": "mktcap",
    "circ_mv": "float_mktcap",
    "float_mktcap": "float_mktcap",
    "industry": "industry",
}


class MarketData:
    """多标的行情容器.

    Parameters
    ----------
    panels :
        字段 -> ``dates × symbols`` DataFrame 的映射。至少包含
        open/high/low/close/volume。
    fundamentals :
        基本面/静态字段 -> DataFrame。与行情共享同一 ``dates × symbols`` 网格
        （静态字段会前向填充）。
    name :
        数据集名称，用于报告标题。
    """

    def __init__(
        self,
        panels: Mapping[str, pd.DataFrame],
        fundamentals: Mapping[str, pd.DataFrame] | None = None,
        name: str = "market",
        meta: dict | None = None,
    ):
        if not isinstance(panels, Mapping):
            raise TypeError("panels 必须是 字段->DataFrame 的映射")

        self.panels: dict[str, pd.DataFrame] = {}
        for field, df in panels.items():
            self.panels[str(field).lower()] = self._normalize(df, field)

        missing = [f for f in _REQUIRED if f not in self.panels]
        if missing:
            raise ValueError(
                f"缺少必需字段 {missing}；行情至少需要 open/high/low/close/volume"
            )

        self._dates = self.panels["close"].index
        self._symbols = self.panels["close"].columns
        for field, df in self.panels.items():
            if not df.index.equals(self._dates):
                raise ValueError(f"字段 {field} 的日期索引与 close 不一致")
            if not df.columns.equals(self._symbols):
                raise ValueError(f"字段 {field} 的标的行列与 close 不一致")

        self.fundamentals: dict[str, pd.DataFrame] = {}
        for field, df in (fundamentals or {}).items():
            self.fundamentals[str(field).lower()] = self._align(df)

        self.name = name
        self.meta = dict(meta or {})

    # ---------------------------------------------------------------- 构造
    @staticmethod
    def _normalize(df: pd.DataFrame, field: str) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"字段 {field} 必须是 DataFrame，收到 {type(df)}")
        out = df.copy()
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).normalize()
        if not out.index.is_monotonic_increasing:
            out = out.sort_index()
        if out.index.has_duplicates:
            out = out[~out.index.duplicated(keep="last")]
        out.columns = [str(c) for c in out.columns]
        out = out.astype(float, errors="ignore")
        return out

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        """把任意 DataFrame 对齐到统一的 dates × symbols 网格。"""
        out = self._normalize(df, "fundamental")
        out = out.reindex(index=self._dates, columns=self._symbols)
        return out

    @classmethod
    def from_dataframes(
        cls,
        frames: Mapping[str, pd.DataFrame],
        field_map: Mapping[str, str] | None = None,
        name: str = "market",
    ) -> "MarketData":
        """从 ``{symbol: DataFrame}`` 构造 MarketData.

        Parameters
        ----------
        frames :
            标的代码 -> 单标的 OHLCV DataFrame（列为 open/high/low/close/volume，
            支持 ``vol``/``成交额`` 等别名）。

            **也接受面板写法**：``{"close": dates×symbols, "open": ..., ...}``，
            或者直接传单个 ``dates × symbols`` 的 DataFrame（此时视为 close 面板）。
            很多研究代码里字段恰好排在前面几个行，这两种形式都很常见。
        field_map :
            自定义列名映射，会与内置别名合并（优先级更高）。

        Examples
        --------
        ```python
        # 形式一：每个标的一份数据
        md = MarketData.from_dataframes({"600519": df1, "000858": df2})

        # 形式二：面板写法（字段 -> 面板）
        md = MarketData.from_dataframes({"close": px, "volume": vol})
        ```
        """
        if isinstance(frames, pd.DataFrame):
            if frames.empty:
                raise ValueError("frames 为空")
            return cls.from_panel(frames.rename(columns=str))
        if not frames:
            raise ValueError("frames 为空")

        # ---- 形式二：单个 DataFrame 视为 close 面板 ----

        # ---- 形式二：{"close": panel, ...} 字段 -> 面板 ----
        keys = list(frames.keys())
        panel_like = [
            k for k in keys
            if isinstance(frames[k], pd.DataFrame)
            and str(k).strip().lower() in _COLUMN_ALIASES
        ]
        if panel_like and all(isinstance(frames[k], pd.DataFrame) for k in keys):
            panels: dict[str, pd.DataFrame] = {}
            for k in keys:
                target = _COLUMN_ALIASES.get(str(k).strip().lower(), str(k).strip().lower())
                panels[target] = frames[k]
            if "close" not in panels:
                raise ValueError(
                    f"面板写法必须包含 close 字段；当前字段：{keys}"
                )
            # 补齐必需字段
            for req in ("open", "high", "low"):
                panels.setdefault(req, panels["close"])
            panels.setdefault(
                "volume", pd.DataFrame(0.0, index=panels["close"].index, columns=panels["close"].columns)
            )
            return cls(panels, name=name)

        aliases = dict(_COLUMN_ALIASES)
        if field_map:
            aliases.update({k.lower(): v for k, v in field_map.items()})

        per_field: dict[str, dict[str, pd.Series]] = {}
        fund_frames: dict[str, pd.DataFrame] = {}

        for symbol, df in frames.items():
            if df is None or len(df) == 0:
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            renamed = {str(c): aliases.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns}
            d = df.rename(columns=renamed)
            for col in d.columns:
                series = d[col]
                if not pd.api.types.is_numeric_dtype(series):
                    continue
                per_field.setdefault(col, {})[str(symbol)] = series

        if "close" not in per_field:
            raise ValueError("未找到 close 列，请检查列名或 field_map")

        # 合成 volume 缺失时补 0（不影响价格类因子）
        panels: dict[str, pd.DataFrame] = {}
        for field, cols in per_field.items():
            panels[field] = pd.DataFrame(cols).sort_index()

        for req in ("open", "high", "low"):
            if req not in panels:
                panels[req] = panels["close"]
        if "volume" not in panels:
            panels["volume"] = pd.DataFrame(0.0, index=panels["close"].index, columns=panels["close"].columns)

        fundamentals = {k: v for k, v in fund_frames.items()} if fund_frames else None
        return cls(panels, fundamentals=fundamentals, name=name)

    @classmethod
    def from_panel(cls, close: pd.DataFrame, **others) -> "MarketData":
        """直接由面板构造（close 必填，其余缺省用 close / 0 填充）。"""
        close = close.copy()
        panels = {
            "open": others.pop("open", close),
            "high": others.pop("high", close),
            "low": others.pop("low", close),
            "close": close,
            "volume": others.pop("volume", pd.DataFrame(0.0, index=close.index, columns=close.columns)),
        }
        for k, v in others.items():
            panels[k] = v
        return cls(panels)

    # ---------------------------------------------------------------- 属性
    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._dates

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def n_dates(self) -> int:
        return len(self._dates)

    @property
    def n_symbols(self) -> int:
        return len(self._symbols)

    @property
    def start(self) -> pd.Timestamp:
        return self._dates[0]

    @property
    def end(self) -> pd.Timestamp:
        return self._dates[-1]

    # ---------------------------------------------------------------- 访问
    def field(self, name: str) -> pd.DataFrame:
        """按名字取任意字段面板（先查行情，再查基本面）。"""
        key = str(name).lower()
        if key in self.panels:
            return self.panels[key]
        if key in self.fundamentals:
            return self.fundamentals[key]
        raise KeyError(f"字段 {name} 不存在；可用字段：{sorted(self.panels) + sorted(self.fundamentals)}")

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.field(name)

    def __contains__(self, name: str) -> bool:
        key = str(name).lower()
        return key in self.panels or key in self.fundamentals

    def open(self) -> pd.DataFrame:  # noqa: A003 - 语义命名
        return self.panels["open"]

    def high(self) -> pd.DataFrame:
        return self.panels["high"]

    def low(self) -> pd.DataFrame:
        return self.panels["low"]

    def close(self) -> pd.DataFrame:
        return self.panels["close"]

    def volume(self) -> pd.DataFrame:
        return self.panels["volume"]

    def vwap(self) -> pd.DataFrame:
        """成交量加权均价；没有 vwap 字段时用 amount/volume 估计，再退回 close。"""
        if "vwap" in self.panels:
            return self.panels["vwap"]
        if "amount" in self.panels:
            from ..utils.misc import safe_div

            return pd.DataFrame(
                safe_div(self.panels["amount"].values, self.panels["volume"].values, fill=np.nan),
                index=self._dates,
                columns=self._symbols,
            )
        return self.panels["close"]

    def amount(self) -> pd.DataFrame:
        if "amount" in self.panels:
            return self.panels["amount"]
        return self.panels["close"] * self.panels["volume"]

    def returns(self, field: str = "close", kind: str = "simple", periods: int = 1) -> pd.DataFrame:
        """收益率面板.

        kind : ``"simple"`` 简单收益 $p_t/p_{t-1}-1$；``"log"`` 对数收益 $\\ln(p_t/p_{t-1})$。
        """
        px = self.field(field)
        if kind == "log":
            r = np.log(px / px.shift(periods))
        elif kind == "simple":
            r = px / px.shift(periods) - 1.0
        else:
            raise ValueError("kind 只能是 'simple' 或 'log'")
        return r

    def fundamental(self, name: str) -> pd.DataFrame:
        if name not in self.fundamentals:
            raise KeyError(f"基本面字段 {name} 不存在；可用：{sorted(self.fundamentals)}")
        return self.fundamentals[name]

    def liquid(self, min_days: int = 60) -> pd.DataFrame:
        """流动性掩码：过去 min_days 天有成交且停牌（volume=0）比例不高。"""
        vol = self.volume()
        tradable = (vol > 0).rolling(min_days, min_periods=max(1, min_days // 2)).sum() >= min_days * 0.6
        return tradable.fillna(False)

    # ---------------------------------------------------------------- 变换
    def slice(self, start=None, end=None) -> "MarketData":
        """按日期切片，返回新的 MarketData。"""
        s = pd.Timestamp(self.start) if start is None else to_datetime(start)
        e = pd.Timestamp(self.end) if end is None else to_datetime(end)
        mask = (self._dates >= s) & (self._dates <= e)
        if not mask.any():
            raise ValueError(f"切片区间 [{s.date()}, {e.date()}] 内没有数据")
        panels = {k: v.loc[mask] for k, v in self.panels.items()}
        funds = {k: v.loc[mask] for k, v in self.fundamentals.items()}
        return type(self)(panels, funds, name=self.name, meta=self.meta)

    def subset(self, symbols: Sequence[str] | None = None, drop: Sequence[str] | None = None) -> "MarketData":
        """按标的筛选。"""
        cols = list(self._symbols)
        if symbols is not None:
            want = [str(s) for s in symbols]
            unknown = [s for s in want if s not in cols]
            if unknown:
                raise KeyError(f"未知标的：{unknown[:5]}")
            cols = [c for c in cols if c in set(want)]
        if drop:
            drop_set = {str(s) for s in drop}
            cols = [c for c in cols if c not in drop_set]
        panels = {k: v[cols] for k, v in self.panels.items()}
        funds = {k: v[cols] for k, v in self.fundamentals.items()}
        return type(self)(panels, funds, name=self.name, meta=self.meta)

    def drop_suspended(self, max_suspend_ratio: float = 0.5) -> "MarketData":
        """剔除停牌过多的标的（volume 为 0 的比例过高）。"""
        vol = self.volume()
        bad = (vol <= 0).mean() > max_suspend_ratio
        return self.subset(drop=list(vol.columns[bad]))

    def resample(self, rule: str = "W-FRI") -> "MarketData":
        """降频（日线 -> 周/月线）。"""
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        panels = {}
        for field, df in self.panels.items():
            how = agg.get(field, "last")
            panels[field] = df.resample(rule).agg(how)
        if "amount" in panels:
            panels["amount"] = self.panels["amount"].resample(rule).sum()
        funds = {k: v.resample(rule).last() for k, v in self.fundamentals.items()}
        return type(self)(panels, funds, name=self.name, meta=self.meta)

    def add_field(self, name: str, df: pd.DataFrame, fundamental: bool = False) -> "MarketData":
        """返回带有新增字段的副本（函数式，不改自身）。"""
        panels = dict(self.panels)
        funds = dict(self.fundamentals)
        target = funds if fundamental else panels
        target[str(name).lower()] = self._align(df)
        return type(self)(panels, funds, name=self.name, meta=self.meta)

    # ---------------------------------------------------------------- IO
    def to_parquet(self, path) -> Path:
        """把 panels 与 fundamentals 写入一个目录（每个字段一个 parquet）。"""
        root = Path(path).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        for field, df in self.panels.items():
            df.to_parquet(root / f"px_{field}.parquet")
        for field, df in self.fundamentals.items():
            df.to_parquet(root / f"fd_{field}.parquet")
        import json

        (root / "meta.json").write_text(
            json.dumps({"name": self.name, "meta": self.meta}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return root

    @classmethod
    def from_parquet(cls, path) -> "MarketData":
        root = Path(path).expanduser()
        panels, funds = {}, {}
        for f in sorted(root.glob("px_*.parquet")):
            panels[f.stem[3:]] = pd.read_parquet(f)
        for f in sorted(root.glob("fd_*.parquet")):
            funds[f.stem[3:]] = pd.read_parquet(f)
        meta_file = root / "meta.json"
        name, meta = "market", {}
        if meta_file.exists():
            import json

            blob = json.loads(meta_file.read_text(encoding="utf-8"))
            name = blob.get("name", "market")
            meta = blob.get("meta", {})
        return cls(panels, funds, name=name, meta=meta)

    # ---------------------------------------------------------------- 诊断
    def sanity_report(self) -> dict:
        """数据体检：缺失、异常跳变、零成交量占比、价格非正等。"""
        px = self.close()
        vol = self.volume()
        ret = px / px.shift(1) - 1.0
        jump = (ret.abs() > 0.5).sum().sum()  # 单日 >50% 视为可疑跳变（未复权/拆股）
        return {
            "name": self.name,
            "start": str(self.start.date()),
            "end": str(self.end.date()),
            "n_dates": int(self.n_dates),
            "n_symbols": int(self.n_symbols),
            "close_nan_ratio": float(px.isna().mean().mean()),
            "zero_volume_ratio": float((vol <= 0).mean().mean()),
            "nonpositive_price": int((px <= 0).sum().sum()),
            "suspicious_daily_jumps": int(jump),
            "fields": sorted(self.panels),
            "fundamentals": sorted(self.fundamentals),
        }

    def __repr__(self) -> str:
        return (
            f"MarketData(name={self.name!r}, {self.n_symbols} symbols, "
            f"{self.n_dates} days, {self.start.date()} -> {self.end.date()})"
        )
