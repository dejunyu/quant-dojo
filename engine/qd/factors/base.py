"""因子基类与注册表."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping

import pandas as pd

__all__ = [
    "Factor", "FactorSpec", "FACTOR_REGISTRY", "FACTOR_CATEGORIES",
    "get_factor", "list_factors", "register", "factor_from_func",
]

FACTOR_CATEGORIES = [
    "动量", "反转", "价值", "成长", "质量", "波动", "流动性", "情绪", "技术", "另类", "自定义",
]


@dataclass
class FactorSpec:
    """因子元信息."""
    name: str
    category: str = "未分类"
    direction: int = 1            # +1 正向 / -1 反向
    params: dict = field(default_factory=dict)
    description: str = ""
    formula: str = ""             # LaTeX 片段，用于课程与 UI 展示
    required_fields: tuple = ("close",)
    reference: str = ""


class Factor:
    """因子对象.

    Parameters
    ----------
    spec :
        元信息。
    fn :
        计算函数 ``fn(md, **params) -> DataFrame(dates × symbols)``。
    """

    def __init__(self, spec: FactorSpec, fn: Callable[..., pd.DataFrame]):
        if spec.category == "未分类":
            spec.category = "自定义"
        if spec.category not in FACTOR_CATEGORIES:
            raise ValueError(f"未知因子分类 {spec.category!r}；可选：{FACTOR_CATEGORIES}")
        if spec.direction not in (1, -1):
            raise ValueError("direction 只能是 +1 或 -1")
        self.spec = spec
        self._fn = fn

    # ---- 属性代理 ----
    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def category(self) -> str:
        return self.spec.category

    @property
    def direction(self) -> int:
        return self.spec.direction

    @property
    def description(self) -> str:
        return self.spec.description

    @property
    def formula(self) -> str:
        return self.spec.formula

    def compute(self, md) -> pd.DataFrame:
        """计算因子面板。返回 ``dates × symbols``。"""
        if md is None or not hasattr(md, "panels"):
            raise TypeError(
                f"因子 {self.name} 需要 MarketData 对象，收到 {type(md).__name__}。\n"
                f"正确用法：f.compute(md)  其中 md = generate_demo_market() 或 AkshareFeed().load(...)"
            )
        # 基本面字段缺失时给出清晰报错，而不是静默出错
        missing = [f for f in self.spec.required_fields if f not in md]
        if missing:
            raise KeyError(
                f"因子 {self.name} 需要字段 {missing}，当前数据集只有 {sorted(md.panels)}"
                f"（基本面：{sorted(md.fundamentals)}）"
            )
            raise KeyError(
                f"因子 {self.name} 需要字段 {missing}，当前数据集只有 {sorted(md.panels)}"
                f"（基本面：{sorted(md.fundamentals)}）"
            )
        out = self._fn(md, **self.spec.params)
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"因子 {self.name} 返回值必须是 DataFrame")
        if not out.index.equals(md.dates) or not out.columns.equals(pd.Index(md.symbols)):
            out = out.reindex(index=md.dates, columns=md.symbols)
        return out

    def signed(self, md) -> pd.DataFrame:
        """按 direction 调整符号后的因子值（越大越好）。"""
        return self.direction * self.compute(md)

    def __call__(self, md) -> pd.DataFrame:
        return self.compute(md)

    def __repr__(self) -> str:
        return f"<Factor {self.name} [{self.category}] dir={self.direction:+d}>"


FACTOR_REGISTRY: dict[str, Factor] = {}


def register(spec: FactorSpec, fn: Callable | None = None):
    """注册因子（可作装饰器使用）。"""
    def deco(f: Callable) -> Factor:
        factor = Factor(spec, f)
        FACTOR_REGISTRY[spec.name] = factor
        return factor

    if fn is None:
        return deco
    return deco(fn)


def factor_from_func(
    name: str,
    fn: Callable | None = None,
    category: str = "未分类",
    direction: int = 1,
    description: str = "",
    formula: str = "",
    required_fields: tuple = ("close",),
    **params,
):
    """用一行代码把函数变成注册因子（可作装饰器）。"""
    spec = FactorSpec(
        name=name, category=category, direction=direction, params=params,
        description=description, formula=formula, required_fields=required_fields,
    )
    if fn is None:
        return register(spec)
    return register(spec)(fn)


# 常见别名：课程与从业者的方言与平台命名不一致时，尽量帮用户找到想要的因子
FACTOR_ALIASES: dict[str, str] = {
    "bp_w1": "bp", "book_to_price": "bp", "book_to_market": "bp", "bm": "bp",
    "ep_ttm": "ep", "earnings_yield": "ep", "ep_w1": "ep",
    "sp_ttm": "sp", "sales_to_price": "sp",
    "roe_q": "roe", "roe_ttm": "roe", "roe_annual": "roe",
    "mom_20": "mom_1m", "mom_60": "mom_3m", "mom_120": "mom_6m",
    "momentum_12_1": "mom_12_1", "momentum": "mom_12_1", "mom": "mom_12_1",
    "mom_6_1": "mom_6m", "mom_12_2": "mom_12_1", "mom_12m": "mom_12_1",
    "reversal": "rev_5d", "rev": "rev_5d", "ret_5d": "rev_5d",
    "rev_20": "rev_20d", "rev_1m": "rev_20d", "rev_6": "rev_5d",
    "volatility": "vol_60d", "vol": "vol_60d", "realized_vol": "vol_60d",
    "iv": "idiosyncratic_vol", "idio_vol": "idiosyncratic_vol",
    "size": "ln_mktcap", "market_cap": "ln_mktcap", "mktcap": "ln_mktcap",
    "turnover": "turnover_20d", "liquidity": "amihud_illiquidity",
    "pe_ratio": "ep", "pb_ratio": "bp",
}

# 后缀模式：如 roe_q / bp_w1 / mom_1m_ttm —— 去掉可疑后缀再匹配
_SUFFIX_PATTERNS = [
    re.compile(r"_(w\d+|ttm|q\d?|y\d|annual|quarterly|monthly|daily|adj|z|rank)$", re.I),
]


def get_factor(name: str) -> Factor:
    """按名字取因子，支持别名与常见写法容错。

    解析顺序：精确名 → 别名表 → 去掉可疑后缀后再精确/别名 → 唯一模糊匹配。
    仍然找不到时，报错信息里会列出最接近的几个候选，而不是让用户去猜。
    """
    key = str(name).strip()
    if key in FACTOR_REGISTRY:
        return FACTOR_REGISTRY[key]

    if key.lower() in FACTOR_ALIASES:
        target = FACTOR_ALIASES[key.lower()]
        if target in FACTOR_REGISTRY:
            return FACTOR_REGISTRY[target]

    # 去掉 _w1 / _q / _ttm 之类的后缀再试
    stripped = key
    changed = True
    while changed:
        changed = False
        for pat in _SUFFIX_PATTERNS:
            new = pat.sub("", stripped)
            if new != stripped:
                stripped, changed = new, True
                break
    if stripped != key:
        if stripped in FACTOR_REGISTRY:
            return FACTOR_REGISTRY[stripped]
        if stripped.lower() in FACTOR_ALIASES:
            target = FACTOR_ALIASES[stripped.lower()]
            if target in FACTOR_REGISTRY:
                return FACTOR_REGISTRY[target]

    # 唯一模糊匹配
    cands = [k for k in FACTOR_REGISTRY if key.lower() in k.lower()]
    if len(cands) == 1:
        return FACTOR_REGISTRY[cands[0]]
    if len(cands) > 1:
        raise KeyError(f"因子名 {name!r} 有歧义，候选：{sorted(cands)}")

    # 给出"最接近"的建议，方便用户改正
    import difflib

    close = difflib.get_close_matches(key, list(FACTOR_REGISTRY), n=5, cutoff=0.4)
    hint = f"你是不是想用：{close}" if close else ""
    raise KeyError(
        f"未知因子 {name!r}。{hint}\n"
        f"可用因子（{len(FACTOR_REGISTRY)} 个）：{sorted(FACTOR_REGISTRY)}"
    )


class FactorInfo(dict):
    """因子元信息（dict 的子类）。

    额外做了两件事，让课程/脚本写起来更顺：

    * 可以像字符串一样用：``get_factor(info)``、``info == "mom_12_1"``；
    * 可以作字典键（按因子名哈希）。
    """

    @property
    def name(self) -> str:
        return str(self.get("name", ""))

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.name == other
        return dict.__eq__(self, other)

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"FactorInfo({self.name!r})"


def list_factors(category: str | None = None) -> list[FactorInfo]:
    """列出因子元信息（给 UI / 文档用）。

    返回的每一项都是 :class:`FactorInfo`——既像字典能取 ``["category"]``，
    又像字符串能被 ``get_factor`` 直接接受。
    """
    out: list[FactorInfo] = []
    for name, f in sorted(FACTOR_REGISTRY.items()):
        if category and f.category != category:
            continue
        out.append(FactorInfo({
            "name": name,
            "category": f.category,
            "direction": f.direction,
            "description": f.description,
            "formula": f.formula,
            "params": f.spec.params,
            "required_fields": list(f.spec.required_fields),
        }))
    return out
