"""回测配置与成本模型."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

__all__ = ["BacktestConfig", "CostModel"]


@dataclass
class CostModel:
    """交易成本模型.

    一笔交易的总成本 = 佣金 + 印花税 + 过户费 + 滑点 + 冲击成本。

    默认值按 A 股典型水平设定（**仅为教学默认，实盘请按自己的券商费率调整**）：
    佣金双边万分之三、印花税卖出千分之一（2023-08 起为万分之五，这里保守取
    千分之一以便演示成本敏感性）、过户费万分之零点一、滑点 5bp、冲击成本
    按成交量参与率的平方根模型。
    """

    commission_rate: float = 0.0003        # 双边佣金率
    min_commission: float = 5.0            # 单笔最低佣金（元）
    stamp_tax: float = 0.001               # 印花税（仅卖出）
    transfer_fee: float = 0.00001          # 过户费（双边，沪市）
    slippage_bps: float = 5.0              # 基础滑点（基点）
    impact_coeff: float = 0.1              # 冲击成本系数（平方根模型）
    max_participation: float = 0.1         # 单日成交量参与率上限
    borrow_rate: float = 0.08              # 年化融券利率（做空）
    short_allowed: bool = False

    def commission(self, notional: float) -> float:
        return max(abs(notional) * self.commission_rate, self.min_commission if abs(notional) > 0 else 0.0)

    def tax(self, notional: float, side: str) -> float:
        if side == "sell":
            return abs(notional) * self.stamp_tax
        return 0.0

    def transfer(self, notional: float) -> float:
        return abs(notional) * self.transfer_fee

    def slippage(self, price: float, side: str) -> float:
        """滑点造成的价格偏移（买高卖低）。"""
        sign = 1.0 if side == "buy" else -1.0
        return price * sign * self.slippage_bps / 10_000.0

    def impact(self, price: float, qty: float, adv: float) -> float:
        """平方根冲击成本：$\\Delta p = \\sigma\\cdot c\\sqrt{q/ADV}$。

        这里用 ``impact_coeff`` 直接乘以参与率的平方根，得到一个保守的价格偏移。
        """
        if adv is None or adv <= 0 or qty <= 0:
            return 0.0
        part = abs(qty) / adv
        sign = 1.0 if qty > 0 else -1.0
        return price * sign * self.impact_coeff * np_sqrt(part)

    def to_dict(self) -> dict:
        return asdict(self)


def np_sqrt(x: float) -> float:
    import math

    return math.sqrt(max(float(x), 0.0))


@dataclass
class BacktestConfig:
    """回测配置.

    成本参数可以两种方式给出：

    * 传 ``cost=CostModel(...)`` —— 显式、可控（推荐用于细调）；
    * 直接传 ``commission_rate=0.0003, slippage_bps=5, ...`` —— 便捷写法，
      会自动写入 ``cost``。两种方式同时出现时，显式传入的成本字段覆盖
      ``CostModel`` 里的对应项。
    """

    initial_cash: float = 1_000_000.0
    cost: CostModel = field(default_factory=CostModel)

    # ---- 交易规则 ----
    t_plus_1: bool = True              # A 股 T+1：当日买入不可当日卖出
    allow_short: bool = False
    limit_up_buy: bool = False         # 涨停能否买入（默认不能，更符合现实）
    limit_down_sell: bool = True       # 跌停能否卖出（默认可以排队，但按滑点惩罚）
    price_limit: float | None = 0.1    # 涨跌停幅度；None 表示无涨跌幅限制（如美股）
    lot_size: int = 100                # 最小交易单位（A 股 100 股）
    cash_reserve: float = 0.0          # 保留现金比例

    # ---- 撮合 ----
    execution: Literal["next_open", "same_close", "next_close", "vwap"] = "next_open"
    rebalance: str = "daily"           # daily / weekly / monthly / quarterly
    rebalance_offset: int = 0          # 调仓日的偏移（避免所有策略同日调仓）
    max_weight: float = 1.0            # 单标的最大权重
    min_weight: float = 0.0
    gross_leverage: float = 1.0        # 总敞口上限
    max_participation: float = 0.1
    min_trade_value: float = 0.005     # 最小交易阈值（占净资产比例）：低于此偏离不交易
    rebalance_tol: float = 0.0         # 权重容差带（buffer 再平衡）

    # ---- 风控 ----
    stop_loss: float | None = None     # 单标的止损（相对成本价跌幅）
    take_profit: float | None = None
    max_drawdown_stop: float | None = None   # 组合回撤熔断
    trailing_stop: float | None = None

    # ---- 其它 ----
    benchmark: str | None = None       # 基准标的（用于相对收益）
    risk_free: float = 0.02            # 年化无风险利率
    warmup: int = 0                    # 前 N 天不下单
    verbose: bool = False

    # ---- 成本参数的便捷入口（见类文档）----
    commission_rate: float | None = None
    stamp_tax: float | None = None
    commission: float | None = None    # commission_rate 的别名
    slippage_bps: float | None = None
    slippage: float | None = None      # slippage_bps 的别名
    slippage_bp: float | None = None   # slippage_bps 的别名（单数写法）
    impact_coeff: float | None = None
    transfer_fee: float | None = None
    min_commission: float | None = None
    stamp_duty: float | None = None    # stamp_tax 的别名
    cost_bp: float | None = None       # 综合单边成本（bp）：佣金+印花税+滑点的合计
    borrow_rate: float | None = None   # 融券年化利率

    # ---- 交易规则的便捷别名 ----
    t_plus_one: bool | None = None     # t_plus_1 的别名
    exec_price: str | None = None      # execution 的别名（'' -> next_open）
    margin_ratio: float | None = None  # 保证金比例（做空时用于折算可用资金）

    def __post_init__(self) -> None:
        """把顶层成本参数合并进 ``cost``，让``BacktestConfig(...)`` 一行就能配齐。"""
        if self.t_plus_one is not None:
            self.t_plus_1 = bool(self.t_plus_one)
        if self.exec_price is not None:
            # exec_price 可以是 "open" / "close" / "next_open" / "vwap"
            m = {
                "open": "next_open", "next_open": "next_open",
                "close": "next_close", "next_close": "next_close",
                "same_close": "same_close", "today_close": "same_close",
                "vwap": "vwap",
            }
            self.execution = m.get(str(self.exec_price).lower(), self.execution)

        if not isinstance(self.cost, CostModel):
            # 允许 cost={"commission_rate": ...} 这样的字典写法
            self.cost = CostModel(**dict(self.cost)) if self.cost else CostModel()

        # 深拷贝，避免多个配置共享同一个 CostModel 对象
        base = CostModel(**{k: getattr(self.cost, k) for k in CostModel.__dataclass_fields__})

        rate = self.commission_rate if self.commission_rate is not None else self.commission
        tax = self.stamp_tax if self.stamp_tax is not None else self.stamp_duty
        slip = self.slippage_bps
        if slip is None:
            slip = self.slippage if self.slippage is not None else self.slippage_bp
        if self.cost_bp is not None:
            # 综合成本（bp）：按"佣金 = 印花税 = 滑点"的简单假设拆分，
            # 适合快速做成本敏感性测试；精确建模请直接用 CostModel。
            bp = float(self.cost_bp) / 1e4
            rate = bp / 3.0
            tax = bp / 3.0
            slip = float(self.cost_bp) / 3.0
        overrides = {
            "commission_rate": rate,
            "stamp_tax": tax,
            "slippage_bps": slip,
            "impact_coeff": self.impact_coeff,
            "transfer_fee": self.transfer_fee,
            "min_commission": self.min_commission,
            "borrow_rate": self.borrow_rate,
        }
        for k, v in overrides.items():
            if v is not None:
                setattr(base, k, float(v))
        if self.allow_short:
            base.short_allowed = True
        self.cost = base

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
