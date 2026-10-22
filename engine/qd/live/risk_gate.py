"""风控闸门（RiskGate）：下单前的最后一道检查.

回测里没被执行到的规则，实盘一定会出事。这里把它们集中起来，每一条都对应
一次真实的亏钱经历：

* 单标的上限 —— 一次黑天鹅就让你出局；
* 单日成交额上限 —— 你的单子就是市场；
* 停牌 / 涨跌停 —— 下单了也成交不了；
* 可用现金 —— 券商不会让你透支；
* 连续亏损降杠杆 —— 模型可能已经失效；
* 回撤熔断 —— 保住本金才有下一局。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["OrderIntent", "GateResult", "RiskGate"]


@dataclass
class OrderIntent:
    """一个下单意图（还没成交）。"""
    symbol: str
    target_weight: float
    current_weight: float = 0.0
    price: float = np.nan
    adv: float = np.nan            # 日均成交额（用于容量检查）
    reason: str = ""

    @property
    def delta_weight(self) -> float:
        return self.target_weight - self.current_weight


@dataclass
class GateResult:
    """闸门结果."""
    allowed: bool
    adjusted_weight: float
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.allowed


class RiskGate:
    """下单前风控闸门.

    Parameters
    ----------
    max_position_weight :
        单标的最大权重。
    max_daily_adv_share :
        单日下单量不超过日均成交额的这个比例（容量约束）。
    max_gross, max_net :
        总敞口与净敞口上限。
    min_cash_ratio :
        最低现金比例。
    max_drawdown_halt :
        组合回撤超过此值则**全部拒绝**并建议清仓。
    max_consecutive_loss_days :
        连续亏损天数达到此值时降低杠杆。
    """

    def __init__(
        self,
        max_position_weight: float = 0.1,
        max_daily_adv_share: float = 0.05,
        max_gross: float = 1.0,
        max_net: float = 1.0,
        min_cash_ratio: float = 0.0,
        max_drawdown_halt: float = 0.2,
        max_consecutive_loss_days: int = 5,
        de_leverage_factor: float = 0.5,
        blocked_symbols: Sequence[str] = (),
    ):
        self.max_position_weight = float(max_position_weight)
        self.max_daily_adv_share = float(max_daily_adv_share)
        self.max_gross = float(max_gross)
        self.max_net = float(max_net)
        self.min_cash_ratio = float(min_cash_ratio)
        self.max_drawdown_halt = float(max_drawdown_halt)
        self.max_consecutive_loss_days = int(max_consecutive_loss_days)
        self.de_leverage_factor = float(de_leverage_factor)
        self.blocked = {str(s) for s in blocked_symbols}

    # ------------------------------------------------------------ 单笔检查
    def check(self, intent: OrderIntent, *, portfolio_value: float = 0.0,
              current_drawdown: float = 0.0, consecutive_loss_days: int = 0,
              suspended: bool = False, limit_up: bool = False,
              limit_down: bool = False, gross_now: float = 0.0) -> GateResult:
        """检查一个下单意图。"""
        reasons: list[str] = []
        warnings: list[str] = []
        w = float(intent.target_weight)

        # --- 硬拒绝 ---
        if intent.symbol in self.blocked:
            return GateResult(False, intent.current_weight, [f"{intent.symbol} 在黑名单中"])
        if current_drawdown <= -abs(self.max_drawdown_halt):
            return GateResult(
                False, 0.0,
                [f"组合回撤 {current_drawdown:.2%} 超过熔断线 {-abs(self.max_drawdown_halt):.2%}，停止一切开仓"],
            )
        if suspended:
            return GateResult(False, intent.current_weight, [f"{intent.symbol} 停牌，无法成交"])
        if limit_up and w > intent.current_weight:
            return GateResult(False, intent.current_weight, [f"{intent.symbol} 涨停，买入无法成交"])
        if limit_down and w < intent.current_weight:
            warnings.append(f"{intent.symbol} 跌停，卖出可能排队不成交")

        # --- 软约束：裁剪权重 ---
        if abs(w) > self.max_position_weight:
            w = np.sign(w) * self.max_position_weight
            warnings.append(f"{intent.symbol} 权重裁剪到 {w:.2%}（单标的上限）")

        # 容量约束：下单金额 / ADV
        if np.isfinite(intent.adv) and intent.adv > 0 and portfolio_value > 0:
            order_value = abs(intent.delta_weight) * portfolio_value
            share = order_value / intent.adv
            if share > self.max_daily_adv_share:
                max_delta = self.max_daily_adv_share * intent.adv / portfolio_value
                w = intent.current_weight + np.sign(intent.delta_weight) * max_delta
                warnings.append(
                    f"{intent.symbol} 下单量占日均成交额 {share:.1%} > {self.max_daily_adv_share:.1%}，"
                    f"裁剪到 {w:.2%}（容量约束）"
                )

        # 连续亏损降杠杆
        if consecutive_loss_days >= self.max_consecutive_loss_days and abs(w) > 0:
            w *= self.de_leverage_factor
            warnings.append(
                f"连续 {consecutive_loss_days} 天亏损，杠杆降到 {self.de_leverage_factor:.0%}（模型可能已失效）"
            )

        return GateResult(True, float(w), reasons, warnings)

    # ------------------------------------------------------------ 组合级检查
    def check_portfolio(self, weights: pd.Series, *, gross_limit: float | None = None) -> GateResult:
        """组合级检查并归一化。"""
        warnings: list[str] = []
        w = pd.Series(weights).fillna(0.0).astype(float)
        w = w.clip(-self.max_position_weight, self.max_position_weight)
        if (w.abs() > self.max_position_weight * 0.999).any():
            warnings.append("部分标的触及单标的上限，已被裁剪")

        gross_limit = gross_limit if gross_limit is not None else self.max_gross
        gross = w.abs().sum()
        if gross > gross_limit + 1e-9 and gross > 0:
            w = w * gross_limit / gross
            warnings.append(f"总敞口从 {gross:.2%} 压缩到 {gross_limit:.2%}")

        net = w.sum()
        if abs(net) > self.max_net + 1e-9:
            warnings.append(f"净敞口 {net:.2%} 超过 {self.max_net:.2%}")

        cash_ratio = 1.0 - w.sum()
        if cash_ratio < self.min_cash_ratio:
            warnings.append(f"现金比例 {cash_ratio:.2%} 低于下限 {self.min_cash_ratio:.2%}")

        return GateResult(True, float(gross), warnings=warnings) if False else GateResult(
            True, float(w.abs().sum()), warnings=warnings
        )

    def to_dict(self) -> dict:
        return {
            "max_position_weight": self.max_position_weight,
            "max_daily_adv_share": self.max_daily_adv_share,
            "max_gross": self.max_gross,
            "max_net": self.max_net,
            "min_cash_ratio": self.min_cash_ratio,
            "max_drawdown_halt": self.max_drawdown_halt,
            "max_consecutive_loss_days": self.max_consecutive_loss_days,
        }
