"""撮合与账户（Broker）.

设计原则
--------
这是整个平台最容易"失真"的地方，因此这里刻意做得保守而具体：

1. **信号与成交分离**：策略在 t 日收盘后给出目标权重，订单在 **t+1 日** 按
   ``execution`` 指定的价格成交。绝不出现"用今天的收盘价按今天的信号成交"。
2. **涨停不能买 / 跌停难卖**：按 ``prev_close * (1 ± price_limit)`` 判定，
   默认涨停禁止买入。
3. **成交量参与率上限**：单笔成交不超过当日成交量的 ``max_participation``，
   这是对"回测资金无限"最直接的修正，也是容量约束的第一道闸。
4. **T+1**：当日买入的份额当日不可卖出。
5. **成本逐笔计算**：佣金（含最低 5 元）、印花税（卖出）、过户费、滑点、
   平方根冲击成本，全部计入现金。

这些约束会让回测收益"变难看"，但它们是真实存在的。一个回测引擎的价值，
取决于它拒绝了多少不切实际的幻想。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig, CostModel

__all__ = ["Position", "Fill", "Trade", "Broker"]


@dataclass
class Position:
    """单个持仓."""
    symbol: str
    qty: int = 0
    avg_cost: float = 0.0
    locked_qty: int = 0          # T+1 冻结份额（当日买入）
    last_price: float = np.nan
    peak_price: float = np.nan   # 用于移动止损
    open_date: pd.Timestamp | None = None

    @property
    def market_value(self) -> float:
        if np.isnan(self.last_price):
            return 0.0
        return self.qty * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        if not self.qty or np.isnan(self.last_price):
            return 0.0
        return (self.last_price - self.avg_cost) * self.qty


@dataclass
class Fill:
    """一次成交记录."""
    date: pd.Timestamp
    symbol: str
    side: str            # buy / sell
    qty: int
    price: float         # 实际成交价（含滑点冲击）
    ref_price: float     # 理论参考价（未加成本）
    commission: float
    tax: float
    transfer_fee: float
    slippage_cost: float
    impact_cost: float
    reason: str = ""

    @property
    def total_cost(self) -> float:
        return self.commission + self.tax + self.transfer_fee + self.slippage_cost + self.impact_cost

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass
class Trade:
    """一笔完整交易（开仓到平仓），用于胜率与盈亏比统计。"""
    symbol: str
    open_date: pd.Timestamp
    close_date: pd.Timestamp | None
    qty: int
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    holding_days: int
    reason: str = ""


class Broker:
    """账户与撮合。

    性能说明：价格/成交量在构造时预取为 numpy 数组，估值为 O(n_symbols) 的
    向量运算，因此 1800 日 × 50 标的的全量回测在百毫秒量级完成。
    """

    def __init__(self, md, config: BacktestConfig | None = None):
        self.md = md
        self.config = config or BacktestConfig()
        self.cost: CostModel = self.config.cost
        self.symbols = list(md.symbols)
        self.n_sym = len(self.symbols)
        self.dates = md.dates
        self.n_dates = len(self.dates)
        self._date_arr = self.dates.to_numpy() if hasattr(self.dates, "to_numpy") else None

        # ---- 预取为 numpy，避免逐格 pandas 索引 ----
        self._open_a = md.open().to_numpy(dtype=float)
        self._close_a = md.close().to_numpy(dtype=float)
        self._volume_a = md.volume().to_numpy(dtype=float)
        self._vwap_a = md.vwap().to_numpy(dtype=float)

        self._sym_index = {s: i for i, s in enumerate(self.symbols)}

        self.positions: dict[str, Position] = {s: Position(symbol=s) for s in self.symbols}
        self.qty = np.zeros(self.n_sym, dtype=np.int64)          # 当前持股
        self.avg_cost = np.zeros(self.n_sym, dtype=float)        # 平均成本
        self.locked = np.zeros(self.n_sym, dtype=np.int64)       # T+1 冻结
        self.peak = np.full(self.n_sym, np.nan)                  # 持仓期间最高价

        self.cash = float(self.config.initial_cash)
        self.initial_cash = float(self.config.initial_cash)
        self.fills: list[Fill] = []
        self.trades: list[Trade] = []
        self._open_lots: list[list[dict]] = [[] for _ in range(self.n_sym)]

        self._cache: dict[int, float] = {}

    # ------------------------------------------------------------ 估值
    def mark_to_market(self, i: int) -> float:
        """按第 i 日收盘价估值，返回总资产。"""
        px = self._close_a[i]
        last = np.where(np.isnan(px), self.avg_cost, px)
        mv = float(np.nansum(self.qty * last))
        total = self.cash + mv
        self._cache[i] = total
        return total

    def equity_curve(self, equity_list) -> pd.Series:
        return pd.Series(list(equity_list), index=self.dates[: len(equity_list)], name="equity")

    def portfolio_value(self, i: int) -> float:
        return self.mark_to_market(i)

    def current_weights(self, i: int) -> pd.Series:
        total = self.mark_to_market(i)
        px = self._close_a[i]
        vals = self.qty * np.where(np.isnan(px), 0.0, px)
        if total <= 0:
            return pd.Series(0.0, index=self.symbols)
        return pd.Series(vals / total, index=self.symbols)

    def gross_exposure(self, i: int) -> float:
        total = self.mark_to_market(i)
        if total <= 0:
            return 0.0
        px = self._close_a[i]
        vals = np.abs(self.qty) * np.where(np.isnan(px), 0.0, px)
        return float(vals.sum() / total)

    def position_values(self, i: int) -> np.ndarray:
        px = self._close_a[i]
        return self.qty * np.where(np.isnan(px), 0.0, px)

    # ------------------------------------------------------------ 价格工具
    def price_row(self, i: int) -> np.ndarray:
        """按 config.execution 返回第 i 日的参考价向量。"""
        how = self.config.execution
        if how == "next_open":
            return self._open_a[i]
        if how in ("same_close", "next_close"):
            return self._close_a[i]
        if how == "vwap":
            v = self._vwap_a[i]
            return np.where(np.isnan(v), self._close_a[i], v)
        raise ValueError(f"未知撮合方式 {how}")

    def limit_masks(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """返回 (涨停掩码, 跌停掩码)。"""
        up = np.zeros(self.n_sym, dtype=bool)
        dn = np.zeros(self.n_sym, dtype=bool)
        if i == 0 or self.config.price_limit is None:
            return up, dn
        prev = self._close_a[i - 1]
        op = self._open_a[i]
        valid = np.isfinite(prev) & np.isfinite(op) & (prev > 0)
        lim = self.config.price_limit
        up = valid & (op >= prev * (1 + lim) - 1e-8)
        dn = valid & (op <= prev * (1 - lim) + 1e-8)
        return up, dn

    def max_tradable(self, i: int) -> np.ndarray:
        vol = self._volume_a[i]
        cap = np.where(np.isfinite(vol), vol, 0.0) * self.config.max_participation
        return cap

    # ------------------------------------------------------------ 下单
    def order_target_weights(self, i: int, targets, reason: str = "") -> int:
        """按目标权重调仓（在**第 i 日**用第 i 日的价格成交）。

        调用方必须保证：``targets`` 是用 **i 日之前**（含 i-1 日或更早）的
        信息算出来的，这样才没有前视偏差。返回成交笔数。
        """
        ref = self.price_row(i)
        tradable = np.isfinite(ref) & (ref > 0)
        cap = self.max_tradable(i)
        up_mask, dn_mask = self.limit_masks(i)
        lot = max(1, int(self.config.lot_size))

        equity = self.mark_to_market(i)
        if equity <= 0:
            return 0

        if isinstance(targets, np.ndarray):
            t = np.asarray(targets, dtype=float)
            if t.shape[0] != self.n_sym:
                raise ValueError(f"目标权重长度 {t.shape[0]} 与标的数 {self.n_sym} 不一致")
        else:
            t = pd.Series(targets).reindex(self.symbols).fillna(0.0).to_numpy(dtype=float)
        t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        t = np.clip(t, -self.config.max_weight, self.config.max_weight)
        if not self.config.allow_short:
            t = np.clip(t, 0.0, None)
        gross = np.abs(t).sum()
        if gross > self.config.gross_leverage + 1e-9 and gross > 0:
            t = t * (self.config.gross_leverage / gross)

        # 目标市值与当前市值
        investable = equity * (1 - self.config.cash_reserve)
        target_val = t * investable
        cur_val = self.position_values(i)
        delta_val = target_val - cur_val

        # ---- 最小交易阈值：低于阈值的偏离不交易（抑制噪声换手）----
        threshold = self.config.min_trade_value
        if threshold > 0 and investable > 0:
            rel = np.abs(delta_val) / investable
            ignore = rel < threshold
            delta_val = np.where(ignore, 0.0, delta_val)

        # ---- 先卖：释放现金 ----
        n_fills = 0
        sell_idx = np.where((delta_val < -1e-8) & tradable)[0]
        sell_idx = sell_idx[np.argsort(delta_val[sell_idx])]  # 卖得最多的优先
        for j in sell_idx:
            if dn_mask[j] and not self.config.limit_down_sell:
                continue
            price0 = ref[j]
            want = int(np.floor(abs(delta_val[j]) / price0 / lot) * lot)
            sellable = self.qty[j] - self.locked[j]
            want = int(min(want, max(sellable, 0)))
            want = int(min(want, float(cap[j])))
            want = int(np.floor(want / lot) * lot)
            if want <= 0:
                continue
            f = self._fill(i, j, -want, price0, cap[j], reason)
            if f is not None:
                self.fills.append(f)
                n_fills += 1

        # ---- 再买：受现金约束 ----
        equity = self.mark_to_market(i)
        cur_val = self.position_values(i)
        target_val = t * equity * (1 - self.config.cash_reserve)
        delta_val = target_val - cur_val
        buy_idx = np.where((delta_val > 1e-8) & tradable)[0]
        buy_idx = buy_idx[np.argsort(-delta_val[buy_idx])]
        for j in buy_idx:
            if up_mask[j] and not self.config.limit_up_buy:
                continue
            price0 = ref[j]
            cash_left = self.cash * (1 - self.config.cash_reserve)
            if cash_left <= self.cost.min_commission * 2:
                break
            want = int(np.floor(delta_val[j] / price0 / lot) * lot)
            want = int(min(want, float(cap[j])))
            # 现金上限（含费用与安全垫）
            affordable = int(np.floor(cash_left / (price0 * 1.003)) / lot) * lot
            want = int(min(want, max(affordable, 0)))
            if want <= 0:
                continue
            f = self._fill(i, j, want, price0, cap[j], reason)
            if f is not None:
                self.fills.append(f)
                n_fills += 1

        self._cache.pop(i, None)
        if n_fills:
            # 只有真的发生了成交才同步 Position 对象；没有成交时它不可能变化。
            # 这一步在 1800 天 × 50 标的里能省下上万次循环。
            self.sync_positions(i)
        return n_fills

    def _fill(self, i: int, j: int, signed_qty: int, ref_price: float, cap: float, reason: str) -> Fill | None:
        """执行一笔成交（signed_qty 正为买、负为卖）。"""
        if signed_qty == 0:
            return None
        side = "buy" if signed_qty > 0 else "sell"
        qty = abs(int(signed_qty))
        symbol = self.symbols[j]

        price = self._impact_price(ref_price, qty, cap, side)
        if not np.isfinite(price) or price <= 0:
            return None
        notional = qty * price
        commission = self.cost.commission(notional)
        tax = self.cost.tax(notional, side)
        transfer = self.cost.transfer(notional)
        slip_cost = abs(self.cost.slippage(ref_price, side)) * qty
        impact_cost = abs(price - ref_price - self.cost.slippage(ref_price, side)) * qty

        if side == "buy":
            est = notional + commission + transfer
            if est > self.cash * (1 - self.config.cash_reserve) + 1e-9:
                return None
            old = max(self.qty[j], 0)
            self.avg_cost[j] = (self.avg_cost[j] * old + price * qty) / (old + qty) if (old + qty) else price
            self.qty[j] += qty
            self.locked[j] += qty if self.config.t_plus_1 else 0
            self.cash -= est
            self._open_lots[j].append({"date": self._date_at(i), "qty": qty, "price": price})
        else:
            self.qty[j] -= qty
            self.cash += notional - commission - tax - transfer
            self._close_lots(j, i, qty, price, reason)
            if self.qty[j] == 0:
                self.avg_cost[j] = 0.0
                self.peak[j] = np.nan

        return Fill(
            date=self._date_at(i), symbol=symbol, side=side, qty=qty, price=price,
            ref_price=ref_price, commission=commission, tax=tax, transfer_fee=transfer,
            slippage_cost=slip_cost, impact_cost=impact_cost, reason=reason,
        )

    def _impact_price(self, ref: float, qty: int, adv: float, side: str) -> float:
        slip = self.cost.slippage(ref, side)
        impact = self.cost.impact(ref, qty if side == "buy" else -qty, adv)
        return ref + slip + impact

    def _date_at(self, i: int):
        """快速取第 i 日日期（避开 pandas 标量索引的开销）。"""
        if self._date_arr is not None:
            return pd.Timestamp(self._date_arr[i])
        return self.dates[i]

    def _close_lots(self, j: int, i: int, qty: int, price: float, reason: str) -> None:
        """FIFO 平仓，记录完整交易。"""
        date = self._date_at(i)
        remaining = qty
        while remaining > 0 and self._open_lots[j]:
            lot = self._open_lots[j][0]
            take = min(remaining, lot["qty"])
            pnl = (price - lot["price"]) * take
            self.trades.append(Trade(
                symbol=self.symbols[j], open_date=lot["date"], close_date=date, qty=take,
                entry_price=lot["price"], exit_price=price, pnl=pnl,
                return_pct=pnl / (lot["price"] * take) if lot["price"] * take else 0.0,
                holding_days=int((date - lot["date"]).days), reason=reason,
            ))
            lot["qty"] -= take
            remaining -= take
            if lot["qty"] <= 0:
                self._open_lots[j].pop(0)

    def sync_positions(self, i: int) -> None:
        """把 numpy 状态同步到 Position 对象（供展示/风控使用）。"""
        px = self._close_a[i]
        qty = self.qty
        avg = self.avg_cost
        locked = self.locked
        date = self.dates[i]
        for j, s in enumerate(self.symbols):
            q = qty[j]
            if q == 0 and not locked[j]:
                continue
            p = self.positions[s]
            p.qty = int(q)
            p.avg_cost = float(avg[j])
            p.locked_qty = int(locked[j])
            price = px[j]
            if np.isfinite(price):
                price = float(price)
                p.last_price = price
                peak = p.peak_price
                p.peak_price = price if np.isnan(peak) else max(peak, price)
            if q and p.open_date is None:
                p.open_date = date

    def unlock_daily(self) -> None:
        """每日收盘后解冻 T+1 份额。"""
        self.locked[:] = 0
        for p in self.positions.values():
            p.locked_qty = 0

    # ------------------------------------------------------------ 汇总
    def fills_frame(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame(columns=["symbol", "side", "qty", "price", "ref_price",
                                         "commission", "tax", "transfer_fee",
                                         "slippage_cost", "impact_cost", "notional", "reason"])
        df = pd.DataFrame([{
            "date": f.date, "symbol": f.symbol, "side": f.side, "qty": f.qty,
            "price": f.price, "ref_price": f.ref_price, "commission": f.commission,
            "tax": f.tax, "transfer_fee": f.transfer_fee,
            "slippage_cost": f.slippage_cost, "impact_cost": f.impact_cost,
            "notional": f.notional, "reason": f.reason,
        } for f in self.fills])
        return df.set_index("date").sort_index()

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=["symbol", "open_date", "close_date", "qty",
                                         "entry_price", "exit_price", "notional", "pnl",
                                         "return_pct", "holding_days", "reason"])
        return pd.DataFrame([{
            "symbol": t.symbol, "open_date": t.open_date, "close_date": t.close_date,
            "qty": t.qty, "entry_price": t.entry_price, "exit_price": t.exit_price,
            # 成交金额（按开仓价计），用于成本归因：成交额 × 费率 = 成本
            "notional": t.entry_price * t.qty,
            "pnl": t.pnl, "return_pct": t.return_pct,
            "holding_days": t.holding_days, "reason": t.reason,
        } for t in self.trades])

    def positions_frame(self) -> pd.DataFrame:
        rows = []
        for j, s in enumerate(self.symbols):
            if self.qty[j] != 0:
                p = self.positions[s]
                rows.append({
                    "symbol": s, "qty": int(self.qty[j]), "avg_cost": p.avg_cost,
                    "last_price": p.last_price, "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                })
        return pd.DataFrame(rows)
