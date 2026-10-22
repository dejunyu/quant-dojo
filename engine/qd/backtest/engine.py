"""回测主引擎.

无前视协议（核心不变量）
------------------------
主循环严格按下面的顺序执行，任何改动都必须保持它：

```
第 i 日：
  1. 解冻 T+1 份额、按 i-1 日收盘价估值（资金曲线只用到 i-1 的信息）
  2. 若第 i-1 日产生了目标订单 → 用**第 i 日**的开盘价成交
  3. 用截至**第 i 日收盘**的数据调用策略 → 产生目标权重（仅记录，不成交）
  4. 风控（止损/熔断）基于 i 日价格判定，标记在下一日执行
```

也就是说：**信号在收盘后生成，订单在下一根 K 线开盘成交**。这是平台保证
回测可信的根本。``tests/test_no_lookahead.py`` 会用"把未来收益打乱"的方式
验证这一点：打乱未来后，任何策略的样本内表现都不应显著变好。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.market import MarketData
from ..metrics.perf import perf_report, to_returns, drawdown_series, monthly_returns_table
from .broker import Broker
from .config import BacktestConfig

__all__ = ["Backtest", "BacktestResult"]


class BacktestResult:
    """回测结果容器."""

    def __init__(self, equity: pd.Series, broker: Broker, config: BacktestConfig,
                 weights: pd.DataFrame, benchmark: pd.Series | None = None,
                 daily_stats: pd.DataFrame | None = None):
        self.equity = equity
        self.broker = broker
        self.config = config
        self.weights = weights
        self.benchmark = benchmark
        self.daily_stats = daily_stats if daily_stats is not None else pd.DataFrame()
        self._metrics: dict | None = None

    # ------------------------------------------------------------ 数据
    @property
    def returns(self) -> pd.Series:
        return to_returns(self.equity)

    @property
    def trades(self) -> pd.DataFrame:
        return self.broker.trades_frame()

    @property
    def fills(self) -> pd.DataFrame:
        return self.broker.fills_frame()

    @property
    def positions(self) -> pd.DataFrame:
        return self.broker.positions_frame()

    @property
    def metrics(self) -> dict:
        if self._metrics is None:
            self._metrics = perf_report(
                self.equity,
                benchmark=self.benchmark,
                rf=self.config.risk_free,
                weights=self.weights,
                n_trials=getattr(self.config, "n_trials", 1),
            )
        return self._metrics

    def monthly_returns(self) -> pd.DataFrame:
        return monthly_returns_table(self.equity)

    def trade_stats(self) -> dict:
        t = self.trades
        if t.empty:
            return {"n_trades": 0}
        win = t[t["pnl"] > 0]
        loss = t[t["pnl"] <= 0]
        plr = float("nan")
        if len(win) and len(loss) and loss["pnl"].mean() != 0:
            plr = float(abs(win["pnl"].mean() / loss["pnl"].mean()))
        return {
            "n_trades": int(len(t)),
            "win_rate": float(len(win) / len(t)),
            "avg_win": float(win["pnl"].mean()) if len(win) else 0.0,
            "avg_loss": float(loss["pnl"].mean()) if len(loss) else 0.0,
            "profit_loss_ratio": plr,
            "avg_holding_days": float(t["holding_days"].mean()),
            "total_pnl": float(t["pnl"].sum()),
            "best_trade": float(t["pnl"].max()),
            "worst_trade": float(t["pnl"].min()),
        }

    def cost_breakdown(self) -> dict:
        f = self.fills
        if f.empty:
            return {}
        cols = ["commission", "tax", "transfer_fee", "slippage_cost", "impact_cost"]
        return {
            "commission": float(f["commission"].sum()),
            "stamp_tax": float(f["tax"].sum()),
            "transfer_fee": float(f["transfer_fee"].sum()),
            "slippage_cost": float(f["slippage_cost"].sum()),
            "impact_cost": float(f["impact_cost"].sum()),
            "total_cost": float(f[cols].sum().sum()),
            "cost_drag_annual": _cost_drag_annual(f, self.equity),
            "n_fills": int(len(f)),
            "avg_fill_cost_bps": float(
                (f[cols].sum(axis=1) / f["notional"].replace(0, np.nan)).mean() * 1e4
            ),
        }

    def summary(self) -> str:
        m = self.metrics
        fm = lambda k, fmt="{:.2f}": (fmt.format(m[k]) if _ok(m.get(k)) else "n/a")  # noqa: E731
        lines = [
            f"回测区间：{m['start']} ~ {m['end']}（{m['years']} 年）",
            f"累计收益：{m['cumulative_return']:.2%}   年化：{m['annual_return']:.2%}   年化波动：{m['annual_vol']:.2%}",
            f"夏普：{fm('sharpe')}   索提诺：{fm('sortino')}   卡玛：{fm('calmar')}",
            f"最大回撤：{m['max_drawdown']:.2%}   Ulcer：{fm('ulcer_index')}   VaR95：{m['var_95']:.2%}",
            f"胜率：{fm('win_rate','{:.2%}')}   盈亏比：{fm('profit_loss_ratio')}   偏度：{fm('skew')} 峰度：{fm('kurtosis')}",
            f"PSR：{fm('probabilistic_sharpe','{:.3f}')}   DSR（{m['n_trials']} 次试验）：{fm('deflated_sharpe','{:.3f}')}",
        ]
        if "benchmark_return" in m:
            lines.append(
                f"基准累计：{m['benchmark_return']:.2%}   超额年化：{m['excess_annual_return']:.2%}   "
                f"Alpha：{m['alpha']:.2%}   Beta：{m['beta']:.2f}   IR：{fm('information_ratio')}"
            )
        ts = self.trade_stats()
        if ts.get("n_trades"):
            lines.append(
                f"交易 {ts['n_trades']} 笔   胜率 {ts['win_rate']:.2%}   盈亏比 {ts['profit_loss_ratio']:.2f}   "
                f"平均持有 {ts['avg_holding_days']:.1f} 天"
            )
        cb = self.cost_breakdown()
        if cb:
            scale = self.broker.initial_cash
            drag = cb["cost_drag_annual"] / scale if scale else float("nan")
            lines.append(
                f"总成本 {cb['total_cost']:,.0f} 元（年均 {cb['cost_drag_annual']:,.0f} 元，"
                f"≈ 初始资金 {drag:.2%}/年）  成交 {cb['n_fills']} 笔，平均 {cb['avg_fill_cost_bps']:.1f} bp"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics,
            "trade_stats": self.trade_stats(),
            "cost_breakdown": self.cost_breakdown(),
            "equity": [{"date": str(d.date()), "equity": float(v)} for d, v in self.equity.items()],
        }

    def __repr__(self) -> str:
        return f"BacktestResult({self.equity.index[0].date()} ~ {self.equity.index[-1].date()}, Sharpe={self.metrics['sharpe']:.2f})"


def _ok(v) -> bool:
    return v is not None and np.isfinite(v)


_REBALANCE_ALIASES = {
    "1d": "daily", "d": "daily", "every_day": "daily",
    "1w": "weekly", "w": "weekly", "week": "weekly",
    "1m": "monthly", "m": "monthly", "month": "monthly", "monthly_first": "monthly",
    "1q": "quarterly", "q": "quarterly", "quarter": "quarterly",
    "3m": "quarterly", "6m": "semiannual", "1y": "annual", "y": "annual",
}


def _is_rebalance_date(dates: pd.DatetimeIndex, i: int, freq: str, offset: int = 0) -> bool:
    """判断第 i 日是否为调仓日。

    ``daily`` 每日；``weekly`` 每周一（可按 offset 偏移到周内其它天）；
    ``monthly`` 每月第一个交易日；``quarterly`` 每季度第一个交易日。
    """
    freq = _REBALANCE_ALIASES.get(str(freq).lower(), str(freq).lower())
    if freq == "daily":
        return True
    d = dates[i]
    if freq == "weekly":
        # 周内序号变化即为一周的开始（用 ISO weekday - offset 判断跨周）
        if i == 0:
            return True
        prev = dates[i - 1]
        return (d.isocalendar()[1] != prev.isocalendar()[1]) or (prev.isocalendar()[0] != d.isocalendar()[0])
    if freq == "monthly":
        if i == 0:
            return True
        return d.month != dates[i - 1].month
    if freq == "quarterly":
        if i == 0:
            return True
        return d.quarter != dates[i - 1].quarter
    if freq == "semiannual":
        if i == 0:
            return True
        return (d.month - 1) // 6 != (dates[i - 1].month - 1) // 6
    if freq == "annual":
        if i == 0:
            return True
        return d.year != dates[i - 1].year
    raise ValueError(
            f"未知调仓频率 {freq}；可选 daily / weekly / monthly / quarterly / semiannual / annual"
            f"（也接受 1M / 3M / 6M / 1W 这类写法）"
        )


def _cost_drag_annual(fills: pd.DataFrame, equity: pd.Series) -> float:
    """成本年化拖累 = 年均成本 / 平均资产规模。

    注意：``equity`` 在这里已经归一化为净值曲线（起点 1.0），所以必须先
    还原成名义资金（用初始现金），否则会把比例放大 1e6 倍。
    """
    if fills.empty or len(equity) < 2:
        return float("nan")
    cols = ["commission", "tax", "transfer_fee", "slippage_cost", "impact_cost"]
    total = float(fills[cols].sum().sum())
    years = len(equity) / 252
    if years <= 0:
        return float("nan")
    return float(total / years)


def _to_target_array(out, symbols: list[str]) -> np.ndarray:
    """把策略返回的目标权重统一成按 ``symbols`` 顺序排列的浮点数组。

    支持三种返回形式：

    * ``pd.Series``（index 为标的代码）—— 最常用；
    * ``np.ndarray`` / ``list``（长度等于标的数，按 ``symbols`` 顺序）；
    * ``dict``（标的 -> 权重）。

    把 ndarray 直接喂给 ``pd.Series()`` 会把它当成"索引 0..N-1"，再 reindex
    到标的代码后全部变成 NaN——这是一个非常隐蔽的错误。
    """
    import numpy as np
    import pandas as pd

    if isinstance(out, pd.Series):
        if out.index.equals(pd.Index(symbols)):
            # 绝大多数策略返回的 Series 本来就是这个顺序，直接取值即可
            arr = out.to_numpy(dtype=float, copy=False)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return out.reindex(symbols).fillna(0.0).to_numpy(dtype=float)
    if isinstance(out, dict):
        return pd.Series(out).reindex(symbols).fillna(0.0).to_numpy(dtype=float)
    arr = np.asarray(out, dtype=float).ravel()
    if arr.shape[0] != len(symbols):
        raise ValueError(
            f"策略返回了 {arr.shape[0]} 个权重，但标的数是 {len(symbols)}。"
            f"请返回 pd.Series(index=标的代码) 或与之等长的数组。"
        )
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


class Backtest:
    """回测引擎.

    Parameters
    ----------
    md :
        行情数据。
    strategy :
        策略对象，必须实现 ``weights(ctx) -> pd.Series``；也可以是可调用对象
        ``f(ctx) -> Series``。``ctx`` 提供到**当日收盘为止**的全部数据切片。
    config :
        回测配置。
    """

    def __init__(self, md: MarketData, strategy, config: BacktestConfig | None = None):
        if not isinstance(md, MarketData):
            raise TypeError("md 必须是 MarketData")
        self.md = md
        self.strategy = strategy
        self.config = config or BacktestConfig()

    # ------------------------------------------------------------ 主循环
    def run(self) -> BacktestResult:
        cfg = self.config
        broker = Broker(self.md, cfg)
        dates = self.md.dates
        n = len(dates)
        symbols = list(self.md.symbols)

        equity_arr = np.empty(n, dtype=float)
        weight_rows = np.empty((n, len(symbols)), dtype=float)
        n_sym = len(symbols)
        close_a = broker._close_a  # noqa: SLF001 - 热路径直接用预取的数组
        cash_arr = np.empty(n, dtype=float)
        gross_arr = np.empty(n, dtype=float)
        npos_arr = np.empty(n, dtype=int)

        pending: np.ndarray | None = None
        pending_reason = ""
        has_target = False          # 是否已有一次目标权重（区分"从未算过"与"已持有"）
        halted = False
        strategy_name = getattr(self.strategy, "name", "strategy")

        # 基准
        bench_arr = None
        if cfg.benchmark and cfg.benchmark in symbols:
            bc = self.md.close()[cfg.benchmark].to_numpy(dtype=float)
            bench_arr = bc / bc[0]

        peak_equity = cfg.initial_cash

        for i in range(n):
            # --- 1. 解冻 T+1；用截至 i-1 日的信息估值 ---
            broker.unlock_daily()
            equity_arr[i] = cfg.initial_cash if i == 0 else broker.mark_to_market(i - 1)

            # --- 2. 执行上一日生成的订单（用第 i 日的价格成交）---
            if pending is not None and i > 0 and not halted:
                broker.order_target_weights(i, pending, reason=pending_reason)
                pending = None

            # --- 3. 风控（基于第 i 日价格）---
            if i > 0 and not halted:
                self._risk_check(broker, i)

            # 回撤熔断
            if cfg.max_drawdown_stop is not None and i > 5:
                peak_equity = max(peak_equity, equity_arr[i])
                dd = equity_arr[i] / peak_equity - 1.0
                if dd < -abs(cfg.max_drawdown_stop):
                    halted = True
                    broker.order_target_weights(i, np.zeros(len(symbols)), reason="回撤熔断清仓")
                    pending = None

            # --- 4. 策略用 ≤ i 日数据生成目标权重（下一日才成交）---
            if i >= cfg.warmup and i < n - 1 and not halted:
                # 调仓频率控制：只在调仓日生成新目标，其余日子沿用上一次的目标。
                # 这既是真实的做法（降低换手成本），也让"调仓周期"成为可研究的参数。
                is_rebalance_day = _is_rebalance_date(dates, i, cfg.rebalance, cfg.rebalance_offset)
                if is_rebalance_day or not has_target:
                    ctx = _Context(self.md, i)
                    try:
                        out = self.strategy.weights(ctx) if hasattr(self.strategy, "weights") else self.strategy(ctx)
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(
                            f"策略在第 {i} 日（{dates[i].date()}）计算失败：{exc}"
                        ) from exc
                    if out is not None:
                        pending = _to_target_array(out, symbols)
                        has_target = True
                        pending_reason = strategy_name
            elif i >= n - 1:
                pending = None

            # --- 5. 记录当日状态（直接用 numpy，避免每天构造 pandas 对象）---
            equity_arr[i] = broker.mark_to_market(i)
            px = close_a[i]
            vals = broker.qty * np.where(np.isfinite(px), px, 0.0)
            total = equity_arr[i]
            weight_rows[i] = vals / total if total > 0 else 0.0
            cash_arr[i] = broker.cash
            gross_arr[i] = float(np.abs(vals).sum() / total) if total > 0 else 0.0
            npos_arr[i] = int(np.count_nonzero(broker.qty))

        equity = pd.Series(equity_arr, index=dates, name="equity")
        equity = equity / equity.iloc[0]

        weights = pd.DataFrame(weight_rows, index=dates, columns=symbols)
        daily = pd.DataFrame({
            "equity": equity_arr, "cash": cash_arr,
            "gross": gross_arr, "n_positions": npos_arr,
        }, index=dates)

        bench_series = None
        if bench_arr is not None:
            bench_series = pd.Series(bench_arr, index=dates, name="benchmark")

        return BacktestResult(equity, broker, self.config, weights, bench_series, daily)

    # ------------------------------------------------------------ 风控
    def _risk_check(self, broker: Broker, i: int) -> None:
        cfg = self.config
        if cfg.stop_loss is None and cfg.take_profit is None and cfg.trailing_stop is None:
            return
        px = broker._close_a[i]
        hit = np.zeros(broker.n_sym, dtype=bool)
        held = broker.qty > 0
        if not held.any():
            return
        with np.errstate(divide="ignore", invalid="ignore"):
            pnl_pct = np.where(broker.avg_cost > 0, px / broker.avg_cost - 1.0, 0.0)
        if cfg.stop_loss is not None:
            hit |= held & (pnl_pct <= -abs(cfg.stop_loss))
        if cfg.take_profit is not None:
            hit |= held & (pnl_pct >= abs(cfg.take_profit))
        if cfg.trailing_stop is not None:
            peak = np.where(np.isnan(broker.peak), px, np.maximum(broker.peak, px))
            dd = px / np.where(peak > 0, peak, np.nan) - 1.0
            hit |= held & (dd <= -abs(cfg.trailing_stop))
        if not hit.any():
            return
        w = np.array(broker.current_weights(i).to_numpy(dtype=float), copy=True)
        w[hit] = 0.0
        broker.order_target_weights(i, w, reason="风控平仓")


class _Context:
    """策略上下文：只暴露截至 ``i`` 日（含）的数据，物理上杜绝前视。

    为了让策略写起来自然，这里对切片结果做了缓存：同一个 ``i`` 下重复调用
    ``ctx.close()`` 不会重复切片。
    """

    __slots__ = ("md", "i", "date", "_cache")

    def __init__(self, md: MarketData, i: int):
        self.md = md
        self.i = i
        self.date = md.dates[i]
        self._cache: dict[str, pd.DataFrame] = {}

    def _slice(self, key: str) -> pd.DataFrame:
        cached = self._cache.get(key)
        if cached is None:
            src = self.md.panels.get(key) if key in self.md.panels else self.md.fundamentals.get(key)
            if src is None:
                raise KeyError(f"字段 {key} 不存在")
            cached = src.iloc[: self.i + 1]
            self._cache[key] = cached
        return cached

    def close(self) -> pd.DataFrame:
        return self._slice("close")

    def open(self) -> pd.DataFrame:
        return self._slice("open")

    def high(self) -> pd.DataFrame:
        return self._slice("high")

    def low(self) -> pd.DataFrame:
        return self._slice("low")

    def volume(self) -> pd.DataFrame:
        return self._slice("volume")

    def amount(self) -> pd.DataFrame:
        return self._slice("amount") if "amount" in self.md else self.close() * self.volume()

    def vwap(self) -> pd.DataFrame:
        return self.md.vwap().iloc[: self.i + 1]

    def returns(self) -> pd.DataFrame:
        # 策略几乎每天都会调用它；整块算一次再切片，比每天对截断后的面板
        # 重新 pct_change 便宜得多（结果完全相同）。
        md = self.md
        cache = getattr(md, "_ret_cache", None)
        if cache is None:
            cache = md.close().pct_change()
            try:
                md._ret_cache = cache  # noqa: SLF001
            except AttributeError:  # pragma: no cover
                pass
        return cache.iloc[: self.i + 1]

    def field(self, name: str) -> pd.DataFrame:
        return self._slice(str(name).lower())

    def fundamental(self, name: str) -> pd.DataFrame:
        return self._slice(str(name).lower())

    @property
    def symbols(self) -> list[str]:
        return list(self.md.symbols)

    @property
    def n_days(self) -> int:
        return self.i + 1

    def __repr__(self) -> str:
        return f"<Context i={self.i} date={self.date.date()}>"
