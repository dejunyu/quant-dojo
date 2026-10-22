"""实盘监控：检测因子失效、模型漂移与异常。

回测结束不是终点，是起点。实盘后每天要回答三个问题：

1. **策略还有效吗？** —— 滚动 IC 是否跌到 0 附近？近期收益是否显著低于
   回测预期？
2. **是市场变了还是策略坏了？** —— 如果同类策略一起亏（beta 解释得掉），
   那是市场；如果你的独家 alpha 没了，那是策略坏了。
3. **该下线吗？** —— 本模块给出机械的下线标准，避免"再等等看"的心理陷阱。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..metrics.perf import sharpe, max_drawdown, roll_sharpe, annual_return

__all__ = ["Monitor", "MonitorReport"]


class MonitorReport:
    """监控报告."""

    def __init__(self, data: dict):
        self.data = data

    @property
    def status(self) -> str:
        return str(self.data.get("status", "未知"))

    @property
    def alerts(self) -> list[str]:
        return list(self.data.get("alerts", []))

    def __getitem__(self, key):
        return self.data[key]

    def to_dict(self) -> dict:
        return self.data

    def __repr__(self) -> str:
        return f"<MonitorReport status={self.status} alerts={len(self.alerts)}>"


class Monitor:
    """实盘/纸面交易监控.

    Parameters
    ----------
    expected_sharpe :
        回测给出的预期年化夏普，用于判断实盘是否偏离。
    lookback :
        用于计算近期表现的窗口（交易日）。
    ic_lookback :
        滚动 IC 的窗口。
    """

    def __init__(self, expected_sharpe: float = 1.0, lookback: int = 60,
                 ic_lookback: int = 120, drawdown_limit: float = 0.2,
                 sharpe_drop_limit: float = 0.5):
        self.expected_sharpe = float(expected_sharpe)
        self.lookback = int(lookback)
        self.ic_lookback = int(ic_lookback)
        self.drawdown_limit = float(drawdown_limit)
        self.sharpe_drop_limit = float(sharpe_drop_limit)

    # ------------------------------------------------------------ 主检查
    def run(self, equity: pd.Series, ic_series: pd.Series | None = None,
            factor_values: pd.DataFrame | None = None) -> MonitorReport:
        """跑一遍全部检查。"""
        alerts: list[str] = []
        e = pd.Series(equity).astype(float).dropna()
        r = e.pct_change().dropna()
        data: dict = {}

        if len(e) < 20:
            return MonitorReport({
                "status": "数据不足",
                "alerts": ["样本少于 20 个观测，无法判断"],
            })

        recent = r.iloc[-self.lookback:] if len(r) >= self.lookback else r
        recent_sharpe = sharpe(recent)
        full_sharpe = sharpe(r)
        cur_dd = max_drawdown(e.iloc[-self.lookback * 3:]) if len(e) > self.lookback else max_drawdown(e)
        rs = roll_sharpe(r, window=min(self.lookback, max(20, len(r) // 4)))

        data.update({
            "n_observations": int(len(r)),
            "full_sharpe": round(float(full_sharpe), 3),
            "recent_sharpe": round(float(recent_sharpe), 3),
            "expected_sharpe": round(self.expected_sharpe, 3),
            "recent_annual_return": round(float(annual_return(e.iloc[-self.lookback:])), 4) if len(e) > self.lookback else None,
            "current_drawdown": round(float(cur_dd), 4),
            "rolling_sharpe_last": round(float(rs.dropna().iloc[-1]), 3) if rs.notna().any() else None,
            "rolling_sharpe_mean": round(float(rs.dropna().mean()), 3) if rs.notna().any() else None,
        })

        # --- 检查 1：近期夏普是否显著低于预期 ---
        if np.isfinite(recent_sharpe):
            if recent_sharpe < 0:
                alerts.append(f"近 {len(recent)} 个观测夏普为负（{recent_sharpe:.2f}），策略可能在亏钱")
            elif recent_sharpe < self.expected_sharpe * (1 - self.sharpe_drop_limit):
                alerts.append(
                    f"近期夏普 {recent_sharpe:.2f} 远低于预期 {self.expected_sharpe:.2f}"
                    f"（跌破 {self.sharpe_drop_limit:.0%} 阈值），Alpha 可能已衰减"
                )

        # --- 检查 2：回撤 ---
        if cur_dd < -abs(self.drawdown_limit):
            alerts.append(f"当前回撤 {cur_dd:.2%} 超过 {self.drawdown_limit:.2%}，触发风险预警")

        # --- 检查 3：夏普的时间趋势（是否在单调恶化）---
        rs_clean = rs.dropna()
        if len(rs_clean) >= 10:
            x = np.arange(len(rs_clean))
            slope = np.polyfit(x, rs_clean.values, 1)[0]
            data["sharpe_trend_slope"] = round(float(slope), 6)
            if slope < 0 and len(rs_clean) > 20:
                corr = sps.spearmanr(x, rs_clean.values).statistic
                if corr < -0.5:
                    alerts.append(
                        f"滚动夏普持续下滑（Spearman 趋势 {corr:.2f}），Alpha 呈系统性衰减"
                    )

        # --- 检查 4：因子 IC 是否衰减 ---
        if ic_series is not None and len(ic_series) > 0:
            ic = pd.Series(ic_series).dropna()
            recent_ic = ic.iloc[-self.ic_lookback:] if len(ic) >= self.ic_lookback else ic
            data.update({
                "ic_mean": round(float(ic.mean()), 4),
                "ic_recent_mean": round(float(recent_ic.mean()), 4),
                "ic_recent_std": round(float(recent_ic.std(ddof=1)), 4) if len(recent_ic) > 2 else None,
            })
            if len(recent_ic) > 10:
                t = recent_ic.mean() / recent_ic.std(ddof=1) * np.sqrt(len(recent_ic) - 1)
                data["ic_recent_tstat"] = round(float(t), 3)
                if abs(t) < 1.0:
                    alerts.append(
                        f"近期 IC 均值 {recent_ic.mean():.4f}（t={t:.2f}），已不显著，因子可能失效"
                    )
                elif np.sign(recent_ic.mean()) != np.sign(ic.mean()):
                    alerts.append("近期 IC 方向与历史相反，因子可能已反向（ regime 切换）")

        # --- 检查 5：因子分布是否漂移 ---
        if factor_values is not None and len(factor_values) > 60:
            fv = pd.DataFrame(factor_values)
            half = len(fv) // 2
            a = fv.iloc[:half].stack().dropna()
            b = fv.iloc[half:].stack().dropna()
            if len(a) > 30 and len(b) > 30:
                ks = sps.ks_2samp(a.values, b.values)
                data["factor_distribution_shift_pvalue"] = round(float(ks.pvalue), 5)
                if ks.pvalue < 0.01:
                    alerts.append(
                        f"因子分布发生显著漂移（KS 检验 p={ks.pvalue:.4f}），特征空间可能已改变"
                    )

        # --- 综合判定 ---
        urgent = [a for a in alerts if ("回撤" in a or "为负" in a)]
        decay = [a for a in alerts if ("衰减" in a or "失效" in a or "不显著" in a or "反向" in a)]
        if urgent:
            status = "红色：立即处置"
        elif len(decay) >= 2:
            status = "橙色：Alpha 可能失效，准备下线"
        elif decay:
            status = "黄色：密切观察"
        else:
            status = "绿色：运行正常"

        data["status"] = status
        data["alerts"] = alerts
        data["recommendation"] = _recommend(status, alerts)
        return MonitorReport(data)


def _recommend(status: str, alerts: list[str]) -> str:
    if status.startswith("红色"):
        return "先降杠杆或暂停，核对数据与成交是否正常，再决定是市场波动还是策略失效"
    if status.startswith("橙色"):
        return "缩小仓位，检查因子是否失效；准备切换到备用模型"
    if status.startswith("黄色"):
        return "维持运行但加强监控，每周复核一次 IC 与回撤"
    return "维持当前配置"
