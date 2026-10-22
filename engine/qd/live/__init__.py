"""实盘层：监控、风控闸门、回测-实盘一致性核对（纸面交易）.

定位
----
本模块**不下单**（那是券商 API 的事），它做三件事：

1. **纸面交易（Paper Trading）**：用真实/合成数据模拟"每天只看到当天数据"
   的下单流程，用来验证你的**代码链路**是否与回测一致；
2. **风控闸门（RiskGate）**：下单前的最后一道检查，把所有"回测里没想到但
   实盘会出事"的规则集中在这里；
3. **监控（Monitor）**：实盘后每天的异常检测——因子失效、回撤超限、成交
   异常、模型漂移。
"""
from .monitor import Monitor, MonitorReport
from .risk_gate import RiskGate, OrderIntent, GateResult
from .paper import PaperTrader

__all__ = ["Monitor", "MonitorReport", "RiskGate", "OrderIntent", "GateResult", "PaperTrader"]
