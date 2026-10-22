"""HTTP API.

启动：
```bash
qd serve            # 或
uvicorn qd.api.app:app --reload
```

设计原则：**每个接口都是幂等的、无状态的**（除了数据缓存），前端可以随
时刷新。所有回测在服务端同步执行，返回结构化 JSON（含净值曲线、指标、
交易、成本分解），前端只负责画图。
"""
from __future__ import annotations

import io
import traceback
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..backtest import Backtest, BacktestConfig, CostModel
from ..data.market import MarketData
from ..data.synthetic import generate_demo_market
from ..factors import get_factor, list_factors
from ..live.monitor import Monitor
from ..live.risk_gate import RiskGate
from ..metrics.perf import perf_report
from ..optimize import grid_search, walk_forward, purged_kfold, monte_carlo
from ..optimize.portfolio import compare_allocators
from ..research.lab import FactorLab
from ..research.logbook import Logbook
from ..research.overfit import OverfitLab
from ..strategy import builtin as S
from ..tutor import load_curriculum

__all__ = ["create_app", "app"]


# ---------------------------------------------------------------- 数据管理
class DataStore:
    """进程内数据缓存（避免每次请求都重新生成合成数据）。"""

    def __init__(self):
        self._cache: dict[str, MarketData] = {}
        self._logbook: Logbook | None = None

    def get(self, key: str = "demo", **kwargs) -> MarketData:
        if key not in self._cache:
            if key == "demo":
                self._cache[key] = generate_demo_market(**kwargs)
            else:
                raise KeyError(f"未知数据集 {key}")
        return self._cache[key]

    def put(self, key: str, md: MarketData) -> None:
        self._cache[key] = md

    @property
    def logbook(self) -> Logbook:
        if self._logbook is None:
            self._logbook = Logbook()
        return self._logbook


STORE = DataStore()


def _curriculum():
    if not hasattr(_curriculum, "_c"):
        _curriculum._c = load_curriculum()
    return _curriculum._c


# ---------------------------------------------------------------- 请求模型
class BacktestRequest(BaseModel):
    dataset: str = "demo"
    strategy: str = "dual_momentum"
    params: dict[str, Any] = Field(default_factory=dict)
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    stamp_tax: float = 0.001
    slippage_bps: float = 5.0
    impact_coeff: float = 0.1
    max_participation: float = 0.1
    execution: str = "next_open"
    t_plus_1: bool = True
    allow_short: bool = False
    benchmark: str | None = None
    risk_free: float = 0.02
    stop_loss: float | None = None
    take_profit: float | None = None
    max_drawdown_stop: float | None = None
    min_trade_value: float = 0.005
    n_trials: int = 1
    rebalance: str = "daily"
    start: str | None = None
    end: str | None = None


STRATEGY_FACTORIES = {
    "buy_and_hold": lambda **kw: S.buy_and_hold(**kw),
    "equal_weight": lambda **kw: S.equal_weight(**kw),
    "sma_cross": lambda **kw: S.sma_cross(**kw),
    "dual_momentum": lambda **kw: S.dual_momentum(**kw),
    "turtle_breakout": lambda **kw: S.turtle_breakout(**kw),
    "bollinger_reversion": lambda **kw: S.bollinger_reversion(**kw),
    "mean_reversion_basket": lambda **kw: S.mean_reversion_basket(**kw),
    "risk_parity": lambda **kw: S.risk_parity(**kw),
    "vol_target": lambda **kw: S.vol_target(**kw),
    "rotation": lambda **kw: S.rotation(**kw),
    "factor_long_only": lambda **kw: S.factor_long_only(**kw),
    "multi_factor_rank": lambda **kw: S.multi_factor_rank(**kw),
}


# ---------------------------------------------------------------- 应用
def create_app() -> FastAPI:
    app = FastAPI(
        title="Quant Dojo API",
        description="量化投资学习与实战平台（中文）",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    # ------------------------------------------------------------ 课程
    @app.get("/api/curriculum")
    def api_curriculum():
        c = _curriculum()
        return {"stats": c.stats(), "categories": c.categories()}

    @app.get("/api/curriculum/lesson/{lesson_id}")
    def api_lesson(lesson_id: str):
        c = _curriculum()
        lesson = c.get(lesson_id)
        if not lesson:
            raise HTTPException(404, f"课程 {lesson_id} 不存在")
        return lesson.to_dict(with_body=True)

    @app.get("/api/curriculum/search")
    def api_search(q: str = ""):
        return {"results": _curriculum().search(q)}

    @app.get("/api/curriculum/path")
    def api_path():
        return {"path": _curriculum().learning_path()}

    # ------------------------------------------------------------ 因子
    @app.get("/api/factors")
    def api_factors():
        return {"factors": list_factors(), "n": len(list_factors())}

    @app.post("/api/factors/compute")
    def api_factor_compute(payload: dict):
        name = payload.get("name", "mom_12_1")
        md = STORE.get(payload.get("dataset", "demo"))
        f = get_factor(name)
        panel = f.compute(md)
        return {
            "name": name,
            "category": f.category,
            "direction": f.direction,
            "description": f.description,
            "formula": f.formula,
            "dates": [str(d.date()) for d in panel.index[:: max(1, len(panel) // 200)]],
            "symbols": list(panel.columns),
            "summary": {
                "mean": _nan(panel.mean().mean()),
                "std": _nan(panel.std().mean()),
                "coverage": _nan(panel.notna().mean().mean()),
            },
            "last_cross_section": panel.iloc[-1].dropna().round(6).to_dict(),
        }

    @app.post("/api/research/factor")
    def api_factor_report(payload: dict):
        name = payload.get("name", "mom_12_1")
        horizon = int(payload.get("horizon", 20))
        md = STORE.get(payload.get("dataset", "demo"))
        lab = FactorLab(md, horizon=horizon)
        rep = lab.report(name, horizons=tuple(payload.get("horizons", (1, 5, 20))))
        return _jsonable(rep)

    @app.post("/api/research/ic")
    def api_ic(payload: dict):
        name = payload.get("name", "mom_12_1")
        md = STORE.get(payload.get("dataset", "demo"))
        lab = FactorLab(md, horizon=int(payload.get("horizon", 20)))
        s = lab.ic_series(name)
        return {
            "name": name,
            "ic_series": [{"date": str(d.date()), "ic": _nan(v)} for d, v in s.items()],
            "summary": lab.ic_summary(name),
            "decay": lab.ic_decay(name).to_dict("records"),
            "turnover": lab.turnover(name),
        }

    # ------------------------------------------------------------ 回测
    @app.get("/api/strategies")
    def api_strategies():
        return {
            "strategies": sorted(STRATEGY_FACTORIES),
            "descriptions": {
                k: getattr(STRATEGY_FACTORIES[k](), "description", "") for k in sorted(STRATEGY_FACTORIES)
            },
        }

    @app.post("/api/backtest")
    def api_backtest(req: BacktestRequest):
        try:
            factory = STRATEGY_FACTORIES.get(req.strategy)
            if factory is None:
                raise HTTPException(400, f"未知策略 {req.strategy}；可选：{sorted(STRATEGY_FACTORIES)}")
            md = STORE.get(req.dataset)
            if req.start or req.end:
                md = md.slice(start=req.start, end=req.end)

            cost = CostModel(
                commission_rate=req.commission_rate,
                stamp_tax=req.stamp_tax,
                slippage_bps=req.slippage_bps,
                impact_coeff=req.impact_coeff,
            )
            cfg = BacktestConfig(
                initial_cash=req.initial_cash, cost=cost,
                execution=req.execution, t_plus_1=req.t_plus_1,
                allow_short=req.allow_short, benchmark=req.benchmark,
                risk_free=req.risk_free, stop_loss=req.stop_loss,
                take_profit=req.take_profit, max_drawdown_stop=req.max_drawdown_stop,
                min_trade_value=req.min_trade_value,
                max_participation=req.max_participation,
                rebalance=req.rebalance,
            )
            cfg.n_trials = req.n_trials  # type: ignore[attr-defined]
            strat = factory(**req.params)
            res = Backtest(md, strat, cfg).run()
            return {
                "strategy": req.strategy,
                "params": req.params,
                "metrics": _jsonable(res.metrics),
                "trade_stats": _jsonable(res.trade_stats()),
                "cost_breakdown": _jsonable(res.cost_breakdown()),
                "equity": [{"date": str(d.date()), "value": round(float(v), 6)} for d, v in res.equity.items()],
                "benchmark": (
                    [{"date": str(d.date()), "value": round(float(v), 6)} for d, v in res.benchmark.items()]
                    if res.benchmark is not None else None
                ),
                "weights_top": _top_weights(res.weights),
                "daily": {
                    "gross": [round(float(x), 4) for x in res.daily_stats["gross"].tolist()],
                    "n_positions": [int(x) for x in res.daily_stats["n_positions"].tolist()],
                },
                "fills_sample": _jsonable(res.fills.tail(50).reset_index().to_dict("records")),
                "trades_sample": _jsonable(res.trades.tail(50).to_dict("records")),
                "monthly": _jsonable(res.monthly_returns().fillna(0).round(4).to_dict("index")),
            }
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=400, content={
                "error": str(exc), "traceback": traceback.format_exc().splitlines()[-3:],
            })

    # ------------------------------------------------------------ 优化
    @app.post("/api/optimize/grid")
    def api_grid(payload: dict):
        md = STORE.get(payload.get("dataset", "demo"))
        name = payload.get("strategy", "dual_momentum")
        grid = payload.get("param_grid", {"lookback": [60, 120], "top_k": [5, 10]})
        factory = STRATEGY_FACTORIES.get(name)
        if factory is None:
            raise HTTPException(400, f"未知策略 {name}")
        out = grid_search(md, factory, grid, metric=payload.get("metric", "sharpe"),
                          verbose=False)
        return {
            "table": _jsonable(out["table"].to_dict("records")),
            "best_params": _jsonable(out["best_params"]),
            "n_trials": out["n_trials"],
            "deflated_sharpe": out["deflated_sharpe"],
            "warning": out["warning"],
            "elapsed_sec": out["elapsed_sec"],
        }

    @app.post("/api/optimize/walk_forward")
    def api_wf(payload: dict):
        md = STORE.get(payload.get("dataset", "demo"))
        name = payload.get("strategy", "dual_momentum")
        factory = STRATEGY_FACTORIES.get(name)
        if factory is None:
            raise HTTPException(400, f"未知策略 {name}")
        out = walk_forward(
            md, factory,
            param_grid=payload.get("param_grid") or {"lookback": [60, 120], "top_k": [5, 10]},
            n_splits=int(payload.get("n_splits", 5)),
            verbose=False,
        )
        return {
            "folds": _jsonable(out["folds"].to_dict("records")),
            "sharpe": out["sharpe"], "annual_return": out["annual_return"],
            "max_drawdown": out["max_drawdown"], "n_folds": out["n_folds"],
            "equity": [{"date": str(d.date()), "value": round(float(v), 6)} for d, v in out["equity"].items()],
            "note": out["note"],
        }

    @app.post("/api/analyze/monte_carlo")
    def api_mc(payload: dict):
        md = STORE.get(payload.get("dataset", "demo"))
        r = md.close().pct_change().mean(axis=1).dropna()  # 等权市场收益作为示例输入
        out = monte_carlo(r, n_sims=int(payload.get("n_sims", 500)),
                          horizon=int(payload.get("horizon", 252)))
        return _jsonable(out)

    @app.post("/api/analyze/allocators")
    def api_allocators(payload: dict):
        md = STORE.get(payload.get("dataset", "demo"))
        r = md.returns().iloc[-int(payload.get("lookback", 252)):]
        return {"table": _jsonable(compare_allocators(r).to_dict("records"))}

    # ------------------------------------------------------------ 过拟合
    @app.post("/api/overfit/white_noise")
    def api_white_noise(payload: dict):
        name = payload.get("strategy", "dual_momentum")
        grid = payload.get("param_grid", {"lookback": [20, 60, 120], "top_k": [3, 5, 10]})
        factory = STRATEGY_FACTORIES.get(name)
        if factory is None:
            raise HTTPException(400, f"未知策略 {name}")
        md = STORE.get(payload.get("dataset", "demo"))
        lab = OverfitLab(md)
        return _jsonable(lab.white_noise_experiment(
            grid, factory, n_symbols=int(payload.get("n_symbols", 30)),
            seed=int(payload.get("seed", 0)),
        ))

    # ------------------------------------------------------------ 实盘监控
    @app.post("/api/monitor")
    def api_monitor(payload: dict):
        md = STORE.get(payload.get("dataset", "demo"))
        name = payload.get("strategy", "dual_momentum")
        factory = STRATEGY_FACTORIES.get(name)
        res = Backtest(md, factory(**payload.get("params", {})), BacktestConfig()).run()
        mon = Monitor(expected_sharpe=float(payload.get("expected_sharpe", 1.0)))
        ic = None
        if payload.get("factor"):
            lab = FactorLab(md, horizon=20)
            ic = lab.ic_series(payload["factor"])
        rep = mon.run(res.equity, ic_series=ic)
        return _jsonable(rep.to_dict())

    @app.get("/api/risk_gate")
    def api_risk_gate():
        return RiskGate().to_dict()

    # ------------------------------------------------------------ 研究日志
    @app.get("/api/logbook")
    def api_logbook():
        df = STORE.logbook.all()
        return {"count": len(df), "records": _jsonable(df.to_dict("records"))}

    @app.post("/api/logbook")
    def api_logbook_add(payload: dict):
        rid = STORE.logbook.log(
            hypothesis=payload.get("hypothesis", ""),
            strategy=payload.get("strategy", ""),
            params=payload.get("params"),
            notes=payload.get("notes", ""),
            tags=payload.get("tags"),
            metrics=payload.get("metrics"),
            verdict=payload.get("verdict", ""),
        )
        return {"id": rid}

    @app.delete("/api/logbook/{rid}")
    def api_logbook_del(rid: int):
        STORE.logbook.delete(rid)
        return {"ok": True}

    # ------------------------------------------------------------ 数据
    @app.get("/api/dataset/info")
    def api_dataset_info(dataset: str = "demo"):
        md = STORE.get(dataset)
        return md.sanity_report()

    @app.get("/api/dataset/prices")
    def api_prices(dataset: str = "demo", symbol: str | None = None, limit: int = 500):
        md = STORE.get(dataset)
        px = md.close()
        if symbol and symbol in px.columns:
            px = px[[symbol]]
        tail = px.tail(int(limit))
        return {
            "dates": [str(d.date()) for d in tail.index],
            "series": {c: [round(float(v), 4) for v in tail[c].tolist()] for c in tail.columns[:10]},
        }

    @app.get("/api/health")
    def api_health():
        return {
            "status": "ok",
            "version": "0.1.0",
            "n_factors": len(list_factors()),
            "n_lessons": len(_curriculum()),
            "datasets": list(STORE._cache),
        }

    # ------------------------------------------------------------ 前端静态文件
    # 如果 web/dist 存在，就直接由后端托管前端：一个端口解决所有问题。
    dist = _web_dist()
    if dist is not None:
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles

        assets = dist / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        @app.get("/")
        def index():
            return FileResponse(dist / "index.html")

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str):
            # 让前端路由刷新时也能正确返回 index.html
            candidate = (dist / full_path).resolve()
            if candidate.is_file() and str(candidate).startswith(str(dist.resolve())):
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


def _web_dist():
    """定位前端构建产物目录（web/dist）。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]  # <quant>/
    dist = root / "web" / "dist"
    return dist if (dist / "index.html").exists() else None


app = create_app()


# ---------------------------------------------------------------- 工具
def _nan(x) -> float | None:
    try:
        f = float(x)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _jsonable(obj):
    """把 numpy / pandas / NaN 转成 JSON 安全的对象。"""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    if isinstance(obj, pd.Series):
        return _jsonable(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict("records"))
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if obj is None or isinstance(obj, (str, int)):
        return obj
    if pd.isna(obj):  # type: ignore[arg-type]
        return None
    return str(obj)


def _top_weights(weights: pd.DataFrame, n: int = 12) -> list[dict]:
    """取最后一期权重最大的 n 个标的 + 权重的时间序列（抽样）。"""
    last = weights.iloc[-1].sort_values(ascending=False)
    top = last.head(n)
    step = max(1, len(weights) // 200)
    w = weights[top.index].iloc[::step]
    return {
        "dates": [str(d.date()) for d in w.index],
        "series": {c: [round(float(v), 5) for v in w[c].tolist()] for c in w.columns},
    }
