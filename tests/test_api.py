"""API、课程体系与实盘监控测试."""
import numpy as np
import pandas as pd
import pytest

from fastapi.testclient import TestClient

from qd.api import create_app
from qd.live.monitor import Monitor
from qd.live.risk_gate import RiskGate, OrderIntent
from qd.live.paper import PaperTrader
from qd.tutor import load_curriculum


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


# ------------------------------------------------------------------ API
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["n_factors"] >= 40
    assert d["n_lessons"] >= 60


def test_curriculum_api(client):
    r = client.get("/api/curriculum")
    assert r.status_code == 200
    d = r.json()
    assert d["stats"]["n_lessons"] >= 60
    assert len(d["categories"]) >= 10

    lid = d["categories"][0]["lessons"][0]["id"]
    r2 = client.get(f"/api/curriculum/lesson/{lid}")
    assert r2.status_code == 200
    assert "body" in r2.json()
    assert r2.json()["title"]

    r3 = client.get("/api/curriculum/lesson/不存在")
    assert r3.status_code == 404


def test_curriculum_search_api(client):
    r = client.get("/api/curriculum/search", params={"q": "夏普"})
    assert r.status_code == 200
    assert len(r.json()["results"]) > 0


def test_curriculum_path_api(client):
    r = client.get("/api/curriculum/path")
    assert r.status_code == 200
    assert len(r.json()["path"]) >= 60


def test_factors_api(client):
    r = client.get("/api/factors")
    assert r.status_code == 200
    assert len(r.json()["factors"]) >= 40


def test_factor_compute_api(client):
    r = client.post("/api/factors/compute", json={"name": "mom_12_1"})
    assert r.status_code == 200
    d = r.json()
    assert d["name"] == "mom_12_1"
    assert "last_cross_section" in d


def test_backtest_api(client):
    r = client.post("/api/backtest", json={"strategy": "dual_momentum",
                                           "params": {"lookback": 120, "top_k": 10}})
    assert r.status_code == 200, r.text[:400]
    d = r.json()
    assert "metrics" in d and "equity" in d
    assert len(d["equity"]) > 100
    assert "sharpe" in d["metrics"]


def test_backtest_api_unknown_strategy(client):
    r = client.post("/api/backtest", json={"strategy": "不存在的策略"})
    assert r.status_code in (400, 404)


def test_grid_api(client):
    r = client.post("/api/optimize/grid", json={
        "strategy": "dual_momentum",
        "param_grid": {"lookback": [60, 120], "top_k": [5, 10]},
    })
    assert r.status_code == 200, r.text[:300]
    d = r.json()
    assert d["n_trials"] == 4
    assert "deflated_sharpe" in d


def test_walk_forward_api(client):
    r = client.post("/api/optimize/walk_forward", json={
        "strategy": "dual_momentum",
        "param_grid": {"lookback": [60, 120], "top_k": [5, 10]},
        "n_splits": 3,
    })
    assert r.status_code == 200, r.text[:300]
    assert r.json()["n_folds"] >= 2


def test_monitor_api(client):
    r = client.post("/api/monitor", json={"strategy": "dual_momentum",
                                          "expected_sharpe": 1.0})
    assert r.status_code == 200, r.text[:400]
    assert "status" in r.json()


def test_logbook_api(client):
    r = client.post("/api/logbook", json={"hypothesis": "API 测试假设",
                                          "strategy": "test", "notes": "x"})
    assert r.status_code == 200
    rid = r.json()["id"]
    r2 = client.get("/api/logbook")
    assert r2.json()["count"] >= 1
    r3 = client.delete(f"/api/logbook/{rid}")
    assert r3.status_code == 200


def test_dataset_api(client):
    r = client.get("/api/dataset/info")
    assert r.status_code == 200
    assert r.json()["n_symbols"] == 50


def test_allocators_api(client):
    r = client.post("/api/analyze/allocators", json={"lookback": 252})
    assert r.status_code == 200
    assert len(r.json()["table"]) >= 5


def test_strategies_api(client):
    r = client.get("/api/strategies")
    assert r.status_code == 200
    assert "dual_momentum" in r.json()["strategies"]


# ------------------------------------------------------------------ 课程体系
def test_curriculum_loads():
    c = load_curriculum()
    assert len(c) >= 60
    st = c.stats()
    assert st["total_words"] > 80_000
    assert st["n_code_blocks"] > 50


def test_curriculum_lessons_have_required_fields():
    c = load_curriculum()
    for l in c:
        assert l.id and l.title and l.category
        assert l.word_count > 500, f"{l.id} 字数只有 {l.word_count}"
        assert len(l.objectives) >= 1, f"{l.id} 缺少学习目标"


def test_curriculum_search():
    c = load_curriculum()
    hits = c.search("夏普")
    assert len(hits) > 0
    assert all(isinstance(h["id"], str) for h in hits)


def test_learning_path_respects_prereq():
    c = load_curriculum()
    path = c.learning_path()
    ids = [p["id"] for p in path]
    assert len(ids) == len(c)
    assert len(set(ids)) == len(ids), "学习路径有重复"
    pos = {i: k for k, i in enumerate(ids)}
    for p in path:
        for pre in p["prereq"]:
            if pre in pos:
                assert pos[pre] < pos[p["id"]], f"{p['id']} 排在了先修 {pre} 之前"


def test_api_index():
    c = load_curriculum()
    idx = c.api_index()
    assert len(idx) > 5
    assert all(isinstance(v, list) for v in idx.values())


# ------------------------------------------------------------------ 风控闸门
def test_risk_gate_blocks():
    gate = RiskGate(max_position_weight=0.1)
    intent = OrderIntent(symbol="600000", target_weight=0.5, price=10.0)
    res = gate.check(intent)
    assert res.allowed
    assert res.adjusted_weight <= 0.1 + 1e-9
    assert any("上限" in w for w in res.warnings)


def test_risk_gate_drawdown_halt():
    gate = RiskGate(max_drawdown_halt=0.2)
    intent = OrderIntent(symbol="600000", target_weight=0.05)
    res = gate.check(intent, current_drawdown=-0.25)
    assert not res.allowed


def test_risk_gate_suspended_and_limit():
    gate = RiskGate()
    intent = OrderIntent(symbol="600000", target_weight=0.05)
    assert not gate.check(intent, suspended=True).allowed
    assert not gate.check(intent, limit_up=True).allowed


def test_risk_gate_capacity():
    gate = RiskGate(max_daily_adv_share=0.05)
    intent = OrderIntent(symbol="600000", target_weight=0.5, adv=1_000_000)
    res = gate.check(intent, portfolio_value=10_000_000)
    assert res.allowed
    assert any("容量" in w for w in res.warnings)


def test_risk_gate_de_leverage():
    gate = RiskGate(max_consecutive_loss_days=5, de_leverage_factor=0.5)
    intent = OrderIntent(symbol="600000", target_weight=0.08)
    res = gate.check(intent, consecutive_loss_days=6)
    assert abs(res.adjusted_weight - 0.04) < 1e-9


def test_risk_gate_portfolio():
    gate = RiskGate(max_position_weight=0.2, max_gross=1.0)
    w = pd.Series({"A": 0.6, "B": 0.6})
    res = gate.check_portfolio(w)
    assert res.allowed
    assert res.adjusted_weight <= 1.0 + 1e-9


# ------------------------------------------------------------------ 监控
def test_monitor_green_on_good_equity():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.008, 800))
    eq = (1 + r).cumprod()
    mon = Monitor(expected_sharpe=1.0)
    rep = mon.run(eq)
    assert rep.status.startswith(("绿色", "黄色")), f"健康曲线被判为 {rep.status}"


def test_monitor_red_on_negative_sharpe():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(-0.002, 0.01, 300))
    eq = (1 + r).cumprod()
    mon = Monitor(expected_sharpe=1.0)
    rep = mon.run(eq)
    assert "为负" in " ".join(rep.alerts) or rep.status.startswith("红色")


def test_monitor_detects_ic_decay():
    rng = np.random.default_rng(2)
    eq = pd.Series((1 + rng.normal(0.0005, 0.01, 600)).cumprod())
    # 前 300 天 IC 稳定为正，后 300 天衰减到 0
    ic = pd.concat([
        pd.Series(rng.normal(0.05, 0.01, 300)),
        pd.Series(rng.normal(0.0, 0.01, 300)),
    ]).reset_index(drop=True)
    mon = Monitor(expected_sharpe=1.0)
    rep = mon.run(eq, ic_series=ic)
    assert "ic_recent_tstat" in rep.data
    assert rep.data["ic_recent_mean"] < rep.data["ic_mean"]


def test_monitor_insufficient_data():
    mon = Monitor()
    rep = mon.run(pd.Series([1.0, 1.01, 1.02, 1.03]))
    assert rep.status == "数据不足"


# ------------------------------------------------------------------ 纸面交易
def test_paper_trader_runs(md_small):
    from qd.strategy.builtin import dual_momentum
    from qd.backtest.config import BacktestConfig

    pt = PaperTrader(md_small, dual_momentum(lookback=60, top_k=5), BacktestConfig())
    out = pt.run()
    assert len(out["equity"]) == md_small.n_dates
    assert abs(out["equity"].iloc[0] - 1.0) < 1e-9
    assert np.isfinite(out["final_value"])


def test_paper_matches_backtest(md_small):
    """纸面交易与回测引擎应给出一致结果（同一套协议）。"""
    from qd.backtest import Backtest
    from qd.strategy.builtin import dual_momentum
    from qd.backtest.config import BacktestConfig

    cfg = BacktestConfig()
    bt = Backtest(md_small, dual_momentum(lookback=60, top_k=5), cfg).run()
    pt = PaperTrader(md_small, dual_momentum(lookback=60, top_k=5), cfg).run()
    np.testing.assert_allclose(
        bt.equity.to_numpy(), pt["equity"].to_numpy(), rtol=1e-9,
        err_msg="纸面交易与回测不一致，说明两条链路的撮合协议不同",
    )
