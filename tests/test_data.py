"""数据层测试."""
import numpy as np
import pandas as pd
import pytest

from qd.data.market import MarketData
from qd.data.synthetic import generate_demo_market, generate_random_walk_market
from qd.utils.calendar import trading_days, shift_date, is_trading_day


def test_generate_demo_market_shape(md_small):
    assert md_small.n_symbols == 20
    assert md_small.n_dates > 900
    for f in ("open", "high", "low", "close", "volume"):
        assert f in md_small
        assert md_small.field(f).shape == (md_small.n_dates, md_small.n_symbols)


def test_no_nan_close_and_positive_price(md_small):
    px = md_small.close()
    assert px.isna().sum().sum() == 0
    assert (px > 0).all().all()


def test_ohlc_consistency(md_small):
    """high >= max(open, close)，low <= min(open, close)。"""
    h, l, o, c = md_small.high(), md_small.low(), md_small.open(), md_small.close()
    assert (h >= np.maximum(o, c) - 1e-9).all().all()
    assert (l <= np.minimum(o, c) + 1e-9).all().all()


def test_fundamentals_present(md_small):
    for f in ("pe", "pb", "roe", "mktcap", "industry"):
        assert f in md_small
    cap = md_small.fundamental("mktcap")
    assert (cap > 0).all().all()


def test_slice_and_subset(md_small):
    sub = md_small.slice("2021-01-01", "2021-12-31")
    assert sub.n_dates < md_small.n_dates
    assert sub.n_symbols == md_small.n_symbols
    assert sub.start >= pd.Timestamp("2021-01-01")

    syms = md_small.symbols[:5]
    ss = md_small.subset(syms)
    assert ss.n_symbols == 5
    assert set(ss.symbols) == set(syms)


def test_resample(md_small):
    weekly = md_small.resample("W-FRI")
    assert weekly.n_dates < md_small.n_dates
    # 周线收盘价应等于该周最后一个交易日的收盘价
    assert np.isclose(weekly.close().iloc[-1], md_small.close().iloc[-1], atol=1e-6).any()


def test_parquet_roundtrip(md_small, tmp_path):
    p = md_small.to_parquet(tmp_path / "ds")
    back = MarketData.from_parquet(p)
    assert back.n_symbols == md_small.n_symbols
    assert back.n_dates == md_small.n_dates
    np.testing.assert_allclose(back.close().to_numpy(), md_small.close().to_numpy(), rtol=1e-9)


def test_sanity_report(md_small):
    rep = md_small.sanity_report()
    assert rep["n_symbols"] == 20
    assert 0.0 <= rep["close_nan_ratio"] <= 1.0
    assert "suspicious_daily_jumps" in rep


def test_random_walk_has_no_autocorrelation(rw):
    """随机游走数据的收益不应有可预测的自相关。"""
    r = rw.returns()
    ac = r.apply(lambda col: col.autocorr(1)).mean()
    assert abs(ac) < 0.15


def test_returns_definitions(md_small):
    simple = md_small.returns(kind="simple")
    log = md_small.returns(kind="log")
    px = md_small.close()
    manual = px / px.shift(1) - 1
    np.testing.assert_allclose(simple.iloc[1:].to_numpy(), manual.iloc[1:].to_numpy(), rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(log.iloc[1:].to_numpy(), np.log1p(manual.iloc[1:]).to_numpy(), rtol=1e-6, equal_nan=True)


def test_calendar():
    d = trading_days("2024-01-01", "2024-01-31", market="A")
    assert len(d) > 15
    assert all(x.weekday() < 5 for x in d)
    assert trading_days("2024-01-01", "2024-01-07", market="CRYPTO").equals(
        pd.date_range("2024-01-01", "2024-01-07", freq="D")
    )
    with pytest.raises(ValueError):
        trading_days("2024-02-01", "2024-01-01")


def test_market_data_rejects_bad_input():
    with pytest.raises(ValueError):
        MarketData.from_dataframes({})


def test_from_dataframes_aliases():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    frames = {
        "AAA": pd.DataFrame({
            "Open": np.linspace(10, 12, 30), "High": np.linspace(10.2, 12.2, 30),
            "Low": np.linspace(9.8, 11.8, 30), "Close": np.linspace(10, 12, 30),
            "Vol": np.full(30, 1000.0),
        }, index=idx),
    }
    md = MarketData.from_dataframes(frames)
    assert "volume" in md
    assert md.n_symbols == 1
