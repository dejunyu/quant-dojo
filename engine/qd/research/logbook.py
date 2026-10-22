"""研究日志（Logbook）.

量化研究最大的敌人不是不会写代码，而是**记不住自己试过什么**。三个月后
回看一条漂亮的净值曲线，你已经分不清它是精心设计的策略，还是第 87 次参数
搜索的幸存者。

Logbook 把每一次实验（假设、参数、样本区间、结果、当时的想法）持久化到
SQLite，并支持按结果排序、按假设检索、以及自动统计"你已经试了多少次"
——这个数字正是 Deflated Sharpe Ratio 需要的 ``n_trials``。
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["Logbook"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    strategy TEXT,
    params TEXT,
    start TEXT,
    end TEXT,
    universe TEXT,
    n_symbols INTEGER,
    sharpe REAL,
    annual_return REAL,
    max_drawdown REAL,
    turnover REAL,
    n_trials INTEGER,
    deflated_sharpe REAL,
    metrics TEXT,
    notes TEXT,
    tags TEXT,
    verdict TEXT
);
CREATE INDEX IF NOT EXISTS idx_hyp ON experiments(created_at);
"""


class Logbook:
    """SQLite 研究日志.

    Parameters
    ----------
    path :
        数据库文件路径，默认 ``storage/db/logbook.db``。
    """

    def __init__(self, path: str | Path | None = None):
        if path is None:
            root = Path(__file__).resolve().parents[3]
            path = root / "storage" / "db" / "logbook.db"
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------ 写入
    def log(
        self,
        hypothesis: str,
        strategy: str = "",
        params: dict | None = None,
        start: str = "",
        end: str = "",
        universe: list[str] | None = None,
        result=None,
        metrics: dict | None = None,
        notes: str = "",
        tags: list[str] | None = None,
        verdict: str = "",
        n_trials: int | None = None,
    ) -> int:
        """记录一次实验，返回记录 id。

        Parameters
        ----------
        hypothesis :
            你这次想验证的**假设**（一句话）。没有假设的实验不值得记录。
        result :
            ``BacktestResult``；给定时自动提取指标。
        """
        m = dict(metrics or {})
        if result is not None:
            mm = result.metrics
            m = {
                "sharpe": mm.get("sharpe"),
                "annual_return": mm.get("annual_return"),
                "max_drawdown": mm.get("max_drawdown"),
                "deflated_sharpe": mm.get("deflated_sharpe"),
                "turnover": (mm.get("turnover") or {}).get("annual_turnover"),
                **m,
            }
            start = start or mm.get("start", "")
            end = end or mm.get("end", "")
        if n_trials is None:
            n_trials = int(m.get("n_trials") or 1)

        cur = self._conn.execute(
            """INSERT INTO experiments
               (created_at, hypothesis, strategy, params, start, end, universe, n_symbols,
                sharpe, annual_return, max_drawdown, turnover, n_trials, deflated_sharpe,
                metrics, notes, tags, verdict)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                hypothesis,
                strategy,
                json.dumps(params or {}, ensure_ascii=False),
                start,
                end,
                json.dumps(universe or [], ensure_ascii=False),
                len(universe) if universe else None,
                _f(m.get("sharpe")),
                _f(m.get("annual_return")),
                _f(m.get("max_drawdown")),
                _f(m.get("turnover")),
                int(n_trials),
                _f(m.get("deflated_sharpe")),
                json.dumps(m, ensure_ascii=False),
                notes,
                json.dumps(tags or [], ensure_ascii=False),
                verdict,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    # ------------------------------------------------------------ 读取
    def all(self) -> pd.DataFrame:
        df = pd.read_sql_query("SELECT * FROM experiments ORDER BY id DESC", self._conn)
        return df

    def search(self, keyword: str) -> pd.DataFrame:
        like = f"%{keyword}%"
        df = pd.read_sql_query(
            "SELECT * FROM experiments WHERE hypothesis LIKE ? OR notes LIKE ? OR strategy LIKE ? ORDER BY id DESC",
            self._conn,
            params=(like, like, like),
        )
        return df

    def best(self, n: int = 10, by: str = "sharpe") -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM experiments WHERE {by} IS NOT NULL ORDER BY {by} DESC LIMIT ?",
            self._conn,
            params=(int(n),),
        )
        return df

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])

    def trials_for(self, strategy: str) -> int:
        """同一策略家族已经试过多少次——这是 DSR 的 n_trials 来源。"""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE strategy LIKE ?",
            (f"%{strategy}%",),
        ).fetchone()
        return int(row[0]) if row else 0

    def delete(self, id_: int) -> None:
        self._conn.execute("DELETE FROM experiments WHERE id = ?", (int(id_),))
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM experiments")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"<Logbook {self.count()} 条实验 @ {self.path}>"


def _f(v) -> float | None:
    try:
        f = float(v)
        return f if f == f else None  # NaN -> None
    except (TypeError, ValueError):
        return None
