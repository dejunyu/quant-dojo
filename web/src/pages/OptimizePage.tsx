import { useState } from 'react';
import { api } from '../api';
import { Chart } from '../components/Chart';

interface GridRow {
  rank?: number;
  sharpe?: number;
  annual_return?: number;
  max_drawdown?: number;
  n_trades?: number;
  [key: string]: unknown;
}

interface GridResp {
  table: GridRow[];
  best_params: Record<string, unknown>;
  n_trials: number;
  deflated_sharpe: number | null;
  warning: string;
  elapsed_sec: number;
}

interface WfResp {
  folds: Record<string, unknown>[];
  sharpe: number;
  annual_return: number;
  max_drawdown: number;
  n_folds: number;
  equity: { date: string; value: number }[];
  note: string;
}

const GRIDS: Record<string, Record<string, number[]>> = {
  dual_momentum: { lookback: [60, 120, 250], top_k: [5, 10, 20] },
  sma_cross: { fast: [5, 10, 20], slow: [30, 60, 120] },
  bollinger_reversion: { window: [10, 20, 40], top_k: [5, 10] },
  mean_reversion_basket: { lookback: [3, 5, 10], top_k: [5, 10] },
  risk_parity: { lookback: [60, 120, 250] },
  turtle_breakout: { entry_window: [20, 55, 90], n: [5, 10] },
  rotation: { lookback: [20, 60, 120], top_k: [2, 3, 5] },
};

export function OptimizePage() {
  const [strategy, setStrategy] = useState('dual_momentum');
  const [grid, setGrid] = useState<GridResp | null>(null);
  const [wf, setWf] = useState<WfResp | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const runGrid = () => {
    setBusy('grid');
    setErr(null);
    api
      .post<GridResp>('/optimize/grid', { strategy, param_grid: GRIDS[strategy] })
      .then(setGrid)
      .catch((e: Error) => setErr(e.message))
      .finally(() => setBusy(null));
  };

  const runWf = () => {
    setBusy('wf');
    setErr(null);
    api
      .post<WfResp>('/optimize/walk_forward', { strategy, param_grid: GRIDS[strategy], n_splits: 5 })
      .then(setWf)
      .catch((e: Error) => setErr(e.message))
      .finally(() => setBusy(null));
  };

  const cols = grid?.table?.length ? Object.keys(grid.table[0]).filter((c) => c !== 'rank') : [];

  return (
    <div className="main-wide">
      <h1 className="page-title">参数优化与样本外验证</h1>
      <div className="callout warn">
        <strong>读这个数字之前，先读这句话。</strong> 在金融数据上搜索 N 组参数，样本内最优结果的
        期望偏误会随 N 增大而增大。所以本页每次搜索都会给出 Deflated Sharpe——它把"你试了多少次"
        直接计入结论。滚动前推那条曲线才是你能对外说的数字。
      </div>

      <div className="panel">
        <div className="btn-row">
          <label className="field">
            策略
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)} style={{ minWidth: 190 }}>
              {Object.keys(GRIDS).map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <button className="btn btn-primary" onClick={runGrid} disabled={busy !== null}>
            {busy === 'grid' ? <span className="spinner" /> : null}
            网格搜索
          </button>
          <button className="btn btn-primary" onClick={runWf} disabled={busy !== null}>
            {busy === 'wf' ? <span className="spinner" /> : null}
            滚动前推
          </button>
          <span style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>
            参数网格：{JSON.stringify(GRIDS[strategy])}
          </span>
        </div>
      </div>

      {err && <div className="callout danger">失败：{err}</div>}

      {grid && (
        <div className="panel">
          <div className="panel-title">
            网格搜索结果
            <span className="hint">
              {grid.n_trials} 组参数 · 耗时 {grid.elapsed_sec}s
            </span>
          </div>
          <div className="grid grid-3" style={{ marginBottom: 12 }}>
            <div className="metric">
              <div className="label">最优 Sharpe</div>
              <div className="value">{Number(grid.table[0]?.sharpe ?? 0).toFixed(3)}</div>
            </div>
            <div className="metric">
              <div className="label">Deflated Sharpe</div>
              <div className={`value ${(grid.deflated_sharpe ?? 0) > 0.95 ? 'pos' : 'neg'}`}>
                {grid.deflated_sharpe?.toFixed(3) ?? '—'}
              </div>
            </div>
            <div className="metric">
              <div className="label">最优参数</div>
              <div className="value" style={{ fontSize: 13 }}>
                {JSON.stringify(grid.best_params)}
              </div>
            </div>
          </div>
          <table className="data">
            <thead>
              <tr>
                <th>#</th>
                {cols.map((c) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {grid.table.slice(0, 15).map((r, i) => (
                <tr key={i}>
                  <td>{r.rank ?? i + 1}</td>
                  {cols.map((c) => {
                    const v = r[c];
                    let cls = '';
                    if (c === 'sharpe' || c === 'annual_return' || c === 'calmar') {
                      cls = Number(v) > 0 ? 'pos' : 'neg';
                    }
                    return (
                      <td key={c} className={cls}>
                        {typeof v === 'number' ? v.toFixed(4) : String(v ?? '—')}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="callout warn" style={{ marginTop: 12, marginBottom: 0 }}>
            {grid.warning}
          </div>
        </div>
      )}

      {wf && (
        <>
          <div className="panel">
            <div className="panel-title">
              滚动前推（样本外拼接）净值
              <span className="hint">{wf.n_folds} 段</span>
            </div>
            <Chart
              series={[{ name: '样本外', data: wf.equity.map((p) => ({ time: p.date, value: p.value })), area: true, color: '#3fb950' }]}
              height={320}
            />
            <div className="grid grid-3" style={{ marginTop: 12 }}>
              <div className="metric">
                <div className="label">样本外夏普</div>
                <div className={`value ${wf.sharpe > 0.5 ? 'pos' : 'neg'}`}>{wf.sharpe.toFixed(3)}</div>
              </div>
              <div className="metric">
                <div className="label">样本外年化</div>
                <div className={`value ${wf.annual_return > 0 ? 'pos' : 'neg'}`}>{(wf.annual_return * 100).toFixed(2)}%</div>
              </div>
              <div className="metric">
                <div className="label">最大回撤</div>
                <div className="value neg">{(wf.max_drawdown * 100).toFixed(2)}%</div>
              </div>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-faint)', marginTop: 8 }}>{wf.note}</div>
          </div>

          <div className="panel">
            <div className="panel-title">每一段的选参与表现</div>
            <table className="data">
              <thead>
                <tr>
                  <th>段</th>
                  <th>训练区间</th>
                  <th>测试区间</th>
                  <th>参数</th>
                  <th>Sharpe</th>
                  <th>年化</th>
                  <th>回撤</th>
                </tr>
              </thead>
              <tbody>
                {wf.folds.map((f, i) => (
                  <tr key={i}>
                    <td>{String(f.fold ?? i + 1)}</td>
                    <td>
                      {String(f.train_start).slice(0, 10)} ~ {String(f.train_end).slice(0, 10)}
                    </td>
                    <td>
                      {String(f.test_start).slice(0, 10)} ~ {String(f.test_end).slice(0, 10)}
                    </td>
                    <td className="txt">{JSON.stringify(f.params)}</td>
                    <td className={Number(f.sharpe) > 0 ? 'pos' : 'neg'}>{Number(f.sharpe).toFixed(3)}</td>
                    <td className={Number(f.annual_return) > 0 ? 'pos' : 'neg'}>
                      {(Number(f.annual_return) * 100).toFixed(2)}%
                    </td>
                    <td className="neg">{(Number(f.max_drawdown) * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
