import { useEffect, useState } from 'react';
import { api, type FactorInfo } from '../api';
import { Chart, BarChart } from '../components/Chart';

interface IcSummary {
  ic_mean: number;
  icir: number;
  t_stat: number;
  ic_positive_ratio: number;
  n_periods: number;
}

interface QB {
  groups: Record<string, { annual_return: number; sharpe: number }>;
  long_short: { annual_return: number; sharpe: number; max_drawdown: number };
  monotonicity: number;
}

interface Report {
  factor: string;
  category: string;
  direction: number;
  description: string;
  formula: string;
  coverage: { avg_coverage: number };
  ic: IcSummary;
  ic_decay: { horizon: number; ic: number; icir: number; t_stat: number }[];
  quantile_backtest: QB;
  turnover: { annual_turnover: number; avg_daily_turnover: number };
  verdict: string;
  reasons: string[];
}

interface IcResp {
  name: string;
  ic_series: { date: string; ic: number }[];
  summary: IcSummary;
  decay: { horizon: number; ic: number; icir: number; t_stat: number; pos_ratio?: number }[];
  turnover: { annual_turnover: number };
}

export function ResearchPage() {
  const [factors, setFactors] = useState<FactorInfo[]>([]);
  const [name, setName] = useState('mom_12_1');
  const [horizon, setHorizon] = useState(20);
  const [report, setReport] = useState<Report | null>(null);
  const [ic, setIc] = useState<IcResp | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.get<{ factors: FactorInfo[] }>('/factors').then((d) => setFactors(d.factors)).catch(() => undefined);
  }, []);

  const run = () => {
    setBusy(true);
    setErr(null);
    Promise.all([
      api.post<Report>('/research/factor', { name, horizon, horizons: [1, 5, 20, 60] }),
      api.post<IcResp>('/research/ic', { name, horizon }),
    ])
      .then(([r, i]) => {
        setReport(r);
        setIc(i);
      })
      .catch((e: Error) => setErr(e.message))
      .finally(() => setBusy(false));
  };

  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pct = (v: number) => (v == null ? '—' : `${(v * 100).toFixed(2)}%`);

  return (
    <div className="main-wide">
      <h1 className="page-title">因子研究实验室</h1>
      <p className="page-desc">
        单因子检验五关：覆盖率 → IC → ICIR/t 检验 → 分层回测 → 衰减与成本。
        五关全过才值得进组合。
      </p>

      <div className="panel">
        <div className="btn-row">
          <label className="field">
            因子
            <select value={name} onChange={(e) => setName(e.target.value)} style={{ minWidth: 220 }}>
              {factors.map((f) => (
                <option key={f.name} value={f.name}>
                  {f.name} ({f.category})
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            预测周期（交易日）
            <input
              type="number"
              value={horizon}
              min={1}
              max={250}
              onChange={(e) => setHorizon(Number(e.target.value))}
              style={{ width: 100 }}
            />
          </label>
          <button className="btn btn-primary" onClick={run} disabled={busy} style={{ alignSelf: 'flex-end' }}>
            {busy ? <span className="spinner" /> : null}
            运行检验
          </button>
        </div>
      </div>

      {err && <div className="callout danger">运行失败：{err}</div>}

      {report && (
        <>
          <div className="grid grid-4">
            <Metric label="IC 均值" value={report.ic.ic_mean.toFixed(4)} good={report.ic.ic_mean > 0.02} />
            <Metric label="ICIR" value={report.ic.icir.toFixed(3)} good={report.ic.icir > 0.3} />
            <Metric label="t 统计量" value={report.ic.t_stat.toFixed(2)} good={Math.abs(report.ic.t_stat) > 2} />
            <Metric label="覆盖率" value={pct(report.coverage.avg_coverage)} good={report.coverage.avg_coverage > 0.7} />
          </div>

          <div className="grid grid-2" style={{ marginTop: 14 }}>
            <div className="panel">
              <div className="panel-title">
                {report.factor}
                <span className="hint">
                  {report.category} · {report.direction > 0 ? '正向' : '反向'}
                </span>
              </div>
              <div style={{ fontSize: 13, color: 'var(--text-dim)', marginBottom: 12 }}>
                {report.description}
              </div>
              <div className="grid grid-2">
                <Metric label="多空年化" value={pct(report.quantile_backtest.long_short.annual_return)} good={report.quantile_backtest.long_short.annual_return > 0} />
                <Metric label="多空夏普" value={report.quantile_backtest.long_short.sharpe.toFixed(3)} good={report.quantile_backtest.long_short.sharpe > 0.5} />
                <Metric label="分组单调性" value={report.quantile_backtest.monotonicity.toFixed(2)} good={report.quantile_backtest.monotonicity > 0.6} />
                <Metric label="年化换手" value={`${report.turnover.annual_turnover.toFixed(1)}×`} good={report.turnover.annual_turnover < 10} />
              </div>
            </div>

            <div className="panel">
              <div className="panel-title">
                分层回测（5 组年化收益）
                <span className="hint">Q1 = 因子值最高组</span>
              </div>
              <BarChart
                data={Object.entries(report.quantile_backtest.groups).map(([k, v]) => ({
                  label: k,
                  value: v.annual_return,
                }))}
              />
            </div>
          </div>

          <div className="panel">
            <div className="panel-title">IC 衰减曲线</div>
            <table className="data">
              <thead>
                <tr>
                  <th>预测周期</th>
                  <th>IC</th>
                  <th>ICIR</th>
                  <th>t 值</th>
                  <th>IC&gt;0 占比</th>
                </tr>
              </thead>
              <tbody>
                {(ic?.decay ?? report.ic_decay).map((d) => (
                  <tr key={d.horizon}>
                    <td>{d.horizon} 日</td>
                    <td className={d.ic > 0 ? 'pos' : 'neg'}>{Number(d.ic).toFixed(4)}</td>
                    <td>{Number(d.icir).toFixed(3)}</td>
                    <td className={Math.abs(Number(d.t_stat)) > 2 ? 'pos' : 'neutral'}>
                      {Number(d.t_stat).toFixed(2)}
                    </td>
                    <td>{'pos_ratio' in d ? pct(Number(d.pos_ratio)) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ fontSize: 12, color: 'var(--text-faint)', marginTop: 8 }}>
              IC 衰减告诉你该多久调一次仓：选 IC 还显著、但换手又不至于吃掉收益的那个周期。
            </div>
          </div>

          {ic && (
            <div className="panel">
              <div className="panel-title">IC 时间序列</div>
              <Chart
                series={[
                  {
                    name: 'IC',
                    data: ic.ic_series.map((p) => ({ time: p.date, value: p.ic })),
                    area: true,
                    color: '#4a9eff',
                  },
                ]}
                baseline={0}
                height={260}
              />
            </div>
          )}

          <div className={`callout ${verdictClass(report.verdict)}`}>
            <strong>机器判定：{report.verdict}</strong>
            <ul style={{ margin: '6px 0 0', paddingLeft: 20 }}>
              {report.reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  );
}

function verdictClass(v: string) {
  if (v.startsWith('通过')) return 'ok';
  if (v.startsWith('可疑')) return 'warn';
  return 'danger';
}

function Metric({ label, value, good }: { label: string; value: string; good: boolean }) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className={`value ${good ? 'pos' : 'neg'}`}>{value}</div>
    </div>
  );
}
