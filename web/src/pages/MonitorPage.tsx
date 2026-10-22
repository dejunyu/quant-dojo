import { useState } from 'react';
import { api } from '../api';

interface MonitorResp {
  status: string;
  alerts: string[];
  full_sharpe?: number;
  recent_sharpe?: number;
  expected_sharpe?: number;
  current_drawdown?: number;
  ic_mean?: number;
  ic_recent_mean?: number;
  ic_recent_tstat?: number;
  recommendation?: string;
  [key: string]: unknown;
}

interface Gate {
  max_position_weight: number;
  max_daily_adv_share: number;
  max_gross: number;
  max_net: number;
  min_cash_ratio: number;
  max_drawdown_halt: number;
  max_consecutive_loss_days: number;
}

export function MonitorPage() {
  const [strategy, setStrategy] = useState('dual_momentum');
  const [expected, setExpected] = useState('1.0');
  const [factor, setFactor] = useState('');
  const [rep, setRep] = useState<MonitorResp | null>(null);
  const [gate, setGate] = useState<Gate | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = () => {
    setBusy(true);
    setErr(null);
    api
      .post<MonitorResp>('/monitor', {
        strategy,
        expected_sharpe: Number(expected),
        ...(factor ? { factor } : {}),
      })
      .then(setRep)
      .catch((e: Error) => setErr(e.message))
      .finally(() => setBusy(false));
  };

  if (!gate) {
    api.get<Gate>('/risk_gate').then(setGate).catch(() => undefined);
  }

  const statusClass = (s: string) =>
    s.startsWith('红色') ? 'danger' : s.startsWith('橙色') ? 'warn' : s.startsWith('黄色') ? 'warn' : 'ok';

  return (
    <div className="main-wide">
      <h1 className="page-title">实盘监控与风控闸门</h1>
      <p className="page-desc">
        回测结束不是终点，是起点。实盘后每天要回答：策略还有效吗？是市场变了还是策略坏了？该下线了吗？
      </p>

      <div className="panel">
        <div className="btn-row">
          <label className="field">
            策略
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)} style={{ minWidth: 190 }}>
              {['dual_momentum', 'sma_cross', 'risk_parity', 'bollinger_reversion', 'equal_weight'].map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            预期夏普（回测值）
            <input type="number" step="0.1" value={expected} onChange={(e) => setExpected(e.target.value)} style={{ width: 100 }} />
          </label>
          <label className="field">
            监控因子（可选）
            <input type="text" value={factor} onChange={(e) => setFactor(e.target.value)} placeholder="如 mom_12_1" style={{ width: 150 }} />
          </label>
          <button className="btn btn-primary" onClick={run} disabled={busy} style={{ alignSelf: 'flex-end' }}>
            {busy ? <span className="spinner" /> : null}
            运行监控
          </button>
        </div>
      </div>

      {err && <div className="callout danger">失败：{err}</div>}

      {rep && (
        <>
          <div className={`callout ${statusClass(rep.status)}`}>
            <strong>{rep.status}</strong>
            {rep.recommendation ? <div style={{ marginTop: 4 }}>{rep.recommendation}</div> : null}
          </div>

          <div className="grid grid-4">
            <M label="全样本夏普" value={rep.full_sharpe?.toFixed(3) ?? '—'} />
            <M label="近期夏普" value={rep.recent_sharpe?.toFixed(3) ?? '—'} good={(rep.recent_sharpe ?? 0) > (rep.expected_sharpe ?? 0) * 0.5} />
            <M label="当前回撤" value={`${((rep.current_drawdown ?? 0) * 100).toFixed(2)}%`} good={(rep.current_drawdown ?? -1) > -0.2} />
            <M label="近期 IC" value={rep.ic_recent_mean?.toFixed(4) ?? '—'} good={(rep.ic_recent_mean ?? 0) > 0.02} />
          </div>

          {rep.alerts.length > 0 ? (
            <div className="panel" style={{ marginTop: 14 }}>
              <div className="panel-title">告警</div>
              <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13 }}>
                {rep.alerts.map((a, i) => (
                  <li key={i} style={{ marginBottom: 5 }}>
                    {a}
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <div className="callout ok" style={{ marginTop: 14 }}>
              没有触发任何告警。
            </div>
          )}
        </>
      )}

      {gate && (
        <div className="panel">
          <div className="panel-title">风控闸门参数（下单前最后一道检查）</div>
          <table className="data">
            <tbody>
              {Object.entries(gate).map(([k, v]) => (
                <tr key={k}>
                  <td className="txt">{GATE_LABELS[k] ?? k}</td>
                  <td>{typeof v === 'number' ? (Math.abs(v) < 1 ? v.toFixed(4) : v) : String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ fontSize: 11.5, color: 'var(--text-faint)', marginTop: 8 }}>
            每一条都对应一次真实的亏钱经历：单标的上限、容量约束、停牌涨跌停、连续亏损降杠杆、回撤熔断。
          </div>
        </div>
      )}
    </div>
  );
}

const GATE_LABELS: Record<string, string> = {
  max_position_weight: '单标的最大权重',
  max_daily_adv_share: '单日下单占日均成交额上限',
  max_gross: '总敞口上限',
  max_net: '净敞口上限',
  min_cash_ratio: '最低现金比例',
  max_drawdown_halt: '回撤熔断线',
  max_consecutive_loss_days: '连续亏损降杠杆阈值（天）',
};

function M({ label, value, good }: { label: string; value: string; good?: boolean }) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className={`value ${good === undefined ? 'neutral' : good ? 'pos' : 'neg'}`}>{value}</div>
    </div>
  );
}
