import { useEffect, useState } from 'react';
import { api, type BacktestResponse } from '../api';
import { Chart } from '../components/Chart';

const PRESETS: Record<string, Record<string, number>> = {
  dual_momentum: { lookback: 120, top_k: 10 },
  sma_cross: { fast: 20, slow: 60 },
  turtle_breakout: { entry_window: 55, exit_window: 20, n: 10 },
  bollinger_reversion: { window: 20, n_std: 2.0, top_k: 10 },
  mean_reversion_basket: { lookback: 5, top_k: 10 },
  risk_parity: { lookback: 120 },
  rotation: { top_k: 3, lookback: 60 },
  vol_target: { target_vol: 0.15, lookback: 60 },
  equal_weight: {},
  buy_and_hold: {},
};

export function BacktestPage() {
  const [strategies, setStrategies] = useState<string[]>([]);
  const [strategy, setStrategy] = useState('dual_momentum');
  const [params, setParams] = useState<Record<string, string>>({ lookback: '120', top_k: '10' });
  const [commission, setCommission] = useState('0.0003');
  const [slippage, setSlippage] = useState('5');
  const [participation, setParticipation] = useState('0.1');
  const [stopLoss, setStopLoss] = useState('');
  const [rebalance, setRebalance] = useState('monthly');
  const [res, setRes] = useState<BacktestResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api
      .get<{ strategies: string[] }>('/strategies')
      .then((d) => setStrategies(d.strategies))
      .catch(() => undefined);
  }, []);

  const pickStrategy = (s: string) => {
    setStrategy(s);
    const p = PRESETS[s] ?? {};
    setParams(Object.fromEntries(Object.entries(p).map(([k, v]) => [k, String(v)])));
  };

  const run = () => {
    setBusy(true);
    setErr(null);
    const numeric = Object.fromEntries(
      Object.entries(params).map(([k, v]) => [k, v === '' ? 0 : Number(v)]),
    );
    api
      .post<BacktestResponse>('/backtest', {
        strategy,
        params: numeric,
        commission_rate: Number(commission),
        slippage_bps: Number(slippage),
        max_participation: Number(participation),
        stop_loss: stopLoss === '' ? null : Number(stopLoss),
        rebalance,
      })
      .then(setRes)
      .catch((e: Error) => setErr(e.message))
      .finally(() => setBusy(false));
  };

  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const m = res?.metrics;
  const pct = (v: number | undefined) => (v == null || !isFinite(v) ? '—' : `${(v * 100).toFixed(2)}%`);

  return (
    <div className="main-wide">
      <h1 className="page-title">策略回测</h1>
      <p className="page-desc">
        引擎默认按 A 股规则：T+1、涨跌停不可买、成交量参与率上限、佣金印花税滑点冲击成本。
        信号在收盘生成，次日开盘成交——没有前视。
      </p>

      <div className="panel">
        <div className="btn-row">
          <label className="field">
            策略
            <select value={strategy} onChange={(e) => pickStrategy(e.target.value)} style={{ minWidth: 190 }}>
              {strategies.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          {Object.keys(params).map((k) => (
            <label className="field" key={k}>
              {k}
              <input
                type="number"
                step="any"
                value={params[k]}
                onChange={(e) => setParams((p) => ({ ...p, [k]: e.target.value }))}
                style={{ width: 96 }}
              />
            </label>
          ))}
          <label className="field">
            佣金率
            <input type="number" step="0.00001" value={commission} onChange={(e) => setCommission(e.target.value)} style={{ width: 96 }} />
          </label>
          <label className="field">
            滑点(bp)
            <input type="number" step="1" value={slippage} onChange={(e) => setSlippage(e.target.value)} style={{ width: 80 }} />
          </label>
          <label className="field">
            参与率上限
            <input type="number" step="0.01" value={participation} onChange={(e) => setParticipation(e.target.value)} style={{ width: 90 }} />
          </label>
          <label className="field">
            调仓频率
            <select value={rebalance} onChange={(e) => setRebalance(e.target.value)}>
              {[
                ['daily', '每日'],
                ['weekly', '每周'],
                ['monthly', '每月'],
                ['quarterly', '每季'],
              ].map(([v, l]) => (
                <option key={v} value={v}>
                  {l}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            止损(空=不设)
            <input type="number" step="0.01" value={stopLoss} onChange={(e) => setStopLoss(e.target.value)} style={{ width: 100 }} />
          </label>
          <button className="btn btn-primary" onClick={run} disabled={busy} style={{ alignSelf: 'flex-end' }}>
            {busy ? <span className="spinner" /> : null}
            运行回测
          </button>
        </div>
        <div style={{ fontSize: 11.5, color: 'var(--text-faint)', marginTop: 8 }}>
          试一试：把调仓频率从「每日」改成「每月」，看同一策略的收益差多少——日频交易的
          成本会吃掉大部分 alpha，这是新手最常踩的坑之一。
        </div>
      </div>

      {err && <div className="callout danger">回测失败：{err}</div>}

      {res && m && (
        <>
          <div className="grid grid-4">
            <M label="累计收益" value={pct(m.cumulative_return)} good={m.cumulative_return > 0} />
            <M label="年化收益" value={pct(m.annual_return)} good={m.annual_return > 0} />
            <M label="年化波动" value={pct(m.annual_vol)} neutral />
            <M label="夏普比率" value={m.sharpe?.toFixed(3) ?? '—'} good={m.sharpe > 1} />
            <M label="索提诺" value={m.sortino?.toFixed(3) ?? '—'} good={m.sortino > 1} />
            <M label="最大回撤" value={pct(m.max_drawdown)} good={false} />
            <M label="Calmar" value={m.calmar?.toFixed(3) ?? '—'} good={m.calmar > 0.5} />
            <M label="Deflated Sharpe" value={m.deflated_sharpe?.toFixed(3) ?? '—'} good={m.deflated_sharpe > 0.95} />
          </div>

          <div className="panel" style={{ marginTop: 14 }}>
            <div className="panel-title">
              净值曲线
              <span className="hint">
                {m.start} ~ {m.end}（{m.years} 年）
              </span>
            </div>
            <Chart
              series={[
                { name: '策略', data: res.equity.map((p) => ({ time: p.date, value: p.value })), color: '#4a9eff', area: true },
                ...(res.benchmark
                  ? [{ name: '基准', data: res.benchmark.map((p) => ({ time: p.date, value: p.value })), color: '#9aa7b4' }]
                  : []),
              ]}
              height={340}
            />
          </div>

          <div className="grid grid-2">
            <div className="panel">
              <div className="panel-title">交易统计</div>
              <table className="data">
                <tbody>
                  {Object.entries(res.trade_stats).map(([k, v]) => (
                    <tr key={k}>
                      <td className="txt">{k}</td>
                      <td>{typeof v === 'number' ? v.toFixed(4) : String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="panel">
              <div className="panel-title">成本分解</div>
              <table className="data">
                <tbody>
                  {Object.entries(res.cost_breakdown).map(([k, v]) => (
                    <tr key={k}>
                      <td className="txt">{COST_LABELS[k] ?? k}</td>
                      <td>{typeof v === 'number' ? (Math.abs(v) > 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : v.toFixed(4)) : String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="panel">
            <div className="panel-title">月度收益</div>
            <table className="data">
              <thead>
                <tr>
                  <th>年份</th>
                  {Array.from({ length: 12 }, (_, i) => (
                    <th key={i}>{i + 1}M</th>
                  ))}
                  <th>全年</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(res.monthly).map(([year, row]) => (
                  <tr key={year}>
                    <td className="txt">{year}</td>
                    {Array.from({ length: 12 }, (_, i) => {
                      const v = row[`${i + 1}M`];
                      return (
                        <td key={i} className={v > 0 ? 'pos' : v < 0 ? 'neg' : 'neutral'}>
                          {v == null ? '—' : (v * 100).toFixed(1)}
                        </td>
                      );
                    })}
                    <td className={(row['全年'] ?? 0) > 0 ? 'pos' : 'neg'}>
                      {row['全年'] == null ? '—' : (row['全年'] * 100).toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="panel">
            <div className="panel-title">最近成交（后 20 笔）</div>
            <table className="data">
              <thead>
                <tr>
                  <th>日期</th>
                  <th>标的</th>
                  <th>方向</th>
                  <th>数量</th>
                  <th>成交价</th>
                  <th>参考价</th>
                  <th>佣金</th>
                  <th>滑点</th>
                  <th>原因</th>
                </tr>
              </thead>
              <tbody>
                {res.fills_sample.slice(-20).reverse().map((f, i) => (
                  <tr key={i}>
                    <td>{String(f.date).slice(0, 10)}</td>
                    <td>{String(f.symbol)}</td>
                    <td className={f.side === 'buy' ? 'pos' : 'neg'}>{f.side === 'buy' ? '买入' : '卖出'}</td>
                    <td>{Number(f.qty).toLocaleString()}</td>
                    <td>{Number(f.price).toFixed(3)}</td>
                    <td>{Number(f.ref_price).toFixed(3)}</td>
                    <td>{Number(f.commission).toFixed(2)}</td>
                    <td>{Number(f.slippage_cost).toFixed(2)}</td>
                    <td className="txt">{String(f.reason)}</td>
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

const COST_LABELS: Record<string, string> = {
  commission: '佣金',
  stamp_tax: '印花税',
  transfer_fee: '过户费',
  slippage_cost: '滑点成本',
  impact_cost: '冲击成本',
  total_cost: '总成本',
  cost_drag_annual: '年均成本(元)',
  n_fills: '成交笔数',
  avg_fill_cost_bps: '平均成本(bp)',
};

function M({ label, value, good, neutral }: { label: string; value: string; good?: boolean; neutral?: boolean }) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className={`value ${neutral ? 'neutral' : good ? 'pos' : 'neg'}`}>{value}</div>
    </div>
  );
}
