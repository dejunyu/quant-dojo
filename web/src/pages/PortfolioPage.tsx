import { useEffect, useState } from 'react';
import { api } from '../api';
import { Chart } from '../components/Chart';

interface Row {
  method: string;
  expected_annual_return: number;
  annual_vol: number;
  sharpe: number;
  max_weight: number;
  effective_n: number;
  risk_concentration: number;
  [key: string]: unknown;
}

const CN: Record<string, string> = {
  equal_weight: '等权',
  inverse_vol: '逆波动率',
  risk_parity: '风险平价',
  hrp: '层次风险平价',
  min_variance: '最小方差',
  max_diversification: '最大分散化',
  mean_variance: '均值方差',
};

const NOTE: Record<string, string> = {
  equal_weight: '没有估计误差。DeMiguel 等人的研究显示它经常打赢复杂模型，永远拿它当基准。',
  inverse_vol: '风险平价的一阶近似，只用波动率，稳健。',
  risk_parity: '每个资产对组合风险的贡献相等，不依赖预期收益估计。',
  hrp: '先聚类再分配，不需要矩阵求逆，对协方差估计误差最不敏感。',
  min_variance: '只用协方差，不用预期收益，因此避开了均值方差最大的弱点。',
  max_diversification: '最大化分散化比率，介于风险平价与最小方差之间。',
  mean_variance: '教科书最优，但预期收益的估计误差会被优化器放大，实践中最脆弱。',
};

export function PortfolioPage() {
  const [rows, setRows] = useState<Row[]>([]);
  const [lookback, setLookback] = useState(252);
  const [busy, setBusy] = useState(false);
  const [equity, setEquity] = useState<{ date: string; value: number }[]>([]);

  const run = () => {
    setBusy(true);
    api
      .post<{ table: Row[] }>('/analyze/allocators', { lookback })
      .then((d) => setRows(d.table))
      .catch(() => setRows([]))
      .finally(() => setBusy(false));
  };

  useEffect(() => {
    run();
    api
      .post<{ equity: { date: string; value: number }[] }>('/backtest', { strategy: 'risk_parity', params: { lookback: 120 } })
      .then((d) => setEquity(d.equity))
      .catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="main-wide">
      <h1 className="page-title">组合构建与资产配置</h1>
      <p className="page-desc">
        七种配置方法的并排对比。注意：这里比的是<strong>权重特征</strong>，
        不是样本外收益——要比较真实表现请用滚动前推。
      </p>

      <div className="panel">
        <div className="btn-row">
          <label className="field">
            估计窗口（交易日）
            <input type="number" value={lookback} onChange={(e) => setLookback(Number(e.target.value))} style={{ width: 110 }} />
          </label>
          <button className="btn btn-primary" onClick={run} disabled={busy} style={{ alignSelf: 'flex-end' }}>
            {busy ? <span className="spinner" /> : null}
            重新计算
          </button>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">方法对比</div>
        <table className="data">
          <thead>
            <tr>
              <th>方法</th>
              <th>预期年化</th>
              <th>年化波动</th>
              <th>夏普</th>
              <th>最大权重</th>
              <th>有效标的数</th>
              <th>风险集中度</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.method}>
                <td className="txt">
                  {CN[r.method] ?? r.method}
                  <span className="tag" style={{ marginLeft: 6 }}>
                    {r.method}
                  </span>
                </td>
                <td className={r.expected_annual_return > 0 ? 'pos' : 'neg'}>
                  {(r.expected_annual_return * 100).toFixed(2)}%
                </td>
                <td>{(r.annual_vol * 100).toFixed(2)}%</td>
                <td className={r.sharpe > 0.5 ? 'pos' : 'neutral'}>{r.sharpe?.toFixed(3)}</td>
                <td>{(r.max_weight * 100).toFixed(2)}%</td>
                <td>{r.effective_n?.toFixed(2)}</td>
                <td>{r.risk_concentration?.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ fontSize: 11.5, color: 'var(--text-faint)', marginTop: 8 }}>
          有效标的数 = 1/Σw²，越接近 N 说明越分散；风险集中度越低说明各资产的风险贡献越均衡。
        </div>
      </div>

      <div className="grid grid-2">
        {rows.slice(0, 6).map((r) => (
          <div className="panel" key={r.method} style={{ marginBottom: 0 }}>
            <div className="panel-title">{CN[r.method] ?? r.method}</div>
            <div style={{ fontSize: 12.5, color: 'var(--text-dim)' }}>
              {NOTE[r.method] ?? '—'}
            </div>
          </div>
        ))}
      </div>

      {equity.length > 0 && (
        <div className="panel">
          <div className="panel-title">风险平价策略净值（示例回测）</div>
          <Chart
            series={[{ name: 'risk_parity', data: equity.map((p) => ({ time: p.date, value: p.value })), area: true, color: '#a371f7' }]}
            height={280}
          />
        </div>
      )}
    </div>
  );
}
