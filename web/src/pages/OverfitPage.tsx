import { useState } from 'react';
import { api } from '../api';

interface WnResp {
  data: string;
  n_trials: number;
  best_in_sample_sharpe: number;
  worst_in_sample_sharpe?: number;
  best_params: Record<string, unknown>;
  sharpe_mean: number;
  sharpe_std: number;
  sharpe_p95: number;
  search_lift?: number;
  deflated_sharpe_of_best: number;
  lesson: string;
}

interface MCResp {
  n_sims: number;
  horizon: number;
  final_p05: number;
  final_p50: number;
  final_p95: number;
  prob_loss: number;
  maxdd_median: number;
}

const GRIDS: Record<string, Record<string, number[]>> = {
  dual_momentum: { lookback: [20, 60, 120], top_k: [3, 5, 10] },
  sma_cross: { fast: [5, 10, 20], slow: [30, 60, 120] },
  bollinger_reversion: { window: [10, 20, 40], top_k: [5, 10] },
  risk_parity: { lookback: [30, 60, 120, 250] },
  mean_reversion_basket: { lookback: [3, 5, 10, 20], top_k: [3, 5, 10] },
};

export function OverfitPage() {
  const [strategy, setStrategy] = useState('dual_momentum');
  const [wn, setWn] = useState<WnResp | null>(null);
  const [mc, setMc] = useState<MCResp | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const runWn = () => {
    setBusy('wn');
    setErr(null);
    api
      .post<WnResp>('/overfit/white_noise', { strategy, param_grid: GRIDS[strategy], n_symbols: 30, seed: 0 })
      .then(setWn)
      .catch((e: Error) => setErr(e.message))
      .finally(() => setBusy(null));
  };

  const runMc = () => {
    setBusy('mc');
    setErr(null);
    api
      .post<MCResp>('/analyze/monte_carlo', { n_sims: 1000, horizon: 252 })
      .then(setMc)
      .catch((e: Error) => setErr(e.message))
      .finally(() => setBusy(null));
  };

  return (
    <div className="main-wide">
      <h1 className="page-title">过拟合实验室</h1>
      <p className="page-desc">
        这一页的目的不是给你更好的策略，而是<strong>让你对"好策略"的标准变严格</strong>。
      </p>

      <div className="callout danger">
        <strong>白噪声实验：</strong>在<strong>真 alpha = 0</strong> 的纯随机数据上，只靠搜索参数，
        就能得到很高的样本内夏普。亲手跑一次，你就再也不会相信"夏普 2.0 就是好策略"。
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
          <button className="btn btn-primary" onClick={runWn} disabled={busy !== null}>
            {busy === 'wn' ? <span className="spinner" /> : null}
            跑白噪声实验
          </button>
          <button className="btn" onClick={runMc} disabled={busy !== null}>
            {busy === 'mc' ? <span className="spinner" /> : null}
            蒙特卡洛路径模拟
          </button>
        </div>
      </div>

      {err && <div className="callout danger">失败：{err}</div>}

      {wn && (
        <div className="panel">
          <div className="panel-title">
            白噪声实验结果
            <span className="hint">{wn.data}</span>
          </div>
          <div className="grid grid-4">
            <div className="metric">
              <div className="label">搜索组数</div>
              <div className="value">{wn.n_trials}</div>
            </div>
            <div className="metric">
              <div className="label">夏普均值</div>
              <div className="value neutral">{wn.sharpe_mean.toFixed(2)}</div>
              <div className="sub">真 alpha = 0 的应有水平</div>
            </div>
            <div className="metric">
              <div className="label">搜索出的最优夏普</div>
              <div className="value warn">{wn.best_in_sample_sharpe.toFixed(2)}</div>
              <div className="sub">凭空多出 {wn.search_lift?.toFixed(2)}</div>
            </div>
            <div className="metric">
              <div className="label">搜索抬升幅度</div>
              <div className="value neg">+{wn.search_lift?.toFixed(2)}</div>
              <div className="sub">这就是纯粹的过拟合偏误</div>
            </div>
            <div className="metric">
              <div className="label">Deflated Sharpe</div>
              <div className="value pos">{wn.deflated_sharpe_of_best.toFixed(3)}</div>
              <div className="sub">校正后不显著</div>
            </div>
          </div>
          <div style={{ marginTop: 12, fontSize: 12, color: 'var(--text-faint)' }}>
            夏普标准差 {wn.sharpe_std.toFixed(3)} · P95 {wn.sharpe_p95.toFixed(2)} · 最差{' '}
            {wn.worst_in_sample_sharpe?.toFixed(2)} · 最优参数 {JSON.stringify(wn.best_params)}
          </div>
          <div className="callout" style={{ marginTop: 12, marginBottom: 0 }}>
            {wn.lesson}
          </div>
        </div>
      )}

      {mc && (
        <div className="panel">
          <div className="panel-title">
            蒙特卡洛路径模拟
            <span className="hint">
              {mc.n_sims} 次 · {mc.horizon} 个交易日
            </span>
          </div>
          <div className="grid grid-4">
            <div className="metric">
              <div className="label">5% 分位终值</div>
              <div className="value neg">{mc.final_p05.toFixed(3)}</div>
            </div>
            <div className="metric">
              <div className="label">中位终值</div>
              <div className="value">{mc.final_p50.toFixed(3)}</div>
            </div>
            <div className="metric">
              <div className="label">95% 分位终值</div>
              <div className="value pos">{mc.final_p95.toFixed(3)}</div>
            </div>
            <div className="metric">
              <div className="label">亏损概率</div>
              <div className={`value ${mc.prob_loss > 0.4 ? 'neg' : 'neutral'}`}>
                {(mc.prob_loss * 100).toFixed(1)}%
              </div>
            </div>
          </div>
          <div style={{ marginTop: 10, fontSize: 12, color: 'var(--text-faint)' }}>
            中位最大回撤 {(mc.maxdd_median * 100).toFixed(1)}%
          </div>
        </div>
      )}

      <div className="panel">
        <div className="panel-title">上线前的自检清单</div>
        <ol style={{ margin: 0, paddingLeft: 20, fontSize: 13.5, color: 'var(--text-dim)' }}>
          <li>参数数量是否远小于样本量？每加一个参数都需要更多数据支撑。</li>
          <li>是否用了滚动前推或净化 K 折，而不是随机切分？</li>
          <li>样本外表现是否与样本内同量级？（落差本身就是过拟合的度量）</li>
          <li>策略有没有经济学解释，还是纯粹的数据挖掘？</li>
          <li>换手率与成本是否完整计入？扣完成本还剩多少？</li>
          <li>你一共试了多少组参数？这个数字必须诚实地填进 Deflated Sharpe。</li>
        </ol>
      </div>
    </div>
  );
}
