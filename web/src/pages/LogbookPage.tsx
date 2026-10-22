import { useEffect, useState } from 'react';
import { api } from '../api';

interface Rec {
  id: number;
  created_at: string;
  hypothesis: string;
  strategy: string;
  params: string;
  sharpe: number | null;
  annual_return: number | null;
  max_drawdown: number | null;
  n_trials: number | null;
  notes: string;
  tags: string;
  verdict: string;
}

export function LogbookPage() {
  const [rows, setRows] = useState<Rec[]>([]);
  const [hyp, setHyp] = useState('');
  const [strategy, setStrategy] = useState('');
  const [notes, setNotes] = useState('');
  const [sharpe, setSharpe] = useState('');
  const [verdict, setVerdict] = useState('');
  const [busy, setBusy] = useState(false);

  const load = () =>
    api
      .get<{ records: Rec[] }>('/logbook')
      .then((d) => setRows(d.records))
      .catch(() => setRows([]));

  useEffect(() => {
    load();
  }, []);

  const add = () => {
    if (!hyp.trim()) return;
    setBusy(true);
    api
      .post('/logbook', {
        hypothesis: hyp,
        strategy,
        notes,
        verdict,
        metrics: sharpe === '' ? undefined : { sharpe: Number(sharpe) },
      })
      .then(() => {
        setHyp('');
        setStrategy('');
        setNotes('');
        setSharpe('');
        setVerdict('');
        return load();
      })
      .finally(() => setBusy(false));
  };

  const remove = (id: number) => {
    api.del(`/logbook/${id}`).then(load).catch(() => undefined);
  };

  return (
    <div className="main-wide">
      <h1 className="page-title">研究日志</h1>
      <div className="callout">
        <strong>为什么要记？</strong> 量化研究最大的敌人不是不会写代码，而是记不住自己试过什么。
        三个月后回看一条漂亮的净值曲线，你已经分不清它是精心设计的策略，还是第 87 次参数搜索的
        幸存者。这里记录的试验次数，正是 Deflated Sharpe 需要的那个数字。
      </div>

      <div className="panel">
        <div className="panel-title">记录一次实验</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
          <input
            type="text"
            placeholder="假设（一句话，例如：12-1 动量在剔除小市值后依然显著）"
            value={hyp}
            onChange={(e) => setHyp(e.target.value)}
            style={{ width: '100%' }}
          />
          <div className="btn-row">
            <input type="text" placeholder="策略名" value={strategy} onChange={(e) => setStrategy(e.target.value)} style={{ width: 170 }} />
            <input type="number" step="0.01" placeholder="夏普" value={sharpe} onChange={(e) => setSharpe(e.target.value)} style={{ width: 110 }} />
            <input type="text" placeholder="结论（通过/可疑/不通过）" value={verdict} onChange={(e) => setVerdict(e.target.value)} style={{ width: 200 }} />
            <button className="btn btn-primary" onClick={add} disabled={busy || !hyp.trim()}>
              {busy ? <span className="spinner" /> : null}
              记录
            </button>
          </div>
          <input type="text" placeholder="备注" value={notes} onChange={(e) => setNotes(e.target.value)} style={{ width: '100%' }} />
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">
          实验记录
          <span className="hint">{rows.length} 条</span>
        </div>
        {!rows.length ? (
          <div className="empty">还没有记录。开始你的第一次实验吧。</div>
        ) : (
          <table className="data">
            <thead>
              <tr>
                <th>#</th>
                <th>时间</th>
                <th>假设</th>
                <th>策略</th>
                <th>夏普</th>
                <th>结论</th>
                <th>备注</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id}>
                  <td>{r.id}</td>
                  <td>{String(r.created_at).slice(0, 16)}</td>
                  <td className="txt">{r.hypothesis}</td>
                  <td className="txt">{r.strategy}</td>
                  <td className={(r.sharpe ?? 0) > 0 ? 'pos' : 'neg'}>
                    {r.sharpe == null ? '—' : r.sharpe.toFixed(3)}
                  </td>
                  <td className="txt">
                    {r.verdict ? (
                      <span className={`tag ${verdictTag(r.verdict)}`}>{r.verdict}</span>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td className="txt">{r.notes}</td>
                  <td>
                    <button className="btn btn-sm" onClick={() => remove(r.id)}>
                      删除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function verdictTag(v: string) {
  if (v.includes('通过') && !v.includes('不')) return 'green';
  if (v.includes('可疑')) return 'amber';
  return 'red';
}
