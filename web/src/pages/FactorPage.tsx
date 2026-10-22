import { useEffect, useMemo, useState } from 'react';
import { api, type Category, type FactorInfo } from '../api';
import { useApp } from '../store';

export function FactorPage() {
  const [factors, setFactors] = useState<FactorInfo[]>([]);
  const [filter, setFilter] = useState('全部');
  const [q, setQ] = useState('');
  const openLesson = useApp((s) => s.openLesson);

  useEffect(() => {
    api.get<{ factors: FactorInfo[] }>('/factors').then((d) => setFactors(d.factors)).catch(() => undefined);
    api
      .get<{ categories: Category[] }>('/curriculum')
      .then(() => undefined)
      .catch(() => undefined);
  }, []);

  const cats = useMemo(() => ['全部', ...Array.from(new Set(factors.map((f) => f.category)))], [factors]);
  const shown = useMemo(
    () =>
      factors.filter(
        (f) =>
          (filter === '全部' || f.category === filter) &&
          (!q || f.name.toLowerCase().includes(q.toLowerCase()) || f.description.includes(q)),
      ),
    [factors, filter, q],
  );

  const grouped = useMemo(() => {
    const m = new Map<string, FactorInfo[]>();
    shown.forEach((f) => {
      const arr = m.get(f.category) ?? [];
      arr.push(f);
      m.set(f.category, arr);
    });
    return Array.from(m.entries());
  }, [shown]);

  return (
    <div className="main-wide">
      <h1 className="page-title">因子库</h1>
      <p className="page-desc">
        共 {factors.length} 个因子。点击任意因子可以跳到对应课程。因子方向 ↑ 表示"值越大越好"，
        ↓ 表示反向因子（值越大越差，做多前应该取负号）。
      </p>

      <div className="btn-row" style={{ marginBottom: 14 }}>
        {cats.map((c) => (
          <button
            key={c}
            className={`btn btn-sm ${filter === c ? 'btn-primary' : ''}`}
            onClick={() => setFilter(c)}
          >
            {c}
          </button>
        ))}
        <input
          type="text"
          placeholder="搜索因子名或说明…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          style={{ marginLeft: 'auto', minWidth: 200 }}
        />
      </div>

      {grouped.map(([cat, list]) => (
        <div className="panel" key={cat}>
          <div className="panel-title">
            {cat}
            <span className="hint">{list.length} 个</span>
          </div>
          <div className="grid grid-3">
            {list.map((f) => (
              <div
                key={f.name}
                className="factor-card"
                onClick={() => openLesson('05-02')}
                title="点击查看因子构造课程"
              >
                <div className="name">
                  <span className="dir">{f.direction > 0 ? '↑' : '↓'}</span>
                  {f.name}
                </div>
                <div className="desc">{f.description}</div>
              </div>
            ))}
          </div>
        </div>
      ))}

      {!shown.length && <div className="empty">没有匹配的因子</div>}
    </div>
  );
}
