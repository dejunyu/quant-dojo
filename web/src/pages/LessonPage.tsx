import { useEffect, useMemo, useState } from 'react';
import { api, type Category, type LessonMeta } from '../api';
import { LessonView } from '../components/LessonView';
import { useApp } from '../store';

export function LessonPage() {
  const [cats, setCats] = useState<Category[]>([]);
  const [q, setQ] = useState('');
  const [hits, setHits] = useState<LessonMeta[]>([]);
  const lessonId = useApp((s) => s.lessonId);
  const openLesson = useApp((s) => s.openLesson);

  useEffect(() => {
    api
      .get<{ categories: Category[] }>('/curriculum')
      .then((d) => setCats(d.categories))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!q.trim()) {
      setHits([]);
      return;
    }
    const t = setTimeout(() => {
      api
        .get<{ results: LessonMeta[] }>(`/curriculum/search?q=${encodeURIComponent(q)}`)
        .then((d) => setHits(d.results))
        .catch(() => setHits([]));
    }, 220);
    return () => clearTimeout(t);
  }, [q]);

  const first = useMemo(() => cats[0]?.lessons?.[0]?.id ?? null, [cats]);
  const active = lessonId ?? first;

  return (
    <div className="lesson-layout">
      <aside className="lesson-toc">
        <div className="search-box">
          <input
            type="text"
            placeholder="搜索课程…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
          {hits.length > 0 && (
            <div className="search-results">
              {hits.slice(0, 12).map((h) => (
                <div key={h.id} className="search-hit" onClick={() => openLesson(h.id)}>
                  {h.title}
                  <span className="cat">{h.category}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {cats.map((c) => (
          <div key={c.id}>
            <div className="toc-cat">
              {c.id} · {c.n_lessons} 篇
            </div>
            {c.lessons.map((l) => (
              <div
                key={l.id}
                className={`toc-item ${active === l.id ? 'active' : ''}`}
                onClick={() => openLesson(l.id)}
                title={l.title}
              >
                <span className="num">{l.id}</span> {l.title}
              </div>
            ))}
          </div>
        ))}
      </aside>

      {active ? (
        <LessonView lessonId={active} onNavigate={openLesson} />
      ) : (
        <div className="empty">加载课程目录…</div>
      )}
    </div>
  );
}
