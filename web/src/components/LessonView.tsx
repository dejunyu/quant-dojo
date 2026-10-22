import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { api, type LessonMeta } from '../api';

/** 把正文里的 [[id|标题]] 内链转成可点击链接（渲染时拦截跳转）。 */
const WIKI_LINK = /\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g;

function withWikiLinks(body: string): string {
  return body.replace(WIKI_LINK, (_m, id: string, label?: string) => {
    const text = label ?? id;
    return `[${text}](#lesson:${id})`;
  });
}

export function LessonView({
  lessonId,
  onNavigate,
}: {
  lessonId: string;
  onNavigate: (id: string) => void;
}) {
  const [lesson, setLesson] = useState<LessonMeta | null>(null);
  const [all, setAll] = useState<LessonMeta[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // 课程目录（用于上一篇/下一篇导航）
  useEffect(() => {
    api
      .get<{ categories: { lessons: LessonMeta[] }[] }>('/curriculum')
      .then((d) => setAll(d.categories.flatMap((c) => c.lessons)))
      .catch(() => setAll([]));
  }, []);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api
      .get<LessonMeta>(`/curriculum/lesson/${encodeURIComponent(lessonId)}`)
      .then((d) => {
        setLesson(d);
        window.scrollTo({ top: 0 });
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [lessonId]);

  if (loading && !lesson) {
    return (
      <div className="empty">
        <span className="spinner" />
        加载课程…
      </div>
    );
  }
  if (error) return <div className="empty">加载失败：{error}</div>;
  if (!lesson) return <div className="empty">未找到课程 {lessonId}</div>;

  const idx = all.findIndex((l) => l.id === lesson.id);
  const prev = idx > 0 ? all[idx - 1] : null;
  const next = idx >= 0 && idx < all.length - 1 ? all[idx + 1] : null;

  return (
    <div className="lesson-body">
      <div className="lesson-meta">
        <span className="tag accent">{lesson.category}</span>
        <span className="tag">{lesson.level}</span>
        <span className="tag">{lesson.word_count} 字</span>
        <span className="tag">约 {lesson.reading_minutes} 分钟</span>
        {lesson.tags.map((t) => (
          <span key={t} className="tag">
            {t}
          </span>
        ))}
      </div>

      {lesson.objectives.length > 0 && (
        <div className="obj-box">
          <h4>学完这一篇，你能做到</h4>
          <ul>
            {lesson.objectives.map((o, i) => (
              <li key={i}>{o}</li>
            ))}
          </ul>
        </div>
      )}

      <article className="md">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[[rehypeKatex, { strict: false, throwOnError: false }]]}
          components={{
            a: ({ href, children }) => {
              const h = String(href ?? '');
              if (h.startsWith('#lesson:')) {
                const id = h.slice('#lesson:'.length);
                return (
                  <a
                    href={h}
                    onClick={(e) => {
                      e.preventDefault();
                      onNavigate(id);
                    }}
                  >
                    {children}
                  </a>
                );
              }
              return (
                <a href={h} target="_blank" rel="noreferrer">
                  {children}
                </a>
              );
            },
          }}
        >
          {withWikiLinks(lesson.body ?? '')}
        </ReactMarkdown>
      </article>

      {lesson.exercises.length > 0 && (
        <div className="obj-box" style={{ marginTop: 28 }}>
          <h4>动手练习</h4>
          <ul>
            {lesson.exercises.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      {lesson.api.length > 0 && (
        <div className="panel" style={{ marginTop: 20 }}>
          <div className="panel-title">本篇涉及的平台接口</div>
          <div className="btn-row">
            {lesson.api.map((a) => (
              <span key={a} className="tag accent">
                {a}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="nav-lesson">
        {prev ? (
          <button className="btn" onClick={() => onNavigate(prev.id)}>
            ← {prev.title}
          </button>
        ) : (
          <span />
        )}
        {next ? (
          <button className="btn" onClick={() => onNavigate(next.id)}>
            {next.title} →
          </button>
        ) : (
          <span />
        )}
      </div>
    </div>
  );
}
