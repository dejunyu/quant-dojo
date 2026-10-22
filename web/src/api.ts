// 开发时由 Vite 代理到 FastAPI（127.0.0.1:8000）；生产环境同域 /api
const BASE = '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = body?.detail ?? body?.error ?? detail;
    } catch {
      /* 忽略解析失败 */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body: unknown) =>
    request<T>(path, { method: 'POST', body: JSON.stringify(body) }),
  del: <T>(path: string) => request<T>(path, { method: 'DELETE' }),
};

// ------------------------------------------------------------------ 类型
export interface LessonMeta {
  id: string;
  title: string;
  category: string;
  level: string;
  order: number;
  prereq: string[];
  tags: string[];
  objectives: string[];
  key_points: string[];
  exercises: string[];
  api: string[];
  word_count: number;
  reading_minutes: number;
  path: string;
  body?: string;
}

export interface Category {
  id: string;
  title: string;
  n_lessons: number;
  word_count: number;
  lessons: LessonMeta[];
}

export interface CurriculumStats {
  n_lessons: number;
  n_categories: number;
  total_words: number;
  total_reading_minutes: number;
  levels: Record<string, number>;
  n_code_blocks: number;
}

export interface FactorInfo {
  name: string;
  category: string;
  direction: number;
  description: string;
  formula: string;
  params: Record<string, unknown>;
  required_fields: string[];
}

export interface EquityPoint {
  date: string;
  value: number;
}

export interface BacktestMetrics {
  start: string;
  end: string;
  years: number;
  cumulative_return: number;
  annual_return: number;
  annual_vol: number;
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  calmar: number;
  win_rate: number;
  var_95: number;
  probabilistic_sharpe: number;
  deflated_sharpe: number;
  n_trials: number;
  [key: string]: unknown;
}

export interface BacktestResponse {
  strategy: string;
  params: Record<string, unknown>;
  metrics: BacktestMetrics;
  trade_stats: Record<string, unknown>;
  cost_breakdown: Record<string, number>;
  equity: EquityPoint[];
  benchmark: EquityPoint[] | null;
  weights_top: { dates: string[]; series: Record<string, number[]> };
  daily: { gross: number[]; n_positions: number[] };
  fills_sample: Record<string, unknown>[];
  trades_sample: Record<string, unknown>[];
  monthly: Record<string, Record<string, number>>;
}
