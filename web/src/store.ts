import { create } from 'zustand';

export type PageId =
  | 'home'
  | 'lesson'
  | 'factors'
  | 'research'
  | 'backtest'
  | 'optimize'
  | 'overfit'
  | 'portfolio'
  | 'monitor'
  | 'logbook';

interface AppState {
  page: PageId;
  lessonId: string | null;
  dataset: string;
  setPage: (p: PageId) => void;
  openLesson: (id: string) => void;
}

export const useApp = create<AppState>((set) => ({
  page: 'home',
  lessonId: null,
  dataset: 'demo',
  setPage: (p) => set({ page: p }),
  openLesson: (id) => set({ page: 'lesson', lessonId: id }),
}));
