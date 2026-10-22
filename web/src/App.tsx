import { useApp, type PageId } from './store';
import { HomePage } from './pages/HomePage';
import { LessonPage } from './pages/LessonPage';
import { FactorPage } from './pages/FactorPage';
import { ResearchPage } from './pages/ResearchPage';
import { BacktestPage } from './pages/BacktestPage';
import { OptimizePage } from './pages/OptimizePage';
import { OverfitPage } from './pages/OverfitPage';
import { PortfolioPage } from './pages/PortfolioPage';
import { MonitorPage } from './pages/MonitorPage';
import { LogbookPage } from './pages/LogbookPage';

interface NavItem {
  id: PageId;
  label: string;
  icon: string;
  group: string;
}

const NAV: NavItem[] = [
  { id: 'home', label: '总览', icon: '◎', group: '开始' },
  { id: 'lesson', label: '课程', icon: '📖', group: '开始' },
  { id: 'factors', label: '因子库', icon: '∑', group: '研究' },
  { id: 'research', label: '因子研究', icon: '🔬', group: '研究' },
  { id: 'backtest', label: '策略回测', icon: '📈', group: '实战' },
  { id: 'optimize', label: '参数与验证', icon: '⚙', group: '实战' },
  { id: 'overfit', label: '过拟合实验室', icon: '⚠', group: '实战' },
  { id: 'portfolio', label: '组合配置', icon: '⚖', group: '实战' },
  { id: 'monitor', label: '监控与风控', icon: '🛡', group: '实盘' },
  { id: 'logbook', label: '研究日志', icon: '📓', group: '实盘' },
];

export default function App() {
  const page = useApp((s) => s.page);
  const setPage = useApp((s) => s.setPage);

  const groups = Array.from(new Set(NAV.map((n) => n.group)));

  return (
    <div className="layout">
      <nav className="sidebar">
        <div className="sidebar-brand">
          <h1>Quant Dojo</h1>
          <div className="sub">量化学习 · 实战平台</div>
        </div>
        <div className="nav">
          {groups.map((g) => (
            <div className="nav-group" key={g}>
              <div className="nav-group-title">{g}</div>
              {NAV.filter((n) => n.group === g).map((n) => (
                <div
                  key={n.id}
                  className={`nav-item ${page === n.id ? 'active' : ''}`}
                  onClick={() => setPage(n.id)}
                >
                  <span className="icon">{n.icon}</span>
                  {n.label}
                </div>
              ))}
            </div>
          ))}
        </div>
        <div className="sidebar-footer">
          qd v0.1.0 · 数据为合成数据
          <br />
          仅供学习，不构成投资建议
        </div>
      </nav>

      <main className="main">
        {page === 'home' && <HomePage />}
        {page === 'lesson' && <LessonPage />}
        {page === 'factors' && <FactorPage />}
        {page === 'research' && <ResearchPage />}
        {page === 'backtest' && <BacktestPage />}
        {page === 'optimize' && <OptimizePage />}
        {page === 'overfit' && <OverfitPage />}
        {page === 'portfolio' && <PortfolioPage />}
        {page === 'monitor' && <MonitorPage />}
        {page === 'logbook' && <LogbookPage />}
      </main>
    </div>
  );
}
