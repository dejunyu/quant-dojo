import { useEffect, useState } from 'react';
import { api, type Category, type CurriculumStats } from '../api';
import { useApp } from '../store';

export function HomePage() {
  const [stats, setStats] = useState<CurriculumStats | null>(null);
  const [cats, setCats] = useState<Category[]>([]);
  const [factors, setFactors] = useState<number>(0);
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const openLesson = useApp((s) => s.openLesson);

  useEffect(() => {
    api
      .get<{ stats: CurriculumStats; categories: Category[] }>('/curriculum')
      .then((d) => {
        setStats(d.stats);
        setCats(d.categories);
      })
      .catch(() => undefined);
    api
      .get<{ n_factors: number }>('/factors')
      .then((d) => setFactors(d.n_factors))
      .catch(() => undefined);
    api
      .get<Record<string, unknown>>('/health')
      .then(setHealth)
      .catch(() => undefined);
  }, []);

  return (
    <div className="main-wide">
      <h1 className="page-title">Quant Dojo · 量化投资学习与实战平台</h1>
      <p className="page-desc">
        一套把「学」和「做」放在同一个地方的系统：左边是体系化的中文课程，右边是可以直接
        跑的因子研究、回测、过拟合检验与风控工具。
      </p>

      {stats && (
        <div className="grid grid-4" style={{ marginBottom: 18 }}>
          <Stat label="课程篇数" value={stats.n_lessons} />
          <Stat label="总字数" value={stats.total_words.toLocaleString()} />
          <Stat label="阅读时长" value={`${stats.total_reading_minutes} 分钟`} />
          <Stat label="代码示例" value={stats.n_code_blocks} />
        </div>
      )}

      <div className="callout">
        <strong>先读这三篇，再动手。</strong>{' '}
        <a
          href="#"
          onClick={(e) => {
            e.preventDefault();
            openLesson('01-01');
          }}
        >
          01-01 量化投资全景图
        </a>{' '}
        →{' '}
        <a
          href="#"
          onClick={(e) => {
            e.preventDefault();
            openLesson('06-03');
          }}
        >
          06-03 前视偏差的十种来源
        </a>{' '}
        →{' '}
        <a
          href="#"
          onClick={(e) => {
            e.preventDefault();
            openLesson('06-05');
          }}
        >
          06-05 过拟合与多重检验
        </a>
        。这三篇决定了你后面做的所有研究是不是可信的。
      </div>

      <div className="grid grid-2">
        <div className="panel">
          <div className="panel-title">课程体系</div>
          {cats.map((c) => (
            <div
              key={c.id}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '6px 0',
                borderBottom: '1px solid var(--border-soft)',
                fontSize: 13,
              }}
            >
              <a
                href="#"
                onClick={(e) => {
                  e.preventDefault();
                  openLesson(c.lessons[0]?.id ?? '01-01');
                }}
              >
                {c.id}
              </a>
              <span style={{ color: 'var(--text-faint)', fontFamily: 'var(--mono)', fontSize: 12 }}>
                {c.n_lessons} 篇 · {(c.word_count / 1000).toFixed(1)}k 字
              </span>
            </div>
          ))}
        </div>

        <div className="panel">
          <div className="panel-title">平台能力</div>
          <table className="data">
            <tbody>
              <tr>
                <td className="txt">因子库</td>
                <td>{factors} 个因子（动量/反转/价值/质量/波动/流动性/情绪/技术）</td>
              </tr>
              <tr>
                <td className="txt">回测引擎</td>
                <td className="txt">T+1、涨跌停、成交量参与率、滑点、冲击成本、印花税佣金</td>
              </tr>
              <tr>
                <td className="txt">无前视保证</td>
                <td className="txt">信号收盘生成 → 次日开盘成交（有自动化测试验证）</td>
              </tr>
              <tr>
                <td className="txt">过拟合检验</td>
                <td className="txt">Deflated Sharpe、白噪声实验、滚动前推、净化 K 折</td>
              </tr>
              <tr>
                <td className="txt">组合优化</td>
                <td className="txt">均值方差、风险平价、HRP、凯利、协方差收缩</td>
              </tr>
              <tr>
                <td className="txt">数据</td>
                <td className="txt">合成数据（离线可用）+ akshare A 股真实数据（可选）</td>
              </tr>
            </tbody>
          </table>
          {health && (
            <div style={{ marginTop: 12, fontSize: 11.5, color: 'var(--text-faint)' }}>
              后端状态：{String(health.status)} · v{String(health.version)}
            </div>
          )}
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">怎么用这套系统学习</div>
        <ol style={{ margin: 0, paddingLeft: 20, fontSize: 13.5, color: 'var(--text-dim)' }}>
          <li>
            先按左侧「课程」顺序读 01–04 章，建立对收益、风险、数据陷阱的基本认识。
          </li>
          <li>
            读 05 章时同步打开「因子研究」，对同一个因子跑 IC 检验和分层回测，看看教科书
            说的和你在数据上看到的是不是一回事。
          </li>
          <li>
            读 06 章时去「过拟合实验室」做一次白噪声实验——这会彻底改变你对"好策略"的判断标准。
          </li>
          <li>
            在「回测」里跑内置策略，然后改参数、加成本、缩容量，看收益怎么被吃掉。
          </li>
          <li>
            用「研究日志」记录每一次实验。三个月后你会感谢自己记下了试过多少次。
          </li>
        </ol>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
    </div>
  );
}
