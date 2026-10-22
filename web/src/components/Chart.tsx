import { useEffect, useRef } from 'react';
import {
  createChart,
  type IChartApi,
  type UTCTimestamp,
} from 'lightweight-charts';

export interface Series {
  name: string;
  data: { time: string; value: number }[];
  color?: string;
  area?: boolean;
}

export interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

const COLORS = ['#4a9eff', '#3fb950', '#d29922', '#f85149', '#a371f7', '#39c5cf'];

function toTs(date: string): UTCTimestamp {
  return Math.floor(new Date(date + 'T00:00:00Z').getTime() / 1000) as UTCTimestamp;
}

interface Props {
  series?: Series[];
  candles?: Candle[];
  histogram?: { name: string; data: { time: string; value: number }[]; color?: string }[];
  baseline?: number;
  height?: number;
}

/**
 * 基于 lightweight-charts 的时间序列图。
 *
 * 每次数据变化时重建图表：这是 v4 里最可靠的"清空旧序列"方式（该版本没有
 * removeSeries 的公开批量接口），代价只是重建开销，对交互式页面无感知。
 */
export function Chart({ series = [], candles, histogram = [], baseline, height = 320 }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    const parent = ref.current;
    if (!parent) return;

    const chart = createChart(parent, {
      height,
      layout: {
        background: { color: 'transparent' },
        textColor: '#9aa7b4',
        fontFamily:
          "-apple-system, BlinkMacSystemFont, 'PingFang SC', 'Helvetica Neue', Arial, sans-serif",
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.04)' },
        horzLines: { color: 'rgba(255,255,255,0.05)' },
      },
      rightPriceScale: { borderColor: '#262d36' },
      timeScale: { borderColor: '#262d36' },
      crosshair: { mode: 0 },
      localization: { locale: 'zh-CN' },
      autoSize: true,
    });
    chartRef.current = chart;

    if (candles && candles.length) {
      const s = chart.addCandlestickSeries({
        upColor: '#3fb950',
        downColor: '#f85149',
        borderUpColor: '#3fb950',
        borderDownColor: '#f85149',
        wickUpColor: '#3fb950',
        wickDownColor: '#f85149',
      });
      s.setData(
        candles.map((c) => ({
          time: toTs(c.time),
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        })),
      );
    }

    series.forEach((sr, i) => {
      const color = sr.color ?? COLORS[i % COLORS.length];
      if (sr.area) {
        const s = chart.addAreaSeries({
          lineColor: color,
          topColor: color + '33',
          bottomColor: color + '05',
          lineWidth: 2,
        });
        s.setData(sr.data.map((d) => ({ time: toTs(d.time), value: d.value })));
      } else {
        const s = chart.addLineSeries({ color, lineWidth: 2 });
        s.setData(sr.data.map((d) => ({ time: toTs(d.time), value: d.value })));
      }
    });

    histogram.forEach((h, i) => {
      const s = chart.addHistogramSeries({ color: h.color ?? COLORS[(i + 2) % COLORS.length] });
      s.setData(h.data.map((d) => ({ time: toTs(d.time), value: d.value })));
    });

    if (baseline !== undefined) {
      const s = chart.addLineSeries({
        color: '#6e7b8a',
        lineWidth: 1,
        lineStyle: 2,
      });
      const times = (series[0]?.data ?? []).map((d) => d.time);
      if (times.length) {
        s.setData([
          { time: toTs(times[0]), value: baseline },
          { time: toTs(times[times.length - 1]), value: baseline },
        ]);
      }
    }

    chart.timeScale().fitContent();

    return () => {
      chart.remove();
      chartRef.current = null;
    };
  }, [series, candles, histogram, baseline, height]);

  if (!series.length && !candles?.length && !histogram.length) {
    return <div className="chart-empty">暂无数据</div>;
  }
  return <div ref={ref} className="chart" />;
}

/** 纯 SVG 柱状图（用于分组收益等不需要时间轴的场景）。 */
export function BarChart({
  data,
  height = 200,
  color = '#4a9eff',
}: {
  data: { label: string; value: number }[];
  height?: number;
  color?: string;
}) {
  if (!data.length) return <div className="chart-empty">暂无数据</div>;
  const max = Math.max(...data.map((d) => Math.abs(d.value)), 1e-9);
  const min = Math.min(...data.map((d) => Math.min(d.value, 0)), 0);
  const range = max - min || 1;
  const zeroY = ((max - 0) / range) * height;

  return (
    <svg width="100%" height={height + 34} style={{ overflow: 'visible' }}>
      <line x1="0" y1={zeroY} x2="100%" y2={zeroY} stroke="#262d36" strokeWidth="1" />
      {data.map((d, i) => {
        const w = 100 / data.length;
        const x = `${i * w + w * 0.15}%`;
        const barW = `${w * 0.7}%`;
        const h = (Math.abs(d.value) / range) * height;
        const y = d.value >= 0 ? zeroY - h : zeroY;
        return (
          <g key={d.label}>
            <rect
              x={x}
              y={y}
              width={barW}
              height={Math.max(h, 1)}
              fill={d.value >= 0 ? color : '#f85149'}
              rx="2"
            />
            <text
              x={`${i * w + w * 0.5}%`}
              y={height + 15}
              fill="#9aa7b4"
              fontSize="11"
              textAnchor="middle"
            >
              {d.label}
            </text>
            <text
              x={`${i * w + w * 0.5}%`}
              y={d.value >= 0 ? y - 5 : y + h + 13}
              fill="#e6edf3"
              fontSize="11"
              textAnchor="middle"
            >
              {(d.value * 100).toFixed(1)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}
