import { useMemo, useState } from 'react';

// Single-series training-loss line, rendered live from loss_history.json.
// Recessive grid, 2px line, rounded data-end, hover crosshair + tooltip.
const W = 640;
const H = 260;
const PAD = { top: 16, right: 16, bottom: 34, left: 44 };

export default function LossCurve({ loss }) {
  const [hover, setHover] = useState(null);

  const { pts, area, yTicks, plotW, plotH } = useMemo(() => {
    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;
    const maxLoss = Math.max(...loss);
    const minLoss = Math.min(...loss);
    const yMax = Math.ceil(maxLoss * 10) / 10;
    const yMin = Math.floor(minLoss * 10) / 10;

    const x = (i) => PAD.left + (i / (loss.length - 1)) * plotW;
    const y = (v) => PAD.top + (1 - (v - yMin) / (yMax - yMin || 1)) * plotH;

    const pts = loss.map((v, i) => ({ x: x(i), y: y(v), v, epoch: i + 1 }));
    const line = pts.map((p, i) => `${i ? 'L' : 'M'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
    const area = `${line} L${x(loss.length - 1).toFixed(1)},${PAD.top + plotH} L${PAD.left},${
      PAD.top + plotH
    } Z`;

    const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => {
      const v = yMin + t * (yMax - yMin);
      return { v, y: y(v) };
    });

    return { pts, line, area, yTicks, plotW, plotH };
  }, [loss]);

  const handleMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let nearest = pts[0];
    for (const p of pts) if (Math.abs(p.x - px) < Math.abs(nearest.x - px)) nearest = p;
    setHover(nearest);
  };

  return (
    <div className="chart">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="chart-svg"
        onMouseMove={handleMove}
        onMouseLeave={() => setHover(null)}
        role="img"
        aria-label="Training loss over epochs"
      >
        {/* grid + y labels */}
        {yTicks.map((t, i) => (
          <g key={i}>
            <line x1={PAD.left} x2={W - PAD.right} y1={t.y} y2={t.y} className="chart-grid" />
            <text x={PAD.left - 8} y={t.y + 3} className="chart-tick" textAnchor="end">
              {t.v.toFixed(2)}
            </text>
          </g>
        ))}

        {/* x labels (epoch) */}
        {[0, Math.floor(pts.length / 2), pts.length - 1].map((i) => (
          <text key={i} x={pts[i].x} y={H - 12} className="chart-tick" textAnchor="middle">
            {pts[i].epoch}
          </text>
        ))}
        <text x={PAD.left + plotW / 2} y={H - 0.5} className="chart-axis-title" textAnchor="middle">
          Epoch
        </text>

        <path d={area} className="chart-area" />
        <path d={pts.map((p, i) => `${i ? 'L' : 'M'}${p.x},${p.y}`).join(' ')} className="chart-line" />

        {/* rounded final data-end */}
        <circle cx={pts[pts.length - 1].x} cy={pts[pts.length - 1].y} r="3.5" className="chart-end" />

        {hover && (
          <g>
            <line
              x1={hover.x}
              x2={hover.x}
              y1={PAD.top}
              y2={PAD.top + plotH}
              className="chart-crosshair"
            />
            <circle cx={hover.x} cy={hover.y} r="4.5" className="chart-hoverdot" />
          </g>
        )}
      </svg>

      {hover && (
        <div className="chart-tip">
          Epoch {hover.epoch} · loss <strong>{hover.v.toFixed(3)}</strong>
        </div>
      )}
    </div>
  );
}
