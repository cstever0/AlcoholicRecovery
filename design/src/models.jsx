// Model comparison, confidence threshold tradeoff and calibration.
// Everything here is read from metrics.json, which export_data.py computes from the fitted models.
import METRICS from './data/metrics.json';
import { cohortFacts } from './model.js';
import { useWidth } from './ui.jsx';

const { useState } = React;

const f3 = (v) => v.toFixed(3);
const p1 = (v) => `${(v * 100).toFixed(1)}%`;
const COLUMNS = [
  ['roc_auc', 'ROC-AUC', f3, 'high'],
  ['pr_auc', 'PR-AUC', f3, 'high'],
  ['brier_score', 'Brier', f3, 'low'],
  ['accuracy', 'Accuracy', p1, 'high'],
  ['sensitivity', 'Sensitivity', p1, 'high'],
  ['specificity', 'Specificity', p1, 'high'],
];

function ComparisonTable() {
  const best = Object.fromEntries(
    COLUMNS.map(([key, , , dir]) => {
      const values = METRICS.table.map((row) => row[key]);
      return [key, dir === 'high' ? Math.max(...values) : Math.min(...values)];
    }),
  );
  return (
    <div className="table-scroll">
      <table className="compare">
        <thead>
          <tr>
            <th scope="col">Model</th>
            {COLUMNS.map(([key, label]) => (
              <th key={key} scope="col">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {METRICS.table.map((row) => (
            <tr key={row.key}>
              <th scope="row">
                {row.name}
                {row.key === 'logistic_regression' && <span className="in-use">In use</span>}
              </th>
              {COLUMNS.map(([key, , format]) => (
                <td key={key} className={row[key] === best[key] ? 'best' : ''}>{format(row[key])}</td>
              ))}
            </tr>
          ))}
          <tr className="pending">
            <th scope="row">Treatment plan model</th>
            <td colSpan={COLUMNS.length}>Not built yet. New models appear here as rows.</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

const CHART_H = 270;
const M = { top: 14, right: 18, bottom: 40, left: 46 };

function TradeoffChart({ threshold }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const data = METRICS.tradeoff.filter((d) => d.accuracy != null);
  const w = Math.max(width, 280);
  const x = (t) => M.left + ((t - 0.5) / 0.49) * (w - M.left - M.right);
  const y = (v) => M.top + ((1 - v) / 0.8) * (CHART_H - M.top - M.bottom);
  const line = (key) => data.map((d, i) => `${i ? 'L' : 'M'}${x(d.t).toFixed(1)} ${y(d[key]).toFixed(1)}`).join('');
  const current = data.find((d) => Math.round(d.t * 100) === threshold) ?? data[0];
  const shown = hover ?? current;

  const onMove = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const t = 0.5 + ((e.clientX - box.left - M.left) / (w - M.left - M.right)) * 0.49;
    setHover(data.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a)));
  };

  return (
    <div className="plot" ref={ref}>
      <div className="legend">
        <span><i className="key key-1" />Patients who get a plan</span>
        <span><i className="key key-2" />Accuracy on those patients</span>
      </div>
      {width > 0 && (
        <svg width={w} height={CHART_H} role="img" aria-label="Plan coverage falls and accuracy rises as the confidence threshold goes up" onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
          {[0.2, 0.4, 0.6, 0.8, 1].map((v) => (
            <g key={v}>
              <line className="grid" x1={M.left} x2={w - M.right} y1={y(v)} y2={y(v)} />
              <text className="tick" x={M.left - 8} y={y(v) + 4} textAnchor="end">{Math.round(v * 100)}%</text>
            </g>
          ))}
          {[0.5, 0.6, 0.7, 0.8, 0.9].map((t) => (
            <text key={t} className="tick" x={x(t)} y={CHART_H - M.bottom + 18} textAnchor="middle">{Math.round(t * 100)}%</text>
          ))}
          <text className="axis-title" x={(M.left + w - M.right) / 2} y={CHART_H - 4} textAnchor="middle">Confidence threshold</text>
          <line className="now" x1={x(current.t)} x2={x(current.t)} y1={M.top} y2={CHART_H - M.bottom} />
          {hover && <line className="cross" x1={x(hover.t)} x2={x(hover.t)} y1={M.top} y2={CHART_H - M.bottom} />}
          <path className="series series-1" d={line('coverage')} />
          <path className="series series-2" d={line('accuracy')} />
          <circle className="dot dot-1" cx={x(shown.t)} cy={y(shown.coverage)} r="5" />
          <circle className="dot dot-2" cx={x(shown.t)} cy={y(shown.accuracy)} r="5" />
        </svg>
      )}
      {hover && (
        <div className="tooltip" style={{ left: Math.min(x(hover.t) + 12, w - 190), top: 44 }}>
          <strong>{Math.round(hover.t * 100)}% threshold</strong>
          <span>{p1(hover.coverage)} get a plan</span>
          <span>{p1(hover.accuracy)} accurate</span>
        </div>
      )}
    </div>
  );
}

function CalibrationChart() {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const w = Math.max(width, 280);
  const h = 320;
  const x = (v) => M.left + v * (w - M.left - M.right);
  const y = (v) => M.top + (1 - v) * (h - M.top - M.bottom);
  const series = [
    ['current', 'series-1', 'dot-1', 'Current model'],
    ['unweighted', 'series-2', 'dot-2', 'Without class_weight="balanced"'],
  ];
  return (
    <div className="plot" ref={ref}>
      <div className="legend">
        <span><i className="key key-1" />{series[0][3]}</span>
        <span><i className="key key-2" />{series[1][3]}</span>
        <span><i className="key key-0" />Perfect calibration</span>
      </div>
      {width > 0 && (
        <svg width={w} height={h} role="img" aria-label="Predicted against observed recovery. The current model predicts too low between 10 and 70 percent.">
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <g key={v}>
              <line className="grid" x1={M.left} x2={w - M.right} y1={y(v)} y2={y(v)} />
              <text className="tick" x={M.left - 8} y={y(v) + 4} textAnchor="end">{v * 100}%</text>
              <text className="tick" x={x(v)} y={h - M.bottom + 18} textAnchor="middle">{v * 100}%</text>
            </g>
          ))}
          <text className="axis-title" x={(M.left + w - M.right) / 2} y={h - 4} textAnchor="middle">Predicted chance of recovery</text>
          <line className="ideal" x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} />
          {series.map(([key, cls, dot]) => (
            <g key={key}>
              <path className={`series ${cls}`} d={METRICS.calibration[key].map((d, i) => `${i ? 'L' : 'M'}${x(d.pred).toFixed(1)} ${y(d.obs).toFixed(1)}`).join('')} />
              {METRICS.calibration[key].map((d) => (
                <g key={d.pred} onMouseEnter={() => setHover({ ...d, name: series.find((s) => s[0] === key)[3] })} onMouseLeave={() => setHover(null)}>
                  <circle className="hit" cx={x(d.pred)} cy={y(d.obs)} r="13" />
                  <circle className={`dot ${dot}`} cx={x(d.pred)} cy={y(d.obs)} r="5" />
                </g>
              ))}
            </g>
          ))}
        </svg>
      )}
      {hover && (
        <div className="tooltip" style={{ left: Math.min(x(hover.pred) + 14, w - 220), top: Math.max(y(hover.obs) - 8, 40) }}>
          <strong>{hover.name}</strong>
          <span>Predicted {Math.round(hover.pred * 100)}%</span>
          <span>Observed {Math.round(hover.obs * 100)}%</span>
          <span>{hover.n} patients</span>
        </div>
      )}
    </div>
  );
}

const MRI_STATS = [
  ['3.86 points', 'Mean LVEF error against expert contours'],
  ['15 of 20', 'Test patients within 5 points'],
  ['0.839', 'Mean Dice overlap'],
];

export function ModelsView({ threshold, setThreshold }) {
  const facts = cohortFacts();
  const row = METRICS.tradeoff.find((d) => Math.round(d.t * 100) === threshold);
  const worst = METRICS.calibration.current.reduce((a, b) => (b.obs - b.pred > a.obs - a.pred ? b : a));
  return (
    <div className="models">
      <header className="models-head">
        <h1>Models</h1>
        <p>
          All scores come from synthetic patients, so they show the model learned the simulator, not that it works on real people. {facts.total.toLocaleString()} records, {facts.eligible.toLocaleString()} in the abstinent group, {METRICS.n_val} in the validation split.
        </p>
      </header>

      <section className="panel">
        <h2>Recovery prediction: validation split</h2>
        <ComparisonTable />
      </section>

      <div className="panel-pair">
        <section className="panel">
          <h2>How sure should the model be before it recommends?</h2>
          <label className="threshold" htmlFor="threshold-main">
            <span>Confidence threshold</span>
            <input id="threshold-main" type="range" min="50" max="99" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
            <output htmlFor="threshold-main">{threshold}%</output>
          </label>
          {row && row.accuracy != null && (
            <p className="takeaway">
              At {threshold}%, <strong>{Math.round(row.coverage * 100)}%</strong> of patients get a plan, and the recovery call is right for <strong>{p1(row.accuracy)}</strong> of them. The rest see "Not sure."
            </p>
          )}
          <TradeoffChart threshold={threshold} />
        </section>
        <section className="panel">
          <h2>Can the percentages be trusted?</h2>
          <p className="takeaway">
            Not in the middle yet. When the current model says <strong>{Math.round(worst.pred * 100)}%</strong>, <strong>{Math.round(worst.obs * 100)}%</strong> of those patients recover. Removing one training setting fixes most of it.
          </p>
          <CalibrationChart />
        </section>
      </div>

      <section className="panel">
        <h2>MRI measurement: U-Net on real ACDC scans</h2>
        <dl className="stat-row">
          {MRI_STATS.map(([value, label]) => (
            <div key={label}>
              <dd>{value}</dd>
              <dt>{label}</dt>
            </div>
          ))}
        </dl>
        <p className="small-note">From the README. Not rerun here, because the ACDC scans are not in the repository.</p>
      </section>
    </div>
  );
}
