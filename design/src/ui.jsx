// Small shared pieces: status marks, the reading scales, glossary terms.
import { STATUS_LABEL } from './model.js';

const { useEffect, useRef, useState } = React;

export function StatusMark({ status }) {
  if (status === 'plan') {
    return (
      <svg className="mark mark-plan" viewBox="0 0 12 12" aria-hidden="true">
        <circle cx="6" cy="6" r="5" />
      </svg>
    );
  }
  if (status === 'unsure') {
    return (
      <svg className="mark mark-unsure" viewBox="0 0 12 12" aria-hidden="true">
        <path d="M6 .8 11.4 11H.6Z" />
      </svg>
    );
  }
  return (
    <svg className="mark mark-off" viewBox="0 0 12 12" aria-hidden="true">
      <circle cx="6" cy="6" r="4.6" />
      <path d="M2.8 9.2 9.2 2.8" />
    </svg>
  );
}

export function StatusChip({ status }) {
  return (
    <span className={`chip chip-${status}`}>
      <StatusMark status={status} />
      {STATUS_LABEL[status]}
    </span>
  );
}

const clamp01 = (x) => Math.min(1, Math.max(0, x));

// PEth spans three orders of magnitude, so the scale is logarithmic.
const pethPos = (v) => clamp01(Math.log10(1 + v) / Math.log10(1 + 2000));

export function PethScale({ value }) {
  return (
    <div className="scale" role="img" aria-label={`PEth ${value} nanograms per milliliter`}>
      <div className="scale-track">
        <span className="band" style={{ width: `${pethPos(20) * 100}%` }} />
        <span className="band band-mid" style={{ width: `${(pethPos(200) - pethPos(20)) * 100}%` }} />
        <span className="band band-high" style={{ flex: 1 }} />
        <span className="pin" style={{ left: `${pethPos(value) * 100}%` }} />
      </div>
      <div className="scale-labels">
        <span style={{ width: `${pethPos(20) * 100}%` }}>Under 20</span>
        <span style={{ width: `${(pethPos(200) - pethPos(20)) * 100}%` }}>20 to 200</span>
        <span style={{ flex: 1 }}>Over 200 ng/mL</span>
      </div>
    </div>
  );
}

// LVEF and RVEF on one shared 0 to 75% scale, so the two read against each other.
const EF_MAX = 75;

export function EfScale({ value, marks }) {
  return (
    <div className="scale" role="img" aria-label={`Ejection fraction ${value} percent`}>
      <div className="scale-track">
        {marks ? (
          <>
            <span className="band band-high" style={{ width: `${(35 / EF_MAX) * 100}%` }} />
            <span className="band band-mid" style={{ width: `${(15 / EF_MAX) * 100}%` }} />
            <span className="band" style={{ flex: 1 }} />
          </>
        ) : (
          <span className="band" style={{ flex: 1 }} />
        )}
        <span className="pin" style={{ left: `${clamp01(value / EF_MAX) * 100}%` }} />
      </div>
    </div>
  );
}

export function ConfidenceBar({ value, threshold }) {
  const pass = value * 100 >= threshold;
  return (
    <div className="conf">
      <div className="conf-head">
        <span>Model confidence</span>
        <span className="conf-value">{Math.round(value * 100)}%</span>
      </div>
      <div
        className="conf-track"
        role="meter"
        aria-label="Model confidence"
        aria-valuenow={Math.round(value * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <span className={`conf-fill ${pass ? '' : 'conf-fill-low'}`} style={{ width: `${value * 100}%` }} />
        <span className="conf-tick" style={{ left: `${threshold}%` }} />
      </div>
      <div className="conf-note" style={{ paddingLeft: `min(${threshold}%, calc(100% - 7.5em))` }}>
        {threshold}% needed
      </div>
    </div>
  );
}

export function Term({ tip, children }) {
  return (
    <span className="term" tabIndex={0} data-tip={tip}>
      {children}
    </span>
  );
}

export function Flag({ value }) {
  if (!value) return null;
  return (
    <abbr className="flag" title={value === 'H' ? 'High' : 'Low'}>
      {value}
    </abbr>
  );
}

export function useWidth() {
  const ref = useRef(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const node = ref.current;
    if (!node) return undefined;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

export function Segmented({ legend, options, value, onChange, name }) {
  return (
    <fieldset className="seg">
      <legend>{legend}</legend>
      <div className="seg-row">
        {options.map(([key, label]) => (
          <label key={key} className={value === key ? 'on' : ''}>
            <input
              id={`${name}-${key}`}
              type="radio"
              name={name}
              checked={value === key}
              onChange={() => onChange(key)}
            />
            {label}
          </label>
        ))}
      </div>
    </fieldset>
  );
}
