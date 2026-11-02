// Entering a patient's results. All three layouts draw from SECTIONS in model.js.
import {
  EMPTY_FORM, MISSING_ACTION, SECTIONS, assess, efBand, fieldError, patients, pct, pethBand,
  readiness, toForm, toRecord,
} from './model.js';
import { ConfidenceBar, Flag, StatusChip } from './ui.jsx';

const { useEffect, useMemo, useRef, useState } = React;

// A complete record that stands in for what an EHR import or MRI upload would return.
const SAMPLE = patients.find(
  (p) => p.peth_available && p.lge_available && p.lge_status === 'POSITIVE'
    && p.ttn_classification === 'NO_PATHOGENIC_OR_LIKELY_PATHOGENIC'
    && p.assumed_abstinence_scenario === 'SUSTAINED_ABSTINENCE' && p.rvesv_ml != null,
);
const MRI_KEYS = ['edv_ml', 'esv_ml', 'rvedv_ml', 'rvesv_ml', 'myocardial_mass_g'];

function derivedValue(field, form) {
  const value = toRecord(form, 'draft')[field.derived];
  return value == null || value < 0 ? null : value;
}

function Control({ field, form, set, origin }) {
  const id = `f-${field.key}`;
  const value = form[field.key];
  if (field.kind === 'choice') {
    return (
      <div className="choices">
        {field.options.map(([key, label]) => (
          <label key={key} className={value === key ? 'on' : ''}>
            <input
              id={`${id}-${key}`}
              type="radio"
              name={id}
              data-entry
              checked={value === key}
              onChange={() => set(field.key, key)}
            />
            {label}
          </label>
        ))}
      </div>
    );
  }
  if (field.kind === 'select') {
    return (
      <select id={id} data-entry value={value} onChange={(e) => set(field.key, e.target.value)}>
        {field.options.map(([key, label]) => (
          <option key={key} value={key}>
            {label}
          </option>
        ))}
      </select>
    );
  }
  if (field.kind === 'date') {
    return <input id={id} data-entry type="date" value={value} onChange={(e) => set(field.key, e.target.value)} />;
  }
  const error = fieldError(field, form);
  return (
    <span className={`number ${error ? 'bad' : ''} ${origin[field.key] ? 'filled' : ''}`}>
      <input
        id={id}
        data-entry
        inputMode="decimal"
        autoComplete="off"
        value={value}
        aria-invalid={Boolean(error)}
        aria-describedby={error ? `${id}-error` : undefined}
        onChange={(e) => set(field.key, e.target.value)}
      />
      {field.unit && <span className="suffix">{field.unit}</span>}
    </span>
  );
}

function FieldFlag({ field, form }) {
  if (field.flag !== 'peth' || form.peth_value_ng_ml === '') return null;
  const band = pethBand(Number(form.peth_value_ng_ml));
  if (!band || fieldError(field, form)) return null;
  return (
    <span className="live-flag">
      <Flag value={band.flag} /> {band.label}
    </span>
  );
}

function Derived({ field, form }) {
  const value = derivedValue(field, form);
  const band = field.derived === 'baseline_lvef' ? efBand(value) : null;
  return (
    <div className="field derived">
      <span className="field-label">{field.label}</span>
      <output htmlFor={field.derived === 'baseline_lvef' ? 'f-edv_ml f-esv_ml' : 'f-rvedv_ml f-rvesv_ml'}>
        {value == null ? '–' : `${value}%`} <Flag value={band?.flag} />
      </output>
      <span className="field-note">Calculated</span>
    </div>
  );
}

function Field({ field, form, set, origin }) {
  if (field.derived) return <Derived field={field} form={form} />;
  const id = `f-${field.key}`;
  if (field.kind === 'check') {
    return (
      <label className={`check ${form[field.key] ? 'on' : ''}`} htmlFor={id}>
        <input id={id} data-entry type="checkbox" checked={form[field.key]} onChange={(e) => set(field.key, e.target.checked)} />
        <span>
          {field.label}
          <small>{field.note}</small>
        </span>
      </label>
    );
  }
  const error = fieldError(field, form);
  const wide = field.kind === 'choice' || field.kind === 'date' || field.kind === 'select' || field.flag || field.wide;
  const Wrapper = field.kind === 'choice' ? 'fieldset' : 'div';
  const Label = field.kind === 'choice' ? 'legend' : 'label';
  return (
    <Wrapper className={`field ${wide ? 'field-wide' : ''}`}>
      <Label className="field-label" {...(field.kind === 'choice' ? {} : { htmlFor: id })} title={field.short ? field.label : undefined}>
        {field.short ?? field.label}
      </Label>
      <Control field={field} form={form} set={set} origin={origin} />
      <FieldFlag field={field} form={form} />
      {error && (
        <span className="field-error" id={`${id}-error`}>
          {error}
        </span>
      )}
    </Wrapper>
  );
}

function SectionBody({ section, form, set, origin, onUpload, measuring }) {
  const visible = section.fields.filter((f) => !f.when || f.when(form));
  const main = visible.filter((f) => !f.more);
  const more = visible.filter((f) => f.more);
  return (
    <>
      {section.hint && <p className="section-hint">{section.hint}</p>}
      {section.upload && (
        <button type="button" className="drop" onClick={onUpload} disabled={measuring}>
          {measuring ? 'Measuring ventricles…' : 'Upload CMR study to fill volumes automatically'}
        </button>
      )}
      <div className="field-grid">
        {main.map((f) => (
          <Field key={f.key ?? f.derived} field={f} form={form} set={set} origin={origin} />
        ))}
      </div>
      {more.length > 0 && (
        <details className="more">
          <summary>Alcohol history (optional)</summary>
          <div className="field-grid">
            {more.map((f) => (
              <Field key={f.key} field={f} form={form} set={set} origin={origin} />
            ))}
          </div>
        </details>
      )}
    </>
  );
}

function SheetLayout(props) {
  const [patient, peth, mri, ttn] = SECTIONS;
  const panel = (section) => (
    <section className="entry" key={section.id}>
      <h2>{section.title}</h2>
      <SectionBody section={section} {...props} />
    </section>
  );
  return (
    <div className="sheet">
      <div className="sheet-col">
        {panel(patient)}
        {panel(peth)}
      </div>
      <div className="sheet-col">
        {panel(mri)}
        {panel(ttn)}
      </div>
    </div>
  );
}

function StepsLayout({ step, setStep, ...props }) {
  const section = SECTIONS[step];
  const last = step === SECTIONS.length - 1;
  return (
    <div className="steps">
      <ol className="step-list">
        {SECTIONS.map((s, i) => (
          <li key={s.id} className={i === step ? 'on' : i < step ? 'done' : ''}>
            <button type="button" onClick={() => setStep(i)} aria-current={i === step ? 'step' : undefined}>
              <span className="step-n">{i + 1}</span>
              {s.title}
            </button>
          </li>
        ))}
      </ol>
      <section className="entry">
        <h2>{section.title}</h2>
        <SectionBody section={section} {...props} />
      </section>
      <div className="step-nav">
        <button type="button" className="btn" onClick={() => setStep(step - 1)} disabled={step === 0}>
          Back
        </button>
        {!last && (
          <button type="button" className="btn btn-primary" onClick={() => setStep(step + 1)}>
            Next: {SECTIONS[step + 1].title}
          </button>
        )}
      </div>
    </div>
  );
}

function FlowsheetLayout({ form, set, origin, onUpload, measuring }) {
  return (
    <div className="flow-wrap">
      <button type="button" className="drop" onClick={onUpload} disabled={measuring}>
        {measuring ? 'Measuring ventricles…' : 'Upload CMR study to fill volumes automatically'}
      </button>
      <table className="flow">
        <thead>
          <tr>
            <th scope="col">Test</th>
            <th scope="col">Result</th>
            <th scope="col">Unit</th>
            <th scope="col">Range</th>
          </tr>
        </thead>
        {SECTIONS.map((section) => (
          <tbody key={section.id}>
            <tr className="flow-group">
              <th colSpan={4} scope="colgroup">{section.title}</th>
            </tr>
            {section.fields
              .filter((f) => !f.when || f.when(form))
              .map((f) => {
                if (f.derived) {
                  const value = derivedValue(f, form);
                  return (
                    <tr key={f.derived} className="flow-derived">
                      <th scope="row">{f.label}</th>
                      <td>{value ?? '–'}</td>
                      <td>{f.unit}</td>
                      <td>Calculated</td>
                    </tr>
                  );
                }
                if (f.kind === 'check') {
                  return (
                    <tr key={f.key}>
                      <td colSpan={4}>
                        <Field field={f} form={form} set={set} origin={origin} />
                      </td>
                    </tr>
                  );
                }
                const error = fieldError(f, form);
                return (
                  <tr key={f.key}>
                    <th scope="row">
                      <label htmlFor={`f-${f.key}`}>{f.label}</label>
                    </th>
                    <td>
                      <Control field={{ ...f, unit: null }} form={form} set={set} origin={origin} />
                      {error && <span className="field-error" id={`f-${f.key}-error`}>{error}</span>}
                    </td>
                    <td>{f.unit}</td>
                    <td>{f.min != null ? f.range ?? `${f.min} to ${f.max}` : ''}</td>
                  </tr>
                );
              })}
          </tbody>
        ))}
      </table>
    </div>
  );
}

function Forecast({ preview, status, threshold, onSubmit, onCancel, editing }) {
  const a = preview;
  return (
    <aside className="forecast" aria-label="What this entry will produce">
      <h2>What you will get</h2>
      {!a && <p>Enter age, sex and LV volumes to see what the model can do with these results.</p>}
      {a && (
        <>
          <StatusChip status={a.status} />
          {a.status === 'plan' && <p>These results are enough for a recommendation.</p>}
          {a.status === 'unsure' && a.missing.length > 0 && (
            <p>
              Missing results leave the estimate between <strong>{pct(a.low)}</strong> and <strong>{pct(a.high)}</strong>.{' '}
              {MISSING_ACTION[a.swings[0].key].order} would settle the most.
            </p>
          )}
          {a.status === 'unsure' && a.missing.length === 0 && <p>The results point both ways, so the model will not recommend.</p>}
          <ConfidenceBar value={a.confidence} threshold={threshold} />
        </>
      )}
      {status.needed.length > 0 && (
        <div className="needed">
          <h3>Still needed</h3>
          <ul>
            {status.needed.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </div>
      )}
      <button type="button" id="generate-plan" className="btn btn-primary btn-wide" disabled={!status.ready} onClick={onSubmit}>
        {editing ? 'Update plan' : 'Generate plan'}
      </button>
      <button type="button" className="btn btn-wide" onClick={onCancel}>
        Cancel
      </button>
    </aside>
  );
}

export function Intake({ layout, editing, threshold, onSubmit, onCancel }) {
  const [form, setForm] = useState(() => (editing ? toForm(editing) : EMPTY_FORM));
  const [origin, setOrigin] = useState({});
  const [step, setStep] = useState(0);
  const [measuring, setMeasuring] = useState(false);
  const rootRef = useRef(null);

  useEffect(() => {
    if (!editing) rootRef.current?.querySelector('[data-entry]')?.focus();
  }, [editing]);

  const set = (key, value) => {
    setForm((f) => ({ ...f, [key]: value }));
    setOrigin((o) => {
      if (!o[key]) return o;
      const { [key]: _gone, ...rest } = o;
      return rest;
    });
  };

  const status = readiness(form);
  const preview = useMemo(() => {
    if (form.age === '' || !form.biological_sex || form.edv_ml === '' || form.esv_ml === '') return null;
    if (status.errors > 0) return null;
    const rec = toRecord(form, 'preview');
    return assess(rec, threshold);
  }, [form, threshold, status.errors]);

  const fromEhr = () => {
    const filled = toForm(SAMPLE);
    setForm({ ...filled, withdrawal: false });
    setOrigin(Object.fromEntries(Object.keys(filled).filter((k) => filled[k] !== '' && typeof filled[k] === 'string').map((k) => [k, true])));
  };

  const upload = () => {
    setMeasuring(true);
    window.setTimeout(() => {
      const filled = toForm(SAMPLE);
      setForm((f) => ({ ...f, ...Object.fromEntries(MRI_KEYS.map((k) => [k, filled[k]])) }));
      setOrigin((o) => ({ ...o, ...Object.fromEntries(MRI_KEYS.map((k) => [k, true])) }));
      setMeasuring(false);
    }, 1100);
  };

  const submit = () => {
    if (status.ready) onSubmit(toRecord(form, editing?.patient_id ?? 'new'), form);
  };

  const onKeyDown = (e) => {
    if (e.key !== 'Enter' || e.target.tagName === 'BUTTON' || e.target.tagName === 'SUMMARY') return;
    e.preventDefault();
    if (e.metaKey || e.ctrlKey) {
      submit();
      return;
    }
    const entries = [...rootRef.current.querySelectorAll('[data-entry]')].filter((el) => el.type !== 'radio' || el.checked || !rootRef.current.querySelector(`input[name="${el.name}"]:checked`));
    const next = entries[entries.indexOf(e.target) + 1];
    if (next) next.focus();
    else document.getElementById('generate-plan')?.focus();
  };

  const body = { form, set, origin, onUpload: upload, measuring };

  return (
    <div className="intake" ref={rootRef}>
      <header className="intake-head">
        <div>
          <h1>{editing ? `Edit ${editing.patient_id}` : 'New patient'}</h1>
          <p>Enter PEth, cardiac MRI and TTN results. Mark anything not done, so missing results are a choice and not a gap.</p>
        </div>
        {!editing && (
          <button type="button" className="btn" onClick={fromEhr}>
            Import from EHR (FHIR)
          </button>
        )}
      </header>
      {Object.keys(origin).length > 0 && (
        <p className="import-note">Shaded values were filled in for you from sample data. Check them before generating a plan.</p>
      )}
      <form className="intake-grid" onSubmit={(e) => e.preventDefault()} onKeyDown={onKeyDown}>
        {layout === 'steps' && <StepsLayout step={step} setStep={setStep} {...body} />}
        {layout === 'flow' && <FlowsheetLayout {...body} />}
        {layout === 'sheet' && <SheetLayout {...body} />}
        <Forecast preview={preview} status={status} threshold={threshold} onSubmit={submit} onCancel={onCancel} editing={Boolean(editing)} />
      </form>
    </div>
  );
}
