import {
  DEFAULT_THRESHOLD, LGE_PATTERN_LABEL, TTN_LABEL, TTN_ORDER, assess, demoPatients,
  patients, pct, predict, sexWord,
} from './model.js';
import { Intake } from './intake.jsx';
import { ModelsView } from './models.jsx';
import { PatientPage } from './patient.jsx';
import { PatientChart, summaryText } from './chart.jsx';
import { Segmented, StatusMark } from './ui.jsx';
import mriOriginal from './data/mri-original.jpg';
import mriOverlay from './data/mri-overlay.jpg';

const { useCallback, useEffect, useMemo, useRef, useState } = React;

const LIST_LIMIT = 100;
const DEFAULT_OPTIONS = { layout: 'chart', intake: 'sheet', threshold: DEFAULT_THRESHOLD, size: 'standard', theme: 'system' };

function loadOptions() {
  try {
    return { ...DEFAULT_OPTIONS, ...JSON.parse(localStorage.getItem('crip.options') ?? '{}') };
  } catch {
    return DEFAULT_OPTIONS;
  }
}

function saveOptions(options) {
  try {
    localStorage.setItem('crip.options', JSON.stringify(options));
  } catch {
    // Storage can be blocked; options then last only for this visit.
  }
}

function applyDisplay({ size, theme }) {
  const root = document.documentElement;
  root.classList.toggle('projector', size === 'projector');
  if (theme === 'system') root.removeAttribute('data-theme');
  else root.setAttribute('data-theme', theme);
}

const demoNote = Object.fromEntries(demoPatients.map((d) => [d.id, d.note]));
const demoOrder = demoPatients.map((d) => d.id);

function initialView() {
  const token = location.hash.slice(1);
  if (token === 'models') return { name: 'models' };
  if (token === 'new') return { name: 'new' };
  if (token && patients.some((p) => p.patient_id === token)) return { name: 'patient', id: token };
  const shareId = token.replace('patient-view.', '');
  if (shareId !== token && patients.some((p) => p.patient_id === shareId)) return { name: 'patient-page', id: shareId, standalone: true };
  return { name: 'patient', id: demoOrder[0] };
}

const FILTERS = {
  sex: { label: 'Sex', options: [['MALE', 'Male'], ['FEMALE', 'Female']], test: (r, v) => r.biological_sex === v },
  ttn: { label: 'TTN', options: TTN_ORDER.map((k) => [k, TTN_LABEL[k]]), test: (r, v) => r.ttn_classification === v },
  lge: {
    label: 'Scar on LGE',
    options: [['NEGATIVE', 'No scar'], ['POSITIVE', 'Scar present'], ['NOT_PERFORMED', 'Not performed']],
    test: (r, v) => r.lge_status === v,
  },
  peth: {
    label: 'PEth',
    options: [['low', 'Under 20'], ['mid', '20 to 200'], ['high', 'Over 200'], ['none', 'Not done']],
    test: (r, v) => {
      const p = r.peth_value_ng_ml;
      if (v === 'none') return !r.peth_available;
      if (p == null) return false;
      return v === 'low' ? p < 20 : v === 'mid' ? p <= 200 : p > 200;
    },
  },
};

function Worklist({ tab, setTab, counts, query, setQuery, filters, setFilters, rows, activeId, onPick, searchRef, edits }) {
  const active = Object.values(filters).filter(Boolean).length;
  return (
    <nav className="list" aria-label="Patients">
      <div className="list-tools">
        <label className="search" htmlFor="patient-search">
          <span className="sr">Search by patient ID</span>
          <input id="patient-search" ref={searchRef} type="search" placeholder="Search patient ID" value={query} onChange={(e) => setQuery(e.target.value)} />
          <kbd aria-hidden="true">/</kbd>
        </label>
        <div className="tabs" role="group" aria-label="Which patients to show">
          {[['demo', 'Demo'], ['all', 'All'], ['unsure', 'Not sure']].map(([key, label]) => (
            <button key={key} type="button" className={tab === key ? 'on' : ''} aria-pressed={tab === key} onClick={() => setTab(key)}>
              {label} <span>{counts[key].toLocaleString()}</span>
            </button>
          ))}
        </div>
        <details className="filters">
          <summary>Filters{active > 0 && ` (${active})`}</summary>
          <div className="filter-grid">
            {Object.entries(FILTERS).map(([key, spec]) => (
              <label key={key} htmlFor={`filter-${key}`}>
                {spec.label}
                <select id={`filter-${key}`} value={filters[key] ?? ''} onChange={(e) => setFilters({ ...filters, [key]: e.target.value || null })}>
                  <option value="">All</option>
                  {spec.options.map(([value, label]) => (
                    <option key={value} value={value}>{label}</option>
                  ))}
                </select>
              </label>
            ))}
          </div>
        </details>
      </div>
      <ul className="rows">
        {rows.slice(0, LIST_LIMIT).map(({ rec, a }) => (
          <li key={rec.patient_id}>
            <button type="button" className={rec.patient_id === activeId ? 'on' : ''} aria-current={rec.patient_id === activeId ? 'true' : undefined} onClick={() => onPick(rec.patient_id)}>
              <StatusMark status={a.status} />
              <span className="row-id">{rec.patient_id.replace('CRIP-', '')}</span>
              <span className="row-who">{rec.age} {sexWord(rec.biological_sex)}</span>
              {tab === 'demo' && <span className="row-note">{edits[rec.patient_id] ? 'Results updated this session' : demoNote[rec.patient_id]}</span>}
            </button>
          </li>
        ))}
      </ul>
      {rows.length === 0 && <p className="list-empty">No patients match. Clear the search or filters.</p>}
      {rows.length > LIST_LIMIT && (
        <p className="list-more">Showing {LIST_LIMIT} of {rows.length.toLocaleString()}. Search or filter to narrow.</p>
      )}
    </nav>
  );
}

function valueGroups(rec) {
  const fmt = (v, unit = '') => (v == null ? 'Not recorded' : `${v}${unit}`);
  const ml = (v) => (v == null ? 'Not recorded' : `${Math.round(v)} mL`);
  const groups = [
    {
      id: 'peth',
      title: 'PEth and alcohol',
      rows: [
        ['PEth', rec.peth_available ? `${rec.peth_value_ng_ml} ng/mL` : 'Not done', 'Blood marker of alcohol use over the last 2 to 4 weeks. Can read falsely high in advanced liver disease.'],
        ['Drawn on', fmt(rec.peth_measurement_date), null],
        ['Drinks per week', fmt(rec.reported_drinks_per_week), 'Self-reported, so less reliable than PEth.'],
        ['Days since last drink', fmt(rec.time_since_last_drink_days), 'Self-reported.'],
        ['Years of heavy drinking', fmt(rec.years_heavy_drinking), 'More than 80 g of alcohol a day.'],
      ],
    },
    {
      id: 'mri',
      title: 'Left ventricle',
      rows: [
        ['LVEF', fmt(rec.baseline_lvef, '%'), 'Share of blood pumped out with each beat. 35% or below is the usual ICD range.'],
        ['LV EDV', ml(rec.edv_ml), 'End-diastolic volume: blood in the chamber when it is fullest. High values mean it has stretched.'],
        ['LV ESV', ml(rec.esv_ml), 'End-systolic volume: blood left after the squeeze.'],
        ['LV stroke volume', ml(rec.stroke_volume_ml), 'EDV minus ESV: blood pumped each beat.'],
        ['LV mass', fmt(rec.myocardial_mass_g, ' g'), 'Weight of the heart muscle, estimated from MRI at 1.05 g/mL.'],
      ],
    },
    {
      id: 'rv',
      title: 'Right ventricle',
      rows: [
        ['RVEF', fmt(rec.rvef_percent, '%'), 'The same pumping measure for the chamber that feeds the lungs.'],
        ['RV EDV', ml(rec.rvedv_ml), 'Blood in the right ventricle when fullest.'],
        ['RV ESV', ml(rec.rvesv_ml), 'Blood left after the squeeze.'],
      ],
    },
    {
      id: 'lge',
      title: 'Scar on LGE',
      rows: [
        ['Result', !rec.lge_available ? 'Not performed' : rec.lge_status === 'NEGATIVE' ? 'No scar' : 'Scar present', 'Late gadolinium enhancement marks scar, which does not recover. The strongest single signal in this model.'],
        ['Pattern', LGE_PATTERN_LABEL[rec.lge_pattern] ?? 'None', 'Midwall scar is the typical non-ischemic pattern.'],
        ['Extent', rec.lge_extent_percent == null ? 'Not recorded' : `${rec.lge_extent_percent}% of LV mass`, null],
      ],
    },
    {
      id: 'ttn',
      title: 'TTN genetics',
      rows: [
        ['Result', TTN_LABEL[rec.ttn_classification], 'Titin variants can cause dilated cardiomyopathy. A VUS is never treated as a confirmed cause.'],
        ['Standard', 'ACMG-AMP 2015, ClinVar', null],
      ],
    },
  ];
  if (rec.follow_up_lvef != null) {
    groups.push({
      id: 'outcome',
      title: 'Simulated 6-month outcome',
      note: 'The synthetic answer key. A real patient would not have this yet.',
      rows: [
        ['LVEF at 6 months', `${rec.follow_up_lvef}%`, null],
        ['Change', `${rec.delta_lvef > 0 ? '+' : ''}${rec.delta_lvef} points`, 'Recovery means a rise of 10 points or more.'],
        ['Result', rec.recovery_category.replaceAll('_', ' ').toLowerCase(), null],
        ['Adverse cardiac events', rec.adverse_cardiac_events === 'NONE' ? 'None' : rec.adverse_cardiac_events, null],
      ],
    });
  }
  return groups;
}

function WhatIf({ rec, threshold }) {
  const [peth, setPeth] = useState(rec.peth_value_ng_ml ?? '');
  const [ttn, setTtn] = useState(rec.ttn_classification);
  const [scar, setScar] = useState(rec.lge_available ? rec.lge_status : 'NOT_PERFORMED');
  const changed = {
    ...rec,
    peth_value_ng_ml: peth === '' ? null : Number(peth),
    peth_available: peth === '' ? 0 : 1,
    ttn_classification: ttn,
    lge_status: scar,
    lge_available: scar === 'NOT_PERFORMED' ? 0 : 1,
    lge_pattern: scar === 'POSITIVE' ? (rec.lge_pattern === 'NONE' ? 'MIDWALL' : rec.lge_pattern) : 'NONE',
    lge_extent_percent: scar === 'POSITIVE' ? (rec.lge_extent_percent || 10) : scar === 'NEGATIVE' ? 0 : null,
  };
  const before = predict(rec).p;
  const after = assess(changed, threshold);
  return (
    <div className="whatif">
      <p>Change a result to see how the model responds. This shows the model's sensitivity. It does not predict what a treatment would do.</p>
      <label htmlFor="whatif-peth">PEth (ng/mL)<input id="whatif-peth" inputMode="decimal" value={peth} onChange={(e) => setPeth(e.target.value)} /></label>
      <label htmlFor="whatif-ttn">TTN
        <select id="whatif-ttn" value={ttn} onChange={(e) => setTtn(e.target.value)}>
          {TTN_ORDER.map((k) => <option key={k} value={k}>{TTN_LABEL[k]}</option>)}
        </select>
      </label>
      <label htmlFor="whatif-lge">Scar on LGE
        <select id="whatif-lge" value={scar} onChange={(e) => setScar(e.target.value)}>
          <option value="NEGATIVE">No scar</option>
          <option value="POSITIVE">Scar present</option>
          <option value="NOT_PERFORMED">Not performed</option>
        </select>
      </label>
      <dl className="whatif-result">
        <div><dt>Current estimate</dt><dd>{pct(before)}</dd></div>
        <div><dt>With these values</dt><dd>{after.status === 'plan' ? pct(after.p) : 'Not sure'}</dd></div>
      </dl>
    </div>
  );
}

function Images() {
  return (
    <div className="images">
      <div className="image-pair">
        <figure><img src={mriOriginal} alt="Short-axis cine MRI slice of the heart" /><figcaption>Original</figcaption></figure>
        <figure><img src={mriOverlay} alt="The same slice with the right ventricle, myocardium and left ventricle outlined in color" /><figcaption>Expert contours</figcaption></figure>
      </div>
      <ul className="mri-legend">
        <li><i style={{ background: 'var(--mri-rv)' }} />Right ventricle</li>
        <li><i style={{ background: 'var(--mri-myo)' }} />Myocardium</li>
        <li><i style={{ background: 'var(--mri-lv)' }} />Left ventricle</li>
      </ul>
      <p><strong>Sample image, not this patient.</strong> ACDC patient001, end-diastole, slice 6 of 10. Synthetic records have numbers only. The slice and phase viewer from the Streamlit MRI page moves here.</p>
      <p className="small-note">ACDC dataset, CC BY-NC-SA 4.0. Bernard et al., IEEE Trans Med Imaging, 2018.</p>
    </div>
  );
}

function Drawer({ rec, threshold, tab, section, setTab, onClose }) {
  const panelRef = useRef(null);
  useEffect(() => {
    panelRef.current?.querySelector('button')?.focus();
    const close = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, [onClose]);
  useEffect(() => {
    if (!section) return;
    const body = panelRef.current?.querySelector('.drawer-body');
    const target = body?.querySelector(`[data-group="${section}"]`);
    if (body && target) body.scrollTop = target.offsetTop - body.offsetTop;
  }, [section]);
  return (
    <div className="scrim" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <aside className="drawer" ref={panelRef} role="dialog" aria-modal="true" aria-label={`Details for ${rec.patient_id}`}>
        <header>
          <div className="tabs" role="group" aria-label="Detail views">
            {[['values', 'All values'], ['images', 'MRI images'], ['whatif', 'What if']].map(([key, label]) => (
              <button key={key} type="button" className={tab === key ? 'on' : ''} aria-pressed={tab === key} onClick={() => setTab(key)}>{label}</button>
            ))}
          </div>
          <button type="button" className="btn btn-icon" onClick={onClose} aria-label="Close details">×</button>
        </header>
        <div className="drawer-body">
          {tab === 'values' && valueGroups(rec).map((group) => (
            <section key={group.id} data-group={group.id}>
              <h3>{group.title}</h3>
              {group.note && <p className="small-note">{group.note}</p>}
              <dl className="values">
                {group.rows.map(([term, value, meaning]) => (
                  <div key={term}>
                    <dt>{term}</dt>
                    <dd>{value}</dd>
                    {meaning && <p>{meaning}</p>}
                  </div>
                ))}
              </dl>
            </section>
          ))}
          {tab === 'images' && <Images />}
          {tab === 'whatif' && <WhatIf rec={rec} threshold={threshold} />}
        </div>
      </aside>
    </div>
  );
}

function OptionsPanel({ options, change, onClose }) {
  return (
    <aside className="options" aria-label="Design options">
      <header>
        <h2>Design options</h2>
        <button type="button" className="btn btn-icon" onClick={onClose} aria-label="Close design options">×</button>
      </header>
      <Segmented legend="Patient view" name="opt-layout" value={options.layout} onChange={(v) => change('layout', v)} options={[['chart', 'Chart'], ['first', 'Plan below'], ['note', 'Consult note']]} />
      <Segmented legend="New patient form" name="opt-intake" value={options.intake} onChange={(v) => change('intake', v)} options={[['sheet', 'One sheet'], ['steps', 'Steps'], ['flow', 'Flowsheet']]} />
      <label className="threshold" htmlFor="threshold-option">
        <span>Confidence threshold</span>
        <input id="threshold-option" type="range" min="50" max="99" value={options.threshold} onChange={(e) => change('threshold', Number(e.target.value))} />
        <output htmlFor="threshold-option">{options.threshold}%</output>
      </label>
      <Segmented legend="Text size" name="opt-size" value={options.size} onChange={(v) => change('size', v)} options={[['standard', 'Standard'], ['projector', 'Projector']]} />
      <Segmented legend="Theme" name="opt-theme" value={options.theme} onChange={(v) => change('theme', v)} options={[['system', 'System'], ['light', 'Light'], ['dark', 'Dark']]} />
    </aside>
  );
}

function App() {
  const [options, setOptions] = useState(loadOptions);
  const [view, setView] = useState(initialView);
  const [added, setAdded] = useState([]);
  const [edits, setEdits] = useState({});
  const [shared, setShared] = useState({});
  const [tab, setTab] = useState('demo');
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState({});
  const [drawer, setDrawer] = useState(null);
  const [showOptions, setShowOptions] = useState(false);
  const [showList, setShowList] = useState(false);
  const [copied, setCopied] = useState(false);
  const searchRef = useRef(null);
  const threshold = options.threshold;

  const change = useCallback((key, value) => {
    setOptions((o) => {
      const next = { ...o, [key]: value };
      saveOptions(next);
      return next;
    });
  }, []);

  useEffect(() => applyDisplay(options), [options.size, options.theme]);

  const scored = useMemo(() => {
    const all = [...added, ...patients].map((r) => edits[r.patient_id] ?? r);
    return all.map((rec) => ({ rec, a: assess(rec, threshold) }));
  }, [added, edits, threshold]);

  const byId = useMemo(() => new Map(scored.map((row) => [row.rec.patient_id, row])), [scored]);
  const demoRows = useMemo(() => demoOrder.map((id) => byId.get(id)), [byId]);

  const rows = useMemo(() => {
    let list = tab === 'demo' ? demoRows : tab === 'unsure' ? scored.filter((r) => r.a.status === 'unsure') : scored;
    const q = query.trim().toLowerCase();
    if (q) list = list.filter((r) => r.rec.patient_id.toLowerCase().includes(q));
    for (const [key, value] of Object.entries(filters)) {
      if (value) list = list.filter((r) => FILTERS[key].test(r.rec, value));
    }
    return list;
  }, [tab, demoRows, scored, query, filters]);

  const counts = useMemo(() => ({
    demo: demoOrder.length,
    all: scored.length,
    unsure: scored.filter((r) => r.a.status === 'unsure').length,
  }), [scored]);

  const open = (id) => {
    setView({ name: 'patient', id });
    setShowList(false);
  };

  const current = view.name === 'patient' ? byId.get(view.id) : null;
  const index = current ? rows.findIndex((r) => r.rec.patient_id === view.id) : -1;
  const step = (delta) => {
    const next = rows[index + delta];
    if (next) open(next.rec.patient_id);
  };

  useEffect(() => {
    const onKey = (e) => {
      const typing = /INPUT|SELECT|TEXTAREA/.test(e.target.tagName);
      if (typing || e.metaKey || e.ctrlKey || e.altKey || drawer || view.name === 'patient-page') return;
      if (e.key === '/') {
        e.preventDefault();
        searchRef.current?.focus();
      } else if (e.key === 'n') {
        setView({ name: 'new' });
      } else if (e.key === 'ArrowRight') {
        step(1);
      } else if (e.key === 'ArrowLeft') {
        step(-1);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  useEffect(() => {
    document.querySelector('.content')?.scrollTo(0, 0);
  }, [view]);

  const submit = (rec) => {
    if (view.name === 'edit') {
      setEdits((e) => ({ ...e, [rec.patient_id]: rec }));
      setShared((list) => ({ ...list, [rec.patient_id]: false }));
      open(rec.patient_id);
      return;
    }
    const id = `CRIP-NEW-${String(added.length + 1).padStart(4, '0')}`;
    setAdded((list) => [{ ...rec, patient_id: id }, ...list]);
    setTab('all');
    setQuery('');
    setFilters({});
    open(id);
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(summaryText(current.rec, current.a));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  const closeDrawer = useCallback(() => setDrawer(null), []);

  if (view.name === 'patient-page') {
    const row = byId.get(view.id);
    return (
      <PatientPage
        rec={row.rec}
        a={row.a}
        preview={!view.standalone}
        shared={Boolean(shared[view.id])}
        onShare={() => setShared((list) => ({ ...list, [view.id]: true }))}
        onBack={() => open(view.id)}
      />
    );
  }

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#" onClick={(e) => { e.preventDefault(); open(view.id ?? demoOrder[0]); }}>
          CRIP
          <span>Cardiomyopathy recovery</span>
        </a>
        <nav className="topnav" aria-label="Main">
          <button type="button" className={view.name === 'models' ? '' : 'on'} onClick={() => open(view.id ?? demoOrder[0])}>Patients</button>
          <button type="button" className={view.name === 'models' ? 'on' : ''} onClick={() => setView({ name: 'models', id: view.id })}>Models</button>
        </nav>
        <p className="research">Research prototype. Synthetic patients. Not for clinical use.</p>
        <button type="button" className="btn btn-primary" onClick={() => setView({ name: 'new', id: view.id })}>New patient</button>
      </header>

      <div className={`main ${view.name === 'models' ? 'main-wide' : ''}`}>
        {view.name !== 'models' && (
          <>
            <button type="button" className="list-toggle" aria-expanded={showList} onClick={() => setShowList(!showList)}>
              {showList ? 'Hide patient list' : `Patient list (${rows.length.toLocaleString()})`}
            </button>
            <div className={`list-wrap ${showList ? 'open' : ''}`}>
              <Worklist tab={tab} setTab={setTab} counts={counts} query={query} setQuery={setQuery} filters={filters} setFilters={setFilters} rows={rows} activeId={view.id} onPick={open} searchRef={searchRef} edits={edits} />
            </div>
          </>
        )}
        <main className="content">
          {current && (
            <PatientChart
              rec={current.rec}
              a={current.a}
              threshold={threshold}
              layout={options.layout}
              note={tab === 'demo' && !edits[view.id] ? demoNote[view.id] : null}
              position={index >= 0 ? { index, total: rows.length } : null}
              onPrev={index > 0 ? () => step(-1) : null}
              onNext={index >= 0 && index < rows.length - 1 ? () => step(1) : null}
              onOpen={(t, section) => setDrawer({ tab: t, section })}
              onEdit={() => setView({ name: 'edit', id: view.id })}
              onCopy={copy}
              copied={copied}
              onPatientView={() => setView({ name: 'patient-page', id: view.id })}
              shared={Boolean(shared[view.id])}
            />
          )}
          {(view.name === 'new' || view.name === 'edit') && (
            <Intake
              key={view.name + (view.id ?? '')}
              layout={options.intake}
              editing={view.name === 'edit' ? byId.get(view.id).rec : null}
              threshold={threshold}
              onSubmit={submit}
              onCancel={() => open(view.id ?? demoOrder[0])}
            />
          )}
          {view.name === 'models' && <ModelsView threshold={threshold} setThreshold={(v) => change('threshold', v)} />}
        </main>
      </div>

      {drawer && current && (
        <Drawer rec={current.rec} threshold={threshold} tab={drawer.tab} section={drawer.section} setTab={(t) => setDrawer({ tab: t })} onClose={closeDrawer} />
      )}

      {showOptions ? (
        <OptionsPanel options={options} change={change} onClose={() => setShowOptions(false)} />
      ) : (
        <button type="button" className="options-open" onClick={() => setShowOptions(true)}>Design options</button>
      )}
    </div>
  );
}

applyDisplay(loadOptions());
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
