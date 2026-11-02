// The single-patient view: three findings, the plan, and three ways of arranging them.
import {
  MISSING_ACTION, TTN_LABEL, TTN_ORDER, LGE_PATTERN_LABEL, driverText, efBand, modelVersion,
  pct, pethBand, planFor, sexWord,
} from './model.js';
import { ConfidenceBar, EfScale, Flag, PethScale, StatusChip, Term } from './ui.jsx';

const round = (v) => (v == null ? '–' : Math.round(v));

function Finding({ title, onOpen, section, children }) {
  return (
    <section className="finding">
      <header className="finding-head">
        <h3>{title}</h3>
        <button type="button" className="link" onClick={() => onOpen('values', section)}>
          All values
        </button>
      </header>
      {children}
    </section>
  );
}

function PethFinding({ rec, onOpen }) {
  const done = rec.peth_available && rec.peth_value_ng_ml != null;
  const band = pethBand(rec.peth_value_ng_ml);
  return (
    <Finding title="PEth" section="peth" onOpen={onOpen}>
      {done ? (
        <>
          <p className="reading">
            <span className="value">{rec.peth_value_ng_ml}</span>
            <span className="unit">ng/mL</span>
            <Flag value={band.flag} />
          </p>
          <p className="reading-note">{band.label} over the last 2 to 4 weeks</p>
          <PethScale value={rec.peth_value_ng_ml} />
        </>
      ) : (
        <p className="absent">Not done</p>
      )}
      {rec.time_since_last_drink_days != null && (
        <p className="meta">Last drink {rec.time_since_last_drink_days} days ago</p>
      )}
    </Finding>
  );
}

function scarLine(rec) {
  if (!rec.lge_available) return 'Not performed';
  if (rec.lge_status === 'NEGATIVE') return 'No scar';
  return `${LGE_PATTERN_LABEL[rec.lge_pattern] ?? 'Scar present'}, ${rec.lge_extent_percent}% of LV mass`;
}

function MriFinding({ rec, onOpen }) {
  const lv = efBand(rec.baseline_lvef);
  return (
    <Finding title="Cardiac MRI" section="mri" onOpen={onOpen}>
      <div className="ef-pair">
        <div>
          <p className="reading">
            <span className="value">{rec.baseline_lvef}</span>
            <span className="unit">%</span>
            <Flag value={lv?.flag} />
          </p>
          <p className="reading-note">
            <Term tip="Left ventricular ejection fraction: the share of blood the main pumping chamber pushes out each beat.">LVEF</Term>
            {' · '}
            {lv?.label}
          </p>
          <EfScale value={rec.baseline_lvef} marks />
        </div>
        <div>
          <p className="reading">
            <span className="value">{rec.rvef_percent ?? '–'}</span>
            <span className="unit">%</span>
          </p>
          <p className="reading-note">
            <Term tip="Right ventricular ejection fraction: the same measure for the chamber that pumps to the lungs.">RVEF</Term>
          </p>
          {rec.rvef_percent != null && <EfScale value={rec.rvef_percent} />}
        </div>
      </div>
      <dl className="inline-stats">
        <div>
          <dt><Term tip="Late gadolinium enhancement shows scar in the heart muscle. Scar does not recover.">Scar on LGE</Term></dt>
          <dd>{scarLine(rec)}</dd>
        </div>
        <div>
          <dt><Term tip="End-diastolic volume: blood in the left ventricle when it is fullest. High values mean the chamber has stretched.">LV EDV</Term></dt>
          <dd>{round(rec.edv_ml)} mL</dd>
        </div>
      </dl>
    </Finding>
  );
}

function TtnFinding({ rec, onOpen }) {
  const value = rec.ttn_classification;
  const missing = value === 'NOT_TESTED';
  return (
    <Finding title="TTN genetics" section="ttn" onOpen={onOpen}>
      {missing ? <p className="absent">Not tested</p> : <p className="reading-text">{TTN_LABEL[value]}</p>}
      {value === 'VARIANT_OF_UNCERTAIN_SIGNIFICANCE' && (
        <p className="reading-note">Uncertain. Not treated as a cause of the cardiomyopathy.</p>
      )}
      <ol className="ladder" aria-hidden="true">
        {TTN_ORDER.slice(0, 4).map((key) => (
          <li key={key} className={key === value ? 'on' : ''} />
        ))}
      </ol>
      <div className="ladder-labels" aria-hidden="true">
        <span>Pathogenic</span>
        <span>None found</span>
      </div>
    </Finding>
  );
}

function Drivers({ a, rec }) {
  const top = a.drivers.slice(0, 4);
  const max = Math.max(...top.map((d) => Math.abs(d.phi)));
  const showsPeth = top.some((d) => d.key === 'peth' && d.phi > 0);
  return (
    <section className="block">
      <h3>What moved the estimate</h3>
      <ul className="drivers">
        {top.map((d) => (
          <li key={d.key}>
            <span className="driver-name">{driverText(d.key, rec)}</span>
            <span className="driver-bar" aria-hidden="true">
              <span className={`half ${d.phi < 0 ? 'on' : ''}`} style={d.phi < 0 ? { '--w': `${(Math.abs(d.phi) / max) * 100}%` } : undefined} />
              <span className={`half ${d.phi > 0 ? 'on' : ''}`} style={d.phi > 0 ? { '--w': `${(d.phi / max) * 100}%` } : undefined} />
            </span>
            <span className="driver-dir">{d.phi > 0 ? 'Raises' : 'Lowers'}</span>
          </li>
        ))}
      </ul>
      {showsPeth && (
        <p className="small-note">
          High PEth raises the estimate only because the model assumes full abstinence, which removes a reversible cause. It is not protective.
        </p>
      )}
    </section>
  );
}

function PlanList({ plan }) {
  return (
    <>
      <section className="block">
        <h3>Medications</h3>
        <ul className="meds">
          {plan.medications.map((m) => (
            <li key={m.cls}>
              <strong>{m.drug}</strong>
              <span>{m.cls}</span>
            </li>
          ))}
        </ul>
        <p className="small-note">Check blood pressure, kidney function and potassium first. They are not captured here.</p>
      </section>
      <section className="block">
        <h3>Procedures and surgery</h3>
        <ul className="plain">
          {plan.procedures.map((p) => (
            <li key={p.title}>
              <strong>{p.title}.</strong> {p.detail}
            </li>
          ))}
        </ul>
      </section>
      <section className="block">
        <h3>Follow-up</h3>
        <ul className="plain">
          {plan.followUp.map((f) => (
            <li key={f}>{f}</li>
          ))}
        </ul>
      </section>
    </>
  );
}

function verdict(a) {
  const likely = a.p >= 0.5;
  return {
    title: likely ? 'Recovery likely' : 'Recovery unlikely',
    line: `${pct(a.p)} chance LVEF rises 10 or more points within 6 months, if the patient stays abstinent.`,
  };
}

function Verdict({ a, threshold }) {
  const v = verdict(a);
  return (
    <div className="verdict">
      <h2>{v.title}</h2>
      <p>{v.line}</p>
      <ConfidenceBar value={a.confidence} threshold={threshold} />
    </div>
  );
}

function Unsure({ a, rec, threshold, onEdit }) {
  const byData = a.missing.length > 0;
  const pulling = a.drivers.filter((d) => Math.abs(d.phi) > 0.2).slice(0, 4);
  return (
    <div className="void">
      <div className="void-hatch" aria-hidden="true" />
      <div className="void-body">
        <h2>Not sure. No recommendation.</h2>
        {byData ? (
          <p>
            Too much is unknown. Depending on the missing results, the estimate could be anywhere from{' '}
            <strong>{pct(a.low)}</strong> to <strong>{pct(a.high)}</strong>.
          </p>
        ) : (
          <p>The results point in both directions, so the model cannot separate recovery from no recovery.</p>
        )}
        <ConfidenceBar value={a.confidence} threshold={threshold} />
        {byData ? (
          <div className="next-steps">
            <h3>To get a recommendation</h3>
            <ol>
              {a.swings.map((s, i) => (
                <li key={s.key}>
                  <strong>{MISSING_ACTION[s.key].order}</strong>
                  {i === 0 && a.swings.length > 1 && <span> Settles the most.</span>}
                </li>
              ))}
            </ol>
            <button type="button" className="btn btn-primary" onClick={onEdit}>
              Add results
            </button>
          </div>
        ) : (
          <div className="next-steps">
            <h3>Pulling each way</h3>
            <ul className="pull">
              {pulling.map((d) => (
                <li key={d.key}>
                  <span className="driver-dir">{d.phi > 0 ? 'Raises' : 'Lowers'}</span>
                  {driverText(d.key, rec)}
                </li>
              ))}
            </ul>
            <p className="small-note">This one needs clinical judgment. Rerun after new results or a 3-month LVEF.</p>
          </div>
        )}
      </div>
    </div>
  );
}

function Ineligible() {
  return (
    <div className="void void-off">
      <div className="void-body">
        <h2>No plan for this patient.</h2>
        <p>
          This record is in the continued drinking group. Plans assume withdrawal is complete and the patient stays abstinent.
        </p>
      </div>
    </div>
  );
}

function PlanPanel({ rec, a, threshold, onEdit, onOpen }) {
  if (a.status === 'ineligible') return <Ineligible />;
  if (a.status === 'unsure') return <Unsure a={a} rec={rec} threshold={threshold} onEdit={onEdit} />;
  return (
    <>
      <Verdict a={a} threshold={threshold} />
      <PlanList plan={planFor(rec, a)} />
      <Drivers a={a} rec={rec} />
      <p className="placeholder-note">Sample plan content. The treatment model is not built yet.</p>
      <button type="button" className="link" onClick={() => onOpen('whatif')}>
        Try different values
      </button>
    </>
  );
}

function ChartLayout({ rec, a, threshold, onOpen, onEdit }) {
  return (
    <div className="layout-chart">
      <div className="findings">
        <PethFinding rec={rec} onOpen={onOpen} />
        <MriFinding rec={rec} onOpen={onOpen} />
        <TtnFinding rec={rec} onOpen={onOpen} />
      </div>
      <aside className="plan-col" aria-label="Assessment and plan">
        <PlanPanel rec={rec} a={a} threshold={threshold} onEdit={onEdit} onOpen={onOpen} />
      </aside>
    </div>
  );
}

function PlanFirstLayout({ rec, a, threshold, onOpen, onEdit }) {
  const ready = a.status === 'plan';
  return (
    <div className="layout-first">
      <div className="findings findings-row">
        <PethFinding rec={rec} onOpen={onOpen} />
        <MriFinding rec={rec} onOpen={onOpen} />
        <TtnFinding rec={rec} onOpen={onOpen} />
      </div>
      {ready ? (
        <section className="first-plan" aria-label="Assessment and plan">
          <Verdict a={a} threshold={threshold} />
          <div className="first-grid">
            <PlanList plan={planFor(rec, a)} />
          </div>
          <div className="first-foot">
            <Drivers a={a} rec={rec} />
            <p className="placeholder-note">Sample plan content. The treatment model is not built yet.</p>
          </div>
        </section>
      ) : (
        <PlanPanel rec={rec} a={a} threshold={threshold} onEdit={onEdit} onOpen={onOpen} />
      )}
    </div>
  );
}

function lgeSentence(rec) {
  if (!rec.lge_available) return 'LGE not performed.';
  if (rec.lge_status === 'NEGATIVE') return 'No scar on LGE.';
  return `${LGE_PATTERN_LABEL[rec.lge_pattern] ?? 'Positive'} scar on LGE, ${rec.lge_extent_percent}% of LV mass.`;
}

function NoteLayout({ rec, a, threshold, onEdit, onOpen }) {
  const band = pethBand(rec.peth_value_ng_ml);
  const who = `${rec.age}-year-old ${rec.biological_sex === 'FEMALE' ? 'woman' : 'man'}`;
  const plan = a.status === 'plan' ? planFor(rec, a) : null;
  return (
    <article className="note">
      <header className="note-head">
        <h2>Consult note</h2>
        <p>
          {who}
          {rec.time_since_last_drink_days != null && `, last drink ${rec.time_since_last_drink_days} days ago`}.{a.status !== 'ineligible' && ' Withdrawal complete.'}
        </p>
      </header>
      <section>
        <h3>Data</h3>
        <table className="note-data">
          <tbody>
            <tr>
              <th scope="row">PEth</th>
              <td>
                {band ? (
                  <>
                    {rec.peth_value_ng_ml} ng/mL <Flag value={band.flag} /> {band.label.toLowerCase()}
                  </>
                ) : (
                  'Not done'
                )}
              </td>
            </tr>
            <tr>
              <th scope="row">Cardiac MRI</th>
              <td>
                LVEF {rec.baseline_lvef}% <Flag value={efBand(rec.baseline_lvef)?.flag} />, RVEF {rec.rvef_percent ?? '–'}%, LV EDV {round(rec.edv_ml)} mL, LV ESV {round(rec.esv_ml)} mL. {lgeSentence(rec)}
              </td>
            </tr>
            <tr>
              <th scope="row">TTN</th>
              <td>{TTN_LABEL[rec.ttn_classification]}</td>
            </tr>
          </tbody>
        </table>
        <button type="button" className="link" onClick={() => onOpen('values')}>
          All values
        </button>
      </section>
      <section>
        <h3>Assessment</h3>
        {a.status === 'plan' ? (
          <>
            <p className="note-lead">{verdict(a).title}.</p>
            <p>{verdict(a).line}</p>
            <ConfidenceBar value={a.confidence} threshold={threshold} />
          </>
        ) : (
          <PlanPanel rec={rec} a={a} threshold={threshold} onEdit={onEdit} onOpen={onOpen} />
        )}
      </section>
      {plan && (
        <section>
          <h3>Plan</h3>
          <ol className="note-plan">
            <li>
              <strong>Medications.</strong> {plan.medications.map((m) => m.drug).join(', ')}. Check blood pressure, kidney function and potassium first.
            </li>
            <li>
              <strong>Procedures and surgery.</strong> {plan.procedures.map((p) => `${p.title}. ${p.detail}`).join(' ')}
            </li>
            <li>
              <strong>Follow-up.</strong> {plan.followUp.join('. ')}.
            </li>
          </ol>
          <p className="placeholder-note">Sample plan content. The treatment model is not built yet.</p>
        </section>
      )}
    </article>
  );
}

export function PatientChart({ rec, a, threshold, layout, position, onPrev, onNext, onOpen, onEdit, onCopy, copied, note, onPatientView, shared }) {
  const Layout = layout === 'first' ? PlanFirstLayout : layout === 'note' ? NoteLayout : ChartLayout;
  const isNew = rec.patient_id.startsWith('CRIP-NEW');
  return (
    <div className="chart">
      <header className="chart-head">
        <div className="chart-id">
          <h1>{rec.patient_id}</h1>
          <p>
            {rec.age} {sexWord(rec.biological_sex)}
            <span className="dot" />
            {isNew ? 'Entered this session' : 'Synthetic record'}
            <span className="dot" />
            Model {modelVersion}
          </p>
          {note && <p className="demo-note">{note}</p>}
        </div>
        <div className="chart-actions">
          <StatusChip status={a.status} />
          <button type="button" className="btn" onClick={onCopy}>
            {copied ? 'Copied' : 'Copy summary'}
          </button>
          <button type="button" className="btn" onClick={onEdit}>
            Edit results
          </button>
          <button type="button" className="btn" onClick={onPatientView}>
            {shared ? 'Patient view (shared)' : 'Patient view'}
          </button>
          {position && (
            <div className="stepper" aria-label="Move between patients">
              <button type="button" className="btn btn-icon" onClick={onPrev} aria-label="Previous patient" disabled={!onPrev}>
                ‹
              </button>
              <span>
                {position.index + 1} of {position.total.toLocaleString()}
              </span>
              <button type="button" className="btn btn-icon" onClick={onNext} aria-label="Next patient" disabled={!onNext}>
                ›
              </button>
            </div>
          )}
        </div>
      </header>
      <Layout rec={rec} a={a} threshold={threshold} onOpen={onOpen} onEdit={onEdit} />
    </div>
  );
}

export function summaryText(rec, a) {
  const lines = [
    `${rec.patient_id}, ${rec.age} ${sexWord(rec.biological_sex)}`,
    `PEth: ${rec.peth_available ? `${rec.peth_value_ng_ml} ng/mL` : 'not done'}`,
    `LVEF ${rec.baseline_lvef}%, RVEF ${rec.rvef_percent ?? '–'}%. ${lgeSentence(rec)}`,
    `TTN: ${TTN_LABEL[rec.ttn_classification]}`,
  ];
  if (a.status === 'plan') {
    const plan = planFor(rec, a);
    lines.push(`${verdict(a).title}. ${verdict(a).line} Confidence ${Math.round(a.confidence * 100)}%.`);
    lines.push(`Medications: ${plan.medications.map((m) => m.drug).join(', ')}.`);
    lines.push(`Procedures and surgery: ${plan.procedures.map((p) => p.title).join('; ')}.`);
  } else if (a.status === 'unsure') {
    lines.push(`Not sure. No recommendation. Confidence ${Math.round(a.confidence * 100)}%.`);
  }
  lines.push('Research prototype on synthetic data. Not for clinical use.');
  return lines.join('\n');
}
