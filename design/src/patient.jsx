// The page a patient reads after their doctor shares it. Plain language, aimed at an 8th-grade
// reading level or below. It never shows a recovery percentage or a confidence score, and a
// "Not sure" result becomes next steps, because a patient cannot act on a model's doubts.
//
// Layout follows patient-education guidance: the main message first, beside the one number that
// matters most (CDC Clear Communication Index); ejection fraction on a banded number line with the
// American Heart Association's bands (number lines read better than tables: Zikmund-Fisher et al.);
// NHS-style do and don't lists and care cards. Two columns only for short, parallel items.
import { planFor, pethBand } from './model.js';

const MEDICINE_INFO = {
  'Sacubitril/valsartan': { brand: 'Entresto', why: 'Relaxes your blood vessels so your heart does not have to work as hard.' },
  'Metoprolol succinate': { brand: 'Toprol-XL', why: 'Slows your heartbeat and gives your heart muscle time to heal.' },
  Spironolactone: { brand: 'Aldactone', why: 'Helps your body get rid of extra fluid and protects your heart.' },
  Dapagliflozin: { brand: 'Farxiga', why: 'Helps your body clear extra fluid and sugar, which eases the strain on your heart.' },
};

// "What happens next" as steps in time order; rank sorts them.
const PROCEDURE_STEP = {
  'None now': { rank: 3, when: 'In 3 months', title: 'Check your heart again', detail: 'You do not need a procedure right now. If your heart is still weak after 3 months of medicine and no alcohol, we may talk about a small device called an ICD.' },
  'None indicated': { rank: 0, when: 'Now', title: 'No procedure needed', detail: null },
  'ICD evaluation': { rank: 1, when: 'Soon', title: 'See a heart rhythm doctor', detail: 'They will talk with you about an ICD. This is a small device placed under the skin that can stop a dangerous heart rhythm.' },
  'Advanced heart failure referral': { rank: 1, when: 'Soon', title: 'See a heart failure specialist', detail: 'They can talk with you about stronger treatments if your symptoms do not get better. These include a heart pump or a heart transplant.' },
};

const FOLLOW_UP_STEP = {
  'Genetic counseling and family screening': { rank: 1, when: 'Soon', title: 'Visit a genetic counselor', detail: 'To learn what your gene result means for you and your family.' },
  'PEth monthly to confirm abstinence': { rank: 2, when: 'Every month', title: 'PEth blood test', detail: 'To help you stay on track.' },
  'Addiction medicine follow-up': { rank: 2, when: 'Ongoing', title: 'Visits with an alcohol treatment doctor', detail: 'For support with not drinking.' },
  'Repeat cardiac MRI at 6 months': { rank: 4, when: 'In 6 months', title: 'Heart MRI', detail: 'To see if your heart is getting stronger.' },
};

const NEXT_TEST = {
  peth: { title: 'PEth blood test', detail: 'Shows if alcohol has been in your body in the past few weeks.' },
  ttn: { title: 'Gene test', detail: 'Looks for a change in the TTN gene that can weaken the heart.' },
  lge: { title: 'Heart MRI with contrast dye', detail: 'Shows if there is any scar in your heart muscle.' },
};

// Line icons that take the text color. They only decorate, so screen readers skip them.
function Icon({ children }) {
  return (
    <svg className="pt-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      {children}
    </svg>
  );
}
const HeartIcon = () => <Icon><path d="M12 20s-7-4.35-7-9.5A4 4 0 0 1 12 8a4 4 0 0 1 7 2.5C19 15.65 12 20 12 20Z" /></Icon>;
const PulseIcon = () => <Icon><path d="M3 12h4l2.5-6 5 12 2.5-6h4" /></Icon>;
const ClipboardIcon = () => <Icon><rect x="5" y="4" width="14" height="17" rx="2" /><path d="M9 4V3h6v1M9 13l2 2 4-4" /></Icon>;
const ArrowIcon = () => <Icon><circle cx="12" cy="12" r="9" /><path d="m10 8 4 4-4 4" /></Icon>;
const CheckCircleIcon = () => <Icon><circle cx="12" cy="12" r="9" /><path d="m8 12.5 2.5 2.5L16 9.5" /></Icon>;
const CheckIcon = () => <Icon><path d="m5 12.5 4.5 4.5L19 7.5" /></Icon>;
const CrossIcon = () => <Icon><path d="M6 6l12 12M18 6 6 18" /></Icon>;
const AlertIcon = () => <Icon><path d="M12 4 2.8 19.5h18.4Z" /><path d="M12 10v4.5M12 17.2v.1" /></Icon>;
const ScanIcon = () => <Icon><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" /><path d="M12 16.5s-4-2.4-4-5.2A2.2 2.2 0 0 1 12 10a2.2 2.2 0 0 1 4 1.3c0 2.8-4 5.2-4 5.2Z" /></Icon>;
const DropIcon = () => <Icon><path d="M12 3.5s-6 6.6-6 10.6a6 6 0 0 0 12 0c0-4-6-10.6-6-10.6Z" /></Icon>;
const DnaIcon = () => <Icon><path d="M7 3c0 5 10 5 10 9s-10 4-10 9M17 3c0 5-10 5-10 9s10 4 10 9M9 6.5h6M9 17.5h6" /></Icon>;
const PillIcon = () => (
  <Icon>
    <g transform="rotate(-45 12 12)">
      <rect x="3" y="8.5" width="18" height="7" rx="3.5" />
      <path d="M6.5 8.5H12v7H6.5a3.5 3.5 0 0 1 0-7Z" fill="currentColor" />
    </g>
  </Icon>
);
const PhoneIcon = () => <Icon><path d="M6.5 3.5h2.7l1.5 4-2.1 1.4a11.5 11.5 0 0 0 6.5 6.5l1.4-2.1 4 1.5v2.7a2 2 0 0 1-2.2 2A16.5 16.5 0 0 1 4.5 5.7a2 2 0 0 1 2-2.2Z" /></Icon>;
const MessageIcon = () => <Icon><path d="M4 5h16v11H9l-5 4Z" /></Icon>;

function SectionTitle({ id, icon, children }) {
  return (
    <h2 className="pt-title" id={id}>
      <span className="pt-badge">{icon}</span>
      {children}
    </h2>
  );
}

function message(rec, a) {
  const weak = rec.baseline_lvef < 50;
  if (a.status === 'ineligible') {
    return {
      title: 'The first step is to stop drinking safely.',
      body: 'Your heart plan can start once you are through alcohol withdrawal. Your care team can help you with this.',
    };
  }
  if (a.status === 'unsure') {
    return {
      title: 'We need more information before we can suggest a plan.',
      body: a.missing.length
        ? 'This is common. The tests listed below will help your care team decide what is best for you.'
        : 'Your results point in different directions. Your doctor will go over them and decide on next steps with you.',
    };
  }
  if (a.p >= 0.5) {
    return {
      title: weak ? 'Your heart is weaker than normal, but it has a good chance of getting stronger.' : 'Your heart is pumping well, and we want to keep it that way.',
      body: 'Staying away from alcohol and taking your medicines give your heart the best chance.',
    };
  }
  return {
    title: 'Your heart is weaker than normal and may need more than medicine to get stronger.',
    body: 'Your care team has a plan for you and will go over it with you. Staying away from alcohol still helps your heart.',
  };
}

// Ejection fraction bands from the American Heart Association.
function efZone(v) {
  if (v <= 40) return { key: 'low', word: 'Reduced' };
  if (v < 50) return { key: 'mid', word: 'Mildly reduced' };
  if (v <= 70) return { key: 'ok', word: 'Normal' };
  return { key: 'hi', word: 'Above normal' };
}

const EF_MAX = 80;
const efAt = (v) => `${(Math.min(Math.max(v, 0), EF_MAX) / EF_MAX) * 100}%`;

function EfScale({ value }) {
  const edge = value < 8 ? ' pt-ef-you-start' : value > 72 ? ' pt-ef-you-end' : '';
  return (
    <div className="pt-ef">
      <div className="pt-ef-track">
        <span className={`pt-ef-you${edge}`} style={{ left: efAt(value) }}>You: {value}%</span>
        <div className="pt-ef-bar" aria-hidden="true">
          <span className="pt-ef-seg low" style={{ width: efAt(40) }} />
          <span className="pt-ef-seg mid" style={{ width: `calc(${efAt(50)} - ${efAt(40)})` }} />
          <span className="pt-ef-seg ok" style={{ width: `calc(${efAt(70)} - ${efAt(50)})` }} />
          <span className="pt-ef-seg hi" style={{ flex: 1 }} />
        </div>
        <span className="pt-ef-pin" style={{ left: efAt(value) }} aria-hidden="true" />
        <div className="pt-ef-ticks" aria-hidden="true">
          {[0, 40, 50, 70].map((t) => <span key={t} style={{ left: efAt(t) }}>{t}%</span>)}
        </div>
      </div>
      <ul className="pt-ef-key">
        <li><i className="pt-ef-sw low" />Reduced: 40% or less</li>
        <li><i className="pt-ef-sw mid" />Mildly reduced: 41% to 49%</li>
        <li><i className="pt-ef-sw ok" />Normal: 50% to 70%</li>
      </ul>
    </div>
  );
}

function scarTag(rec) {
  if (!rec.lge_available) return 'Not checked';
  return rec.lge_status === 'NEGATIVE' ? 'No scar seen' : 'Scar seen';
}

function scarLine(rec) {
  if (!rec.lge_available) return 'Your MRI did not include the scan that looks for scar.';
  if (rec.lge_status === 'NEGATIVE') return 'Your MRI did not show any scar in your heart muscle. That is good news.';
  return 'Your MRI showed some scar in your heart muscle. Scar does not heal, so your care team will plan around it.';
}

function pethLevel(rec) {
  if (!rec.peth_available || rec.peth_value_ng_ml == null) return null;
  const band = pethBand(rec.peth_value_ng_ml);
  if (band.range === 'below 20') return 'low';
  if (band.range === '20 to 200') return 'middle';
  return 'high';
}

const PETH_TAG = { low: 'Low', middle: 'Middle range', high: 'High' };
const PETH_STEP = { low: 'Low', middle: 'Middle', high: 'High' };

function geneTag(value) {
  switch (value) {
    case 'PATHOGENIC':
    case 'LIKELY_PATHOGENIC':
      return 'Change found';
    case 'VARIANT_OF_UNCERTAIN_SIGNIFICANCE':
      return 'Unclear result';
    case 'NO_PATHOGENIC_OR_LIKELY_PATHOGENIC':
      return 'No change found';
    default:
      return 'Not done yet';
  }
}

function geneLine(value) {
  switch (value) {
    case 'PATHOGENIC':
    case 'LIKELY_PATHOGENIC':
      return 'The test found a change in the TTN gene that can weaken the heart muscle.';
    case 'VARIANT_OF_UNCERTAIN_SIGNIFICANCE':
      return 'The test found a change in the TTN gene, but doctors do not know yet if it matters. We are not treating it as the cause of your heart problem.';
    case 'NO_PATHOGENIC_OR_LIKELY_PATHOGENIC':
      return 'The test did not find a TTN gene change known to weaken the heart.';
    default:
      return 'This test has not been done yet.';
  }
}

function Tile({ icon, title, tag, children }) {
  return (
    <article className="pt-tile">
      <div className="pt-tile-head">
        <span className="pt-badge pt-badge-quiet">{icon}</span>
        <h3>{title}</h3>
      </div>
      <p><span className="pt-tag">{tag}</span></p>
      {children}
    </article>
  );
}

function Results({ rec }) {
  const level = pethLevel(rec);
  return (
    <section className="pt-block" aria-labelledby="pt-results">
      <SectionTitle id="pt-results" icon={<PulseIcon />}>Your other test results</SectionTitle>
      <div className="pt-tiles">
        <Tile icon={<ScanIcon />} title="Scar in your heart" tag={scarTag(rec)}>
          <p>{scarLine(rec)}</p>
        </Tile>
        <Tile icon={<DropIcon />} title="Alcohol blood test (PEth)" tag={level ? PETH_TAG[level] : 'Not done yet'}>
          {level && (
            <div className="pt-level" aria-hidden="true">
              {['low', 'middle', 'high'].map((l) => <span key={l} className={l === level ? 'on' : undefined}>{PETH_STEP[l]}</span>)}
            </div>
          )}
          <p>{level ? 'This test shows alcohol use over the past 2 to 4 weeks. It helps your care team see what may be affecting your heart.' : 'This test has not been done yet.'}</p>
        </Tile>
        <Tile icon={<DnaIcon />} title="Gene test (TTN)" tag={geneTag(rec.ttn_classification)}>
          <p>{geneLine(rec.ttn_classification)}</p>
        </Tile>
      </div>
    </section>
  );
}

function Steps({ steps }) {
  return (
    <ol className="pt-steps">
      {steps.map((s) => (
        <li key={s.title}>
          <span className="pt-when">{s.when}</span>
          <strong>{s.title}</strong>
          {s.detail && <p>{s.detail}</p>}
        </li>
      ))}
    </ol>
  );
}

function Plan({ rec, a }) {
  const plan = planFor(rec, a);
  const steps = [
    ...plan.procedures.map((p) => PROCEDURE_STEP[p.title]),
    ...plan.followUp.map((f) => FOLLOW_UP_STEP[f]),
  ].sort((x, y) => x.rank - y.rank);
  return (
    <section className="pt-block" aria-labelledby="pt-plan">
      <SectionTitle id="pt-plan" icon={<ClipboardIcon />}>Your plan</SectionTitle>
      <div className="pt-split">
        <article className="pt-card">
          <h3>Your heart medicines</h3>
          <ul className="pt-meds">
            {plan.medications.map((m) => (
              <li key={m.drug}>
                <span className="pt-badge pt-badge-quiet"><PillIcon /></span>
                <div>
                  <p><strong>{m.drug}</strong> <span className="pt-brand">({MEDICINE_INFO[m.drug].brand})</span></p>
                  <p>{MEDICINE_INFO[m.drug].why}</p>
                </div>
              </li>
            ))}
          </ul>
          <p className="pt-note">Your doses may go up slowly over time. Do not stop a medicine on your own.</p>
        </article>
        <article className="pt-card">
          <h3>What happens next</h3>
          <Steps steps={steps} />
        </article>
      </div>
    </section>
  );
}

function NextSteps({ a }) {
  let steps = [];
  if (a.status === 'unsure') steps = a.swings.map((s, i) => ({ when: i === 0 ? 'First' : 'Then', ...NEXT_TEST[s.key] }));
  if (a.status === 'ineligible') {
    steps = [{ when: 'First', title: 'Stop drinking safely', detail: 'Talk with your care team about how to stop drinking safely. They can set up the support you need.' }];
  }
  return (
    <section className="pt-block" aria-labelledby="pt-next">
      <SectionTitle id="pt-next" icon={<ArrowIcon />}>What happens next</SectionTitle>
      <article className="pt-card pt-card-narrow">
        {steps.length > 0 ? <Steps steps={steps} /> : <p>Your doctor will talk with you about your results and your choices.</p>}
        <p className="pt-note">Keep taking the medicines you already have, unless your doctor tells you to stop.</p>
      </article>
    </section>
  );
}

function SelfCare({ hasPlan }) {
  return (
    <section className="pt-block" aria-labelledby="pt-self">
      <SectionTitle id="pt-self" icon={<CheckCircleIcon />}>Taking care of yourself</SectionTitle>
      <div className="pt-split">
        <article className="pt-card pt-do">
          <h3>Do</h3>
          <ul className="pt-ticks">
            {hasPlan && <li><CheckIcon /><span>Take your medicines every day, even when you feel fine.</span></li>}
            <li><CheckIcon /><span>Weigh yourself every morning and write it down.</span></li>
            <li><CheckIcon /><span>Go to all of your visits and tests.</span></li>
          </ul>
        </article>
        <article className="pt-card pt-dont">
          <h3>Don't</h3>
          <ul className="pt-ticks">
            <li><CrossIcon /><span><strong>Do not drink any alcohol.</strong> This is the most important thing you can do for your heart.</span></li>
          </ul>
          <p className="pt-note">If you drink, or have strong urges to drink, tell your care team. They can help.</p>
        </article>
      </div>
    </section>
  );
}

function GetHelp() {
  return (
    <section className="pt-block" aria-labelledby="pt-help">
      <SectionTitle id="pt-help" icon={<PhoneIcon />}>When to get help</SectionTitle>
      <div className="pt-split">
        <article className="pt-care pt-care-911">
          <h3><PhoneIcon />Call 911 now if you have</h3>
          <ul className="pt-list">
            <li>Chest pain</li>
            <li>Severe trouble breathing, even when resting</li>
            <li>Fainting</li>
            <li>Symptoms that suddenly get much worse</li>
          </ul>
        </article>
        <article className="pt-care pt-care-team">
          <h3><AlertIcon />Call your care team today if you</h3>
          <ul className="pt-list">
            <li>Gain 2 to 3 pounds in a day, or 5 pounds in a week</li>
            <li>Have more swelling in your legs, ankles, feet or belly</li>
            <li>Get short of breath more easily than usual</li>
            <li>Have a new dry, hacking cough</li>
            <li>Feel confused, or have a heartbeat that feels very fast or uneven</li>
          </ul>
        </article>
      </div>
      <article className="pt-card pt-support">
        <h3>Support with alcohol, any time</h3>
        <p>If you start drinking heavily again, do not stop all at once on your own. Stopping suddenly can be dangerous. Call your care team first.</p>
        <dl className="pt-lines">
          <div>
            <dt>SAMHSA National Helpline</dt>
            <dd><a className="pt-call" href="tel:+18006624357"><PhoneIcon />1-800-662-4357</a></dd>
            <dd className="pt-note">Free and private, 24 hours a day, in English and Spanish.</dd>
          </div>
          <div>
            <dt>988 Lifeline</dt>
            <dd className="pt-calls">
              <a className="pt-call" href="tel:988"><PhoneIcon />Call 988</a>
              <a className="pt-call" href="sms:988"><MessageIcon />Text 988</a>
            </dd>
            <dd className="pt-note">Free crisis support, 24 hours a day.</dd>
          </div>
        </dl>
      </article>
    </section>
  );
}

export function PatientPage({ rec, a, preview, shared, onShare, onBack }) {
  const headline = message(rec, a);
  const hasPlan = a.status === 'plan';
  const ef = Math.round(rec.baseline_lvef);
  const zone = efZone(ef);
  const constructed = rec.record_type === 'CONSTRUCTED_MULTIMODAL_SCENARIO';
  return (
    <div className="pt">
      {preview && (
        <div className="pt-bar">
          <p>{shared ? 'Shared. The patient can see this page.' : 'Preview. The patient cannot see this until you share it.'}</p>
          <div className="pt-bar-actions">
            <button type="button" className="btn" onClick={onBack}>Back to chart</button>
            <button type="button" className="btn btn-primary" onClick={onShare} disabled={shared}>
              {shared ? 'Shared with patient' : 'Share with patient'}
            </button>
          </div>
        </div>
      )}
      <p className={constructed ? 'pt-research pt-research-strong' : 'pt-research'}>
        {constructed
          ? `Research prototype. Demo case: a real MRI scan (ACDC ${rec.acdc_id}) with a made-up health history. Not a real patient. Not medical advice.`
          : 'Research prototype. This patient is made up. This is not medical advice.'}
      </p>
      <article className="pt-page">
        <header className="pt-hero">
          <div className="pt-hero-text">
            <p className="pt-eyebrow"><HeartIcon />Your heart care summary</p>
            <h1>{headline.title}</h1>
            <p className="pt-lede">{headline.body}</p>
          </div>
          <section className="pt-pump" aria-labelledby="pt-pump">
            <h2 id="pt-pump">How well your heart pumps</h2>
            <p className="pt-pump-value">
              <span className="pt-figure">{ef}%</span>
              <span className={`pt-tag ${zone.key}`}>{zone.word}</span>
            </p>
            <EfScale value={ef} />
            <p className="pt-note">This is called your ejection fraction. It is how much blood your heart pumps out with each beat.</p>
          </section>
        </header>
        <Results rec={rec} />
        {hasPlan ? <Plan rec={rec} a={a} /> : <NextSteps a={a} />}
        <SelfCare hasPlan={hasPlan} />
        <GetHelp />
        <footer className="pt-foot">
          <p>Your own care team may give you different numbers or steps. Follow what they tell you.</p>
          {hasPlan && <p>Sample plan content. The treatment model is not built yet.</p>}
        </footer>
      </article>
    </div>
  );
}
