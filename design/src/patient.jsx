// The page a patient reads after their doctor shares it. Plain language, aimed at an 8th-grade
// reading level or below. It never shows a recovery percentage or a confidence score, and a
// "Not sure" result becomes next steps, because a patient cannot act on a model's doubts.
import { planFor, pethBand } from './model.js';

const MEDICINE_INFO = {
  'Sacubitril/valsartan': { brand: 'Entresto', why: 'Relaxes your blood vessels so your heart does not have to work as hard.' },
  'Metoprolol succinate': { brand: 'Toprol-XL', why: 'Slows your heartbeat and gives your heart muscle time to heal.' },
  Spironolactone: { brand: 'Aldactone', why: 'Helps your body get rid of extra fluid and protects your heart.' },
  Dapagliflozin: { brand: 'Farxiga', why: 'Helps your body clear extra fluid and sugar, which eases the strain on your heart.' },
};

const PROCEDURE_INFO = {
  'None now': 'You do not need a procedure right now. We will check your heart again after 3 months of medicine and no alcohol. If it is still weak, we may talk about a small device called an ICD.',
  'None indicated': 'You do not need a procedure right now.',
  'ICD evaluation': 'We would like you to see a heart rhythm doctor. They will talk with you about an ICD. This is a small device placed under the skin that can stop a dangerous heart rhythm.',
  'Advanced heart failure referral': 'We would also like you to see a heart failure specialist. They can talk with you about stronger treatments if your symptoms do not get better. These include a heart pump or a heart transplant.',
};

const FOLLOW_UP_INFO = {
  'Repeat cardiac MRI at 6 months': 'A heart MRI in 6 months to see if your heart is getting stronger',
  'PEth monthly to confirm abstinence': 'A PEth blood test each month to help you stay on track',
  'Addiction medicine follow-up': 'Visits with an alcohol treatment doctor for support',
  'Genetic counseling and family screening': 'A visit with a genetic counselor to learn what your gene result means for you and your family',
};

const NEXT_TEST = {
  peth: { title: 'PEth blood test', why: 'Shows if alcohol has been in your body in the past few weeks.' },
  ttn: { title: 'Gene test', why: 'Looks for a change in the TTN gene that can weaken the heart.' },
  lge: { title: 'Heart MRI with contrast dye', why: 'Shows if there is any scar in your heart muscle.' },
};

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

function scarLine(rec) {
  if (!rec.lge_available) return 'Your MRI did not include the scan that looks for scar.';
  if (rec.lge_status === 'NEGATIVE') return 'Your MRI did not show any scar in your heart muscle. That is good news.';
  return 'Your MRI showed some scar in your heart muscle. Scar does not heal, so your care team will plan around it.';
}

function pethLine(rec) {
  if (!rec.peth_available || rec.peth_value_ng_ml == null) return null;
  const band = pethBand(rec.peth_value_ng_ml);
  if (band.range === 'below 20') return 'Your level was low.';
  if (band.range === '20 to 200') return 'Your level was in the middle range.';
  return 'Your level was high.';
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

function HeartStrength({ value }) {
  const scale = 80;
  return (
    <div className="pt-meter" role="img" aria-label={`Your heart pumps out about ${value} percent of its blood with each beat. Healthy hearts pump out about 50 to 70 percent.`}>
      <div className="pt-meter-track">
        <span className="pt-meter-fill" style={{ width: `${(value / scale) * 100}%` }} />
        <span className="pt-meter-range" style={{ left: `${(50 / scale) * 100}%`, width: `${(20 / scale) * 100}%` }} />
      </div>
      <div className="pt-meter-labels">
        <span style={{ width: `${(50 / scale) * 100}%` }} />
        <span style={{ width: `${(20 / scale) * 100}%` }}>Healthy range</span>
      </div>
    </div>
  );
}

function Results({ rec }) {
  const ef = Math.round(rec.baseline_lvef);
  const peth = pethLine(rec);
  return (
    <section className="pt-section">
      <h2>What your tests showed</h2>
      <div className="pt-result">
        <h3>How well your heart pumps</h3>
        <p className="pt-figure">{ef}%</p>
        <HeartStrength value={ef} />
        <p>This is called your ejection fraction. It is how much blood your heart pumps out with each beat. Healthy hearts pump out about 50% to 70%.</p>
        <p>{scarLine(rec)}</p>
      </div>
      <div className="pt-result">
        <h3>Alcohol blood test (PEth)</h3>
        {peth ? <p><strong>{peth}</strong> This test shows alcohol use over the past 2 to 4 weeks. It helps your care team see what may be affecting your heart.</p> : <p>This test has not been done yet.</p>}
      </div>
      <div className="pt-result">
        <h3>Gene test (TTN)</h3>
        <p>{geneLine(rec.ttn_classification)}</p>
      </div>
    </section>
  );
}

function Plan({ rec, a }) {
  const plan = planFor(rec, a);
  return (
    <section className="pt-section">
      <h2>Your plan</h2>
      <h3>Medicines</h3>
      <ul className="pt-medicines">
        {plan.medications.map((m) => (
          <li key={m.drug}>
            <strong>{m.drug}</strong> <span>({MEDICINE_INFO[m.drug].brand})</span>
            <p>{MEDICINE_INFO[m.drug].why}</p>
          </li>
        ))}
      </ul>
      <h3>Procedures</h3>
      {plan.procedures.map((p) => (
        <p key={p.title}>{PROCEDURE_INFO[p.title]}</p>
      ))}
      <h3>Follow-up</h3>
      <ul className="pt-list">
        {plan.followUp.map((f) => (
          <li key={f}>{FOLLOW_UP_INFO[f]}</li>
        ))}
      </ul>
    </section>
  );
}

function NextSteps({ a }) {
  return (
    <section className="pt-section">
      <h2>What happens next</h2>
      {a.status === 'unsure' && a.swings.length > 0 && (
        <ul className="pt-medicines">
          {a.swings.map((s) => (
            <li key={s.key}>
              <strong>{NEXT_TEST[s.key].title}</strong>
              <p>{NEXT_TEST[s.key].why}</p>
            </li>
          ))}
        </ul>
      )}
      {a.status === 'unsure' && a.swings.length === 0 && <p>Your doctor will talk with you about your results and your choices.</p>}
      {a.status === 'ineligible' && <p>Talk with your care team about how to stop drinking safely. They can set up the support you need.</p>}
      <p>Keep taking the medicines you already have, unless your doctor tells you to stop.</p>
    </section>
  );
}

function SelfCare({ hasPlan }) {
  return (
    <section className="pt-section">
      <h2>What you can do</h2>
      <ul className="pt-list">
        <li><strong>Do not drink any alcohol.</strong> This is the most important thing you can do for your heart.</li>
        {hasPlan && <li>Take your medicines every day, even when you feel fine.</li>}
        <li>Weigh yourself every morning and write it down.</li>
        <li>Go to all of your visits and tests.</li>
      </ul>
    </section>
  );
}

function GetHelp() {
  return (
    <section className="pt-section pt-help">
      <h2>When to get help</h2>
      <div className="pt-urgent">
        <h3>Call 911 if you have</h3>
        <ul className="pt-list">
          <li>Chest pain</li>
          <li>Severe trouble breathing</li>
          <li>Fainting</li>
          <li>Symptoms that suddenly get much worse</li>
        </ul>
      </div>
      <h3>Call your care team if you</h3>
      <ul className="pt-list">
        <li>Gain 2 to 3 pounds in a day, or 5 pounds in a week</li>
        <li>Have more swelling in your legs, ankles or feet</li>
        <li>Get short of breath more easily than usual</li>
        <li>Feel confused, or have a heartbeat that feels very fast or uneven</li>
        <li>Drink alcohol again, or feel you might</li>
      </ul>
      <p>If you start drinking heavily again, do not stop all at once on your own. Stopping suddenly can be dangerous. Call your care team first.</p>
      <h3>Support with alcohol, any time</h3>
      <dl className="pt-lines">
        <div>
          <dt>SAMHSA National Helpline</dt>
          <dd>1-800-662-4357</dd>
          <p>Free and private, 24 hours a day, in English and Spanish.</p>
        </div>
        <div>
          <dt>988 Lifeline</dt>
          <dd>Call or text 988</dd>
          <p>Free crisis support, 24 hours a day.</p>
        </div>
      </dl>
    </section>
  );
}

export function PatientPage({ rec, a, preview, shared, onShare, onBack }) {
  const headline = message(rec, a);
  const hasPlan = a.status === 'plan';
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
      <p className="pt-research">Research prototype. This patient is made up. This is not medical advice.</p>
      <article className="pt-page">
        <header className="pt-head">
          <p className="pt-eyebrow">Your heart care summary</p>
          <h1>{headline.title}</h1>
          <p>{headline.body}</p>
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
