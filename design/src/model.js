// Scoring, confidence and plan logic for the prototype.
// predict() is the real fitted model (v1.0-m3b) running in the browser. confidence and
// planFor() are stand-ins until the treatment model exists: replace those two only.
import MODEL from './data/model.json';
import COHORT from './data/cohort.json';

export const modelVersion = MODEL.version;
export const DEFAULT_THRESHOLD = 85;

export const patients = COHORT.rows.map((row) =>
  Object.fromEntries(COHORT.columns.map((col, i) => [col, row[i]])),
);

export const demoPatients = Object.values(COHORT.demo);

const DRIVER_GROUP = {
  peth_value_ng_ml: 'peth', peth_available: 'peth',
  lge_status: 'lge', lge_pattern: 'lge', lge_extent_percent: 'lge', lge_available: 'lge',
  ttn_classification: 'ttn',
  baseline_lvef: 'lv', edv_ml: 'lv', esv_ml: 'lv', stroke_volume_ml: 'lv', myocardial_mass_g: 'lv',
  rvedv_ml: 'rv', rvef_percent: 'rv',
  reported_drinks_per_week: 'history', time_since_last_drink_days: 'history',
  years_heavy_drinking: 'history', alcohol_history_available: 'history',
  age: 'age', biological_sex: 'age',
};

export function predict(rec) {
  let z = MODEL.intercept;
  const groups = {};
  const add = (feature, contribution, phi) => {
    z += contribution;
    const g = DRIVER_GROUP[feature];
    groups[g] = (groups[g] ?? 0) + phi;
  };
  for (const n of MODEL.numeric) {
    const raw = rec[n.f];
    const x = ((raw == null ? n.median : raw) - n.mean) / n.scale;
    add(n.f, n.coef * x, n.coef * (x - n.xbar));
  }
  for (const c of MODEL.categorical) {
    const value = String(rec[c.f]);
    for (const [level, { coef, xbar }] of Object.entries(c.levels)) {
      const on = level === value ? 1 : 0;
      add(c.f, coef * on, coef * (on - xbar));
    }
  }
  for (const b of MODEL.binary) {
    const x = rec[b.f] ? 1 : 0;
    add(b.f, b.coef * x, b.coef * (x - b.xbar));
  }
  return { p: 1 / (1 + Math.exp(-z)), groups };
}

// The same best-case and worst-case stand-ins Streamlit's Recovery Estimate page uses, with
// one change: only results that are really missing get replaced, never observed ones.
const BEST = {
  peth: { peth_value_ng_ml: 450, peth_available: 1 },
  ttn: { ttn_classification: 'NO_PATHOGENIC_OR_LIKELY_PATHOGENIC' },
  lge: { lge_status: 'NEGATIVE', lge_pattern: 'NONE', lge_extent_percent: 0, lge_available: 1 },
};
const WORST = {
  peth: { peth_value_ng_ml: 15, peth_available: 1 },
  ttn: { ttn_classification: 'PATHOGENIC' },
  lge: { lge_status: 'POSITIVE', lge_pattern: 'MIDWALL', lge_extent_percent: 15, lge_available: 1 },
};

export const MISSING_ACTION = {
  peth: { label: 'PEth', order: 'Draw PEth' },
  ttn: { label: 'TTN', order: 'Order TTN sequencing' },
  lge: { label: 'LGE', order: 'Get an LGE sequence' },
};

function missingResults(rec) {
  const missing = [];
  if (!rec.peth_available || rec.peth_value_ng_ml == null) missing.push('peth');
  if (rec.ttn_classification === 'NOT_TESTED') missing.push('ttn');
  if (!rec.lge_available) missing.push('lge');
  return missing;
}

const scoreCache = new WeakMap();

// Everything about a patient that does not depend on the threshold.
export function score(rec) {
  if (scoreCache.has(rec)) return scoreCache.get(rec);
  const { p, groups } = predict(rec);
  const missing = missingResults(rec);
  const fill = (table, keys) => ({ ...rec, ...Object.assign({}, ...keys.map((k) => table[k])) });
  const low = missing.length ? predict(fill(WORST, missing)).p : p;
  const high = missing.length ? predict(fill(BEST, missing)).p : p;
  // Prototype confidence: distance from a coin flip, capped by how far missing results could move the answer.
  const confidence = Math.min(Math.max(p, 1 - p), 1 - (high - low));
  const swings = missing
    .map((key) => ({
      key,
      swing: predict(fill(BEST, [key])).p - predict(fill(WORST, [key])).p,
    }))
    .sort((a, b) => b.swing - a.swing);
  const drivers = Object.entries(groups)
    .map(([key, phi]) => ({ key, phi }))
    .sort((a, b) => Math.abs(b.phi) - Math.abs(a.phi));
  const result = { p, low, high, confidence, missing, swings, drivers };
  scoreCache.set(rec, result);
  return result;
}

export function assess(rec, threshold) {
  const scored = score(rec);
  const eligible = rec.assumed_abstinence_scenario === 'SUSTAINED_ABSTINENCE';
  const status = !eligible ? 'ineligible' : scored.confidence * 100 >= threshold ? 'plan' : 'unsure';
  return { ...scored, status };
}

export const STATUS_LABEL = {
  plan: 'Plan ready',
  unsure: 'Not sure',
  ineligible: 'Not eligible',
};

export const TTN_LABEL = {
  PATHOGENIC: 'Pathogenic',
  LIKELY_PATHOGENIC: 'Likely pathogenic',
  VARIANT_OF_UNCERTAIN_SIGNIFICANCE: 'Uncertain significance (VUS)',
  NO_PATHOGENIC_OR_LIKELY_PATHOGENIC: 'No pathogenic variant',
  NOT_TESTED: 'Not tested',
};
export const TTN_ORDER = [
  'PATHOGENIC', 'LIKELY_PATHOGENIC', 'VARIANT_OF_UNCERTAIN_SIGNIFICANCE',
  'NO_PATHOGENIC_OR_LIKELY_PATHOGENIC', 'NOT_TESTED',
];
export const LGE_PATTERN_LABEL = {
  NONE: 'None', MIDWALL: 'Midwall', SUBEPICARDIAL: 'Subepicardial',
  SUBENDOCARDIAL: 'Subendocardial', PATCHY_TRANSMURAL: 'Patchy transmural',
};

export function pethBand(v) {
  if (v == null) return null;
  if (v < 20) return { label: 'Abstinent or light use', range: 'below 20', flag: null };
  if (v <= 200) return { label: 'Moderate use', range: '20 to 200', flag: 'H' };
  return { label: 'Heavy use', range: 'above 200', flag: 'H' };
}

export function efBand(v) {
  if (v == null) return null;
  if (v <= 35) return { label: 'In the ICD range', flag: 'L' };
  if (v < 50) return { label: 'Reduced', flag: 'L' };
  return { label: 'Preserved', flag: null };
}

export const pct = (x) => {
  const v = x * 100;
  if (v < 1) return '<1%';
  if (v > 99) return '>99%';
  return `${Math.round(v)}%`;
};

export const sexWord = (s) => (s === 'FEMALE' ? 'F' : s === 'MALE' ? 'M' : '');

export function driverText(key, rec) {
  switch (key) {
    case 'lge':
      if (!rec.lge_available) return 'LGE not performed';
      if (rec.lge_status === 'NEGATIVE') return 'No scar on LGE';
      return `${LGE_PATTERN_LABEL[rec.lge_pattern] ?? 'Positive'} scar, ${rec.lge_extent_percent ?? '?'}% of LV mass`;
    case 'ttn':
      return `TTN: ${TTN_LABEL[rec.ttn_classification].toLowerCase()}`;
    case 'peth':
      return rec.peth_available && rec.peth_value_ng_ml != null
        ? `PEth ${rec.peth_value_ng_ml} ng/mL`
        : 'PEth not done';
    case 'lv':
      return `LVEF ${rec.baseline_lvef}%, LV EDV ${Math.round(rec.edv_ml)} mL`;
    case 'rv':
      return rec.rvef_percent == null ? 'RV function not measured' : `RVEF ${rec.rvef_percent}%`;
    case 'history': {
      if (rec.reported_drinks_per_week == null) return 'No alcohol history';
      const years = rec.years_heavy_drinking == null ? '' : `, ${rec.years_heavy_drinking} years`;
      return `${Math.round(rec.reported_drinks_per_week)} drinks a week${years}`;
    }
    case 'age':
      return `Age ${rec.age}, ${rec.biological_sex === 'FEMALE' ? 'female' : 'male'}`;
    default:
      return key;
  }
}

// Illustrative content for the plan slot. No model produces this yet. The four drug classes
// are standard HFrEF therapy, the same for nearly everyone here, because the model has no
// blood pressure, kidney function, potassium, NYHA class or QRS duration to choose on.
export function planFor(rec, a) {
  const ef = rec.baseline_lvef;
  const likely = a.p >= 0.5;
  const pathogenicTtn = ['PATHOGENIC', 'LIKELY_PATHOGENIC'].includes(rec.ttn_classification);

  const medications = [
    { cls: 'ARNI', drug: 'Sacubitril/valsartan' },
    { cls: 'Beta blocker', drug: 'Metoprolol succinate' },
    { cls: 'MRA', drug: 'Spironolactone' },
    { cls: 'SGLT2 inhibitor', drug: 'Dapagliflozin' },
  ];

  const procedures = [];
  if (ef <= 35 && !likely) {
    procedures.push({
      title: 'ICD evaluation',
      detail: 'Refer to electrophysiology. LVEF is at or below 35% and is unlikely to recover.',
    });
    if (ef <= 25) {
      procedures.push({
        title: 'Advanced heart failure referral',
        detail: 'LVAD or transplant evaluation if symptoms persist on medical therapy.',
      });
    }
  } else if (ef <= 35) {
    procedures.push({
      title: 'None now',
      detail: 'Recheck LVEF after 3 months of therapy and abstinence. ICD evaluation if it stays at or below 35%.',
    });
  } else {
    procedures.push({ title: 'None indicated', detail: 'LVEF is above the device range.' });
  }

  const followUp = ['Repeat cardiac MRI at 6 months', 'PEth monthly to confirm abstinence', 'Addiction medicine follow-up'];
  if (pathogenicTtn) followUp.push('Genetic counseling and family screening');

  return { medications, procedures, followUp };
}

const num = (v) => (v === '' || v == null ? null : Number(v));

// Form values to a model record. Volumes are entered; LVEF, RVEF and stroke volume are computed.
export function toRecord(form, id) {
  const edv = num(form.edv_ml);
  const esv = num(form.esv_ml);
  const rvedv = num(form.rvedv_ml);
  const rvesv = num(form.rvesv_ml);
  const lgeDone = form.lge_status === 'NEGATIVE' || form.lge_status === 'POSITIVE';
  const hasHistory = [form.reported_drinks_per_week, form.time_since_last_drink_days, form.years_heavy_drinking].some((v) => v !== '');
  const round1 = (v) => Math.round(v * 10) / 10;
  return {
    patient_id: id,
    age: num(form.age),
    biological_sex: form.biological_sex,
    edv_ml: edv,
    esv_ml: esv,
    stroke_volume_ml: edv != null && esv != null ? round1(edv - esv) : null,
    baseline_lvef: edv && esv != null ? round1(((edv - esv) / edv) * 100) : null,
    rvedv_ml: rvedv,
    rvesv_ml: rvesv,
    rvef_percent: rvedv && rvesv != null ? round1(((rvedv - rvesv) / rvedv) * 100) : null,
    myocardial_mass_g: num(form.myocardial_mass_g),
    peth_value_ng_ml: form.peth_status === 'DONE' ? num(form.peth_value_ng_ml) : null,
    peth_available: form.peth_status === 'DONE' ? 1 : 0,
    peth_measurement_date: form.peth_measurement_date || null,
    reported_drinks_per_week: num(form.reported_drinks_per_week),
    time_since_last_drink_days: num(form.time_since_last_drink_days),
    years_heavy_drinking: num(form.years_heavy_drinking),
    alcohol_history_available: hasHistory ? 1 : 0,
    ttn_classification: form.ttn_classification || 'NOT_TESTED',
    lge_status: lgeDone ? form.lge_status : 'NOT_PERFORMED',
    lge_pattern: form.lge_status === 'POSITIVE' ? form.lge_pattern : 'NONE',
    lge_extent_percent: form.lge_status === 'POSITIVE' ? num(form.lge_extent_percent) : lgeDone ? 0 : null,
    lge_available: lgeDone ? 1 : 0,
    assumed_abstinence_scenario: 'SUSTAINED_ABSTINENCE',
    follow_up_lvef: null,
    delta_lvef: null,
    recovery_category: 'PENDING_EVALUATION',
    adverse_cardiac_events: 'NONE',
  };
}

export const EMPTY_FORM = {
  age: '', biological_sex: '', withdrawal: false,
  peth_status: '', peth_value_ng_ml: '', peth_measurement_date: '',
  reported_drinks_per_week: '', time_since_last_drink_days: '', years_heavy_drinking: '',
  edv_ml: '', esv_ml: '', rvedv_ml: '', rvesv_ml: '', myocardial_mass_g: '',
  lge_status: '', lge_pattern: 'MIDWALL', lge_extent_percent: '',
  ttn_classification: '',
};

const text = (v) => (v == null ? '' : String(v));

export function toForm(rec) {
  return {
    ...EMPTY_FORM,
    age: text(rec.age),
    biological_sex: rec.biological_sex,
    withdrawal: true,
    peth_status: rec.peth_available ? 'DONE' : 'NOT_DONE',
    peth_value_ng_ml: text(rec.peth_value_ng_ml),
    peth_measurement_date: text(rec.peth_measurement_date),
    reported_drinks_per_week: text(rec.reported_drinks_per_week),
    time_since_last_drink_days: text(rec.time_since_last_drink_days),
    years_heavy_drinking: text(rec.years_heavy_drinking),
    edv_ml: text(rec.edv_ml),
    esv_ml: text(rec.esv_ml),
    rvedv_ml: text(rec.rvedv_ml),
    rvesv_ml: text(rec.rvesv_ml),
    myocardial_mass_g: text(rec.myocardial_mass_g),
    lge_status: rec.lge_available ? rec.lge_status : 'NOT_PERFORMED',
    lge_pattern: rec.lge_pattern === 'NONE' ? 'MIDWALL' : rec.lge_pattern,
    lge_extent_percent: rec.lge_available && rec.lge_status === 'POSITIVE' ? text(rec.lge_extent_percent) : '',
    ttn_classification: rec.ttn_classification,
  };
}

// The whole intake form is this list. New or changed fields go here and every layout follows.
// Ranges are the Pydantic limits in src/crip/synthetic/schemas.py.
export const SECTIONS = [
  {
    id: 'patient',
    title: 'Patient',
    fields: [
      { key: 'age', label: 'Age', unit: 'years', min: 18, max: 105, required: true },
      { key: 'biological_sex', label: 'Sex', kind: 'choice', required: true, options: [['MALE', 'Male'], ['FEMALE', 'Female']] },
      { key: 'withdrawal', label: 'Alcohol withdrawal complete', kind: 'check', required: true, note: 'Required. The plan assumes this.' },
    ],
  },
  {
    id: 'peth',
    title: 'PEth',
    hint: 'Reflects drinking over the last 2 to 4 weeks.',
    fields: [
      { key: 'peth_status', label: 'PEth test', kind: 'choice', required: true, options: [['DONE', 'Result available'], ['NOT_DONE', 'Not done']] },
      { key: 'peth_value_ng_ml', label: 'PEth', unit: 'ng/mL', min: 0, max: 5000, required: true, when: (f) => f.peth_status === 'DONE', range: 'Below 20 light, 20 to 200 moderate, above 200 heavy', flag: 'peth' },
      { key: 'peth_measurement_date', label: 'Drawn on', kind: 'date', when: (f) => f.peth_status === 'DONE' },
      { key: 'time_since_last_drink_days', label: 'Days since last drink', unit: 'days', min: 0, max: 365, more: true },
      { key: 'reported_drinks_per_week', label: 'Drinks per week', unit: 'drinks', min: 0, max: 200, more: true },
      { key: 'years_heavy_drinking', label: 'Years of heavy drinking', unit: 'years', min: 0, max: 70, more: true },
    ],
  },
  {
    id: 'mri',
    title: 'Cardiac MRI',
    hint: 'Enter volumes. LVEF and RVEF are calculated for you.',
    upload: true,
    fields: [
      { key: 'edv_ml', label: 'LV end-diastolic volume', short: 'LV EDV', unit: 'mL', min: 40, max: 600, required: true },
      { key: 'esv_ml', label: 'LV end-systolic volume', short: 'LV ESV', unit: 'mL', min: 15, max: 500, required: true },
      { derived: 'baseline_lvef', label: 'LVEF', unit: '%' },
      { key: 'rvedv_ml', label: 'RV end-diastolic volume', short: 'RV EDV', unit: 'mL', min: 30, max: 600 },
      { key: 'rvesv_ml', label: 'RV end-systolic volume', short: 'RV ESV', unit: 'mL', min: 10, max: 500 },
      { derived: 'rvef_percent', label: 'RVEF', unit: '%' },
      { key: 'myocardial_mass_g', label: 'LV mass', unit: 'g', min: 40, max: 500 },
      { key: 'lge_status', label: 'Scar on LGE', kind: 'choice', required: true, options: [['NEGATIVE', 'No scar'], ['POSITIVE', 'Scar present'], ['NOT_PERFORMED', 'Not performed']] },
      { key: 'lge_pattern', label: 'Scar pattern', kind: 'select', when: (f) => f.lge_status === 'POSITIVE', options: Object.entries(LGE_PATTERN_LABEL).filter(([k]) => k !== 'NONE') },
      { key: 'lge_extent_percent', label: 'Scar extent', unit: '% of LV mass', min: 0, max: 50, wide: true, when: (f) => f.lge_status === 'POSITIVE', required: true },
    ],
  },
  {
    id: 'ttn',
    title: 'TTN genetics',
    hint: 'A VUS is never treated as a pathogenic result.',
    fields: [
      { key: 'ttn_classification', label: 'TTN result', kind: 'choice', required: true, options: TTN_ORDER.map((k) => [k, k === 'VARIANT_OF_UNCERTAIN_SIGNIFICANCE' ? 'VUS' : TTN_LABEL[k]]) },
    ],
  },
];

export function fieldError(field, form) {
  const value = form[field.key];
  if (field.min == null || value === '') return null;
  const n = Number(value);
  if (Number.isNaN(n)) return 'Enter a number.';
  if (n < field.min || n > field.max) return `Enter a value from ${field.min} to ${field.max}.`;
  if (field.key === 'esv_ml' && form.edv_ml !== '' && n >= Number(form.edv_ml)) return 'Must be less than end-diastolic volume.';
  if (field.key === 'rvesv_ml' && form.rvedv_ml !== '' && n >= Number(form.rvedv_ml)) return 'Must be less than end-diastolic volume.';
  return null;
}

// What still blocks a plan, in form order.
export function readiness(form) {
  const needed = [];
  let errors = 0;
  for (const section of SECTIONS) {
    for (const field of section.fields) {
      if (field.derived || (field.when && !field.when(form))) continue;
      if (fieldError(field, form)) errors += 1;
      const empty = form[field.key] === '' || form[field.key] === false;
      if (field.required && empty) needed.push(field.label);
    }
  }
  return { needed, errors, ready: needed.length === 0 && errors === 0 };
}

export function cohortFacts() {
  const eligible = patients.filter((p) => p.assumed_abstinence_scenario === 'SUSTAINED_ABSTINENCE').length;
  return { total: patients.length, eligible };
}
