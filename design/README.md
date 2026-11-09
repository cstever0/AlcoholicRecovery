# CRIP physician dashboard prototype

A clickable prototype of the physician dashboard and the page a patient sees after their doctor shares it. It runs entirely in the browser: 3,000 synthetic patients, scored live by the same logistic regression the repo trains (`models/recovery`, version v1.0-m3b).

Research prototype. Synthetic patients only. Not for clinical use.

## Open the demo

Double-click **`design/CRIP Physician Dashboard.html`**. Nothing to install, no server. It works offline; only the fonts need a connection.

To share it without the repo, send that one HTML file.

## Walk through the demo

The **Demo** tab in the patient list holds six patients chosen to show every state the dashboard can be in. They come from the validation and test splits, so the model never trained on them. Within each state, `export_data.py` picks the patient whose LVEF is closest to that group's median, so each demo shows a typical heart. Click through them in order, or use the ← and → keys.

| # | Patient | LVEF | What it shows |
|---|---------|------|---------------|
| 1 | CRIP-SYN-02677 | 30% | Recovery likely: toxic picture, no scar, no TTN variant |
| 2 | CRIP-SYN-00554 | 31% | Recovery unlikely: TTN variant with scar |
| 3 | CRIP-SYN-02823 | 28% | Not sure: TTN and LGE missing. Shows which test to order first and **Add results** |
| 4 | CRIP-SYN-01241 | 28% | Not sure: complete data, model on the fence |
| 5 | CRIP-SYN-02679 | 36% | VUS handled as uncertain, not pathogenic |
| 6 | CRIP-SYN-00871 | 25% | Severe LV dysfunction, recovery unlikely (device referral) |

Every patient has a weak heart on purpose: the cohort simulates alcohol-associated cardiomyopathy, so baseline LVEF runs from 16% to 44% (median 30%) and nobody is in the 50–70% normal range.

For each one:

- **The chart** shows the findings and the recommendation (a plan, or "Not sure" when confidence is below the 85% threshold). **All values**, **MRI images** and **What if** open in a side panel; What if rescores the patient as you change the PEth, TTN or scar result.
- **Patient view** previews the plain-language page for the patient. **Share with patient** marks it shared; **Back to chart** returns.
- **Models** (top bar) shows how the recovery model scores on the validation split, the confidence threshold slider, calibration, and the MRI model's accuracy on ACDC and on other hospitals' scanners (M&Ms).
- **New patient** (or the `n` key) enters a patient and scores them. `/` jumps to search.
- **Design options** (bottom right) switches the chart layout and the new-patient form, and sets the confidence threshold, text size (Projector for presenting) and theme.

Direct links: add `#models`, `#new`, `#CRIP-SYN-02677` (a chart) or `#patient-view.CRIP-SYN-02677` (the patient page on its own, as the patient would see it) to the end of the file's address.

### Real scans tab

Three demo cases that pair a **real MRI** with a **made-up health history**: Cory's `CRIP-DEMO-CASE-A/B/C` from `src/crip/integration/demo_cases.py`, built on ACDC patient001–003 (all dilated cardiomyopathy). They show the pipeline running end to end: real scan, AI-measured LVEF, recovery model, patient page.

- Every value carries a source tag: **Real MRI**, **Simulated**, or **Waiting for export**. A banner on the chart and on the patient page says the case is not a real patient with this disease and the outcome is simulated.
- They are kept out of the 3,000-patient cohort, the counts and every metric on the Models page. They are not evidence that the model works on real patients.
- Today only the LVEF values (23.7%, 29.1%, 12.9%, from our U-Net on these training cases) and patient001's image are in the repo. The LV/RV volumes, mass and the other two images need the ACDC scans, so until someone runs the export the model fills the volumes with its training medians. To complete them, on a machine with the ACDC data, from the repo root with `.venv` active:

```bash
python design/export_real_cases.py
cd design && npm run build
```

ACDC images: Bernard et al., IEEE TMI 2018, CC BY-NC-SA 4.0, non-commercial research use only (`docs/acdc/LICENSE_TERMS.md`).

## What is real and what is sample content

| Real, from this repo | Sample or placeholder |
|---|---|
| The 3,000 synthetic patients (`crip.synthetic.cohort_generator`, seed 42) | Treatment plan text: no treatment model exists yet |
| Recovery probabilities, computed from the trained model's own coefficients | The 85% confidence threshold |
| Validation metrics, tradeoff and calibration charts | The MRI image (ACDC patient001, not the selected patient) |
| MRI accuracy figures (copied from the README and the M&Ms report) | Patient names, doctors and dates: there are none |

The newer "empirically informed" cohort (`python -m crip.pipeline synthesize --empirical`) is not used. It needs the raw ACDC, M&Ms and EMIDEC scans, which are not in the repo, and its model files are not committed.

## Change the prototype

Source is in `design/src`; the HTML file is built from it.

```bash
cd design
npm install
npm run build
```

`npm run build` rewrites `CRIP Physician Dashboard.html` and `dist/artifact.html`. Commit the rebuilt HTML with your source changes so the double-click file stays current.

## Refresh the data

`design/src/data/*.json` is a snapshot of the cohort and the recovery model. Rerun this after either one changes. From the repo root, with `.venv` active (setup is in the main README):

```bash
python -m crip.synthetic.cohort_generator --n-patients 3000 --seed 42 --abstinence-prob 0.70 --output-dir data/synthetic
python -m crip.modeling.train
python design/export_data.py
cd design && npm run build
```

The first two commands create the gitignored cohort CSV and `models/recovery/recovery_pipeline.joblib`. They are seeded, so on a fresh clone the export matches the committed JSON byte for byte (checked on 2026-11-08 at commit fc9984e). If the export prints different demo patients, the cohort or model changed: update the table above.
