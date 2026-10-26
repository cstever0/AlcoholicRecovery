# Milestone 4 Scientific & Clinical Data Audit Report
## Critical Examination of Synthetic Assumptions, Data Contracts, and Evaluator Integrity

**Project:** Cardiomyopathy Recovery Intelligence Platform (CRIP)  
**Milestone:** 4 — Final Application Integration & End-to-End Validation  
**Date:** October 2026  
**Status:** Audit Complete  

---

> [!WARNING]
> ### CLINICAL RESEARCH DISCLAIMER — NOT FOR CLINICAL USE
> **CRIP is an academic research proof of concept. The predictive models, simulated trajectories, and clinical recovery labels documented herein are derived from synthetic simulation and MUST NEVER be used for clinical diagnosis, patient risk stratification, or clinical treatment decisions. Phosphatidylethanol (PEth) quantification measures recent ethanol exposure (preceding 2–4 weeks); it does not alone diagnose alcoholic cardiomyopathy, does not substitute for comprehensive clinical evaluation, and does not predict the severity of alcohol withdrawal syndrome.**

---

## 1. Audit Overview & Objectives

Before finalizing the end-to-end integrated application, an exhaustive audit was performed across five scientific and data integrity dimensions:
1. **Discrepancy in Cohort Size:** Resolving the difference between the 2,123 abstinent patients reported in Milestone 3A narrative versus the 2,095 eligible modeling patients in Milestone 3B.
2. **Synthetic Model Performance:** Examining why test ROC-AUC reached 0.9565 and contextualizing simulation optimism.
3. **Biomarker Interpretation (PEth):** Documenting simulator-specific positive recovery weighting versus real-world prognostic nuance.
4. **Test-Set Integrity:** Verifying that held-out test data remained uncompromised throughout model selection and evaluation.
5. **Genetics (*TTN*) and CMR Tissue (*LGE*) Boundaries:** Enforcing strict boundaries between real ACDC cine MRI and synthetic LGE/genetic variables.

---

## 2. Issue 1: Synthetic Cohort Discrepancy (2,123 vs. 2,095)

### Findings & Root Cause Analysis
- **Milestone 3A Narrative Report:** Mentioned *2,123* simulated abstinent patients ($70.8\%$ of $3,000$) with a stated recovery rate of $78.3\%$ ($1,661 / 2,123$).
- **Milestone 3B Modeling Dataset:** Loaded `data/synthetic/synthetic_cardiomyopathy_cohort.csv` and filtered by `assumed_abstinence_scenario == 'SUSTAINED_ABSTINENCE'`, yielding exactly **2,095** patients with **1,641** recoveries ($1,641 / 2,095 = 78.33\%$).

```text
================================================================================
                    COHORT GROUND-TRUTH RECONCILIATION
================================================================================
Data Entity                        Reported in M3A Text     Actual Ground-Truth on Disk
--------------------------------------------------------------------------------
Total Synthetic Records            3,000                   3,000
Sustained Abstinence Subgroup      2,123 (70.8%)           2,095 (69.83%)
Continued Heavy Drinking Subgroup  877 (29.2%)             905 (30.17%)
Recovered Patients in Abstinent    1,661                   1,641
Recovery Rate in Abstinent         78.24%                  78.33%
Invalid / Dropped Records          0                       0
================================================================================
```

### Explanation:
The discrepancy was identified as a **narrative documentation discrepancy** from an exploratory preliminary generation draft during Milestone 3A drafting, whereas the actual serialized CSV file produced by `SyntheticCohortGenerator(seed=42)` and committed to disk contained exactly **2,095** abstinent patients and **905** continued heavy drinkers (an exact $69.83\% / 30.17\%$ split from the Bernoulli trial $\text{Bernoulli}(p=0.70)$).

No records were dropped, corrupted, or excluded due to schema violations. All 2,095 patients loaded by `load_modeling_dataset()` represent 100% of the ground-truth abstinent cases on disk.

---

## 3. Issue 2: Synthetic Model Performance (ROC-AUC = 0.9565)

### Simulation Optimism vs. Clinical Predictive Validity
In Milestone 3B, the primary Logistic Regression model achieved a held-out test ROC-AUC of **0.9565** (95% CI: 0.9264–0.9802).

### Critical Scientific Context:
1. **Mathematical Shared Family:** The synthetic data-generating process in `cohort_generator.py` calculates latent score $z$ via a linear combination of log-transformed PEth, linear fibrosis penalty, genetic penalty, dilation penalty, and age penalty, followed by logistic sigmoid transformation:
   $$\mathbb{E}[\Delta \text{LVEF}] = 32.0 \times \left(\frac{1}{1 + e^{-z}} - 0.35\right)$$
   When a Logistic Regression pipeline with regularized preprocessing is fitted on these data, the statistical model’s functional assumption matches the true functional form of the generator.
2. **Residual Stochastic Noise:** Although Gaussian noise ($\sigma = 4.25\%$ points) was injected to prevent deterministic separability, the underlying signal-to-noise ratio remains favorable for linear models.
3. **Clinical Limitation:** This high ROC-AUC demonstrates that the machine learning pipeline is capable of learning and recovering the multi-factorial dependencies formulated in the code. **It does NOT represent clinical predictive accuracy on real human patients.** Real-world patient recovery involves unmeasured confounders, medication adherence variations, neurohormonal factors, nutritional deficits, and complex myocardial biology not captured by parametric equations.

---

## 4. Issue 3: PEth Interpretation & Directionality

### Simulator Assumption vs. Real Clinical Biology
In the trained logistic regression model, baseline PEth carries a positive standardized coefficient ($\beta = +1.48$, $\text{OR} = 4.39$):
- **Simulator Mechanism:** The model was trained *specifically on patients who achieve sustained 6-month abstinence*. Under this conditional estimand, a patient with high baseline PEth ($\ge 200\text{ ng/mL}$) who completely stops drinking experiences the removal of a severe toxic repressor, permitting marked reverse remodeling. Conversely, a patient with low baseline PEth whose LVEF is 25% likely has non-toxic, irreversible idiopathic cardiomyopathy.
- **Clinical Governance Warning:** In clinical practice, high PEth is a direct marker of severe alcohol intake. It must NEVER be presented to clinicians or patients as a "protective factor" or an instruction that higher drinking promotes recovery. The positive association exists **strictly conditional on complete and sustained cessation**.

---

## 5. Issue 4: Test-Set Integrity & Leakage Audit

A comprehensive review of modeling logs, scripts, and checkpoints confirmed that the test partition ($N = 419$ cases, $20\%$ of eligible cohort) was strictly preserved:
- **No Data Leakage:** Preprocessing imputers, standard scalers, and one-hot encoders were fitted strictly on `X_train` ($N = 1,257$) and applied statelessly to test data.
- **No Model Selection on Test:** All model architecture comparisons (Logistic Regression vs. Random Forest vs. Dummy) and all 6 feature ablation experiments were executed and scored on the validation partition (`X_val`, $N = 419$).
- **Single Evaluation:** The finalized model pipeline was evaluated on `X_test` exactly once to produce the final evaluation report (`recovery_test_metrics.json`). No hyperparameters or decision thresholds were tuned against the test split.

---

## 6. Issue 5: Genetics (*TTN*) & Tissue (*LGE*) Boundaries

### Strict Categorical and Imaging Isolation
1. **ACMG-AMP Titin (*TTN*) Genetics:**
   - Pathogenic truncating variants (*TTNtv*) are treated as independent susceptibility modifiers that attenuate reverse remodeling.
   - Variants of Uncertain Significance (*TTN* VUS) have zero penalty in the generator and zero weight in the model ($\beta \approx 0.00$, $\text{OR} \approx 1.00$). They are strictly classified as non-pathogenic in accordance with ACMG-AMP guidelines.
2. **ACDC Cine MRI vs. Synthetic LGE Fibrosis:**
   - The real ACDC dataset consists exclusively of 2D/3D steady-state free precession (SSFP) cine cardiac MRI acquisitions.
   - ACDC cine MRI contains **NO Late Gadolinium Enhancement (LGE)** sequences.
   - Late Gadolinium Enhancement fibrosis patterns (midwall striae) and quantified extent (% of LV mass) in CRIP are **100% synthetic**. They were NOT automatically detected or segmented from ACDC scans.
   - When real ACDC cine MRI volumetrics are integrated with synthetic clinical variables in Milestone 4, the resulting record must be assigned a new hypothetical case identifier (e.g., `CRIP-SCENARIO-XXXX`) and explicitly labeled as a constructed multimodal demonstration scenario.

---

## 7. Audit Conclusion & Next Steps

All five scientific and data contracts are resolved, verified, and transparently documented. Development proceeds directly to Phase 3: implementing the unified patient feature layer, mock FHIR interoperability, and end-to-end application workflow integration.
