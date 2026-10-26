# CRIP Synthetic Clinical Cohort: Literature Registry & Modeling Assumptions

**Cardiomyopathy Recovery Intelligence Platform (CRIP)**  
*Clinical AI Research Proof of Concept — Milestone 3A*  
**Date:** October 2026  
**Document Status:** Approved Research Specification  

---

> [!CAUTION]
> **INVESTIGATIVE RESEARCH SPECIFICATION — NOT A CLINICAL TOOL**  
> This document specifies the mathematical and epidemiological assumptions used to synthesize an in silico multimodal patient cohort for research algorithm development. All generated profiles, biomarker values, genetic variants, and simulated six-month outcomes are artificial. They must **never** be used for clinical diagnosis, patient risk stratification, treatment decisions, or patient counseling.

---

## 1. Clinical Research Question & Recovery Endpoint

### 1.1 The Research Objective
To evaluate whether multimodal baseline clinical variables—combining **objective toxicology (blood PEth)**, **structural cardiac magnetic resonance (CMR volumetrics)**, **genetic predisposition (*TTN* variants)**, and **cardiac tissue characterization (LGE replacement fibrosis)**—can provide transparent, interpretable prediction of 6-month left ventricular functional recovery under a hypothetical sustained-abstinence clinical scenario.

### 1.2 Binary Recovery Endpoint Definition
In accordance with cardiovascular magnetic resonance reverse-remodeling literature (Fauchier et al., 2000; Nicolas et al., 2002; Halliday et al., 2017), the primary experimental recovery endpoint is defined as:

$$\Delta \text{LVEF} = \text{LVEF}_{6\text{m}} - \text{LVEF}_{\text{baseline}}$$

$$\text{Recovery} = \begin{cases} 1, & \text{if } \Delta \text{LVEF} \ge 10.0\text{ percentage points} \\ 0, & \text{if } \Delta \text{LVEF} < 10.0\text{ percentage points} \end{cases}$$

### 1.3 Methodological Safety Distinctions
- **Baseline Predictors Only:** Future predictive models (Milestones 3B/4) must strictly evaluate baseline clinical features available at the initial outpatient consultation. Follow-up observations, confirmed sustained abstinence, and 6-month LVEF values are simulated outcomes and must **never** be included as baseline predictors.
- **Biomarker Limitations:** Phosphatidylethanol (PEth) quantifies recent ethanol exposure over the preceding 2–4 weeks. It does **not** prove alcohol-induced cardiomyopathy by itself, nor does it predict acute alcohol withdrawal syndrome severity or CIWA-Ar scores.
- **ACDC Data Distinction:** Real ACDC imaging provides structural short-axis cine geometries (EDV, ESV, LVEF) for physical grounding. The ACDC dataset contains **zero** alcohol exposure or genetic data. Any simulated profile referencing ACDC metrics receives a new synthetic identifier and clear data provenance tags.

---

## 2. Modeled Variables & Parameter Specifications

The following table distinguishes **Empirically Supported Literature Parameters** from **Illustrative Engineering Assumptions**:

| Variable | Physical Units | Valid Range | Distribution Type | Literature Reference | Evidence Strength | Classification |
|---|---|---|---|---|---|---|
| **Age** | Years | 18 – 85 | Truncated Normal ($\mu=52, \sigma=11$) | Fauchier 2000, Nicolas 2002 | High | Empirical |
| **Biological Sex** | Categorical | MALE (78%), FEMALE (22%) | Bernoulli | Fernandez-Sola 2006 | High | Empirical |
| **Baseline LVEF** | Percentage (%) | 15.0% – 45.0% | Beta-scaled ($\mu=31.5\%, \sigma=6.5\%$) | ACDC DCM Cohort, Bernard 2018 | High | Empirical |
| **LVEDV** | Milliliters (mL) | 120.0 – 450.0 mL | Log-Normal ($\mu=230\text{ mL}$) | ACDC DCM Cohort, Bernard 2018 | High | Empirical |
| **LVESV** | Milliliters (mL) | 70.0 – 380.0 mL | Derived via LVEF consistency | Physical Geometry ($ESV < EDV$) | High | Deterministic Physical |
| **RVEDV** (Optional) | Milliliters (mL) | 100.0 – 380.0 mL | Normal ($\mu=185\text{ mL}, \sigma=45$) | ACDC DCM Cohort | Moderate | Empirical |
| **RVEF** (Optional) | Percentage (%) | 20.0% – 65.0% | Normal ($\mu=44.0\%, \sigma=10.5$) | ACDC DCM Cohort | Moderate | Empirical |
| **Myocardial Mass** | Grams (g) | 80.0 – 280.0 g | Derived via density ($\rho=1.05\text{ g/mL}$) | SCMR Guidelines, Schulz-Menger 2020 | High | Empirical |
| **PEth Concentration** | ng/mL | 0.0 – 2500.0 ng/mL | Mixture: 20% <20, 35% 20–200, 45% >200 | Ulwelling 2018, Stewart 2014 | High | Empirical |
| **Reported Drinks/Wk** | Drinks/week | 0 – 120 drinks | Correlated with PEth ($r \approx 0.72$) | Kechagias 2015 | Moderate | Empirical |
| **Time Since Last Drink**| Days | 0 – 60 days | Exponential decay | Stewart 2014 | Moderate | Engineering Assumption |
| **TTN Variant** | Categorical | 5 ACMG-AMP Tiers | Categorical | Herman 2012, Ware 2016 | High | Empirical |
| **LGE Status** | Categorical | Negative (65%), Positive (35%) | Bernoulli | Gulati 2013, Halliday 2017 | High | Empirical |
| **LGE Pattern** | Categorical | Midwall (75%), Subepicardial, etc. | Multinomial conditional on Positive | Gulati 2013 | High | Empirical |
| **LGE Extent** | % of LV Mass | 0.0% – 25.0% | Zero-inflated Exponential | Halliday 2017 | Moderate | Empirical |
| **Assumed Abstinence** | Categorical | Sustained (70%), Non-Sustained (30%) | Simulation Scenario | Nicolas 2002 | Moderate | Scenario Assumption |
| **6-Month LVEF** | Percentage (%) | 10.0% – 70.0% | Multi-factor Stochastic Model | Simulated Endpoint | — | Research Simulation |
| **Recovery Label** | Binary (0 or 1) | {0, 1} | Deterministic from $\Delta \text{LVEF} \ge 10\%$ | Fauchier 2000 | High | Deterministic Definition |

---

## 3. Genetic Classification Rules (*TTN* Variants)

In strict adherence to the **ACMG-AMP 2015 Guidelines** (Richards et al., 2015) and *TTN* dilated cardiomyopathy genomics (Herman et al., 2012; Ware et al., 2016):

1. **Pathogenic (PATHOGENIC):** Truncating variants (*TTNtv*: nonsense, frameshift, essential splice-site) located in constitutively expressed exons of the titin A-band with high percent spliced-in (PSI > 0.9). Prevalence modeled at **12.5%** in the severe DCM cohort.
2. **Likely Pathogenic (LIKELY_PATHOGENIC):** Truncating variants in titin I-band or splice regions with strong computational and family co-segregation evidence. Prevalence modeled at **4.5%**.
3. **Variant of Uncertain Significance (VARIANT_OF_UNCERTAIN_SIGNIFICANCE / VUS):** Non-truncating missense variants or variants in poorly characterized titin domains. Prevalence modeled at **11.0%**.
   > [!IMPORTANT]
   > **CLINICAL GENETICS SAFETY RULE:**  
   > A *TTN* VUS must **never** be treated as a confirmed inherited genetic etiology. In the synthetic generator and subsequent clinical modeling, VUS cases must not be pooled with pathogenic truncating variants.
4. **No Pathogenic/Likely Pathogenic Variant Identified (NO_PATHOGENIC_OR_LIKELY_PATHOGENIC):** Comprehensive next-generation sequencing panel completed; no disease-causing *TTN* variant detected. Modeled at **54.0%**.
5. **Not Tested (NOT_TESTED):** Genetic sequencing not performed or sample refused. Modeled at **18.0%** to capture real-world clinical missingness.

---

## 4. Realistic Clinical Missingness Mechanisms

Real-world clinical registries and EHR datasets routinely encounter missing diagnostic investigations. The synthetic generator implements explicit, realistic missing-data mechanisms:

1. **Missing PEth (5.0%):** Patient blood draw refused, hemolyzed sample, or test unavailable at initial community visit.
2. **Missing Genetic Testing (18.0%):** Insurance denial, patient opt-out, or genetic counseling pending.
3. **Missing LGE CMR (10.0%):** Contrast contraindication due to severe renal impairment (eGFR $< 30\text{ mL/min/1.73m}^2$), severe claustrophobia, or metallic foreign bodies.
4. **Missing Reported Drinking History (7.0%):** Incomplete clinician documentation or patient refusal.

---

## 5. Mathematical Formulation of Simulated Six-Month Outcomes

To prevent an artificially trivial classification task (e.g. single-feature linear separability), the 6-month LVEF trajectory is modeled as a non-linear, multi-factorial stochastic function with biological interactions and Gaussian residual variance.

### 5.1 Latent Reverse Remodeling Potential ($\eta_i$)
The latent recovery propensity for patient $i$ is calculated from baseline parameters and simulated abstinence:

$$\eta_i = \beta_0 + \beta_{\text{abs}} \cdot A_i + \beta_{\text{peth}} \cdot f_{\text{toxic}}(P_i, A_i) - \beta_{\text{lge}} \cdot L_i - \beta_{\text{ttn}} \cdot T_i - \beta_{\text{dilation}} \cdot D_i - \beta_{\text{age}} \cdot \text{Age}_i$$

Where:
- $A_i \in \{0, 1\}$: Simulated sustained complete abstinence ($A_i = 1$ in sustained abstinence scenario).
- $P_i$: Baseline PEth level ($\text{ng/mL}$). Under sustained abstinence, high baseline PEth identifies true toxic alcohol etiology with high recovery capacity:
  $$f_{\text{toxic}}(P_i, A_i) = \begin{cases} \ln\left(1 + \frac{P_i}{50}\right), & \text{if } A_i = 1 \\ -0.5 \cdot \ln\left(1 + \frac{P_i}{100}\right), & \text{if } A_i = 0 \text{ (continued toxic exposure)} \end{cases}$$
- $L_i$: Myocardial fibrosis penalty based on LGE status and extent:
  $$L_i = \begin{cases} 0, & \text{if LGE is Negative} \\ 1.0 + 0.15 \times \text{LGE Extent (\%)} & \text{if LGE is Positive} \end{cases}$$
- $T_i$: Genetic modifier penalty:
  $$T_i = \begin{cases} 1.2, & \text{if TTN is Pathogenic} \\ 0.8, & \text{if TTN is Likely Pathogenic} \\ 0.0, & \text{if TTN is VUS, Negative, or Not Tested} \end{cases}$$
- $D_i$: Severe cavity dilation penalty ($D_i = \max(0, \frac{\text{LVEDV} - 250}{50})$).
- $\text{Age}_i$: Age penalty ($\frac{\text{Age} - 50}{25}$).

### 5.2 Six-Month LVEF Realization
$$\Delta \text{LVEF}_i = \text{Scale} \times \left( \frac{1}{1 + e^{-\eta_i}} - 0.35 \right) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2 = 18.0)$$

$$\text{LVEF}_{6\text{m}, i} = \text{Clip}(\text{LVEF}_{\text{baseline}, i} + \Delta \text{LVEF}_i, 10.0\%, 75.0\%)$$

$$\text{Recovery}_i = \mathbb{I}(\text{LVEF}_{6\text{m}, i} - \text{LVEF}_{\text{baseline}, i} \ge 10.0\%)$$

### 5.3 Non-Trivial Separability Guarantee
The inclusion of substantial stochastic residual noise ($\sigma = 4.25\%$ points) guarantees that:
- No single clinical feature (e.g. PEth alone or LGE alone) can achieve a univariate ROC-AUC $> 0.80$.
- Predicting recovery requires realistic, non-linear multimodal feature combination.
- The outcome reflects genuine clinical uncertainty observed in prospective reverse remodeling studies.
