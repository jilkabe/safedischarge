# SafeDischarge: Fair, Explainable and Privacy-Preserving Readmission Risk Prediction for Diabetic Patients

**Team:** SafeDischarge
**Members:** Nguyen Hoang Truong Giang (HOANGTRU003@e.ntu.edu.sg)

> Drafting notes (delete before submission): every number below comes from the working prototype in `prototype/` (`results/*.json`). Figures: 1 = `design/architecture.svg`, 2 = `prototype/results/figures/dashboard.png`, 3 = `prototype/results/figures/fairness_recall_by_age.png`, 4 = `prototype/results/figures/privacy_tradeoff.png`. With 4 figures and 6 tables it fills about 9.4 pages at Arial 10pt. If it runs over, trim Table 1 first.

---

## 1. Introduction

### 1.1 The problem

Patients with diabetes are readmitted to hospital more often than other inpatients. Unplanned readmission within 30 days of discharge is costly, often preventable, and a sign that discharge planning or follow-up care was not enough. Hospitals widely use it as a quality indicator. In the United States, the Hospital Readmissions Reduction Program (HRRP) financially penalises hospitals with excess readmissions [1].

Interventions that reduce readmission exist and are well understood: medication reconciliation, diabetes education before discharge, an early follow-up appointment, and a nurse phone call within 72 hours. These interventions take staff time, so they cannot be given to everyone. The practical question for a discharge team is therefore:

> *Of the diabetic patients being discharged today, which ones should receive the limited follow-up resources?*

### 1.2 Why existing approaches fall short

1. **Simple scores are coarse.** Hospitals often use rule-based indices such as LACE [2] or the HOSPITAL score [3]. They are transparent but use few variables and have modest accuracy [4]. They ignore diabetes-specific signals such as HbA1c testing or medication changes during the stay.
2. **Machine-learning models raise trust problems.** More accurate models can be black boxes that nurses will not act on, and they can quietly perform worse for some patient groups [5, 6].
3. **Single-hospital data is small.** One hospital has too few readmissions to train a good model, especially for small subgroups. Pooling patient records across hospitals raises privacy, legal and governance barriers.

### 1.3 Our solution

**SafeDischarge** is a discharge-planning assistant. Each day it ranks diabetic patients due for discharge by their risk of readmission within 30 days, explains the main factors behind each patient's risk, and suggests a follow-up action for each factor. The discharge nurse uses the list to allocate a fixed number of follow-up slots.

The design is **practical and implemented**. We built a working prototype on the public **Diabetes 130-US Hospitals (1999–2008)** dataset [7] using open-source tools (scikit-learn [8], SHAP [9], Fairlearn [10], PyTorch with Opacus [11]). It includes a nurse dashboard. Every result in this report was measured on that prototype.

**What is new.** Most readmission models are judged by AUROC alone. SafeDischarge differs in four ways:
- **Evaluated where it is used.** Accuracy and fairness are measured at a fixed number of follow-up slots (recall at capacity). This exposed two problems a standard evaluation hides: tie-breaking that changed fairness results (Section 3.2), and a fairness-constrained model that became *less* fair at capacity (Section 4.2).
- **Explanations lead to actions, and are tested.** Each reason maps to a follow-up action, and a faithfulness test shows the reasons are what actually drives each flag (Section 4.1).
- **A statistical fairness gate.** Permutation tests separate real gaps from noise, so small groups are not passed or failed by chance.
- **The three practices tested together.** One release gate re-tests every practice after every change, which revealed interactions: differential privacy narrowed the age gap, and re-weighting by age widened the sex gap.

**All three Responsible AI practices are addressed and measured:**
- **Explainability:** top-3 reasons per flagged patient, each linked to an action and tested for faithfulness.
- **Fairness:** the audit found that patients aged 80+ were under-detected; per-age-group thresholds removed the gap for 1.1 points of recall.
- **Privacy:** federated training with secure aggregation and differential privacy nearly matches pooled-data accuracy, with no measurable leakage.

---

## 2. Application Description

### 2.1 Stakeholders and users

> **Table 1.** Stakeholders and their needs.

| Stakeholder | Role | Key need |
|---|---|---|
| Discharge nurses and case managers | Main users; allocate follow-up slots | A short ranked list with reasons they can act on |
| Ward physicians | Review flagged patients; adjust discharge plan | Clinically sensible explanations |
| Patients | Receive follow-up support | Fair access to support; protection of their data |
| Hospitals (data custodians) | Run local nodes; join the federation | Data stays on premises; compliance with health-privacy law |
| Model governance committee (clinicians, data-protection officer, equity lead) | Approves each model release | Evidence of accuracy, fairness and privacy |

### 2.2 System architecture

SafeDischarge has three layers (Figure 1):

- **Hospital node** (inside each hospital's network): feature pipeline, local trainer, scoring service, explanation engine, nurse dashboard.
- **Federation server** (run by a neutral party such as a regional health authority): coordinates training rounds, performs secure aggregation, tracks the privacy budget, stores model versions. It never receives patient records or any single hospital's update.
- **Governance layer**: fairness and accuracy audit, model cards, release gate and monitoring.

> **Figure 1.** SafeDischarge system architecture.

### 2.3 User journey

1. **Daily trigger.** Each morning, SafeDischarge scores every diabetic inpatient due for discharge, using data already in the hospital record.
2. **Ranked list at capacity.** The nurse sets today's number of follow-up slots (*k*). The top-*k* patients are marked "High: follow-up", the next *k* "Medium", the rest "Low". Each patient shows a calibrated 30-day risk (e.g. "25%").
3. **Reasons and actions.** Each flagged patient shows up to three risk factors in plain language, each with a suggested action (Figure 2). A real example from the prototype:
   - "3 inpatient admissions in the past year" → *refer to case management*
   - "2 earlier 30-day readmissions" → *review why earlier discharges failed*
   - "Discharge destination: care facility" → *hand-over call to receiving facility*
4. **Decision.** The nurse accepts or overrides each plan. Overrides are logged with a reason.
5. **Outcome capture.** 30-day readmissions are recorded automatically and become new training data.

> **Figure 2.** Prototype nurse dashboard: today's ranked list with calibrated risk and top reason (top), and one flagged patient's three reasons, suggested actions and the nurse's decision (bottom).

### 2.4 Data

The **Diabetes 130-US Hospitals** dataset [7] (CC BY 4.0) contains 101,766 inpatient stays of 1–14 days with a diabetes diagnosis, from 130 US hospitals. It stands in for the records a real hospital would hold.

> **Table 2.** Data used by SafeDischarge.

| Group | Fields | Use |
|---|---|---|
| Demographics | race, sex, age (10-year bands) | Fairness audit groups; age is also a model input. Race is **not** an input |
| Admission | admission type and source, discharge destination, length of stay | Inputs |
| Prior use | outpatient, emergency and inpatient visits in the past year; the patient's earlier stays in the data and how many of them ended in a 30-day readmission | Inputs |
| Clinical | diagnoses (ICD-9, grouped into 9 clinical categories), number of diagnoses, lab tests, procedures, medications | Inputs |
| Diabetes-specific | HbA1c and glucose test results, insulin and metformin status, number of diabetes medication changes | Inputs; linked to follow-up actions |
| Outcome | readmitted within 30 days | Target (11.4% positive after exclusions) |

**Data preparation decisions:**
- **Deaths and hospice.** 2,423 stays ending in death or hospice are excluded, plus 3 with unknown sex. This leaves 99,340 stays from 69,987 patients.
- **Earlier stays.** 23% of patients have more than one stay, so 30% of stays have an earlier one. For each stay we count the patient's *earlier* stays and earlier 30-day readmissions. The data has no dates, so stay-ID order is used as time order. Only earlier stays are used, and their outcomes are always known by the current discharge, so no future information leaks in. Patients with two earlier readmissions are readmitted 31% of the time, versus 10% for those with none.
- **Repeat patients.** Splits are made **by patient** (70/15/15 train/validation/test), so no patient appears in two sets. Splitting by stay would let the model memorise patients and inflate test results.
- **Missing data.** `weight` (97% missing) is dropped. `payer_code` (40% missing) is kept only as an audit attribute. `medical_specialty` (49% missing) gets an explicit "Unknown" category. HbA1c was not measured in 83% of stays, and that absence is itself informative, so it is kept as a "not measured" category.
- **No hospital ID.** The dataset does not record which hospital each stay came from, so multiple hospitals are **simulated** (Section 3.4).

---

## 3. Use of AI Technologies

### 3.1 Readmission risk model

- **Task:** binary classification. Will the patient be readmitted within 30 days, given what is known at discharge?
- **Models compared:** L2-regularised logistic regression (the transparent baseline), gradient-boosted trees (scikit-learn HistGradientBoosting [8]), and a small 2-layer neural network (MLP, used for federated training). All use class weighting because readmissions are the minority class. Gradient-boosting settings were chosen by a 40-trial random search scored on validation AUPRC; the test set was not used for any choice.
- **Main metric, recall at capacity:** of all patients actually readmitted, the share who appear in the top 20% of the ranked list. 20% is the assumed follow-up capacity. Random selection would give 20%.

> **Table 3.** Model comparison on the held-out test set (14,859 stays, 1,681 readmissions).

| Model | AUROC | AUPRC | Recall at 20% capacity |
|---|---|---|---|
| Logistic regression | 0.667 | 0.211 | 36.1% |
| **Gradient boosting (chosen)** | **0.681** | **0.233** | **40.8%** |
| MLP | 0.681 | 0.230 | 39.9% |
| Gradient boosting + race as input | 0.681 | 0.234 | 40.1% |

**Reading the results.** Discrimination is modest (AUROC ≈ 0.68), which matches what is generally reported for readmission models [4]. But it doubles the yield of a random selection: about 41% of future readmissions fall in the top 20% of the list. Gradient boosting catches 4.6 points more readmissions than logistic regression, about 78 more patients per 1,681. Our rule is to prefer the simpler model unless the complex one adds at least 2 points of recall, so gradient boosting is chosen. Adding race as an input changes nothing (same AUROC, slightly lower recall), so race is left out of the model and used only for auditing.

### 3.2 Calibration and capacity-based ranking

- **Calibration.** Isotonic regression fitted on the validation set turns scores into honest probabilities. The Brier score falls from 0.215 (class-weighted raw scores) to 0.095, slightly better than the 0.100 obtained by predicting the base rate for everyone. The calibrated value is what the nurse sees.
- **Ranking on the raw score.** Isotonic calibration maps many patients to the same probability, creating ties. In our first prototype, which patients were flagged at the capacity cut-off, and therefore the fairness results, depended on how those ties were broken. The final design ranks on the model's raw (continuous) score and uses the calibrated probability only for display. This is a practical lesson: fairness checks should be run on exactly the ranking that is deployed.

### 3.3 Explanation engine

- **Per-patient explanations:** TreeSHAP [9] values for each prediction. Values for one-hot columns of the same variable are summed, so "discharge destination" counts as one reason. Only factors that *increase* risk are shown.
- **Reason → action table:** a short clinician-authored table maps each variable to a follow-up action, e.g. prior inpatient visits → case management; HbA1c not measured → order HbA1c or diabetes clinic review.
- **Global check:** SHAP importance ranks prior inpatient admissions first, followed by discharge destination, the patient's earlier stays, age, number of diagnoses and earlier 30-day readmissions. This matches clinical knowledge and published risk indices [2, 3], which is evidence that the model relies on sensible signals.

### 3.4 Federated learning across hospitals

- **Simulation.** Training patients are assigned to *H* = 5 virtual hospitals. Within each race × age stratum, hospital shares are drawn from a Dirichlet(0.5) distribution, which gives each hospital a very different case mix, as in real networks. Hospital sizes range from 6,518 to 20,524 stays. The African American share ranges from 3% to 47%, and the share aged 80+ from 3% to 35%.
- **Training.** FedAvg [12]: 20 rounds of 1 local epoch each, with updates weighted by hospital size. In production we would use the Flower framework [13]; the prototype implements the same protocol directly in PyTorch for transparency.
- **Secure aggregation [14].** Each pair of hospitals shares a random mask; one adds it, the other subtracts it. The server sees only masked updates, which are uncorrelated with the real updates (correlation 0.02), but the masks cancel in the sum (maximum error 3 × 10⁻⁷). The prototype simulates the key agreement; a real deployment would use the cryptographic protocol of [14].
- **Results (Table 4).** Federated training nearly matches pooling all the data (AUROC 0.675 vs 0.677) and beats every hospital training alone (0.649–0.666, mean 0.660). In other words, hospitals get almost all the benefit of pooled data without pooling it.

> **Table 4.** Federated learning and differential privacy (MLP; mean of 3 random seeds; AUROC standard deviation ≤ 0.002, except 0.008 for the over-fitted control).

| Setting | AUROC | Recall at 20% | Membership-inference AUC |
|---|---|---|---|
| Each hospital alone (mean of 5) | 0.660 | 37.2% | – |
| All data pooled (not allowed in practice) | 0.677 | 39.2% | 0.511 |
| Federated + secure aggregation | 0.675 | 38.8% | 0.508 |
| + DP, ε = 8 | 0.662 | 36.9% | 0.500 |
| + DP, ε = 3 | 0.657 | 36.2% | 0.500 |
| + DP, ε = 1 | 0.644 | 35.1% | 0.499 |
| Over-fitted control (no DP) | 0.601 | 29.1% | 0.592 |

### 3.5 Monitoring and prototype

**Monitoring.** Drift in key features and in the readmission rate is checked monthly. Recall, calibration and the fairness tests are tracked once 30-day outcomes are known (the dashboard's *Fairness monitor* tab). The model is retrained quarterly by federated learning, and each version must pass the release gate (Section 4.4).

**Prototype.** The prototype (`prototype/`) runs every experiment in this report end to end in about 9 minutes on a laptop (`run_all.sh`). It covers data preparation and model comparison, the explanation faithfulness test, the fairness audit and mitigations, federated training with DP and the membership-inference attack, and the Streamlit nurse dashboard (Figure 2).

---

## 4. Responsible AI Practices Addressed

### 4.1 Explainability

**Why it matters:** nurses must decide quickly and will not act on an unexplained score. Explanations also let clinicians catch a model that relies on a wrong or unfair signal.

**Mechanisms (all implemented):**
1. **Top-3 reasons per flagged patient** from SHAP, in plain language.
2. **Reason → action mapping**, so every explanation leads to something the nurse can do.
3. **Calibrated probabilities**, so "25% risk" means about 25 in 100 similar patients are readmitted.
4. **Transparent baseline:** logistic regression is always reported. The complex model is used only if it beats the baseline by a set margin (Section 3.1).
5. **Global review:** SHAP importance is checked against clinical knowledge before each release (Section 3.3).
6. **Model card** in the dashboard: intended use, data, performance, fairness results and limitations.

**Are the explanations faithful?** An explanation helps only if it reflects what the model actually uses. For each of the 2,972 patients flagged in the test set, we replaced the variables behind their top-3 reasons with values from randomly chosen training patients (averaged over 20 draws) and re-scored them. Their risk score fell by 29% on average, and 97% dropped off the flagged list. Replacing 3 randomly chosen variables lowered risk by only 2.7%, and 15% dropped off. The top-3 reasons outperformed the random variables for 99% of patients. The reasons shown to nurses are therefore the factors that actually cause each flag.

**Further evaluation (not yet done):** a usability test with clinical reviewers. Can they say *why* a patient was flagged, and do they agree with the reason? Target ≥ 80% agreement.

### 4.2 Fairness

**Why it matters:** SafeDischarge decides who gets *extra help*. If high-risk patients in one group are under-detected, that group systematically receives less follow-up care [5].

**Definition.** The harm is *missing* a patient who needed help, so we require **equal opportunity** [15]: recall at capacity should be similar across groups of each protected attribute (race, sex, age). Insurance type is monitored as an audit-only proxy for socio-economic status.

**Testing for real gaps, not noise.** Some groups are tiny. For example, Asian patients are 0.6% of stays and have only 7 readmissions in the test set. The largest-minus-smallest recall gap among small groups is dominated by noise, so a fixed tolerance such as "gap ≤ 5 points" would fail or pass largely by chance. For each attribute we therefore run a **permutation test**: we shuffle group labels among the readmitted patients 2,000 times and ask how often chance alone produces a gap as large as the one observed. Groups with fewer than 30 readmissions are reported with confidence intervals but not tested. They are flagged as needing more data, not declared fair.

**Audit result.** Race gap: 9.7 points, p = 0.31, not significant. Sex gap: 1.8 points, p = 0.49, not significant. **Age gap: 21.5 points, p < 0.001, significant.** The model catches 58% of readmissions in patients under 40 but only **36% in patients aged 80+**, the group most vulnerable after discharge. A likely cause is that older patients are readmitted for reasons that are less visible in prior-use counts.

**Mitigations compared** (Figure 3), each also re-tested for race and sex:

> **Table 5.** Fairness mitigation results on the test set (capacity fixed at 20%).

| Method | Overall recall | Age gap (p) | Sex gap (p) | Release gate |
|---|---|---|---|---|
| Baseline | 40.8% | 21.5 pts (< 0.001) | 1.8 pts (0.49) | **Fails** (age) |
| Pre-processing: re-weight under-detected age groups | 39.7% | 13.5 pts (**0.02**) | 4.4 pts (0.06) | **Fails** (age) |
| In-processing: Fairlearn exponentiated gradient, TPR parity [10] | 32.7% | 56.6 pts (< 0.001) | 4.1 pts (0.09) | **Fails** |
| **Post-processing: per-age-group thresholds** | **39.7%** | **6.8 pts (0.39)** | **1.7 pts (0.50)** | **Passes** |

> **Figure 3.** Recall at 20% capacity by age group for each mitigation (95% bootstrap confidence intervals).

**What we learned:**
- **Per-age-group thresholds work best.** Thresholds fitted on the validation set give each age band the same recall while keeping the total number flagged at capacity. Recall for 80+ rises from 36% to 38%, the gap is no longer significant, and overall recall falls by only 1.1 points (about 18 of 1,681 readmissions). This method treats age groups explicitly differently. We consider this acceptable because age is a clinically legitimate factor and the adjustment *raises* priority for the group that was being missed. It still requires approval by the governance committee.
- **Fixing one attribute can strain another.** Re-weighting narrowed the age gap but not enough (still significant), and it widened the sex gap from 1.8 to 4.4 points (p = 0.06, borderline). In an earlier version of our prototype the same method made the sex gap significant. Checking every attribute after every change is essential.
- **Constraint methods must match how the model is used.** Exponentiated gradient enforces equal recall at the classifier's own decision threshold. SafeDischarge instead flags a fixed top 20%. At that operating point the constrained model was both less accurate and *less* fair. Fairness must be evaluated at the operating point actually deployed.
- **Leaving race out of the model is not enough** on its own [16]. That is why race is still audited.

### 4.3 Privacy preservation

**Why it matters:** hospital records are highly sensitive. Pooling them needs legal agreements and increases the damage from a breach. Trained models can also reveal whether a person was in the training data (membership inference) [17].

**Mechanisms:**
1. **Federated learning:** patient records never leave a hospital; only model updates do.
2. **Secure aggregation:** the server sees only masked updates and learns only their sum (Section 3.4).
3. **Differential privacy:** each hospital trains with **DP-SGD** [18] in Opacus [11]. Each patient's gradient is clipped (norm 1.0) and Gaussian noise is added. The noise level is set by Opacus's privacy accountant so that each hospital spends at most the target ε over all 20 rounds (δ = 10⁻⁵).
4. **Data minimisation and local scoring:** only fields needed for prediction are used, the data is de-identified, and daily scoring runs inside each hospital.
5. **Privacy attack test:** a loss-threshold membership-inference attack [19] tries to tell training patients from unseen patients. An AUC of 0.5 means it can do no better than guessing.

**Results (Table 4, Figure 4).**
- **The attack works when a model leaks:** against a deliberately over-fitted model it reaches AUC 0.592.
- **Normal models leak a little:** pooled and federated models without DP give AUC 0.508–0.511.
- **DP removes measurable leakage:** at every ε tested the attack is at chance (0.499–0.500).
- **Accuracy cost:** 1.2 AUROC points at ε = 8 and 1.7 at ε = 3. At ε = 8 the shared model is slightly better than the average hospital training alone (0.662 vs 0.660), though not better than the largest hospital alone (0.666). At ε = 3 it is slightly below the average single hospital, but with a much stronger privacy guarantee. Most of the federation gain is spent on privacy.
- **Recommended setting:** ε = 8. It is the strongest guarantee we tested that keeps a small benefit over training alone for most hospitals. ε = 3 is the conservative alternative.

> **Figure 4.** Privacy–utility trade-off: AUROC (left) and membership-inference AUC (right) for federated training at ε = 1, 3, 8 and without DP (mean ± SD, 3 seeds).

### 4.4 Trade-offs and the release gate

- **Privacy vs fairness.** DP is known to hurt small groups most [20]. In our runs the opposite happened for age: the DP models had a *smaller* age gap (6–10 points vs 23 without DP), probably because the noise stops the model relying heavily on age-specific patterns. This is one dataset and one model, so the release gate re-tests fairness after DP rather than assuming either direction.
- **Fairness vs accuracy.** Equalising recall across age groups costs 1.1 points of overall recall.
- **Accuracy vs explainability.** The more complex model is accepted only because it beats the interpretable baseline by more than the agreed margin.

**Release gate.** A new model version is deployed only if:
1. recall at capacity is no worse than the current model;
2. no protected attribute shows a significant recall gap (permutation p ≥ 0.05), and small groups are listed as "insufficient data";
3. ε is within the agreed budget and the membership-inference AUC is ≤ 0.52;
4. the governance committee signs off the global explanations and any group-specific thresholds;
5. the model card is updated.

### 4.5 Accountability and human oversight

- **The nurse decides.** SafeDischarge only prioritises extra follow-up. No patient loses standard discharge care, and a nurse can add any patient to the list.
- **Overrides are logged and reviewed.** Every accept or override is recorded with a reason. Override rates are reviewed monthly by patient group; a group whose flags are often overridden triggers a review of the model or the action table.
- **Patients can ask why.** Patients are told that a decision-support tool is used, and can ask why they were or were not offered follow-up. Staff answer using the same plain-language reasons the nurse sees.
- **Clear ownership.** The governance committee owns each release. Every model version is stored with its data snapshot, ε spent, fairness results and model card, so any past decision can be traced to the model that made it.

> **Table 6.** Summary: Responsible AI mechanisms and measured results.

| Practice | Mechanisms | Measured result |
|---|---|---|
| Explainability | SHAP top-3 reasons → actions, faithfulness test, calibration, interpretable baseline, global review, model card | Removing top-3 reasons cuts risk 29% vs 2.7% for random variables; Brier 0.215 → 0.095 |
| Fairness | Equal-opportunity audit, permutation test, per-age-group thresholds, re-testing every attribute | Age gap 21.5 → 6.8 pts (p < 0.001 → 0.39) for −1.1 pts recall |
| Privacy | Federated learning, secure aggregation, DP-SGD with accountant, local scoring, attack test | Federated ≈ pooled (0.675 vs 0.677); DP ε = 8 costs 1.2 pts; attack AUC 0.508 → 0.500 |

---

## 5. Limitations

- **Old, US-only data.** The dataset covers 1999–2008 in the US. Practice and populations have changed. A real deployment must retrain and validate on current local data.
- **Simulated hospitals.** Without hospital IDs, the virtual-hospital split approximates real between-site differences but cannot reproduce them fully.
- **Simulated secure aggregation.** The masking protocol is simulated in one process. Real deployments need the full cryptographic protocol, including handling hospitals that drop out mid-round.
- **Modest accuracy.** AUROC ≈ 0.68. SafeDischarge is a prioritisation aid, not an automated decision-maker.
- **Prediction is not prevention.** Benefit depends on the follow-up actions working. A prospective before/after evaluation is needed.
- **Small groups.** Several race groups have too few readmissions to audit. Collaborating hospitals with more diverse populations are needed to check fairness for them.

## 6. Conclusion

SafeDischarge helps discharge teams direct limited follow-up resources to the diabetic patients most likely to be readmitted. The prototype shows the design is practical: on public data, it doubles the readmissions caught compared with random selection. Its explanations pass a faithfulness test, and its audit found and fixed an age disparity that would have left the oldest patients under-served. Federated training gives every hospital close to pooled-data accuracy without pooling records, and differential privacy removes measurable leakage at a cost of about one AUROC point. Each Responsible AI practice is a tested mechanism enforced by a release gate.

---

## References

[1] Centers for Medicare & Medicaid Services. *Hospital Readmissions Reduction Program (HRRP)*. https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp
[2] C. van Walraven et al. Derivation and validation of an index to predict early death or unplanned readmission after discharge from hospital to the community. *CMAJ*, 182(6), 2010.
[3] J. Donzé, D. Aujesky, D. Williams, J. L. Schnipper. Potentially avoidable 30-day hospital readmissions in medical patients: derivation and validation of a prediction model. *JAMA Internal Medicine*, 173(8), 2013.
[4] D. Kansagara et al. Risk prediction models for hospital readmission: a systematic review. *JAMA*, 306(15), 2011.
[5] Z. Obermeyer, B. Powers, C. Vogeli, S. Mullainathan. Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 2019.
[6] A. Rajkomar, M. Hardt, M. D. Howell, G. Corrado, M. H. Chin. Ensuring fairness in machine learning to advance health equity. *Annals of Internal Medicine*, 169(12), 2018.
[7] B. Strack, J. P. DeShazo, C. Gennings, J. L. Olmo, S. Ventura, K. J. Cios, J. N. Clore. Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. *BioMed Research International*, 2014. Dataset: UCI Machine Learning Repository, "Diabetes 130-US Hospitals for Years 1999–2008", CC BY 4.0.
[8] F. Pedregosa et al. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2011.
[9] S. M. Lundberg, S.-I. Lee. A unified approach to interpreting model predictions. *NeurIPS*, 2017.
[10] S. Bird et al. Fairlearn: A toolkit for assessing and improving fairness in AI. Microsoft Technical Report MSR-TR-2020-32, 2020.
[11] A. Yousefpour et al. Opacus: User-friendly differential privacy library in PyTorch. arXiv:2109.12298, 2021.
[12] B. McMahan, E. Moore, D. Ramage, S. Hampson, B. A. y Arcas. Communication-efficient learning of deep networks from decentralized data. *AISTATS*, 2017.
[13] D. J. Beutel et al. Flower: A friendly federated learning research framework. arXiv:2007.14390, 2020.
[14] K. Bonawitz et al. Practical secure aggregation for privacy-preserving machine learning. *ACM CCS*, 2017.
[15] M. Hardt, E. Price, N. Srebro. Equality of opportunity in supervised learning. *NeurIPS*, 2016.
[16] I. Chen, F. D. Johansson, D. Sontag. Why is my classifier discriminatory? *NeurIPS*, 2018.
[17] R. Shokri, M. Stronati, C. Song, V. Shmatikov. Membership inference attacks against machine learning models. *IEEE S&P*, 2017.
[18] M. Abadi et al. Deep learning with differential privacy. *ACM CCS*, 2016.
[19] S. Yeom, I. Giacomelli, M. Fredrikson, S. Jha. Privacy risk in machine learning: analyzing the connection to overfitting. *IEEE CSF*, 2018.
[20] E. Bagdasaryan, O. Poursaeed, V. Shmatikov. Differential privacy has disparate impact on model accuracy. *NeurIPS*, 2019.
