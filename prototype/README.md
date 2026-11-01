# SafeDischarge prototype

Working prototype of the design in [`../report/draft.md`](../report/draft.md): 30-day readmission risk for diabetic inpatients, with explanations, a fairness audit and mitigation, and a federated + differentially private training simulation.

## Run it

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 and all packages are installed automatically.

```bash
cd prototype
./run_all.sh                 # downloads data, runs all experiments (about 9 minutes)
uv run streamlit run app.py  # nurse dashboard at http://localhost:8501
```

## What each step does

| Script | What it does | Outputs |
|---|---|---|
| `scripts/00_tune.py` | Random search (40 trials) over gradient-boosting settings, scored on validation AUPRC. The test set is never used | `results/tuning.json` |
| `scripts/01_central.py` | Cleans data, splits by patient (70/15/15), compares logistic regression, gradient boosting and an MLP, tests race-as-input, calibrates, runs SHAP | `results/central.json`, `calibration.png`, `shap_summary.png`, `artifacts.joblib` |
| `scripts/02_fairness.py` | Audits recall by age, race and sex at 20% capacity, with permutation tests. Compares re-weighting, Fairlearn exponentiated gradient and per-age-group thresholds | `results/fairness.json`, `fairness_recall_by_age.png` |
| `scripts/03_federated.py` | Splits the data into 5 virtual hospitals with different patient mixes. Compares single-hospital, pooled, FedAvg with secure aggregation, and FedAvg + DP-SGD (ε = 1, 3, 8). Runs a membership-inference attack. 3 seeds | `results/federated.json`, `privacy_tradeoff.png` |
| `scripts/04_explain.py` | Faithfulness test for explanations: for each flagged test patient, replaces the top-3 SHAP reasons with values from random training patients and compares the score drop with replacing 3 random variables | `results/explain.json` |
| `app.py` | Streamlit dashboard: ranked discharge list, top-3 reasons → actions, fairness monitor, model card | — |

Code lives in `src/safedischarge/`: `data.py` (cleaning, patient split, virtual hospitals), `metrics.py` (recall at capacity, subgroup recall with CIs, permutation test), `models.py` (MLP and training loop).

## Key design decisions

- **Ranking at capacity:** patients are ranked and exactly the top 20% are flagged, to match a fixed number of follow-up slots. Ranking uses the model's raw score. Isotonic calibration creates ties, which made results depend on tie-breaking, so the calibrated probability is used only for display.
- **Earlier stays as features.** Each stay gets the patient's number of earlier stays and earlier 30-day readmissions, using stay-ID order as time order (the data has no dates). Only earlier stays are counted, so nothing from the future leaks in.
- **Race is not a model input.** Adding it does not change accuracy (AUROC 0.6814 with and without). Race is used only for auditing.
- **Fairness gate:** for each protected attribute, the gap is tested with a permutation test, and it fails if p < 0.05. A fixed-size tolerance would be dominated by noise in small groups.
- **Secure aggregation** is simulated with pairwise additive masks; the code checks that the masks cancel. A real deployment would use cryptographic key agreement and finite-field arithmetic.

## Data

UCI Machine Learning Repository, *Diabetes 130-US Hospitals for Years 1999–2008* (CC BY 4.0). Strack et al., *BioMed Research International*, 2014. `run_all.sh` downloads it into `data/raw/`. The data is not committed to git.
