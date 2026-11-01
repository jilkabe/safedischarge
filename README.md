# SafeDischarge

Fair, explainable and privacy-preserving 30-day readmission risk prediction for diabetic inpatients.
System design and working prototype for the NTU AI6101 (Introduction to AI and AI Ethics) mini-project.

**Report:** [report/SafeDischarge-final-report.pdf](report/SafeDischarge-final-report.pdf) · LaTeX source in [report/latex/](report/latex/)

## What it does

Each day SafeDischarge ranks diabetic patients due for discharge by their risk of readmission within 30 days, explains the top reasons for each flagged patient, and suggests a follow-up action for each reason. A discharge nurse uses the list to allocate a fixed number of follow-up slots.

It addresses all three Responsible AI practices, each with a measured result:

| Practice | Mechanism | Result (held-out test set) |
|---|---|---|
| Explainability | TreeSHAP top-3 reasons → follow-up actions; faithfulness test | Removing the top-3 reasons lowers risk 29%, vs 2.7% for 3 random variables |
| Fairness | Equal-opportunity audit with permutation tests; per-age-group thresholds | Age recall gap 21.5 → 6.8 points (p < 0.001 → 0.39) for −1.1 points of recall |
| Privacy | Federated learning, simulated secure aggregation, DP-SGD (Opacus), membership-inference test | Federated ≈ pooled (AUROC 0.675 vs 0.677); with DP the attack is at chance |

## Run it

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd prototype
./run_all.sh                 # downloads the data and reproduces every number in the report (~9 minutes)
uv run streamlit run app.py  # nurse dashboard at http://localhost:8501
```

See [prototype/README.md](prototype/README.md) for what each script does.

Rebuild the report PDF with [Tectonic](https://tectonic-typesetting.github.io): `report/latex/build.sh`.

## Layout

- `prototype/` — data pipeline, models, fairness audit, federated + DP training, explanation test, Streamlit dashboard
- `prototype/results/` — every number and figure used in the report
- `design/` — architecture diagram
- `report/latex/` — report source; `report/draft.md` is an earlier Markdown/Word version

## Data

UCI Machine Learning Repository, *Diabetes 130-US Hospitals for Years 1999–2008* (CC BY 4.0).
Strack et al., *BioMed Research International*, 2014. Downloaded by `run_all.sh`; not stored in this repository.

Research prototype for coursework. Not for clinical use.
