"""Fairness audit and mitigation for the gradient-boosting model.

The audit (01_central.py) covers race, sex and age group at a fixed follow-up capacity
(top 20% of discharges flagged). Race groups other than Caucasian/African American have too
few readmissions for reliable estimates, and the large race groups are within tolerance.
The clear failure is AGE: patients aged 80+ are under-detected. Mitigations therefore target
age group, while race and sex are re-audited after every mitigation.

  baseline         - calibrated class-weighted gradient boosting
  reweighting      - pre-processing: upweight positives in under-detected age groups
  exp_gradient     - in-processing: Fairlearn ExponentiatedGradient, TruePositiveRateParity(age)
  group_thresholds - post-processing: capacity-aware per-age-group thresholds (fit on validation)

Outputs: results/fairness.json, results/figures/fairness_recall_by_age.png
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safedischarge import metrics as M  # noqa: E402
from safedischarge.data import load_clean, patient_split  # noqa: E402
from safedischarge.models import hgb_params  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
ATTR = "age_group"
art = joblib.load(RES / "artifacts.joblib")
enc = art["encoder"]

df = load_clean()
sp = patient_split(df)
Xtr, Xva, Xte = (enc.transform(d) for d in (sp.train, sp.val, sp.test))
ytr, yva, yte = sp.train.y.to_numpy(), sp.val.y.to_numpy(), sp.test.y.to_numpy()
gtr, gva, gte = (d[ATTR].to_numpy() for d in (sp.train, sp.val, sp.test))


def hgb(**kw):
    params = {k: v for k, v in hgb_params().items() if k != "class_weight"}  # weights set per method
    return HistGradientBoostingClassifier(**{**params, **kw})


def class_weights(y):
    p = y.mean()
    return np.where(y == 1, 0.5 / p, 0.5 / (1 - p))


def evaluate(scores, flagged=None, cal=None):
    """Rank on continuous `scores`; Brier uses calibrated probabilities `cal` when given."""
    flagged = M.flag_top_k(scores) if flagged is None else flagged
    out = {**M.overall(yte, scores), "flag_rate": round(float(flagged.mean()), 4),
           "recall_at_capacity": round(float(flagged[yte == 1].mean()), 4)}
    if cal is not None:
        out["brier"] = M.overall(yte, cal)["brier"]
    for col in ["age_group", "race", "gender"]:
        tab = M.group_recall(yte, flagged, sp.test[col].to_numpy())
        out[f"{col}_gap"] = M.recall_gap(tab)
        out[f"{col}_gap_p"] = M.gap_pvalue(yte, flagged, sp.test[col].to_numpy())
        out[f"{col}_table"] = tab.round(4).to_dict("records")
    return out


def raw_and_calibrated(model):
    iso = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(Xva)[:, 1], yva)
    raw = model.predict_proba(Xte)[:, 1]
    return raw, iso.predict(raw)


results = {}
# Rank on the raw (continuous) score: isotonic calibration creates many ties, which would make
# capacity-based flagging depend on tie-breaking. Calibrated values are used for display/Brier only.
base_val = art["model"].predict_proba(Xva)[:, 1]
base_te = art["model"].predict_proba(Xte)[:, 1]
base_cal = art["calibrator"].predict(base_te)
results["baseline"] = evaluate(base_te, cal=base_cal)

# 1. Pre-processing: upweight positives in each age group in proportion to its recall shortfall
#    on the validation set, so under-detected groups count more during training.
val_flag = M.flag_top_k(base_val)
val_rec = pd.Series({g: val_flag[(gva == g) & (yva == 1)].mean() for g in np.unique(gva)})
boost = (val_rec.max() / val_rec).clip(upper=2.0)
w = class_weights(ytr) * np.where(ytr == 1, pd.Series(gtr).map(boost).to_numpy(), 1.0)
m_rw = hgb().fit(Xtr, ytr, sample_weight=w / w.mean())
rw_raw, rw_cal = raw_and_calibrated(m_rw)
results["reweighting"] = evaluate(rw_raw, cal=rw_cal)
results["reweighting"]["positive_boost"] = boost.round(3).to_dict()

# 2. In-processing: exponentiated gradient with true-positive-rate parity across age groups.
eg = ExponentiatedGradient(hgb(max_iter=150, class_weight="balanced"),
                           constraints=TruePositiveRateParity(difference_bound=0.02),
                           max_iter=20, nu=1e-6)
eg.fit(Xtr, ytr, sensitive_features=gtr)
results["exp_gradient"] = evaluate(eg._pmf_predict(Xte)[:, 1])


# 3. Post-processing: per-age-group thresholds so every group has the same recall, with the
#    common recall chosen so total flags equal capacity (fit on validation data only).
def thresholds_for(target):
    return {g: np.quantile(base_val[(gva == g) & (yva == 1)], 1 - target) for g in np.unique(gva)}


lo, hi = 0.0, 1.0
for _ in range(40):
    mid = (lo + hi) / 2
    th = thresholds_for(mid)
    rate = np.mean(base_val >= np.vectorize(th.get)(gva))
    lo, hi = (mid, hi) if rate < M.CAPACITY else (lo, mid)
th = thresholds_for(lo)
# Apply as a per-group score shift, then flag exactly top-k so capacity is respected on test.
shift = {g: th[g] - np.median(list(th.values())) for g in th}
adj_te = base_te - np.vectorize(shift.get)(gte)
results["group_thresholds"] = evaluate(base_te, flagged=M.flag_top_k(adj_te), cal=base_cal)
results["group_thresholds"]["thresholds"] = {k: float(v) for k, v in th.items()}
results["group_thresholds"]["score_shift"] = {k: float(v) for k, v in shift.items()}

# Figure: recall by age group with 95% CIs
methods = ["baseline", "reweighting", "exp_gradient", "group_thresholds"]
labels = {"baseline": "Baseline", "reweighting": "Re-weighting", "exp_gradient": "Exp. gradient",
          "group_thresholds": "Group thresholds"}
ages = ["<40", "40-59", "60-79", "80+"]
fig, ax = plt.subplots(figsize=(7, 3.2))
width = 0.2
for i, mth in enumerate(methods):
    tab = {r["group"]: r for r in results[mth]["age_group_table"]}
    rec = [tab[g]["recall"] for g in ages]
    err = [[tab[g]["recall"] - tab[g]["ci_low"] for g in ages],
           [tab[g]["ci_high"] - tab[g]["recall"] for g in ages]]
    ax.bar(np.arange(4) + (i - 1.5) * width, rec, width, yerr=err, capsize=2,
           label=f"{labels[mth]} (gap {results[mth]['age_group_gap']:.2f}, "
                 f"overall {results[mth]['recall_at_capacity']:.2f})")
npos = {r["group"]: r["n_pos"] for r in results["baseline"]["age_group_table"]}
ax.set_xticks(np.arange(4), [f"{g}\n({npos[g]} readmitted)" for g in ages])
ax.set_ylabel("Recall at 20% capacity")
ax.set_ylim(0, 0.85)
ax.legend(fontsize=7, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "fairness_recall_by_age.png", dpi=200)

(RES / "fairness.json").write_text(json.dumps(results, indent=2, default=float))
cols = ["auroc", "recall_at_capacity", "brier", "age_group_gap", "age_group_gap_p", "race_gap", "race_gap_p", "gender_gap", "gender_gap_p"]
print(pd.DataFrame({m: {c: results[m][c] for c in cols} for m in methods}).T.to_string())
for m in methods:
    t = pd.DataFrame(results[m]["age_group_table"]).set_index("group")
    print(m, t.loc[ages, "recall"].round(3).to_dict())
