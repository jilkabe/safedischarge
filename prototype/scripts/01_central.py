"""Train and compare central models (logistic regression, gradient boosting, MLP),
calibrate the chosen model, and compute SHAP explanations.

Outputs: results/central.json, results/figures/*.png, results/artifacts.joblib
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import shap
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safedischarge import metrics as M  # noqa: E402
from safedischarge.data import AUDIT, Encoder, load_clean, patient_split, summary  # noqa: E402
from safedischarge.models import MLP, hgb_params, make_loader, predict_proba, train_epochs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)

df = load_clean()
sp = patient_split(df)
pos_w = float((1 - sp.train.y.mean()) / sp.train.y.mean())
results = {"data": summary(df), "split": {k: len(getattr(sp, k)) for k in ("train", "val", "test")}}


def fit_lr(enc):
    m = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
    m.fit(enc.transform(sp.train, scale=True), sp.train.y)
    return m, lambda d: m.predict_proba(enc.transform(d, scale=True))[:, 1]


def fit_hgb(enc):
    m = HistGradientBoostingClassifier(**hgb_params())  # tuned by 00_tune.py
    m.fit(enc.transform(sp.train), sp.train.y)
    return m, lambda d: m.predict_proba(enc.transform(d))[:, 1]


def fit_mlp(enc):
    Xtr = enc.transform(sp.train, scale=True).to_numpy()
    m = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    train_epochs(m, make_loader(Xtr, sp.train.y.to_numpy()), opt, epochs=8, pos_weight=pos_w)
    return m, lambda d: predict_proba(m, enc.transform(d, scale=True).to_numpy())


# 1. Model comparison (race NOT used as a feature)
enc = Encoder(use_race=False).fit(sp.train)
models = {}
results["models"] = {}
for name, fit in [("logistic_regression", fit_lr), ("gradient_boosting", fit_hgb), ("mlp", fit_mlp)]:
    model, score = fit(enc)
    models[name] = (model, score)
    s = score(sp.test)
    results["models"][name] = {
        **M.overall(sp.test.y, s),
        "race_recall_gap": M.fairness_report(sp.test.y, s, sp.test[AUDIT])["race"]["recall_gap"],
    }
    print(name, results["models"][name])

# 2. Race as an input: does including it change accuracy or fairness?
enc_r = Encoder(use_race=True).fit(sp.train)
_, score_r = fit_hgb(enc_r)
s_r = score_r(sp.test)
results["race_as_feature"] = {
    "without_race": results["models"]["gradient_boosting"],
    "with_race": {**M.overall(sp.test.y, s_r),
                  "race_recall_gap": M.fairness_report(sp.test.y, s_r, sp.test[AUDIT])["race"]["recall_gap"]},
}

# 3. Calibrate chosen model (gradient boosting) on validation set
hgb, score_hgb = models["gradient_boosting"]
iso = IsotonicRegression(out_of_bounds="clip").fit(score_hgb(sp.val), sp.val.y)
raw_test = score_hgb(sp.test)
cal_test = iso.predict(raw_test)
results["calibration"] = {"brier_raw": M.overall(sp.test.y, raw_test)["brier"],
                          "brier_calibrated": M.overall(sp.test.y, cal_test)["brier"]}
fig, ax = plt.subplots(figsize=(4, 4))
for lab, s in [("raw (class-weighted)", raw_test), ("isotonic-calibrated", cal_test)]:
    bins = np.quantile(s, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(s, bins[1:-1]), 0, 9)
    ax.plot([s[idx == b].mean() for b in range(10)], [sp.test.y.to_numpy()[idx == b].mean() for b in range(10)],
            "o-", label=lab)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set(xlabel="Predicted risk", ylabel="Observed 30-day readmission", xlim=(0, 1), ylim=(0, 0.5))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "calibration.png", dpi=200)
plt.close(fig)

# 4. Full fairness report for the chosen model (baseline for 02_fairness.py)
results["fairness_baseline"] = M.fairness_report(sp.test.y, raw_test, sp.test[AUDIT])  # rank on raw score (no ties)

# 5. SHAP explanations
X_test = enc.transform(sp.test)
explainer = shap.TreeExplainer(hgb)
sample = X_test.sample(2000, random_state=0)
sv = explainer.shap_values(sample)
sv = sv[1] if isinstance(sv, list) else sv
mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(mean_abs)[::-1][:15]
results["shap_top15"] = [{"feature": sample.columns[i], "mean_abs_shap": round(float(mean_abs[i]), 4)}
                         for i in order]
plt.figure()
shap.summary_plot(sv, sample, max_display=15, show=False, plot_size=(7, 5))
plt.tight_layout()
plt.savefig(FIG / "shap_summary.png", dpi=200)
plt.close()

joblib.dump({"encoder": enc, "model": hgb, "calibrator": iso, "explainer": explainer,
             "threshold_capacity": M.CAPACITY}, RES / "artifacts.joblib")
(RES / "central.json").write_text(json.dumps(results, indent=2, default=float))
print(json.dumps({k: results[k] for k in ("data", "race_as_feature", "calibration", "shap_top15")},
                 indent=1, default=float))
