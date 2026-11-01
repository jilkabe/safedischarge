"""Faithfulness test for the per-patient explanations.

For every patient flagged at 20% capacity on the test set, take the top-3 risk-increasing reasons
from SHAP (one-hot columns of a variable summed, as in the dashboard). "Remove" those variables by
replacing them with the values of random training patients, and measure how much the risk score
falls. Compare with removing 3 variables chosen at random. If the explanations are faithful, removing
the top-3 reasons should lower risk much more than removing random variables.

Outputs: results/explain.json
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safedischarge import metrics as M  # noqa: E402
from safedischarge.data import load_clean, patient_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_BACKGROUND = 20  # random training patients used as replacement values per patient
rng = np.random.default_rng(0)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

art = joblib.load(RES / "artifacts.joblib")
enc, model, explainer = art["encoder"], art["model"], art["explainer"]
sp = patient_split(load_clean())
X_test = enc.transform(sp.test).to_numpy()
X_train = enc.transform(sp.train).to_numpy()
cols = enc.columns
var_of = np.array([c.split("=")[0] for c in cols])
variables = list(dict.fromkeys(var_of))
cols_of = {v: np.where(var_of == v)[0] for v in variables}

raw = model.predict_proba(X_test)[:, 1]
flagged = M.flag_top_k(raw)
cutoff = raw[flagged].min()
Xf = X_test[flagged]
base = raw[flagged]

sv = explainer.shap_values(pd.DataFrame(Xf, columns=cols))
sv = sv[1] if isinstance(sv, list) else sv
contrib = np.stack([sv[:, cols_of[v]].sum(axis=1) for v in variables], axis=1)  # patients x variables


def removed_score(chosen: list[list[int]]) -> np.ndarray:
    """Mean score after replacing the chosen variables with values from random training patients."""
    out = np.zeros(len(Xf))
    for b in range(N_BACKGROUND):
        Xb = Xf.copy()
        donors = X_train[rng.integers(0, len(X_train), len(Xf))]
        for i, vs in enumerate(chosen):
            for v in vs:
                c = cols_of[variables[v]]
                Xb[i, c] = donors[i, c]
        out += model.predict_proba(Xb)[:, 1]
    return out / N_BACKGROUND


top3 = [[v for v in np.argsort(-row)[:3] if row[v] > 0] for row in contrib]
rand3 = [list(rng.choice(len(variables), 3, replace=False)) for _ in range(len(Xf))]
s_top, s_rand = removed_score(top3), removed_score(rand3)


def summarise(s):
    drop = base - s
    return {"mean_score_drop": round(float(drop.mean()), 4),
            "mean_relative_drop": round(float((drop / base).mean()), 4),
            "share_leaving_flagged_list": round(float((s < cutoff).mean()), 4)}


res = {
    "flagged_patients": int(flagged.sum()),
    "mean_flagged_score": round(float(base.mean()), 4),
    "background_draws": N_BACKGROUND,
    "patients_with_3_positive_reasons": round(float(np.mean([len(t) == 3 for t in top3])), 4),
    "remove_top3_shap": summarise(s_top),
    "remove_random3": summarise(s_rand),
}
res["ratio_top3_vs_random"] = round(res["remove_top3_shap"]["mean_score_drop"]
                                    / res["remove_random3"]["mean_score_drop"], 1)
# Paired comparison: how often removing the top-3 lowers risk more than removing random variables
res["share_patients_top3_beats_random"] = round(float(((base - s_top) > (base - s_rand)).mean()), 4)
(RES / "explain.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))
