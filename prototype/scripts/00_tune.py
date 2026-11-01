"""Random search over gradient-boosting settings, scored on the validation set.

Objective: validation AUPRC. It rewards ranking readmitted patients near the top, like recall at
capacity, but is less noisy. The test set is never used here.
Output: results/tuning.json (read by the other scripts through models.hgb_params()).
"""
import json
import sys
import time
from pathlib import Path

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import ParameterSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safedischarge import metrics as M  # noqa: E402
from safedischarge.data import Encoder, load_clean, patient_split  # noqa: E402
from safedischarge.models import HGB_DEFAULT  # noqa: E402

N_TRIALS = 40
RES = Path(__file__).resolve().parents[1] / "results"
RES.mkdir(exist_ok=True)

sp = patient_split(load_clean())
enc = Encoder().fit(sp.train)
Xtr, Xva = enc.transform(sp.train), enc.transform(sp.val)

space = {
    "learning_rate": loguniform(0.01, 0.2),
    "max_iter": randint(150, 800),
    "max_leaf_nodes": randint(8, 64),
    "min_samples_leaf": randint(20, 400),
    "l2_regularization": loguniform(0.01, 10),
    "max_features": uniform(0.3, 0.7),
}
candidates = [HGB_DEFAULT] + list(ParameterSampler(space, N_TRIALS - 1, random_state=0))

trials = []
t0 = time.time()
for i, p in enumerate(candidates):
    p = {k: (round(float(v), 5) if isinstance(v, float) else int(v)) for k, v in p.items()}
    m = HistGradientBoostingClassifier(**p, class_weight="balanced", early_stopping=True,
                                       validation_fraction=0.1, random_state=0)
    m.fit(Xtr, sp.train.y)
    s = m.predict_proba(Xva)[:, 1]
    r = {"params": p, **M.overall(sp.val.y, s), "iterations_used": int(m.n_iter_)}
    trials.append(r)
    print(f"{i:2d} AUPRC {r['auprc']:.4f} AUROC {r['auroc']:.4f} recall@20% {r['recall_at_capacity']:.4f}")

best = max(trials, key=lambda r: r["auprc"])
out = {"objective": "validation AUPRC", "n_trials": len(trials), "seconds": round(time.time() - t0),
       "default": trials[0], "best": best, "best_params": best["params"], "trials": trials}
(RES / "tuning.json").write_text(json.dumps(out, indent=2, default=float))
print("default", trials[0]["auprc"], "best", best["auprc"], best["params"])
