"""Evaluation metrics: ranking quality, recall at capacity, calibration, subgroup fairness."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

CAPACITY = 0.20  # share of discharges that can receive follow-up (team to set)
MIN_POS = 30     # minimum readmissions in a subgroup to report its recall


def flag_top_k(scores: np.ndarray, capacity: float = CAPACITY, seed: int = 0) -> np.ndarray:
    """Flag exactly the top `capacity` share of patients (ties broken at random)."""
    scores = np.asarray(scores, dtype=float)
    jitter = np.random.default_rng(seed).random(len(scores)) * 1e-9
    k = int(round(capacity * len(scores)))
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[np.argsort(-(scores + jitter))[:k]] = True
    return flagged


def recall_at_capacity(y, scores, capacity: float = CAPACITY) -> float:
    y = np.asarray(y)
    return float(flag_top_k(scores, capacity)[y == 1].mean())


def overall(y, scores, capacity: float = CAPACITY) -> dict:
    return {
        "auroc": round(float(roc_auc_score(y, scores)), 4),
        "auprc": round(float(average_precision_score(y, scores)), 4),
        "recall_at_capacity": round(recall_at_capacity(y, scores, capacity), 4),
        "brier": round(float(brier_score_loss(y, np.clip(scores, 0, 1))), 4),
    }


def group_recall(y, flagged, groups, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Recall (true positive rate) per group with bootstrap 95% CIs."""
    rng = np.random.default_rng(seed)
    y, flagged, groups = np.asarray(y), np.asarray(flagged), np.asarray(groups)
    rows = []
    for g in pd.unique(groups):
        m = (groups == g) & (y == 1)
        n_pos = int(m.sum())
        f = flagged[m]
        rec = float(f.mean()) if n_pos else np.nan
        if n_pos >= 2:
            boots = rng.choice(f, size=(n_boot, n_pos), replace=True).mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
        else:
            lo = hi = np.nan
        rows.append({"group": g, "n": int((groups == g).sum()), "n_pos": n_pos,
                     "recall": rec, "ci_low": lo, "ci_high": hi,
                     "flag_rate": float(flagged[groups == g].mean()),
                     "reportable": n_pos >= MIN_POS})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def recall_gap(table: pd.DataFrame) -> float:
    t = table[table["reportable"]]
    return round(float(t["recall"].max() - t["recall"].min()), 4) if len(t) > 1 else float("nan")


def fairness_report(y, scores, audit: pd.DataFrame, capacity: float = CAPACITY,
                    flagged: np.ndarray | None = None) -> dict:
    flagged = flag_top_k(scores, capacity) if flagged is None else flagged
    out = {}
    for col in ["race", "gender", "age_group"]:
        tab = group_recall(y, flagged, audit[col].to_numpy())
        out[col] = {"recall_gap": recall_gap(tab), "table": tab.round(4).to_dict("records")}
    return out


def gap_pvalue(y, flagged, groups, n_perm: int = 2000, seed: int = 0) -> float:
    """Permutation test: how often does shuffling group labels among readmitted patients
    produce a recall gap at least as large as the observed one? Small p = real disparity."""
    rng = np.random.default_rng(seed)
    y, flagged, groups = np.asarray(y), np.asarray(flagged), np.asarray(groups)
    f, g = flagged[y == 1], groups[y == 1]
    labels, counts = np.unique(g, return_counts=True)
    keep = labels[counts >= MIN_POS]

    def gap(gg):
        r = [f[gg == k].mean() for k in keep]
        return max(r) - min(r)

    obs = gap(g)
    null = np.array([gap(rng.permutation(g)) for _ in range(n_perm)])
    return round(float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1)), 4)
