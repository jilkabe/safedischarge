"""Federated learning simulation with secure aggregation and differential privacy.

Settings compared (same MLP, same test set):
  local_only    - each virtual hospital trains alone (the status quo)
  central       - all data pooled (upper bound; not allowed in practice)
  fedavg        - federated averaging with secure aggregation
  fedavg_dp_e*  - fedavg + per-hospital DP-SGD (Opacus) at several privacy budgets

Also runs a loss-threshold membership-inference attack (Yeom et al., 2018) on each model,
plus a deliberately over-fitted model as a positive control showing the attack works.

Outputs: results/federated.json, results/figures/privacy_tradeoff.png
"""
import copy
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safedischarge import metrics as M  # noqa: E402
from safedischarge.data import dirichlet_hospitals, load_clean, patient_split  # noqa: E402
from safedischarge.models import MLP, make_loader, per_example_loss, predict_proba, train_epochs  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
N_HOSP, ROUNDS, LOCAL_EPOCHS, BATCH, LR = 5, 20, 1, 256, 1e-3
EPSILONS, DELTA, CLIP = [1.0, 3.0, 8.0], 1e-5, 1.0
torch.manual_seed(0)

enc = joblib.load(RES / "artifacts.joblib")["encoder"]
df = load_clean()
sp = patient_split(df)
Xtr = enc.transform(sp.train, scale=True).to_numpy(np.float32)
Xte = enc.transform(sp.test, scale=True).to_numpy(np.float32)
ytr, yte = sp.train.y.to_numpy(), sp.test.y.to_numpy()
hosp = dirichlet_hospitals(sp.train, N_HOSP)
pos_w = float((1 - ytr.mean()) / ytr.mean())
n_in = Xtr.shape[1]

results = {"config": dict(hospitals=N_HOSP, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, batch=BATCH,
                          lr=LR, delta=DELTA, clip=CLIP, dirichlet_alpha=0.5)}
results["hospitals"] = [
    {"hospital": h, "encounters": int((hosp == h).sum()), "readmit_rate": round(float(ytr[hosp == h].mean()), 4),
     **{f"share_{r}": round(float((sp.train.race[hosp == h] == r).mean()), 3) for r in ["Caucasian", "AfricanAmerican"]},
     "share_80plus": round(float((sp.train.age_group[hosp == h] == "80+").mean()), 3)}
    for h in range(N_HOSP)]


def evaluate(model):
    s = predict_proba(model, Xte)
    flagged = M.flag_top_k(s)
    out = M.overall(yte, s)
    for col in ["age_group", "race", "gender"]:
        out[f"{col}_gap"] = M.recall_gap(M.group_recall(yte, flagged, sp.test[col].to_numpy()))
    return out


def membership_auc(model, n=5000, seed=0):
    """Loss-threshold attack: lower loss -> guess 'member'. AUC 0.5 = no leakage."""
    rng = np.random.default_rng(seed)
    mi = rng.choice(len(Xtr), n, replace=False)
    ni = rng.choice(len(Xte), n, replace=False)
    loss = np.concatenate([per_example_loss(model, Xtr[mi], ytr[mi]), per_example_loss(model, Xte[ni], yte[ni])])
    member = np.r_[np.ones(n), np.zeros(n)]
    return round(float(roc_auc_score(member, -loss)), 4)


# ---------- secure aggregation (pairwise additive masking, Bonawitz et al. 2017) ----------
def flatten(sd):
    return np.concatenate([v.detach().cpu().numpy().ravel().astype(np.float64) for v in sd.values()])


def unflatten(vec, like):
    out, i = {}, 0
    for k, v in like.items():
        n = v.numel()
        out[k] = torch.tensor(vec[i:i + n].reshape(v.shape), dtype=v.dtype)
        i += n
    return out


def secure_sum(vectors, round_seed):
    """Each pair (i<j) derives a shared random mask; i adds it, j subtracts it.
    The server only ever sees masked vectors, but the masks cancel in the sum."""
    k, dim = len(vectors), len(vectors[0])
    masked = [v.copy() for v in vectors]
    for i in range(k):
        for j in range(i + 1, k):
            mask = np.random.default_rng([round_seed, i, j]).normal(0, 1e8, dim)
            masked[i] += mask
            masked[j] -= mask
    return np.sum(masked, axis=0), masked


# ---------- training settings ----------
def train_central(epochs=8, hidden=64, weight_decay=1e-4):
    m = MLP(n_in, hidden)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=weight_decay)
    train_epochs(m, make_loader(Xtr, ytr, batch_size=BATCH), opt, epochs=epochs, pos_weight=pos_w)
    return m


def train_local_only(epochs=8):
    out = []
    for h in range(N_HOSP):
        idx = hosp == h
        m = MLP(n_in)
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-4)
        train_epochs(m, make_loader(Xtr[idx], ytr[idx], batch_size=BATCH, seed=h), opt, epochs=epochs, pos_weight=pos_w)
        out.append({"hospital": h, **evaluate(m)})
    return out


def train_fedavg(target_eps=None, check_secure=False):
    """FedAvg over the virtual hospitals; optional per-hospital DP-SGD and secure-sum check."""
    global_model = MLP(n_in)
    clients = []
    for h in range(N_HOSP):
        idx = hosp == h
        model = MLP(n_in)
        opt = torch.optim.Adam(model.parameters(), lr=LR * (2 if target_eps else 1))
        loader = make_loader(Xtr[idx], ytr[idx], batch_size=BATCH, seed=h)
        engine = None
        if target_eps is not None:
            sample_rate = BATCH / idx.sum()
            steps = ROUNDS * LOCAL_EPOCHS * int(np.ceil(idx.sum() / BATCH))
            sigma = get_noise_multiplier(target_epsilon=target_eps, target_delta=DELTA,
                                         sample_rate=sample_rate, steps=steps, accountant="rdp")
            engine = PrivacyEngine(accountant="rdp")
            model, opt, loader = engine.make_private(module=model, optimizer=opt, data_loader=loader,
                                                     noise_multiplier=sigma, max_grad_norm=CLIP)
        clients.append(dict(model=model, opt=opt, loader=loader, engine=engine, n=int(idx.sum())))

    total = sum(c["n"] for c in clients)
    max_err = 0.0
    for r in range(ROUNDS):
        gsd = global_model.state_dict()
        updates = []
        for c in clients:
            base = c["model"]._module if c["engine"] else c["model"]
            base.load_state_dict(gsd)
            train_epochs(c["model"], c["loader"], c["opt"], epochs=LOCAL_EPOCHS, pos_weight=pos_w)
            updates.append(flatten(base.state_dict()) * c["n"])  # weight by hospital size
        agg, masked = secure_sum(updates, round_seed=r)
        if check_secure:
            max_err = max(max_err, float(np.abs(agg - np.sum(updates, axis=0)).max()))
        global_model.load_state_dict(unflatten(agg / total, gsd))

    info = {}
    if target_eps is not None:
        info["epsilon_spent_per_hospital"] = [round(float(c["engine"].get_epsilon(DELTA)), 3) for c in clients]
    if check_secure:
        info["secure_agg_max_abs_error"] = max_err
        info["masked_vs_plain_correlation"] = round(float(np.corrcoef(masked[0], updates[0])[0, 1]), 4)
    return global_model, info


SEEDS = [0, 1, 2]
runs = {}  # setting -> list of metric dicts (one per seed)


def add(name, d):
    runs.setdefault(name, []).append(d)


for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    local = train_local_only()
    if seed == 0:
        results["local_only_per_hospital"] = local
    add("local_only (mean of 5)", {k: float(np.mean([l[k] for l in local])) for k in local[0] if k != "hospital"})
    central = train_central()
    add("central (pooled)", {**evaluate(central), "membership_auc": membership_auc(central, seed=seed)})
    fed, info = train_fedavg(check_secure=(seed == 0))
    if seed == 0:
        results["secure_aggregation_check"] = info
    add("fedavg + secure agg", {**evaluate(fed), "membership_auc": membership_auc(fed, seed=seed)})
    for eps in EPSILONS:
        m, info = train_fedavg(target_eps=eps)
        add(f"fedavg + DP (eps={eps:g})", {**evaluate(m), "membership_auc": membership_auc(m, seed=seed),
                                           "max_epsilon_spent": max(info["epsilon_spent_per_hospital"])})
    overfit = train_central(epochs=60, hidden=512, weight_decay=0.0)
    add("over-fitted control (no DP)", {**evaluate(overfit), "membership_auc": membership_auc(overfit, seed=seed)})
    print("done seed", seed, flush=True)

settings = {}
for name, lst in runs.items():
    keys = [k for k in lst[0] if isinstance(lst[0][k], (int, float))]
    settings[name] = {**{k: round(float(np.nanmean([d[k] for d in lst])), 4) for k in keys},
                      **{f"{k}_sd": round(float(np.nanstd([d[k] for d in lst])), 4) for k in keys}}
results["seeds"] = SEEDS
results["settings"] = settings
(RES / "federated.json").write_text(json.dumps(results, indent=2, default=float))

table = pd.DataFrame(settings).T
cols = ["auroc", "auroc_sd", "recall_at_capacity", "recall_at_capacity_sd", "age_group_gap", "age_group_gap_sd",
        "race_gap", "gender_gap", "membership_auc", "membership_auc_sd", "max_epsilon_spent"]
print(pd.DataFrame(results["hospitals"]).to_string())
print(results["secure_aggregation_check"])
print(table[[c for c in cols if c in table]].to_string())

# Figure: privacy-utility trade-off (mean +/- sd over seeds)
keys = [f"fedavg + DP (eps={e:g})" for e in EPSILONS] + ["fedavg + secure agg"]
xs = np.arange(len(keys))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
for metric, ax, title in [("auroc", a1, "AUROC"), ("membership_auc", a2, "Membership-inference AUC")]:
    ax.errorbar(xs, [settings[k][metric] for k in keys], yerr=[settings[k][f"{metric}_sd"] for k in keys],
                fmt="o-", capsize=3, label="Federated")
    ref = [("central (pooled)", "--", "Pooled (central)")]
    if metric == "auroc":
        ref.append(("local_only (mean of 5)", ":", "Single hospital"))
    else:
        ref.append(("over-fitted control (no DP)", "-.", "Over-fitted control"))
    for k, ls, lab in ref:
        ax.axhline(settings[k][metric], ls=ls, c="grey", lw=1, label=lab)
    if metric == "membership_auc":
        ax.axhline(0.5, c="k", lw=0.6)
    ax.set_xticks(xs, [f"ε={e:g}" for e in EPSILONS] + ["no DP"])
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6.5)
a1.set_ylim(0.62, 0.69)
a2.set_ylim(0.48, 0.61)
fig.tight_layout()
fig.savefig(FIG / "privacy_tradeoff.png", dpi=200)
