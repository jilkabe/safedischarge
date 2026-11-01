"""Model definitions shared by central and federated training."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

torch.set_num_threads(4)


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_loader(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
                batch_size: int = 256, shuffle: bool = True, seed: int = 0):
    w = np.ones(len(y)) if w is None else w
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32),
    )
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


def train_epochs(model: nn.Module, loader, optimizer, epochs: int = 1, pos_weight: float = 1.0):
    model.train()
    for _ in range(epochs):
        for xb, yb, wb in loader:
            if len(yb) == 0:  # Poisson sampling (DP) can yield empty batches
                continue
            optimizer.zero_grad()
            logits = model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, weight=wb * (1 + (pos_weight - 1) * yb), reduction="mean")
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


@torch.no_grad()
def per_example_loss(model: nn.Module, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    model.eval()
    logits = model(torch.tensor(X, dtype=torch.float32))
    return nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor(y, dtype=torch.float32), reduction="none").numpy()


# Gradient-boosting settings. scripts/00_tune.py searches these on the validation set and
# writes results/tuning.json; every script reads the tuned values through hgb_params().
HGB_DEFAULT = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                   l2_regularization=1.0)


def hgb_params() -> dict:
    import json
    from pathlib import Path
    f = Path(__file__).resolve().parents[2] / "results" / "tuning.json"
    best = json.loads(f.read_text())["best_params"] if f.exists() else HGB_DEFAULT
    return {**best, "class_weight": "balanced", "early_stopping": True,
            "validation_fraction": 0.1, "random_state": 0}
