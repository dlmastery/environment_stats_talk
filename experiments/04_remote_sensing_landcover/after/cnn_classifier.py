"""AFTER — a small PyTorch CNN on the raw 5-band patches.

Where the BEFORE pipeline throws away the spatial structure (it averages every
patch down to four numbers), the CNN learns *texture* directly from the raw
(N, 5, H, W) reflectance cube. The model is deliberately small so it trains in
minutes on CPU for tests, but it AUTO-USES CUDA when a GPU is present (the repo
box has an RTX 4090, where the headline run uses more data / epochs).

Design choices (all standard, all defensible to a skeptical audience):
- 3 conv blocks (Conv-BN-ReLU + MaxPool) -> global average pool -> linear head.
- AdamW optimiser with weight decay (decoupled L2, the modern default).
- Early stopping on a validation split (patience), restoring the best weights.
- Deterministic seeding so committed metrics reproduce.

Contract (mirrors the BEFORE script):

    train_cnn(X_train, y_train, ...) -> (model, history)
    predict(model, X, device) -> y_pred (int labels into common.CLASSES)
    run_after(X_train, y_train, X_test, y_test, ...) -> metrics dict

Run from the REPO ROOT so ``import common`` resolves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from common import metrics, BANDS, CLASSES  # noqa: E402

__all__ = [
    "SmallCNN", "pick_device", "set_seed", "train_cnn", "predict",
    "run_after", "embedding_knn_baseline",
]

N_BANDS = len(BANDS)
N_CLASSES = len(CLASSES)


def pick_device(prefer_cuda: bool = True) -> torch.device:
    """CPU for tests; auto-use CUDA when available (RTX 4090 on the repo box)."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 0) -> None:
    """Seed python/numpy/torch RNGs for reproducible committed metrics."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SmallCNN(nn.Module):
    """A compact 5-band patch classifier: 3 conv blocks -> GAP -> linear head.

    Global average pooling makes the head independent of patch size, so the same
    architecture works on 8x8 test patches and larger real Sentinel-2 chips.
    """

    def __init__(self, in_ch: int = N_BANDS, n_classes: int = N_CLASSES,
                 width: int = 16):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_ch, width),
            block(width, width * 2),
            block(width * 2, width * 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(width * 4, n_classes),
        )

    def forward(self, x):  # x: (B, 5, H, W)
        return self.head(self.features(x))


def _loaders(X_train, y_train, val_frac, batch_size, seed):
    """Make train/val DataLoaders from a deterministic shuffle of the train set."""
    rng = np.random.default_rng(seed)
    n = len(X_train)
    idx = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def ds(ii):
        return TensorDataset(
            torch.as_tensor(np.asarray(X_train)[ii], dtype=torch.float32),
            torch.as_tensor(np.asarray(y_train)[ii], dtype=torch.long),
        )

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(ds(tr_idx), batch_size=batch_size, shuffle=True,
                              generator=g)
    val_loader = DataLoader(ds(val_idx), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_cnn(X_train, y_train, *, epochs: int = 20, batch_size: int = 64,
              lr: float = 1e-3, weight_decay: float = 1e-4, val_frac: float = 0.2,
              patience: int = 5, width: int = 16, seed: int = 0,
              device: torch.device | None = None, verbose: bool = False):
    """Train ``SmallCNN`` with AdamW + early stopping; restore best weights.

    Returns ``(model, history)`` where history holds per-epoch train/val loss and
    val accuracy plus the best epoch. Early stopping monitors validation loss.
    """
    set_seed(seed)
    device = device or pick_device()
    model = SmallCNN(width=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, val_loader = _loaders(X_train, y_train, val_frac, batch_size, seed)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "best_epoch": 0}
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            tl += float(loss) * len(xb)
        tl /= max(1, len(train_loader.dataset))

        # validation
        model.eval()
        vl, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl += float(loss_fn(out, yb)) * len(xb)
                correct += int((out.argmax(1) == yb).sum())
                total += len(xb)
        vl /= max(1, total)
        va = correct / max(1, total)

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        if verbose:
            print(f"epoch {epoch+1:3d}  train_loss={tl:.4f}  val_loss={vl:.4f}  val_acc={va:.3f}")

        if vl < best_val - 1e-5:
            best_val = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, X, device: torch.device | None = None,
            batch_size: int = 256) -> np.ndarray:
    """Predict integer land-cover labels for raw (N,5,H,W) patches."""
    device = device or next(model.parameters()).device
    model.eval()
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    preds = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds).astype(int)


def run_after(X_train, y_train, X_test, y_test, *, epochs: int = 20,
              batch_size: int = 64, lr: float = 1e-3, width: int = 16,
              patience: int = 5, seed: int = 0,
              device: torch.device | None = None, verbose: bool = False) -> dict:
    """Train the CNN and evaluate on the held-out test patches.

    Returns a metrics dict mirroring ``before.rf_indices.run_before`` plus the
    training history and the device used.
    """
    device = device or pick_device()
    model, history = train_cnn(
        X_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr,
        width=width, patience=patience, seed=seed, device=device, verbose=verbose,
    )
    y_pred = predict(model, X_test, device=device)
    rep = metrics.classification_report_simple(y_test, y_pred)
    n_params = int(sum(p.numel() for p in model.parameters()))
    return {
        "model": "SmallCNN-on-bands",
        "device": str(device),
        "accuracy": rep["accuracy"],
        "macro_f1": rep["macro_f1"],
        "macro_precision": rep["macro_precision"],
        "macro_recall": rep["macro_recall"],
        "n_params": n_params,
        "best_epoch": int(history["best_epoch"]),
        "epochs_run": int(len(history["train_loss"])),
        "history": history,
        "y_pred": np.asarray(y_pred, int).tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def embedding_knn_baseline(X_train, y_train, X_test, y_test, *,
                           encoder=None, k: int = 5):
    """OPTIONAL — pretrained-RS-encoder embeddings + kNN (documented, skippable).

    The modern "foundation-model" route to land-cover classification does not
    train a CNN from scratch at all. Instead you take a *pretrained* remote-sensing
    encoder -- e.g. a Sentinel-2 self-supervised backbone such as SatMAE / Prithvi
    / Clay, or the vision tower of a remote-sensing MLLM in the spirit of
    EagleVision (arXiv:2503.23330) -- run every patch through it to get a fixed
    embedding vector, and classify those embeddings with a cheap kNN (or a linear
    probe). With a good encoder this is competitive *without any task-specific
    training*, which is the whole point of foundation models.

    This function is intentionally a thin, dependency-free shim:

    - If you pass an ``encoder`` callable mapping (N,5,H,W)->(N,D) embeddings, we
      run kNN on those embeddings and return real accuracy / macro-F1.
    - If no encoder is supplied (the default, and the case in CI / runs-anywhere),
      we DO NOT fabricate a result: we return ``{"skipped": True, ...}`` with a
      note. No invented numbers ever reach a slide.

    To wire a real encoder on the 4090::

        from some_rs_backbone import load_encoder        # e.g. SatMAE / Prithvi
        enc = load_encoder(pretrained=True).eval()
        def encode(x):  # (N,5,H,W) float -> (N,D) numpy
            import torch
            with torch.no_grad():
                return enc(torch.as_tensor(x)).cpu().numpy()
        embedding_knn_baseline(Xtr, ytr, Xte, yte, encoder=encode, k=5)
    """
    if encoder is None:
        return {
            "skipped": True,
            "reason": "no pretrained RS encoder supplied (runs-anywhere default)",
            "how_to_enable": "pass encoder=callable mapping (N,5,H,W)->(N,D); "
                             "see docstring for SatMAE/Prithvi/Clay wiring",
        }

    from sklearn.neighbors import KNeighborsClassifier

    emb_tr = np.asarray(encoder(np.asarray(X_train)))
    emb_te = np.asarray(encoder(np.asarray(X_test)))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(emb_tr, y_train)
    y_pred = knn.predict(emb_te).astype(int)
    rep = metrics.classification_report_simple(y_test, y_pred)
    return {
        "skipped": False,
        "model": f"pretrained-encoder + {k}NN",
        "accuracy": rep["accuracy"],
        "macro_f1": rep["macro_f1"],
        "embedding_dim": int(emb_tr.shape[1]),
        "y_pred": y_pred.tolist(),
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common import multispectral_patches

    Xtr, ytr = multispectral_patches(n=600, size=16, seed=0)
    Xte, yte = multispectral_patches(n=200, size=16, seed=1)
    res = run_after(Xtr, ytr, Xte, yte, epochs=15, seed=0, verbose=True)
    print("== AFTER: SmallCNN on raw 5-band patches ==")
    print(f"  device   : {res['device']}")
    print(f"  params   : {res['n_params']:,}")
    print(f"  accuracy : {res['accuracy']:.3f}")
    print(f"  macro-F1 : {res['macro_f1']:.3f}  (best epoch {res['best_epoch']})")
