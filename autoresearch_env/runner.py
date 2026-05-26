"""Minimal env-stats experiment runner — ONE experiment per call.

This is the lightweight, self-contained heart of the env-stats autoresearch loop.
It re-implements (does not import) the orchestration contract of the source
framework's ``core/runner.py`` and keeps only what the talk needs:

  (a) load a config dict;
  (b) REFUSE to run unless a valid pre-run reasoning entry exists
      (Citation-Rigor + Reasoning-Completeness gates from ``reasoning.py``);
  (c) check the composite-metric fingerprint against the project's frozen one
      (Goodhart protection from ``composite.py``);
  (d) build the chosen backbone (a tiny MLP or the persistence baseline) on a
      lag-feature matrix derived from ``common.daily_temperature`` — runs on CPU,
      no GPU, no keys;
  (e) evaluate with a walk-forward or super-fold splitter (``splits.py``) and the
      env composite metric;
  (f) append a row to ``experiment_log.jsonl``, write a champion record
      (``best_config.json``) whenever the composite improves, and rewrite a
      crash-recovery checkpoint markdown.

Claude Code is the OUTER loop: Claude authors the pre-run reasoning, calls
``run_experiment`` for ONE experiment, reads the result, then authors the next
reasoning entry. The runner never invents reasoning — it only enforces it.

Backbones live here (tiny, dependency-light) so the package is self-contained:
  - "persistence"  : carries the last observed value forward (the skill reference).
  - "mlp"          : a 1-hidden-layer numpy MLP trained by gradient descent on the
                     standardized lag features. Deterministic given a seed; trains
                     in milliseconds on CPU. (A torch/CUDA backbone is the 4090
                     upgrade path documented in the README.)

Reuses this repo's ``common`` package (daily_temperature, metrics). Does NOT import
the source autoresearch repo.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Make `import common` resolve whether run via pytest or as a script from the repo.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import metrics as env_metrics  # noqa: E402
from common import daily_temperature  # noqa: E402

# Package-relative when imported as autoresearch_env.runner; flat when run directly.
if __package__:
    from .reasoning import ReasoningAnnotationsFile, ReasoningEntry, validate_pre_run_entry
    from .composite import CompositeCalculator
    from .splits import create_splitter, validate_no_overlap
else:  # pragma: no cover - direct-script convenience
    from reasoning import ReasoningAnnotationsFile, ReasoningEntry, validate_pre_run_entry
    from composite import CompositeCalculator
    from splits import create_splitter, validate_no_overlap


# ------------------------------------------------------------------ record

@dataclass
class ExperimentRecord:
    experiment_num: int
    backbone: str
    description: str
    composite: float
    val_primary: float
    test_primary: float
    per_fold_test: list[float]
    per_fold_val: list[float]
    status: str
    seconds_elapsed: float
    timestamp: str
    composite_fingerprint: str
    primary_metric: str
    per_fold_reports: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ data

def _make_lag_features(series: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a supervised one-step-ahead matrix from a 1-D series.

    X[i] = [series[i-1], ..., series[i-n_lags], doy_sin, doy_cos]; y[i] = series[i].
    Rows where the full lag window is unavailable are dropped, so X/y are aligned
    and there is no leakage (every feature precedes its target).
    """
    series = np.asarray(series, dtype=float).ravel()
    n = len(series)
    rows_X, rows_y = [], []
    for i in range(n_lags, n):
        lags = series[i - n_lags:i][::-1]  # most-recent first
        doy = (i % 365) / 365.0
        feats = np.concatenate([lags, [np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy)]])
        rows_X.append(feats)
        rows_y.append(series[i])
    return np.asarray(rows_X), np.asarray(rows_y)


def load_dataset(cfg: dict[str, Any]):
    """Load the synthetic climate series and turn it into (X, y, regimes).

    Config keys (all optional, with env-friendly defaults):
      data.n_years, data.seed, data.n_lags, data.n_regimes
    Regimes are synthetic climate states assigned by binning a slow warm/cool
    oscillation (an ENSO-like phase) so SuperFoldSplit has something to group on.
    """
    data_cfg = cfg.get("data", {})
    n_years = int(data_cfg.get("n_years", 12))
    seed = int(data_cfg.get("seed", 0))
    n_lags = int(data_cfg.get("n_lags", 7))
    n_regimes = int(data_cfg.get("n_regimes", 3))

    df = daily_temperature(n_years=n_years, seed=seed)
    series = df["t2m"].to_numpy()
    X, y = _make_lag_features(series, n_lags)

    # Synthetic regime label per (post-lag) sample: a slow oscillation binned into
    # n_regimes phases — a stand-in for ENSO-like warm/neutral/cool states.
    t = np.arange(len(y))
    phase = np.sin(2 * np.pi * t / max(1, len(y) / 3.0))
    regimes = np.clip(((phase + 1) / 2 * n_regimes).astype(int), 0, n_regimes - 1)
    return X, y, regimes


# ------------------------------------------------------------------ backbones

def _standardize(X_tr, X_other):
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-8
    out = [(X_tr - mu) / sd]
    for X in X_other:
        out.append((X - mu) / sd if X is not None else None)
    return (*out, mu, sd)


def _fit_predict_persistence(X_tr, y_tr, X_te):
    """Persistence: predict the most recent lag (column 0 of the lag block)."""
    return X_te[:, 0].copy()


def _fit_predict_mlp(X_tr, y_tr, X_va, y_va, X_te, cfg, seed):
    """Tiny 1-hidden-layer numpy MLP (tanh), trained by full-batch gradient descent.

    Deterministic given the seed. Standardizes targets for stable training and
    un-standardizes predictions. Pure numpy -> runs in milliseconds on CPU.
    """
    rng = np.random.default_rng(seed)
    hidden = int(cfg.get("hidden", 16))
    epochs = int(cfg.get("epochs", 300))
    lr = float(cfg.get("lr", 0.05))
    l2 = float(cfg.get("weight_decay", 1e-4))

    Xtr, Xva, Xte, mu, sd = _standardize(X_tr, [X_va, X_te])
    y_mu, y_sd = y_tr.mean(), y_tr.std() + 1e-8
    ytr = (y_tr - y_mu) / y_sd

    n_in = Xtr.shape[1]
    W1 = rng.normal(0, 1.0 / np.sqrt(n_in), size=(n_in, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1.0 / np.sqrt(hidden), size=(hidden, 1))
    b2 = np.zeros(1)

    n = len(Xtr)
    for _ in range(epochs):
        h_pre = Xtr @ W1 + b1
        h = np.tanh(h_pre)
        out = (h @ W2 + b2).ravel()
        err = out - ytr  # (n,)
        # gradients (MSE / n) + L2
        g_out = (2.0 / n) * err[:, None]  # (n,1)
        gW2 = h.T @ g_out + l2 * W2
        gb2 = g_out.sum(axis=0)
        g_h = g_out @ W2.T  # (n,hidden)
        g_hpre = g_h * (1 - h ** 2)
        gW1 = Xtr.T @ g_hpre + l2 * W1
        gb1 = g_hpre.sum(axis=0)
        W2 -= lr * gW2; b2 -= lr * gb2
        W1 -= lr * gW1; b1 -= lr * gb1

    def _predict(Xs):
        h = np.tanh(Xs @ W1 + b1)
        return (h @ W2 + b2).ravel() * y_sd + y_mu

    return _predict(Xte)


# ------------------------------------------------------------------ metrics

def _primary(name: str, y_true, y_pred, reference_pred=None) -> float:
    """Compute the env-stats primary metric by name (delegates to common.metrics)."""
    if name in ("rmse", "lat_weighted_rmse"):  # 1-D series: lat-weighting is identity
        return env_metrics.rmse(y_true, y_pred)
    if name == "mae":
        return env_metrics.mae(y_true, y_pred)
    if name == "acc":
        return env_metrics.anomaly_correlation(y_true, y_pred)
    if name == "skill_vs_persistence":
        ref = env_metrics.rmse(y_true, reference_pred)
        mod = env_metrics.rmse(y_true, y_pred)
        return env_metrics.skill_score(mod, ref, higher_is_better=False)
    raise ValueError(f"Unknown primary metric {name!r}")


# ------------------------------------------------------------------ runner core

def _next_experiment_num(log_path: Path) -> int:
    if not log_path.exists():
        return 1
    max_num = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            max_num = max(max_num, int(json.loads(line).get("experiment_num", 0)))
        except Exception:
            continue
    return max_num + 1


def _frozen_fingerprint(log_path: Path) -> str | None:
    """Return the composite fingerprint recorded on the FIRST experiment, if any."""
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line).get("composite_fingerprint")
        except Exception:
            continue
    return None


def _write_checkpoint(results_dir: Path, record: ExperimentRecord, champion: dict | None,
                      next_command: str) -> None:
    """Rewrite the self-contained crash-recovery checkpoint markdown."""
    ckpt = results_dir / "project_autoresearch_checkpoint.md"
    champ_line = (
        f"Exp{champion['experiment_num']} ({champion['backbone']}) "
        f"composite={champion['composite']:.4f}"
        if champion else "none yet"
    )
    lines = [
        "# Env-Stats AutoResearch — Crash-Recovery Checkpoint",
        "",
        f"_Last updated: {record.timestamp}_  ",
        "Driver: Claude Code (outer loop). Hardware: CPU smoke / RTX 4090 for full runs.",
        "",
        "## Current champion",
        f"- {champ_line}",
        f"- Primary metric: `{record.primary_metric}` "
        f"(composite = min(val,test) − penalty·n_below_threshold)",
        f"- Composite fingerprint (frozen): `{record.composite_fingerprint}`",
        "",
        "## Last experiment",
        f"- Exp{record.experiment_num} ({record.backbone}) — **{record.status}**",
        f"- composite={record.composite:.4f}, "
        f"val={record.val_primary:.4f}, test={record.test_primary:.4f}",
        f"- per-fold test {record.primary_metric}: "
        f"{[round(x, 4) for x in record.per_fold_test]}",
        f"- elapsed: {record.seconds_elapsed:.2f}s",
        "",
        "## Exact next command",
        "```bash",
        next_command,
        "```",
        "",
        "## How to resume",
        "1. Read this checkpoint + `experiment_log.jsonl` (tail).",
        "2. Author the next pre-run reasoning entry in `reasoning_annotations.json` "
        "(must pass the Citation-Rigor + Reasoning-Completeness gates).",
        "3. Call `run_experiment(config)` for ONE experiment, then analyze + re-checkpoint.",
        "",
    ]
    ckpt.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(config: dict[str, Any]) -> ExperimentRecord:
    """Run ONE env-stats experiment. Writes runner-owned files. Returns the record."""
    t0 = time.time()
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "experiment_log.jsonl"
    exp_num = _next_experiment_num(log_path)

    # ---- (b) reasoning gate -------------------------------------------------
    annotations = ReasoningAnnotationsFile(results_dir / "reasoning_annotations.json")
    entries = annotations.load()
    key = str(exp_num)
    if key not in entries:
        raise ValueError(
            f"Experiment {exp_num} has no pre-run reasoning annotation in {annotations.path}.\n"
            "Author diagnosis/citations/hypothesis/prediction (with _manual: true) before launching."
        )
    entry = ReasoningEntry.from_dict(entries[key])
    violations = validate_pre_run_entry(entry)
    if violations:
        msg = "\n  - ".join(violations)
        raise ValueError(
            f"Experiment {exp_num} pre-run reasoning failed the gate. Fix first:\n  - {msg}"
        )

    # ---- composite calculator + fingerprint freeze (c) ----------------------
    comp_cfg = config.get("composite", {})
    primary_metric = config["primary_metric"]
    higher_is_better = bool(comp_cfg.get("higher_is_better", primary_metric in ("acc", "skill_vs_persistence")))
    calc = CompositeCalculator(
        primary_metric_name=primary_metric,
        higher_is_better=higher_is_better,
        penalty_weight=float(comp_cfg.get("penalty_weight", 0.1)),
        below_threshold=float(comp_cfg.get("below_threshold", 0.0)),
    )
    calc.assert_fingerprint(_frozen_fingerprint(log_path))  # raises if metric was rewritten

    # ---- data + split -------------------------------------------------------
    X, y, regimes = load_dataset(config)
    split_cfg = dict(config["split"])
    split_name = split_cfg.pop("name")
    splitter = create_splitter(split_name, **split_cfg)
    folds = splitter.split(len(X), y=y, groups=regimes)
    if not folds:
        raise ValueError("Splitter produced no folds — check split config / data size.")
    validate_no_overlap(folds)

    # ---- per-fold train/eval (d, e) -----------------------------------------
    backbone = config["backbone"]
    bb_cfg = dict(config.get("backbone_config", {}))
    seed = int(config.get("seed", 0))

    per_fold_val: list[float] = []
    per_fold_test: list[float] = []
    per_fold_reports: list[dict[str, Any]] = []

    for fa in folds:
        X_tr, y_tr = X[fa.train_idx], y[fa.train_idx]
        X_va, y_va = X[fa.val_idx], y[fa.val_idx]
        X_te, y_te = X[fa.test_idx], y[fa.test_idx]

        if backbone == "persistence":
            yhat_va = _fit_predict_persistence(X_tr, y_tr, X_va)
            yhat_te = _fit_predict_persistence(X_tr, y_tr, X_te)
        elif backbone == "mlp":
            yhat_va = _fit_predict_mlp(X_tr, y_tr, X_va, y_va, X_va, bb_cfg, seed)
            yhat_te = _fit_predict_mlp(X_tr, y_tr, X_va, y_va, X_te, bb_cfg, seed)
        else:
            raise ValueError(f"Unknown backbone {backbone!r} (have: persistence, mlp)")

        # persistence reference for skill scoring (always column-0 lag)
        ref_va, ref_te = X_va[:, 0], X_te[:, 0]
        val_m = _primary(primary_metric, y_va, yhat_va, reference_pred=ref_va)
        test_m = _primary(primary_metric, y_te, yhat_te, reference_pred=ref_te)
        per_fold_val.append(float(val_m))
        per_fold_test.append(float(test_m))
        per_fold_reports.append({
            "fold_id": fa.fold_id,
            "regime": fa.regime_label,
            "n_test": int(fa.n_test()),
            primary_metric: float(test_m),
            "rmse": env_metrics.rmse(y_te, yhat_te),
            "mae": env_metrics.mae(y_te, yhat_te),
            "acc": env_metrics.anomaly_correlation(y_te, yhat_te),
        })

    val_agg = float(np.mean(per_fold_val))
    test_agg = float(np.mean(per_fold_test))
    composite = calc.compute(val_agg, test_agg, per_fold_test)

    # ---- champion check (d) -------------------------------------------------
    best_path = results_dir / "best_config.json"
    prev_best = float("-inf")
    if best_path.exists():
        try:
            prev_best = float(json.loads(best_path.read_text(encoding="utf-8")).get("composite", float("-inf")))
        except Exception:
            prev_best = float("-inf")
    is_champion = composite > prev_best
    status = "KEEP" if is_champion else "DISCARD"

    record = ExperimentRecord(
        experiment_num=exp_num,
        backbone=backbone,
        description=config.get("description", ""),
        composite=composite,
        val_primary=val_agg,
        test_primary=test_agg,
        per_fold_test=per_fold_test,
        per_fold_val=per_fold_val,
        status=status,
        seconds_elapsed=time.time() - t0,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        composite_fingerprint=calc.fingerprint(),
        primary_metric=primary_metric,
        per_fold_reports=per_fold_reports,
        config=config,
    )

    # ---- write JSONL log ----------------------------------------------------
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), default=str) + "\n")

    # ---- champion record + checkpoint --------------------------------------
    champion = None
    if is_champion:
        best_payload = {
            "experiment_num": exp_num,
            "backbone": backbone,
            "composite": composite,
            "val_primary": val_agg,
            "test_primary": test_agg,
            "primary_metric": primary_metric,
            "composite_fingerprint": calc.fingerprint(),
            "config": config,
            "timestamp": record.timestamp,
        }
        best_path.write_text(json.dumps(best_payload, indent=2, default=str), encoding="utf-8")
        champion = best_payload
    elif best_path.exists():
        try:
            champion = json.loads(best_path.read_text(encoding="utf-8"))
        except Exception:
            champion = None

    next_cmd = (
        f"python -m autoresearch_env.runner --config <next_config.yaml>  "
        f"# author Exp{exp_num + 1} reasoning first"
    )
    _write_checkpoint(results_dir, record, champion, next_cmd)

    print(
        f"[env-runner] Exp{exp_num} ({backbone}) done in {record.seconds_elapsed:.2f}s — "
        f"composite={composite:.4f} status={status}"
    )
    return record


# ------------------------------------------------------------------ CLI

def _load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml  # type: ignore
        return yaml.safe_load(text)
    return json.loads(text)


def _cli():  # pragma: no cover - thin CLI wrapper
    import argparse
    ap = argparse.ArgumentParser(description="Env-stats AutoResearch runner (one experiment).")
    ap.add_argument("--config", required=True, help="Path to YAML or JSON config file")
    ap.add_argument("--description", default="", help="Override experiment description")
    args = ap.parse_args()
    cfg = _load_config(args.config)
    if args.description:
        cfg["description"] = args.description
    run_experiment(cfg)


if __name__ == "__main__":  # pragma: no cover
    _cli()
