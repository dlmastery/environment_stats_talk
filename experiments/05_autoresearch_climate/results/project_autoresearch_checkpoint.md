# Env-Stats AutoResearch — Crash-Recovery Checkpoint

_Last updated: 2026-05-26T20:56:36_  
Driver: Claude Code (outer loop). Hardware: CPU smoke / RTX 4090 for full runs.

## Current champion
- Exp5 (mlp) composite=-1.7967
- Primary metric: `rmse` (composite = min(val,test) − penalty·n_below_threshold)
- Composite fingerprint (frozen): `942df6a8450d8d57`

## Last experiment
- Exp6 (mlp) — **DISCARD**
- composite=-1.9622, val=1.9622, test=1.8228
- per-fold test rmse: [1.8228]
- elapsed: 4.55s

## Exact next command
```bash
python -m autoresearch_env.runner --config <next_config.yaml>  # author Exp7 reasoning first
```

## How to resume
1. Read this checkpoint + `experiment_log.jsonl` (tail).
2. Author the next pre-run reasoning entry in `reasoning_annotations.json` (must pass the Citation-Rigor + Reasoning-Completeness gates).
3. Call `run_experiment(config)` for ONE experiment, then analyze + re-checkpoint.
