# CHECKPOINT — live progress log

Append-only log for crash recovery. Newest at top. Pushed to the public repo periodically.

---

## 2026-05-28 — Wave 6 COMPLETE (REAL domain-specific foundation models)

Closed the "did you actually run the SOTA foundation models from the transcript?" gap end-to-end:

- **Generic TS foundations on Exp01 (4090, zero-shot, 548×3 forecasts each):**
  - **TimesFM-2.0-500m (Google)** RMSE 1.82/3.20/3.83 (skill +0.031/+0.087/+0.094) — only clean win.
  - Chronos-Bolt-small (Amazon) ~ persistence.
  - MOMENT-1-small (AutonLab): random-init forecast head — unusable zero-shot.
  - Moirai-1.0-R-small (Salesforce): underperforms on this seasonal series.
  - **SARIMA still beats every foundation here** — exact "AFTER ≠ always wins" data point.

- **Domain-specific weather FMs (`experiments/00_foundation_models_climate/`, 5 of 7 RAN):**
  - GraphCast (small): REAL ERA5 6-hr forecast (60s CPU, bundled 1° example).
  - NeuralGCM (2.8°): REAL 4-day forecast on public ARCO-ERA5 (944s CPU, no CDS key).
  - ClimaX (5.625°): full pipeline on synthetic in 0.91s on 4090 (exact grid match for `common.gridded_temperature_field` — best "hello world").
  - FourCastNet: 3.35s on 4090, 75M params (pipeline test on synthetic).
  - Pangu-Weather (24h): 210s CPU ONNX pipeline.
  - CLLMate: dataset/benchmark only — no released model (honestly logged).
  - WeatherNext/2: service-only (BigQuery/EE/Vertex AI) — documented.
  - ClimateLLM (arXiv:2502.11059): arXiv + OpenReview + GitHub searched -> NO public code/weights -> logged `{status: not_available}` in metrics.json (never fabricated).

- **Domain-specific RS foundation on Exp04:**
  - NASA-IBM **Prithvi-EO-100M** (ViT-B/16, MAE on HLS), frozen + linear probe = 0.997 acc (matches from-scratch CNN with **0 gradient steps**); embedding 4000 patches in 32.8s on 4090.
  - EagleVision (arXiv:2503.23330): no clean embedding endpoint in the public repo — documented honestly.

- New docs: `docs/FOUNDATION_MODELS.md` — definitive runnability matrix + per-model section.
- RESULTS.md + README updated with the foundation-models story; CHECKPOINT logged.
- Solved real install friction: truststore (HF SSL on Windows); transformers 4.57->5.9; safetensors 0.4->0.7; h5py numpy-2 ABI; jax/jaxlib==0.4.30 + chex<0.1.86 + optax<0.2.4 for GraphCast; gcsfs SSL -> fsspec HTTPS workaround for ARCO-ERA5; ClimaX API-drift patches.
- 1 borderline-large file: experiments/00_foundation_models_climate/fourcastnet/results/fcn_forecast.npy 79MB (under GitHub hard limit; kept for reproducibility).

---

## 2026-05-26 — Wave 5 COMPLETE (11 experiments + presenter walkthrough)
- Added Exp09 (Bayesian hierarchical: MH 11.6s/dataset -> amortized 0.008s scoring, ~1451x speedup, coverage parity) and Exp10 (SDM: GLM AUC 0.71 -> GBM 0.73, suit-corr 0.70 -> 0.88).
- Built docs/talk/demo_walkthrough.md with REAL captured stdout for the 5 live-demo experiments (presenter backup).
- common/__init__ exports synthetic_station_offsets + synthetic_sdm_dataset. run_all_tests covers 13 components; full suite green.
- RESULTS.md + README + EXPERIMENTS_INDEX refreshed to span all 11 experiments; honest mixed-verdict framing intact.
- One regression caught & fixed: the demowalk agent's --quick re-runs had overwritten Exp01/02/03/12 full-run results; restored from origin/master before commit.

---

## 2026-05-26 — Wave 4 COMPLETE (heavy docs + README overhaul + 9 experiments)
- Added Exp06 (spatial kriging 2.91 -> RF+cov 2.20) and Exp07 (air quality ARIMA 17.4 -> GBM+weather 12.1). 9 experiments total; **11 test components green**.
- common/__init__ exports all generators (climate/biodiversity/remote-sensing+hard/hydrology/spatial/airquality). deck.pptx now 20 slides with 12 real result figures embedded.
- **Heavy documentation set written** (parallel doc agents): GETTING_STARTED, EXPERIMENTS_INDEX, TUTORIAL_run_an_experiment, TUTORIAL_autoresearch, GUIDE_using_claude_code_for_envstats, GUIDE_adapt_to_real_data, GLOSSARY, FAQ, ADDENDUM_methodology, CONTRIBUTING.
- **README fully overhauled**: TOC, thesis, 9-experiment table, repo map, docs index, autoresearch section, hardware/reproducibility, honest-results philosophy, credits. All 39 internal links verified.
- Synced to public GitHub after each unit. v1++ complete.

---

## 2026-05-26 — Wave 3 COMPLETE (backlog expansion + verification)
- Second parallel SME wave (5 agents) all integrated & pushed:
  - **deck.pptx** (18 slides, reproducible build_deck.py) — tangible presentation artifact.
  - **Exp04 hard-mode**: texture-only classes -> genuine CNN win RF 0.642 vs CNN 0.998 (+0.356 acc) on the 4090.
  - **Exp08 hydrology**: linear NSE 0.142 -> LSTM NSE 0.704 (+0.562) — the clean ML-wins example.
  - **Exp12 conformal UQ**: normal-theory 80%->86.9% miscalibrated; conformal closes gap 0.033->0.004 (~7x) with narrower bands.
  - **Citation verification (web)**: all 5 arXiv anchors + Brown&Spillias (MEE 2026 DOI) + Co-Scientist Nature (19 May 2026 DOI) CONFIRMED as real (postdate my Jan-2026 cutoff). Corrections: AI-Scientist-v2 Nature title is 'Towards end-to-end automation of AI research' (workshop peer-review); Aletheia = Gemini 3 Deep Think not 3.5. Round-number performance STATS remain UNVERIFIED/flagged.
- RESULTS.md + README refreshed: 7 experiments, honest mixed verdicts (3 clean AFTER-wins, 1 rigor win, 1 recall win w/ ceiling, ties where deserved incl SARIMA>LSTM at 1-step). run_all_tests.py now covers 9 components.
- **NEXT (future sessions):** Exp06 spatial kriging-vs-ML, Exp07 air quality, Exp09 Bayesian-vs-amortized, Exp10 SDM; embed deck figures from results/*.png; optional real-data (ERA5/CAMELS) smoke; record the live-demo backup video.

---

## 2026-05-26 — Wave 2 COMPLETE (v1 build done)
- All 5 experiments run for real; results committed & pushed. RESULTS.md scoreboard live.
- Real headline numbers: Exp01 (4090) LSTM skill +0.06/+0.21/+0.32 at h=1/7/14 — but **SARIMA wins at h>=7** (honest: value = fair fast benchmarking). Exp02 GEV 100-yr 55.8mm[46.7-80.2] + stationarity flag. Exp03 recall 10%->100% (synthetic ceiling). Exp04 (4090) RF=CNN=1.0 on easy synthetic + change map (effort/capability shift). Exp05 flagship: gated 6-experiment loop in ~38s, champion +8.0% skill, gates provably block shallow entries.
- Fixed numpy-2 ABI breaks: matplotlib->3.10.9, statsmodels->0.14.6 (pinned in requirements).
- Test runner extended to 7 components; full suite green.
- **NEXT (optional polish / future sessions):** Exp04 texture 'hard mode' so CNN beats RF; backlog experiments (spatial kriging, hydrology/CAMELS, air quality, Bayesian, SDM, conformal UQ); verify the flagged citations in CITATIONS-TO-VERIFY.md before any slide uses them; generate the actual .pptx from docs/talk/outline.md.

---

## 2026-05-26 — Wave 1 COMPLETE (all docs + code + unit tests built)
- Public repo live: https://github.com/dlmastery/environment_stats_talk (periodic pushes after each unit).
- Dispatched **10 parallel SME agents**; all integrated & checkpointed:
  - Docs: taxonomy, before/after, zero-to-hero, autoresearch protocol; ledgers INSIGHTS/ITEMS/CITATIONS-TO-VERIFY; talk outline/handout/demo; 3 skills.
  - Code+tests (all green individually): Exp01 climate forecasting (13), Exp02 extremes/trends (12), Exp03 biodiversity text (11), Exp04 remote sensing (8), common/ (16), autoresearch_env/ (10).
- Honest headline findings so far: Exp01 LSTM +0.16 skill at h=7 (wins at longer horizon); Exp02 GEV adds CIs+extrapolation+validation; Exp03 recall 10%→100% (synthetic ceiling, caveated); Exp04 capability/effort shift + change detection (RF≈CNN on easy synthetic data — honest).
- Known issues to fix in Wave 2 prep: (1) `statsmodels` 0.14.2 vs numpy 2.2 ABI break → SARIMAX falls back to AutoReg; upgrade statsmodels. (2) Per-experiment `before`/`after` package names collide in a single pytest collection → use `run_all_tests.py` (one subprocess per component).
- **NEXT (Wave 2):** fix statsmodels; full experiment runs on the 4090 (Exp01/02/04 full + Exp05 autoresearch loop); aggregate `RESULTS.md` 100×-faster scoreboard; final review.

---

## 2026-05-26 — Session start (Claude Code, Opus)
- Read full source transcript (6,988 lines) chunk by chunk. Synthesis captured in `CLAUDE.md` §1–2.
- Studied autoresearch: `dlmastery/autoresearch` (private, also at `C:/Users/evija/autoresearch/`) and ecosystem (karpathy/autoresearch + forks). Protocol captured in `CLAUDE.md` §5.
- Confirmed dev env: Python 3.12, numpy/pandas/sklearn/scipy/statsmodels present, **torch 2.6 + CUDA on RTX 4090 Laptop (~17 GB)**. matplotlib has an init error to fix (use Agg backend / reinstall).
- Created master task ledger (17 tasks) — see `TODO.md`.
- **DONE:** repo scaffold, `README.md`, `requirements.txt`, `.gitignore`, `CLAUDE.md`, `TODO.md`, this file.
- **NEXT:** build `common/` utilities + tests (lead), then dispatch hierarchical SME agent teams for docs + experiment code + tests (no full runs yet). Set up public GitHub repo + periodic push.

### Decisions / defaults locked
- Results reproducible from synthetic + small data, no keys/GPU; real-data & 4090 variants documented.
- Before/After framing for every experiment; flagged claims quarantined in `CITATIONS-TO-VERIFY.md`.
