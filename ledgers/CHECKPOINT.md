# CHECKPOINT — live progress log

Append-only log for crash recovery. Newest at top. Pushed to the public repo periodically.

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
