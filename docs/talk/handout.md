# One-Page Handout — From Data to Discovery
### Transforming Environmental Statistics with Claude Code & AI-for-Science
*Companion to the talk — Mexico City, Dec 2026. Repo: `environment_stats_talk`.*

---

## 3 key messages

1. **Hold the science fixed; change only the workflow.** For each task, keep the same data, metric, and validation — then compare BEFORE (traditional) vs AFTER (agent-driven). Speed is the headline; **rigor is the hard gate**.
2. **It's a ladder, not a leap (zero → hero).** L1 assistant → L2 pipeline builder → L3 reusable skills → L4 autonomous research loop. Level 1 pays off on day one; you climb at your own pace.
3. **The human owns five checks.** Leakage/look-ahead, multiple testing, uncertainty (intervals not points), citation verification, and reproducibility. The agent does the labor; you own validity. Unverified claims stay quarantined — never stated as fact.

---

## Copy-paste prompts

> Tip: start on **synthetic or small public data — no API keys, no GPU**. Scale to real data and the 4090 only after the pipeline is green. Replace bracketed parts with your own context.

**A) Hypothesis generation (gated, no fishing)**
```
You are my environmental-statistics research partner. Context: I have [daily station
temperature, 1990–2024, mid-latitude]. Propose 5 *testable* hypotheses about
[trends / extremes / seasonality]. For each: (1) state it as a falsifiable prediction,
(2) name the appropriate estimator and its assumptions, (3) specify the validation
split that avoids leakage, (4) note the multiple-testing risk if I scan many stations,
(5) cite only prior work you can verify (give an arXiv ID or dataset name; if you are
unsure, say "unverified" — do NOT invent a citation). Rank by signal-to-effort.
Do not run anything yet; wait for me to pick one.
```

**B) "Claude Code, fetch ERA5 and fit a trend with validation"**
```
Claude Code: build a small, reproducible pipeline in this repo.
1. DATA: load 2m temperature. Default to the synthetic generator
   (common.synthetic_climate.daily_temperature, fixed seed) so it runs with no keys/GPU.
   Add a clearly-commented REAL-DATA branch that fetches ERA5 via cdsapi+xarray
   (2m_temperature) and maps it to the SAME DataFrame schema (date index + value column).
2. SPLIT: chronological train/val/test via common.synthetic_climate.time_split
   (never shuffle a time series — no look-ahead).
3. TREND: fit a trend with a Mann-Kendall test + Sen's slope; report the slope WITH a
   confidence interval, not a bare point estimate. If I later scan many series, apply a
   multiple-testing correction and say which one.
4. VALIDATE: run the rigor checklist (leakage, split discipline, multiplicity, uncertainty);
   refuse to report a result if any check fails and tell me why.
5. OUTPUT: save a trend plot with its CI band using common.plotting, write metrics to a
   results/ file, and print a 3-line plain-English summary. Use deterministic seeds.
   Quote no number you have not actually computed.
```

---

## 3-step action plan

- **Today (15 min):** Install Claude Code. Ask it to reproduce **one figure** from your last paper on synthetic data. Goal: feel the loop.
- **This week (a few hours):** Hand it a **full small pipeline** end-to-end — use prompt (B): fetch/generate data → fit → validate → plot → summarize. Keep it on synthetic/small data first.
- **This month (a day):** Write **one reusable skill** that encodes your group's validation conventions (splits, multiplicity, UQ, reporting). Now every future task inherits your standards. Then clone this repo and use `experiments/01_climate_timeseries_forecast/run_before_after.py` as your template.

---

## Verified resources

**Papers (arXiv IDs — verify the page before citing):**
- ClimateLLM — arXiv:2502.11059
- CLLMate — arXiv:2409.19058
- Towards an AI co-scientist — arXiv:2502.18864
- The AI Scientist-v2 — arXiv:2504.08066
- EagleVision — arXiv:2503.23330

**Datasets (canonical open sources):** ERA5 (Copernicus CDS) · GBIF · iNaturalist · Sentinel-2 (Copernicus) · CAMELS · OpenAQ

**In this repo:** `docs/talk/outline.md` (deck + notes) · `docs/talk/demo_script.md` (live demo) · `docs/taxonomy.md` · `docs/before_after.md` · `docs/zero_to_hero.md` · `docs/autoresearch_protocol.md` · `RESULTS.md` (all measured numbers) · `ledgers/CITATIONS-TO-VERIFY.md` (quarantined claims — never cite as fact).

*All measured numbers live in `RESULTS.md`. This handout names no figure that has not been computed; speculative or future-dated claims are deliberately excluded.*
