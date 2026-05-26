# CLAUDE.md — Project Brief, Operating Rules & Checkpoint

**Read this first.** This file is the single source of truth for the `environment_stats_talk` project. It records, in detail, everything the user (Vijay / `dlmastery`, eranti@gmail.com) has asked for, plus the operating rules every contributor (human or agent) must follow.

_Last checkpoint: see git log. Build driver: Claude Code (Opus). Hardware: Windows 11, Python 3.12, NVIDIA RTX 4090 Laptop GPU (~16–17 GB), CUDA 12.4, torch 2.6._

---

## 1. The mission

Build a comprehensive companion repository for a **conference talk** the user is giving:

- **Event:** International Environment Statistics Institute conference, **Mexico City, 7–11 December 2026**.
- **Session:** _"Modern Approaches to Environmental Statistics."_
- **Audience:** Environmental statisticians / environmetricians (TIES-type community) — rigorous, busy, skeptical of hype, value statistical validity and concrete takeaways.
- **Goal of the talk:** Prove to climate/environmental scientists a **"zero to hero"** path across a **taxonomy of environmental-statistics tasks**, leveraging **Claude Code and agentic coding** + **AI-for-science models**, showing that **science is now ~100× faster** — emphasizing _the art of telling the agent what to do_.

## 2. Exactly what the user asked for (chronological, verbatim intent)

1. **Source material:** `~/Downloads/grok-llms-ai-environmental-statistics-transcript.md` (6,988 lines). Read it **chunk by chunk** (not all at once) and build **ledgers of TODO tasks, insights, and items**. _(Done — see `ledgers/`.)_
2. **Study** `https://github.com/dlmastery/autoresearch/blob/master/docs/paper.md` and other **autoresearch forks** on GitHub. _(Done — see §5.)_
3. **Create a new repo** `environment_stats_talk` containing a **ton of supporting artifacts**: already-written code **with results that can be run easily from Claude Code**, **skills**, an **autoresearch protocol**, etc.
4. **Run for days.** Go over **all** the material to present. Provide **experiments runnable on a local 4090 GPU**. Showcase **science 100× faster** with Claude Code and agentic coders.
5. Produce a **very, very long list of tasks** and **knock them off one at a time**.
6. **Framing:** _"Transform this space with AI + AI coding capabilities + AI-for-science models. **Before and after** will be a great way to present slides with all artifacts and results."_ → Every topic is a BEFORE (traditional) vs AFTER (agentic/AI-for-science) pair, each with real artifacts + results.
7. **Sequencing:** _"First do all the documentation formalities (extensive) and all the coding and unit testing — before starting the run of experiments. Finish a lot of stuff. **Parallelize** the documentation, coding, and unit testing by **deploying hierarchical agent teams with SME agents.**"_
8. **Resilience:** _"Do periodic check-in to a **public GitHub repo** so I can recover from power failure."_
9. **Memory:** _"Update CLAUDE.md with all the things I said so far in great detail and **checkpoint** the whole thing."_ → this file.

## 3. Operating rules (non-negotiable)

- **Build order:** documentation + code + unit tests **first** (extensively), experiment *runs* **after** everything is built and green.
- **Parallelism:** use hierarchical SME agent teams for independent docs/code/test workstreams; the lead (Claude Code main session) integrates and runs experiments.
- **Before/After everywhere:** every experiment = `before/` (traditional) + `after/` (agentic/AI-for-science) + `results/` (real committed metrics, plots, tables) + a README with a time/effort/quality comparison.
- **Runs-anywhere first:** headline results must be reproducible from **synthetic + small public data, no API keys, no GPU** (deterministic seeds). Then document **real-data** and **4090-GPU** variants separately.
- **Zero-hallucination discipline (audience = scientists):** build only on **verifiable** papers/tools/datasets. Any speculative or future-dated claim from the transcript (e.g., "Co-Scientist Nature May 2026", "OpenAI Erdős May 2026", "GPT-5.4", fabricated meta-analyses/stats) goes to `ledgers/CITATIONS-TO-VERIFY.md` and is **never** stated as fact on a slide or in code comments.
- **Statistical rigor preserved:** AI accelerates; the human keeps final say on model choice, assumptions, multiple-testing, leakage, and uncertainty (the `env-stats-validate` skill encodes this).
- **Checkpoint often:** commit + push to the public GitHub repo after each meaningful unit (power-failure recovery).
- **Credit sources:** the autoresearch methodology is adapted from the user's own `dlmastery/autoresearch` (`generalized_ml_autoresearch`).

## 4. The build plan (see `ledgers/TODO.md` for the live list)

- **Foundations:** repo scaffold, README, this file, `common/` synthetic-data + plotting utilities (+ unit tests).
- **Documentation:** `ledgers/{INSIGHTS,ITEMS,CITATIONS-TO-VERIFY}.md`; `docs/{taxonomy,before_after,zero_to_hero,autoresearch_protocol}.md`; `docs/talk/{outline+notes, handout, demo script}`.
- **Code + tests (no full runs yet):** `common/`, experiments 01–04 (before/after scripts), `autoresearch_env/` adaptation, three skills.
- **Experiment runs (after build is green):** Exp01–04 produce committed results; Exp05 runs the autoresearch loop on the 4090 (flagship); aggregate `RESULTS.md`.

## 5. Autoresearch reference (studied)

- `dlmastery/autoresearch` (the user's own private repo; also cloned locally at `C:/Users/evija/autoresearch/`): _"Karpathy-style autonomous FX prediction optimization."_ Contains `generalized_ml_autoresearch/` — a **domain-agnostic** loop where **Claude Code is the researcher**, with:
  - A **7-step Research-Driven Experiment Selection** loop: **Diagnose → Cite → Hypothesize → Predict → Execute (one experiment) → Analyze → Checkpoint**.
  - **Hard gates:** Citation-Rigor + Reasoning-Blob-Completeness must pass before an experiment runs; composite-metric fingerprint detects mid-project goal rewrites.
  - Pluggable **backbones** (MLP, LSTM, FT-Transformer, GBMs, foundation-model stubs: TimesFM/Chronos/MOMENT/Moirai/TiRex/Sundial/Time-MoE), **splits** (holdout/kfold/stratified/group/timeseries/walk-forward/super-fold), **composite metric** `min(val,test) − penalty·n_below_threshold`.
  - **Winner archive** (self-contained champion folder + audit + inference + Colab), **crash-recovery checkpoint**, **interactive setup skill** (`/ml-autoresearch-setup`).
- Ecosystem (for the talk's "ladder"): `karpathy/autoresearch` (single-GPU, `program.md`-driven), and forks/skills (uditgoenka, AutoResearchClaw 23-stage idea→paper, autoresearch-cli, etc.).
- **Our adaptation:** `autoresearch_env/` brings this loop to environmental statistics — env composite metrics (e.g., latitude-weighted RMSE/ACC), env splits (spatial GroupKFold, walk-forward by year, super-fold by climate regime), citation gate seeded with the real env-stats SOTA catalog.

## 6. Verified anchors vs flagged claims

- **Verified, safe to build on:** ClimateLLM (arXiv:2502.11059), CLLMate (arXiv:2409.19058), "Towards an AI co-scientist" (arXiv:2502.18864), AI-Scientist-v2 (arXiv:2504.08066), EagleVision (arXiv:2503.23330); datasets ERA5, GBIF, iNaturalist, Sentinel-2, CAMELS, OpenAQ; platforms BioDT, DestinE; TS foundation models TimesFM/Chronos/MOMENT/Moirai. (Confirm exact versions/dates before citing — see ledger.)
- **Flagged (do NOT assert):** any "Nature, 19 May 2026" dating, "OpenAI Erdős breakthrough", "GPT-5.4 / Gemini 3.5", "WeatherNext 2" specifics, and invented statistics. See `ledgers/CITATIONS-TO-VERIFY.md`.

## 7. How to resume after a crash

1. `git pull` the public repo.
2. Read this file, then `ledgers/TODO.md` (task status) and the most recent commit messages.
3. Check `ledgers/CHECKPOINT.md` for the live progress log.
4. Continue the lowest-numbered unfinished task; keep before/after + runs-anywhere + rigor rules.
