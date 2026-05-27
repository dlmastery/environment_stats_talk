# TODO — Master Task Ledger

Mirror of the live task list. Status: ☐ pending · ◐ in progress · ☑ done.
Rules: **docs + code + unit tests first**, experiment **runs** after everything is green. Before/After everywhere. Runs-anywhere (synthetic, no keys/GPU) then real-data + 4090 variants. Verifiable claims only.

## Phase 0 — Foundations
- ☑ 1. Scaffold repo + README (before/after thesis) + requirements + .gitignore + CLAUDE.md
- ◐ 9. `common/` utilities: synthetic env-data generators + plotting (Agg) + unit tests

## Phase 1 — Documentation (parallel SME agents)
- ☐ 2. `ledgers/INSIGHTS.md`
- ☐ 3. `ledgers/ITEMS.md` (papers/tools/datasets/benchmarks, VERIFIED vs NEEDS-VERIFICATION)
- ☐ 4. `ledgers/CITATIONS-TO-VERIFY.md` (rigor)
- ☐ 5. `docs/taxonomy.md` (env-stats taxonomy + where AI/Claude Code fits)
- ☐ 6. `docs/before_after.md` (framework + master comparison table)
- ☐ 7. `docs/zero_to_hero.md` (L1–L4 Claude-Code ladder)
- ☐ 8. `docs/autoresearch_protocol.md` (env-stats adaptation of the 7-step loop)
- ☐ 16. `docs/talk/` (slide outline + speaker notes, handout, live-demo script)

## Phase 2 — Code + unit tests (parallel SME agents) ✅ DONE (75 tests green)
- ☑ 10. Exp01 climate time-series forecasting — code + 13 tests
- ☑ 11. Exp02 extreme value & trend detection — code + 12 tests
- ☑ 12. Exp03 biodiversity text → interactions — code + 11 tests
- ☑ 13. Exp04 remote-sensing land-cover (4090) — code + 8 tests
- ☑ 14. Exp05 autoresearch loop adaptation (`autoresearch_env/`) — code + 10 tests
- ☑ 15. Skills: env-autoresearch-setup, climate-data-fetch, env-stats-validate

## Phase 3 — Experiment runs ✅ DONE
- ☑ Ran Exp01–04 on the 4090/CPU; real results committed (metrics, plots, summaries)
- ☑ Ran Exp05 autoresearch loop (flagship); champion + journal + checkpoint committed
- ☑ 17. Aggregated `RESULTS.md` scoreboard + README/CHECKPOINT sync + full-suite re-run

## Backlog (expand — "very very long list")
- ☐ Exp06 spatial statistics: kriging vs ML interpolation
- ☐ Exp07 air-quality source apportionment / PM2.5 nowcasting
- ☐ Exp08 hydrology: streamflow (CAMELS-style) LSTM vs conceptual model
- ☐ Exp09 Bayesian hierarchical (PyMC) vs amortized inference
- ☐ Exp10 species distribution modeling (MaxEnt vs gradient-boosted / NN)
- ☐ Exp11 multimodal fusion (raster + text) event description
- ☐ Exp12 uncertainty quantification & conformal prediction
- ☐ Skill: env-report-writer (auto policy-brief from results)
- ☐ Skill: arxiv-env-scout (weekly SOTA scan)
- ☐ Digital-twin mini-demo (scenario what-if on synthetic ecosystem)
- ☐ docs/talk/: backup slides, QR/reference sheet, image-prompt list
