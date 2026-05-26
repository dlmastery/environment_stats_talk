# env-stats-validate — compact checklist (copy into the report and fill in)

Result under review: ____________________   Reviewer: ____________________   Date: __________

Mark each: PASS / FAIL / N-A. Add a one-line note + pointer to evidence (plot/table/code).

| # | Check | Status | Note / evidence pointer |
|---|---|---|---|
| 1a | Temporal split: test strictly in the future; no shuffle; purge+embargo ≥ label horizon | | |
| 1b | Spatial split: whole basins/cells/stations held out together (GroupKFold) | | |
| 1c | Preprocessing (scalers/imputers/feature-selection/climatology) fit on TRAIN only | | |
| 2a | Stationarity assessed (trend / variance change); Mann-Kendall / change-point if needed | | |
| 2b | Residual autocorrelation checked (ACF/PACF, Ljung-Box); SEs adjusted if present | | |
| 3a | Distributional assumption matches data (non-Gaussian: heavy tails, counts, compositional) | | |
| 3b | EVT: threshold/block justified; shape parameter + its uncertainty reported | | |
| 4 | Multiple testing corrected (Bonferroni / FDR); test family + corrected threshold stated | | |
| 4b | Selection bias from many experiments addressed (winner confirmed on unseen fold) | | |
| 5a | Calibrated CIs / conformal intervals provided (not point predictions only) | | |
| 5b | Coverage checked (empirical vs nominal); return-level CIs for EVT | | |
| 6a | Honest baseline present (climatology/persistence/majority/RF-on-indices) | | |
| 6b | Skill score reported (1 perfect / 0 no-gain / <0 worse); field-appropriate metric used | | |
| 7a | Seeds fixed and recorded; result regenerable from committed code+config | | |
| 7b | Package versions + data provenance (.meta.json) recorded; runs-anywhere path exists | | |
| 8a | Every AI-cited paper/stat/date verified or (verify)-flagged → CITATIONS-TO-VERIFY.md | | |
| 8b | Composite-metric fingerprint unchanged (no Goodhart rewrite) | | |
| 8c | Headline claim re-checked against raw evidence, not the AI's summary | | |

## Sign-off (AI proposes, human decides)

- All applicable rows PASS or justified: [ ]
- Named human statistician: ____________________
- Decision: [ ] approved  [ ] approved-with-caveats  [ ] rejected
- Caveats / conditions: ____________________
- Date: __________
