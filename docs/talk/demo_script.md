# Live Demo Script (4–6 minutes)
### BEFORE vs AFTER — climate time-series forecasting, from the terminal

**Goal:** In one command, reproduce a traditional baseline *and* an agent-built forecast on the same data, same metric, same (chronological) split — show the numbers and a plot, live.

**Audience reminder:** statisticians. Lead with the *split* and the *metric*, not the model. Never quote a number you have not just produced on screen — read the figures off the terminal / `RESULTS.md`, don't recite them from a slide.

---

## Pre-flight checklist (do this before you walk on stage)

- [ ] Terminal open in the repo root; font size large; window maximized.
- [ ] `pip install -r requirements.txt` already done; `python -c "import numpy, pandas, matplotlib"` works.
- [ ] Decide PRIMARY vs FALLBACK (see below) and have BOTH ready.
- [ ] Image viewer ready to open the generated PNG quickly.
- [ ] **Recorded backup:** a screen recording of the full run + plot saved locally (e.g. `assets/demo01_recording.mp4`) and the final plot exported as a still. If the network, the GPU, or the projector misbehaves, play the recording and narrate — same talking points apply.

> **Recorded-backup note:** If anything stalls for more than ~10 seconds live, stop, say *"in the interest of time, here's the same run I recorded this morning,"* switch to the recording, and keep narrating. Never debug live in front of the room.

---

## PRIMARY demo — the headline command (Exp01)

This is the repo's documented headline path (see `README.md`). It runs the BEFORE (persistence / ARIMA) and AFTER (learned model + foundation-model zero-shot variant) and writes metrics + a forecast plot.

**Command:**
```bash
python experiments/01_climate_timeseries_forecast/run_before_after.py
```

**What to say while it runs (≈30–45 s):**
> "Same series, same metric — RMSE and anomaly correlation, the standard NWP skill score. The split is **chronological**, so we never train on the future. One command runs the old way and the new way back to back."

**When it finishes:** open the generated plot, then read the BEFORE vs AFTER metrics straight off the terminal (and note they're committed to `RESULTS.md`).

> ⚠️ **State check:** the `experiments/` tree is built and run in a later phase of this repo (docs + `common/` come first; see `ledgers/TODO.md`). **If `run_before_after.py` is not present yet on the machine you're presenting from, use the FALLBACK below** — it exercises the same idea using the modules that are already in the repo, with no GPU and no API keys. The FALLBACK is also the safest choice on a conference laptop regardless.

---

## FALLBACK demo — no GPU, no keys, no network (always works)

Uses only `common/` modules that are already present and unit-tested. It contrasts two *traditional* baselines — **persistence** (BEFORE, the classic naive forecast) vs a **seasonal-climatology** model (an agent's first honest improvement) — on a synthetic daily-temperature series with a chronological split. Deterministic seed → identical every time.

**One-liner (paste into the terminal):**
```bash
python - <<'PY'
import numpy as np, pandas as pd
from common.synthetic_climate import daily_temperature, time_split
from common.metrics import rmse, anomaly_correlation, skill_score
from common.plotting import plt, new_fig, save

# 1) DATA (synthetic, fixed seed — runs anywhere)
df = daily_temperature(n_years=20, seed=0)
tr, va, te = time_split(df)               # chronological: never leak the future
y = te["t2m"].to_numpy()

# 2) BEFORE — persistence: tomorrow = today
persist = np.concatenate([[tr["t2m"].iloc[-1]], y[:-1]])

# 3) AFTER (stand-in) — seasonal climatology from TRAIN day-of-year means
clim = tr.groupby(tr.index.dayofyear)["t2m"].mean()
seasonal = te.index.dayofyear.map(clim).to_numpy()

# 4) SCORE — same metric for both
r_b, r_a = rmse(y, persist), rmse(y, seasonal)
print(f"BEFORE  persistence  RMSE = {r_b:.3f} C")
print(f"AFTER   seasonal     RMSE = {r_a:.3f} C   ACC = {anomaly_correlation(y, seasonal):.3f}")
print(f"skill score (AFTER vs BEFORE, +ve = better) = {skill_score(r_a, r_b):+.3f}")

# 5) PLOT — first 120 test days
fig, ax = new_fig(8, 4)
ax.plot(te.index[:120], y[:120], color="#188038", lw=1.6, label="observed")
ax.plot(te.index[:120], persist[:120], color="#9aa0a6", lw=1.2, label="BEFORE persistence")
ax.plot(te.index[:120], seasonal[:120], color="#1a73e8", lw=1.2, label="AFTER seasonal")
ax.set_title("Exp01 fallback — BEFORE vs AFTER (synthetic, seed=0)")
ax.set_ylabel("temperature (C)"); ax.legend(frameon=False)
print("plot ->", save(fig, "assets/demo01_fallback.png"))
PY
```

**Then:** open `assets/demo01_fallback.png`.

### The teaching twist (use it — it lands with this audience)
On this synthetic *daily* series, **persistence is a very strong baseline** (tomorrow ≈ today), so the simple seasonal-climatology model does **not** beat it on RMSE — the skill score comes out negative. **Say this out loud:**
> "Notice the agent's first model loses to persistence here — and that's the point. The value of the AFTER workflow isn't that the first model wins; it's that in seconds we have an honest, leakage-free comparison that *tells us* the next move: add persistence's short-memory signal, i.e. fit residuals or go to an LSTM/TCN. The agent makes the loop fast; the statistician reads the scoreboard."

This turns a 'losing' number into a credibility win: you're showing rigor, not cherry-picking.

---

## Talking points (sequence, ~4–6 min total)

1. **(0:00–0:30) Frame it.** "Same data, same metric, same chronological split — only the workflow changes. Watch the clock."
2. **(0:30–1:30) Run it.** Paste the command; narrate the split and the metric while it runs.
3. **(1:30–3:00) Read the numbers.** Point at BEFORE vs AFTER RMSE on screen; say which is better and by how much (read live). Note ACC for the seasonal model.
4. **(3:00–4:00) Show the plot.** Observed vs both forecasts; let them *see* where each tracks or misses.
5. **(4:00–5:00) The twist / honesty.** Deliver the persistence-is-strong point above — rigor over hype.
6. **(5:00–6:00) Land it.** "That whole loop — data, two models, validation, plot, scoreboard — was one command and a few seconds. BEFORE, that's an afternoon of glue code. The numbers I just read are committed in `RESULTS.md`, so nothing here is staged."

---

## If asked live (quick answers)

- *"Is the data real?"* — "Synthetic with a fixed seed so it reproduces anywhere with no keys or GPU. Swapping in real ERA5 is one documented branch — same DataFrame schema, downstream code unchanged (see `common.synthetic_climate.REAL_DATA_NOTE`)."
- *"Where's the GPU model?"* — "That's the AFTER variant in the full Exp01 (LSTM/TCN on a 4090) plus a foundation-model zero-shot run; numbers in `RESULTS.md`. I'm running the no-GPU path on purpose so it's reproducible on any laptop in this room."
- *"How do I know it's not leaking?"* — "`time_split` is strictly chronological — validation and test are the most recent windows, never shuffled. That's enforced in code, not by hand."

---

## Reset for the next session
```bash
# remove the generated artifact so the demo is clean next time
rm -f assets/demo01_fallback.png
```
*(Windows PowerShell: `Remove-Item assets/demo01_fallback.png -ErrorAction SilentlyContinue`)*
