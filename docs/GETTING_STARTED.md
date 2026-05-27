# Getting Started

A friendly setup guide for the `environment_stats_talk` repository — written for a
statistician who knows the math cold but is **new to Python tooling**. By the end
of this page you will have installed the stack, run your first before/after
experiment in about five minutes, learned how to read the metrics it prints and
the artifacts it writes, and know how to run the full test suite.

> **The single most important rule:** every command in this repo is run **from the
> repository root** (the folder that contains `common/`, `experiments/`, and this
> `docs/` folder). Every experiment does `import common`, and that import only
> resolves when the root is your working directory. If you see
> `ModuleNotFoundError: No module named 'common'`, you are almost certainly in the
> wrong directory.

---

## 1. What you need (and what you do *not*)

**You need:**

- **Python 3.12** (the version this repo is developed and tested on).
- About 2 GB of disk for the Python packages (mostly PyTorch).
- A terminal: PowerShell or Command Prompt on Windows, any shell on macOS/Linux.

**You do *not* need:**

- **No GPU.** Every committed headline result is reproducible on **CPU**. The deep
  models in Experiments 01, 04, and 08 *auto-detect* a CUDA GPU and use it if
  present, but they fall back to CPU cleanly. (This repo was developed on an NVIDIA
  RTX 4090 Laptop GPU; the GPU only makes runs faster and lets you scale up — it is
  never required.)
- **No API keys.** Every headline number comes from **synthetic data with
  deterministic seeds**. There is nothing to download, no Copernicus/ERA5 account,
  no Anthropic key. (Real-data and real-LLM variants are *documented and optional* —
  see each experiment's README — but the committed results never depend on them.)
- **No internet at run time** for the core experiments.

This "runs-anywhere" property is deliberate: the talk's claims have to be
reproducible by a skeptical audience on a laptop with one command.

---

## 2. Get the code

Clone the repository and move into it. If you do not have `git`, you can also
download the repo as a ZIP from its GitHub page and unzip it.

```bash
git clone <repository-url> environment_stats_talk
cd environment_stats_talk
```

From here on, "**repo root**" means this `environment_stats_talk` directory you are
now standing in. Confirm you are in the right place — you should see `common`,
`experiments`, `docs`, `requirements.txt`, and `run_all_tests.py`:

```bash
# macOS / Linux
ls
# Windows PowerShell
dir
```

---

## 3. Install Python and the dependencies

### 3.1 Install Python 3.12

If you do not already have it, install **Python 3.12** from
[python.org](https://www.python.org/downloads/) (on Windows, tick *"Add Python to
PATH"* during install). Verify:

```bash
python --version
# should print: Python 3.12.x
```

### 3.2 Create a virtual environment (recommended)

A virtual environment keeps this project's packages isolated from the rest of your
system. This is the standard Python hygiene step that saves you from version
conflicts down the road.

```bash
# create it (once)
python -m venv .venv

# activate it (every new terminal session)
#   macOS / Linux:
source .venv/bin/activate
#   Windows PowerShell:
.venv\Scripts\Activate.ps1
#   Windows Command Prompt:
.venv\Scripts\activate.bat
```

When the environment is active your prompt is prefixed with `(.venv)`.

### 3.3 Install the requirements

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs the core scientific stack (`numpy`, `pandas`, `scikit-learn`,
`scipy`, `statsmodels`, `matplotlib`, `pyyaml`, `networkx`) plus **PyTorch**
(`torch`) for the deep-learning experiments. The optional real-data / foundation-
model packages (xarray, cdsapi, rasterio, pygbif, anthropic, chronos, …) are
**commented out** in `requirements.txt` on purpose — you do not need them to
reproduce any committed result.

### 3.4 Two pins that matter — the numpy-2 ABI notes

`requirements.txt` deliberately pins two versions *higher* than you might expect.
These are not arbitrary; they exist because older releases have a **binary
(ABI) incompatibility with NumPy 2.x** that produces confusing import-time crashes:

```text
statsmodels>=0.14.4   # versions < 0.14.4 break against numpy 2.x:
                      #   the statespace / SARIMAX code fails to import
matplotlib>=3.9       # versions < 3.9 break against numpy 2.x:
                      #   matplotlib._path fails to load
```

**Why this happens:** NumPy 2.0 changed its C-level ABI. Packages with compiled
extensions (statsmodels' state-space filters, matplotlib's C `_path` module) that
were built against NumPy 1.x will not load cleanly under NumPy 2.x. The fix is
simply to use the newer releases that were rebuilt for NumPy 2 — which is exactly
what the pins guarantee.

**Symptoms you would see if these pins were violated** (e.g. if you reused an old
environment): an `ImportError` mentioning `matplotlib._path`, or a SARIMAX/state-
space import failure in Experiment 01 or 07. If you hit either, upgrade:

```bash
pip install --upgrade "statsmodels>=0.14.4" "matplotlib>=3.9"
```

> Note: Experiment 01's classical baseline is written defensively — if the
> state-space (SARIMAX) extension is unavailable it transparently falls back to a
> plain `AutoReg` fit, so it always runs. But to reproduce the committed SARIMA
> numbers you want a working `statsmodels>=0.14.4`.

### 3.5 (Optional) PyTorch with CUDA for the 4090 path

`pip install -r requirements.txt` gives you a working PyTorch that runs on CPU.
If you have an NVIDIA GPU and want the experiments to use it, install the CUDA
build that matches your driver (this repo was built against CUDA 12.4 / torch 2.6).
See [pytorch.org](https://pytorch.org/get-started/locally/) for the exact command.
The experiment scripts call `get_device()` / `pick_device()` and pick up CUDA
automatically — **no flag needed**, and they print the device they used.

---

## 4. The five-minute first run

Run the cleanest "hello world" of environmental forecasting — Experiment 01,
daily temperature forecasting — in its fast smoke mode. **From the repo root:**

```bash
python experiments/01_climate_timeseries_forecast/run_before_after.py --quick
```

`--quick` uses a tiny series and a few epochs with SARIMA off, so it finishes in
**well under a minute** even on CPU. (Drop `--quick` for the modest default,
≈ 2 min on CPU; that is the run that produced the committed numbers below.)

### 4.1 How to read the printed metrics

The script trains and scores four forecasters — **persistence**, **seasonal-naive**,
**(S)ARIMA**, and the **AFTER LSTM** — at one or more forecast horizons (lead
times), and prints RMSE/MAE/ACC plus a **skill score** for the LSTM. The numbers to
understand:

- **RMSE (°C)** — root-mean-square forecast error, lower is better. This is the
  headline accuracy number.
- **MAE (°C)** — mean absolute error, lower is better.
- **ACC** — anomaly correlation coefficient, higher is better (1 = perfect).
- **Skill score** = `1 − RMSE_after / RMSE_persistence`. It measures the LSTM's
  improvement *over the naive persistence baseline*: `0` means "tied persistence",
  `> 0` means "beats it", `< 0` means "worse than doing nothing". This single
  number keeps everyone honest — a model is only impressive if it beats the trivial
  reference.

**The committed full-run numbers** (from
`experiments/01_climate_timeseries_forecast/results/metrics.json`, 20 years,
seed 0, device=cuda) tell an honest story:

| Lead time | persistence RMSE | LSTM (AFTER) RMSE | LSTM skill | SARIMA RMSE |
|----------:|-----------------:|------------------:|-----------:|------------:|
| 1 day  | 1.876 | **1.767** | +0.058 | 1.826 |
| 7 days | 3.580 | 2.811 | +0.215 | **2.054** |
| 14 days | 4.308 | 2.937 | +0.318 | **2.095** |

Read this carefully: the LSTM **beats persistence at every horizon** (skill > 0,
growing with lead time), but **classical SARIMA wins at 7- and 14-day leads** (RMSE
2.05 / 2.10 beats the LSTM's 2.81 / 2.94). That is the point — the agent lets you
build and *fairly* compare all four methods in one command, and the honest finding
is that on this near-AR(1) single-station problem, classical statistics is hard to
beat. (Your `--quick` numbers will differ — they come from a much smaller, faster
configuration — but the *structure* of the comparison is the same.)

### 4.2 What lands in `results/`

After the run, look in
`experiments/01_climate_timeseries_forecast/results/`:

- **`metrics.json`** — the machine-readable scoreboard: RMSE/MAE/ACC for each
  method at each horizon, the LSTM's skill vs persistence, the config used (years,
  seed, device, model hyper-parameters), wall-times, and the
  foundation-model status (skipped unless you install Chronos/TimesFM).
- **`forecast_plot.png`** — observed vs persistence vs LSTM on the held-out test
  tail. *Open this image* to see the forecasts visually.
- **`before_after_bars.png`** — grouped RMSE bars, persistence vs LSTM.
- **`summary.md`** — the same scoreboard as a human-readable table.

Every experiment follows this same `results/` convention (a `metrics.json`, one or
more PNG plots, and a `summary.md`).

---

## 5. The no-GPU / no-API path (read this if you are worried about setup)

You can reproduce **the entire repository** with nothing but Python and the
`requirements.txt` packages:

- All data is **synthetic** and generated on the fly by the deterministic
  generators in `common/` (seeds fixed). Nothing is downloaded.
- Deep models **auto-fall-back to CPU**; the fast `--quick` modes are sized so
  that CPU is plenty.
- No experiment requires an API key to produce its committed numbers. Experiment 03
  ships a *deterministic offline* extractor as its default; the real-LLM variant is
  gated behind `ANTHROPIC_API_KEY` and is entirely optional. Experiment 01 and 04's
  foundation-model paths are skipped cleanly when the (optional) packages are absent
  — they return a documented "skipped" status, never a fabricated number.

So the recommended first experience is: **CPU, `--quick`, no keys.** Once that
works, drop `--quick` for the committed numbers, and only then consider the GPU and
real-data variants documented in each experiment's README.

---

## 6. Run all the tests

Each experiment is a self-contained mini-project with its own local `before/` and
`after/` packages. Those package names repeat across experiments on purpose, so
collecting them into one pytest process would make the module names collide. The
provided runner sidesteps this by running **each component in its own subprocess** —
exactly how a real user would run one experiment at a time:

```bash
# from the repo root — runs every component's test suite
python run_all_tests.py
```

It runs, in order: `common/tests`, `autoresearch_env/tests`, then the test suite
for each of the nine experiments (01, 02, 03, 04, 05, 06, 07, 08, 12). At the end it
prints a `SUMMARY` block with `PASS` / `FAIL` per component and a final
"All N components passed." line (exit code 0) or a count of failures (exit code 1).

You can pass pytest arguments straight through — they are forwarded to every
component's `pytest` call:

```bash
python run_all_tests.py -k exp01      # only tests whose name matches "exp01"
python run_all_tests.py -q            # quieter output
```

To run a single experiment's tests directly (also from the repo root):

```bash
python -m pytest experiments/01_climate_timeseries_forecast/tests -q
```

The tests are intentionally tiny and **CPU-only** (e.g. 2-year series, 1 epoch,
small fits). Most of the wall-clock time you observe is one-time library start-up
(importing `torch` and `statsmodels`) rather than the assertions themselves, so do
not be alarmed if a "fast" test takes a few seconds to begin.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'common'` | You are not in the repo root. | `cd` to the `environment_stats_talk` directory and re-run. Every command assumes the root as the working directory. |
| `ImportError` mentioning `matplotlib._path` | `matplotlib < 3.9` against NumPy 2.x (ABI break). | `pip install --upgrade "matplotlib>=3.9"` |
| SARIMAX / state-space import error in Exp01 or Exp07 | `statsmodels < 0.14.4` against NumPy 2.x (ABI break). | `pip install --upgrade "statsmodels>=0.14.4"` (Exp01 will otherwise fall back to `AutoReg` and still run, but you want this for the SARIMA numbers). |
| `command not found: python` / wrong Python version | Python 3.12 not installed or not on PATH; venv not activated. | Install Python 3.12; activate the venv (§3.2); check `python --version`. |
| Deep experiment is slow on CPU | No GPU, full (non-`--quick`) run. | Use `--quick` to verify the pipeline first; the full CPU run for Exp01 is still only ≈ 2 min. A CUDA build of torch makes it sub-second per epoch. |
| `RuntimeError` / extraction fails in Exp03 | You triggered the real-LLM variant without a key. | The default offline `extract()` needs no key. The `extract_with_llm()` path is gated behind `ANTHROPIC_API_KEY` and raises immediately if it is absent — set the key only if you deliberately want the live call. |
| A foundation-model row says "skipped" in Exp01/Exp04 | Chronos/TimesFM (or an RS encoder) not installed. | This is expected and correct — the optional package is absent, so the code reports a documented skip rather than inventing a number. Install the optional package only if you want that row. |
| Plots not generated / matplotlib backend error in a headless environment | No display available. | The scripts write PNGs to disk (a non-interactive backend); you do not need a display. Just open the PNG files in `results/`. |
| A `pip install` fails to build a wheel | Older pip or a missing toolchain. | `python -m pip install --upgrade pip` first, then re-run. PyTorch can also be installed separately per [pytorch.org](https://pytorch.org/get-started/locally/). |

---

## 8. Where to go next

- **`docs/TUTORIAL_run_an_experiment.md`** — a deep, end-to-end walkthrough of
  Experiment 01: the synthetic data, every baseline, what skill score means,
  reading `metrics.json` and the plots, the honest "SARIMA wins at long leads"
  finding, and how to scale to the 4090 and to real ERA5.
- **`docs/EXPERIMENTS_INDEX.md`** — a one-stop index of all nine experiments:
  one-line description, BEFORE vs AFTER, headline numbers, exact run command, key
  artifacts, whether a GPU helps, and the honest verdict for each.
- **`RESULTS.md`** (repo root) — the full before/after scoreboard with the
  committed numbers for every experiment.
- **`CLAUDE.md`** (repo root) — the project brief, mission, and operating rules.
- Each **`experiments/NN_*/README.md`** — the deep dive for that experiment,
  including the real-data swap and the GPU scaling notes.
