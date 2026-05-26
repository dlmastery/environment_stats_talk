# ITEMS — Inventory of Papers, Tools, Datasets, Benchmarks & Models

Inventory of every concrete item surfaced in the source transcript and project brief, tagged for trust.

**Legend**
- **[VERIFIED]** — A real, locatable anchor with a stable identifier (arXiv ID, project URL, or well-known dataset). *Still confirm exact version, date, and any quoted numbers against the primary source before citing on a slide.*
- **[NEEDS-VERIFICATION]** — Plausible and likely real, but the transcript's identifier, venue, date, or stats could not be confirmed from the transcript alone. Verify before use.
- **[FLAGGED]** — Future-dated, unfindable, or fabricated-looking. **Do not assert.** Full reasoning is in `CITATIONS-TO-VERIFY.md`.

> Note: "[VERIFIED]" here means *the anchor exists and is citeable*, per the project brief's verified-anchor list. The transcript was machine-generated; treat every **number** it attaches to a paper as `[NEEDS-VERIFICATION]` even when the paper itself is verified.

---

## 1. Papers (climate / weather forecasting)

| Item | ID / Venue | Tag | Note |
|---|---|---|---|
| **ClimateLLM** — frequency-aware LLM (2D FFT + Frequency-MoE + dynamic prompting, GPT-2 backbone, latitude-weighted RMSE) | arXiv:2502.11059 | **[VERIFIED]** (ID is a project anchor) | Core "climate forecasting" anchor; runs on ERA5. Quoted gains ("19–43% RMSE," "87–98% training-time reduction," "ACC 0.98–1.00") are **[NEEDS-VERIFICATION]** — check abstract/tables before quoting. |
| **CLLMate** — multimodal LLM for Weather & Climate Event Forecasting (WCEF); Llama3-8B + CLIP ViT-L/14 + knowledge graph | arXiv:2409.19058 (transcript says EMNLP 2025) | **[VERIFIED]** (ID is a project anchor) | "Numbers → natural-language events" anchor. Confirm EMNLP-2025 acceptance and BLEU/ROUGE/BERTScore figures before quoting. |
| GPT-4 zero-shot rainfall forecasting (conservative / mean-reverting) | arXiv:2411.13724 (transcript) | **[NEEDS-VERIFICATION]** | Useful "LLM fails without grounding" example. Verify ID and claims. |
| ClimAgent — agentic open-ended climate analysis | arXiv:2604.16922 (transcript, "Apr 2026") | **[FLAGGED]** | Future-dated ID (26xx = 2026). See ledger. |
| VLMs for textual weather forecasts | arXiv:2512.03623 (transcript, "Dec 2025") | **[FLAGGED]** | Future-dated ID. See ledger. |
| STELLM (spatio-temporal LLM for wind speed) | (transcript, no ID) | **[NEEDS-VERIFICATION]** | No locatable identifier given. |

## 2. Papers (agentic science / autonomous discovery)

| Item | ID / Venue | Tag | Note |
|---|---|---|---|
| **"Towards an AI co-scientist"** — multi-agent generate–debate–evolve + tournament (Gemini) | arXiv:2502.18864 | **[VERIFIED]** (ID is a project anchor) | The *preprint* is the safe anchor. The transcript's **"Accelerating scientific discovery with Co-Scientist, Nature, 19 May 2026 (Gottweis et al.)"** is **[FLAGGED]** — future-dated Nature claim, see ledger. Cite the preprint, not the Nature date. |
| **AI-Scientist-v2** — agentic tree search, end-to-end idea→paper | arXiv:2504.08066 | **[VERIFIED]** (ID is a project anchor) | Safe to cite as a preprint. Transcript claims "published in Nature, early 2026" and "first AI-generated peer-reviewed paper" — both **[FLAGGED]**, see ledger. |
| Robin — multi-agent discovery | arXiv:2505.13400 (transcript) | **[NEEDS-VERIFICATION]** | Plausible 2025 ID; verify. |
| "Expert-level empirical software" agent (LLM + tree search; geospatial examples) | arXiv:2509.06503 (transcript) | **[NEEDS-VERIFICATION]** | Verify ID/claims. |
| EarthLink — self-evolving climate-science agent | arXiv:2507.17311 (transcript) | **[NEEDS-VERIFICATION]** | Verify ID. |
| Kosmos | arXiv:2511.02824 (transcript) | **[FLAGGED]** | Future-dated ID (2511 = Nov 2025, after the conversation's own "May 2026" framing is inconsistent; ID unverifiable). See ledger. |
| "AI Co-Scientist for Ranking" (claims models GPT-5.2, Gemini Pro 3, Claude Opus 4.5) | arXiv:2603.22376 (transcript, "Mar 2026") | **[FLAGGED]** | Future-dated ID + unreleased model versions. See ledger. |
| OpenScientist (open agentic co-scientist eval) | medRxiv (transcript, "Mar 2026") | **[NEEDS-VERIFICATION]** | No DOI; verify. |
| "Survey of AI Scientists" (Tie, Zhou, Sun) | arXiv:2510.23045 (transcript, "v5 rev Jan 2026") | **[NEEDS-VERIFICATION]** | Plausible survey ID; verify ID, authors, and version. |
| "From AI for Science to Agentic Science" (Wei et al.) | arXiv:2508.14111 (transcript) | **[NEEDS-VERIFICATION]** | Verify. |
| AlphaEvolve (DeepMind evolutionary coding agent) | arXiv:2506.13131 (transcript) | **[NEEDS-VERIFICATION]** | Real product line; verify the specific arXiv ID and any optimization claims. |

## 3. Papers (remote sensing / EO multimodal LLMs)

| Item | ID / Venue | Tag | Note |
|---|---|---|---|
| **EagleVision** — object-level attribute MLLM for remote sensing | arXiv:2503.23330 | **[VERIFIED]** (ID is a project anchor) | Remote-sensing anchor. Quoted "70–95% accuracy" is **[NEEDS-VERIFICATION]** (also stated as a generic "2025 survey" range). |
| RSGPT / SkyEyeGPT / GeoChat (RS-MLLM family) | (transcript, no IDs) | **[NEEDS-VERIFICATION]** | Named families exist in the literature; locate specific papers before citing. |
| RS-MLLM / EO-MLLM survey(s) 2024–2025 | (transcript, no ID) | **[NEEDS-VERIFICATION]** | "70–95%" task-accuracy figures trace to unspecified "2025 surveys." |
| Time-VLM (vision-language + time series) | (transcript, no ID) | **[NEEDS-VERIFICATION]** | Verify. |

## 4. Papers (ecology / biodiversity / evaluation)

| Item | ID / Venue | Tag | Note |
|---|---|---|---|
| **Brown & Spillias — "Prompting LLMs for quality ecological statistics"** | *Methods in Ecology and Evolution* (transcript: "2026"; preprint ~arXiv:2505.06120) | **[NEEDS-VERIFICATION]** | **High-value, central to the talk's rigor message** (decompose workflow, supply context, CoT, human oversight). Likely real; **verify authors, venue, year, DOI / arXiv ID** before naming on a slide. The *practice* is adoptable now regardless. |
| LLM evaluation-policy extraction for ecological modeling (metric learning + NL policy; GPP/CO₂ flux) | arXiv:2505.13794 (transcript, "May 2025") | **[NEEDS-VERIFICATION]** | Useful "beyond-RMSE evaluation" idea; verify ID/claims. |
| "LLMs unlock the ecology of species interactions" (citizen-science + literature extraction) | bioRxiv (transcript, "Feb 2026") | **[NEEDS-VERIFICATION]** | No DOI; verify. |
| CURIE benchmark (long-context scientific reasoning, incl. biodiversity) | (transcript, "2025") | **[NEEDS-VERIFICATION]** | Verify. |

## 5. Papers / claims (math & frontier-model breakthroughs) — mostly flagged

| Item | ID / Venue | Tag | Note |
|---|---|---|---|
| OpenAI "Erdős breakthrough" (disproved planar unit-distance conjecture; 125-page CoT proof) | arXiv:2605.20695 (transcript, "20 May 2026") | **[FLAGGED]** | Future-dated ID + extraordinary unverifiable claim. **Do not use.** See ledger. |
| Gemini Deep Think / "Aletheia" PhD-level arithmetic-geometry result | DeepMind blog (transcript, "Feb 2026") | **[FLAGGED]** | Unverifiable; see ledger. |
| FrontierMath "40%+ postdoc-level" progress | (transcript, no source) | **[FLAGGED]** | Round-number, no primary source. |

## 6. Cognitive-load / communication-science claims — flagged

| Item | Source as given | Tag | Note |
|---|---|---|---|
| "Sweller 2025 update" — disengage after 5–7 items | (transcript) | **[FLAGGED]** | Cognitive Load Theory (Sweller) is real; the "2025 update" and the "7-item" threshold attribution are unverified. See ledger. Apply the *practice* (sparse slides), not the citation. |
| "Nature Human Behaviour 2026 meta-analysis" — clear takeaways + live demos → 2.5× application; pre-writing verification −65% overload, +72% usefulness | (transcript) | **[FLAGGED]** | No locatable paper; round-number stats. See ledger. |
| "+40–60% / +60–80% retention from audience-tailored talks" | (transcript) | **[FLAGGED]** | Free-floating ranges, no source. |

## 7. Tools, agents & frameworks

| Item | Type | Tag | Note |
|---|---|---|---|
| **Claude Code** (Anthropic) | Agentic coder | **[VERIFIED]** | The talk/repo build driver. |
| OpenAI Codex / o-series | Coding/reasoning models | **[VERIFIED]** (product line) | Generic capability claims OK; **specific "GPT-5.4" version is [FLAGGED]** — see ledger. |
| LangGraph / LangChain / CrewAI / AutoGen | Multi-agent orchestration | **[VERIFIED]** | Real OSS frameworks for L3 workflows. |
| **dlmastery/autoresearch** (`generalized_ml_autoresearch`) | Autoresearch loop (user's own) | **[VERIFIED]** | The methodology source; 7-step loop + hard gates. Private repo; also local at `C:/Users/evija/autoresearch/`. |
| karpathy/autoresearch (single-GPU, `program.md`-driven) | Autoresearch reference | **[NEEDS-VERIFICATION]** | Confirm repo handle/URL before citing publicly. |
| `conradry/open-coscientist-agents`, `The-Swarm-Corporation/AI-CoScientist`, `EvolvingLMMs-Lab/co-scientist`, `slapglif/Sifu`, `Future-House/robin` | Community co-scientist repos | **[NEEDS-VERIFICATION]** | Confirm each handle exists and does what the transcript claims before recommending. |
| Sakana AI-Scientist (GitHub) | Autonomous discovery | **[NEEDS-VERIFICATION]** | Real project; verify repo/handle. |

## 8. Time-series foundation models

| Item | Tag | Note (scale / 4090 use) |
|---|---|---|
| **TimesFM** (Google) | **[VERIFIED]** (project anchor) | Decoder-only TS foundation model; small variants run inference comfortably on a 4090. Good "use existing SOTA" demo backbone. |
| **Chronos** (Amazon) | **[VERIFIED]** (project anchor) | Tokenized-TS LLM; t5-family sizes (mini/small/base) are 4090-friendly for zero/few-shot forecasting. |
| **MOMENT** | **[VERIFIED]** (project anchor) | General TS foundation model; base sizes fit a 4090. |
| **Moirai** (Salesforce) | **[VERIFIED]** (project anchor) | Universal forecasting; small/base run on a 4090. |
| TimesBERT (BERT-style, "260B+ points") | **[NEEDS-VERIFICATION]** | Pre-train scale figure unverified; treat as a leads-item. |
| Tiny-TSM / SMETimes (sub-3B SLMs) | **[NEEDS-VERIFICATION]** | "Sub-3B, SOTA on Weather/ETT" unverified; if real, ideal for 4090. |
| TiRex / Sundial / Time-MoE | **[NEEDS-VERIFICATION]** | Listed as backbone stubs in the autoresearch loop; verify each. |

## 9. Datasets (with access / scale notes for 4090 use)

| Dataset | Domain | Tag | Access & 4090 scale notes |
|---|---|---|---|
| **ERA5** (ECMWF reanalysis) | Climate / weather gridded | **[VERIFIED]** | Via Copernicus CDS (`cdsapi`) — **API key + license acceptance required**. Full archive is petabytes; **for 4090 work, pull a small subset**: few variables (t2m, u10/v10, z, t@850hPa), coarse grid (e.g., 5.625° as ClimateLLM uses), short time window/region. Single-variable monthly subsets are MB–GB. |
| **GBIF** | Biodiversity occurrences | **[VERIFIED]** | Free API + bulk downloads (DOI'd). Filter by taxon + bounding box (e.g., Mexico) to keep CSVs in the MB–low-GB range; tabular, trivially 4090/CPU-friendly. |
| **iNaturalist** | Citizen-science obs + images | **[VERIFIED]** | API + open-data on AWS. Metadata is light; **image corpora are large** — sample for any vision task on a 4090. |
| **Sentinel-2** | Optical satellite (10–60 m) | **[VERIFIED]** | Copernicus / AWS / Microsoft Planetary Computer; **no key on some STAC endpoints**. Tiles are ~100s MB–GB each — **work at single-tile / single-AOI scale** and downsample for 4090 vision demos. |
| **CAMELS** | Hydrology (catchment attributes + streamflow) | **[VERIFIED]** | Free static download (US/other variants); **small enough to fit fully in RAM** — ideal LSTM-vs-conceptual streamflow demo (Exp08) on a 4090. |
| **OpenAQ** | Air quality (PM2.5, etc.) | **[VERIFIED]** | Free API. Tabular, lightweight; query by city/region. Good PM2.5 nowcasting demo (Exp07), runs on CPU/4090 easily. |
| **SoilGrids** (ISRIC) | Global soil properties | **[VERIFIED]** | Free WCS/REST + tiles. Global rasters are large; **clip to an AOI** for soil-carbon work on a 4090. |
| MODIS / Landsat 8-9 / Sentinel-5P | EO imagery | **[NEEDS-VERIFICATION]** (as cited) | Real datasets; verify the specific access path the experiment uses. Subset by AOI for 4090. |
| CMIP6 | Climate model ensembles | **[NEEDS-VERIFICATION]** (as cited) | Real (ESGF); very large — subset heavily. |
| ETT / Weather / Solar (TS benchmark sets) | TS benchmarks | **[VERIFIED]** | Standard small public TS datasets; fully 4090/CPU-friendly; good for forecasting BEFORE/AFTER. |
| NOAA archives, EM-DAT, EJSCREEN, Eurostat, UNSD FDES, OECD | Various env/policy | **[NEEDS-VERIFICATION]** (as cited) | Real sources; confirm exact dataset + access before use. |
| **CLLMate dataset** (~26k aligned ERA5+text; 41k+ news; KG 6k nodes/19k edges) | Multimodal | **[NEEDS-VERIFICATION]** | Scale figures from transcript; confirm release/availability before relying on it. |

## 10. Platforms / digital twins

| Item | Tag | Note |
|---|---|---|
| **BioDT** (Biodiversity Digital Twin, EU) | **[VERIFIED]** (project anchor) | biodt.eu; prototype DTs. Transcript's "FAIR digital twins for biodiversity, npj Biodiversity, Feb 2026 (Islam et al.)" specific paper is **[NEEDS-VERIFICATION]**. |
| **DestinE** (Destination Earth, EC) | **[VERIFIED]** (project anchor) | destination-earth.eu; includes a Climate Digital Twin. "May 2026 data releases / four new services" specifics are **[NEEDS-VERIFICATION]**. |
| SyncED-Ocean (marine DT) | **[NEEDS-VERIFICATION]** | "OCEANS 2025 Brest, Mansfield et al." — verify. |

## 11. Benchmarks

| Item | Tag | Note |
|---|---|---|
| GIFT-Eval; Chronos benchmarks (time series) | **[NEEDS-VERIFICATION]** | Plausible; verify exact names/availability. |
| WCEF (Weather & Climate Event Forecasting) | **[NEEDS-VERIFICATION]** | Introduced by CLLMate (arXiv:2409.19058); confirm via that paper. |
| CURIE; EarthSE | **[NEEDS-VERIFICATION]** | Verify. |
| HeurekaBench (AI co-scientist eval, "atomic-fact") | arXiv:2601.01678 (transcript) | **[FLAGGED]** | Future-dated ID (2601). See ledger. |
| RS-MLLM evaluation suites; EVA (extreme value) | **[NEEDS-VERIFICATION]** | Generic references; locate specifics. |

## 12. Frontier models named in the transcript

| Model | Tag | Note |
|---|---|---|
| Gemini 2.x (used by co-scientist preprint) | **[VERIFIED]** (line exists) | Generic reference OK. |
| Llama3-8B / Llama3-70B (CLLMate components) | **[VERIFIED]** | Real; as described in CLLMate. |
| CLIP ViT-L/14 (CLLMate visual encoder) | **[VERIFIED]** | Real. |
| GPT-2 (ClimateLLM backbone) | **[VERIFIED]** | Real. |
| **GPT-5.4 (Standard/Thinking/Pro); Gemini 3.5; Gemini Deep Think "Aletheia"; Claude Opus 4.5; GPT-5.2 / Gemini Pro 3; WeatherNext 2 specifics** | **[FLAGGED]** | Unreleased / unconfirmed version names and capability claims. **Do not assert.** See ledger. |
