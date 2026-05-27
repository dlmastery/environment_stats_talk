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
| **ClimateLLM** — "ClimateLLM: Efficient Weather Forecasting via Frequency-Aware Large Language Models" (2D FFT + Frequency-MoE + dynamic prompting, GPT-2 backbone, latitude-weighted RMSE); first author Shixuan Li, Feb 2025 | arXiv:2502.11059 | **[VERIFIED]** (web-checked 2026-05-26: title/author/year confirmed) | Core "climate forecasting" anchor; runs on ERA5. Quoted gains ("19–43% RMSE," "87–98% training-time reduction," "ACC 0.98–1.00") remain **[NEEDS-VERIFICATION]** — not re-derived; check abstract/tables before quoting. URL: https://arxiv.org/abs/2502.11059 |
| **CLLMate** — exact title "CLLMate: A Multimodal Benchmark for Weather and Climate Events Forecasting"; first author Haobo Li, Sep 2024; 26,156 ERA5-aligned news articles | arXiv:2409.19058 (transcript says EMNLP 2025) | **[VERIFIED]** (web-checked 2026-05-26: title/author/year confirmed) | "Numbers → natural-language events" anchor. Title corrected (it is framed as a *benchmark/dataset*). EMNLP-2025 acceptance and BLEU/ROUGE/BERTScore figures still **[NEEDS-VERIFICATION]**. URL: https://arxiv.org/abs/2409.19058 |
| GPT-4 zero-shot rainfall forecasting (conservative / mean-reverting) | arXiv:2411.13724 (transcript) | **[NEEDS-VERIFICATION]** | Useful "LLM fails without grounding" example. Verify ID and claims. |
| ClimAgent — agentic open-ended climate analysis | arXiv:2604.16922 (transcript, "Apr 2026") | **[FLAGGED]** | Future-dated ID (26xx = 2026). See ledger. |
| VLMs for textual weather forecasts | arXiv:2512.03623 (transcript, "Dec 2025") | **[FLAGGED]** | Future-dated ID. See ledger. |
| STELLM (spatio-temporal LLM for wind speed) | (transcript, no ID) | **[NEEDS-VERIFICATION]** | No locatable identifier given. |

## 2. Papers (agentic science / autonomous discovery)

| Item | ID / Venue | Tag | Note |
|---|---|---|---|
| **"Towards an AI co-scientist"** — multi-agent generate–debate–evolve + tournament (Gemini); first author Juraj Gottweis, Feb 2025 | arXiv:2502.18864 | **[VERIFIED]** (web-checked 2026-05-26: preprint title/author/year confirmed) | The *preprint* is the safe anchor. The Nature paper **"Accelerating scientific discovery with Co-Scientist" (Gottweis et al., Nature, DOI 10.1038/s41586-026-10644-y, online 19 May 2026)** is now **[VERIFIED]** — citeable with the DOI. Preprint URL: https://arxiv.org/abs/2502.18864 · Nature DOI: https://doi.org/10.1038/s41586-026-10644-y |
| **AI-Scientist-v2** — "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search"; first author Yutaro Yamada, Apr 2025 | arXiv:2504.08066 | **[VERIFIED]** (web-checked 2026-05-26: preprint title/author/year confirmed) | Preprint confirmed. The Nature paper is **[VERIFIED]** but titled **"Towards end-to-end automation of AI research," Nature 651, 914–919 (2026), DOI 10.1038/s41586-026-10265-5** (NOT "AI-Scientist-v2"). The "first AI-generated paper to pass peer review" is specifically a **workshop** (ICLR) acceptance — qualify the superlative. Preprint URL: https://arxiv.org/abs/2504.08066 · Nature URL: https://www.nature.com/articles/s41586-026-10265-5 |
| Robin — multi-agent discovery | arXiv:2505.13400 (transcript) | **[NEEDS-VERIFICATION]** | Plausible 2025 ID; verify. |
| "Expert-level empirical software" agent (LLM + tree search; geospatial examples) | arXiv:2509.06503 (transcript) | **[NEEDS-VERIFICATION]** | Verify ID/claims. |
| EarthLink — self-evolving climate-science agent | arXiv:2507.17311 (transcript) | **[NEEDS-VERIFICATION]** | Verify ID. |
| Kosmos | arXiv:2511.02824 (transcript) | **[FLAGGED]** | Future-dated ID (2511 = Nov 2025, after the conversation's own "May 2026" framing is inconsistent; ID unverifiable). See ledger. |
| "AI Co-Scientist for Ranking" (claims models GPT-5.2, Gemini Pro 3, Claude Opus 4.5) | arXiv:2603.22376 (transcript, "Mar 2026") | **[FLAGGED]** (arXiv ID not yet checked) | The *model versions* (GPT-5.2, Gemini Pro 3, Claude Opus 4.5) are now confirmed real (see §12). The **arXiv ID 2603.22376 itself was not verified in the 2026-05-26 pass** — confirm it resolves before citing. |
| OpenScientist (open agentic co-scientist eval) | medRxiv (transcript, "Mar 2026") | **[NEEDS-VERIFICATION]** | No DOI; verify. |
| "Survey of AI Scientists" (Tie, Zhou, Sun) | arXiv:2510.23045 (transcript, "v5 rev Jan 2026") | **[NEEDS-VERIFICATION]** | Plausible survey ID; verify ID, authors, and version. |
| "From AI for Science to Agentic Science" (Wei et al.) | arXiv:2508.14111 (transcript) | **[NEEDS-VERIFICATION]** | Verify. |
| AlphaEvolve (DeepMind evolutionary coding agent) | arXiv:2506.13131 (transcript) | **[NEEDS-VERIFICATION]** | Real product line; verify the specific arXiv ID and any optimization claims. |

## 3. Papers (remote sensing / EO multimodal LLMs)

| Item | ID / Venue | Tag | Note |
|---|---|---|---|
| **EagleVision** — "EagleVision: Object-level Attribute Multimodal LLM for Remote Sensing"; first author Hongxiang Jiang, Mar 2025; introduces EVAttrs-95K | arXiv:2503.23330 | **[VERIFIED]** (web-checked 2026-05-26: title/author/year confirmed) | Remote-sensing anchor. Quoted "70–95% accuracy" remains **[NEEDS-VERIFICATION]** (also stated as a generic "2025 survey" range). URL: https://arxiv.org/abs/2503.23330 |
| RSGPT / SkyEyeGPT / GeoChat (RS-MLLM family) | (transcript, no IDs) | **[NEEDS-VERIFICATION]** | Named families exist in the literature; locate specific papers before citing. |
| RS-MLLM / EO-MLLM survey(s) 2024–2025 | (transcript, no ID) | **[NEEDS-VERIFICATION]** | "70–95%" task-accuracy figures trace to unspecified "2025 surveys." |
| Time-VLM (vision-language + time series) | (transcript, no ID) | **[NEEDS-VERIFICATION]** | Verify. |

## 4. Papers (ecology / biodiversity / evaluation)

| Item | ID / Venue | Tag | Note |
|---|---|---|---|
| **Brown & Spillias — "Prompting large language models for quality ecological statistics"** (C. J. Brown & Scott Spillias) | *Methods in Ecology and Evolution* **17, 1012–1021 (2026)**, **DOI 10.1111/2041-210x.70267** | **[VERIFIED]** (web-checked 2026-05-26: title/authors/venue/year/DOI confirmed) | **High-value, central to the talk's rigor message.** Now a published journal article — cite the MEE DOI, NOT the transcript's guessed "arXiv:2505.06120." Key finding: generic prompts missed the appropriate count model for fish-abundance data; detailed prompts (variable types, sample size, design) reliably produced it. URL: https://doi.org/10.1111/2041-210x.70267 |
| LLM evaluation-policy extraction for ecological modeling (metric learning + NL policy; GPP/CO₂ flux) | arXiv:2505.13794 (transcript, "May 2025") | **[NEEDS-VERIFICATION]** | Useful "beyond-RMSE evaluation" idea; verify ID/claims. |
| "LLMs unlock the ecology of species interactions" (citizen-science + literature extraction) | bioRxiv (transcript, "Feb 2026") | **[NEEDS-VERIFICATION]** | No DOI; verify. |
| CURIE benchmark (long-context scientific reasoning, incl. biodiversity) | (transcript, "2025") | **[NEEDS-VERIFICATION]** | Verify. |

## 5. Papers / claims (math & frontier-model breakthroughs) — mostly flagged

| Item | ID / Venue | Tag | Note |
|---|---|---|---|
| OpenAI "Erdős breakthrough" — companion note "Remarks on the disproof of the unit distance conjecture" (Alon, Bloom, Gowers, Litt, Sawin, Shankar, Tsimerman, V. Wang, M. M. Wood); human-verified write-up of an OpenAI-model-generated counterexample to the Erdős (1946) unit-distance conjecture | arXiv:2605.20695 (20 May 2026) | **[VERIFIED]** (web-checked 2026-05-26: ID resolves; authors/date confirmed) | ID now resolves (real "today" is 2026-05-26). Tangential to environmental statistics — **omit from the talk on relevance grounds, not falsity**. URL: https://arxiv.org/abs/2605.20695 |
| Gemini Deep Think / "Aletheia" — autonomous arithmetic-geometry result ("Feng26," computes "eigenweights," Level-A2 autonomy); also resolved 4 open Erdős problems | Google DeepMind blog (Feb 2026) | **[VERIFIED]** (web-checked 2026-05-26) | **Correction:** Aletheia is powered by **Gemini 3 *Deep Think***, NOT "Gemini 3.5." Cite as Gemini 3 Deep Think. URL: https://deepmind.google/blog/accelerating-mathematical-and-scientific-discovery-with-gemini-deep-think/ |
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
| OpenAI Codex / o-series / GPT-5.x | Coding/reasoning models | **[VERIFIED]** (product line; GPT-5.2 Dec 2025, GPT-5.4 Mar 2026 confirmed 2026-05-26) | Generic capability claims OK; specific version names GPT-5.2/5.4 are now confirmed real (see §12). Capability *numbers* still verify per source. |
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
| **GPT-5.4 (Thinking/Pro)** — released 5 Mar 2026 | **[VERIFIED]** (web-checked 2026-05-26) | Real OpenAI release (native computer use, 1M context). https://en.wikipedia.org/wiki/GPT-5.4 |
| **GPT-5.2** — released 11 Dec 2025 | **[VERIFIED]** (web-checked 2026-05-26) | Real (instant/thinking/Pro + Codex). https://openai.com/index/introducing-gpt-5-2/ |
| **Claude Opus 4.5** — released ~24 Nov 2025 (`claude-opus-4-5-20251101`) | **[VERIFIED]** (web-checked 2026-05-26) | Real (Opus 4.7 also now exists). https://www.anthropic.com/news/claude-opus-4-5 |
| **Gemini Pro 3 / "Gemini 3 Pro"** — Nov 2025 (Gemini 3.1 Pro 19 Feb 2026) | **[VERIFIED]** (web-checked 2026-05-26) | Real. https://blog.google/products/gemini/gemini-3/ |
| **Gemini 3.5** — family announced (3.5 Flash shipped; 3.5 Pro slated June 2026) | **[VERIFIED]** (web-checked 2026-05-26) | Real family. **Note:** Aletheia runs on Gemini 3 *Deep Think*, NOT Gemini 3.5. https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/ |
| **WeatherNext 2** (model itself) | **[VERIFIED]** (web-checked 2026-05-26) | Real Google DeepMind release (8× faster, 0–15 day lead, FGN). Transcript's ">2B people / 150-country / 15-day cyclone" impact figures remain **[NEEDS-VERIFICATION]**. https://deepmind.google/science/weathernext/ |
