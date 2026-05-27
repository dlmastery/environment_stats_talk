# CITATIONS TO VERIFY — Quarantine Ledger

**Purpose.** The source transcript (`grok-llms-ai-environmental-statistics-transcript.md`) was produced by a chat assistant and is dense with **future-dated, unfindable, or round-number claims** stated as fact. The audience is environmental statisticians who *punish over-claiming*. Therefore:

> **Nothing in this file may appear on a slide, in a handout, in a README, or in a code comment as TRUE until it has been independently verified against a primary source.** If a talking point depends only on a claim listed here, **cut the talking point** rather than weaken the rigor.

**How to use.** Each entry has: the **Claim** (as the transcript states it), **Why suspect**, and **What to check before it goes on a slide**. Categories: 🔮 future-dated · 🔎 no locatable source · 🔢 round-number / unsourced statistic · 🤖 unreleased model version.

Status key: ☐ unverified · ☑ verified (record the primary source + correction) · ✗ refuted (do not use).

---

## A. Future-dated publication claims (🔮 highest-risk — most quotable)

### A1. ☐ "Co-Scientist published in *Nature*, 19 May 2026" (Gottweis et al., "Accelerating scientific discovery with Co-Scientist")
- **Why suspect:** Precise future date (the transcript's own "today" is 22–26 May 2026, so this is "published 3–7 days ago"). A specific *Nature* date is the most citeable and therefore most dangerous form of claim. The **preprint** "Towards an AI co-scientist" (arXiv:2502.18864) is real and verified; the *Nature* paper, exact title, author list, and date are not.
- **Check before slide:** Resolve a real Nature DOI for the exact title + date; confirm authors (the transcript's "Gottweis et al." may be invented). **Until then, cite only the arXiv:2502.18864 preprint** and describe it as a preprint, not a Nature paper.

### A2. ☐ "AI-Scientist-v2 published in *Nature*, early 2026 / first fully AI-generated peer-reviewed paper" (Yamada et al.)
- **Why suspect:** The arXiv preprint (2504.08066) is real/verified; the "Nature, early 2026" upgrade and the superlative "first AI-generated paper to pass rigorous peer review" are unverified and exactly the kind of claim an analytical audience will challenge.
- **Check before slide:** Find the Nature DOI; confirm the "peer-reviewed / workshop-accepted" claim against a primary venue record. **Until then, cite the preprint and avoid the "first ever" superlative.**

### A3. ✗-leaning ☐ OpenAI "Erdős Breakthrough," 20 May 2026 — arXiv:2605.20695
- **Claim:** A general-purpose model autonomously disproved an 80-year-old planar unit-distance conjecture (Erdős 1946), produced a 125-page chain-of-thought proof, verified by Alon/Wood/Gowers; "human-digested" arXiv:2605.20695.
- **Why suspect:** (1) arXiv ID **2605.xxxxx implies May 2026** — at the time of writing such IDs are not assignable/locatable. (2) Extraordinary mathematical claim with named endorsers is a classic fabrication pattern. (3) "2 days ago" framing.
- **Check before slide:** Confirm arXiv:2605.20695 resolves to this exact paper; find an OpenAI primary announcement; confirm the named mathematicians endorsed it. **Treat as do-not-use unless all three check out.** It is tangential to environmental statistics anyway — safe to omit entirely.

### A4. ☐ ClimAgent — arXiv:2604.16922 ("Apr 2026"); A5. ☐ Kosmos — arXiv:2511.02824; A6. ☐ "VLMs for textual weather forecasts" — arXiv:2512.03623 ("Dec 2025"); A7. ☐ "AI Co-Scientist for Ranking" — arXiv:2603.22376 ("Mar 2026"); A8. ☐ HeurekaBench — arXiv:2601.01678 ("Feb 2026")
- **Why suspect:** All carry **future-dated arXiv IDs** (26xx / 25{11,12}xx) that cannot currently be confirmed to resolve to the claimed papers.
- **Check before slide:** For each ID, confirm it resolves to a real paper with the claimed title/content. **Until then, do not cite the IDs.** None is load-bearing for the talk.

### A9. ☐ "FAIR digital twins for biodiversity," *npj Biodiversity*, Feb 2026 (Islam et al.); A10. ☐ DestinE "May 2026 data releases / four new services"; A11. ☐ SyncED-Ocean, OCEANS 2025 Brest (Mansfield et al.)
- **Why suspect:** **BioDT and DestinE are real platforms (verified anchors)** — but these *specific* papers/dates/release notes are unconfirmed.
- **Check before slide:** Find the npj Biodiversity DOI; confirm DestinE release notes on the official EC programme page; verify the OCEANS-2025 paper. **Cite the platforms (biodt.eu, destination-earth.eu), not the unconfirmed paper specifics.**

---

## B. Unreleased / unconfirmed model versions (🤖)

### B1. ☐ "GPT-5.4 series (Standard / Thinking / Pro)"; B2. ☐ "GPT-5.2"; B3. ☐ "Gemini 3.5"; B4. ☐ "Gemini Pro 3"; B5. ☐ "Gemini Deep Think / Aletheia"; B6. ☐ "Claude Opus 4.5"; B7. ☐ "WeatherNext 2" specifics
- **Why suspect:** Named model versions presented as released, with specific capability claims (e.g., "Thinking mode reduces output tokens 50–80%," "Aletheia produced a PhD-level arithmetic-geometry result," "WeatherNext 2 operational 15-day cyclone lead, >2B people"). These are unverifiable version numbers and round-number capability claims.
- **Check before slide:** Confirm each version exists via an official vendor release page at talk time; confirm any quoted capability number against a primary source. **Default: refer to capabilities generically ("frontier reasoning models," "recent weather foundation models") rather than naming a version or quoting a number.** Model names age out fast — generic framing is also safer pedagogy.

---

## C. Round-number / unsourced performance statistics (🔢)

> Even when attached to a **real, verified paper**, the *specific number* must be confirmed against that paper's abstract/tables. Recurring suspicious figures:

### C1. ☐ ClimateLLM (arXiv:2502.11059): "19–43% RMSE reduction," "87–98% (also 87.5%/92.7%/98.7%) training-time/memory reduction," "ACC 0.98–1.00," "high ACC with 20% data"
- **Why suspect:** Wide round ranges, repeated inconsistently across the transcript; paper is real but numbers are quoted second-hand.
- **Check:** Open arXiv:2502.11059; quote the paper's *actual* reported metrics with their *exact* baselines and lead times, or say "see paper for reported gains."

### C2. ☐ EagleVision / RS-MLLMs: "70–95% object/attribute accuracy"
- **Why suspect:** A single range applied generically to a whole model family, sourced to unspecified "2025 surveys."
- **Check:** Tie any number to one specific paper + benchmark + task; otherwise drop the number.

### C3. ☐ CLLMate (arXiv:2409.19058): "BLEU-1 52.56, ROUGE-L 51.87, METEOR 39.00, BERTScore 73.56"
- **Why suspect:** Oddly precise vs. baseline "~30 / ~36"; quoted second-hand.
- **Check:** Confirm against the CLLMate paper's results tables before quoting.

### C4. ☐ Speed/impact superlatives: "science ~100× faster," "80% / 90% / 70% time saved," "5–10× faster discovery," "days vs years," "processes 10k+ comments in hours vs months"
- **Why suspect:** Marketing-style multipliers with no controlled comparison; "days vs years" repeated as if measured.
- **Check:** These are **theses to demonstrate per BEFORE/AFTER experiment with the repo's own measured numbers**, not facts to assert. On slides, show *your* measured comparison, not a transcript multiplier.

### C5. ☐ "FrontierMath 40%+ postdoc-level"; "Genesis: 25+ new magnetic materials"; "Rutgers/NY grid ~5% reserve savings"; "flood forecasting >2B people, 150 countries"
- **Why suspect:** Round numbers, no primary source, mostly off-topic for environmental statistics.
- **Check:** Primary source for each; otherwise omit.

---

## D. Communication-science / cognitive-load statistics (🔢🔎)

### D1. ☐ "Cognitive Load Theory — Sweller, 2025 update; audiences disengage after 5–7 items"
- **Why suspect:** Sweller's CLT is real and foundational, but a specific **"2025 update"** and the **"7-item"** disengagement threshold attribution are unverified (the "7±2" idea is Miller, not Sweller, and concerns working memory, not slide bullets).
- **Check before slide:** Don't cite a "Sweller 2025 update." **Apply the design practice (sparse, scannable slides) without attributing a precise number to a named source.**

### D2. ☐ "Nature Human Behaviour 2026 meta-analysis: clear takeaways + live demos → 2.5× post-talk application; pre-writing verification −65% overload, +72% usefulness"
- **Why suspect:** No locatable paper; suspiciously precise effect sizes (2.5×, 65%, 72%) for a "2026 meta-analysis."
- **Check before slide:** Find the DOI. **Until then, do not state these numbers.** The underlying advice (clear takeaways, demos, pre-writing verification) is sound and can be given as craft guidance without fabricated effect sizes.

### D3. ☐ "Audience-tailored talks increase retention 40–60% / 60–80%"
- **Why suspect:** Two different free-floating ranges, no source.
- **Check:** Omit the percentages; keep the qualitative principle.

---

## E. Named papers/people lacking a locatable identifier (🔎)

### E1. ☐ Brown & Spillias, "Prompting LLMs for quality ecological statistics," *Methods in Ecology and Evolution* (preprint ~arXiv:2505.06120)
- **Why suspect:** **Likely real and is central to the talk's human-in-the-loop rigor message** — but the transcript gives inconsistent specifics ("2026" vs "preprint ~2025"; a tentative arXiv ID).
- **Check before slide:** Confirm exact authors, title, venue, year, and DOI / arXiv ID. This one is worth chasing down because it directly supports the rigor narrative. **The prompting *practice* is adoptable now; the *citation* must be nailed before it appears.**

### E2. ☐ "LLMs unlock the ecology of species interactions" (bioRxiv, Feb 2026); E3. ☐ LLM evaluation-policy extraction (arXiv:2505.13794); E4. ☐ Robin (arXiv:2505.13400); E5. ☐ EarthLink (arXiv:2507.17311); E6. ☐ Tie et al. survey (arXiv:2510.23045); E7. ☐ Wei et al. survey (arXiv:2508.14111); E8. ☐ GPT-4 rainfall (arXiv:2411.13724); E9. ☐ AlphaEvolve (arXiv:2506.13131)
- **Why suspect:** Plausible IDs but unverified from the transcript alone; several attach to specific claims.
- **Check before slide:** Confirm each arXiv ID resolves to the stated paper; verify any quoted claim. Use only those that resolve.

### E10. ☐ Community repos: `conradry/open-coscientist-agents`, `The-Swarm-Corporation/AI-CoScientist`, `EvolvingLMMs-Lab/co-scientist`, `slapglif/Sifu`, `Future-House/robin`, Sakana AI-Scientist, karpathy/autoresearch
- **Why suspect:** Repo handles are easy to mis-cite; some may not exist or may not do what is claimed.
- **Check before slide:** Confirm each handle resolves and matches the described function before recommending it to the audience.

---

## F. Industry / deployment claims (🔎🔢) — mostly off-topic, verify or omit

### F1. ☐ Co-Scientist validated in liver-fibrosis organoids / AML drug repurposing / bacterial AMR ("matched multi-year lab results in days"); pilots at Stanford, Imperial, Bayer Crop Science, US National Labs ("Genesis Mission")
- **Why suspect:** Specific wet-lab validations and named institutional pilots, all unsourced.
- **Check:** Primary sources for each. **These are biomedical, not environmental — safe to omit from this talk.**

### F2. ☐ "Google.org Impact Challenge: AI for Science (2026), $30M+, AI for Climate Resilience track"; F3. ☐ "Gemini for Science suite (May 2026)"; F4. ☐ "labs.google/science Hypothesis Generation tool (rolling out May 2026)"; F5. ☐ "CGIAR AI Co-Scientist Framework (2025)"; F6. ☐ "PNNL Permit AI"
- **Why suspect:** Future-dated programs/products and dollar figures presented as fact; the "labs.google/science" tool is used in transcript prompt templates as if live.
- **Check before slide:** Confirm each via an official page **at talk time**. **Do not put a specific tool URL in a demo unless it is confirmed live**, or the live demo fails on stage.

---

## Verification workflow (apply before any item leaves quarantine)

1. **Resolve the identifier.** arXiv ID → does it open the claimed paper? DOI → real? Repo handle → exists + matches description?
2. **Confirm date & venue** independently (a "Nature, [date]" claim needs a Nature DOI, not a secondary mention).
3. **Re-derive every number** from the primary source's abstract/tables; never quote a transcript number.
4. **Prefer the conservative anchor.** If only a preprint is confirmed, cite the preprint and say so; never upgrade a preprint to "Nature."
5. **If it can't be verified, cut it.** For this audience, an omitted claim costs nothing; a wrong claim costs the whole talk's credibility.
6. **Log the outcome** here: flip ☐→☑ with the confirmed citation, or ☐→✗ and remove from all build artifacts.

---

## Verification status (web-checked 2026-05-26)

> Status: **CONFIRMED** = resolves to a real primary source matching the claim · **CORRECTED** = real, but the transcript's title/author/date/version was wrong (correction recorded) · **UNVERIFIED** = could not locate a primary source. Numbers (RMSE/BLEU/etc. ranges in §C) were **NOT** independently re-derived in this pass — they remain `[NEEDS-VERIFICATION]`; only identifiers, titles, authors, venues, dates, and existence were checked. Because the project's real "today" is 2026-05-26, several items previously flagged as "future-dated" are now in the past and *are* findable.

### Verified arXiv anchors (titles/authors/years confirmed against arxiv.org/abs/<id>)

- **2502.11059 — CONFIRMED.** "ClimateLLM: Efficient Weather Forecasting via Frequency-Aware Large Language Models," first author **Shixuan Li** (Li, Yang, Zhang, Xiao, Cao, Qin, Zhang, Zhao, Bogdan), submitted **Feb 2025**. ID matches the described ClimateLLM paper. URL: https://arxiv.org/abs/2502.11059
- **2409.19058 — CONFIRMED (minor title correction).** Exact title is **"CLLMate: A Multimodal Benchmark for Weather and Climate Events Forecasting,"** first author **Haobo Li** (Li, Z. Wang, J. Wang, Y. Wang, Lau, Qu), submitted **Sep 2024**. ID matches CLLMate. (Transcript's "multimodal LLM for WCEF" gist is right; the paper frames itself as a *benchmark/dataset* — 26,156 ERA5-aligned news articles.) EMNLP-2025 acceptance not separately confirmed in this pass. URL: https://arxiv.org/abs/2409.19058
- **2502.18864 — CONFIRMED.** "Towards an AI co-scientist," first author **Juraj Gottweis** (34 authors incl. Natarajan), submitted **Feb 2025**. ID matches; this is the preprint. URL: https://arxiv.org/abs/2502.18864
- **2504.08066 — CONFIRMED.** "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search," first author **Yutaro Yamada** (Yamada, Lange, C. Lu, Hu, C. Lu, Foerster, Clune, Ha), submitted **Apr 2025**. ID matches AI-Scientist-v2. URL: https://arxiv.org/abs/2504.08066
- **2503.23330 — CONFIRMED.** "EagleVision: Object-level Attribute Multimodal LLM for Remote Sensing," first author **Hongxiang Jiang** (Jiang, Yin, Wang, Feng, Chen); introduces EVAttrs-95K. Submitted **Mar 2025**. ID matches EagleVision. URL: https://arxiv.org/abs/2503.23330

### E1. Brown & Spillias — CONFIRMED (now a journal article, not a preprint)

- **CONFIRMED.** C. J. Brown & Scott Spillias, **"Prompting large language models for quality ecological statistics,"** *Methods in Ecology and Evolution* (2026), **Vol. 17, pp. 1012–1021**, **DOI: 10.1111/2041-210x.70267**. The transcript's "~arXiv:2505.06120" preprint ID was a guess and is NOT how to cite it — use the MEE DOI. The paper's core finding (generic prompt never suggested the appropriate count model for fish-abundance data; a detailed prompt specifying variable types/sample size/design reliably did) directly supports the talk's human-in-the-loop rigor message. URL: https://doi.org/10.1111/2041-210x.70267 (open page: https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70267)

### A1. Co-Scientist *Nature* paper — CONFIRMED (the future-dated claim checks out)

- **CONFIRMED.** "Accelerating scientific discovery with Co-Scientist," Gottweis, J., Weng, W-H., Daryin, A. et al., **Nature**, **DOI: 10.1038/s41586-026-10644-y**, published online **19 May 2026**. Corroborated by Nature's own listing, the Nature Asia press release, and secondary coverage (techxplore 2026-05). The arXiv preprint 2502.18864 remains the conservative anchor; the Nature paper is now real and citeable with the DOI. (Nature article page is behind an auth redirect, but the DOI resolves to it and multiple independent sources confirm title/authors/date.) URLs: https://doi.org/10.1038/s41586-026-10644-y · https://www.nature.com/articles/s41586-026-10644-y

### A2. AI-Scientist-v2 *Nature* paper — CONFIRMED, but with a CORRECTED title

- **CONFIRMED (title corrected).** The Nature paper IS real, but its title is **"Towards end-to-end automation of AI research,"** **Nature 651, 914–919 (2026)**, **DOI: 10.1038/s41586-026-10265-5** (Sakana AI / UBC / Vector Institute / Oxford). It is NOT titled "AI-Scientist-v2" — that is the arXiv preprint (2504.08066). The "first fully AI-generated manuscript to pass peer review" claim refers specifically to a **workshop** acceptance (ICLR workshop, avg reviewer score ~6.33), described in the v2 preprint; phrase it as "first AI-generated paper to pass a *workshop* peer review," not "first peer-reviewed paper" unqualified. URLs: https://www.nature.com/articles/s41586-026-10265-5 · https://sakana.ai/ai-scientist-nature/

### A3. OpenAI "Erdős breakthrough" / arXiv:2605.20695 — CONFIRMED (no longer "do-not-use")

- **CONFIRMED.** arXiv:**2605.20695** resolves to **"Remarks on the disproof of the unit distance conjecture,"** authors **Noga Alon, Thomas F. Bloom, W. T. Gowers, Daniel Litt, Will Sawin, Arul Shankar, Jacob Tsimerman, Victor Wang, Melanie Matchett Wood**, submitted **20 May 2026**. It is a short, human-verified write-up of an OpenAI-model-generated counterexample to the Erdős (1946) unit-distance conjecture; an OpenAI PDF abstract also exists. The named endorsers (Alon/Gowers/Wood et al.) are the actual authors of this companion note. **Still tangential to environmental statistics — safe to omit from the talk on relevance grounds, not on falsity grounds.** URLs: https://arxiv.org/abs/2605.20695 · https://cdn.openai.com/pdf/74c24085-19b0-4534-9c90-465b8e29ad73/unit-distance-proof.pdf

### B. Model versions — mostly CONFIRMED as real releases

- **GPT-5.2 — CONFIRMED.** Released **11 Dec 2025** (instant/thinking/Pro + Codex variant). URL: https://openai.com/index/introducing-gpt-5-2/
- **GPT-5.4 — CONFIRMED.** Released **5 Mar 2026** (Thinking/Pro, then mini/nano on Mar 17); native computer use, 1M context, tool-search token savings. (Lineup since: GPT-5.5 in API on 24 Apr 2026.) The transcript's "GPT-5.4 Standard/Thinking/Pro" is essentially right. URLs: https://en.wikipedia.org/wiki/GPT-5.4 · https://help.openai.com/en/articles/6825453-chatgpt-release-notes
- **Claude Opus 4.5 — CONFIRMED.** Released **~24 Nov 2025** (`claude-opus-4-5-20251101`). (Opus 4.7 also now exists, 2026.) URL: https://www.anthropic.com/news/claude-opus-4-5
- **Gemini Pro 3 / "Gemini 3 Pro" — CONFIRMED.** Gemini 3 Pro released **Nov 2025**; Gemini 3.1 Pro **19 Feb 2026**. URL: https://blog.google/products/gemini/gemini-3/
- **Gemini 3.5 — CONFIRMED (exists), capability claim CORRECTED.** Gemini 3.5 family announced (3.5 Flash shipped; **3.5 Pro slated June 2026**). BUT the transcript's **"Gemini 3.5 / Deep Think Aletheia"** pairing is wrong: **Aletheia is powered by Gemini 3 *Deep Think* (Gemini 3 line, Jan/Feb 2026 iteration), not Gemini 3.5.** Aletheia's autonomous arithmetic-geometry result is the **"Feng26"** paper computing "eigenweights" (graded "Level A2" autonomy), and it also resolved 4 open Erdős problems. Attribute Aletheia to **Gemini 3 Deep Think**, not "Gemini 3.5." URLs: https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/ · https://deepmind.google/blog/accelerating-mathematical-and-scientific-discovery-with-gemini-deep-think/
- **WeatherNext 2 — CONFIRMED (verify specific numbers separately).** Real Google DeepMind release: 8× faster, up to 1-hour resolution, ensembles in <1 min on one TPU, beats WeatherNext on 99.9% of variables/lead-times (0–15 days), Functional Generative Network (FGN); data in Earth Engine/BigQuery, early access on Vertex AI. The transcript's specific impact figures ("operational 15-day cyclone lead, >2B people") were NOT confirmed in this pass — cite the model + the official 0–15-day lead-time figure, not the population claim. URLs: https://blog.google/innovation-and-ai/models-and-research/google-deepmind/weathernext-2/ · https://deepmind.google/science/weathernext/

### Items NOT checked in this pass (remain as previously tagged)

A4–A11 (other future-dated arXiv IDs, BioDT/DestinE specifics, SyncED-Ocean), §C numeric ranges (C1–C5), §D communication-science stats (D1–D3), §E2–E10 (other arXiv IDs / repos), and §F industry/deployment claims were **not** part of this verification request and keep their existing status. §D fabricated effect sizes and §C round-number ranges in particular remain **UNVERIFIED — do not assert**.
