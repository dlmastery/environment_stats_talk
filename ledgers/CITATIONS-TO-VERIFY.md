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
