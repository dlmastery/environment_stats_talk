# WeatherNext / WeatherNext 2 (Google DeepMind) — availability note

**Status (2026-05-28): NOT-PUBLIC as downloadable weights.**

Verified via the official landing page <https://deepmind.google/science/weathernext/>
on 2026-05-26:

- WeatherNext and WeatherNext 2 are **service-only** releases. There is
  **no open GitHub repository**, **no downloadable checkpoints**, **no
  public PyPI package**.
- Access modes published by Google as of the date above:
  - **BigQuery** — `bigquery-public-data.weathernext_graph_forecasts`-style
    tables for historical and short-lead forecasts.
  - **Google Earth Engine** — ImageCollection assets.
  - **Vertex AI / Gemini Enterprise Agent Platform** — for businesses to
    customize and create WeatherNext-based forecasts.
  - **Developer docs** — <https://developers.google.com/weathernext>.
  - **Weather Lab** — an experimental UI for exploring forecasts.

Therefore there is nothing for this experiment to "run" in the
zero-API-key, offline setting of this repo. The WeatherNext entry in
`docs/FOUNDATION_MODELS.md` records this status precisely so the talk
does not overclaim availability.

If a viewer of the talk wants to consume WeatherNext outputs, the
honest path is to:
1. Authenticate to Google Cloud,
2. Query BigQuery (or pull EE assets) for the geography + period of
   interest, and
3. Compare against a baseline model (e.g., the GraphCast_small run we
   ran in this repo, or persistence).

See the project's `climate-data-fetch` skill for the auth boilerplate.
