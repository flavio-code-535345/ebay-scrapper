# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

A Flask app that searches eBay Germany (and Kleinanzeigen) for secondhand video-game listings, scores each
one for resale profitability with Gemini AI plus deterministic rules, and serves the results through a
dark-themed web UI backed by SQLite. See [README.md](README.md) for the full feature/env-var/API reference.

## Commands

```bash
# Setup
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt                  # adds pytest, ruff, coverage
cp .env.example .env                                 # fill in EBAY_*/GEMINI_API_KEY as needed

# Run
python app.py                                         # http://localhost:5000

# Test
pytest                                                 # full suite
pytest test_scraper.py                                 # single file
pytest test_scraper.py::test_name -v                    # single test
pytest -k "scam"                                        # by keyword, across files
python -m pytest -v --tb=short --cov --cov-report=term-missing   # what CI runs

# Lint / format (must pass — CI blocks on both)
ruff check .
ruff format --check .
ruff check --fix . && ruff format .                     # auto-fix locally
```

CI (`.github/workflows/docker-build.yml`) runs `lint → test → build-and-push` on every push/PR to `main`;
the Docker multi-arch build/push step only runs on non-PR events (i.e. after merge). Tests run with
`GEMINI_API_KEY`/`EBAY_CLIENT_ID`/`EBAY_CLIENT_SECRET` all empty, so tests must not require real credentials.

## Architecture

**Request flow for `/api/search` (the core of the app), in [app.py](app.py):**

1. Accept one `query` or a `queries[]` array; expand each with German synonym groups
   (`_QUERY_SYNONYMS` — e.g. "Sammlung"/"Konvolut"/"Paket"/"Bundle", console name variants) up to 8 total.
2. Pick a search engine via `_resolve_engine(data_source_setting)`: `"api"` → `EbayApiClient`,
   `"scraper"` → `EbayScraper`, `"auto"` → API if credentials are configured, else scraper.
3. Run every (query × source) combination — the chosen eBay engine, `EbayApiClient.search_auctions` (if
   API configured), and `KleinanzeigenScraper` (if importable) — **concurrently** via a thread pool
   (`_build_search_jobs`/`_run_search_jobs`), then merge results in a fixed, deterministic order
   (query-major, eBay → auctions → Kleinanzeigen), de-duplicating by normalized URL and by `title|price`
   (`_merge_deal`/`_merge_deals`). `EbayScraper` and `KleinanzeigenScraper` each gate their own outgoing
   requests behind an instance-level lock (`_rate_limit`) so concurrent callers still hit those sites with
   human-ish spacing instead of a simultaneous burst — eBay's anti-bot heuristic reacts to bursts, not
   steady volume; verified by hand (a naive unthrottled parallel version got most requests HTTP 403'd).
4. Apply post-filters in order: previously-skipped URLs (from SQLite) → Germany-only location filter
   (`_is_german_location`) → sports/Kinect filter (drops FIFA/Kinect/etc.-only listings unless the title
   also has ≥3 non-sports tokens, letting Gemini judge mixed bundles) → cap to 30 deals.
5. Send the surviving deals to Gemini in **one batched request** (`assessor.assess_deals_batch`) rather
   than per-deal calls, to conserve API quota — passed a `deadline` (request-start + `_SEARCH_DEADLINE_S`,
   env `SEARCH_DEADLINE_SECONDS`, default 75s) so a slow search phase leaves correspondingly less time for
   AI scoring instead of the two stacking into a request long enough for a reverse proxy in front (e.g. a
   Cloudflare Tunnel, whose proxied-HTTP default timeout is 100s) to kill the connection with its own HTML
   error page — degrading gracefully (fewer/no AI ratings) rather than losing the response entirely.
6. Sort: "Must Have"/"Must Buy" first, then everything else, each group newest → oldest by `listing_date`.
7. Persist the search + results via `database.save_search`.

**Data source abstraction** — `EbayScraper` (HTML scraping, [scraper.py](scraper.py)) and `EbayApiClient`
(OAuth2 Browse API, [ebay_api_client.py](ebay_api_client.py)) both expose
`search(query, max_results) -> (deals, errors)` and normalize to the *same* deal dict schema
(`title`, `price`, `condition`, `seller_rating`, `url`, `shipping`, `item_location`, `listing_date`,
`description`, `seller_count`, …) so the rest of the app — assessors, database, frontend — is agnostic to
which one produced a result. `KleinanzeigenScraper` ([kleinanzeigen_scraper.py](kleinanzeigen_scraper.py))
follows the same contract and is optional (imported defensively; `app.py` runs without it).

**AI assessment layer (`ai_providers/`)** — `create_assessor()` ([ai_providers/__init__.py](ai_providers/__init__.py))
is the sole entry point; it wires in the concrete provider (currently only `GeminiAssessor`,
[ai_providers/gemini.py](ai_providers/gemini.py)). All AI-agnostic logic lives in
`BaseAssessor` ([ai_providers/base.py](ai_providers/base.py)) — a new provider only needs to implement
`assess_deal`/`assess_deals_batch` and reuse everything else:
- **Deterministic overrides always win over the AI's own rating**, applied by `_finalize_assessment`
  in this order: garbage/trash keywords → sports/Kinect keywords → bait-and-switch scam detection
  (`_apply_garbage_overrides` → `_apply_sports_kinect_override` → `_apply_scam_override`). Each can force
  the rating to `"Garbage"` or `"Avoid"` regardless of what Gemini returned. Some of the same checks also
  run *before* calling the AI at all via `_try_deterministic_assessment`, to skip the API call entirely
  for obvious cases.
- Rating scale (from [prompts/system_prompt.txt](prompts/system_prompt.txt)): `Must Have` (profit ≥ cost,
  and *always* forced when a working item costs ≤ €2 total) → `Good` → `Okay` → `Avoid` → `Garbage`.
- Bundle titles (matched via `_BUNDLE_TITLE_KEYWORDS_RE`: Sammlung/Konvolut/Paket/Lot/Bundle/…) get their
  individual game titles extracted (`_extract_potential_game_titles`) and priced separately by querying
  `EbayApiClient.get_median_sold_price` per game (parallel prefetch with a time budget, then cached for
  5 minutes) so Gemini gets real per-game market prices instead of guessing.
- `assess_deals_batch(deals, deadline=...)` treats `deadline` as an absolute `time.monotonic()` cutoff
  (checked before each batch is submitted and before each retry/timeout) rather than measuring its own
  elapsed time from an independent start — this way whatever's left of the caller's overall time budget,
  not a fixed allowance stacked on top of it, bounds how much assessment happens. Omitting `deadline`
  falls back to the standalone `_ASSESS_TOTAL_BUDGET_S` default for callers outside a request/response
  cycle. Batches themselves run **concurrently** (bounded by `_BATCH_MAX_CONCURRENCY`, submissions still
  paced `_BATCH_DELAY_SECONDS` apart to respect the free-tier RPM limit) rather than one at a time, so
  total wall time is roughly "submission pacing + one call's duration" instead of the sum of every call's
  duration plus every stagger — the difference between all 30 deals getting an AI rating within a search's
  overall deadline versus only the first batch or two.
- Two prompt templates in `prompts/`: `system_prompt.txt` (single-deal) and `batch_system_prompt.txt`
  (batch — explicitly instructed to return **one entry per deal, no aggregation**, since a past bug had
  the model collapsing multiple listings into one summary).
- Response parsing (`_parse_response`/`_parse_batch_response`) is defensive: strips markdown code fences,
  retries with bracket-matching heuristics, and falls back to `_DEFAULT_PARSE_ERROR` per item rather than
  failing the whole batch.

**Persistence ([database.py](database.py))** — raw `sqlite3` (no ORM), WAL mode, one row per search and
one row per deal (FK to search), plus a generic `settings` key/value table for runtime toggles
(`ai_enabled`, `gemini_model`, `data_source`) that must survive across Gunicorn worker processes — hence
`app.py` re-reads `ai_enabled`/`data_source` from the DB on every request rather than trusting in-memory
state. Saved/skipped deals are separate tables keyed by URL; skipped URLs are filtered out of all future
search results.

**Frontend** — server-rendered [templates/index.html](templates/index.html) + vanilla JS
([static/app.js](static/app.js)) calling the JSON API; no build step or frontend framework.

## Conventions

- Deal dicts flow untyped through the whole pipeline (scraper/API → filters → assessor → DB → JSON
  response) — when adding a field, thread it through all four normalized producers (`EbayScraper`,
  `EbayApiClient`, `KleinanzeigenScraper`) or code that reads it must treat it as optional.
- Regex-driven German/English keyword lists (trash titles, broken/defective terms, sports franchises,
  scam phrasing, platform names) live at module level in `ai_providers/base.py`, each documented with a
  short rationale — extend the existing `_..._RE` pattern rather than adding ad-hoc string checks.
- `ruff` config in [pyproject.toml](pyproject.toml): line length 120, target py311, `select = ["E","F","I","W","UP","B","SIM"]`.
  Test files are exempt from `SIM115`/`SIM117`/`B`.
