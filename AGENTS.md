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
pytest -m live                                         # opt-in: hit the real eBay/Kleinanzeigen sites

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

1. Accept one `query` or a `queries[]` array (quick-search chips send several phrasings) and turn it into
   a `SearchPlan` ([search/query.py](search/query.py)): eBay's web search *and* Browse API both honor OR
   groups, so every German bundle synonym fits in **one** request — `xbox 360 (sammlung,konvolut,paket,
   bundle,spielesammlung,lot,spielepaket)`, trimmed to the API's 100-character `q` cap, with no `-word`
   terms for the API (undocumented there); the web query adds a few exclusions (`-skylanders -amiibo …`).
   Kleinanzeigen has no OR support and IP-bans quickly, so it gets at most two plain variants.
2. Pick a search engine via `_resolve_engine(data_source_setting)`: `"api"` → `EbayApiClient`,
   `"scraper"` → `EbayScraper`, `"auto"` → API if credentials are configured, else scraper.
3. `search.pipeline.run_search` ([search/pipeline.py](search/pipeline.py)) fetches every source
   **concurrently** under a search-phase budget (`min(_SEARCH_PHASE_MAX_S, 45% of the deadline)`): a
   source that overruns it is reported (`timed_out`) and dropped rather than stalling the response. It then
   de-duplicates by stable listing ID (`models.canonical_listing_id` — the same eBay item arrives under
   different URL shapes from the API and the web) and by normalized title+price for cross-posted listings.
   `EbayScraper` and `KleinanzeigenScraper` each gate their own outgoing requests behind an instance-level
   lock (`_rate_limit`) so concurrent callers never burst a site.
4. Filters: previously-skipped listings (by URL *or* listing ID) → Germany-only location → **auctions that
   don't end within 2 days** (`AUCTION_MAX_TIME_LEFT`; an auction with no known end time is dropped too — each
   engine's `search_auctions` already asks only for auctions ending by then, this re-checks every auction
   whichever leg it came from) → sports/Kinect-only
   titles (mixed bundles with ≥3 other meaningful words are kept for Gemini to judge) → **platform guard**
   (a title naming only a different platform than the one searched is dropped; titles naming no platform, or
   the generic "Xbox"/"PlayStation", are kept). Then **ranked, source-diverse selection** of the 30 deals that
   get AI budget: score = query match + freshness + price per game (from counts like "26 Spiele"), and each
   source bucket (eBay Buy-It-Now, eBay auctions, Kleinanzeigen) first gets an equal share of its own best
   listings — so one prolific source can't crowd the others out. The response's `sources` array reports each
   source's queries, count, errors and time, so a dead source is visible instead of silent.
5. Send the surviving deals to Gemini in **one batched request** (`assessor.assess_deals_batch`) rather
   than per-deal calls, to conserve API quota — passed a `deadline` (request-start + `_SEARCH_DEADLINE_S`,
   env `SEARCH_DEADLINE_SECONDS`, default 75s) so a slow search phase leaves correspondingly less time for
   AI scoring instead of the two stacking into a request long enough for a reverse proxy in front (e.g. a
   Cloudflare Tunnel, whose proxied-HTTP default timeout is 100s) to kill the connection with its own HTML
   error page — degrading gracefully (fewer/no AI ratings) rather than losing the response entirely.
6. Sort: "Must Have"/"Must Buy" first, then everything else, each group newest → oldest by `listing_date`.
7. Persist the search + results via `database.save_search`.

**Data source abstraction** — `EbayScraper` (HTML scraping, [scraper.py](scraper.py)), `EbayApiClient`
(OAuth2 Browse API, [ebay_api_client.py](ebay_api_client.py)), and `KleinanzeigenScraper`
([kleinanzeigen_scraper.py](kleinanzeigen_scraper.py), optional — imported defensively; `app.py` runs
without it) all expose `search(query, max_results) -> (deals, errors)` and normalize to the same deal dict
schema documented in [models.py](models.py) (`Deal` — `title`, `price`, `condition`,
`condition_normalized`, `seller_rating`, `url`, `shipping`, `shipping_note`, `item_location`,
`listing_date`, `description`, `seller_count`, …) so the rest of the app — assessors, database, frontend —
is agnostic to which one produced a result. Each producer stamps its own `source` and `listing_id`.
`condition_normalized` (`models.normalize_condition`) gives a cross-source-comparable value without touching
the raw `condition` text each source still populates as-is; `listing_date` is always a real ISO-8601 string
or `None` (`models.parse_listing_date`: German local time for Kleinanzeigen, eBay's coarse "Vor 5 Std.
eingestellt" for the scraper — never fabricated), which is what lets `models.sort_key_for_deal` sort dated
deals newest-first ahead of undated ones within each rating tier.

Source specifics worth knowing: `EbayScraper` parses eBay's `ul.srp-results > li.s-card` markup
(`_sop=10` is newest-first, `LH_PrefLoc=1` is Germany). Like the API client it runs two legs: `search`
(Buy It Now, `LH_BIN=1`, newest first) and `search_auctions` (`LH_Auction=1`, `_sop=1` ending soonest —
the only sort order whose cards print the time left, "Noch 1 T 1 Std", which becomes `auction_end`; the
bracketed end clock beside it is rendered in a US time zone and ignored). eBay's bot protection often answers HTTP 403
while setting session cookies and serves the request once they come back, so the scraper retries a 403
exactly once on its cookie-keeping session; if it's still refused, it says so and points at the Browse API,
which is the reliable path. `KleinanzeigenScraper` parses the structured `resultAds[]` data each results page embeds
(an Astro island), with the `article[data-adid]` cards as fallback; it searches the Videospiele category
(`c227`), decodes UTF-8 explicitly (the server sends no charset), and pauses itself for 10 minutes after an
IP-ban 403. Parser tests run against **real captured pages** in `fixtures/` — hand-written HTML is what let
both scrapers break silently before; recapture a fixture when a site's markup changes.

**AI assessment layer (`ai_providers/`)** — `create_assessor()` ([ai_providers/__init__.py](ai_providers/__init__.py))
is the sole entry point; it wires in the concrete provider (currently only `GeminiAssessor`,
[ai_providers/gemini.py](ai_providers/gemini.py)). All AI-agnostic logic lives in
`BaseAssessor` ([ai_providers/base.py](ai_providers/base.py)) — a new provider only needs to implement
`assess_deal`/`assess_deals_batch` and reuse everything else. Assessment runs in three explicit phases so
nothing network-bound is ever unbounded by the caller's `deadline`:
- **Phase A — enrichment** ([ai_providers/enrichment.py](ai_providers/enrichment.py)'s `Enricher.enrich_deals`):
  one concurrent, deadline-bounded pass fetching every eBay price lookup (bundle titles get one lookup per
  extracted game via `EbayApiClient.get_lowest_market_price`, cached 5 minutes) and every image for the
  *whole* filtered deal set at once — not per Gemini sub-batch. Whatever isn't done when its own budget runs
  out (capped at `_ENRICH_MAX_BUDGET_S`, itself never more than half of whatever deadline remains) is simply
  omitted; nothing here ever blocks past it.
- **Phase B — content-building** (`GeminiAssessor._build_contents`/`_build_batch_contents`): pure formatting
  over already-enriched data, zero network calls — this is what makes Phase C's deadline actually sufficient.
  A past bug had bundle-price lookups running live and unbounded *inside* this phase (up to 8 sequential
  calls per bundle listing), so a slow/uncached lookup could alone exceed a whole search's deadline before
  Gemini was ever called — the actual root cause of "some listings never get an AI rating."
- **Phase C — the API call** (`_assess_batch_with_retry`): a real `future.result(timeout=...)` bound to
  `min(_GEMINI_REQUEST_TIMEOUT, deadline - now)`.
- **Deterministic overrides always win over the AI's own rating**, applied by `_finalize_assessment`
  in this order: garbage/trash keywords → sports/Kinect keywords → bait-and-switch scam detection
  (`_apply_garbage_overrides` → `_apply_sports_kinect_override` → `_apply_scam_override`). Each can force
  the rating to `"Garbage"` or `"Avoid"` regardless of what Gemini returned. The same checks also run
  *before* calling the AI at all via `_try_deterministic_assessment` (one shared implementation used by
  both the single-deal and batch paths), to skip the API call entirely for obvious cases.
- Rating scale (from [prompts/system_prompt.txt](prompts/system_prompt.txt)): `Must Have` (profit ≥ cost,
  and *always* forced when a working item costs ≤ €2 total) → `Good` → `Okay` → `Avoid` → `Garbage`.
- Bundle titles (matched via `_BUNDLE_TITLE_KEYWORDS_RE`: Sammlung/Konvolut/Paket/Lot/Bundle/…) get their
  individual game titles extracted (`_extract_potential_game_titles`) for Phase A's per-game price lookups.
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

**Persistence ([database.py](database.py))** — raw `sqlite3` (no ORM), WAL mode, indexed on
`deals.search_id` and `deals.created_at`, one row per search and one row per deal (FK to search), plus a
generic `settings` key/value table for runtime toggles
(`ai_enabled`, `gemini_model`, `data_source`) that must survive across Gunicorn worker processes — hence
`app.py` re-reads `ai_enabled`/`data_source` from the DB on every request rather than trusting in-memory
state. Saved/skipped deals are separate tables keyed by URL; skipped URLs are filtered out of all future
search results.

**Frontend** — server-rendered [templates/index.html](templates/index.html) + vanilla JS
([static/app.js](static/app.js)) calling the JSON API; no build step or frontend framework.

## Conventions

- Deal dicts flow untyped through the whole pipeline (scraper/API → filters → assessor → DB → JSON
  response) — the canonical field reference is [models.py](models.py)'s `Deal` TypedDict; when adding a
  field, thread it through all three deal-producing sources (`EbayScraper`, `EbayApiClient`,
  `KleinanzeigenScraper`) or code that reads it must treat it as optional.
- Regex-driven German/English keyword lists (trash titles, broken/defective terms, sports franchises,
  scam phrasing, platform names) live at module level in `ai_providers/base.py`, each documented with a
  short rationale — extend the existing `_..._RE` pattern rather than adding ad-hoc string checks.
- `ruff` config in [pyproject.toml](pyproject.toml): line length 120, target py311, `select = ["E","F","I","W","UP","B","SIM"]`.
  Test files are exempt from `SIM115`/`SIM117`/`B`.
