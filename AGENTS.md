# AGENTS.md

## Commands

- Install deps: `pip install -r requirements-dev.txt` (runtime deps in `requirements.txt`)
- Run tests: `pytest` (repo root) · single test: `pytest test_app.py::TestHealth::test_health_returns_healthy`
- Lint + format (CI order): `ruff check .` then `ruff format --check .` — run both after any Python change
- Run app: `python app.py` (port 5000) · Docker: `docker compose up -d`
- Python 3.11 only (pinned in pyproject, Dockerfile, CI)

## Architecture

- Flask monolith: `app.py` holds all REST routes. Module-level singletons (`assessor`, `scraper`, `ebay_api`, `kleinanzeigen`) and `database.init_db()` run at **import time**.
- Data engines: `ebay_api_client.py` (official Browse API) → `scraper.py` (HTML fallback) + `kleinanzeigen_scraper.py` (separate source). Selected via `DATA_SOURCE` env or persisted DB setting (`auto`/`api`/`scraper`).
- AI: `ai_providers/` — `create_assessor()` returns `GeminiAssessor`; `base.py` has shared rules/parsing. Prompts load from `prompts/*.txt` at import. Settings (`ai_enabled`, `gemini_model`, `data_source`) persist in SQLite and are re-read per request (multi-worker gunicorn safety).
- Frontend: `templates/index.html` + `static/app.js` + `static/style.css`. Static assets are cache-busted with `?v=N` query strings — bump the version in `index.html` whenever you change `app.js`/`style.css`.
- `APP_VERSION` in `app.py` and `version` in `pyproject.toml` must change together; the UI reads the version via `/api/settings`.

## Search pipeline gotchas

- Queries auto-expand through German synonym groups (`_expand_queries`, capped at 8); results dedupe by URL and title+price.
- Server-side hard filters before AI: Germany-only locations, sports/Kinect lots (bundle-token heuristic), previously skipped URLs. Only `_MAX_DISPLAY` (30) deals reach AI assessment.
- `/api/search` returns an `insights` block (totals, Must Have/Good counts, est. profit) rendered by the UI insights strip — keep it populated when changing the pipeline.

## Testing quirks

- No `conftest.py`. `test_app.py` sets `GEMINI_API_KEY`/`EBAY_CLIENT_ID`/`EBAY_CLIENT_SECRET` to `""` at module top **before** `import app` — preserve that ordering.
- Fixtures swap `database.DB_PATH` to a tmp dir; no network needed (requests mocked via `requests-mock`/`responses`).
- Gunicorn worker timeout is 180s (Dockerfile). `test_gemini_assessor.py` asserts the AI prefetch + batch budget stays under it — don't raise AI timeouts without revisiting that test.

## Gotchas

- Files contain cosmetic mojibake (`�?` where em-dashes should be) — pre-existing; don't mass-fix unless asked.
- Never commit `.env`; `ebay_deals.db` is local state. CI (lint → test → multi-arch Docker push) triggers on push to `main`; `v*` tags get semver image tags.
