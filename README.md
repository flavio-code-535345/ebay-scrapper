# eBay Deal Finder

Find resale-worthy secondhand gaming deals on **eBay Germany** (`EBAY_DE`), also cross-searching
**Kleinanzeigen.de**. Dual eBay data sources (official Browse API + HTML scraper fallback, both
fixed-price and auction listings), Gemini multimodal scoring, deterministic anti-scam/junk rules,
SQLite persistence, and a dark-themed web UI.

## Features

- Web UI for search, filters, save/skip, history, and CSV export
- Official **eBay Browse API** with automatic HTML scraper fallback — fixed-price ("Buy It Now") and
  auction listings, searched in parallel; only auctions ending within the next 2 days are shown
- Also searches **Kleinanzeigen.de** for matching classified listings
- **Gemini AI** deal ratings: Must Have / Good / Okay / Avoid / Garbage
- Deterministic overrides for scams, sports/Kinect lots, broken/untested junk
- Reads what a listing says about its price: a per-game price ("Stückpreis 7 €") on a bundle is flagged, or
  replaced by the whole-lot price the description states ("komplett Paket 120 €"); "1 € VB" placeholders and single
  games dressed up as bundles are recognised too
- Per-game resale estimates from live eBay market data
- **Sell page** (`/sell`): list a game on your own eBay account from your phone — scan the barcode, take photos,
  publish; price suggested from the cheapest comparable offer, settings copied from one of your listings
- Runtime settings (AI on/off, model, data source) persisted in SQLite
- Docker multi-arch images (`linux/amd64`, `linux/arm64`) for Portainer

---

## Requirements

- **Python 3.11+**
- Optional: eBay developer credentials (Browse API)
- Optional: Gemini API key (AI assessment)

---

## eBay Official API

| Feature | Official API | HTML Scraper |
|---------|-------------|--------------|
| Reliability | Stable structured data | Breaks on markup changes |
| Speed | Faster | Slower |
| Extra metadata | Seller score, condition, images | Limited |
| TOS compliant | Yes | Restricted |
| Requires credentials | Free dev account | No |

### Credentials

1. Sign up at <https://developer.ebay.com/>
2. Create an application and copy **App ID** + **Cert ID**
3. Ensure **Browse API** is in the OAuth scope list

```env
EBAY_CLIENT_ID=your-app-id-here
EBAY_CLIENT_SECRET=your-cert-id-here
EBAY_MARKETPLACE_ID=EBAY_DE
EBAY_ENVIRONMENT=production
DATA_SOURCE=auto
```

### `DATA_SOURCE` modes

| Value | Behaviour |
|-------|-----------|
| `auto` | Use API when credentials are set; otherwise HTML scraper |
| `api` | Always Browse API (falls back to scraper if creds missing) |
| `scraper` | Always HTML scraping |

Change at runtime via the UI or:

```bash
curl -X POST http://localhost:5000/api/settings \
  -H 'Content-Type: application/json' \
  -d '{"data_source": "api"}'
```

---

## Selling: list a game in under a minute (`/sell`)

A phone-friendly page for listing a game on your own eBay.de account: scan the barcode → eBay's catalog fills in the
title and item specifics → the price is suggested from the cheapest comparable Buy-It-Now offer (item + shipping,
your own listings excluded) → take photos → **Publish**. Shipping, returns, location, business policies and the
description are copied from one of your existing listings, so every new listing looks like the ones you made by hand.

Listings are created with eBay's Trading API, so they are ordinary listings you can still edit, discount and accept
offers on in the eBay app. ("Check with eBay" validates a listing without creating it.)

### One-time setup

1. **RuName** — at <https://developer.ebay.com/> → *Application Keys* → *User Tokens* (production keyset) →
   *Get a Token from eBay via Your Application* → *Add eBay Redirect URL*. Tick *OAuth Enabled* and set
   *Your auth accepted URL* to `https://<your app's domain>/sell/ebay/callback`. Copy the RuName it shows
   (looks like `Your_Name-YourApp-PRD-abc123-def456`).
2. **Environment** (next to the existing `EBAY_CLIENT_ID` / `EBAY_CLIENT_SECRET`):

   ```env
   APP_PASSWORD=choose-a-long-password   # the Sell page only opens with it
   EBAY_RUNAME=Your_Name-YourApp-PRD-abc123-def456
   # SECRET_KEY=...                      # optional; otherwise generated once and kept in the database
   ```
3. Open `/sell`, sign in with `APP_PASSWORD`, press **Connect eBay** and agree on eBay's own page (you sign in to eBay
   there — this app never sees your eBay password). If eBay can't send you back to the app, paste the address you
   landed on into the box under the button.
4. Paste one of your listings (URL or item number) under **Listing settings** — its setup becomes the template.

The connection lasts about 18 months; **Disconnect** or changing `APP_PASSWORD` ends it early.

---

## Local install

```bash
git clone https://github.com/flavio-code-535345/ebay-scrapper.git
cd ebay-scrapper
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in keys
python app.py
```

Open <http://localhost:5000>.

Dev tooling:

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
ruff format --check .
```

---

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Web UI |
| POST | `/api/search` | Search + assess deals |
| GET | `/api/health` | Health + AI/API status |
| GET/POST | `/api/settings` | Model, AI toggle, data source |
| GET | `/api/history` | Search history |
| GET | `/api/deals/<id>` | Deals for a search |
| GET | `/api/export` | CSV export |
| GET | `/api/stats` | Database stats |
| POST | `/api/deals/save` | Favourite a deal |
| POST | `/api/deals/unsave` | Remove favourite |
| GET | `/api/deals/saved` | List saved deals |
| POST | `/api/deals/skip` | Hide deal forever |
| POST | `/api/deals/unskip` | Restore skipped deal |
| GET | `/api/deals/skipped` | List skipped deals |
| GET | `/sell` | Sell page (password: `APP_PASSWORD`) |
| GET | `/api/sell/status` | Connection, template, recent listings |
| POST | `/api/sell/template` | Copy settings from one of your listings |
| GET | `/api/sell/product?q=` | Catalog product + price suggestion for an EAN/title |
| POST | `/api/sell/photos` | Upload a photo to eBay |
| POST | `/api/sell/publish` | Check (`verify_only`) or publish a listing |

---

## Project structure

```
ebay-scrapper/
├── app.py                 # Flask app & REST API
├── database.py            # SQLite persistence
├── models.py              # Shared Deal schema, condition/date normalization, sort key
├── scraper.py             # Legacy HTML scraper (ebay.de)
├── ebay_api_client.py     # Browse API client (OAuth + search + auctions + price lookup)
├── ebay_seller.py         # Your eBay account: OAuth connection, listing template, photos, publishing
├── sell.py                # Sell page routes (password-protected)
├── kleinanzeigen_scraper.py # Kleinanzeigen.de HTML scraper
├── search/
│   ├── query.py           # Query planner: one OR-grouped request per source
│   └── pipeline.py        # Concurrent fetch, dedupe, filters, ranked selection
├── fixtures/              # Real captured result pages the parser tests run on
├── ai_providers/
│   ├── __init__.py        # create_assessor() factory
│   ├── base.py            # Shared rules, JSON parse
│   ├── enrichment.py      # Phase A: concurrent, deadline-bounded price/image fetching
│   └── gemini.py          # Google Gemini multimodal assessor (Phase B/C)
├── prompts/               # System prompts for single + batch AI
├── templates/             # index.html + sell.html
├── static/                # app.js, style.css, sell.js, sell.css
├── test_*.py              # pytest suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Stack

- Python 3.11 · Flask 3 · gunicorn · requests · BeautifulSoup4
- google-genai (Gemini) · SQLite · vanilla JS frontend

---

## Docker / Portainer

**Image:** `flavio11113/ebay-scrapper:latest`

```bash
# Pull and run via Compose
curl -O https://raw.githubusercontent.com/flavio-code-535345/ebay-scrapper/main/docker-compose.yml
# Create a .env with GEMINI_API_KEY / EBAY_* as needed
docker compose up -d
```

SQLite data lives in the `ebay_db` named volume.

If the app sits behind a reverse proxy or tunnel with its own request timeout (e.g. a Cloudflare
Tunnel, whose proxied-HTTP default is ~100s), set `SEARCH_DEADLINE_SECONDS` (default `75`) to the
total time budget `/api/search` should stay within — see `.env.example` for the full list of
environment variables.

### CI/CD

Push to `main` → GitHub Actions (lint → test → multi-arch push to Docker Hub).

| Secret | Value |
|--------|-------|
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub access token |

### Portainer

1. **Stacks → Add Stack** and paste `docker-compose.yml`, or
2. **Git repository** → `https://github.com/flavio-code-535345/ebay-scrapper` with compose path `docker-compose.yml`
3. Set `GEMINI_API_KEY`, `EBAY_CLIENT_ID`, `EBAY_CLIENT_SECRET` as needed
4. Deploy — app listens on port **5000**
