# eBay Deal Finder

Find resale-worthy secondhand gaming deals on **eBay Germany** (`EBAY_DE`). Dual data sources (official Browse API + HTML scraper fallback), Gemini multimodal scoring, deterministic anti-scam/junk rules, SQLite persistence, and a dark-themed web UI.

## Features

- Web UI for search, filters, save/skip, history, and CSV export
- Official **eBay Browse API** with automatic HTML scraper fallback
- **Multi-provider AI** deal ratings: **Gemini** or **OpenCode Go** (Grok 4.5, DeepSeek, Kimi, …)
- Ratings: Must Have / Good / Okay / Avoid / Garbage
- Deterministic overrides for scams, sports/Kinect lots, broken/untested junk
- Per-game resale estimates from live eBay market data
- Runtime settings (provider, model, AI on/off, data source) **persisted in SQLite**
- Docker multi-arch images (`linux/amd64`, `linux/arm64`) for Portainer

---

## Requirements

- **Python 3.11+**
- Optional: eBay developer credentials (Browse API)
- Optional: `GEMINI_API_KEY` and/or `OPENCODE_GO_API_KEY` (AI assessment)

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
| GET/POST | `/api/settings` | AI provider, model, AI toggle, data source |
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

---

## Project structure

```
ebay-scrapper/
├── app.py                 # Flask app & REST API
├── database.py            # SQLite persistence
├── scraper.py             # Legacy HTML scraper (ebay.de)
├── ebay_api_client.py     # Browse API client (OAuth + search)
├── ai_providers/
│   ├── __init__.py        # Multi-provider factory (gemini | opencode-go)
│   ├── base.py            # Shared rules, JSON parse, price helpers
│   ├── gemini.py          # Google Gemini multimodal assessor
│   └── opencode_go.py     # OpenCode Go (Grok etc.) text assessor
├── prompts/               # System prompts for single + batch AI
├── templates/index.html
├── static/                # app.js + style.css
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
