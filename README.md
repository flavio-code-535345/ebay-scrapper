# eBay Scraper with GUI and Deal Assessment Engine

## Comprehensive Documentation

### Features
- Graphical User Interface (GUI)
- Automated scraping of eBay listings **and** native eBay Browse API integration
- Deal assessment engine to identify the best deals
- Gemini AI scoring with an on/off toggle
- Option to filter listings by various criteria
- User-friendly and intuitive design

---

## eBay Official API Integration

The app supports the **eBay Browse API** (part of the [eBay Developer Program](https://developer.ebay.com/)) as a first-class data source alongside the legacy HTML scraper.

### Why use the official API?
| Feature | Official API | HTML Scraper |
|---------|-------------|--------------|
| Reliability | ✅ Stable structured data | ⚠️ Breaks on eBay markup changes |
| Speed | ✅ Faster, no HTML parsing | Slower |
| Extra metadata | ✅ (seller score, condition ID, images) | Limited |
| TOS compliant | ✅ Yes | ⚠️ Restricted |
| Requires credentials | Yes (free dev account) | No |

### Getting eBay API credentials

1. Sign up (free) at <https://developer.ebay.com/>.
2. Create a new **application** in the developer portal.
3. Under **Credentials**, locate your production **App ID (Client ID)** and **Cert ID (Client Secret)**.
4. Ensure your application has the **Browse API** in its OAuth scope list.

### Configuring credentials

Copy `.env.example` to `.env` and fill in the eBay section:

```env
EBAY_CLIENT_ID=your-app-id-here
EBAY_CLIENT_SECRET=your-cert-id-here
EBAY_MARKETPLACE_ID=EBAY_DE   # or EBAY_US, EBAY_GB, EBAY_FR, etc.
EBAY_ENVIRONMENT=production   # or sandbox for testing
DATA_SOURCE=auto              # auto | api | scraper
```

> **Never commit `.env`** — it is already listed in `.gitignore`.

### `DATA_SOURCE` modes

| Value | Behaviour |
|-------|-----------|
| `auto` | **(default)** Use the official API when credentials are set; fall back to the HTML scraper otherwise. |
| `api` | Always use the official eBay Browse API. Returns an error if credentials are missing. |
| `scraper` | Always use legacy HTML scraping, even if API credentials are present. |

The setting can also be changed **at runtime** via the settings panel in the UI or the `POST /api/settings` endpoint:

```bash
curl -X POST http://localhost:5000/api/settings \
  -H 'Content-Type: application/json' \
  -d '{"data_source": "api"}'
```

The active value is persisted in the SQLite database and survives restarts.

### Docker / Portainer environment variables

Add the following to your `docker-compose.yml` environment section (or Portainer stack variables):

```yaml
- EBAY_CLIENT_ID=${EBAY_CLIENT_ID:-}
- EBAY_CLIENT_SECRET=${EBAY_CLIENT_SECRET:-}
- EBAY_MARKETPLACE_ID=${EBAY_MARKETPLACE_ID:-EBAY_DE}
- EBAY_ENVIRONMENT=${EBAY_ENVIRONMENT:-production}
- DATA_SOURCE=${DATA_SOURCE:-auto}
```

---

### Installation Instructions
1. Clone the repository:
   ```
   git clone https://github.com/flavio-code-535345/ebay-scrapper.git
   cd ebay-scrapper
   ```
2. Ensure you have Python installed (version 3.8 or higher).
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your credentials.
5. Run the application:
   ```
   python app.py
   ```

### API Endpoints
- **POST /api/search**: Search for eBay deals. Body: `{"query": "...", "max_results": 50}`
- **GET /api/health**: Health check; includes `data_source`, `ebay_api_configured`, and AI status.
- **GET /api/settings**: Returns current settings.
- **POST /api/settings**: Updates settings. Body fields: `gemini_model`, `ai_enabled`, `data_source`.
- **GET /api/history**: Recent search history.
- **GET /api/export**: Export all deals as CSV.
- **GET /api/stats**: Database statistics.

### AI Assessment — Per-Game Price Breakdown

When Gemini AI is enabled and the eBay API is configured, the assessment pipeline automatically looks up **real eBay prices for each individual game** identified in a bundle listing:

1. **Title parsing** — The listing title is scanned for comma/plus-separated game names (e.g. `"PS4 Bundle: God of War, Spider-Man, Horizon"`).
2. **eBay price lookup** — Each identified game is searched on eBay:
   - **Primary**: eBay Marketplace Insights API (recently *sold* listings — most accurate).
   - **Fallback**: eBay Browse API (current *active* listings — proxy for market value).
   - If neither returns data the game is marked as `"no_result"` and Gemini estimates the price.
3. **AI analysis** — The fetched prices are injected into the Gemini prompt so the model uses real market data instead of guesswork.

The deal response includes three new AI fields:

| Field | Type | Description |
|-------|------|-------------|
| `ai_itemized_resale_estimates` | list | Per-game breakdown: `game`, `price_eur`, `price_source` |
| `ai_estimated_total_cost` | float | Asking price + shipping |
| `ai_estimated_gross_profit` | float | Estimated resale total − total cost |

`price_source` values:
- `"ebay_sold"` — price from eBay sold/completed listings via Marketplace Insights API
- `"ebay_active"` — price from current eBay active listings (Browse API fallback)
- `"ai_estimate"` — AI estimate (no eBay data available)
- `"no_result"` — no eBay data found; AI estimate used in verdict

### Project Structure
```
/ebay-scrapper
    ├── app.py               Flask application & API routes
    ├── scraper.py           Legacy HTML scraper (eBay.de)
    ├── ebay_api_client.py   eBay Browse API client (OAuth + search)
    ├── deal_assessor.py     Rules-based deal scoring
    ├── gemini_assessor.py   Gemini AI assessment
    ├── database.py          SQLite persistence
    ├── templates/
    │   └── index.html
    ├── static/
    │   ├── app.js
    │   └── style.css
    ├── .env.example         Environment variable template
    ├── Dockerfile
    ├── docker-compose.yml
    └── requirements.txt
```

### Technology Stack
- **Python 3.8+**: Primary language
- **Flask**: REST API framework
- **requests**: HTTP client (used by both scraper and API client)
- **Beautiful Soup**: HTML parsing (legacy scraper)
- **google-genai**: Gemini AI SDK
- **SQLite**: Persistent storage

---

## Docker Hub & Automated Builds

The Docker image is automatically built and published to Docker Hub via GitHub Actions on every push to `main`. Portainer (or any Docker host) can pull the image directly — no local build required.

**Docker Hub image:** `flavio11113/ebay-scrapper:latest`

### How the CI/CD pipeline works

```
Push to main  →  GitHub Actions builds image  →  Pushes to Docker Hub  →  Portainer pulls & runs
```

The workflow (`.github/workflows/docker-build.yml`) supports:
- Multi-platform builds (`linux/amd64`, `linux/arm64`)
- Automatic `latest` tag on `main` branch pushes
- Semantic version tags from git tags (e.g. `v1.2.3` → `1.2.3` and `1.2`)
- GitHub Actions layer caching for faster builds
- Manual trigger via `workflow_dispatch`

---

## Docker Hub Setup

### 1. Create a Docker Hub account
1. Go to <https://hub.docker.com/> and sign up (or log in).
2. Create a **public** repository named `ebay-scrapper` under your account (`flavio11113/ebay-scrapper`).

### 2. Generate a Docker Hub access token
1. In Docker Hub, go to **Account Settings → Security → New Access Token**.
2. Give it a descriptive name (e.g. `github-actions`) and set permission to **Read, Write, Delete**.
3. Copy the generated token — you will not be able to see it again.

---

## GitHub Secrets Configuration

Add the following secrets to your GitHub repository (**Settings → Secrets and variables → Actions → New repository secret**):

| Secret name       | Value                              |
|-------------------|------------------------------------|
| `DOCKER_USERNAME` | Your Docker Hub username           |
| `DOCKER_PASSWORD` | The access token you generated above |

Once set, every push to `main` will automatically build and push a fresh image to Docker Hub.

---

## Portainer Deployment

### Option 1 — Portainer Stacks (recommended)
1. Open Portainer (typically `http://<your-host>:9000`).
2. Go to **Stacks → + Add Stack**.
3. Paste the contents of `docker-compose.yml` from this repository.
4. Set any environment variables (e.g. `EBAY_CLIENT_ID`, `GEMINI_API_KEY`).
5. Click **Deploy the stack**.
6. The container starts automatically, pulling the latest image from Docker Hub.

### Option 2 — Portainer Git repository
1. Go to **Stacks → + Add Stack → Git repository**.
2. Repository URL: `https://github.com/flavio-code-535345/ebay-scrapper`
3. Compose file path: `docker-compose.yml`
4. Enable **Automatic updates** if you want Portainer to redeploy on new commits.
5. Click **Deploy the stack**.

### Option 3 — Docker Compose (command line)
```bash
curl -O https://raw.githubusercontent.com/flavio-code-535345/ebay-scrapper/main/docker-compose.yml
# Create a .env file with your API keys
docker compose up -d
# Access the app at http://localhost:5000
```

The SQLite database is stored in the `ebay_db` named volume so it persists across container restarts and image updates.
