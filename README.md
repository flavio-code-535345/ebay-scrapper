# eBay Scraper with GUI and Deal Assessment Engine

## Comprehensive Documentation

### Features
- Graphical User Interface (GUI)
- Automated scraping of eBay listings
- Deal assessment engine to identify the best deals
- Option to filter listings by various criteria
- User-friendly and intuitive design

### Installation Instructions
1. Clone the repository:
   ```
   git clone https://github.com/flavio-code-535345/ebay-scrapper.git
   cd ebay-scrapper
   ```
2. Ensure you have Python installed (version 3.6 or higher).
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   python gui.py
   ```

### API Endpoints
- **GET /api/scrape**: Initiates the scraping process.
- **GET /api/deals**: Retrieves the list of assessed deals.
- **POST /api/settings**: Updates user preferences for scraping.

### Project Structure
```
/ebay-scrapper
    ├── gui.py
    ├── scraper
    │   ├── __init__.py
    │   ├── ebay_scraper.py
    ├── api
    │   ├── __init__.py
    │   ├── api_routes.py
    ├── requirements.txt
    └── README.md
```

### Technology Stack
- **Python**: The primary programming language.
- **Flask**: For building the RESTful API.
- **Beautiful Soup**: For web scraping.
- **Tkinter**: For creating the GUI.

### Usage Examples
- Launch the application and navigate through the GUI to start scraping eBay listings.
- Use filters to refine your search for deals based on categories, prices, etc.
- View results and assess deals directly through the GUI interface.

## Conclusion
This project aims to simplify the process of finding the best deals on eBay using a user-friendly interface and robust back-end scraping capabilities.

---

## Docker Hub & Automated Builds

The Docker image is automatically built and published to Docker Hub via GitHub Actions on every push to `main`. Portainer (or any Docker host) can pull the image directly — no local build required.

**Docker Hub image:** `flavio-code-535345/ebay-scrapper:latest`

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
2. Create a **public** repository named `ebay-scrapper` under your account (`flavio-code-535345/ebay-scrapper`).

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
4. Click **Deploy the stack**.
5. The container starts automatically, pulling the latest image from Docker Hub.

### Option 2 — Portainer Git repository
1. Go to **Stacks → + Add Stack → Git repository**.
2. Repository URL: `https://github.com/flavio-code-535345/ebay-scrapper`
3. Compose file path: `docker-compose.yml`
4. Enable **Automatic updates** if you want Portainer to redeploy on new commits.
5. Click **Deploy the stack**.

### Option 3 — Docker Compose (command line)
```bash
curl -O https://raw.githubusercontent.com/flavio-code-535345/ebay-scrapper/main/docker-compose.yml
docker compose up -d
# Access the app at http://localhost:5000
```

### docker-compose.yml (for reference)
```yaml
version: "3.8"

services:
  ebay-scrapper:
    image: flavio-code-535345/ebay-scrapper:latest
    container_name: ebay-scrapper
    restart: unless-stopped
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - FLASK_APP=app.py
      - DB_PATH=/data/ebay_deals.db
    volumes:
      - ebay_db:/data
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s

volumes:
  ebay_db:
    driver: local
```

The SQLite database is stored in the `ebay_db` named volume so it persists across container restarts and image updates.
