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
