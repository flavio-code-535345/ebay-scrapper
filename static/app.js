// eBay Deal Finder - Frontend Application

document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('searchForm');
    if (searchForm) {
        searchForm.addEventListener('submit', handleSearch);
    }

    // Check URL parameters for auto-search
    const params = new URLSearchParams(window.location.search);
    const searchParam = params.get('search');
    if (searchParam) {
        document.getElementById('searchQuery').value = searchParam;
        handleSearch(new Event('submit'));
    }
});

async function handleSearch(e) {
    e.preventDefault();

    const query = document.getElementById('searchQuery').value.trim();
    const searchBtn = document.getElementById('searchBtn');
    const spinner = searchBtn.querySelector('.spinner-border');
    const btnText = searchBtn.querySelector('.btn-text');

    if (!query) {
        showError('Please enter a search term');
        return;
    }

    // Show loading state
    searchBtn.disabled = true;
    spinner.classList.remove('d-none');
    btnText.textContent = 'Searching...';
    document.getElementById('loadingContainer').classList.remove('d-none');
    document.getElementById('resultsContainer').classList.add('d-none');
    document.getElementById('errorContainer').classList.add('d-none');
    document.getElementById('emptyState').classList.add('d-none');

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                max_results: 50
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Search failed');
        }

        // Hide loading
        document.getElementById('loadingContainer').classList.add('d-none');

        if (!data.deals || data.deals.length === 0) {
            showError('No deals found. Try a different search term.');
            document.getElementById('emptyState').classList.remove('d-none');
            return;
        }

        displayResults(data);

    } catch (error) {
        console.error('Search error:', error);
        document.getElementById('loadingContainer').classList.add('d-none');
        showError(error.message || 'An error occurred during search');
    } finally {
        searchBtn.disabled = false;
        spinner.classList.add('d-none');
        btnText.textContent = 'Search';
    }
}

function displayResults(data) {
    document.getElementById('dealCount').textContent = data.deal_count;
    
    const dealsGrid = document.getElementById('dealsGrid');
    dealsGrid.innerHTML = data.deals.map(deal => createDealCard(deal)).join('');
    
    document.getElementById('resultsContainer').classList.remove('d-none');

    // Setup export button
    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) {
        exportBtn.onclick = () => exportToCSV(data.query);
    }
}

function createDealCard(deal) {
    const score = deal.overall_score || 0;
    const scoreColor = getScoreColor(score);
    const recommendation = deal.recommendation || 'N/A';
    
    const priceScore = deal.price_score || 0;
    const sellerScore = deal.seller_score || 0;
    const conditionScore = deal.condition_score || 0;
    const trendScore = deal.trend_score || 0;

    return `
        <div class="deal-card">
            <div class="deal-header">
                <div class="deal-score" style="color: ${scoreColor}">
                    ${score.toFixed(1)}
                </div>
                <div class="deal-recommendation">${recommendation}</div>
            </div>
            
            <div class="deal-body">
                <div class="deal-title">${escapeHtml(deal.title)}</div>
                
                <div class="deal-price">
                    $${(deal.price || 0).toFixed(2)}
                </div>
                
                <div class="deal-details">
                    <div class="detail-item">
                        <span class="detail-label">Condition</span>
                        <span class="detail-value">${escapeHtml(deal.condition || 'Unknown')}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Seller Rating</span>
                        <span class="detail-value">${(deal.seller_rating || 0).toFixed(1)}%</span>
                    </div>
                </div>

                <div class="deal-details">
                    <div class="detail-item">
                        <span class="detail-label">Shipping</span>
                        <span class="detail-value">${escapeHtml(deal.shipping || 'TBD')}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Trending</span>
                        <span class="detail-value">${deal.is_trending ? '🔥 Yes' : 'No'}</span>
                    </div>
                </div>

                <div class="scores-breakdown">
                    <div class="score-row">
                        <span>Price</span>
                        <span>${priceScore.toFixed(0)}</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${priceScore}%"></div>
                    </div>

                    <div class="score-row">
                        <span>Seller</span>
                        <span>${sellerScore.toFixed(0)}</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${sellerScore}%"></div>
                    </div>

                    <div class="score-row">
                        <span>Condition</span>
                        <span>${conditionScore.toFixed(0)}</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${conditionScore}%"></div>
                    </div>

                    <div class="score-row">
                        <span>Trend</span>
                        <span>${trendScore.toFixed(0)}</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${trendScore}%"></div>
                    </div>
                </div>
            </div>
            
            <div class="deal-footer">
                <a href="${deal.url}" target="_blank" rel="noopener noreferrer" class="btn-view">
                    View on eBay →
                </a>
            </div>
        </div>
    `;
}

function getScoreColor(score) {
    if (score >= 85) return '#1B998B';     // Green
    if (score >= 70) return '#F7B32B';     // Yellow
    if (score >= 50) return '#FF6B35';     // Orange
    if (score >= 30) return '#EF476F';     // Red
    return '#666';                         // Gray
}

function showError(message) {
    const errorContainer = document.getElementById('errorContainer');
    errorContainer.textContent = message;
    errorContainer.classList.remove('d-none');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function exportToCSV(query) {
    try {
        const response = await fetch('/api/export');
        if (!response.ok) throw new Error('Export failed');
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ebay_deals_${new Date().toISOString().slice(0,10)}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    } catch (error) {
        console.error('Export error:', error);
        showError('Failed to export data');
    }
}