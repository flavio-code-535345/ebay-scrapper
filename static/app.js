// eBay Deal Finder - Frontend Application

// ---------------------------------------------------------------------------
// Progress bar helpers
// ---------------------------------------------------------------------------

let _progressTimer = null;

const _PROGRESS_STAGES = [
    { target: 35,  label: '🔍 Searching eBay listings…',  duration: 3500 },
    { target: 82,  label: '🤖 Running AI scoring…',       duration: 9000 },
    { target: 96,  label: '⚙️ Processing results…',       duration: 2500 },
];

function startProgress() {
    const fill  = document.getElementById('progressBarFill');
    const label = document.getElementById('progressLabel');
    const pct   = document.getElementById('progressPct');
    if (!fill) return;

    let stageIdx   = 0;
    let stageStart = Date.now();
    let current    = 0;

    _progressTimer = setInterval(() => {
        const stage    = _PROGRESS_STAGES[Math.min(stageIdx, _PROGRESS_STAGES.length - 1)];
        const prev     = stageIdx === 0 ? 0 : _PROGRESS_STAGES[stageIdx - 1].target;
        const elapsed  = Date.now() - stageStart;
        const stagePct = Math.min(1, elapsed / stage.duration);

        current = prev + (stage.target - prev) * stagePct;

        if (stagePct >= 1 && stageIdx < _PROGRESS_STAGES.length - 1) {
            stageIdx++;
            stageStart = Date.now();
        }

        const rounded = Math.round(current);
        fill.style.width  = rounded + '%';
        pct.textContent   = rounded + '%';
        label.textContent = stage.label;
    }, 100);
}

function stopProgress(success) {
    if (_progressTimer) {
        clearInterval(_progressTimer);
        _progressTimer = null;
    }
    const fill  = document.getElementById('progressBarFill');
    const label = document.getElementById('progressLabel');
    const pct   = document.getElementById('progressPct');
    if (!fill) return;
    if (success) {
        fill.style.width  = '100%';
        pct.textContent   = '100%';
        label.textContent = '✅ Done!';
    }
}

// ---------------------------------------------------------------------------
// Main search handler
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('searchForm');
    if (searchForm) {
        searchForm.addEventListener('submit', handleSearch);
    }

    // Load and display the active Gemini model and AI-enabled state from settings.
    loadModelSettings();

    const saveModelBtn = document.getElementById('saveModelBtn');
    if (saveModelBtn) {
        saveModelBtn.addEventListener('click', saveModelSettings);
    }

    const aiToggleBtn = document.getElementById('aiToggleBtn');
    if (aiToggleBtn) {
        aiToggleBtn.addEventListener('click', toggleAiEnabled);
    }

    // Check URL parameters for auto-search
    const params = new URLSearchParams(window.location.search);
    const searchParam = params.get('search');
    if (searchParam) {
        document.getElementById('searchQuery').value = searchParam;
        handleSearch(new Event('submit'));
    }
});

async function loadModelSettings() {
    try {
        const response = await fetch('/api/settings');
        if (!response.ok) return;
        const data = await response.json();
        const input = document.getElementById('geminiModelInput');
        const status = document.getElementById('modelStatus');
        if (input && data.gemini_model) {
            input.value = data.gemini_model;
        }
        if (status && data.gemini_model) {
            status.textContent = `Active model: ${data.gemini_model}`;
            status.className = 'model-status model-status--active';
        }
        // Sync the AI toggle button to the persisted state.
        if (typeof data.ai_enabled === 'boolean') {
            _setAiToggleState(data.ai_enabled);
        }
    } catch (err) {
        console.warn('Failed to load model settings:', err);
    }
}

/**
 * Update the AI toggle button visual state without making an API call.
 * @param {boolean} enabled
 */
function _setAiToggleState(enabled) {
    const btn = document.getElementById('aiToggleBtn');
    if (!btn) return;
    btn.setAttribute('aria-pressed', String(enabled));
    const label = btn.querySelector('.ai-toggle-label');
    const icon = btn.querySelector('.ai-toggle-icon');
    if (label) label.textContent = enabled ? 'AI: ON' : 'AI: OFF';
    if (icon) icon.textContent = enabled ? '✨' : '⭕';
}

async function toggleAiEnabled() {
    const btn = document.getElementById('aiToggleBtn');
    if (!btn) return;
    const currentlyEnabled = btn.getAttribute('aria-pressed') === 'true';
    const newState = !currentlyEnabled;

    // Optimistically update UI immediately.
    _setAiToggleState(newState);
    btn.disabled = true;

    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ai_enabled: newState }),
        });
        const data = await response.json();
        if (!response.ok) {
            // Revert on failure.
            _setAiToggleState(currentlyEnabled);
            const status = document.getElementById('modelStatus');
            if (status) {
                status.textContent = `⚠️ ${data.error || 'Failed to save AI toggle.'}`;
                status.className = 'model-status model-status--error';
            }
        } else {
            _setAiToggleState(data.ai_enabled);
        }
    } catch (err) {
        // Revert on error.
        _setAiToggleState(currentlyEnabled);
        const status = document.getElementById('modelStatus');
        if (status) {
            status.textContent = `⚠️ Error: ${err.message}`;
            status.className = 'model-status model-status--error';
        }
    } finally {
        btn.disabled = false;
    }
}

async function saveModelSettings() {
    const input = document.getElementById('geminiModelInput');
    const status = document.getElementById('modelStatus');
    const btn = document.getElementById('saveModelBtn');
    if (!input || !status || !btn) return;

    const model = input.value.trim();
    if (!model) {
        status.textContent = '⚠️ Please enter a model name.';
        status.className = 'model-status model-status--error';
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Saving…';
    status.textContent = '';
    status.className = 'model-status';

    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ gemini_model: model }),
        });
        const data = await response.json();
        if (!response.ok) {
            const msg = (data.errors && data.errors.gemini_model) || data.error || 'Failed to save.';
            status.textContent = `⚠️ ${msg}`;
            status.className = 'model-status model-status--error';
        } else {
            status.textContent = `✅ Active model: ${data.gemini_model}`;
            status.className = 'model-status model-status--active';
        }
    } catch (err) {
        status.textContent = `⚠️ Error: ${err.message}`;
        status.className = 'model-status model-status--error';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Save';
    }
}

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
    startProgress();

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
            console.error('API error response:', data);
            throw new Error(data.error || `HTTP ${response.status}: Search failed`);
        }

        if (data.errors && data.errors.length) {
            console.warn('Search completed with warnings:', data.errors);
        }

        if (!data.deals || data.deals.length === 0) {
            stopProgress(false);
            document.getElementById('loadingContainer').classList.add('d-none');
            const errorLines = (data.errors && data.errors.length)
                ? data.errors
                : ['No matching items found on eBay for this search term.'];
            console.warn('No deals found. Details:', errorLines);
            showDetailedError('No deals found. Try a different search term.', errorLines);
            document.getElementById('emptyState').classList.remove('d-none');
            return;
        }

        stopProgress(true);
        // Brief pause so the user sees "Done!" before results render.
        await new Promise(r => setTimeout(r, 300));
        displayResults(data);

    } catch (error) {
        console.error('Search error:', error);
        stopProgress(false);
        document.getElementById('loadingContainer').classList.add('d-none');
        showDetailedError(
            error.message || 'An error occurred during search',
            ['Check the browser console (F12) for more details.']
        );
    } finally {
        searchBtn.disabled = false;
        spinner.classList.add('d-none');
        btnText.textContent = 'Search';
    }
}

function displayResults(data) {
    document.getElementById('loadingContainer').classList.add('d-none');
    document.getElementById('dealCount').textContent = data.deal_count;

    // Show a warning banner if Gemini quota is exhausted or AI is disabled.
    const aiWarning = document.getElementById('aiWarningContainer');
    if (aiWarning) {
        if (!data.ai_enabled) {
            aiWarning.textContent =
                '⭕ AI evaluation is OFF — showing rules-based scores only. ' +
                'Toggle AI ON to enable Gemini scoring.';
            aiWarning.className = 'alert alert-warning';
            aiWarning.classList.remove('d-none');
        } else if (data.ai_rate_limited) {
            const secs = data.ai_paused_seconds || 0;
            aiWarning.textContent =
                `⚠️ Gemini AI is temporarily paused due to quota exhaustion` +
                (secs > 0 ? ` (resumes in ~${secs}s)` : '') +
                `. Showing rules-based scores only.`;
            aiWarning.className = 'alert alert-warning';
            aiWarning.classList.remove('d-none');
        } else {
            aiWarning.classList.add('d-none');
        }
    }
    
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

    // ── Parse condition text into structured parts ─────────────────────────
    const conditionParts = parseConditionParts(deal.condition || '');
    const sellerType   = conditionParts.sellerType;
    const conditionStr = conditionParts.condition;
    const deviceStr    = conditionParts.device;

    // ── Clean shipping text ────────────────────────────────────────────────
    const shippingClean = cleanShippingText(deal.shipping);

    // ── Image issue warning section ────────────────────────────────────────
    const imageIssues = Array.isArray(deal.image_issues) ? deal.image_issues : [];
    const imageWarningSection = buildImageIssueSection(imageIssues);

    // ── AI (Gemini) verdict section ────────────────────────────────────────
    let aiSection = '';
    if (deal.ai_assessed) {
        aiSection = buildAiSection(deal);
    } else if (deal.ai_error_type === 'rate_limit') {
        aiSection = buildAiErrorSection('⏳ AI paused (quota limit reached)');
    } else if (deal.ai_error_type === 'parse_error') {
        aiSection = buildAiErrorSection('⚠️ AI response could not be parsed');
    }

    // ── Build meta rows ────────────────────────────────────────────────────
    const metaRows = [];

    if (sellerType) {
        metaRows.push(metaRow('Type', escapeHtml(sellerType)));
    }
    metaRows.push(metaRow('Condition', escapeHtml(conditionStr || 'Unknown')));
    if (deviceStr) {
        metaRows.push(metaRow('Device', escapeHtml(deviceStr)));
    }
    metaRows.push(metaRow('Seller', (deal.seller_rating || 0).toFixed(1) + '%'));
    metaRows.push(metaRow('Shipping', escapeHtml(shippingClean)));
    if (deal.is_trending) {
        metaRows.push(metaRow('Trending', '🔥 Yes'));
    }

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
                    €${(deal.price || 0).toFixed(2)}
                </div>

                <div class="deal-meta">
                    ${metaRows.join('')}
                </div>

                ${imageWarningSection}
                ${aiSection}

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

/**
 * Build a single meta-info row (label + value).
 * @param {string} label
 * @param {string} value - already HTML-escaped where required
 * @returns {string} HTML string
 */
function metaRow(label, value) {
    return `<div class="meta-row"><span class="meta-label">${label}</span><span class="meta-value">${value}</span></div>`;
}

/**
 * Parse eBay's combined SECONDARY_INFO / condition text into structured parts.
 * eBay.de typically shows: "Neu | Gewerblich | Microsoft Xbox 360"
 * @param {string} conditionText
 * @returns {{ condition: string, sellerType: string, device: string }}
 */
function parseConditionParts(conditionText) {
    const result = { condition: conditionText, sellerType: '', device: '' };

    if (!conditionText || !conditionText.includes('|')) {
        return result;
    }

    const parts = conditionText.split('|').map(p => p.trim()).filter(Boolean);
    const remaining = [];

    for (const part of parts) {
        const lower = part.toLowerCase();
        if (lower.includes('gewerblich') || lower.includes('unternehmen')) {
            result.sellerType = 'Gewerblich';
        } else if (lower.includes('privat')) {
            result.sellerType = 'Privat';
        } else {
            remaining.push(part);
        }
    }

    if (remaining.length > 0) {
        result.condition = remaining[0];
    }
    if (remaining.length > 1) {
        result.device = remaining.slice(1).join(' | ');
    }

    return result;
}

/**
 * Extract a clean, short shipping summary from raw eBay shipping text.
 * @param {string} shippingText
 * @returns {string}
 */
function cleanShippingText(shippingText) {
    if (!shippingText) return 'N/A';

    const lower = shippingText.toLowerCase();

    // Free shipping
    if (lower.includes('kostenlos') || lower.startsWith('gratis')) {
        return 'Gratis';
    }

    // Extract first EUR price (handles "EUR 9,90 bis EUR 11,90…")
    const priceMatch = shippingText.match(/EUR\s*[\d,.]+(?:\s+bis\s+EUR\s*[\d,.]+)?/);
    if (priceMatch) {
        return priceMatch[0].replace(/\s+/g, ' ').trim();
    }

    // Truncate long text as fallback
    return shippingText.length > 55 ? shippingText.slice(0, 55).trimEnd() + '…' : shippingText;
}

/**
 * Build an image issue warning section for a deal card.
 * @param {string[]} issues - Array of image issue identifiers.
 * @returns {string} HTML string, or empty string when there are no issues.
 */
function buildImageIssueSection(issues) {
    if (!issues || issues.length === 0) return '';

    const issueLabels = {
        'no_images': '📷 No product images available',
    };

    const items = issues.map(issue => {
        const label = issueLabels[issue] || escapeHtml(issue);
        return `<span class="image-issue-badge">${label}</span>`;
    }).join('');

    return `<div class="image-issues">${items}</div>`;
}

/**
 * Build the Gemini AI verdict section HTML for a deal card.
 * @param {Object} deal - Deal object with ai_* fields set.
 * @returns {string} HTML string.
 */
function buildAiSection(deal) {
    const rating = deal.ai_deal_rating || 'Unknown';
    const confidence = deal.ai_confidence_score || 0;
    const summary = deal.ai_verdict_summary || '';
    const estimate = deal.ai_fair_market_estimate || '';
    const visualFindings = Array.isArray(deal.ai_visual_findings) ? deal.ai_visual_findings : [];
    const redFlags = Array.isArray(deal.ai_red_flags) ? deal.ai_red_flags : [];

    const badgeClass = getAiBadgeClass(rating);

    const findingsHtml = visualFindings.length
        ? `<ul class="ai-list">${visualFindings.map(f => `<li>${escapeHtml(f)}</li>`).join('')}</ul>`
        : '';

    const redFlagsHtml = redFlags.length
        ? `<ul class="ai-list ai-red-flags">${redFlags.map(f => `<li>⚠️ ${escapeHtml(f)}</li>`).join('')}</ul>`
        : '';

    const estimateHtml = estimate
        ? `<div class="ai-estimate">💰 Fair market estimate: <strong>${escapeHtml(estimate)}</strong></div>`
        : '';

    return `
        <div class="ai-verdict">
            <div class="ai-verdict-header">
                <span class="ai-badge ${badgeClass}">${escapeHtml(rating)}</span>
                <span class="ai-confidence">AI Confidence: ${confidence}%</span>
            </div>
            ${summary ? `<p class="ai-summary">${escapeHtml(summary)}</p>` : ''}
            ${estimateHtml}
            ${findingsHtml}
            ${redFlagsHtml}
        </div>
    `;
}

/**
 * Build a small AI error notice for a deal card.
 * @param {string} message - Human-friendly error description.
 * @returns {string} HTML string.
 */
function buildAiErrorSection(message) {
    return `
        <div class="ai-verdict ai-verdict-error">
            <div class="ai-verdict-header">
                <span class="ai-badge badge-unknown">${escapeHtml(message)}</span>
            </div>
        </div>
    `;
}

/**
 * Map an AI deal rating to its CSS badge class.
 * @param {string} rating - The AI deal rating string (e.g. "Must Buy", "Fair", "Avoid").
 * @returns {string} CSS class name for the badge element.
 */
function getAiBadgeClass(rating) {
    const r = (rating || '').toLowerCase();
    if (r.includes('must') || r === 'must buy') return 'badge-must-buy';
    if (r === 'fair') return 'badge-fair';
    if (r.includes('avoid') || r.includes('hard pass')) return 'badge-avoid';
    return 'badge-unknown';
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

function showDetailedError(summary, details) {
    const errorContainer = document.getElementById('errorContainer');
    const detailsHtml = details.map(d => `<li><pre class="error-detail">${escapeHtml(d)}</pre></li>`).join('');
    errorContainer.innerHTML = `
        <strong>${escapeHtml(summary)}</strong>
        <ul class="error-details-list">${detailsHtml}</ul>
    `;
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