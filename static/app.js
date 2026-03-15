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
// Save / Skip / Age-filter state
// ---------------------------------------------------------------------------

/** @type {Set<string>} URLs currently saved by the user. */
let _savedUrls = new Set();

/** Current age filter in days (0 = no filter). */
let _maxAgeDays = 0;

/** Last search result deals array (for re-filtering without re-searching). */
let _lastDeals = [];

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

    const germanyOnlyToggleBtn = document.getElementById('germanyOnlyToggleBtn');
    if (germanyOnlyToggleBtn) {
        germanyOnlyToggleBtn.addEventListener('click', toggleGermanyOnly);
    }

    const dataSourceSelect = document.getElementById('dataSourceSelect');
    if (dataSourceSelect) {
        dataSourceSelect.addEventListener('change', saveDataSource);
    }

    // Age filter change handler.
    const ageFilterSelect = document.getElementById('ageFilterSelect');
    if (ageFilterSelect) {
        ageFilterSelect.addEventListener('change', () => {
            _maxAgeDays = parseInt(ageFilterSelect.value, 10) || 0;
            if (_lastDeals.length > 0) {
                _renderDeals(_lastDeals);
            }
        });
    }

    // Saved deals panel toggle.
    const savedDealsBtn = document.getElementById('savedDealsBtn');
    if (savedDealsBtn) {
        savedDealsBtn.addEventListener('click', toggleSavedPanel);
    }
    const closeSavedBtn = document.getElementById('closeSavedBtn');
    if (closeSavedBtn) {
        closeSavedBtn.addEventListener('click', () => {
            document.getElementById('savedPanel').classList.add('d-none');
        });
    }

    // Event delegation for save/skip buttons inside deal cards.
    const dealsGrid = document.getElementById('dealsGrid');
    if (dealsGrid) {
        dealsGrid.addEventListener('click', handleDealCardAction);
    }

    // Load saved URLs so the UI reflects saved state on initial render.
    loadSavedUrls();

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
        // Sync the data source selector.
        _setDataSourceState(data.data_source, data.active_data_source, data.ebay_api_configured);
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

/**
 * Update the data source selector and status badge.
 * @param {string} setting  - Persisted setting: "auto", "api", or "scraper".
 * @param {string} active   - The engine actually in use: "api" or "scraper".
 * @param {boolean} apiConfigured - Whether eBay API credentials are set.
 */
function _setDataSourceState(setting, active, apiConfigured) {
    const sel = document.getElementById('dataSourceSelect');
    const statusEl = document.getElementById('dataSourceStatus');
    if (sel && setting) {
        sel.value = setting;
    }
    if (statusEl) {
        if (active === 'api') {
            statusEl.textContent = '🟢 eBay API active';
            statusEl.className = 'data-source-status data-source-status--api';
        } else if (active === 'scraper') {
            const hint = (!apiConfigured && setting !== 'scraper')
                ? ' (API creds not set)'
                : '';
            statusEl.textContent = `🔵 Scraper active${hint}`;
            statusEl.className = 'data-source-status data-source-status--scraper';
        } else {
            statusEl.textContent = '';
            statusEl.className = 'data-source-status';
        }
    }
}

async function saveDataSource() {
    const sel = document.getElementById('dataSourceSelect');
    const statusEl = document.getElementById('dataSourceStatus');
    if (!sel) return;

    const newSource = sel.value;
    if (statusEl) {
        statusEl.textContent = '⏳ Saving…';
        statusEl.className = 'data-source-status';
    }

    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data_source: newSource }),
        });
        const data = await response.json();
        if (!response.ok) {
            const msg = (data.errors && data.errors.data_source) || data.error || 'Failed to save.';
            if (statusEl) {
                statusEl.textContent = `⚠️ ${msg}`;
                statusEl.className = 'data-source-status data-source-status--error';
            }
        } else {
            _setDataSourceState(data.data_source, data.active_data_source, data.ebay_api_configured);
        }
    } catch (err) {
        if (statusEl) {
            statusEl.textContent = `⚠️ Error: ${err.message}`;
            statusEl.className = 'data-source-status data-source-status--error';
        }
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

    // Show the data-source badge in the results header.
    const dsBadge = document.getElementById('dataSourceBadge');
    if (dsBadge) {
        if (data.data_source === 'api') {
            dsBadge.textContent = '🟢 Via eBay Official API';
            dsBadge.className = 'data-source-badge data-source-badge--api';
            dsBadge.classList.remove('d-none');
        } else if (data.data_source === 'scraper') {
            dsBadge.textContent = '🔵 Via Legacy Scraper';
            dsBadge.className = 'data-source-badge data-source-badge--scraper';
            dsBadge.classList.remove('d-none');
        } else {
            dsBadge.classList.add('d-none');
        }
    }

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

    // Store deals and render with current filters.
    _lastDeals = data.deals || [];
    _renderDeals(_lastDeals);

    document.getElementById('resultsContainer').classList.remove('d-none');

    // Setup export button
    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) {
        exportBtn.onclick = () => exportToCSV(data.query);
    }
}

/**
 * Render the deals grid, applying age filter and saved state.
 * @param {Array} deals - Full unfiltered deal list from the last search.
 */
function _renderDeals(deals) {
    const cutoffDate = _maxAgeDays > 0
        ? new Date(Date.now() - _maxAgeDays * 86400 * 1000)
        : null;

    const filtered = deals.filter(deal => {
        if (cutoffDate && deal.listing_date) {
            const listed = new Date(deal.listing_date);
            if (!isNaN(listed) && listed < cutoffDate) return false;
        }
        return true;
    });

    const dealsGrid = document.getElementById('dealsGrid');
    dealsGrid.innerHTML = filtered.map(deal => createDealCard(deal)).join('');

    // Update visible deal count to reflect filtered total.
    const dealCountEl = document.getElementById('dealCount');
    if (dealCountEl) {
        if (filtered.length !== deals.length) {
            dealCountEl.textContent = `${filtered.length} of ${deals.length}`;
        } else {
            dealCountEl.textContent = deals.length;
        }
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

    // ── Image section ──────────────────────────────────────────────────────
    const imageUrls = Array.isArray(deal.image_urls) ? deal.image_urls : [];
    let imageSection = '';
    if (imageUrls.length > 0) {
        const mainUrl = escapeHtml(imageUrls[0]);
        const titleAlt = escapeHtml(deal.title || 'Deal image');
        imageSection = `
            <div class="deal-image-section">
                <img class="deal-img" src="${mainUrl}" alt="${titleAlt}" loading="lazy"
                     onerror="this.style.display='none';this.parentElement.classList.add('deal-img-error')">
            </div>`;
    } else {
        imageSection = `<div class="deal-image-section"><div class="deal-img-placeholder">🎮</div></div>`;
    }

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
    // Show the item's physical location (e.g. "Berlin, DE") when available.
    // With Germany-only filtering active this should always be a German location.
    if (deal.item_location) {
        const loc = deal.item_location.trim().toUpperCase();
        const isGerman = loc === 'DE' || loc.endsWith(', DE') ||
                         loc.includes('DEUTSCHLAND') || loc.includes('GERMANY');
        const locationFlag = isGerman ? ' 🇩🇪' : '';
        metaRows.push(metaRow('Location', escapeHtml(deal.item_location) + locationFlag));
    }
    if (deal.is_trending) {
        metaRows.push(metaRow('Trending', '🔥 Yes'));
    }

    // ── Listing age ────────────────────────────────────────────────────────
    const ageHtml = buildListingAgeBadge(deal.listing_date);

    // ── Save / Skip buttons ────────────────────────────────────────────────
    const isSaved = deal.is_saved || _savedUrls.has(deal.url);
    const encodedUrl = escapeHtml(deal.url);
    const saveLabel = isSaved ? '★ Saved' : '☆ Save';
    const saveBtnClass = isSaved ? 'btn-deal-action btn-save btn-save--saved' : 'btn-deal-action btn-save';
    const actionsHtml = `
        <div class="deal-actions">
            <button class="${saveBtnClass}" data-action="${isSaved ? 'unsave' : 'save'}"
                    data-url="${encodedUrl}"
                    data-title="${escapeHtml(deal.title || '')}"
                    data-price="${deal.price || 0}"
                    title="${isSaved ? 'Remove from saved' : 'Save this deal'}">${saveLabel}</button>
            <button class="btn-deal-action btn-skip" data-action="skip"
                    data-url="${encodedUrl}"
                    title="Hide this deal from future searches">✕ Skip</button>
        </div>`;

    return `
        <div class="deal-card">
            <div class="deal-header">
                <div class="deal-score" style="color: ${scoreColor}">
                    ${score.toFixed(1)}
                </div>
                <div class="deal-header-right">
                    <div class="deal-recommendation">${recommendation}</div>
                    ${ageHtml}
                </div>
            </div>
            ${imageSection}
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
                ${actionsHtml}
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
 * Build a listing-age badge from an ISO-8601 listing date string.
 * Returns an empty string when no date is available.
 * @param {string|null|undefined} listingDate
 * @returns {string} HTML string
 */
function buildListingAgeBadge(listingDate) {
    if (!listingDate) return '';
    const listed = new Date(listingDate);
    if (isNaN(listed)) return '';
    const ageDays = Math.floor((Date.now() - listed) / (86400 * 1000));
    let label, cls;
    if (ageDays === 0) {
        label = 'Today';
        cls = 'deal-age-badge deal-age-badge--fresh';
    } else if (ageDays === 1) {
        label = '1 day old';
        cls = 'deal-age-badge deal-age-badge--fresh';
    } else if (ageDays <= 7) {
        label = `${ageDays} days old`;
        cls = 'deal-age-badge deal-age-badge--fresh';
    } else if (ageDays <= 30) {
        label = `${ageDays} days old`;
        cls = 'deal-age-badge deal-age-badge--recent';
    } else {
        label = ageDays >= 365
            ? `${Math.floor(ageDays / 365)}y old`
            : `${ageDays} days old`;
        cls = 'deal-age-badge deal-age-badge--old';
    }
    return `<span class="${cls}" title="Listed: ${listed.toLocaleDateString()}">${label}</span>`;
}

// ---------------------------------------------------------------------------
// Save / Skip event delegation
// ---------------------------------------------------------------------------

/**
 * Delegated click handler for save/skip buttons inside deal cards.
 * @param {MouseEvent} e
 */
async function handleDealCardAction(e) {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;

    const action = btn.dataset.action;
    const url = btn.dataset.url;
    if (!url) return;

    if (action === 'save') {
        await handleSaveDeal(btn, url);
    } else if (action === 'unsave') {
        await handleUnsaveDeal(btn, url);
    } else if (action === 'skip') {
        await handleSkipDeal(btn, url);
    }
}

async function handleSaveDeal(btn, url) {
    const title = btn.dataset.title || '';
    const price = parseFloat(btn.dataset.price) || 0;
    btn.disabled = true;
    try {
        const resp = await fetch('/api/deals/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url, title, price }),
        });
        if (resp.ok) {
            _savedUrls.add(url);
            btn.textContent = '★ Saved';
            btn.className = 'btn-deal-action btn-save btn-save--saved';
            btn.dataset.action = 'unsave';
            btn.title = 'Remove from saved';
        }
    } catch (err) {
        console.error('Save deal error:', err);
    } finally {
        btn.disabled = false;
    }
}

async function handleUnsaveDeal(btn, url) {
    btn.disabled = true;
    try {
        const resp = await fetch('/api/deals/unsave', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url }),
        });
        if (resp.ok) {
            _savedUrls.delete(url);
            btn.textContent = '☆ Save';
            btn.className = 'btn-deal-action btn-save';
            btn.dataset.action = 'save';
            btn.title = 'Save this deal';
            // Also refresh saved panel if visible.
            const savedPanel = document.getElementById('savedPanel');
            if (savedPanel && !savedPanel.classList.contains('d-none')) {
                loadSavedPanel();
            }
        }
    } catch (err) {
        console.error('Unsave deal error:', err);
    } finally {
        btn.disabled = false;
    }
}

async function handleSkipDeal(btn, url) {
    btn.disabled = true;
    try {
        const resp = await fetch('/api/deals/skip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url }),
        });
        if (resp.ok) {
            // Hide the card immediately.
            const card = btn.closest('.deal-card');
            if (card) {
                card.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
                card.style.opacity = '0';
                card.style.transform = 'scale(0.95)';
                setTimeout(() => card.remove(), 270);
            }
            // Remove from local deals array so age-filter re-render won't bring it back.
            _lastDeals = _lastDeals.filter(d => d.url !== url);
        }
    } catch (err) {
        console.error('Skip deal error:', err);
        btn.disabled = false;
    }
}

// ---------------------------------------------------------------------------
// Saved deals panel
// ---------------------------------------------------------------------------

async function loadSavedUrls() {
    try {
        const resp = await fetch('/api/deals/saved');
        if (!resp.ok) return;
        const deals = await resp.json();
        _savedUrls = new Set(deals.map(d => d.url));
    } catch (err) {
        console.warn('Could not load saved deals:', err);
    }
}

async function toggleSavedPanel() {
    const panel = document.getElementById('savedPanel');
    if (!panel) return;
    if (panel.classList.contains('d-none')) {
        panel.classList.remove('d-none');
        await loadSavedPanel();
    } else {
        panel.classList.add('d-none');
    }
}

async function loadSavedPanel() {
    const savedList = document.getElementById('savedList');
    if (!savedList) return;
    try {
        const resp = await fetch('/api/deals/saved');
        if (!resp.ok) throw new Error('Failed to load saved deals');
        const deals = await resp.json();
        if (!deals.length) {
            savedList.innerHTML = '<p class="saved-empty">No saved deals yet. Click ☆ Save on any deal card.</p>';
            return;
        }
        savedList.innerHTML = deals.map(d => `
            <div class="saved-item" data-url="${escapeHtml(d.url)}">
                <div class="saved-item-info">
                    <div class="saved-item-title" title="${escapeHtml(d.title || '')}">${escapeHtml(d.title || 'Unknown deal')}</div>
                    ${d.price ? `<div class="saved-item-price">€${(+d.price).toFixed(2)}</div>` : ''}
                </div>
                <div class="saved-item-actions">
                    <a href="${escapeHtml(d.url)}" target="_blank" rel="noopener noreferrer" class="btn-view-saved">View →</a>
                    <button class="btn-unsave" data-url="${escapeHtml(d.url)}" onclick="removeSavedItem(this)">✕</button>
                </div>
            </div>
        `).join('');
    } catch (err) {
        savedList.innerHTML = '<p class="saved-empty">⚠️ Could not load saved deals.</p>';
    }
}

async function removeSavedItem(btn) {
    const url = btn.dataset.url;
    if (!url) return;
    btn.disabled = true;
    try {
        const resp = await fetch('/api/deals/unsave', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url }),
        });
        if (resp.ok) {
            _savedUrls.delete(url);
            const item = btn.closest('.saved-item');
            if (item) item.remove();
            // Update save button state in deal cards if present.
            const saveBtn = document.querySelector(`.btn-save[data-url="${CSS.escape(url)}"]`);
            if (saveBtn) {
                saveBtn.textContent = '☆ Save';
                saveBtn.className = 'btn-deal-action btn-save';
                saveBtn.dataset.action = 'save';
            }
            // Show empty state if no more saved deals.
            const savedList = document.getElementById('savedList');
            if (savedList && !savedList.querySelector('.saved-item')) {
                savedList.innerHTML = '<p class="saved-empty">No saved deals yet. Click ☆ Save on any deal card.</p>';
            }
        }
    } catch (err) {
        console.error('Unsave error:', err);
        btn.disabled = false;
    }
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
 * Parse a German-formatted number string (e.g. "9,90" or "1.234,56") into a
 * JavaScript float.  Also handles standard English notation ("9.90").
 * @param {string} s - Raw number string extracted from eBay text.
 * @returns {number}
 */
function parseGermanNumber(s) {
    const clean = s.trim();
    // Both separators present – determine which is the decimal marker.
    // In German notation the decimal comma always comes last ("1.234,56"),
    // while in English notation the decimal point comes last ("1,234.56").
    if (clean.includes(',') && clean.includes('.')) {
        const lastComma = clean.lastIndexOf(',');
        const lastDot   = clean.lastIndexOf('.');
        if (lastComma > lastDot) {
            // German: "1.234,56" → remove thousands dots, replace decimal comma
            return parseFloat(clean.replace(/\./g, '').replace(',', '.')) || 0;
        }
        // English: "1,234.56" → remove thousands commas
        return parseFloat(clean.replace(/,/g, '')) || 0;
    }
    if (clean.includes(',')) {
        // German-only decimal: "9,90" → "9.90"
        return parseFloat(clean.replace(',', '.')) || 0;
    }
    return parseFloat(clean) || 0;
}

/**
 * Extract a clean, user-friendly shipping summary from raw eBay shipping text.
 *
 * Handles the following real-world eBay.de patterns:
 *   - "Kostenloser Versand" / "Gratis" / "Free shipping"  → "Free"
 *   - "EUR 7,95 Versand"                                  → "€7.95"
 *   - "EUR 7,95 bis EUR 69,00 Versand"                    → "€7.95 – €69.00"
 *   - "€ 9,90"                                            → "€9.90"
 *   - "Nicht angegeben"                                   → "N/A"
 *
 * @param {string} shippingText - Raw shipping text from the eBay listing.
 * @returns {string} Formatted shipping display string.
 */
function cleanShippingText(shippingText) {
    if (!shippingText) return 'N/A';

    const lower = shippingText.toLowerCase();

    // "Not specified" / "Nicht angegeben" – no shipping info available.
    if (lower.includes('nicht angegeben')) return 'N/A';

    // Free shipping – covers German and English phrasings used on eBay.
    if (
        lower.includes('kostenlos') ||
        lower.includes('gratis') ||
        lower.includes('free shipping') ||
        lower.includes('free postage')
    ) {
        return 'Free';
    }

    // Price range: "EUR 7,95 bis EUR 69,00" (German "bis" = "up to").
    // Display as "€7.95 – €69.00" so both bounds are clearly shown.
    const rangeMatch = shippingText.match(/EUR\s*([\d,.]+)\s+bis\s+EUR\s*([\d,.]+)/i);
    if (rangeMatch) {
        const lo = parseGermanNumber(rangeMatch[1]);
        const hi = parseGermanNumber(rangeMatch[2]);
        return `€${lo.toFixed(2)} – €${hi.toFixed(2)}`;
    }

    // Single EUR amount: "EUR 9,90 Versand".
    const singleEurMatch = shippingText.match(/EUR\s*([\d,.]+)/i);
    if (singleEurMatch) {
        const amount = parseGermanNumber(singleEurMatch[1]);
        return `€${amount.toFixed(2)}`;
    }

    // Euro symbol directly: "€ 9,90" or "€9,90".
    const euroSymbolMatch = shippingText.match(/€\s*([\d,.]+)/);
    if (euroSymbolMatch) {
        const amount = parseGermanNumber(euroSymbolMatch[1]);
        return `€${amount.toFixed(2)}`;
    }

    // Fallback: truncate overly long raw text to keep cards tidy.
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
    const potentialScam = !!deal.ai_potential_scam;
    const scamWarning = deal.ai_scam_warning || '';

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

    const scamBannerHtml = potentialScam
        ? `<div class="scam-warning">
               <span class="scam-warning-icon">🚨</span>
               <span class="scam-warning-label">POTENTIAL SCAM</span>
               ${scamWarning ? `<p class="scam-warning-detail">${escapeHtml(scamWarning)}</p>` : ''}
           </div>`
        : '';

    return `
        <div class="ai-verdict${potentialScam ? ' ai-verdict-scam' : ''}">
            ${scamBannerHtml}
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