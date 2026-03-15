// eBay Deal Finder – Frontend Application (Dark JobOps Redesign)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_SEARCH_RESULTS = 200;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let _currentPipeline = 'ready';
let _lastDeals       = [];
let _maxAgeDays      = 0;
let _keywordFilter   = '';
let _ratingFilter    = '';
let _savedUrls       = new Set();
let _selectedUrls    = new Set();
let _abortController = null;
let _progressTimer   = null;

const _PROGRESS_STAGES = [
    { target: 35, label: '🔍 Searching eBay listings…', duration: 3500 },
    { target: 82, label: '🤖 Running AI scoring…',      duration: 9000 },
    { target: 96, label: '⚙️ Processing results…',      duration: 2500 },
];

// ---------------------------------------------------------------------------
// Progress helpers
// ---------------------------------------------------------------------------

function startProgress() {
    const fill  = document.getElementById('progressBarFill');
    const label = document.getElementById('progressLabel');
    const pct   = document.getElementById('progressPct');
    if (!fill) return;

    let stageIdx   = 0;
    let stageStart = Date.now();

    _progressTimer = setInterval(() => {
        const stage   = _PROGRESS_STAGES[Math.min(stageIdx, _PROGRESS_STAGES.length - 1)];
        const prev    = stageIdx === 0 ? 0 : _PROGRESS_STAGES[stageIdx - 1].target;
        const elapsed = Date.now() - stageStart;
        const stagePct = Math.min(1, elapsed / stage.duration);
        const current  = prev + (stage.target - prev) * stagePct;

        if (stagePct >= 1 && stageIdx < _PROGRESS_STAGES.length - 1) {
            stageIdx++;
            stageStart = Date.now();
        }

        const rounded = Math.round(current);
        fill.style.width    = rounded + '%';
        pct.textContent     = rounded + '%';
        label.textContent   = stage.label;
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
    } else {
        fill.style.width  = '0%';
        pct.textContent   = '0%';
        label.textContent = '';
    }
}

// ---------------------------------------------------------------------------
// DOMContentLoaded – wire up all event handlers
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    // Search form
    const searchForm = document.getElementById('searchForm');
    if (searchForm) searchForm.addEventListener('submit', handleSearch);

    // Cancel search
    const cancelBtn = document.getElementById('cancelSearchBtn');
    if (cancelBtn) cancelBtn.addEventListener('click', cancelSearch);

    // Pipeline tabs
    document.querySelectorAll('.pipeline-tab').forEach(btn => {
        btn.addEventListener('click', () => switchPipeline(btn.dataset.pipeline));
    });

    // Keyword filter
    const keywordInput = document.getElementById('keywordFilter');
    if (keywordInput) {
        keywordInput.addEventListener('input', () => {
            _keywordFilter = keywordInput.value.trim().toLowerCase();
            if (_currentPipeline === 'ready') _renderDeals(_lastDeals);
        });
    }

    // Age filter
    const ageSelect = document.getElementById('ageFilterSelect');
    if (ageSelect) {
        ageSelect.addEventListener('change', () => {
            _maxAgeDays = parseInt(ageSelect.value, 10) || 0;
            if (_currentPipeline === 'ready') _renderDeals(_lastDeals);
        });
    }

    // Rating filter
    const ratingSelect = document.getElementById('ratingFilter');
    if (ratingSelect) {
        ratingSelect.addEventListener('change', () => {
            _ratingFilter = ratingSelect.value.toLowerCase();
            if (_currentPipeline === 'ready') _renderDeals(_lastDeals);
        });
    }

    // History search
    const historySearch = document.getElementById('historySearch');
    if (historySearch) {
        historySearch.addEventListener('input', () => filterHistoryList(historySearch.value.trim().toLowerCase()));
    }

    // Batch selection
    const selectAllBtn = document.getElementById('selectAllBtn');
    if (selectAllBtn) selectAllBtn.addEventListener('click', toggleSelectAll);

    const batchSaveBtn = document.getElementById('batchSaveBtn');
    if (batchSaveBtn) batchSaveBtn.addEventListener('click', handleBatchSave);

    const batchSkipBtn = document.getElementById('batchSkipBtn');
    if (batchSkipBtn) batchSkipBtn.addEventListener('click', handleBatchSkip);

    const clearSelectionBtn = document.getElementById('clearSelectionBtn');
    if (clearSelectionBtn) clearSelectionBtn.addEventListener('click', clearSelection);

    // Export
    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) exportBtn.addEventListener('click', () => exportToCSV());

    // Settings
    const saveModelBtn = document.getElementById('saveModelBtn');
    if (saveModelBtn) saveModelBtn.addEventListener('click', saveModelSettings);

    const aiToggleBtn = document.getElementById('aiToggleBtn');
    if (aiToggleBtn) aiToggleBtn.addEventListener('click', toggleAiEnabled);

    const dataSourceSelect = document.getElementById('dataSourceSelect');
    if (dataSourceSelect) dataSourceSelect.addEventListener('change', saveDataSource);

    // Event delegation for deal cards (save/skip/unskip/checkbox)
    const dealsGrid = document.getElementById('dealsGrid');
    if (dealsGrid) {
        dealsGrid.addEventListener('click', handleDealCardAction);
        dealsGrid.addEventListener('change', handleDealCardCheckbox);
    }

    // URL params auto-search
    const params = new URLSearchParams(window.location.search);
    const autoSearch = params.get('search');
    if (autoSearch) {
        document.getElementById('searchQuery').value = autoSearch;
        handleSearch(new Event('submit'));
    }

    // Init
    loadModelSettings();
    loadSavedUrls();
    updateTabBadges();
});

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

async function handleSearch(e) {
    e.preventDefault();

    const query    = document.getElementById('searchQuery').value.trim();
    const searchBtn = document.getElementById('searchBtn');
    const spinner   = searchBtn.querySelector('.spinner-border');
    const btnText   = searchBtn.querySelector('.btn-text');

    if (!query) {
        showError('Please enter a search term');
        return;
    }

    // Abort any prior in-flight search
    if (_abortController) _abortController.abort();
    _abortController = new AbortController();

    // UI – loading state
    searchBtn.disabled = true;
    spinner.classList.remove('d-none');
    btnText.textContent = 'Searching…';

    document.getElementById('activePipelineBar').classList.remove('d-none');
    document.getElementById('errorContainer').classList.add('d-none');
    document.getElementById('aiWarningContainer').classList.add('d-none');
    document.getElementById('emptyState').classList.add('d-none');
    document.getElementById('dealsGrid').innerHTML = '';

    startProgress();
    switchPipeline('ready', /* suppressLoad */ true);

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, max_results: MAX_SEARCH_RESULTS }),
            signal: _abortController.signal,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || `HTTP ${response.status}: Search failed`);
        }

        stopProgress(true);
        await new Promise(r => setTimeout(r, 300));

        if (!data.deals || data.deals.length === 0) {
            const errorLines = (data.errors && data.errors.length)
                ? data.errors
                : ['No matching items found on eBay for this search term.'];
            showDetailedError('No deals found. Try a different search term.', errorLines);
            document.getElementById('emptyState').classList.remove('d-none');
        } else {
            _applySearchResults(data);
        }

        updateTabBadges();

    } catch (err) {
        if (err.name === 'AbortError') {
            stopProgress(false);
        } else {
            stopProgress(false);
            showDetailedError(
                err.message || 'An error occurred during search',
                ['Check the browser console (F12) for more details.']
            );
        }
    } finally {
        searchBtn.disabled = false;
        spinner.classList.add('d-none');
        btnText.textContent = 'Search';
        document.getElementById('activePipelineBar').classList.add('d-none');
        _abortController = null;
    }
}

function cancelSearch() {
    if (_abortController) {
        _abortController.abort();
        _abortController = null;
    }
}

function _applySearchResults(data) {
    // Update data-source badge
    const dsBadge = document.getElementById('dataSourceBadge');
    if (dsBadge) {
        if (data.data_source === 'api') {
            dsBadge.textContent = '🟢 Via eBay Official API';
            dsBadge.className   = 'data-source-badge data-source-badge--api';
            dsBadge.classList.remove('d-none');
        } else if (data.data_source === 'scraper') {
            dsBadge.textContent = '🔵 Via Legacy Scraper';
            dsBadge.className   = 'data-source-badge data-source-badge--scraper';
            dsBadge.classList.remove('d-none');
        } else {
            dsBadge.classList.add('d-none');
        }
    }

    // AI warning banner
    const aiWarning = document.getElementById('aiWarningContainer');
    if (aiWarning) {
        if (!data.ai_enabled) {
            aiWarning.textContent = '⭕ AI evaluation is OFF — showing rules-based scores only. Toggle AI ON to enable Gemini scoring.';
            aiWarning.className   = 'alert alert-warning';
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

    _lastDeals = data.deals || [];
    _renderDeals(_lastDeals);
}

// ---------------------------------------------------------------------------
// Pipeline switching
// ---------------------------------------------------------------------------

/**
 * Switch the active pipeline tab and load appropriate content.
 * @param {string} name - 'ready' | 'saved' | 'skipped' | 'history'
 * @param {boolean} [suppressLoad] - Skip loading data (used during search init)
 */
async function switchPipeline(name, suppressLoad = false) {
    _currentPipeline = name;

    // Update tab active state
    document.querySelectorAll('.pipeline-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.pipeline === name);
    });

    const dealsGrid  = document.getElementById('dealsGrid');
    const historyView = document.getElementById('historyView');
    const filtersBar = document.getElementById('filtersBar');
    const emptyState = document.getElementById('emptyState');

    // Reset empty state
    emptyState.classList.add('d-none');

    if (name === 'history') {
        dealsGrid.classList.add('d-none');
        historyView.classList.remove('d-none');
        filtersBar.classList.add('d-none');
        if (!suppressLoad) await loadHistoryView();
    } else {
        dealsGrid.classList.remove('d-none');
        historyView.classList.add('d-none');
        filtersBar.classList.remove('d-none');

        if (!suppressLoad) {
            if (name === 'ready') {
                _renderDeals(_lastDeals);
            } else if (name === 'saved') {
                await loadSavedView();
            } else if (name === 'skipped') {
                await loadSkippedView();
            }
        }
    }

    // Clear selection when switching pipelines
    clearSelection();
}

// ---------------------------------------------------------------------------
// Filters & Render
// ---------------------------------------------------------------------------

/**
 * Apply keyword, age, and rating filters to a deal array.
 * @param {Array} deals
 * @returns {Array} filtered deals
 */
function _applyFilters(deals) {
    const cutoff = _maxAgeDays > 0
        ? new Date(Date.now() - _maxAgeDays * 86400 * 1000)
        : null;

    return deals.filter(deal => {
        // Age filter
        if (cutoff && deal.listing_date) {
            const listed = new Date(deal.listing_date);
            if (!isNaN(listed) && listed < cutoff) return false;
        }

        // Keyword filter
        if (_keywordFilter) {
            const title = (deal.title || '').toLowerCase();
            if (!title.includes(_keywordFilter)) return false;
        }

        // Rating filter
        if (_ratingFilter) {
            const rating = (deal.ai_deal_rating || '').toLowerCase();
            // Treat legacy "fair" ratings as "okay" and "must buy" as "must have" for filter purposes
            const normRating = rating === 'fair' ? 'okay' : rating === 'must buy' ? 'must have' : rating;
            if (normRating !== _ratingFilter) return false;
        }

        return true;
    });
}

/**
 * Render deals to the grid with current filters applied.
 * @param {Array} deals - Full unfiltered deal list.
 * @param {string} [mode] - 'ready' | 'saved' | 'skipped'
 */
function _renderDeals(deals, mode) {
    const renderMode = mode || _currentPipeline;
    const filtered   = _applyFilters(deals);

    const dealsGrid = document.getElementById('dealsGrid');
    if (filtered.length === 0) {
        dealsGrid.innerHTML = '';
        if (deals.length === 0) {
            document.getElementById('emptyState').classList.remove('d-none');
        } else {
            // Deals exist but all filtered out – show inline message
            dealsGrid.innerHTML = `<div class="empty-state" style="grid-column:1/-1"><p>😕 No deals match current filters.</p></div>`;
        }
    } else {
        document.getElementById('emptyState').classList.add('d-none');
        dealsGrid.innerHTML = filtered.map(deal => createDealCard(deal, renderMode)).join('');
    }

    // Update badge
    const badge = document.getElementById('dealCountBadge');
    if (badge) {
        badge.textContent = filtered.length !== deals.length
            ? `${filtered.length} of ${deals.length} deals`
            : `${deals.length} deal${deals.length !== 1 ? 's' : ''}`;
    }

    // Update ready tab badge
    if (renderMode === 'ready') {
        const readyBadge = document.getElementById('badge-ready');
        if (readyBadge) readyBadge.textContent = filtered.length;
    }
}

// ---------------------------------------------------------------------------
// Saved / Skipped views
// ---------------------------------------------------------------------------

async function loadSavedView() {
    const dealsGrid = document.getElementById('dealsGrid');
    dealsGrid.innerHTML = '<div class="history-loading">⏳ Loading saved deals…</div>';
    try {
        const resp = await fetch('/api/deals/saved');
        if (!resp.ok) throw new Error('Failed to load');
        const deals = await resp.json();
        _renderDeals(deals, 'saved');
    } catch (err) {
        dealsGrid.innerHTML = '<div class="history-empty">⚠️ Could not load saved deals.</div>';
    }
}

async function loadSkippedView() {
    const dealsGrid = document.getElementById('dealsGrid');
    dealsGrid.innerHTML = '<div class="history-loading">⏳ Loading skipped deals…</div>';
    try {
        const resp = await fetch('/api/deals/skipped');
        if (!resp.ok) throw new Error('Failed to load');
        const deals = await resp.json();
        _renderDeals(deals, 'skipped');
    } catch (err) {
        dealsGrid.innerHTML = '<div class="history-empty">⚠️ Could not load skipped deals.</div>';
    }
}

// ---------------------------------------------------------------------------
// Tab badge counts
// ---------------------------------------------------------------------------

async function updateTabBadges() {
    try {
        const [savedResp, skippedResp] = await Promise.all([
            fetch('/api/deals/saved'),
            fetch('/api/deals/skipped'),
        ]);
        if (savedResp.ok) {
            const saved = await savedResp.json();
            const el = document.getElementById('badge-saved');
            if (el) el.textContent = saved.length;
        }
        if (skippedResp.ok) {
            const skipped = await skippedResp.json();
            const el = document.getElementById('badge-skipped');
            if (el) el.textContent = skipped.length;
        }
    } catch (_) { /* silently ignore */ }
}

// ---------------------------------------------------------------------------
// Deal card creation
// ---------------------------------------------------------------------------

/**
 * Build a deal card HTML string.
 * @param {Object} deal
 * @param {string} [mode] - 'ready' | 'saved' | 'skipped'
 * @returns {string} HTML
 */
function createDealCard(deal, mode) {
    const renderMode = mode || _currentPipeline;
    const score = deal.overall_score || 0;
    const scoreColor = getScoreColor(score);
    const recommendation = deal.recommendation || 'N/A';

    const priceScore     = deal.price_score     || 0;
    const sellerScore    = deal.seller_score     || 0;
    const conditionScore = deal.condition_score  || 0;
    const trendScore     = deal.trend_score      || 0;

    const conditionParts = parseConditionParts(deal.condition || '');
    const sellerType     = conditionParts.sellerType;
    const conditionStr   = conditionParts.condition;
    const deviceStr      = conditionParts.device;

    const shippingClean = cleanShippingText(deal.shipping);

    // Image section
    const imageUrls = Array.isArray(deal.image_urls) ? deal.image_urls : [];
    let imageSection = '';
    if (imageUrls.length > 0) {
        const mainUrl  = escapeHtml(imageUrls[0]);
        const titleAlt = escapeHtml(deal.title || 'Deal image');
        imageSection = `<div class="deal-image-section">
            <img class="deal-img" src="${mainUrl}" alt="${titleAlt}" loading="lazy"
                 onerror="this.style.display='none';this.parentElement.classList.add('deal-img-error')">
        </div>`;
    } else {
        imageSection = `<div class="deal-image-section"><div class="deal-img-placeholder">🎮</div></div>`;
    }

    const imageIssues       = Array.isArray(deal.image_issues) ? deal.image_issues : [];
    const imageWarningSection = buildImageIssueSection(imageIssues);

    // AI section
    let aiSection = '';
    if (deal.ai_assessed) {
        aiSection = buildAiSection(deal);
    } else if (deal.ai_error_type === 'rate_limit') {
        aiSection = buildAiErrorSection('⏳ AI paused (quota limit reached)');
    } else if (deal.ai_error_type === 'parse_error') {
        aiSection = buildAiErrorSection('⚠️ AI response could not be parsed');
    }

    // Meta rows
    const metaRows = [];
    if (sellerType) metaRows.push(metaRow('Type', escapeHtml(sellerType)));
    metaRows.push(metaRow('Condition', escapeHtml(conditionStr || 'Unknown')));
    if (deviceStr) metaRows.push(metaRow('Device', escapeHtml(deviceStr)));
    if (deal.seller_rating) metaRows.push(metaRow('Seller', deal.seller_rating.toFixed(1) + '%'));
    metaRows.push(metaRow('Shipping', escapeHtml(shippingClean)));
    if (deal.item_location) {
        const loc = deal.item_location.trim().toUpperCase();
        const flag = (loc === 'DE' || loc.endsWith(', DE') || loc.includes('DEUTSCHLAND') || loc.includes('GERMANY')) ? ' 🇩🇪' : '';
        metaRows.push(metaRow('Location', escapeHtml(deal.item_location) + flag));
    }
    if (deal.is_trending) metaRows.push(metaRow('Trending', '🔥 Yes'));

    // For skipped view, show skipped date
    if (renderMode === 'skipped' && deal.skipped_at) {
        const skippedDate = new Date(deal.skipped_at * 1000).toLocaleDateString();
        metaRows.push(metaRow('Skipped', escapeHtml(skippedDate)));
    }

    const ageHtml = buildListingAgeBadge(deal.listing_date);

    const encodedUrl = escapeHtml(deal.url);
    const isSelected = _selectedUrls.has(deal.url);
    const isChecked  = isSelected ? 'checked' : '';

    // Action buttons based on mode
    let actionsHtml = '';
    if (renderMode === 'skipped') {
        actionsHtml = `<div class="deal-actions">
            <button class="btn-deal-action btn-restore" data-action="unskip"
                    data-url="${encodedUrl}" title="Restore this deal">↩ Restore</button>
        </div>`;
    } else if (renderMode === 'saved') {
        actionsHtml = `<div class="deal-actions">
            <button class="btn-deal-action btn-unsave" data-action="unsave"
                    data-url="${encodedUrl}" title="Remove from saved">✕ Remove</button>
        </div>`;
    } else {
        // ready mode
        const isSaved    = deal.is_saved || _savedUrls.has(deal.url);
        const saveLabel  = isSaved ? '★ Saved' : '☆ Save';
        const saveCls    = isSaved ? 'btn-deal-action btn-save btn-save--saved' : 'btn-deal-action btn-save';
        const saveAction = isSaved ? 'unsave' : 'save';
        actionsHtml = `<div class="deal-actions">
            <button class="${saveCls}" data-action="${saveAction}"
                    data-url="${encodedUrl}"
                    data-title="${escapeHtml(deal.title || '')}"
                    data-price="${deal.price || 0}"
                    title="${isSaved ? 'Remove from saved' : 'Save this deal'}">${saveLabel}</button>
            <button class="btn-deal-action btn-skip" data-action="skip"
                    data-url="${encodedUrl}"
                    data-title="${escapeHtml(deal.title || '')}"
                    data-price="${deal.price || 0}"
                    title="Hide this deal from future searches">✕ Skip</button>
        </div>`;
    }

    // Scores breakdown (only for deals with full data)
    let scoresHtml = '';
    if (deal.overall_score != null) {
        scoresHtml = `<div class="scores-breakdown">
            <div class="score-row"><span>Price</span><span>${priceScore.toFixed(0)}</span></div>
            <div class="score-bar"><div class="score-fill" style="width:${priceScore}%"></div></div>
            <div class="score-row"><span>Seller</span><span>${sellerScore.toFixed(0)}</span></div>
            <div class="score-bar"><div class="score-fill" style="width:${sellerScore}%"></div></div>
            <div class="score-row"><span>Condition</span><span>${conditionScore.toFixed(0)}</span></div>
            <div class="score-bar"><div class="score-fill" style="width:${conditionScore}%"></div></div>
            <div class="score-row"><span>Trend</span><span>${trendScore.toFixed(0)}</span></div>
            <div class="score-bar"><div class="score-fill" style="width:${trendScore}%"></div></div>
        </div>`;
    }

    // Deal header: only show the rules-based score and recommendation when they
    // carry real data (overall_score is set by the legacy rules engine).  After
    // the rules engine was removed, these fields are no longer populated and
    // would otherwise display a confusing "0.0 / N/A" placeholder.
    const headerScoreHtml = (deal.overall_score != null)
        ? `<div class="deal-score" style="color:${scoreColor}">${score.toFixed(1)}</div>`
        : '';
    const headerRecommendationHtml = (deal.overall_score != null)
        ? `<div class="deal-recommendation">${escapeHtml(recommendation)}</div>`
        : '';

    return `<div class="deal-card${isSelected ? ' selected' : ''}" data-url="${encodedUrl}">
        <div class="deal-card-select">
            <input type="checkbox" class="deal-checkbox" data-url="${encodedUrl}" ${isChecked} aria-label="Select deal">
        </div>
        <div class="deal-header">
            ${headerScoreHtml}
            <div class="deal-header-right">
                ${headerRecommendationHtml}
                ${ageHtml}
            </div>
        </div>
        ${imageSection}
        <div class="deal-body">
            <div class="deal-title">${escapeHtml(deal.title || '(no title)')}</div>
            <div class="deal-price">€${(deal.price || 0).toFixed(2)}</div>
            <div class="deal-meta">${metaRows.join('')}</div>
            ${imageWarningSection}
            ${aiSection}
            ${scoresHtml}
        </div>
        <div class="deal-footer">
            <a href="${encodedUrl}" target="_blank" rel="noopener noreferrer" class="btn-view">View on eBay →</a>
            ${actionsHtml}
        </div>
    </div>`;
}

function metaRow(label, value) {
    return `<div class="meta-row"><span class="meta-label">${label}</span><span class="meta-value">${value}</span></div>`;
}

function buildListingAgeBadge(listingDate) {
    if (!listingDate) return '';
    const listed = new Date(listingDate);
    if (isNaN(listed)) return '';
    const ageDays = Math.floor((Date.now() - listed) / (86400 * 1000));
    let label, cls;
    if (ageDays === 0) {
        label = 'Today'; cls = 'deal-age-badge deal-age-badge--fresh';
    } else if (ageDays === 1) {
        label = '1 day old'; cls = 'deal-age-badge deal-age-badge--fresh';
    } else if (ageDays <= 7) {
        label = `${ageDays} days old`; cls = 'deal-age-badge deal-age-badge--fresh';
    } else if (ageDays <= 30) {
        label = `${ageDays} days old`; cls = 'deal-age-badge deal-age-badge--recent';
    } else {
        label = ageDays >= 365 ? `${Math.floor(ageDays / 365)}y old` : `${ageDays} days old`;
        cls = 'deal-age-badge deal-age-badge--old';
    }
    return `<span class="${cls}" title="Listed: ${listed.toLocaleDateString()}">${label}</span>`;
}

// ---------------------------------------------------------------------------
// Deal card event delegation
// ---------------------------------------------------------------------------

async function handleDealCardAction(e) {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;

    const action = btn.dataset.action;
    const url    = btn.dataset.url;
    if (!url) return;

    if (action === 'save')   await handleSaveDeal(btn, url);
    if (action === 'unsave') await handleUnsaveDeal(btn, url);
    if (action === 'skip')   await handleSkipDeal(btn, url);
    if (action === 'unskip') await handleUnskipDeal(btn, url);
}

function handleDealCardCheckbox(e) {
    if (!e.target.classList.contains('deal-checkbox')) return;
    const url = e.target.dataset.url;
    if (!url) return;

    const card = e.target.closest('.deal-card');
    if (e.target.checked) {
        _selectedUrls.add(url);
        card && card.classList.add('selected');
    } else {
        _selectedUrls.delete(url);
        card && card.classList.remove('selected');
    }
    updateBatchActionsBar();
}

// ---------------------------------------------------------------------------
// Batch selection
// ---------------------------------------------------------------------------

function toggleSelectAll() {
    const checkboxes = document.querySelectorAll('#dealsGrid .deal-checkbox');
    const anyUnchecked = Array.from(checkboxes).some(cb => !cb.checked);

    checkboxes.forEach(cb => {
        const url  = cb.dataset.url;
        const card = cb.closest('.deal-card');
        if (anyUnchecked) {
            cb.checked = true;
            _selectedUrls.add(url);
            card && card.classList.add('selected');
        } else {
            cb.checked = false;
            _selectedUrls.delete(url);
            card && card.classList.remove('selected');
        }
    });
    updateBatchActionsBar();
}

function clearSelection() {
    _selectedUrls.clear();
    document.querySelectorAll('#dealsGrid .deal-checkbox').forEach(cb => {
        cb.checked = false;
        const card = cb.closest('.deal-card');
        card && card.classList.remove('selected');
    });
    updateBatchActionsBar();
}

function updateBatchActionsBar() {
    const batchEl  = document.getElementById('batchActions');
    const countEl  = document.getElementById('selectedCount');
    const count    = _selectedUrls.size;

    if (count > 0) {
        batchEl && batchEl.classList.remove('d-none');
        countEl && (countEl.textContent = `${count} selected`);
    } else {
        batchEl && batchEl.classList.add('d-none');
    }
}

/**
 * Extract { title, price } from a deal card DOM element.
 * @param {Element|null} card
 * @returns {{ title: string, price: number }}
 */
function _dealDataFromCard(card) {
    if (!card) return { title: '', price: 0 };
    const title = card.querySelector('.deal-title')?.textContent?.trim() || '';
    const price = parseFloat(card.querySelector('.deal-price')?.textContent?.replace('€', '') ?? '0') || 0;
    return { title, price };
}

async function handleBatchSave() {
    const urls = Array.from(_selectedUrls);
    if (!urls.length) return;

    const cards = document.querySelectorAll('#dealsGrid .deal-card');
    const cardsByUrl = {};
    cards.forEach(c => { cardsByUrl[c.dataset.url] = c; });

    for (const url of urls) {
        const card  = cardsByUrl[url];
        const { title, price } = _dealDataFromCard(card);
        try {
            await fetch('/api/deals/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url, title, price }),
            });
            _savedUrls.add(url);
            // Update save button state in card
            const saveBtn = card && card.querySelector('[data-action="save"]');
            if (saveBtn) {
                saveBtn.textContent   = '★ Saved';
                saveBtn.className     = 'btn-deal-action btn-save btn-save--saved';
                saveBtn.dataset.action = 'unsave';
                saveBtn.title         = 'Remove from saved';
            }
        } catch (_) { /* best-effort */ }
    }
    clearSelection();
    updateTabBadges();
}

async function handleBatchSkip() {
    const urls = Array.from(_selectedUrls);
    if (!urls.length) return;

    const cards = document.querySelectorAll('#dealsGrid .deal-card');
    const cardsByUrl = {};
    cards.forEach(c => { cardsByUrl[c.dataset.url] = c; });

    for (const url of urls) {
        const card  = cardsByUrl[url];
        const { title, price } = _dealDataFromCard(card);
        try {
            await fetch('/api/deals/skip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url, title, price }),
            });
            _lastDeals = _lastDeals.filter(d => d.url !== url);
            if (card) {
                card.style.transition = 'opacity 0.2s ease';
                card.style.opacity    = '0';
                setTimeout(() => card.remove(), 220);
            }
        } catch (_) { /* best-effort */ }
    }
    _selectedUrls.clear();
    updateBatchActionsBar();
    updateTabBadges();
}

// ---------------------------------------------------------------------------
// Individual save / skip / unskip
// ---------------------------------------------------------------------------

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
            btn.textContent    = '★ Saved';
            btn.className      = 'btn-deal-action btn-save btn-save--saved';
            btn.dataset.action = 'unsave';
            btn.title          = 'Remove from saved';
            updateTabBadges();
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
            if (_currentPipeline === 'saved') {
                // Remove card from saved view
                const card = btn.closest('.deal-card');
                if (card) {
                    card.style.transition = 'opacity 0.2s ease';
                    card.style.opacity    = '0';
                    setTimeout(() => card.remove(), 220);
                }
            } else {
                btn.textContent    = '☆ Save';
                btn.className      = 'btn-deal-action btn-save';
                btn.dataset.action = 'save';
                btn.title          = 'Save this deal';
            }
            updateTabBadges();
        }
    } catch (err) {
        console.error('Unsave deal error:', err);
    } finally {
        btn.disabled = false;
    }
}

async function handleSkipDeal(btn, url) {
    btn.disabled = true;
    const title = btn.dataset.title || '';
    const price = parseFloat(btn.dataset.price) || 0;
    try {
        const resp = await fetch('/api/deals/skip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url, title, price }),
        });
        if (resp.ok) {
            const card = btn.closest('.deal-card');
            if (card) {
                card.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
                card.style.opacity    = '0';
                card.style.transform  = 'scale(0.95)';
                setTimeout(() => card.remove(), 270);
            }
            _lastDeals = _lastDeals.filter(d => d.url !== url);
            updateTabBadges();
        }
    } catch (err) {
        console.error('Skip deal error:', err);
        btn.disabled = false;
    }
}

async function handleUnskipDeal(btn, url) {
    btn.disabled = true;
    try {
        const resp = await fetch('/api/deals/unskip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url }),
        });
        if (resp.ok) {
            const card = btn.closest('.deal-card');
            if (card) {
                card.style.transition = 'opacity 0.2s ease';
                card.style.opacity    = '0';
                setTimeout(() => card.remove(), 220);
            }
            updateTabBadges();
        }
    } catch (err) {
        console.error('Unskip deal error:', err);
        btn.disabled = false;
    }
}

// ---------------------------------------------------------------------------
// Saved URLs cache
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

// ---------------------------------------------------------------------------
// History view
// ---------------------------------------------------------------------------

async function loadHistoryView() {
    const historyList = document.getElementById('historyList');
    const countBadge  = document.getElementById('historyCountBadge');
    if (!historyList) return;

    historyList.innerHTML = '<div class="history-loading">⏳ Loading history…</div>';

    try {
        const resp = await fetch('/api/history?limit=50');
        if (!resp.ok) throw new Error('Failed to load history');
        const searches = await resp.json();

        if (countBadge) countBadge.textContent = `${searches.length} search${searches.length !== 1 ? 'es' : ''}`;

        if (!searches.length) {
            historyList.innerHTML = '<div class="history-empty">📭 No search history yet.</div>';
            return;
        }

        historyList.innerHTML = searches.map(s => buildHistoryEntry(s)).join('');

        // Wire click handlers for expand/collapse
        historyList.querySelectorAll('.history-entry-header').forEach(header => {
            header.addEventListener('click', () => toggleHistoryEntry(header.parentElement));
        });

    } catch (err) {
        historyList.innerHTML = '<div class="history-empty">⚠️ Could not load search history.</div>';
    }
}

function buildHistoryEntry(search) {
    const date = new Date(search.created_at * 1000);
    const dateStr = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    const timeStr = date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });

    return `<div class="history-entry" data-search-id="${search.id}">
        <div class="history-entry-header">
            <span class="history-toggle-icon">▶</span>
            <span class="history-query">${escapeHtml(search.query)}</span>
            <span class="history-count">${search.result_count} deal${search.result_count !== 1 ? 's' : ''}</span>
            <span class="history-date">${dateStr} ${timeStr}</span>
        </div>
        <div class="history-entry-deals d-none">
            <div class="history-loading">⏳ Loading deals…</div>
        </div>
    </div>`;
}

async function toggleHistoryEntry(entry) {
    const dealsEl = entry.querySelector('.history-entry-deals');
    if (!dealsEl) return;

    const isExpanded = entry.classList.contains('expanded');

    if (isExpanded) {
        entry.classList.remove('expanded');
        dealsEl.classList.add('d-none');
    } else {
        entry.classList.add('expanded');
        dealsEl.classList.remove('d-none');

        // Load deals if not already loaded
        if (dealsEl.querySelector('.history-loading')) {
            const searchId = entry.dataset.searchId;
            try {
                const resp = await fetch(`/api/deals/${searchId}`);
                if (!resp.ok) throw new Error();
                const deals = await resp.json();
                if (deals.length === 0) {
                    dealsEl.innerHTML = '<div class="history-empty">No deals recorded.</div>';
                } else {
                    dealsEl.innerHTML = `<div class="history-deals-grid">${deals.map(d => createDealCard(d, 'ready')).join('')}</div>`;
                    // Wire up deal card actions in history
                    dealsEl.addEventListener('click', handleDealCardAction);
                    dealsEl.addEventListener('change', handleDealCardCheckbox);
                }
            } catch (_) {
                dealsEl.innerHTML = '<div class="history-empty">⚠️ Could not load deals.</div>';
            }
        }
    }
}

function filterHistoryList(term) {
    const entries = document.querySelectorAll('#historyList .history-entry');
    let visible = 0;
    entries.forEach(entry => {
        const query = entry.querySelector('.history-query')?.textContent?.toLowerCase() || '';
        if (!term || query.includes(term)) {
            entry.style.display = '';
            visible++;
        } else {
            entry.style.display = 'none';
        }
    });
    const badge = document.getElementById('historyCountBadge');
    if (badge) badge.textContent = `${visible} search${visible !== 1 ? 'es' : ''}`;
}

// ---------------------------------------------------------------------------
// Settings: Gemini model, AI toggle, data source
// ---------------------------------------------------------------------------

async function loadModelSettings() {
    try {
        const response = await fetch('/api/settings');
        if (!response.ok) return;
        const data = await response.json();
        const input  = document.getElementById('geminiModelInput');
        const status = document.getElementById('modelStatus');
        if (input && data.gemini_model) input.value = data.gemini_model;
        if (status && data.gemini_model) {
            status.textContent = `Active model: ${data.gemini_model}`;
            status.className   = 'model-status model-status--active';
        }
        if (typeof data.ai_enabled === 'boolean') _setAiToggleState(data.ai_enabled);
        _setDataSourceState(data.data_source, data.active_data_source, data.ebay_api_configured);
    } catch (err) {
        console.warn('Failed to load model settings:', err);
    }
}

function _setAiToggleState(enabled) {
    const btn = document.getElementById('aiToggleBtn');
    if (!btn) return;
    btn.setAttribute('aria-pressed', String(enabled));
    const label = btn.querySelector('.ai-toggle-label');
    const icon  = btn.querySelector('.ai-toggle-icon');
    if (label) label.textContent = enabled ? 'AI: ON' : 'AI: OFF';
    if (icon)  icon.textContent  = enabled ? '✨' : '⭕';
}

async function toggleAiEnabled() {
    const btn = document.getElementById('aiToggleBtn');
    if (!btn) return;
    const currentlyEnabled = btn.getAttribute('aria-pressed') === 'true';
    const newState = !currentlyEnabled;
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
            _setAiToggleState(currentlyEnabled);
            const status = document.getElementById('modelStatus');
            if (status) { status.textContent = `⚠️ ${data.error || 'Failed to save.'}`; status.className = 'model-status model-status--error'; }
        } else {
            _setAiToggleState(data.ai_enabled);
        }
    } catch (err) {
        _setAiToggleState(currentlyEnabled);
        const status = document.getElementById('modelStatus');
        if (status) { status.textContent = `⚠️ Error: ${err.message}`; status.className = 'model-status model-status--error'; }
    } finally {
        btn.disabled = false;
    }
}

function _setDataSourceState(setting, active, apiConfigured) {
    const sel      = document.getElementById('dataSourceSelect');
    const statusEl = document.getElementById('dataSourceStatus');
    if (sel && setting) sel.value = setting;
    if (statusEl) {
        if (active === 'api') {
            statusEl.textContent = '🟢 eBay API active';
            statusEl.className   = 'data-source-status data-source-status--api';
        } else if (active === 'scraper') {
            const hint = (!apiConfigured && setting !== 'scraper') ? ' (API creds not set)' : '';
            statusEl.textContent = `🔵 Scraper active${hint}`;
            statusEl.className   = 'data-source-status data-source-status--scraper';
        } else {
            statusEl.textContent = '';
            statusEl.className   = 'data-source-status';
        }
    }
}

async function saveDataSource() {
    const sel      = document.getElementById('dataSourceSelect');
    const statusEl = document.getElementById('dataSourceStatus');
    if (!sel) return;
    const newSource = sel.value;
    if (statusEl) { statusEl.textContent = '⏳ Saving…'; statusEl.className = 'data-source-status'; }
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data_source: newSource }),
        });
        const data = await response.json();
        if (!response.ok) {
            const msg = (data.errors && data.errors.data_source) || data.error || 'Failed to save.';
            if (statusEl) { statusEl.textContent = `⚠️ ${msg}`; statusEl.className = 'data-source-status data-source-status--error'; }
        } else {
            _setDataSourceState(data.data_source, data.active_data_source, data.ebay_api_configured);
        }
    } catch (err) {
        if (statusEl) { statusEl.textContent = `⚠️ Error: ${err.message}`; statusEl.className = 'data-source-status data-source-status--error'; }
    }
}

async function saveModelSettings() {
    const input  = document.getElementById('geminiModelInput');
    const status = document.getElementById('modelStatus');
    const btn    = document.getElementById('saveModelBtn');
    if (!input || !status || !btn) return;

    const model = input.value.trim();
    if (!model) {
        status.textContent = '⚠️ Please enter a model name.';
        status.className   = 'model-status model-status--error';
        return;
    }

    btn.disabled    = true;
    btn.textContent = 'Saving…';
    status.textContent = '';
    status.className   = 'model-status';

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
            status.className   = 'model-status model-status--error';
        } else {
            status.textContent = `✅ Active model: ${data.gemini_model}`;
            status.className   = 'model-status model-status--active';
        }
    } catch (err) {
        status.textContent = `⚠️ Error: ${err.message}`;
        status.className   = 'model-status model-status--error';
    } finally {
        btn.disabled    = false;
        btn.textContent = 'Save';
    }
}

// ---------------------------------------------------------------------------
// Condition / shipping helpers (preserved from original)
// ---------------------------------------------------------------------------

function parseConditionParts(conditionText) {
    const result = { condition: conditionText, sellerType: '', device: '' };
    if (!conditionText || !conditionText.includes('|')) return result;

    const parts     = conditionText.split('|').map(p => p.trim()).filter(Boolean);
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

    if (remaining.length > 0) result.condition = remaining[0];
    if (remaining.length > 1) result.device    = remaining.slice(1).join(' | ');
    return result;
}

function parseGermanNumber(s) {
    const clean = s.trim();
    if (clean.includes(',') && clean.includes('.')) {
        const lastComma = clean.lastIndexOf(',');
        const lastDot   = clean.lastIndexOf('.');
        if (lastComma > lastDot) return parseFloat(clean.replace(/\./g, '').replace(',', '.')) || 0;
        return parseFloat(clean.replace(/,/g, '')) || 0;
    }
    if (clean.includes(',')) return parseFloat(clean.replace(',', '.')) || 0;
    return parseFloat(clean) || 0;
}

function cleanShippingText(shippingText) {
    if (!shippingText) return 'N/A';
    const lower = shippingText.toLowerCase();
    if (lower.includes('nicht angegeben')) return 'N/A';
    if (lower.includes('kostenlos') || lower.includes('gratis') ||
        lower.includes('free shipping') || lower.includes('free postage')) return 'Free';

    const rangeMatch = shippingText.match(/EUR\s*([\d,.]+)\s+bis\s+EUR\s*([\d,.]+)/i);
    if (rangeMatch) {
        return `€${parseGermanNumber(rangeMatch[1]).toFixed(2)} – €${parseGermanNumber(rangeMatch[2]).toFixed(2)}`;
    }
    const singleEurMatch = shippingText.match(/EUR\s*([\d,.]+)/i);
    if (singleEurMatch) return `€${parseGermanNumber(singleEurMatch[1]).toFixed(2)}`;

    const euroMatch = shippingText.match(/€\s*([\d,.]+)/);
    if (euroMatch) return `€${parseGermanNumber(euroMatch[1]).toFixed(2)}`;

    return shippingText.length > 55 ? shippingText.slice(0, 55).trimEnd() + '…' : shippingText;
}

// ---------------------------------------------------------------------------
// AI verdict sections
// ---------------------------------------------------------------------------

function buildImageIssueSection(issues) {
    if (!issues || issues.length === 0) return '';
    const issueLabels = { 'no_images': '📷 No product images available' };
    const items = issues.map(issue => {
        const label = issueLabels[issue] || escapeHtml(issue);
        return `<span class="image-issue-badge">${label}</span>`;
    }).join('');
    return `<div class="image-issues">${items}</div>`;
}

function buildAiSection(deal) {
    const rating         = deal.ai_deal_rating || 'Unknown';
    const confidence     = deal.ai_confidence_score || 0;
    const summary        = deal.ai_verdict_summary || '';
    const estimate       = deal.ai_fair_market_estimate || '';
    const visualFindings = Array.isArray(deal.ai_visual_findings) ? deal.ai_visual_findings : [];
    const redFlags       = Array.isArray(deal.ai_red_flags) ? deal.ai_red_flags : [];
    const potentialScam  = !!deal.ai_potential_scam;
    const scamWarning    = deal.ai_scam_warning || '';
    const itemized       = Array.isArray(deal.ai_itemized_resale_estimates) ? deal.ai_itemized_resale_estimates : [];
    const totalCost      = deal.ai_estimated_total_cost || 0;
    const grossProfit    = deal.ai_estimated_gross_profit || 0;

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

    // Build itemized resale breakdown table when available.
    let itemizedHtml = '';
    if (itemized.length > 0) {
        const sourceLabel = src => {
            if (src === 'ebay_sold') return '<span class="price-source price-source-sold" title="Based on eBay sold/completed listings">📊 eBay sold</span>';
            if (src === 'ebay_active') return '<span class="price-source price-source-active" title="Based on eBay active listings (proxy)">🔍 eBay active</span>';
            if (src === 'no_result') return '<span class="price-source price-source-none" title="No eBay data found">❓ no data</span>';
            return '<span class="price-source price-source-ai" title="AI estimate (no eBay data)">🤖 AI est.</span>';
        };
        const rows = itemized.map(item => {
            const priceStr = (item.price_eur != null) ? `€${Number(item.price_eur).toFixed(2)}` : '—';
            return `<tr><td>${escapeHtml(item.game || '?')}</td><td class="price-cell">${escapeHtml(priceStr)}</td><td>${sourceLabel(item.price_source)}</td></tr>`;
        }).join('');
        const totalResale = itemized.reduce((s, i) => s + (i.price_eur || 0), 0);
        const profitSign = grossProfit >= 0 ? '+' : '';
        const profitClass = grossProfit >= 0 ? 'profit-positive' : 'profit-negative';
        itemizedHtml = `
        <div class="ai-itemized">
            <div class="ai-itemized-header">🎮 Per-game resale breakdown</div>
            <table class="ai-itemized-table">
                <thead><tr><th>Game</th><th>Est. price</th><th>Source</th></tr></thead>
                <tbody>${rows}</tbody>
                <tfoot>
                    <tr class="itemized-total">
                        <td><strong>Total resale</strong></td>
                        <td class="price-cell"><strong>€${totalResale.toFixed(2)}</strong></td>
                        <td></td>
                    </tr>
                    ${totalCost > 0 ? `<tr class="itemized-cost"><td>Total cost (buy + ship)</td><td class="price-cell">€${Number(totalCost).toFixed(2)}</td><td></td></tr>` : ''}
                    ${totalCost > 0 ? `<tr class="itemized-profit ${profitClass}"><td><strong>Est. gross profit</strong></td><td class="price-cell"><strong>${profitSign}€${Number(grossProfit).toFixed(2)}</strong></td><td></td></tr>` : ''}
                </tfoot>
            </table>
        </div>`;
    }

    return `<div class="ai-verdict${potentialScam ? ' ai-verdict-scam' : ''}">
        ${scamBannerHtml}
        <div class="ai-verdict-header">
            <span class="ai-badge ${badgeClass}">${escapeHtml(rating)}</span>
            <span class="ai-confidence">AI Confidence: ${confidence}%</span>
        </div>
        ${summary  ? `<p class="ai-summary">${escapeHtml(summary)}</p>` : ''}
        ${estimateHtml}
        ${itemizedHtml}
        ${findingsHtml}
        ${redFlagsHtml}
    </div>`;
}

function buildAiErrorSection(message) {
    return `<div class="ai-verdict ai-verdict-error">
        <div class="ai-verdict-header">
            <span class="ai-badge badge-unknown">${escapeHtml(message)}</span>
        </div>
    </div>`;
}

function getAiBadgeClass(rating) {
    const r = (rating || '').toLowerCase();
    if (r.includes('must')) return 'badge-must-buy';
    if (r === 'good') return 'badge-good';
    if (r === 'okay' || r === 'fair') return 'badge-okay';
    if (r.includes('avoid') || r.includes('hard pass')) return 'badge-avoid';
    return 'badge-unknown';
}

function getScoreColor(score) {
    if (score >= 85) return '#3fb950';
    if (score >= 70) return '#e3b341';
    if (score >= 50) return '#f0883e';
    if (score >= 30) return '#f85149';
    return '#8b949e';
}

// ---------------------------------------------------------------------------
// Error display
// ---------------------------------------------------------------------------

function showError(message) {
    const el = document.getElementById('errorContainer');
    if (!el) return;
    el.textContent = message;
    el.classList.remove('d-none');
}

function showDetailedError(summary, details) {
    const el = document.getElementById('errorContainer');
    if (!el) return;
    const detailsHtml = details.map(d => `<li><pre class="error-detail">${escapeHtml(d)}</pre></li>`).join('');
    el.innerHTML = `<strong>${escapeHtml(summary)}</strong><ul class="error-details-list">${detailsHtml}</ul>`;
    el.classList.remove('d-none');
}

// ---------------------------------------------------------------------------
// CSV export
// ---------------------------------------------------------------------------

async function exportToCSV() {
    try {
        const response = await fetch('/api/export');
        if (!response.ok) throw new Error('Export failed');
        const blob = await response.blob();
        const url  = window.URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = `ebay_deals_${new Date().toISOString().slice(0, 10)}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    } catch (error) {
        console.error('Export error:', error);
        showError('Failed to export data');
    }
}

// ---------------------------------------------------------------------------
// HTML escape utility
// ---------------------------------------------------------------------------

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}
