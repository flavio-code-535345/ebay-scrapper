/* Sell page — one game from barcode to eBay listing.
 *
 * Flow: scan/type EAN → catalog product + cheapest comparable offers (price
 * suggestion) → condition → photos (resized here, uploaded in the background
 * while you keep going) → Publish → "Next game".
 */
'use strict';

const $ = (id) => document.getElementById(id);

// Used when eBay's condition list for the category can't be loaded
// (eBay's list for Videospiele, 139973, as returned in September 2026).
const FALLBACK_CONDITIONS = [
    { id: '1000', name: 'Neu' },
    { id: '2750', name: 'Neuwertig' },
    { id: '4000', name: 'Sehr gut' },
    { id: '5000', name: 'Gut' },
    { id: '6000', name: 'Akzeptabel' },
];
const MAX_PHOTO_EDGE = 1600;
const ZXING_URL = 'https://cdn.jsdelivr.net/npm/@zxing/library@0.23.0/umd/index.min.js';

const state = {
    status: null,
    ean: '',
    product: null, // {epid, title, ean, image, platform, game} or null when eBay's catalog doesn't know it
    suggested: null,
    second: null, // {title, suggested} for two-game sets
    addingSecond: false,
    conditionId: '',
    photos: [], // {key, preview, url, state: 'uploading'|'done'|'error', error}
    uploadChain: Promise.resolve(),
    scanner: null,
};

// ── Helpers ─────────────────────────────────────────────────────────────────

async function api(path, options = {}) {
    const resp = await fetch(path, options);
    let body = {};
    try { body = await resp.json(); } catch (_) { /* non-JSON error page */ }
    if (resp.status === 401) throw new Error('Signed out — reload the page and sign in again.');
    if (!resp.ok) throw new Error(body.error || `HTTP ${resp.status}`);
    return body;
}

function postJson(path, data) {
    return api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
}

const euro = (n) => (n == null || isNaN(n) ? '' : `€${Number(n).toFixed(2)}`);
const parsePrice = (text) => parseFloat(String(text || '').replace(/\s|€/g, '').replace(',', '.'));
const withoutPlatform = (title) => String(title || '').replace(/\s*\([^()]*\)\s*$/, '').trim();
const platformOf = (title) => (String(title || '').match(/\(([^()]*)\)\s*$/) || [])[1] || '';
const isEan = (text) => /^\d{8,14}$/.test(String(text || '').replace(/\s/g, ''));

function show(el, visible) { el.hidden = !visible; }

function setText(id, text) { $(id).textContent = text || ''; }

// ── Status, connection, template ────────────────────────────────────────────

async function loadStatus() {
    try {
        state.status = await api('/api/sell/status');
    } catch (err) {
        setText('sellLoading', err.message);
        return;
    }
    show($('sellLoading'), false);
    const s = state.status;
    show($('connectCard'), !s.connected);
    show($('templateCard'), s.connected);
    show($('listingCard'), s.connected && !!s.template);
    setText('accountName', s.username ? `eBay: ${s.username}` : '');
    renderTemplate(s.template);
    renderAllowance(s.free_listings);
    renderConditions();
    renderRecent(s.recent || []);
}

// eBay.de: 320 new listings a month without an insertion fee, then €0.50 each.
function renderAllowance(usage) {
    const el = $('allowance');
    show(el, !!usage);
    if (!usage) return;
    const over = usage.used >= usage.allowance;
    el.textContent = `Free listings this month: ${usage.used} of ${usage.allowance} used`
        + (over ? ' — each further new listing costs €0.50 until next month.' : '.');
    el.classList.toggle('sell-warning', over);
}

function allowanceNote() {
    const usage = state.status && state.status.free_listings;
    return usage ? `${usage.used} of ${usage.allowance} free listings used this month` : '';
}

function renderTemplate(t) {
    const box = $('templateSummary');
    box.innerHTML = '';
    $('templateForm').open = !t;
    setText('templateFormSummary', t ? 'Copy from a different listing' : 'Copy settings from one of your listings');
    if (!t) {
        box.textContent = 'No settings yet — copy them from one of your listings below.';
        return;
    }
    const rows = [
        ['From', `<a href="https://www.ebay.de/itm/${encodeURIComponent(t.source_item_id)}" target="_blank" rel="noopener noreferrer">${escapeHtml(t.title || t.source_item_id)}</a>`],
        ['Shipping', `${escapeHtml(t.shipping_service || 'business policy')}${t.shipping_cost != null ? ' · ' + euro(t.shipping_cost) : ''}`],
        ['Returns', escapeHtml(t.business_policies ? 'business policy' : (t.returns || '—'))],
        ['Location', escapeHtml(t.location || '—')],
        ['Description', escapeHtml(t.description_preview || '—')],
    ];
    box.innerHTML = rows.map(([k, v]) => `<div><span>${k}</span><span>${v}</span></div>`).join('');
    $('bestOffer').checked = !!t.best_offer;
}

function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

async function importTemplate() {
    const item = $('templateItem').value.trim();
    if (!item) return;
    $('templateBtn').disabled = true;
    try {
        const { template } = await postJson('/api/sell/template', { item });
        state.status.template = template;
        $('templateItem').value = '';
        await loadStatus();
    } catch (err) {
        alert(err.message);
    } finally {
        $('templateBtn').disabled = false;
    }
}

async function connectFromPastedUrl() {
    try {
        await postJson('/api/sell/ebay/code', { url: $('pastedAuthUrl').value.trim() });
        await loadStatus();
    } catch (err) {
        alert(err.message);
    }
}

function renderRecent(listings) {
    show($('recentCard'), listings.length > 0);
    $('recent').innerHTML = listings.map((l) => `
        <li><a href="${escapeHtml(l.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(l.title)}</a>
            <span>${euro(l.price)} · ${new Date(l.created_at * 1000).toLocaleString()}</span></li>`).join('');
}

// ── Product lookup and price ────────────────────────────────────────────────

async function lookup(query) {
    query = String(query || '').trim();
    if (!query) return;
    setText('lookupStatus', 'Looking up…');
    $('productChoices').innerHTML = '';
    let result;
    try {
        result = await api('/api/sell/product?q=' + encodeURIComponent(query));
    } catch (err) {
        setText('lookupStatus', err.message);
        return;
    }
    const products = result.products || [];
    // A title search can match several products: let the seller pick one.
    if (!isEan(query) && products.length > 1) {
        setText('lookupStatus', 'Which one?');
        $('productChoices').innerHTML = products.map((p, i) =>
            `<button type="button" class="sell-choice" data-i="${i}">${escapeHtml(p.title)}</button>`).join('');
        $('productChoices').querySelectorAll('button').forEach((btn) => btn.addEventListener('click', () => {
            const p = products[Number(btn.dataset.i)];
            if (p.ean) lookup(p.ean); else useProduct(p, result);
        }));
        return;
    }
    useProduct(products[0] || null, result);
}

function useProduct(product, result) {
    setText('lookupStatus', result.errors && result.errors.length ? result.errors.join(' · ') : '');
    $('productChoices').innerHTML = '';
    if (state.addingSecond) {
        addSecondGame(product, result);
        return;
    }
    state.product = product;
    state.ean = (product && product.ean) || result.ean || '';
    state.suggested = result.suggested_price;
    show($('draft'), true);
    $('titleInput').value = product ? product.title : '';
    updateTitleCount();
    const img = $('productImage');
    img.src = product && product.image ? product.image : '';
    show(img, !!(product && product.image));
    setText('productMeta', product
        ? [product.platform, state.ean && `EAN ${state.ean}`].filter(Boolean).join(' · ')
        : 'Not in eBay\'s catalog — enter the title and platform yourself.');
    show($('platformRow'), !product);
    show($('addGameBtn'), true);
    renderOffers(result.offers || [], result.shipping_cost);
    $('priceInput').value = state.suggested != null ? state.suggested.toFixed(2).replace('.', ',') : '';
    updateBuyerTotal();
    if (!product) $('titleInput').focus();
    $('draft').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function addSecondGame(product, result) {
    state.addingSecond = false;
    $('lookupInput').placeholder = 'EAN or title';
    if (!product) {
        setText('lookupStatus', 'Second game not found in eBay\'s catalog — edit the title yourself.');
        return;
    }
    state.second = { title: product.title, suggested: result.suggested_price };
    const platform = platformOf($('titleInput').value) || product.platform;
    const suffix = platform ? ` (${platform})` : '';
    const first = withoutPlatform($('titleInput').value);
    const second = withoutPlatform(product.title);
    let combined = `${first} + ${second}${suffix}`;
    if (combined.length > 80) {
        // eBay allows 80 characters: drop the subtitles ("Uncharted 2: Among Thieves" → "Uncharted 2").
        const short = (name) => name.split(/:| - /)[0].trim();
        combined = `${short(first)} + ${short(second)}${suffix}`;
    }
    $('titleInput').value = combined;
    updateTitleCount();
    if (state.suggested != null && result.suggested_price != null) {
        $('priceInput').value = (state.suggested + result.suggested_price).toFixed(2).replace('.', ',');
        updateBuyerTotal();
    }
    setText('productMeta', `${$('productMeta').textContent} · set with ${product.title}`);
    show($('addGameBtn'), false);
}

function renderOffers(offers, shippingCost) {
    state.shippingCost = shippingCost;
    const list = $('offers');
    if (!offers.length) {
        list.innerHTML = `<li class="sell-hint">${$('sellApp').dataset.prices === 'on'
            ? 'No comparable Buy-It-Now offers found — set the price yourself.'
            : 'Price lookup needs EBAY_CLIENT_ID/SECRET.'}</li>`;
        return;
    }
    list.innerHTML = offers.map((o) => `
        <li><a href="${escapeHtml(o.url)}" target="_blank" rel="noopener noreferrer">
            <strong>${euro(o.total)}</strong>
            <span>${euro(o.price)} + ${euro(o.shipping)} shipping · ${escapeHtml(o.condition || '')} · ${escapeHtml(o.seller || '')}</span>
        </a></li>`).join('');
}

function updateBuyerTotal() {
    const price = parsePrice($('priceInput').value);
    const ship = state.shippingCost;
    setText('buyerTotal', isNaN(price) ? '' : (ship != null ? `+ ${euro(ship)} shipping = ${euro(price + ship)}` : ''));
    updatePublishState();
}

function updateTitleCount() {
    setText('titleCount', `${$('titleInput').value.length}/80`);
    updatePublishState();
}

// ── Condition ───────────────────────────────────────────────────────────────

function renderConditions() {
    const s = state.status || {};
    const conditions = (s.conditions && s.conditions.length) ? s.conditions : FALLBACK_CONDITIONS;
    const preferred = state.conditionId || (s.template && s.template.condition_id) || '';
    state.conditionId = conditions.some((c) => c.id === preferred) ? preferred : '';
    $('conditionChips').innerHTML = conditions.map((c) => `
        <button type="button" class="sell-chip${c.id === state.conditionId ? ' is-active' : ''}" role="radio"
                aria-checked="${c.id === state.conditionId}" data-id="${escapeHtml(c.id)}">${escapeHtml(c.name)}</button>`).join('');
    $('conditionChips').querySelectorAll('button').forEach((btn) => btn.addEventListener('click', () => {
        state.conditionId = btn.dataset.id;
        renderConditions();
        updatePublishState();
    }));
}

// ── Photos ──────────────────────────────────────────────────────────────────

async function resizePhoto(file) {
    const bitmap = await createImageBitmap(file).catch(() => null);
    if (!bitmap) return file; // unknown format: let eBay deal with it
    const scale = Math.min(1, MAX_PHOTO_EDGE / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob || file), 'image/jpeg', 0.88));
}

function addPhotos(files) {
    for (const file of files) {
        const photo = { key: Math.random().toString(36).slice(2), preview: URL.createObjectURL(file), url: null, state: 'uploading' };
        state.photos.push(photo);
        // One upload at a time, in the order taken — the seller keeps going meanwhile.
        state.uploadChain = state.uploadChain.then(() => uploadPhoto(photo, file));
    }
    renderPhotos();
}

async function uploadPhoto(photo, file) {
    if (!state.photos.includes(photo)) return; // removed before its turn
    try {
        const blob = await resizePhoto(file);
        const form = new FormData();
        form.append('photo', blob, 'photo.jpg');
        const { url } = await api('/api/sell/photos', { method: 'POST', body: form });
        photo.url = url;
        photo.state = 'done';
    } catch (err) {
        photo.state = 'error';
        photo.error = err.message;
    }
    renderPhotos();
}

function renderPhotos() {
    $('photos').innerHTML = state.photos.map((p, i) => `
        <figure class="sell-photo sell-photo--${p.state}" data-key="${p.key}" title="${escapeHtml(p.error || '')}">
            <img src="${p.preview}" alt="Photo ${i + 1}">
            <span class="sell-photo-state">${p.state === 'uploading' ? '⏳' : p.state === 'error' ? '⚠️' : i === 0 ? '★' : ''}</span>
            ${i > 0 ? '<button type="button" class="sell-photo-first" data-act="first" aria-label="Make gallery photo">★</button>' : ''}
            <button type="button" class="sell-photo-remove" data-act="remove" aria-label="Remove photo">✕</button>
        </figure>`).join('');
    const done = state.photos.filter((p) => p.state === 'done').length;
    setText('photoCount', state.photos.length ? `${done}/${state.photos.length} uploaded` : '');
    updatePublishState();
}

function onPhotoClick(event) {
    const btn = event.target.closest('button[data-act]');
    if (!btn) return;
    const key = btn.closest('figure').dataset.key;
    const index = state.photos.findIndex((p) => p.key === key);
    if (index < 0) return;
    const [photo] = state.photos.splice(index, 1);
    if (btn.dataset.act === 'first') state.photos.unshift(photo);
    else URL.revokeObjectURL(photo.preview);
    renderPhotos();
}

// ── Publish ─────────────────────────────────────────────────────────────────

function draftBody(verifyOnly) {
    const title = $('titleInput').value.trim();
    const body = {
        title,
        ean: state.ean,
        epid: state.product ? state.product.epid : '',
        price: parsePrice($('priceInput').value),
        condition_id: state.conditionId,
        photos: state.photos.filter((p) => p.state === 'done').map((p) => p.url),
        best_offer: $('bestOffer').checked,
        verify_only: verifyOnly,
    };
    if (!state.product) {
        // Not in eBay's catalog: no catalog match to ask for, and the listing
        // needs its own item specifics (the EAN becomes one of them).
        body.ean = '';
        body.item_specifics = { Spielname: withoutPlatform(title), Plattform: $('platformSelect').value, EAN: state.ean };
    }
    return body;
}

const WAITING_FOR_PHOTOS = 'Waiting for photos to finish uploading…';

function updatePublishState() {
    const uploading = state.photos.some((p) => p.state === 'uploading');
    const ready = $('titleInput').value.trim() && parsePrice($('priceInput').value) > 0 && state.conditionId
        && state.photos.some((p) => p.state === 'done') && !uploading;
    $('publishBtn').disabled = !ready;
    $('verifyBtn').disabled = !ready;
    // Only manage its own note; eBay's answer to "Check with eBay" stays visible.
    if (uploading) setText('publishStatus', WAITING_FOR_PHOTOS);
    else if ($('publishStatus').textContent === WAITING_FOR_PHOTOS) setText('publishStatus', '');
}

async function publish(verifyOnly) {
    const error = $('publishError');
    show(error, false);
    $('publishBtn').disabled = $('verifyBtn').disabled = true;
    setText('publishStatus', verifyOnly ? 'Checking with eBay…' : 'Publishing…');
    try {
        const result = await postJson('/api/sell/publish', draftBody(verifyOnly));
        const fees = Object.entries(result.fees || {}).map(([k, v]) => `${k} ${euro(v)}`).join(', ');
        const notes = [fees ? `Fees: ${fees}` : 'No fees', allowanceNote(), ...(result.warnings || [])]
            .filter(Boolean).join(' · ');
        if (verifyOnly) {
            setText('publishStatus', `eBay accepts it. ${notes}`);
        } else {
            setText('doneText', `${$('titleInput').value.trim()} — ${euro(parsePrice($('priceInput').value))}. ${notes}`);
            $('doneLink').href = result.url;
            show($('listingCard'), false);
            show($('doneCard'), true);
            $('doneCard').scrollIntoView({ behavior: 'smooth' });
            api('/api/sell/status').then((s) => renderRecent(s.recent || [])).catch(() => {});
        }
    } catch (err) {
        error.textContent = err.message;
        show(error, true);
        setText('publishStatus', '');
    } finally {
        updatePublishState();
    }
}

function nextGame() {
    state.photos.forEach((p) => URL.revokeObjectURL(p.preview));
    Object.assign(state, { ean: '', product: null, suggested: null, second: null, addingSecond: false, photos: [] });
    $('lookupInput').value = '';
    $('titleInput').value = '';
    $('priceInput').value = '';
    $('platformSelect').value = '';
    $('offers').innerHTML = '';
    setText('lookupStatus', '');
    setText('publishStatus', '');
    show($('publishError'), false);
    $('bestOffer').checked = !!(state.status.template && state.status.template.best_offer);
    renderPhotos();
    renderConditions();
    show($('draft'), false);
    show($('doneCard'), false);
    show($('listingCard'), true);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    $('lookupInput').focus();
}

// ── Barcode scanner ─────────────────────────────────────────────────────────

function loadScript(src) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = () => reject(new Error('Could not load the barcode scanner.'));
        document.head.appendChild(script);
    });
}

async function nativeDetector() {
    if (!('BarcodeDetector' in window)) return null;
    const formats = await BarcodeDetector.getSupportedFormats().catch(() => []);
    return formats.includes('ean_13') ? new BarcodeDetector({ formats: ['ean_13', 'ean_8', 'upc_a'] }) : null;
}

async function startScanner() {
    const video = $('scannerVideo');
    show($('scanner'), true);
    const found = (code) => { stopScanner(); $('lookupInput').value = code; lookup(code); };
    try {
        const detector = await nativeDetector();
        if (detector) {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            video.srcObject = stream;
            await video.play();
            state.scanner = { stop: () => stream.getTracks().forEach((t) => t.stop()) };
            const tick = async () => {
                if (!state.scanner) return;
                const codes = await detector.detect(video).catch(() => []);
                if (codes.length) found(codes[0].rawValue); else requestAnimationFrame(tick);
            };
            tick();
        } else {
            // Safari has no BarcodeDetector: ZXing, loaded only when needed.
            if (!window.ZXing) await loadScript(ZXING_URL);
            const reader = new window.ZXing.BrowserMultiFormatReader();
            state.scanner = { stop: () => reader.reset() };
            await reader.decodeFromConstraints({ video: { facingMode: 'environment' } }, video, (result) => {
                if (result && state.scanner) found(result.getText());
            });
        }
    } catch (err) {
        stopScanner();
        setText('lookupStatus', `Camera unavailable (${err.message}) — type the EAN instead.`);
    }
}

function stopScanner() {
    if (state.scanner) state.scanner.stop();
    state.scanner = null;
    const video = $('scannerVideo');
    if (video.srcObject) video.srcObject = null;
    show($('scanner'), false);
}

// ── Wiring ──────────────────────────────────────────────────────────────────

function init() {
    if (!$('sellApp')) return;
    $('pastedAuthBtn').addEventListener('click', connectFromPastedUrl);
    $('templateBtn').addEventListener('click', importTemplate);
    $('lookupInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); lookup(e.target.value); } });
    $('scanBtn').addEventListener('click', startScanner);
    $('scanCancel').addEventListener('click', stopScanner);
    $('addGameBtn').addEventListener('click', () => {
        state.addingSecond = true;
        $('lookupInput').value = '';
        $('lookupInput').placeholder = 'EAN or title of the second game';
        window.scrollTo({ top: 0, behavior: 'smooth' });
        $('lookupInput').focus();
    });
    $('titleInput').addEventListener('input', updateTitleCount);
    $('priceInput').addEventListener('input', updateBuyerTotal);
    $('photoCamera').addEventListener('change', (e) => { addPhotos([...e.target.files]); e.target.value = ''; });
    $('photoPicker').addEventListener('change', (e) => { addPhotos([...e.target.files]); e.target.value = ''; });
    $('photos').addEventListener('click', onPhotoClick);
    $('verifyBtn').addEventListener('click', () => publish(true));
    $('publishBtn').addEventListener('click', () => publish(false));
    $('nextBtn').addEventListener('click', nextGame);
    updatePublishState();
    loadStatus();
}

document.addEventListener('DOMContentLoaded', init);
