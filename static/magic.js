// setpiece — Magic Mix recommender panel (self-contained add-on)
//
// Self-initializing IIFE. Injects its own <section id="magic-panel">
// into <main>, loads its own /magic.css via a programmatic <link>, and
// talks to the existing backend:
//   POST /api/magic/suggest  {top_n: 6}  -> {ok, suggestions:[...]}
//   POST /api/scratch/fire   {path}      -> fire a file to LIVE
//
// Zero edits to control.html / control.css / control.js — the only
// integration step is adding <script src="/magic.js"></script> before
// </body> (see MAGIC_PANEL_INTEGRATION.md).
(function () {
    'use strict';

    // Auto-refresh cadence — recommendations drift as LIVE/scratch change.
    const REFRESH_MS = 15000;
    const TOP_N = 6;

    let refreshTimer = null;
    let inFlight = false;

    // ── CSS injection ─────────────────────────────────────────────────
    // Inject <link rel="stylesheet" href="/magic.css"> into <head> so the
    // panel styles load without touching control.html.
    function injectStylesheet() {
        if (document.querySelector('link[data-magic-css]')) return;
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = '/magic.css';
        link.setAttribute('data-magic-css', '1');
        document.head.appendChild(link);
    }

    // ── basename helper (mirrors control.js fmtBasename) ──────────────
    function fmtBasename(path) {
        if (!path) return '(unknown)';
        const parts = String(path).replace(/\\/g, '/').split('/');
        return parts[parts.length - 1] || path;
    }

    // ── DOM scaffold ──────────────────────────────────────────────────
    function buildPanel() {
        const main = document.querySelector('main');
        if (!main) return null;
        if (document.getElementById('magic-panel')) {
            return document.getElementById('magic-panel');
        }

        const section = document.createElement('section');
        section.id = 'magic-panel';
        section.className = 'panel';

        const header = document.createElement('div');
        header.className = 'magic-header';

        const h2 = document.createElement('h2');
        h2.textContent = 'Magic Mix · what’s next?';

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'magic-refresh-btn';
        btn.textContent = 'Refresh';
        btn.addEventListener('click', function () { loadSuggestions(); });

        header.appendChild(h2);
        header.appendChild(btn);

        const list = document.createElement('ul');
        list.className = 'magic-list';
        list.id = 'magic-list';
        const placeholder = document.createElement('li');
        placeholder.className = 'empty';
        placeholder.textContent = 'Loading suggestions…';
        list.appendChild(placeholder);

        section.appendChild(header);
        section.appendChild(list);
        main.appendChild(section);
        return section;
    }

    // ── rendering ─────────────────────────────────────────────────────
    function setListMessage(cls, text) {
        const list = document.getElementById('magic-list');
        if (!list) return;
        list.innerHTML = '';
        const li = document.createElement('li');
        li.className = cls;
        li.textContent = text;
        list.appendChild(li);
    }

    function renderSuggestions(suggestions) {
        const list = document.getElementById('magic-list');
        if (!list) return;
        if (!suggestions || !suggestions.length) {
            setListMessage('empty', 'No suggestions yet — fire a few clips.');
            return;
        }
        list.innerHTML = '';
        suggestions.forEach(function (s) {
            const row = document.createElement('li');
            row.className = 'magic-suggestion';
            row.dataset.path = s.path || '';

            // top row: filename + score percentage
            const top = document.createElement('div');
            top.className = 'magic-row-top';

            const name = document.createElement('span');
            name.className = 'magic-name';
            name.textContent = s.name || fmtBasename(s.path);

            // clamp score to 0..1, render as a whole percentage
            let score = Number(s.score);
            if (!isFinite(score)) score = 0;
            score = Math.max(0, Math.min(1, score));
            const pct = document.createElement('span');
            pct.className = 'magic-score-pct';
            pct.textContent = Math.round(score * 100) + '%';

            top.appendChild(name);
            top.appendChild(pct);

            // score bar: track + green-gradient fill
            const bar = document.createElement('div');
            bar.className = 'magic-score-bar';
            const fill = document.createElement('div');
            fill.className = 'magic-score-fill';
            fill.style.width = (score * 100) + '%';
            bar.appendChild(fill);

            // reasons: small muted text, comma-joined
            const reasons = document.createElement('div');
            reasons.className = 'magic-reasons';
            const reasonList = Array.isArray(s.reasons) ? s.reasons : [];
            reasons.textContent = reasonList.length
                ? reasonList.join('  ·  ')
                : 'similar vibe';

            row.appendChild(top);
            row.appendChild(bar);
            row.appendChild(reasons);

            // tap the row -> fire that file to LIVE
            row.addEventListener('click', function () {
                fireToLive(row.dataset.path, row);
            });

            list.appendChild(row);
        });
    }

    // ── network ───────────────────────────────────────────────────────
    function setBusy(busy) {
        inFlight = busy;
        const btn = document.querySelector('#magic-panel .magic-refresh-btn');
        if (btn) btn.classList.toggle('busy', busy);
    }

    async function loadSuggestions() {
        if (inFlight) return;
        setBusy(true);
        try {
            const r = await fetch('/api/magic/suggest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ top_n: TOP_N }),
                cache: 'no-store'
            });
            if (!r.ok) throw new Error('HTTP ' + r.status);
            const data = await r.json();
            // Backend wraps callbacks as {ok, result}; the magic payload
            // may be at the top level OR nested under .result. Handle both.
            const payload = (data && data.result && typeof data.result === 'object')
                ? data.result
                : data;
            if (!payload || payload.ok === false) {
                setListMessage('magic-unavailable', 'Magic mix unavailable');
                return;
            }
            renderSuggestions(payload.suggestions || []);
        } catch (e) {
            // iOS Safari swallows fetch errors silently — show it inline.
            setListMessage('magic-unavailable', 'Magic mix unavailable');
        } finally {
            setBusy(false);
        }
    }

    async function fireToLive(path, row) {
        if (!path) return;
        try {
            const r = await fetch('/api/scratch/fire', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: path }),
                cache: 'no-store'
            });
            if (!r.ok) throw new Error('HTTP ' + r.status);
            // Recommendation drifts the moment LIVE changes — refresh soon.
            setTimeout(loadSuggestions, 600);
        } catch (e) {
            if (row) {
                const reasons = row.querySelector('.magic-reasons');
                if (reasons) reasons.textContent = 'fire failed: ' + e.message;
            }
        }
    }

    // ── auto-refresh loop ─────────────────────────────────────────────
    function startAutoRefresh() {
        if (refreshTimer) clearInterval(refreshTimer);
        refreshTimer = setInterval(loadSuggestions, REFRESH_MS);
    }

    // ── init ──────────────────────────────────────────────────────────
    function init() {
        injectStylesheet();
        const panel = buildPanel();
        if (!panel) return; // no <main> — nothing to attach to
        loadSuggestions();
        startAutoRefresh();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
