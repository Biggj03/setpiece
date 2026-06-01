// setpiece iPad control script
// Hoisted module-level state vars (avoiding TDZ ReferenceErrors)
let pollTimer = null;
let lastBeatAge = null;
let connectionFailures = 0;

// Library browser state.
// loadingFile: the basename the user just tapped, kept "loading" until the
// polled state confirms playback has switched to it. Set null on
// confirm/error/folder-change so the row stops pulsing.
let loadingFile = null;
// Cache the last rendered library payload so we can compare and avoid
// rebuilding the DOM on every 500ms poll (prevents scroll-jank on iPad).
let lastLibrarySig = '';
let lastCurrentBasename = '';

// Library search: case-insensitive substring filter applied client-side.
// Empty string = no filter. Hoisted to module scope for the same TDZ
// reason as the other library state vars above.
let librarySearchQuery = '';

// Decks: which slot is currently "selected" (waiting for the next clip
// or library tap to fill it). null = no selection (taps fire immediately).
let selectedDeck = null;

// Channels: cached signature so we only repaint chips when the underlying
// channel list (or active idx) actually changes. The editor open/close is
// tracked separately so the chip row repaints when a chip becomes "open".
// editingChannelIdx: which channel the inline editor is currently showing,
// or null if the editor is closed. Hoisted to module scope to dodge TDZ.
let lastChannelsSig = '';
let editingChannelIdx = null;
// Default color palette must mirror channels.py DEFAULT_COLORS so the
// swatch picker offers the same options the server seeds with.
const CHANNEL_COLOR_PALETTE = [
    '#ff3355', '#ff9933', '#33cc66', '#3399ff',
    '#cc66ff', '#66ddff', '#ffcc00', '#ff66aa',
];

// Clips filter & sort state.
// activeFilter:
//   'all'    = show every clip (default)
//   'star'   = only starred clips
//   'tag:X'  = only clips with tag X (X is the tag string)
// sortMode:
//   'added'  = original insertion order (default — like Spotify "just added")
//   'recent' = last_played_ts desc (most recent fire at top)
//   'most'   = play_count desc (heaviest hitters at top)
// editingTagsFor: clip_id whose row is currently showing the inline
//   "+ tag" input. Only one row at a time gets the input — keeps the
//   list dense and avoids stray autofill banners on iPad.
let activeFilter = 'all';
let sortMode = 'added';
let editingTagsFor = null;
// Cache for the all_tags list so we only rebuild the filter chips when
// the union actually changes.
let lastAllTagsSig = '';
// Cache last-rendered deck signature so we don't tear down img backgrounds
// every 500ms poll (filmstrips would re-fetch and lose the CSS animation
// timing).
let lastDecksSig = '';

// Visible error handlers (iOS Safari swallows JS errors silently otherwise)
window.onerror = function (msg, src, line, col, err) {
    showError('JS Error: ' + msg + ' @' + line + ':' + col);
    return false;
};
window.addEventListener('unhandledrejection', function (e) {
    showError('Promise rejection: ' + (e.reason?.message || e.reason));
});

function showError(msg) {
    const banner = document.getElementById('error-banner');
    banner.textContent = msg;
    banner.classList.remove('hidden');
    setTimeout(() => banner.classList.add('hidden'), 5000);
}

function setStatus(msg) {
    // Status panel was deleted in the iPad audit (2026-05-16) — those
    // toast messages were noise no one read. Kept the call sites & this
    // shim so we don't have to rewrite every fire/save/load action; route
    // to console for dev visibility only.
    if (msg) console.debug('[status]', msg);
}

function setConnection(online) {
    const dot = document.getElementById('connection-status');
    dot.classList.toggle('online', online);
    dot.classList.toggle('offline', !online);
}

function fmtTime(seconds) {
    if (!seconds && seconds !== 0) return '--:--.---';
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(3).padStart(6, '0');
    return `${String(m).padStart(2, '0')}:${s}`;
}

function fmtBasename(path) {
    if (!path) return 'No video loaded';
    const parts = path.replace(/\\/g, '/').split('/');
    return parts[parts.length - 1] || path;
}

async function pollState() {
    try {
        const r = await fetch('/api/state', { cache: 'no-store' });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const state = await r.json();
        connectionFailures = 0;
        setConnection(true);
        renderState(state);
    } catch (e) {
        connectionFailures++;
        setConnection(false);
        if (connectionFailures > 3) {
            setStatus('Connection lost (retry ' + connectionFailures + ')');
        }
    }
}

// ── Mode Dashboard — "what mode am I in" at a glance ─────────────────
// Read-only pill grid populated from state.modes each /api/state poll.
// Stable display order; pill colour comes from each mode's `state`.
function _renderModeDashboard(state) {
    const host = document.getElementById('mode-dash');
    if (!host) return;
    const modes = (state && state.modes) || null;
    if (!modes || !Object.keys(modes).length) {
        host.innerHTML = '<span class="mode-dash-empty">modes unavailable — restart the app to load them</span>';
        return;
    }
    const order = ['autoflip', 'audio', 'setarc', 'lyric', 'layer',
                   'mk2page', 'spotter', 'cohesion', 'hold', 'blackout'];
    const keys = order.filter(k => modes[k])
        .concat(Object.keys(modes).filter(k => !order.includes(k)));
    host.innerHTML = keys.map(k => {
        const m = modes[k] || {};
        const st = m.state || 'neutral';
        return `<div class="mode-pill mode-pill-${escapeHTML(st)}">`
            + `<span class="mode-pill-label">${escapeHTML(m.label || k)}</span>`
            + `<span class="mode-pill-value">${escapeHTML(String(m.value == null ? '' : m.value))}</span>`
            + `</div>`;
    }).join('');
}

function renderState(state) {
    // Sticky status strip (replaces old Now Playing + Markers + Audio
    // Reactive + S2 mirror panels per audit 2026-05-16). Read-only — the
    // hardware IS the control surface; the strip just confirms state.
    _renderModeDashboard(state);
    const pb = state.playback || {};
    document.getElementById('current-file').textContent = fmtBasename(pb.file);
    document.getElementById('position').textContent = fmtTime(pb.position);
    document.getElementById('duration').textContent = fmtTime(pb.duration);
    const pct = pb.duration > 0 ? (pb.position / pb.duration) * 100 : 0;
    document.getElementById('progress-fill').style.width = pct + '%';
    const playIcon = document.getElementById('play-icon');
    playIcon.textContent = pb.is_playing ? 'PLAYING' : 'PAUSED';
    playIcon.classList.toggle('ss-playing', !!pb.is_playing);
    playIcon.classList.toggle('ss-paused', !pb.is_playing);

    // Audio reactive — minimal: BPM number + ON/OFF chip in the strip.
    // (Sensitivity readout + beat-pulse bar dropped — they were "data we
    // have, why not show" cruft. BPM is the only number that drives perf
    // decisions; ON/OFF tells you whether auto-flip will even fire.)
    const ar = state.audio_reactive || {};
    const arState = document.getElementById('audio-state');
    arState.textContent = ar.enabled ? 'REACT ON' : 'REACT OFF';
    arState.classList.toggle('ss-react-on', !!ar.enabled);
    arState.classList.toggle('ss-react-off', !ar.enabled);
    document.getElementById('audio-bpm').textContent = (ar.bpm ? Math.round(ar.bpm) : '---') + ' BPM';

    // Stem-onset pulse: read state.stems block (populated by main
    // app's _on_stem_drum_onset handler). Pulses magenta when
    // boost_active = true (within the 200ms post-kick window) or
    // recently faded. Hidden when stem listener disabled.
    const stems = state.stems || {};
    const stemsEl = document.getElementById('ss-stems');
    if (stemsEl) {
        if (!stems.enabled) {
            stemsEl.textContent = '';
            stemsEl.className = 'ss-stems ss-stems-idle';
        } else {
            const onsets = stems.onsets_60s || 0;
            const lastAge = stems.last_onset_age_s;
            const boostNow = !!stems.boost_active;
            // tier: hot (boost active or <0.5s) → cool (<3s) → idle
            let tier;
            if (boostNow || (lastAge !== null && lastAge < 0.5)) tier = 'hot';
            else if (lastAge !== null && lastAge < 3) tier = 'cool';
            else tier = 'listening';
            // Label: "♫ 142/min" so user sees onset rate
            const rate = Math.round(onsets);
            stemsEl.textContent = tier === 'listening'
                ? '♫ listening'
                : `♫ ${rate}/min`;
            stemsEl.className = `ss-stems ss-stems-${tier}`;
        }
    }

    // Cohesion anchor: poll independently (every ~3s, throttled below
    // via window._cohesionLastPoll). Shows seed file + seconds left so
    // the operator sees the current "music-video shoot" lock. Tap = refresh.
    const cohEl = document.getElementById('ss-cohesion');
    if (cohEl && !cohEl.dataset.wired) {
        cohEl.dataset.wired = '1';
        cohEl.addEventListener('click', () => {
            // POST {refresh:true} → forces re-roll, returns new status
            postAction('cohesion/status', { refresh: true }).then(() => {
                window._cohesionLastPoll = 0;   // trigger immediate re-fetch
            });
        });
    }
    const nowMs = performance.now();
    if (cohEl && (!window._cohesionLastPoll
                  || nowMs - window._cohesionLastPoll > 3000)) {
        window._cohesionLastPoll = nowMs;
        postAction('cohesion/status', {}).then(r => {
            const res = (r && r.result) || {};
            if (!res.enabled) {
                cohEl.textContent = '';
                cohEl.className = 'ss-cohesion ss-cohesion-idle';
                return;
            }
            const subset = res.subset || [];
            const remaining = res.seconds_remaining || 0;
            if (!subset.length) {
                cohEl.textContent = '⚓ (no anchor)';
                cohEl.className = 'ss-cohesion ss-cohesion-empty';
                return;
            }
            // Find the seed entry — first in subset
            const seedEntry = subset[0] || {};
            const seedShort = (seedEntry.name || '?').slice(0, 18);
            // Tier by time remaining — hot when fresh, cool when expiring
            let tier;
            if (remaining > 60) tier = 'hot';
            else if (remaining > 20) tier = 'cool';
            else tier = 'fading';
            cohEl.textContent = `⚓ ${seedShort} ·${Math.round(remaining)}s`;
            cohEl.className = `ss-cohesion ss-cohesion-${tier}`;
            cohEl.title = `cohesion anchor (${subset.length} clips):\n`
                        + subset.map(s => '• ' + (s.name || '?')).join('\n')
                        + `\n\ntap to re-roll`;
        }).catch(() => {});
    }

    // ARC = set-arc phase indicator. Render here so each poll updates
    // the color/label as the phase changes (manual cycle OR auto detect).
    // Also surfaces predicted_phase hint when the auto-detector thinks
    // a change is coming (e.g. "ARC BUILD → PEAK ↗"). The hint
    // displays even when auto mode is OFF -- it's just advisory.
    const arcEl = document.getElementById('ss-arc');
    if (arcEl) {
        const arc = state.set_arc || {};
        if (arc.enabled) {
            const phaseColors = {
                opening:   { bg: '#1e3a8a', fg: '#bfdbfe' },
                build:     { bg: '#d97706', fg: '#fef3c7' },
                peak:      { bg: '#dc2626', fg: '#fee2e2' },
                breakdown: { bg: '#1e293b', fg: '#94a3b8' },
            };
            const pc = phaseColors[arc.phase] || phaseColors.opening;
            const lbl = (arc.label || arc.phase || '').toUpperCase();
            // Prediction hint: prefer the 30s LOOKAHEAD over the
            // "would-be-now" prediction since lookahead actually
            // tells the operator "heads up, change is coming." Falls
            // back to predicted_is_change for backward compatibility
            // when lookahead isn't in the state yet.
            const ENERGY = {
                opening: 0, build: 1, peak: 2, breakdown: 0,
            };
            let hint = '';
            const look = arc.lookahead || {};
            const t30 = look.t30 || null;
            if (t30 && t30.is_change && t30.phase) {
                const pp = String(t30.phase).toUpperCase();
                const arrow = (ENERGY[t30.phase] ?? 0)
                              > (ENERGY[arc.phase] ?? 0) ? '↗' : '↘';
                // Show "→ PEAK 30s ↗" — concise heads-up
                hint = ` → ${pp} 30s ${arrow}`;
            } else if (arc.predicted_is_change && arc.predicted_phase) {
                const pp = String(arc.predicted_phase).toUpperCase();
                const arrow = (ENERGY[arc.predicted_phase] ?? 0)
                              > (ENERGY[arc.phase] ?? 0) ? '↗' : '↘';
                hint = ` → ${pp} ${arrow}`;
            }
            arcEl.textContent = (arc.auto ? 'A·' : '') + 'ARC ' + lbl + hint;
            arcEl.classList.remove('ss-arc-off');
            arcEl.classList.add('ss-arc-on');
            arcEl.style.background = pc.bg;
            arcEl.style.color = pc.fg;
            arcEl.style.boxShadow = arc.auto
                ? `0 0 10px ${pc.bg}, 0 0 0 1px rgba(255,255,255,0.6) inset`
                : `0 0 6px ${pc.bg}aa`;
        } else {
            arcEl.textContent = 'ARC';
            arcEl.classList.add('ss-arc-off');
            arcEl.classList.remove('ss-arc-on');
            arcEl.style.background = '';
            arcEl.style.color = '';
            arcEl.style.boxShadow = '';
        }
    }

    // CDJ Tier-2 phase meter. Cache snapshot of (bpm, last_beat_age_ms, when-received)
    // so the animation loop can extrapolate phase locally at 60Hz between polls.
    // The poll itself only fires every 500ms — without local extrapolation
    // the bar would jump in 500ms chunks, useless. With extrapolation, smooth.
    if (typeof window._phaseState === 'undefined') window._phaseState = {};
    if (ar.bpm && ar.bpm > 30 && ar.bpm < 250) {
        window._phaseState = {
            bpm: ar.bpm,
            beatAgeMs: ar.last_beat_age_ms || 0,
            receivedAt: performance.now(),
            enabled: !!ar.enabled,
            barIdx: window._phaseState.barIdx || 0,
        };
    } else {
        window._phaseState.enabled = false;
    }

    // (Markers panel + S2 mirror panel deleted per audit — Clip Markers
    // are set on S2 LOOP IN/OUT with no need to read back, S2 mirror was
    // a visual duplicate of buttons the user's hand is on. Both removed.)

    // Saved clips: render via dedicated function so filter/sort changes
    // can re-trigger it without going through the whole pollState path.
    renderClips(state.clips || [], state.all_tags || []);

    // Bank: cross-video shortlist of starred + recent clips.
    renderBank(state.bank_clips || []);

    // Scenes: saved 4-deck layout snapshots.
    renderScenes(state.scenes || []);

    // Scratch: session basket of library files.
    renderScratch(state.scratch || []);
    renderScratchSets(state.scratch_sets || []);
    renderBanks(state.banks || []);
    // Bank LAYER OVERLAY chip — sync to whatever the server reports
    // as the active layer (could have been cycled via MK2 STEP).
    updateBankLayer(state.bank_layer || { active: 'default', order: [] });

    // Path-tag filter chips above the library list.
    renderPathTags(state.path_tags || {});
    // Smart playlists row in library header.
    renderSmartPlaylists(state.smart_playlists || []);
    renderActiveTag(state.active_tag || "");

    // (Pad click handlers are wired once on DOMContentLoaded, not here)

    // Crossfade indicator (mirrors S2 crossfader position + preview file)
    renderCrossfade(state.crossfade || {}, state.playback || {});

    // Preview panel — live MJPEG of the staged "next up" deck
    renderPreviewMonitor(state);

    // Decks (launchpad)
    renderDecks(state.decks || [], state.playback || {});
    renderProxyStatus(state.proxy_status || {});

    // Channels (saved library-folder presets bound to S2 FX channel buttons)
    renderChannels(state.channels || {});

    // Library browser
    renderLibrary(state.library || {}, state.playback || {});

    // Status / messages
    const status = state.status || {};
    if (status.error) {
        showError(status.error);
    } else if (status.message) {
        setStatus(status.message);
    }
}

// ── Clips: filter, sort, render ──────────────────────────────────────

// Cache the unfiltered clip array so filter/sort taps can re-render
// without waiting for the next poll. Refreshed every renderClips call.
let _lastClipsRaw = [];
let _lastAllTagsList = [];
// Signature so we don't tear down DOM on every poll (image fetch storm).
let _lastClipsRenderSig = '';

function _clipFilterMatches(clip) {
    if (activeFilter === 'all') return true;
    if (activeFilter === 'star') return !!clip.starred;
    if (activeFilter.startsWith('tag:')) {
        const wanted = activeFilter.slice(4);
        return Array.isArray(clip.tags) && clip.tags.includes(wanted);
    }
    if (activeFilter.startsWith('bpm:')) {
        // Format: bpm:LO-HI (HI=999 means "+"; e.g. bpm:150-999 = 150+).
        // Clips with bpm==0 (unknown / not yet analysed) never match.
        const bpm = +clip.bpm || 0;
        if (!bpm) return false;
        const m = activeFilter.slice(4).match(/^(\d+)-(\d+)$/);
        if (!m) return false;
        const lo = +m[1], hi = +m[2];
        return bpm >= lo && bpm < hi;
    }
    return true;
}

function _clipsSorted(clips) {
    const arr = clips.slice();
    if (sortMode === 'recent') {
        // Most recently played first; never-played sink to bottom.
        arr.sort((a, b) => (b.last_played_ts || 0) - (a.last_played_ts || 0));
    } else if (sortMode === 'most') {
        arr.sort((a, b) => (b.play_count || 0) - (a.play_count || 0));
    } else if (sortMode === 'bpm') {
        // Ascending; clips with no BPM yet sink to the bottom (so the
        // sorted list isn't dominated by zeros while backfill runs).
        arr.sort((a, b) => {
            const ab = +a.bpm || 0, bb = +b.bpm || 0;
            if (!ab && !bb) return 0;
            if (!ab) return 1;
            if (!bb) return -1;
            return ab - bb;
        });
    }
    // 'added' = caller's order, no sort.
    return arr;
}

// Pretty-print a BPM band filter for the chip label.
function _fmtBpmBand(bandStr) {
    const m = (bandStr || '').match(/^(\d+)-(\d+)$/);
    if (!m) return bandStr || '';
    const hi = +m[2];
    return hi >= 999 ? `${m[1]}+` : `${m[1]}-${hi}`;
}

function _renderFilterChips(allTags) {
    const bar = document.getElementById('clips-filter-bar');
    if (!bar) return;
    // Only rebuild chip set when the tag union (or active filter) changes
    // — chip taps would otherwise lose their listeners on every 500ms poll.
    const sig = activeFilter + '|' + (allTags || []).join(',');
    if (sig === lastAllTagsSig) return;
    lastAllTagsSig = sig;

    // Keep "All" + "★" + "BPM" anchored on the left, then one chip per tag.
    const allActive = activeFilter === 'all';
    const starActive = activeFilter === 'star';
    const bpmActive = activeFilter.startsWith('bpm:');
    const bpmLabel = bpmActive ? _fmtBpmBand(activeFilter.slice(4)) + ' BPM' : 'BPM';
    let html = `
        <button class="filter-chip filter-all ${allActive ? 'active' : ''}"
                data-filter="all">All</button>
        <button class="filter-chip filter-star ${starActive ? 'active' : ''}"
                data-filter="star" title="Starred only">&#9733;</button>
        <button class="filter-chip filter-bpm ${bpmActive ? 'active' : ''}"
                data-filter="__bpm" title="Filter by BPM range">${escapeHTML(bpmLabel)}</button>
    `;
    (allTags || []).forEach(tag => {
        const active = activeFilter === 'tag:' + tag;
        html += `<button class="filter-chip filter-tag ${active ? 'active' : ''}"
                 data-filter="tag:${escapeHTML(tag)}">#${escapeHTML(tag)}</button>`;
    });
    bar.innerHTML = html;

    bar.querySelectorAll('.filter-chip').forEach(btn => {
        btn.addEventListener('click', () => {
            const f = btn.dataset.filter || 'all';
            // BPM chip is special: it opens the inline band picker
            // instead of being a filter on its own. Re-tap closes the
            // picker (and clears the band filter if one was active).
            if (f === '__bpm') {
                const picker = document.getElementById('bpm-band-picker');
                const isOpen = picker && !picker.classList.contains('hidden');
                if (isOpen) {
                    picker.classList.add('hidden');
                    if (activeFilter.startsWith('bpm:')) {
                        activeFilter = 'all';
                        lastAllTagsSig = '';
                        _lastClipsRenderSig = '';
                        _renderFilterChips(_lastAllTagsList);
                        renderClips(_lastClipsRaw, _lastAllTagsList);
                    }
                } else if (picker) {
                    picker.classList.remove('hidden');
                    _markActiveBpmBand();
                }
                return;
            }
            // Tap the active chip (anything other than 'all') to clear.
            if (activeFilter === f && f !== 'all') {
                activeFilter = 'all';
            } else {
                activeFilter = f;
            }
            // Hide the BPM picker if the user moved to another filter.
            const picker = document.getElementById('bpm-band-picker');
            if (picker) picker.classList.add('hidden');
            // Force chips + list to rebuild this tick.
            lastAllTagsSig = '';
            _lastClipsRenderSig = '';
            _renderFilterChips(_lastAllTagsList);
            renderClips(_lastClipsRaw, _lastAllTagsList);
        });
    });
}

// Highlight whichever band button matches the active BPM filter (if any).
function _markActiveBpmBand() {
    const picker = document.getElementById('bpm-band-picker');
    if (!picker) return;
    const wanted = activeFilter.startsWith('bpm:') ? activeFilter.slice(4) : '';
    picker.querySelectorAll('.bpm-band-opt').forEach(b => {
        b.classList.toggle('active', b.dataset.band === wanted);
    });
}

// One-time wiring for the BPM band picker (innerHTML inside this picker
// never gets touched, so addEventListener stays attached for the page's
// lifetime — same pattern as wireLibrarySearch / wirePadHandlers).
function wireBpmBandPicker() {
    const picker = document.getElementById('bpm-band-picker');
    if (!picker || picker.dataset.wired) return;
    picker.dataset.wired = 'true';
    picker.querySelectorAll('.bpm-band-opt').forEach(btn => {
        btn.addEventListener('click', (ev) => {
            ev.stopPropagation();
            const band = btn.dataset.band || '';
            const next = 'bpm:' + band;
            // Tap the active band again = clear filter (consistent with tag chips).
            if (activeFilter === next) {
                activeFilter = 'all';
                picker.classList.add('hidden');
            } else {
                activeFilter = next;
            }
            _markActiveBpmBand();
            lastAllTagsSig = '';
            _lastClipsRenderSig = '';
            _renderFilterChips(_lastAllTagsList);
            renderClips(_lastClipsRaw, _lastAllTagsList);
        });
    });
}

function _renderTagsDatalist(allTags) {
    const dl = document.getElementById('all-tags-datalist');
    if (!dl) return;
    const sig = (allTags || []).join('|');
    if (dl.dataset.sig === sig) return;
    dl.dataset.sig = sig;
    dl.innerHTML = (allTags || [])
        .map(t => `<option value="${escapeHTML(t)}"></option>`)
        .join('');
}

// Sticky active tag — auto-applied to every Mark OUT and SHIFT+Loop Out.
// Reconciles input value from server state (without clobbering an in-progress
// edit) and toggles the .set glow when a tag is active.
let _activeTagInputFocused = false;
function renderActiveTag(tag) {
    const input = document.getElementById('active-tag-input');
    const row = input ? input.closest('.active-tag-row') : null;
    if (!input || !row) return;
    if (!_activeTagInputFocused && input.value !== tag) {
        input.value = tag || "";
    }
    row.classList.toggle('set', !!(tag && tag.length));
}

// ── Smart playlists (saved folder + tag-filter combos) ──────────────
let _lastSmartPLSig = "";
let _smartPLSaveWired = false;
function renderSmartPlaylists(items) {
    const row = document.getElementById('smart-playlists-row');
    if (!row) return;
    const saveBtn = document.getElementById('smart-playlist-save');
    if (saveBtn && !_smartPLSaveWired) {
        _smartPLSaveWired = true;
        saveBtn.addEventListener('click', () => {
            const name = prompt('Name this playlist:', '');
            if (name !== null) {
                postAction('smart_playlist/save', { name: name.trim() }).catch(() => {});
            }
        });
    }
    const sig = JSON.stringify((items || []).map(s => [s.id, s.name, s.tags, s.library_root]));
    if (sig === _lastSmartPLSig) return;
    _lastSmartPLSig = sig;
    // Keep the save button, append chips.
    Array.from(row.querySelectorAll('.smart-pl-chip')).forEach(el => el.remove());
    (items || []).forEach(s => {
        const chip = document.createElement('span');
        chip.className = 'smart-pl-chip';
        chip.dataset.id = s.id || '';
        const tagsStr = (s.tags && s.tags.length) ? ` · ${s.tags.join(', ')}` : '';
        chip.innerHTML =
            `<button type="button" class="spl-load" data-id="${escapeHTML(s.id || '')}"`
            + ` title="Apply ${escapeHTML(tagsStr)}">${escapeHTML(s.name || '?')}</button>`
            + `<button type="button" class="spl-del" data-id="${escapeHTML(s.id || '')}" title="Delete">×</button>`;
        chip.querySelector('.spl-load').addEventListener('click', () => {
            postAction('smart_playlist/apply', { id: s.id }).catch(() => {});
        });
        chip.querySelector('.spl-del').addEventListener('click', (ev) => {
            ev.stopPropagation();
            if (confirm(`Delete "${s.name}"?`)) {
                postAction('smart_playlist/delete', { id: s.id }).catch(() => {});
            }
        });
        row.appendChild(chip);
    });
}

// ── Path-tag filter chips (auto-derived from folder/file names) ─────
let _lastPathTagsSig = "";
function renderPathTags(pt) {
    const bar = document.getElementById('path-tags-bar');
    if (!bar) return;
    const top = pt.top || [];
    const filter = new Set(pt.filter || []);
    const sig = JSON.stringify([top.map(t => [t.tag, t.count]), Array.from(filter).sort()]);
    if (sig === _lastPathTagsSig) return;
    _lastPathTagsSig = sig;
    if (!top.length) {
        bar.innerHTML = '';
        return;
    }
    bar.innerHTML = top.map(t => {
        const tag = escapeHTML(t.tag || '');
        const cnt = Number(t.count || 0);
        const active = filter.has(t.tag);
        return `<button type="button" class="path-tag-chip${active ? ' active' : ''}"
            data-tag="${tag}">${tag}<span class="pt-count">${cnt}</span></button>`;
    }).join('');
    bar.querySelectorAll('.path-tag-chip').forEach(btn => {
        btn.addEventListener('click', () => {
            const tag = btn.dataset.tag;
            const next = new Set(filter);
            if (next.has(tag)) {
                next.delete(tag);
            } else {
                next.add(tag);
            }
            postAction('path_tags/set_filter', { tags: Array.from(next) }).catch(() => {});
        });
    });
}

// ── Auto-bank A-H button + tag-search-to-banks shortcut ─────────────
// Two paths:
//   1. Type a name + ENTER in the search input -> hits bank/tag_to_banks
//      which finds best-matching tag, applies as filter, then auto-banks.
//      ONE action for "I want this artist now."
//   2. Tap chips first, then tap AUTO-BANK button -> hits bank/auto_split
//      which uses whatever filter+folder is currently active.
function _resetAutoBankHint() {
    const hint = document.getElementById('auto-bank-hint');
    if (hint) hint.textContent = 'type + ENTER or tap chips + AUTO-BANK';
}

function _showAutoBankResult(res) {
    const btn = document.getElementById('auto-bank-btn');
    const hint = document.getElementById('auto-bank-hint');
    if (!btn) return;
    if (res.ok) {
        const dist = res.distribution || {};
        const parts = Object.entries(dist)
            .filter(([_, n]) => n > 0)
            .map(([l, n]) => `${l}:${n}`)
            .join(' ');
        btn.textContent = `${res.total} -> ${parts}`;
        if (hint) {
            const matched = res.matched_tag
                ? ` [tag: ${res.matched_tag}]` : '';
            hint.textContent = `name: "${res.name}"${matched}`;
        }
    } else {
        btn.textContent = res.error || 'failed';
    }
    setTimeout(() => {
        btn.textContent = 'AUTO-BANK A-H';
        _resetAutoBankHint();
    }, 5000);
}

function wireAutoBankButton() {
    const btn = document.getElementById('auto-bank-btn');
    if (!btn || btn.dataset.wired) return;
    btn.dataset.wired = '1';
    btn.addEventListener('click', async () => {
        btn.disabled = true;
        btn.textContent = 'splitting…';
        try {
            const r = await postAction('bank/auto_split', {});
            const res = (r && r.result) || {};
            _showAutoBankResult(res);
        } catch (e) {
            btn.textContent = 'error';
            console.error('auto-bank failed', e);
            setTimeout(() => { btn.textContent = 'AUTO-BANK A-H'; }, 3000);
        } finally {
            btn.disabled = false;
        }
    });
}

// ── Picker decisions panel ──────────────────────────────────────────
async function _loadPickerHistory() {
    const list = document.getElementById('picker-list');
    if (!list) return;
    try {
        const r = await postAction('picker/recent', { limit: 10 });
        const res = (r && r.result) || {};
        const picks = res.picks || [];
        if (!picks.length) {
            list.innerHTML = `<div class="picker-empty">
                no picks yet — auto-flip or hit > to fire some
            </div>`;
            return;
        }
        list.innerHTML = picks.map((p, i) => {
            const age = (p.age_s || 0) < 1
                ? 'now'
                : `${Math.floor(p.age_s || 0)}s ago`;
            const kindCls = `pick-kind-${(p.kind || 'other').replace(/[^a-z0-9-]/g, '')}`;
            return `<div class="picker-row ${kindCls}">
                <span class="pick-num">${i + 1}</span>
                <span class="pick-name">${escapeHTML(p.name || '?')}</span>
                <span class="pick-reason">${escapeHTML(p.reason || '')}</span>
                <span class="pick-age">${age}</span>
            </div>`;
        }).join('');
    } catch (e) {
        list.innerHTML = `<div class="picker-empty">
            failed: ${escapeHTML(String(e))}
        </div>`;
    }
}

async function _loadCohesionStatus(forceRefresh = false) {
    const filesEl = document.getElementById('cohesion-files');
    const timerEl = document.getElementById('cohesion-timer');
    if (!filesEl || !timerEl) return;
    try {
        const r = await postAction('cohesion/status',
            forceRefresh ? { refresh: true } : {});
        const s = (r && r.result) || {};
        if (!s.enabled) {
            filesEl.innerHTML = '<span class="cohesion-empty">cohesion off (config picker_cohesion_enabled=false)</span>';
            timerEl.textContent = '—';
            return;
        }
        const subset = s.subset || [];
        if (!subset.length) {
            filesEl.innerHTML = '<span class="cohesion-empty">no anchor yet — fire a clip to seed</span>';
            timerEl.textContent = '—';
            return;
        }
        const remaining = Math.round(s.seconds_remaining || 0);
        const lock = s.lock_seconds || 90;
        timerEl.textContent = `${remaining}s of ${lock}s remaining`;
        filesEl.innerHTML = subset.map(f => {
            const isSeed = f.path === s.seed;
            const mark = isSeed ? '★' : '·';
            const cls = isSeed ? 'cohesion-file seed' : 'cohesion-file';
            return `<div class="${cls}">
                <span class="cohesion-file-mark">${mark}</span>
                <span class="cohesion-file-name">${escapeHTML(f.name || '?')}</span>
            </div>`;
        }).join('');
    } catch (e) {
        timerEl.textContent = '(error)';
    }
}

function wirePickerPanel() {
    const panel = document.querySelector('.picker-panel');
    if (!panel || panel.dataset.wired) return;
    panel.dataset.wired = '1';
    let refreshInterval = null;
    panel.addEventListener('toggle', () => {
        if (panel.open) {
            _loadPickerHistory();
            _loadCohesionStatus();
            // Auto-refresh every 2s while open so new picks + cohesion
            // timer countdown stay live.
            refreshInterval = setInterval(() => {
                _loadPickerHistory();
                _loadCohesionStatus();
            }, 2000);
        } else if (refreshInterval) {
            clearInterval(refreshInterval);
            refreshInterval = null;
        }
    });
    // Manual refresh button on the cohesion strip
    const refreshBtn = document.getElementById('cohesion-refresh');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            _loadCohesionStatus(true);
        });
    }
}

// ── Votes & corrections panel ────────────────────────────────────────
// Shows what the hardware vote system has learned. Current clip's
// score + corrections at top so the user can see MK2 hold-gestures
// registering live. Top/bottom-voted lists below for historical view.
async function _loadVotesSummary() {
    const curEl = document.getElementById('votes-current');
    const topEl = document.getElementById('votes-top');
    const botEl = document.getElementById('votes-bottom');
    if (!curEl || !topEl || !botEl) return;
    try {
        const r = await postAction('votes/summary', { limit: 10 });
        const res = (r && r.result) || {};
        if (res.ok === false) {
            curEl.innerHTML = `<div class="votes-empty">
                ${escapeHTML(res.error || 'unavailable')}
            </div>`;
            return;
        }
        // Current clip block
        const c = res.current;
        if (c) {
            const corrEntries = Object.entries(c.corrections || {})
                .sort((a, b) => b[1] - a[1])
                .filter(([_, n]) => n > 0)
                .map(([l, n]) => `${l}=${n}`)
                .join(' ');
            const corrHtml = corrEntries
                ? `<span class="vote-corr">→ ${escapeHTML(corrEntries)}</span>`
                : '';
            const weightTag = c.picker_weight !== 1.0
                ? ` (weight ${c.picker_weight}x)`
                : '';
            curEl.innerHTML = `
                <div class="votes-current-name">${escapeHTML(c.name)}</div>
                <div class="votes-current-stats">
                    score ${c.score >= 0 ? '+' : ''}${c.score}${weightTag}
                    &nbsp;<span class="vote-up">↑${c.ups}</span>
                    &nbsp;<span class="vote-dn">↓${c.downs}</span>
                    &nbsp;${corrHtml}
                </div>`;
        } else {
            curEl.innerHTML = '<div class="votes-empty">no clip playing</div>';
        }
        const renderList = (rows, sign) => {
            if (!rows.length) return '<div class="votes-empty">—</div>';
            return rows.map(r => {
                const cls = sign > 0 ? 'pos' : 'neg';
                const score = (sign > 0 && r.score > 0 ? '+' : '')
                    + r.score;
                return `<div class="votes-row ${cls}">
                    <span class="votes-row-score">${score}</span>
                    <span class="votes-row-name">${escapeHTML(r.name)}</span>
                </div>`;
            }).join('');
        };
        topEl.innerHTML = renderList(res.top || [], +1);
        botEl.innerHTML = renderList(res.bottom || [], -1);
    } catch (e) {
        curEl.innerHTML = `<div class="votes-empty">
            failed: ${escapeHTML(String(e))}
        </div>`;
    }
}

function wireVotesPanel() {
    const panel = document.querySelector('.votes-panel');
    if (!panel || panel.dataset.wired) return;
    panel.dataset.wired = '1';
    let refreshInterval = null;
    panel.addEventListener('toggle', () => {
        if (panel.open) {
            _loadVotesSummary();
            refreshInterval = setInterval(_loadVotesSummary, 2000);
        } else if (refreshInterval) {
            clearInterval(refreshInterval);
            refreshInterval = null;
        }
    });
}

// ── CLIP semantic search ─────────────────────────────────────────────
// Natural-language query over the library's CLIP visual embeddings.
// First query loads the text encoder + matrix (~5s warmup); subsequent
// queries are sub-second. Tap a result row to fire that clip.
async function _runClipSearch() {
    const input = document.getElementById('clip-search-input');
    const btn = document.getElementById('clip-search-btn');
    const meta = document.getElementById('clip-search-meta');
    const list = document.getElementById('clip-search-results');
    if (!input || !btn || !list || !meta) return;
    const q = (input.value || '').trim();
    if (!q) {
        list.innerHTML = '<div class="clip-search-empty">enter a query first</div>';
        return;
    }
    btn.disabled = true;
    btn.textContent = '…';
    meta.textContent = 'searching…';
    list.innerHTML = '<div class="clip-search-empty">querying CLIP — first call ~5s, rest are instant</div>';
    try {
        const r = await postAction('clip/search', { query: q, top: 20 });
        const res = (r && r.result) || {};
        if (!res.ok) {
            meta.textContent = '';
            list.innerHTML = `<div class="clip-search-empty">error: ${escapeHTML(res.error || 'unknown')}</div>`;
            return;
        }
        const results = res.results || [];
        meta.textContent = `${results.length} results in ${res.elapsed_ms || 0}ms for "${res.query || q}"`;
        if (!results.length) {
            list.innerHTML = '<div class="clip-search-empty">no matches — embeddings may still be building. tail logs/clip_tagger.log</div>';
            return;
        }
        list.innerHTML = results.map(r => {
            const score = r.score >= 0 ? `+${r.score.toFixed(3)}` : r.score.toFixed(3);
            return `<div class="clip-search-row" data-path="${escapeHTML(r.path)}">
                <span class="clip-search-score">${score}</span>
                <span class="clip-search-name">${escapeHTML(r.name || '?')}</span>
            </div>`;
        }).join('');
        // Wire tap-to-fire on each row.
        list.querySelectorAll('.clip-search-row').forEach(row => {
            row.addEventListener('click', () => {
                const path = row.dataset.path;
                if (!path) return;
                // Reuse the existing load-file endpoint
                postAction('load', { path: path }).then(() => {
                    row.style.borderLeftColor = '#fbbf24';
                });
            });
        });
    } catch (e) {
        meta.textContent = '';
        list.innerHTML = `<div class="clip-search-empty">failed: ${escapeHTML(String(e))}</div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'SEARCH';
    }
}

// Take the current query, fire CLIP search, then categorically split
// the top-N visual matches into banks A-H. The user types a vibe and
// gets 8 instantly-playable banks. Shows the per-bank distribution
// inline as a status line.
async function _runClipToBanks() {
    const input = document.getElementById('clip-search-input');
    const btn = document.getElementById('clip-to-banks-btn');
    const meta = document.getElementById('clip-search-meta');
    if (!input || !btn || !meta) return;
    const q = (input.value || '').trim();
    if (!q) {
        meta.textContent = 'type a query first';
        return;
    }
    btn.disabled = true;
    btn.textContent = '…';
    meta.textContent = `routing "${q}" to banks A-H…`;
    try {
        const r = await postAction('bank/clip_to_banks', { q: q, top: 64 });
        const res = (r && r.result) || {};
        if (!res.ok) {
            meta.textContent = `error: ${res.error || 'unknown'}`;
            return;
        }
        const dist = res.distribution || {};
        const parts = ['A','B','C','D','E','F','G','H'].map(l => {
            const n = dist[l];
            if (n === undefined || n < 0) return `${l}·-`;
            return `${l}·${n}`;
        }).join(' ');
        meta.textContent = `→BANKS "${res.name || q}": ${res.matches} hits / ${res.elapsed_ms}ms search · ${parts}`;
    } catch (e) {
        meta.textContent = `failed: ${String(e)}`;
    } finally {
        btn.disabled = false;
        btn.textContent = '→BANKS';
    }
}

// ── Saved CLIP-search vibes (star + recall chips) ──────────────────
async function _saveCurrentVibe() {
    const input = document.getElementById('clip-search-input');
    const meta = document.getElementById('clip-search-meta');
    if (!input) return;
    const q = (input.value || '').trim();
    if (!q) { if (meta) meta.textContent = 'type a query first'; return; }
    const r = await postAction('saved_vibes/save', { q: q });
    const res = (r && r.result) || {};
    if (!res.ok) {
        if (meta) meta.textContent = `save failed: ${res.error || '?'}`;
        return;
    }
    if (meta) meta.textContent = `saved "${q}" — ${res.total} vibes total`;
    _renderSavedVibes();
}

// ── Curated theme chips ──────────────────────────────────────────────
// Tag-driven vibes (theme_apply.py). Tap a chip -> bank/theme loads
// that theme's clips, categorically split across banks A-H. Unlike the
// CLIP saved-vibes, these match branded/narrative themes ("horror")
// that visual search can't.
async function _renderThemes() {
    const strip = document.getElementById('theme-chips');
    if (!strip) return;
    const r = await postAction('theme/list', {});
    const res = (r && r.result) || {};
    const themes = (res && res.themes) || [];
    if (!themes.length) {
        strip.innerHTML = '<span class="saved-vibe-empty">no themes yet — run theme_apply.py --apply</span>';
        return;
    }
    strip.innerHTML = themes.map(t => {
        const name = escapeHTML(t.name);
        return `<button class="theme-chip" data-theme="${name}" title="load ${name} clips into banks A-H">${name} <span class="theme-chip-n">${t.count}</span></button>`;
    }).join('');
    strip.querySelectorAll('.theme-chip').forEach(chip => {
        chip.addEventListener('click', async () => {
            if (chip.classList.contains('theme-chip-loading')) return;
            chip.classList.add('theme-chip-loading');
            try {
                const rr = await postAction('bank/theme', { theme: chip.dataset.theme });
                const rres = (rr && rr.result) || {};
                if (rres && rres.ok) {
                    strip.querySelectorAll('.theme-chip').forEach(c => c.classList.remove('theme-chip-active'));
                    chip.classList.add('theme-chip-active');
                }
            } finally {
                chip.classList.remove('theme-chip-loading');
            }
        });
    });
}

async function _renderSavedVibes() {
    const strip = document.getElementById('saved-vibes');
    if (!strip) return;
    const r = await postAction('saved_vibes/list', {});
    const res = (r && r.result) || {};
    const vibes = (res && res.vibes) || [];
    if (!vibes.length) {
        strip.innerHTML = '<span class="saved-vibe-empty">⭐ star a query to save it here</span>';
        return;
    }
    strip.innerHTML = vibes.map(v => {
        const q = escapeHTML(v.q);
        return `<button class="saved-vibe-chip" data-q="${q}" title="tap = re-fire to banks · long-press = delete">${q}</button>`;
    }).join('');
    // Wire taps + long-press
    strip.querySelectorAll('.saved-vibe-chip').forEach(chip => {
        const q = chip.dataset.q;
        let longPressTimer = null;
        let longPressFired = false;
        const startPress = () => {
            longPressFired = false;
            longPressTimer = setTimeout(async () => {
                longPressFired = true;
                if (confirm(`Delete saved vibe "${q}"?`)) {
                    await postAction('saved_vibes/delete', { q: q });
                    _renderSavedVibes();
                }
            }, 700);
        };
        const cancelPress = () => {
            if (longPressTimer) {
                clearTimeout(longPressTimer);
                longPressTimer = null;
            }
        };
        chip.addEventListener('mousedown', startPress);
        chip.addEventListener('touchstart', startPress, { passive: true });
        chip.addEventListener('mouseup', cancelPress);
        chip.addEventListener('mouseleave', cancelPress);
        chip.addEventListener('touchend', cancelPress);
        chip.addEventListener('click', async (e) => {
            if (longPressFired) { e.preventDefault(); return; }
            // Re-fire this vibe to banks
            const input = document.getElementById('clip-search-input');
            if (input) input.value = q;
            const meta = document.getElementById('clip-search-meta');
            if (meta) meta.textContent = `routing "${q}" to banks…`;
            await postAction('bank/clip_to_banks', { q: q, top: 64 })
                .then(r => {
                    const res = (r && r.result) || {};
                    if (!res.ok) {
                        if (meta) meta.textContent = `error: ${res.error || '?'}`;
                        return;
                    }
                    const dist = res.distribution || {};
                    const parts = ['A','B','C','D','E','F','G','H']
                        .map(l => {
                            const n = dist[l];
                            if (n === undefined || n < 0) return `${l}·-`;
                            return `${l}·${n}`;
                        }).join(' ');
                    if (meta) meta.textContent =
                        `→BANKS "${res.name || q}": ${res.matches} hits · ${parts}`;
                });
        });
    });
}

async function _runOpeners() {
    const btn = document.getElementById('openers-btn');
    const meta = document.getElementById('clip-search-meta');
    if (!btn || !meta) return;
    btn.disabled = true;
    const orig = btn.textContent;
    btn.textContent = '…';
    meta.textContent = 'building OPENERS banks from intro-tagged files…';
    try {
        const r = await postAction('bank/openers', {});
        const res = (r && r.result) || {};
        if (!res.ok) {
            meta.textContent = `OPENERS error: ${res.error || 'unknown'}`;
            return;
        }
        const dist = res.distribution || {};
        const parts = ['A','B','C','D','E','F','G','H'].map(l => {
            const n = dist[l];
            if (n === undefined || n < 0) return `${l}·-`;
            return `${l}·${n}`;
        }).join(' ');
        meta.textContent = `OPENERS: ${res.total} files, split by intro-length · ${parts}`;
    } catch (e) {
        meta.textContent = `failed: ${String(e)}`;
    } finally {
        btn.disabled = false;
        btn.textContent = orig;
    }
}

function wireClipSearchPanel() {
    // Panel id is 'clip-search-panel-top' since the 2026-05-19 restructure
    // promoted it to a top-level section. The old '.clip-search-panel'
    // class selector silently matched nothing -> SEARCH, Enter and the
    // saved-vibe chips all went unwired. (Fixed 2026-05-20.)
    const panel = document.getElementById('clip-search-panel-top');
    if (!panel || panel.dataset.wired) return;
    panel.dataset.wired = '1';
    const input = document.getElementById('clip-search-input');
    const btn = document.getElementById('clip-search-btn');
    const toBanksBtn = document.getElementById('clip-to-banks-btn');
    const saveBtn = document.getElementById('clip-save-btn');
    const openersBtn = document.getElementById('openers-btn');
    if (btn) btn.addEventListener('click', _runClipSearch);
    if (toBanksBtn) toBanksBtn.addEventListener('click', _runClipToBanks);
    if (saveBtn) saveBtn.addEventListener('click', _saveCurrentVibe);
    if (openersBtn) openersBtn.addEventListener('click', _runOpeners);
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                // SHIFT+Enter = route to banks; plain Enter = preview results
                if (e.shiftKey) _runClipToBanks();
                else _runClipSearch();
            }
        });
    }
    _renderSavedVibes();
    _renderThemes();
}

// ── MagicMix top-N suggestions ───────────────────────────────────────
async function _loadMagicMix() {
    const list = document.getElementById('magic-list');
    if (!list) return;
    list.innerHTML = '<div class="magic-empty">thinking…</div>';
    try {
        const r = await postAction('magic/suggest', { top_n: 5 });
        const res = (r && r.result) || {};
        const sugg = res.suggestions || [];
        if (!sugg.length) {
            list.innerHTML = `<div class="magic-empty">
                no suggestions (active bank may be empty)
            </div>`;
            return;
        }
        list.innerHTML = sugg.map((s, i) => {
            const fname = (s.path || '').split(/[\\\/]/).pop() || '';
            const reasons = (s.reasons || []).slice(0, 3)
                .map(rs => `<span class="mm-reason">${escapeHTML(rs)}</span>`)
                .join('');
            return `<button type="button" class="magic-row"
                data-path="${escapeHTML(s.path || '')}"
                title="${escapeHTML(fname)}  score=${(s.score || 0).toFixed(2)}">
                <span class="mm-rank">${i + 1}</span>
                <span class="mm-name">${escapeHTML(s.name || fname)}</span>
                <span class="mm-score">${(s.score || 0).toFixed(2)}</span>
                <span class="mm-reasons">${reasons}</span>
            </button>`;
        }).join('');
        list.querySelectorAll('.magic-row').forEach(btn => {
            btn.addEventListener('click', async () => {
                const path = btn.dataset.path;
                if (!path) return;
                btn.classList.add('magic-firing');
                try {
                    await postAction('scratch/fire', { path });
                } catch (e) {
                    console.error('magic fire failed', e);
                }
                setTimeout(() => btn.classList.remove('magic-firing'), 800);
            });
        });
    } catch (e) {
        list.innerHTML = `<div class="magic-empty">
            failed: ${escapeHTML(String(e))}
        </div>`;
    }
}

function wireMagicMix() {
    const panel = document.querySelector('.magic-mix-panel');
    const refresh = document.getElementById('magic-refresh');
    if (panel && !panel.dataset.wired) {
        panel.dataset.wired = '1';
        panel.addEventListener('toggle', () => {
            if (panel.open) _loadMagicMix();
        });
    }
    if (refresh && !refresh.dataset.wired) {
        refresh.dataset.wired = '1';
        refresh.addEventListener('click', _loadMagicMix);
    }
}

// ── Openers launcher ────────────────────────────────────────────────
// Lists clips tagged `opener` / `transition-rich`. Tap = fire that
// specific clip segment via /api/bank/fire (which honors in/out marks).
// Re-fetches each time the panel is expanded so newly-marked clips
// appear without a page refresh.
function _fmtOpenerTime(seconds) {
    const s = Math.max(0, Math.floor(seconds || 0));
    const m = Math.floor(s / 60);
    const rs = String(s % 60).padStart(2, '0');
    return `${m}:${rs}`;
}

// Active tag for the openers panel. Defaults to 'opener'; pills swap it.
let _openersActiveTag = 'opener';

async function _loadOpeners() {
    const list = document.getElementById('openers-list');
    const countEl = document.getElementById('openers-count');
    const titleEl = document.getElementById('openers-title');
    if (!list) return;
    if (titleEl) titleEl.textContent = _openersActiveTag.toUpperCase();
    list.innerHTML = '<div class="openers-empty">loading…</div>';
    try {
        const r = await postAction('clips/by_tag', { tag: _openersActiveTag });
        const res = (r && r.result) || {};
        const clips = res.clips || [];
        if (countEl) {
            countEl.textContent = `${clips.length} clip${clips.length === 1 ? '' : 's'}`;
        }
        if (!clips.length) {
            list.innerHTML = `<div class="openers-empty">
                No clips tagged '${escapeHTML(_openersActiveTag)}'.
                Mark some via the form below.
            </div>`;
            return;
        }
        list.innerHTML = clips.map(c => {
            const star = c.starred ? '★' : '';
            const fname = (c.filepath || '').split(/[\\\/]/).pop() || '';
            const inT = _fmtOpenerTime(c.in_seconds);
            const outT = _fmtOpenerTime(c.out_seconds);
            const tagBadges = (c.tags || [])
                .filter(t => t !== _openersActiveTag)
                .map(t => `<span class="op-tag">${escapeHTML(t)}</span>`)
                .join('');
            // Row is a DIV (not button) so the inline ✕ delete can
            // stopPropagation -- a button-inside-button is invalid HTML.
            return `<div class="opener-row" tabindex="0"
                data-clip-id="${escapeHTML(c.id || '')}"
                title="${escapeHTML(fname)} [${inT}-${outT}]">
                <span class="op-star">${star}</span>
                <span class="op-name">${escapeHTML(c.name || fname)}</span>
                <span class="op-time">${inT}-${outT}</span>
                <span class="op-tags">${tagBadges}</span>
                <button type="button" class="op-del"
                    data-clip-id="${escapeHTML(c.id || '')}"
                    aria-label="delete clip"
                    title="delete this clip mark (not the file)">&times;</button>
            </div>`;
        }).join('');
        // Fire on row click (excluding the delete button)
        list.querySelectorAll('.opener-row').forEach(row => {
            row.addEventListener('click', async (ev) => {
                if (ev.target.classList.contains('op-del')) return;
                const clipId = row.dataset.clipId;
                row.classList.add('opener-firing');
                try {
                    await postAction('bank/fire', { clip_id: clipId });
                } catch (e) {
                    console.error('opener fire failed', e);
                }
                setTimeout(() => row.classList.remove('opener-firing'), 800);
            });
        });
        // Delete clip on ✕ click
        list.querySelectorAll('.op-del').forEach(btn => {
            btn.addEventListener('click', async (ev) => {
                ev.stopPropagation();
                if (!confirm('Delete this clip mark? (file itself stays)')) return;
                const clipId = btn.dataset.clipId;
                try {
                    await postAction('clip/delete', { clip_id: clipId });
                    _loadOpeners();   // refresh list
                } catch (e) {
                    console.error('delete failed', e);
                }
            });
        });
    } catch (e) {
        list.innerHTML = `<div class="openers-empty">
            failed to load: ${escapeHTML(String(e))}
        </div>`;
    }
}

function wireOpenersLauncher() {
    const details = document.getElementById('openers-details');
    if (!details || details.dataset.wired) return;
    details.dataset.wired = '1';
    // Fetch the count once at boot so the summary shows e.g. "6 clips"
    // even before the user opens the panel.
    _loadOpeners();
    // Re-fetch every time the panel is opened (newly marked clips
    // should appear without a manual refresh).
    details.addEventListener('toggle', () => {
        if (details.open) {
            _loadOpeners();
        }
    });
    // Tag-filter pills: tap to switch which tag we're viewing.
    const pillsEl = document.getElementById('openers-pills');
    if (pillsEl && !pillsEl.dataset.wired) {
        pillsEl.dataset.wired = '1';
        pillsEl.querySelectorAll('.op-pill').forEach(p => {
            p.addEventListener('click', () => {
                _openersActiveTag = p.dataset.tag || 'opener';
                pillsEl.querySelectorAll('.op-pill').forEach(
                    o => o.classList.toggle('active', o === p));
                _loadOpeners();
            });
        });
    }
    // Mark-clip form: submit + clear + show result.
    const submitBtn = document.getElementById('mc-submit');
    const resultEl = document.getElementById('mc-result');
    if (submitBtn && !submitBtn.dataset.wired) {
        submitBtn.dataset.wired = '1';
        submitBtn.addEventListener('click', async () => {
            const fileVal = document.getElementById('mc-file').value.trim();
            const inVal = parseFloat(document.getElementById('mc-in').value || '0');
            const outVal = parseFloat(document.getElementById('mc-out').value || '0');
            const nameVal = document.getElementById('mc-name').value.trim();
            const tagsVal = document.getElementById('mc-tags').value.trim();
            const starredVal = document.getElementById('mc-starred').checked;
            if (!fileVal) { resultEl.textContent = '✗ need a file'; return; }
            if (!(outVal > inVal)) { resultEl.textContent = '✗ OUT must be > IN'; return; }
            const tagList = tagsVal
                ? tagsVal.split(',').map(s => s.trim()).filter(Boolean)
                : ['opener', 'transition-rich'];
            if (starredVal && !tagList.includes('starred-opener')) {
                tagList.push('starred-opener');
            }
            resultEl.textContent = 'creating…';
            submitBtn.disabled = true;
            try {
                const r = await postAction('clips/create_segment', {
                    file: fileVal, in: inVal, out: outVal,
                    name: nameVal, tags: tagList, starred: starredVal,
                });
                const res = (r && r.result) || {};
                if (res.ok) {
                    const c = res.clip || {};
                    resultEl.textContent =
                        `✓ ${c.name}  [${Math.floor(inVal)}-${Math.floor(outVal)}s]`;
                    // Clear form for next entry
                    document.getElementById('mc-file').value = '';
                    document.getElementById('mc-in').value = '';
                    document.getElementById('mc-out').value = '';
                    document.getElementById('mc-name').value = '';
                    document.getElementById('mc-tags').value = '';
                    document.getElementById('mc-starred').checked = false;
                    // Refresh openers list so new clip appears
                    _loadOpeners();
                } else {
                    let msg = `✗ ${res.error || 'failed'}`;
                    if (res.matches) msg += ` (${res.matches.slice(0,3).join(', ')})`;
                    resultEl.textContent = msg;
                }
            } catch (e) {
                resultEl.textContent = `✗ ${e}`;
            } finally {
                submitBtn.disabled = false;
            }
        });
    }
}

function wireTagSearchInput() {
    const input = document.getElementById('tag-search-input');
    const resultsDiv = document.getElementById('tag-search-results');
    if (!input || input.dataset.wired) return;
    input.dataset.wired = '1';
    let searchTimer = null;
    const doLiveSearch = async (q) => {
        if (!q.trim() || !resultsDiv) {
            if (resultsDiv) resultsDiv.innerHTML = '';
            return;
        }
        try {
            const r = await postAction('path_tags/search',
                                       { q: q.trim(), limit: 10 });
            const res = (r && r.result) || {};
            const list = res.results || [];
            if (!list.length) {
                resultsDiv.innerHTML =
                    `<span class="tag-search-empty">no tags match "${q}"</span>`;
                return;
            }
            // Render matching tags as mini-chips. Click = run tag_to_banks
            // with that tag directly (skip the best-match fuzzy step).
            resultsDiv.innerHTML = list.map(t =>
                `<button type="button" class="tag-search-hit"
                    data-tag="${t.tag}">${t.tag}
                    <span class="tsh-count">${t.count}</span></button>`
            ).join('');
            resultsDiv.querySelectorAll('.tag-search-hit').forEach(b => {
                b.addEventListener('click', async () => {
                    const tag = b.dataset.tag;
                    input.value = tag;
                    resultsDiv.innerHTML = '';
                    await _runTagToBanks(tag);
                });
            });
        } catch (e) {
            console.error('tag search failed', e);
        }
    };
    input.addEventListener('input', () => {
        clearTimeout(searchTimer);
        const q = input.value;
        // Debounce live search 200ms
        searchTimer = setTimeout(() => doLiveSearch(q), 200);
    });
    input.addEventListener('keydown', async (ev) => {
        if (ev.key === 'Enter') {
            ev.preventDefault();
            const q = input.value.trim();
            if (!q) return;
            await _runTagToBanks(q);
        } else if (ev.key === 'Escape') {
            input.value = '';
            if (resultsDiv) resultsDiv.innerHTML = '';
        }
    });
}

// ── Phrase ticker: latest spotter hit ────────────────────────────
// Polls /api/phrase/recent every 3s (spotter latency is already in
// seconds; 500ms /api/state poll cadence would be overkill). Shows
// the most recent hit with age countdown, dims as it ages, hides
// after 30s. Element is empty when spotter is off or has no hits.
function wirePhraseTicker() {
    const el = document.getElementById('ss-phrase');
    if (!el || el.dataset.wired) return;
    el.dataset.wired = '1';
    const POLL_MS = 3000;
    const tick = async () => {
        try {
            const r = await fetch('/api/phrase/recent?within=30',
                                  { cache: 'no-store' });
            const d = await r.json();
            const running = !!d.running;
            const hits = d.hits || [];
            if (!running) {
                el.textContent = '';
                el.className = 'ss-phrase ss-phrase-idle';
                return;
            }
            if (!hits.length) {
                // Spotter is listening but nothing heard recently.
                el.textContent = '♪ listening';
                el.className = 'ss-phrase ss-phrase-listening';
                return;
            }
            const h = hits[0];
            const age = Math.floor(h.age_s || 0);
            const label = (h.phrase || h.slug || '').toString();
            const short = label.length > 28
                ? label.slice(0, 27) + '…' : label;
            el.textContent = `♪ ${short} (${age}s)`;
            // Color tier: <4s = bright orange, <10s = mid amber,
            // <30s = dim gray.
            let tier = 'fresh';
            if (age > 10) tier = 'stale';
            else if (age > 4) tier = 'mid';
            el.className = `ss-phrase ss-phrase-${tier}`;
        } catch (e) {
            el.textContent = '';
            el.className = 'ss-phrase ss-phrase-idle';
        }
    };
    tick();
    setInterval(tick, POLL_MS);
}

// ── ARC badge: tap = cycle phase, long-press = toggle auto-detect ────
function wireArcBadge() {
    const el = document.getElementById('ss-arc');
    if (!el || el.dataset.wired) return;
    el.dataset.wired = '1';
    let pressTimer = null;
    let longPressed = false;
    const LONG_PRESS_MS = 550;
    const post = (path) => fetch('/api' + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}'
    }).catch(e => console.error('arc post failed', e));
    const onDown = () => {
        longPressed = false;
        clearTimeout(pressTimer);
        pressTimer = setTimeout(() => {
            longPressed = true;
            post('/set-arc/auto');
        }, LONG_PRESS_MS);
    };
    const onUp = (ev) => {
        clearTimeout(pressTimer);
        if (!longPressed) {
            if (ev && ev.preventDefault) ev.preventDefault();
            post('/set-arc/cycle');
        }
        longPressed = false;
    };
    const onCancel = () => { clearTimeout(pressTimer); longPressed = false; };
    // Touch events. preventDefault on touchend in ALL cases (not just
    // tap) -- without that, iOS Safari synthesises a mouse event ~300ms
    // later that re-fires onDown+onUp and undoes the long-press action.
    el.addEventListener('touchstart', onDown, { passive: true });
    el.addEventListener('touchend',   (ev) => {
        if (ev.cancelable) ev.preventDefault();
        onUp(ev);
    });
    el.addEventListener('touchcancel', onCancel);
    // Mouse fallback for desktop testing.
    el.addEventListener('mousedown',  onDown);
    el.addEventListener('mouseup',    onUp);
    el.addEventListener('mouseleave', onCancel);
}

async function _runTagToBanks(q) {
    const btn = document.getElementById('auto-bank-btn');
    const input = document.getElementById('tag-search-input');
    const resultsDiv = document.getElementById('tag-search-results');
    if (btn) {
        btn.disabled = true;
        btn.textContent = `banking "${q}"…`;
    }
    try {
        const r = await postAction('bank/tag_to_banks', { q });
        const res = (r && r.result) || {};
        _showAutoBankResult(res);
        if (res.ok && resultsDiv) {
            resultsDiv.innerHTML = '';
            if (input) input.blur();
        }
    } catch (e) {
        if (btn) {
            btn.textContent = 'error';
            setTimeout(() => { btn.textContent = 'AUTO-BANK A-H'; }, 3000);
        }
        console.error('tag_to_banks failed', e);
    } finally {
        if (btn) btn.disabled = false;
    }
}

// ── Scratch (session basket of library files) ───────────────────────
let _lastScratchSig = "";
let _scratchWired = false;
let _scratchPaths = new Set();  // for library-row +/- toggle state
function renderScratch(items) {
    const list = document.getElementById('scratch-list');
    if (!list) return;
    const sig = JSON.stringify((items || []).map(i => i.path));
    _scratchPaths = new Set((items || []).map(i => i.path));
    if (sig !== _lastScratchSig) {
        _lastScratchSig = sig;
        if (!items || items.length === 0) {
            list.innerHTML = '<li class="empty">Nothing in scratch yet</li>';
        } else {
            list.innerHTML = items.map(it => {
                const path = escapeHTML(it.path || '');
                const name = escapeHTML(it.name || it.path || '');
                return `<li><span class="scratch-chip">`
                     + `<button type="button" class="scratch-fire" data-path="${path}" title="Fire to LIVE">${name}</button>`
                     + `<button type="button" class="scratch-remove" data-path="${path}" title="Remove from scratch">×</button>`
                     + `</span></li>`;
            }).join('');
            list.querySelectorAll('.scratch-fire').forEach(btn => {
                btn.addEventListener('click', () => {
                    postAction('scratch/fire', { path: btn.dataset.path }).catch(() => {});
                });
            });
            list.querySelectorAll('.scratch-remove').forEach(btn => {
                btn.addEventListener('click', (ev) => {
                    ev.stopPropagation();
                    postAction('scratch/remove', { path: btn.dataset.path }).catch(() => {});
                });
            });
        }
    }
    if (!_scratchWired) {
        const addBtn = document.getElementById('scratch-add-current');
        const clearBtn = document.getElementById('scratch-clear');
        const shuffleBtn = document.getElementById('scratch-shuffle');
        const saveBtn = document.getElementById('scratch-save-set');
        if (addBtn && clearBtn) {
            _scratchWired = true;
            addBtn.addEventListener('click', () => {
                postAction('scratch/add_current', {}).catch(() => {});
            });
            clearBtn.addEventListener('click', () => {
                if (confirm('Clear the scratch list?')) {
                    postAction('scratch/clear', {}).catch(() => {});
                }
            });
            if (shuffleBtn) {
                shuffleBtn.addEventListener('click', () => {
                    postAction('scratch/shuffle', {}).catch(() => {});
                });
            }
            if (saveBtn) {
                saveBtn.addEventListener('click', () => {
                    const name = prompt('Name this scratch set:', '');
                    if (name !== null) {
                        postAction('scratch/save_set', { name: name.trim() }).catch(() => {});
                    }
                });
            }
        }
    }
}

// ── Banks A-H (8-slot quick-switch for scratch lists) ───────────────
let _lastBanksSig = "";
function renderBanks(slots) {
    const row = document.getElementById('banks-row');
    if (!row) return;
    const sig = JSON.stringify((slots || []).map(s => [s.letter, s.name, s.count, s.active]));
    // Track active bank globally so the bank-preview drawer can highlight
    // it in its left-side grid (separate from the peeked bank).
    const activeSlot = (slots || []).find(s => s.active);
    const newActive = (activeSlot && activeSlot.letter) || '';
    const prevActive = window._lastActiveBank || '';
    window._lastActiveBank = newActive;
    // Auto-peek when active bank changes — keeps the always-visible
    // drawer in sync with what's loaded. First-time load (no peeked yet)
    // also kicks the peek so the drawer isn't blank on startup.
    // (2026-05-16 v2 — bank drawer is the primary view, not a popup.)
    if (newActive && (newActive !== prevActive || !_lastPeekedLetter)) {
        peekBank(newActive);
    }
    // Horizontal banks-row is hidden in HTML now; renderBanks is a
    // no-op for the row but still tracks _lastActiveBank above.
    if (sig === _lastBanksSig) return;
    _lastBanksSig = sig;
    if (!slots || !slots.length) { row.innerHTML = ''; return; }
    row.innerHTML = slots.map(s => {
        const ltr = escapeHTML(s.letter || '?');
        const name = escapeHTML(s.name || '');
        const cnt = Number(s.count || 0);
        const active = !!s.active;
        const hasContent = cnt > 0;
        const cls = ['bank-slot'];
        if (active) cls.push('active');
        if (hasContent) cls.push('has-content');
        // Chip body = load gesture (unchanged). bs-peek button at the
        // end = preview gesture (new 2026-05-16). Stop-propagation on
        // peek so tapping it doesn't ALSO trigger a load.
        return `<div class="${cls.join(' ')}" data-letter="${ltr}" data-count="${cnt}"
            title="${name ? name + ' · ' : ''}${cnt} files. Tap to load. Tap ▸ to preview. Shift+tap (or long-press) to save current.">`
             + `<span class="bs-letter">${ltr}</span>`
             + `${name ? `<span class="bs-name">${name}</span>` : (hasContent ? '' : '<span class="bs-save-hint">empty</span>')}`
             + `<span class="bs-count">${cnt || ''}</span>`
             + (hasContent
                 ? `<button type="button" class="bs-peek" data-letter="${ltr}"
                            title="Preview without loading">▸</button>`
                 : '')
             + `</div>`;
    }).join('');
    row.querySelectorAll('.bank-slot').forEach(el => {
        let pressTimer = null;
        const ltr = el.dataset.letter;
        const cnt = Number(el.dataset.count || 0);
        const doSave = () => {
            const name = prompt(`Name for bank ${ltr}:`, '') || '';
            postAction('bank/save', { letter: ltr, name }).catch(() => {});
        };
        const doLoad = () => {
            if (cnt === 0) {
                // Empty bank: same gesture as Save (initial population).
                doSave();
            } else {
                postAction('bank/load', { letter: ltr }).catch(() => {});
            }
        };
        el.addEventListener('click', (ev) => {
            if (ev.shiftKey || ev.altKey || ev.metaKey) {
                doSave();
            } else {
                doLoad();
            }
        });
        // Long-press to save (iPad-friendly since shift-tap isn't native).
        el.addEventListener('touchstart', () => {
            pressTimer = setTimeout(doSave, 700);
        });
        el.addEventListener('touchend', () => {
            if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; }
        });
        el.addEventListener('touchmove', () => {
            if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; }
        });
    });
    // Peek buttons get their own wiring — stopPropagation prevents the
    // chip-body click handler from ALSO firing a load. Long-press on
    // peek is ignored (no save semantics from inside peek).
    row.querySelectorAll('.bs-peek').forEach(btn => {
        btn.addEventListener('click', (ev) => {
            ev.stopPropagation();
            peekBank(btn.dataset.letter);
        });
        // Suppress the parent's touchstart -> long-press -> save flow.
        btn.addEventListener('touchstart', (ev) => ev.stopPropagation());
    });
}

// ── Bank Preview drawer ──────────────────────────────────────────────
// User can tap ▸ on any bank chip to peek its contents without
// loading. Drawer renders below the banks-row; "Load" commits.
// (2026-05-16 — built so the user can "see exactly what I want to
// take next" before committing.)

let _lastPeekedLetter = '';

// Auto-retry for bank-preview thumbs. Server kicks a background
// ffmpeg backfill on the first peek of a never-browsed bank; thumbs
// take ~1-2s/file to generate. Without this, the iPad would 404 →
// hide → never refetch. We retry up to 6 times at 3s intervals with
// a cache-buster query so the browser actually re-hits the server
// (it caches 404s aggressively otherwise). After 6 fails the slot
// shows the hatched placeholder permanently.
window._bpThumbRetry = function(img) {
    const retry = parseInt(img.dataset.retry || '0', 10);
    const hash = img.dataset.hash || '';
    if (retry >= 6 || !hash) {
        img.classList.add('bp-thumb-missing');
        img.removeAttribute('src');
        return;
    }
    img.dataset.retry = String(retry + 1);
    setTimeout(() => {
        img.src = `/lib_thumbnails/${hash}.jpg?r=${Date.now()}`;
    }, 3000);
};

async function peekBank(letter) {
    if (!letter) return;
    try {
        const r = await fetch('/api/bank/peek', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ letter }),
        });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        // The HTTP handler wraps every action's return value as
        // {ok, result: <handler-return>} — unwrap before rendering.
        const wrapped = await r.json();
        if (!wrapped.ok) throw new Error(wrapped.error || 'peek failed');
        const data = wrapped.result || {};
        if (!data.ok) throw new Error(data.error || 'peek failed');
        _renderBankPreview(data);
    } catch (e) {
        showError('Peek failed: ' + e.message);
    }
}

// Verticals (8 folder shortcuts + reroll). NOT cached -- labels
// change when the active MK2 page changes (mk2_vertical_pages binds
// different folders per page), so we re-fetch on every bank-preview
// render. The /api/mk2/verticals endpoint is cheap; this is fine.
async function _ensureVerticals() {
    try {
        const r = await fetch('/api/mk2/verticals', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: '{}',
        });
        const wrapped = await r.json();
        return wrapped.result || { verticals: [], page_name: '', active_page: 0 };
    } catch (e) {
        return { verticals: [], page_name: '', active_page: 0 };
    }
}

function _renderBankPreview(data) {
    const panel = document.getElementById('bank-preview');
    if (!panel) return;
    _lastPeekedLetter = data.letter || '';
    document.getElementById('bp-letter').textContent = data.letter || '?';
    document.getElementById('bp-name').textContent = data.name || '(unnamed)';
    document.getElementById('bp-count').textContent =
        `${data.count || 0} clip${data.count === 1 ? '' : 's'}`;
    // === LEFT: bank-group grid (A-H, 2 cols × 4 rows = MK2 layout) ===
    // Active = whichever bank is loaded right now (different from peeked).
    // Peeked = the one currently shown in the drawer. Both get visual
    // styles so the user knows which is which.
    const groups = document.getElementById('bp-groups');
    const peeked = (data.letter || '').toUpperCase();
    const activeBank = (window._lastActiveBank || '').toUpperCase();
    if (groups) {
        const letters = ['A','B','C','D','E','F','G','H'];
        groups.innerHTML = letters.map(L => {
            const cls = ['bp-group'];
            if (L === activeBank) cls.push('active');
            if (L === peeked) cls.push('peeked');
            return `<button type="button" class="${cls.join(' ')}"
                            data-letter="${L}" title="Peek bank ${L}">${L}</button>`;
        }).join('');
        groups.querySelectorAll('.bp-group').forEach(btn => {
            btn.addEventListener('click', () => peekBank(btn.dataset.letter));
            _wireGroupLongPress(btn);
        });
    }
    // === MIDDLE: verticals (8 folder shortcuts + reroll) ===
    // Re-fetched on every bank-preview render so it tracks active
    // MK2 page changes. Tap fires the same action as the physical
    // MK2 vertical-button press would.
    _ensureVerticals().then(vinfo => {
        const col = document.getElementById('bp-verts');
        if (!col) return;
        const verts = (vinfo && vinfo.verticals) || [];
        const pageName = (vinfo && vinfo.page_name) || '';
        const activePage = (vinfo && vinfo.active_page) || 0;
        // Tiny page header row so the operator knows which page
        // these vertical labels belong to.
        let html = '';
        if (pageName) {
            html += `<div class="bp-verts-header"
                title="MK2 active page ${activePage + 1}: ${escapeHTML(pageName)}">P${activePage + 1} ${escapeHTML(pageName)}</div>`;
        }
        html += verts.map(v => {
            const label = escapeHTML(v.label || '');
            const idx = Number(v.idx || 0);
            const cls = label ? 'bp-vert' : 'bp-vert bp-vert-empty';
            return `<button type="button" class="${cls}" data-idx="${idx}"
                            title="Vertical ${idx + 1}: ${label || '(unbound)'}">${label || '·'}</button>`;
        }).join('');
        col.innerHTML = html;
        col.querySelectorAll('.bp-vert').forEach(btn => {
            btn.addEventListener('click', () => {
                const idx = Number(btn.dataset.idx);
                fetch('/api/mk2/fire_vertical', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ idx }),
                }).then(() => {
                    // Refresh the peek after a beat — vertical fires a
                    // folder-load into the ACTIVE bank, so peek of the
                    // active one will refresh; others unchanged.
                    setTimeout(() => peekBank(_lastPeekedLetter), 500);
                }).catch(e => showError('Vertical failed: ' + e.message));
            });
        });
    });
    // === RIGHT: pads grid (4×4 thumbs of the peeked bank) ===
    const list = document.getElementById('bp-pads');
    if (!data.files || !data.files.length) {
        list.innerHTML = '<div class="bp-empty">Bank is empty</div>';
        return;
    }
    {
        // Thumb-first grid layout (2026-05-16 user req: "would like
        // that to be the default just a thumb there rather than name").
        // Each card = big thumb + badges as corner overlays + filename
        // demoted to a small caption below the thumb. Auto-retries
        // thumbs that 404 (cold-cache files mid-backfill) so they pop
        // in over the next 30-60s without needing a re-peek.
        list.innerHTML = data.files.map((f, i) => {
            const name = (f && f.name) || String(f || '');
            const bpm = Number((f && f.bpm) || 0);
            const cues = Number((f && f.cues) || 0);
            const starred = Number((f && f.starred) || 0);
            const hash = (f && f.hash) || '';
            const safeName = escapeHTML(name);
            const safeHash = escapeHTML(hash);
            // Badges as corner overlays on the thumb.
            const corners = [];
            if (bpm > 0) corners.push(`<span class="bp-cnr bp-cnr-bpm">${bpm}</span>`);
            if (starred > 0) corners.push(`<span class="bp-cnr bp-cnr-star" title="${starred} starred">★${starred > 1 ? starred : ''}</span>`);
            if (cues > 0) corners.push(`<span class="bp-cnr bp-cnr-cues" title="${cues} cue points">${cues}c</span>`);
            // Thumb with auto-retry: onerror schedules a refetch with
            // a cache-buster query, so as backfill completes thumbs
            // pop in without the user re-peeking. Retry count is
            // capped in data-retry to avoid infinite loops.
            const thumb = hash
                ? `<img class="bp-thumb" loading="lazy" alt=""
                        src="/lib_thumbnails/${safeHash}.jpg"
                        data-hash="${safeHash}" data-retry="0"
                        onerror="_bpThumbRetry(this)">`
                : '<span class="bp-thumb bp-thumb-missing"></span>';
            // Square pad-shaped thumb, no caption — the filename
            // lives in the card `title` (hover/long-press) so the
            // grid stays a clean quick-reference mirror of the MK2's
            // 4×4 pads. (2026-05-20 user req: drop the long names.)
            return `<div class="bp-card" title="${safeName}">`
                 + `<div class="bp-thumb-wrap">${thumb}`
                 + (corners.length ? `<div class="bp-corners">${corners.join('')}</div>` : '')
                 + `</div>`
                 + `</div>`;
        }).join('');
    }
    panel.classList.remove('hidden');
}

function _loadPeekedBank() {
    if (!_lastPeekedLetter) return;
    const letter = _lastPeekedLetter;
    postAction('bank/load', { letter })
        .catch((e) => showError('Load failed: ' + e.message));
}

async function _rerollPeekedBank() {
    if (!_lastPeekedLetter) return;
    const letter = _lastPeekedLetter;
    const btn = document.getElementById('bp-reroll');
    if (btn) { btn.disabled = true; btn.textContent = '…'; }
    try {
        const r = await postAction('bank/reroll', { letter });
        const res = (r && r.result) || {};
        if (btn) {
            btn.textContent = res.ok ? '✓' : '!';
            setTimeout(() => { btn.textContent = '↻'; btn.disabled = false; }, 1200);
        }
        if (res.ok) {
            // Refresh peek so the new contents show.
            peekBank(letter);
        }
    } catch (e) {
        if (btn) {
            btn.textContent = '!';
            setTimeout(() => { btn.textContent = '↻'; btn.disabled = false; }, 1500);
        }
        console.error('reroll failed', e);
    }
}

function wireBankPreview() {
    // No close button anymore — drawer is always visible (2026-05-16 v2).
    const loadBtn = document.getElementById('bp-load');
    if (loadBtn && !loadBtn.dataset.wired) {
        loadBtn.dataset.wired = '1';
        loadBtn.addEventListener('click', _loadPeekedBank);
    }
    const rerollBtn = document.getElementById('bp-reroll');
    if (rerollBtn && !rerollBtn.dataset.wired) {
        rerollBtn.dataset.wired = '1';
        rerollBtn.addEventListener('click', _rerollPeekedBank);
    }
    // Layer overlay chip (2026-05-18). Tap = cycle to next layer.
    // The chip label is refreshed every state poll in updateBankLayer().
    const layerBtn = document.getElementById('bp-layer');
    if (layerBtn && !layerBtn.dataset.wired) {
        layerBtn.dataset.wired = '1';
        layerBtn.addEventListener('click', _cycleBankLayer);
    }
}

// LAYER OVERLAY CHIP — POST /api/bank/cycle_layer, then refresh the
// chip label from the response. The state poll will also pick it up
// within ~500ms but this gives instant tactile feedback.
async function _cycleBankLayer() {
    const btn = document.getElementById('bp-layer');
    if (btn) btn.disabled = true;
    try {
        const r = await postAction('bank/cycle_layer', {});
        const res = (r && r.result) || {};
        if (res.ok && res.active_layer) {
            updateBankLayer({
                active: res.active_layer,
                order: res.order || [],
            });
        }
    } catch (e) {
        showError('Cycle layer failed: ' + e.message);
    } finally {
        if (btn) btn.disabled = false;
    }
}

// Repaint the layer chip from whatever the server says is active.
// Called from the state-poll cycle AND from the click handler. Keeps
// the chip in sync even if the layer is cycled via MK2 hardware.
function updateBankLayer(info) {
    const btn = document.getElementById('bp-layer');
    if (!btn) return;
    const active = (info && info.active) || 'default';
    btn.textContent = 'L:' + active;
    btn.classList.toggle('non-default', active !== 'default');
    const order = (info && info.order) || [];
    if (order.length > 1) {
        btn.title = 'Bank LAYER: ' + active +
            '  (cycle order: ' + order.join(' → ') +
            '). Tap to rotate. Takes effect on next bank rebuild.';
    }
}

// Long-press handler for bank-letter buttons (bp-groups). Moved here
// from the old horizontal banks-row chip — same gesture, new home.
// Tap = peek; long-press (700ms) = save current scratch into that
// bank slot.
function _wireGroupLongPress(btn) {
    if (btn.dataset.longpressWired) return;
    btn.dataset.longpressWired = '1';
    const letter = btn.dataset.letter;
    const doSave = () => {
        const name = prompt(`Name for bank ${letter}:`, '') || '';
        postAction('bank/save', { letter, name }).catch(() => {});
    };
    let pressTimer = null;
    btn.addEventListener('touchstart', () => {
        pressTimer = setTimeout(doSave, 700);
    });
    btn.addEventListener('touchend', () => {
        if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; }
    });
    btn.addEventListener('touchmove', () => {
        if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; }
    });
    // Mouse fallback for desktop testing — Shift+click = save.
    btn.addEventListener('click', (ev) => {
        if (ev.shiftKey || ev.altKey || ev.metaKey) {
            ev.stopImmediatePropagation();
            doSave();
        }
    }, true);
}

let _lastScratchSetsSig = "";
function renderScratchSets(sets) {
    const row = document.getElementById('scratch-sets-row');
    if (!row) return;
    const sig = JSON.stringify((sets || []).map(s => [s.id, s.name, s.count]));
    if (sig === _lastScratchSetsSig) return;
    _lastScratchSetsSig = sig;
    if (!sets || !sets.length) {
        row.innerHTML = '';
        return;
    }
    row.innerHTML = sets.map(s => {
        const sid = escapeHTML(s.id || '');
        const name = escapeHTML(s.name || 'Set');
        const cnt = Number(s.count || 0);
        return `<span class="scratch-set-chip" data-set-id="${sid}">`
             + `<button type="button" class="ss-name" data-set-id="${sid}" title="Load this scratch set">${name}</button>`
             + `<span class="ss-count">${cnt}</span>`
             + `<button type="button" class="ss-delete" data-set-id="${sid}" title="Delete">×</button>`
             + `</span>`;
    }).join('');
    row.querySelectorAll('.ss-name').forEach(btn => {
        btn.addEventListener('click', () => {
            if (confirm('Replace current scratch with this set?')) {
                postAction('scratch/load_set', { set_id: btn.dataset.setId }).catch(() => {});
            }
        });
    });
    row.querySelectorAll('.ss-delete').forEach(btn => {
        btn.addEventListener('click', (ev) => {
            ev.stopPropagation();
            if (confirm('Delete this saved set?')) {
                postAction('scratch/delete_set', { set_id: btn.dataset.setId }).catch(() => {});
            }
        });
    });
}

// ── Scenes (saved 4-deck layouts) ───────────────────────────────────
let _lastScenesSig = "";
let _scenesWired = false;
function renderScenes(scenes) {
    const list = document.getElementById('scenes-list');
    if (!list) return;
    const sig = JSON.stringify((scenes || []).map(s => [s.id, s.name, s.slot_count]));
    if (sig !== _lastScenesSig) {
        _lastScenesSig = sig;
        if (!scenes || scenes.length === 0) {
            list.innerHTML = '<li class="empty">No scenes saved yet</li>';
        } else {
            list.innerHTML = scenes.map(s => {
                const sid = escapeHTML(s.id || '');
                const name = escapeHTML(s.name || 'Untitled');
                const cnt = Number(s.slot_count || 0);
                return `<li><span class="scene-chip" data-scene-id="${sid}">`
                     + `<button type="button" class="scene-load" data-scene-id="${sid}" title="Load scene">${name}</button>`
                     + `<span class="scene-count">${cnt}/4</span>`
                     + `<button type="button" class="scene-delete" data-scene-id="${sid}" title="Delete scene">×</button>`
                     + `</span></li>`;
            }).join('');
            list.querySelectorAll('.scene-load').forEach(btn => {
                btn.addEventListener('click', () => {
                    postAction('scene/load', { scene_id: btn.dataset.sceneId }).catch(() => {});
                });
            });
            list.querySelectorAll('.scene-delete').forEach(btn => {
                btn.addEventListener('click', (ev) => {
                    ev.stopPropagation();
                    if (confirm('Delete this scene?')) {
                        postAction('scene/delete', { scene_id: btn.dataset.sceneId }).catch(() => {});
                    }
                });
            });
        }
    }
    // One-time wire of the Save button + name input.
    if (!_scenesWired) {
        const saveBtn = document.getElementById('scene-save-btn');
        const nameInput = document.getElementById('scene-name-input');
        if (saveBtn && nameInput) {
            _scenesWired = true;
            const doSave = () => {
                const name = (nameInput.value || '').trim();
                postAction('scene/save', { name }).catch(() => {});
                nameInput.value = '';
            };
            saveBtn.addEventListener('click', doSave);
            nameInput.addEventListener('keydown', (ev) => {
                if (ev.key === 'Enter') {
                    ev.preventDefault();
                    doSave();
                }
            });
        }
    }
}

// Renders the global Clip Bank (starred + recent across all videos).
// Tap any chip → POST /api/action bank/fire with the clip_id. Server
// loads the source video if needed, sets the loop, and seeks.
let _lastBankSig = "";
function renderBank(bankClips) {
    const list = document.getElementById('bank-list');
    if (!list) return;
    const sig = JSON.stringify((bankClips || []).map(c => [c.id, c.starred, c.name, c.bpm]));
    if (sig === _lastBankSig) return;
    _lastBankSig = sig;
    if (!bankClips || bankClips.length === 0) {
        list.innerHTML = '<li class="empty">No clips saved yet</li>';
        return;
    }
    list.innerHTML = bankClips.map(c => {
        const cid = escapeHTML(c.id || '');
        const name = escapeHTML(c.name || (c.id || '').slice(0, 8));
        const star = c.starred ? '<span class="bank-star">&#9733;</span>' : '';
        const bpm = (c.bpm && c.bpm > 0) ? `<span class="bank-bpm">${Math.round(c.bpm)}</span>` : '';
        const cls = c.starred ? 'bank-chip starred' : 'bank-chip';
        return `<li><button type="button" class="${cls}" data-clip-id="${cid}">`
             + `${star}<span class="bank-name">${name}</span>${bpm}</button></li>`;
    }).join('');
    list.querySelectorAll('.bank-chip').forEach(btn => {
        btn.addEventListener('click', () => {
            postAction('bank/fire', { clip_id: btn.dataset.clipId });
        });
    });
}

function renderClips(clipsRaw, allTags) {
    _lastClipsRaw = clipsRaw || [];
    _lastAllTagsList = allTags || [];

    _renderFilterChips(_lastAllTagsList);
    _renderTagsDatalist(_lastAllTagsList);

    const clipsList = document.getElementById('clips-list');
    if (!clipsList) return;

    // Apply filter, then sort. Original index is preserved as data-idx
    // because play_clip is keyed by per-video index, not clip_id.
    const indexed = _lastClipsRaw.map((c, i) => ({ c, i }));
    const filtered = indexed.filter(x => _clipFilterMatches(x.c));
    const sorted = _clipsSorted(filtered.map(x => x.c)).map(c => {
        // Recover the original idx after sort (linear scan; n is tiny).
        return { c, i: _lastClipsRaw.indexOf(c) };
    });

    // Signature: clip identity + the metadata that affects rendering +
    // active filter / sort / editing state. Without this we'd flicker
    // every 500ms poll (tearing down <img>s and re-fetching thumbnails).
    const sig = JSON.stringify([
        sorted.map(x => [
            x.c.id || x.c.name, x.c.starred ? 1 : 0,
            x.c.play_count || 0, (x.c.tags || []).join(','),
            +x.c.bpm || 0,
        ]),
        activeFilter, sortMode, editingTagsFor,
    ]);
    if (sig === _lastClipsRenderSig) return;
    _lastClipsRenderSig = sig;

    if (sorted.length === 0) {
        const msg = (_lastClipsRaw.length === 0)
            ? 'No cue points on this video'
            : 'No cue points match the current filter';
        clipsList.innerHTML = `<li class="empty">${msg}</li>`;
        return;
    }

    clipsList.innerHTML = sorted.map(({ c, i }) => {
        const id = c.id || '';
        const name = escapeHTML(c.name || 'unnamed');
        const thumbSrc = id ? `/thumbnails/${id}.jpg` : '';
        const starred = !!c.starred;
        const tags = Array.isArray(c.tags) ? c.tags : [];
        const playCount = c.play_count || 0;
        const isEditingTags = editingTagsFor === id;

        // Star badge in the thumb corner so even a glance at the
        // thumbnail tells you which clips are favourites.
        const starBadge = starred
            ? `<span class="clip-star-badge" title="Starred">&#9733;</span>`
            : '';
        // 🔥 N — only show on clips with non-trivial play counts so the
        // visual stays meaningful (not "🔥 1" on everything).
        const playBadge = playCount >= 3
            ? `<span class="clip-play-badge" title="Fired ${playCount}x">&#128293; ${playCount}</span>`
            : '';

        const tagPills = tags.map(t => `
            <span class="clip-tag-pill">
                #${escapeHTML(t)}
                <button class="clip-tag-remove" data-clip-id="${id}"
                        data-tag="${escapeHTML(t)}" title="Remove tag">&times;</button>
            </span>
        `).join('');

        // Inline tag input — only one row at a time owns it (editingTagsFor).
        const tagInput = isEditingTags
            ? `<input class="clip-tag-input" type="text"
                       list="all-tags-datalist" placeholder="tag…"
                       autocomplete="off" autocorrect="off"
                       autocapitalize="off" spellcheck="false"
                       data-clip-id="${id}">`
            : `<button class="clip-tag-add" data-clip-id="${id}"
                       title="Add tag">+ tag</button>`;

        return `
            <li class="clip-row${starred ? ' clip-starred' : ''}"
                data-idx="${i}" data-clip-id="${id}">
                <div class="clip-thumb-wrap">
                    <div class="clip-thumb-placeholder">no preview</div>
                    ${thumbSrc ? `<img class="clip-thumb" src="${thumbSrc}" alt=""
                         onload="this.classList.add('loaded')"
                         onerror="this.style.display='none'">` : ''}
                    <span class="clip-slot">${i}</span>
                    ${starBadge}
                    ${playBadge}
                </div>
                <div class="clip-meta">
                    <div class="clip-meta-row">
                        <span class="clip-name">${name}</span>
                        <button class="clip-star ${starred ? 'on' : ''}"
                                data-clip-id="${id}"
                                title="Toggle star">${starred ? '&#9733;' : '&#9734;'}</button>
                    </div>
                    <span class="clip-time">${fmtTime(c.in_seconds)} &rarr; ${fmtTime(c.out_seconds)}${
                        (+c.bpm > 0)
                            ? ` <span class="clip-bpm" title="Detected tempo">${Math.round(+c.bpm)} BPM</span>`
                            : ''
                    }</span>
                    <div class="clip-tags-row">
                        ${tagPills}
                        ${tagInput}
                    </div>
                </div>
                <button class="clip-delete" data-clip-id="${id}" title="Delete clip">&times;</button>
            </li>
        `;
    }).join('');

    // ── Wire handlers (innerHTML drops old listeners on every render) ──

    // Whole-row tap = fire. Star/tag/delete buttons stopPropagation.
    clipsList.querySelectorAll('.clip-row').forEach(row => {
        row.addEventListener('click', (ev) => {
            // Only fire if the tap actually landed on the row body, not
            // an interactive child (input/button/pill). Without this you
            // can't tap a tag pill without firing the clip too.
            const t = ev.target;
            if (t.closest('.clip-star, .clip-tag-pill, .clip-tag-add, .clip-tag-input, .clip-delete, .clip-tag-remove')) {
                return;
            }
            const idx = parseInt(row.dataset.idx);
            row.classList.add('firing');
            setTimeout(() => row.classList.remove('firing'), 250);
            const clip = _lastClipsRaw[idx];
            fireClip(idx, clip);
        });
    });

    // Star toggle — optimistic UI (flip class immediately, server reconciles)
    clipsList.querySelectorAll('.clip-star').forEach(btn => {
        btn.addEventListener('click', async (ev) => {
            ev.stopPropagation();
            const id = btn.dataset.clipId;
            if (!id) return;
            const wasOn = btn.classList.contains('on');
            // Optimistic flip
            btn.classList.toggle('on', !wasOn);
            btn.innerHTML = !wasOn ? '&#9733;' : '&#9734;';
            try {
                await postAction('clip/star', { clip_id: id, value: !wasOn });
            } catch (e) {
                // Revert on failure (postAction already showed the error)
                btn.classList.toggle('on', wasOn);
                btn.innerHTML = wasOn ? '&#9733;' : '&#9734;';
            }
        });
    });

    // Add-tag pill — open the inline input on this row
    clipsList.querySelectorAll('.clip-tag-add').forEach(btn => {
        btn.addEventListener('click', (ev) => {
            ev.stopPropagation();
            editingTagsFor = btn.dataset.clipId || null;
            _lastClipsRenderSig = '';
            renderClips(_lastClipsRaw, _lastAllTagsList);
            // Focus the freshly-rendered input
            setTimeout(() => {
                const inp = clipsList.querySelector('.clip-tag-input');
                if (inp) inp.focus();
            }, 0);
        });
    });

    // Tag input — Enter commits, Escape cancels, blur cancels.
    clipsList.querySelectorAll('.clip-tag-input').forEach(inp => {
        inp.addEventListener('click', (ev) => ev.stopPropagation());
        inp.addEventListener('keydown', async (ev) => {
            ev.stopPropagation();
            if (ev.key === 'Enter') {
                ev.preventDefault();
                const tag = (inp.value || '').trim();
                const id = inp.dataset.clipId;
                editingTagsFor = null;
                if (tag && id) {
                    try {
                        await postAction('clip/add_tag', { clip_id: id, tag });
                    } catch (e) {}
                }
                _lastClipsRenderSig = '';
                renderClips(_lastClipsRaw, _lastAllTagsList);
            } else if (ev.key === 'Escape') {
                editingTagsFor = null;
                _lastClipsRenderSig = '';
                renderClips(_lastClipsRaw, _lastAllTagsList);
            }
        });
        inp.addEventListener('blur', () => {
            // Defer so a tap on a datalist option still gets to commit.
            setTimeout(() => {
                if (editingTagsFor && document.activeElement !== inp) {
                    editingTagsFor = null;
                    _lastClipsRenderSig = '';
                    renderClips(_lastClipsRaw, _lastAllTagsList);
                }
            }, 200);
        });
    });

    // Tag pill remove (×) — POST clip/remove_tag
    clipsList.querySelectorAll('.clip-tag-remove').forEach(btn => {
        btn.addEventListener('click', async (ev) => {
            ev.stopPropagation();
            const id = btn.dataset.clipId;
            const tag = btn.dataset.tag;
            if (!id || !tag) return;
            try {
                await postAction('clip/remove_tag', { clip_id: id, tag });
            } catch (e) {}
        });
    });

    // Delete (×) — same as before, just inside the new render path.
    clipsList.querySelectorAll('.clip-delete').forEach(btn => {
        btn.addEventListener('click', async (ev) => {
            ev.stopPropagation();
            const id = btn.dataset.clipId;
            if (!id) return;
            btn.disabled = true;
            try {
                await postAction('clip/delete', { clip_id: id });
            } catch (e) {
                showError('Delete failed: ' + e.message);
            }
        });
    });
}

async function fireClip(idx, clipObj) {
    // If a deck slot is selected, assigning takes priority over firing.
    if (selectedDeck !== null && clipObj && clipObj.id) {
        const deck = selectedDeck;
        clearDeckSelection();
        await assignClipToDeck(deck, clipObj.id);
        return;
    }
    // (Pad mirror flash was removed with the S2 mirror panel — audit
    // 2026-05-16. Hardware pads have their own LED; no need to double up.)
    try {
        const r = await fetch('/api/play_clip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ idx })
        });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        setStatus(`Fired clip ${idx}`);
    } catch (e) {
        showError('Fire clip failed: ' + e.message);
    }
}

// ── Library browser ──────────────────────────────────────────────────

function escapeHTML(s) {
    return String(s).replace(/[<>&"]/g, ch => (
        { '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' }[ch]
    ));
}

// Coerce a library file entry to {name, hash, has_thumb}. Server now
// emits dicts but we keep the legacy bare-string path tolerated so a
// stale cached state.json can't blank the panel.
function normalizeLibFile(f) {
    if (typeof f === 'string') return { name: f, hash: '', has_thumb: false };
    if (!f || typeof f !== 'object') return { name: '', hash: '', has_thumb: false };
    return {
        name: String(f.name || ''),
        hash: String(f.hash || ''),
        has_thumb: !!f.has_thumb,
    };
}

// One-time wire of the Working Set / Library shortcut buttons.
let _wsButtonsWired = false;
function wireLibraryShortcuts() {
    if (_wsButtonsWired) return;
    const ws = document.getElementById('library-working-set');
    const back = document.getElementById('library-default-root');
    if (!ws || !back) return;
    _wsButtonsWired = true;
    ws.addEventListener('click', () => {
        postAction('working_set/open', {}).catch(() => {});
    });
    back.addEventListener('click', () => {
        postAction('working_set/exit', {}).catch(() => {});
    });
}

function renderLibrary(lib, pb) {
    wireLibraryShortcuts();
    // Toggle which button is shown based on whether we're currently
    // pointed at the working-set folder. Best-effort string match.
    try {
        const ws = document.getElementById('library-working-set');
        const back = document.getElementById('library-default-root');
        const root = String(lib.root || '');
        const inWorkingSet = root.toLowerCase().includes('setpiece-working');
        if (ws && back) {
            ws.classList.toggle('hidden', inWorkingSet);
            back.classList.toggle('hidden', !inWorkingSet);
        }
    } catch (_) {}

    const list = document.getElementById('library-list');
    const crumbBar = document.getElementById('library-breadcrumb');
    if (!list || !crumbBar) return;

    const subs = lib.subfolders || [];
    const filesRaw = (lib.files || []).map(normalizeLibFile);
    const rel = lib.rel_path || '';
    const atRoot = !!lib.at_root;
    const selectedIdx = (typeof lib.selected_idx === 'number') ? lib.selected_idx : -1;
    const currentBasename = fmtBasename(pb.file || '');

    // Clear loading state once the polled state confirms the file is now
    // playing. Without this the row would pulse forever after a successful load.
    if (loadingFile && currentBasename === loadingFile) {
        loadingFile = null;
    }

    // Apply the search filter. Subfolders are hidden when there's an
    // active query — browsing into folders mid-search is more disruptive
    // than helpful, and clearing the box restores everything anyway.
    const q = (librarySearchQuery || '').trim().toLowerCase();
    const filteredSubs = q ? [] : subs;
    const filteredFiles = q
        ? filesRaw.filter(f => f.name.toLowerCase().includes(q))
        : filesRaw;

    // Cheap signature: only rebuild DOM when contents actually change.
    // Includes loadingFile, search query, current/selected idx so highlight
    // transitions render.
    const sig = JSON.stringify([
        rel, atRoot, filteredSubs, filteredFiles, currentBasename,
        loadingFile, selectedIdx, q,
    ]);
    if (sig === lastLibrarySig) return;
    lastLibrarySig = sig;
    lastCurrentBasename = currentBasename;

    // Breadcrumb: root + each segment, last segment is non-tappable "current"
    const segments = rel ? rel.split('/').filter(Boolean) : [];
    let crumbHTML = `<span class="crumb${segments.length === 0 ? ' current' : ''}"
        data-rel="">root</span>`;
    let acc = '';
    segments.forEach((seg, i) => {
        acc = acc ? `${acc}/${seg}` : seg;
        const isLast = i === segments.length - 1;
        crumbHTML += `<span class="crumb-sep">/</span>`;
        crumbHTML += `<span class="crumb${isLast ? ' current' : ''}"
            data-rel="${escapeHTML(acc)}">${escapeHTML(seg)}</span>`;
    });
    crumbBar.innerHTML = crumbHTML;
    crumbBar.querySelectorAll('.crumb').forEach(el => {
        if (el.classList.contains('current')) return;
        el.addEventListener('click', () => navigateAbsolute(el.dataset.rel || ''));
    });

    if (!lib.folder) {
        list.innerHTML = `<div class="library-empty">
            No library configured. POST /api/library/scan with {folder: "..."}
        </div>`;
        return;
    }
    if (subs.length === 0 && filesRaw.length === 0) {
        list.innerHTML = `<div class="library-empty">Empty folder</div>`;
        return;
    }
    if (q && filteredFiles.length === 0) {
        list.innerHTML = `<div class="library-empty">
            No files match &ldquo;${escapeHTML(q)}&rdquo;
        </div>`;
        return;
    }

    let html = '';
    if (!atRoot && !q) {
        html += `<div class="lib-row lib-up" data-action="up">
            <span class="lib-icon">&#8593;</span>
            <span class="lib-name">..</span>
            <span class="lib-tag">UP</span>
        </div>`;
    }
    filteredSubs.forEach(name => {
        html += `<div class="lib-row lib-folder" data-action="cd" data-name="${escapeHTML(name)}">
            <span class="lib-icon">&#128193;</span>
            <span class="lib-name">${escapeHTML(name)}</span>
            <button type="button" class="lib-folder-bank"
                data-folder="${escapeHTML(name)}"
                aria-label="Navigate in and auto-bank A-H from this folder"
                title="Navigate in and auto-bank A-H from this folder">&#8862;</button>
            <span class="lib-tag">FOLDER</span>
        </div>`;
    });
    // At root, loose files clutter the folder browser -- the user has
    // to scroll past N loose files to find a [folder]. Collapse them
    // behind an expand toggle. Default = hidden, folders dominate.
    // The toggle itself has a BANK button so the operator can bank
    // the loose set without expanding. State persisted across renders
    // via localStorage so the expand sticks (was snapping closed on
    // every poll without it).
    if (window._libShowLoose === undefined) {
        try {
            window._libShowLoose =
                localStorage.getItem('libShowLoose') === '1';
        } catch (_) { window._libShowLoose = false; }
    }
    const looseAtRoot = atRoot && !q && filteredFiles.length > 0;
    const looseHidden = looseAtRoot && !window._libShowLoose;
    if (looseAtRoot) {
        const expanded = !looseHidden;
        const arrow = expanded ? '&#9662;' : '&#9656;';   // ▾ / ▸
        const tag = expanded ? 'HIDE' : 'SHOW';
        html += `<div class="lib-row lib-loose-toggle"
            data-action="toggle-loose">
            <span class="lib-icon">${arrow}</span>
            <span class="lib-name">${filteredFiles.length} loose files at root</span>
            <button type="button" class="lib-folder-bank"
                data-loose-bank="1"
                aria-label="Auto-bank just the loose root files"
                title="Auto-bank the loose root files into A-H">&#8862;</button>
            <span class="lib-tag">${tag}</span>
        </div>`;
    }
    // When loose files are collapsed at root, skip rendering them
    // (the toggle row above already showed the count + bank action).
    const filesToRender = looseHidden ? [] : filteredFiles;
    filesToRender.forEach((entry) => {
        const name = entry.name;
        // Real index in the unfiltered list — keeps cursor / S2 browse
        // semantics working even when a filter is active.
        const fi = filesRaw.indexOf(entry);
        const isCurrent = name === currentBasename;
        const isLoading = name === loadingFile;
        const isSelected = fi === selectedIdx;
        const cls = ['lib-row', 'lib-file'];
        if (isCurrent) cls.push('lib-current');
        if (isLoading) cls.push('lib-loading');
        if (isSelected) cls.push('lib-selected');
        const tag = isSelected ? 'BROWSE' : (isCurrent ? 'PLAYING' : (isLoading ? 'LOADING' : ''));
        // Thumbnail: lazy-loaded native (loading="lazy" is supported in
        // iOS Safari 15.4+). Server returns 404 silently when the thumb
        // hasn't been generated yet — onerror hides the broken image so
        // the placeholder gradient stays visible.
        const thumbSrc = entry.hash ? `/lib_thumbnails/${entry.hash}.jpg` : '';
        const thumbHTML = `
            <div class="lib-thumb-wrap">
                <span class="lib-thumb-placeholder">&#9658;</span>
                ${thumbSrc ? `<img class="lib-thumb" src="${thumbSrc}" alt=""
                    loading="lazy" decoding="async"
                    onload="this.classList.add('loaded')"
                    onerror="this.style.display='none'">` : ''}
            </div>`;
        // Tiny ✚ button drops this file into the Scratch basket without
        // playing it. Already-in-scratch shows ✓ instead.
        const inScratch = _scratchPaths && Array.from(_scratchPaths).some(p =>
            p && p.toLowerCase().endsWith(name.toLowerCase()));
        const scratchBtn = `<button type="button" class="lib-scratch-add"
            data-scratch="${escapeHTML(name)}"
            title="${inScratch ? 'Already in scratch' : 'Add to scratch'}">${inScratch ? '✓' : '+'}</button>`;
        html += `<div class="${cls.join(' ')}" data-action="load" data-name="${escapeHTML(name)}" data-fi="${fi}">
            ${thumbHTML}
            <span class="lib-name">${escapeHTML(name)}</span>
            <span class="lib-tag">${tag}</span>
            ${scratchBtn}
        </div>`;
    });
    list.innerHTML = html;

    // Auto-scroll the browse-cursor row into view if it's not currently
    // visible. block:'nearest' avoids re-centering when it's already on
    // screen, so the user can still scroll manually without fighting us.
    const sel = list.querySelector('.lib-row.lib-selected');
    if (sel) {
        const lr = list.getBoundingClientRect();
        const sr = sel.getBoundingClientRect();
        if (sr.top < lr.top || sr.bottom > lr.bottom) {
            sel.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
    }

    list.querySelectorAll('.lib-row').forEach(row => {
        const action = row.dataset.action;
        if (action === 'up') {
            row.addEventListener('click', () => navigateRelative('..'));
        } else if (action === 'cd') {
            row.addEventListener('click', () => navigateRelative(row.dataset.name));
        } else if (action === 'load') {
            row.addEventListener('click', () => loadLibraryFile(row.dataset.name, row));
        } else if (action === 'toggle-loose') {
            row.addEventListener('click', () => {
                window._libShowLoose = !window._libShowLoose;
                try {
                    localStorage.setItem(
                        'libShowLoose',
                        window._libShowLoose ? '1' : '0'
                    );
                } catch (_) {}
                lastLibrarySig = '';  // force re-render
            });
        }
    });
    // Wire the "bank just the loose files" button on the loose-toggle
    // row. Hits bank/auto_split (which uses current visible filter =
    // root files when at root with no path-tag filter active).
    list.querySelectorAll('.lib-folder-bank[data-loose-bank]').forEach(btn => {
        btn.addEventListener('click', async (ev) => {
            ev.stopPropagation();
            const orig = btn.textContent;
            btn.disabled = true;
            btn.textContent = '...';
            try {
                const r = await postAction('bank/auto_split',
                                           { prefix: 'loose' });
                const res = (r && r.result) || {};
                _showAutoBankResult(res);
                btn.textContent = res.ok ? '✓' : '!';
            } catch (e) {
                console.error('loose-bank failed', e);
                btn.textContent = '!';
            } finally {
                setTimeout(() => {
                    btn.textContent = orig;
                    btn.disabled = false;
                }, 1500);
            }
        });
    });
    // Wire the inline ✚ scratch-add buttons. stopPropagation so the row's
    // load handler doesn't ALSO fire.
    list.querySelectorAll('.lib-scratch-add').forEach(btn => {
        btn.addEventListener('click', (ev) => {
            ev.stopPropagation();
            postAction('scratch/add_file', { file: btn.dataset.scratch }).catch(() => {});
        });
    });
    // Wire the inline BANK buttons on folder rows. One tap =
    // recursively gather every video under that folder and split
    // across A-H. No navigation needed -- backend walks subfolders
    // so banking "Lil Humpers" (parent with no direct .mp4s, only
    // year subfolders) finds the full set.
    list.querySelectorAll('.lib-folder-bank').forEach(btn => {
        btn.addEventListener('click', async (ev) => {
            ev.stopPropagation();
            const folder = btn.dataset.folder;
            const orig = btn.textContent;
            btn.disabled = true;
            btn.textContent = '...';
            try {
                const r = await postAction('bank/from_folder', {
                    folder: folder,
                    recursive: true,
                });
                const res = (r && r.result) || {};
                _showAutoBankResult(res);
                btn.textContent = res.ok ? '✓' : '!';
            } catch (e) {
                console.error('folder bank failed', e);
                btn.textContent = '!';
            } finally {
                setTimeout(() => {
                    btn.textContent = orig;
                    btn.disabled = false;
                }, 1500);
            }
        });
    });
}

function wireLibrarySearch() {
    const input = document.getElementById('library-search-input');
    const clearBtn = document.getElementById('library-search-clear');
    if (!input || input.dataset.wired) return;
    input.dataset.wired = 'true';

    const sync = () => {
        librarySearchQuery = input.value || '';
        clearBtn.classList.toggle('hidden', librarySearchQuery.length === 0);
        // Force the next renderLibrary to actually rebuild even if the
        // server snapshot didn't change.
        lastLibrarySig = '';
    };

    // 'input' fires per keystroke on iOS Safari (also on backspace).
    input.addEventListener('input', sync);
    // Some keyboards fire 'search' on the return key — handle both.
    input.addEventListener('search', sync);

    clearBtn.addEventListener('click', () => {
        input.value = '';
        librarySearchQuery = '';
        clearBtn.classList.add('hidden');
        lastLibrarySig = '';
        input.focus();
    });
}

async function navigateRelative(path) {
    // Tapping a folder/up: clear any pending load indicator first so the
    // pulse doesn't follow you into the new folder.
    loadingFile = null;
    try {
        const r = await fetch('/api/library/cd', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok || !data.ok) throw new Error(data.error || ('HTTP ' + r.status));
        // Force re-render on next poll by invalidating signature.
        lastLibrarySig = '';
    } catch (e) {
        showError('Navigate failed: ' + e.message);
    }
}

async function navigateAbsolute(relPath) {
    // Breadcrumb tap: cd to absolute relative-to-root path. The endpoint
    // resolves "" as root and any subpath inside root the same way as a
    // sequence of cd's would.
    loadingFile = null;
    try {
        // The cd endpoint joins to current folder, so first reset to root
        // then descend if needed.
        let r = await fetch('/api/library/cd', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: '' })
        });
        let data = await r.json().catch(() => ({}));
        if (!r.ok || !data.ok) throw new Error(data.error || ('HTTP ' + r.status));
        if (relPath) {
            r = await fetch('/api/library/cd', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: relPath })
            });
            data = await r.json().catch(() => ({}));
            if (!r.ok || !data.ok) throw new Error(data.error || ('HTTP ' + r.status));
        }
        lastLibrarySig = '';
    } catch (e) {
        showError('Navigate failed: ' + e.message);
    }
}

async function loadLibraryFile(name, rowEl) {
    if (!name) return;
    // If a deck slot is selected, assign the file to that slot instead of
    // loading it into the live player.
    if (selectedDeck !== null) {
        const deck = selectedDeck;
        clearDeckSelection();
        await assignFileToDeck(deck, name);
        return;
    }
    // Optimistic UI: pulse immediately, clear on confirm/error.
    loadingFile = name;
    if (rowEl) {
        rowEl.classList.add('lib-loading');
        // A tag refresh next poll will set "LOADING" too, but show it now.
        const tag = rowEl.querySelector('.lib-tag');
        if (tag) tag.textContent = 'LOADING';
    }
    lastLibrarySig = '';
    try {
        const r = await fetch('/api/library/load_file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file: name })
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok || !data.ok) {
            // Server-side failure (file gone, bad path, etc) — surface and clear.
            throw new Error(data.error || ('HTTP ' + r.status));
        }
        // Success: leave loadingFile set; the next pollState() that sees
        // current_file == name will clear it and flip the row to "PLAYING".
    } catch (e) {
        loadingFile = null;
        if (rowEl) rowEl.classList.remove('lib-loading');
        showError('Load failed: ' + e.message);
    }
}

// ── Decks (launchpad) ────────────────────────────────────────────────

function clearDeckSelection() {
    selectedDeck = null;
    document.querySelectorAll('.deck-slot.selected').forEach(el => {
        el.classList.remove('selected');
    });
    updateDecksHint();
}

function updateDecksHint() {
    const hint = document.getElementById('decks-hint');
    if (!hint) return;
    if (selectedDeck === null) {
        hint.textContent = 'Tap a slot, then tap a clip';
        hint.classList.remove('armed');
    } else {
        hint.textContent = `Slot ${selectedDeck + 1} ARMED — tap a clip or library file`;
        hint.classList.add('armed');
    }
}

function selectDeck(slot) {
    // Toggle off if tapping the already-selected slot
    if (selectedDeck === slot) {
        clearDeckSelection();
        return;
    }
    selectedDeck = slot;
    document.querySelectorAll('.deck-slot').forEach(el => {
        el.classList.toggle('selected', parseInt(el.dataset.slot) === slot);
    });
    updateDecksHint();
}

async function fireDeck(slot) {
    const slotEl = document.querySelector(`.deck-slot[data-slot="${slot}"]`);
    if (slotEl) {
        slotEl.classList.add('firing');
        setTimeout(() => slotEl.classList.remove('firing'), 350);
    }
    try {
        const r = await fetch('/api/deck/fire', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ deck: slot })
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok || data.ok === false) {
            throw new Error(data.error || ('HTTP ' + r.status));
        }
        setStatus(`GO LIVE: deck ${slot + 1}`);
    } catch (e) {
        showError('Fire deck failed: ' + e.message);
    }
}

async function clearDeckSlot(slot) {
    try {
        const r = await fetch('/api/deck/clear', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ deck: slot })
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok || !data.ok) throw new Error(data.error || ('HTTP ' + r.status));
        // Force re-render on next poll
        lastDecksSig = '';
    } catch (e) {
        showError('Clear deck failed: ' + e.message);
    }
}

async function assignClipToDeck(slot, clipId) {
    try {
        const r = await fetch('/api/deck/load_clip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ deck: slot, clip_id: clipId })
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok || !data.ok) throw new Error(data.error || ('HTTP ' + r.status));
        setStatus(`Assigned to deck ${slot + 1}`);
        lastDecksSig = '';
    } catch (e) {
        showError('Assign failed: ' + e.message);
    }
}

async function assignFileToDeck(slot, filename) {
    try {
        const r = await fetch('/api/deck/load_file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ deck: slot, file: filename })
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok || !data.ok) throw new Error(data.error || ('HTTP ' + r.status));
        setStatus(`Assigned to deck ${slot + 1}`);
        lastDecksSig = '';
    } catch (e) {
        showError('Assign failed: ' + e.message);
    }
}

// ── Crossfade indicator ──────────────────────────────────────────────

// Cache last rendered xfade signature so we don't repaint identical state.
// Position changes constantly though, so we always update bar widths.
let lastXfadeSig = '';

function renderCrossfade(xf, pb) {
    // Text-only since audit 2026-05-16: the S2 fader IS the bar
    // (user's hand is on it). Names + status badge are all that's
    // useful screen-side.
    const liveName = fmtBasename(xf.live_file || pb.file || '');
    const previewPath = xf.preview_file || '';
    const previewName = previewPath ? fmtBasename(previewPath) : '';
    const position = typeof xf.position === 'number' ? xf.position : 0;
    const blendActive = !!xf.blend_active;
    const deckIdx = (typeof xf.preview_deck_idx === 'number') ? xf.preview_deck_idx : 0;

    // Names + deck-num are slow-changing — only re-render text on change
    const sig = JSON.stringify([liveName, previewName, blendActive, deckIdx]);
    if (sig !== lastXfadeSig) {
        lastXfadeSig = sig;
        const liveEl = document.getElementById('xfade-live-name');
        const prevEl = document.getElementById('xfade-preview-name');
        const deckNumEl = document.getElementById('xfade-deck-num');
        if (liveEl) liveEl.textContent = liveName || 'No video';
        if (prevEl) {
            if (previewName) {
                prevEl.textContent = previewName;
                prevEl.classList.remove('xfade-empty');
            } else {
                prevEl.textContent = blendActive ? '(loading)' : 'No preview staged';
                prevEl.classList.add('xfade-empty');
            }
        }
        if (deckNumEl) deckNumEl.textContent = String(deckIdx + 1);
    }

    // Status badge: A LIVE / BLEND / B LIVE — the one visual confirmation
    // that's still worth pixels (color-shift catches the eye on stage).
    const statusEl = document.getElementById('crossfade-status');
    if (statusEl) {
        let label, cls;
        if (position < 0.05) {
            label = 'A LIVE';
            cls = 'xfade-live-on';
        } else if (position > 0.95) {
            label = 'B LIVE';
            cls = 'xfade-preview-on';
        } else {
            label = 'BLEND';
            cls = 'xfade-blend';
        }
        statusEl.textContent = label;
        statusEl.classList.remove('xfade-live-on', 'xfade-preview-on', 'xfade-blend');
        statusEl.classList.add(cls);
    }
}

// Per-slot cache-buster timestamp for the MJPEG <img> src. Bumped only
// when a slot's underlying file/in/out changes — NOT on every state
// poll. Otherwise iOS Safari would re-open the multipart connection
// every 500ms (each render = new src = drop existing connection).
const deckStreamTokens = [0, 0, 0, 0];
const deckStreamSig = ['', '', '', ''];

// Global so the inline onerror= handler in the rendered <img> can find
// it. Hides the broken stream <img> so the .deck-placeholder hatch
// shows through. Live dot + pulse ring stay (they read .streaming on
// the parent slot, not the <img>'s state) since the deck IS still
// loaded — we just couldn't open the stream right now.
function onDeckStreamError(slot) {
    const img = document.querySelector(
        `.deck-stream[data-slot="${slot}"]`
    );
    if (img) img.classList.add('broken');
}

let _proxyWired = false;
function renderProxyStatus(ps) {
    const text = document.getElementById('ps-text');
    const pauseBtn = document.getElementById('ps-pause');
    if (!text || !pauseBtn) return;
    const paused = !!ps.paused;
    const queued = Number(ps.queued || 0);
    const inFlight = String(ps.in_flight || '');
    let label;
    if (paused) {
        label = `Proxies: PAUSED · ${queued} queued`;
        text.className = 'ps-text paused';
    } else if (inFlight) {
        label = `Proxies: ${inFlight}  (${queued} more)`;
        text.className = 'ps-text busy';
    } else if (queued > 0) {
        label = `Proxies: ${queued} queued`;
        text.className = 'ps-text';
    } else {
        label = 'Proxies: idle';
        text.className = 'ps-text';
    }
    text.textContent = label;
    pauseBtn.textContent = paused ? 'Resume' : 'Pause';
    pauseBtn.classList.toggle('active', paused);
    if (!_proxyWired) {
        const cancelCur = document.getElementById('ps-cancel-current');
        const cancelPend = document.getElementById('ps-cancel-pending');
        _proxyWired = true;
        pauseBtn.addEventListener('click', () => {
            const ep = pauseBtn.classList.contains('active') ? 'proxy/resume' : 'proxy/pause';
            postAction(ep, {}).catch(() => {});
        });
        if (cancelCur) {
            cancelCur.addEventListener('click', () => {
                postAction('proxy/cancel_current', {}).catch(() => {});
            });
        }
        if (cancelPend) {
            cancelPend.addEventListener('click', () => {
                if (confirm('Drop all queued transcodes?')) {
                    postAction('proxy/cancel_pending', {}).catch(() => {});
                }
            });
        }
    }
}

let _decksClearAllWired = false;
function renderDecks(decks, pb) {
    if (!_decksClearAllWired) {
        const clearBtn = document.getElementById('decks-clear-all');
        if (clearBtn) {
            _decksClearAllWired = true;
            clearBtn.addEventListener('click', () => {
                if (confirm('Clear all 4 deck slots?')) {
                    postAction('deck/clear_all', {}).catch(() => {});
                }
            });
        }
    }
    const grid = document.getElementById('deck-grid');
    if (!grid) return;
    // Pad / truncate to 4 (server should already do this, but be safe)
    const slots = (decks || []).slice(0, 4);
    while (slots.length < 4) slots.push(null);

    // Update per-slot cache-busting tokens BEFORE the signature check —
    // we need to detect "same slots overall, but slot N's content
    // changed" so the <img> reloads with a fresh stream.
    slots.forEach((d, i) => {
        const slotSig = d
            ? `${d.filepath || ''}|${d.in_sec || 0}|${d.out_sec || 0}`
            : '';
        if (slotSig !== deckStreamSig[i]) {
            deckStreamSig[i] = slotSig;
            deckStreamTokens[i] = Date.now();
        }
    });

    const currentBasename = fmtBasename(pb.file || '');
    // Cheap signature so we don't repaint the slots on every 500ms poll —
    // re-rendering the <img> would close the open MJPEG connection.
    const sig = JSON.stringify(
        slots.map((d, i) => d
            ? [d.slot, d.name, d.filepath, deckStreamTokens[i]]
            : null)
    ) + '|' + selectedDeck + '|' + currentBasename;
    if (sig === lastDecksSig) return;
    lastDecksSig = sig;

    grid.innerHTML = slots.map((d, i) => {
        if (!d) {
            return `
                <div class="deck-slot empty" data-slot="${i}">
                    <div class="deck-placeholder"></div>
                    <div class="deck-badge">${i + 1}</div>
                    <div class="deck-name">Drop a clip here</div>
                </div>
            `;
        }
        const name = escapeHTML(d.name || `slot ${i + 1}`);
        const isLive = d.filepath && fmtBasename(d.filepath) === currentBasename;
        const cls = ['deck-slot', 'streaming'];
        if (isLive) cls.push('live');
        // The <img>'s src includes a per-slot cache-buster so iOS
        // doesn't try to reuse a stale connection after deck content
        // changes. onerror swaps the .deck-stream → .deck-placeholder
        // fallback if the multipart endpoint 503s (no ffmpeg / not
        // streaming yet).
        const streamUrl = `/preview/${i}.mjpg?t=${deckStreamTokens[i]}`;
        return `
            <div class="${cls.join(' ')}" data-slot="${i}">
                <div class="deck-placeholder"></div>
                <img class="deck-stream" data-slot="${i}"
                     src="${streamUrl}"
                     alt="deck ${i + 1} preview"
                     onerror="onDeckStreamError(${i})">
                <div class="deck-badge">${i + 1}</div>
                <button class="deck-go" data-action="fire" data-slot="${i}">GO LIVE</button>
                <button class="deck-clear" data-action="clear" data-slot="${i}" title="Clear slot">&times;</button>
                <div class="deck-live-dot">LIVE</div>
                <div class="deck-name">${name}</div>
            </div>
        `;
    }).join('');

    // Restore selected highlight (since innerHTML wiped it)
    if (selectedDeck !== null) {
        const sel = grid.querySelector(`.deck-slot[data-slot="${selectedDeck}"]`);
        if (sel) sel.classList.add('selected');
    }

    // Wire handlers (fresh each render — innerHTML drops old listeners)
    grid.querySelectorAll('.deck-slot').forEach(slotEl => {
        const slot = parseInt(slotEl.dataset.slot);
        slotEl.addEventListener('click', (ev) => {
            // GO LIVE button → fire, regardless of selection
            const action = ev.target?.dataset?.action;
            if (action === 'fire') {
                ev.stopPropagation();
                fireDeck(slot);
                return;
            }
            if (action === 'clear') {
                ev.stopPropagation();
                clearDeckSlot(slot);
                return;
            }
            // Tap empty slot → select it. Tap filled slot with no
            // selection active → fire it (the "primary" action). Tap a
            // filled slot while a different slot is selected → switch
            // selection to this one (so re-targeting is one tap).
            const isEmpty = slotEl.classList.contains('empty');
            if (isEmpty) {
                selectDeck(slot);
            } else if (selectedDeck === null) {
                fireDeck(slot);
            } else {
                selectDeck(slot);
            }
        });
    });
}

async function postAction(action, data = {}) {
    // Returns the parsed JSON response so callers can read
    // `r.result` for action-specific data. Returns `null` on network /
    // HTTP errors (caller's `r && r.result` pattern handles both
    // cleanly). Bug seen 2026-05-18: this used to return `undefined`
    // implicitly, causing AUTO-BANK and other actions that DID succeed
    // on the backend to show "failed" in the UI because the iPad
    // never saw the response body.
    try {
        const r = await fetch('/api/' + action, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        try {
            return await r.json();
        } catch (e) {
            // Non-JSON body (some endpoints return plain text or empty)
            return { ok: true };
        }
    } catch (e) {
        showError(action + ' failed: ' + e.message);
        return null;
    }
}

// ── Channels (saved library-folder presets) ───────────────────────────

function fmtChannelFolder(folder) {
    // Show only the basename so the chip stays readable on iPad. Empty
    // folder = special placeholder text (chip will style it muted).
    if (!folder) return '(no folder set)';
    const cleaned = String(folder).replace(/[\\/]+$/, '');
    const parts = cleaned.replace(/\\/g, '/').split('/');
    return parts[parts.length - 1] || cleaned;
}

function renderChannels(chData) {
    const row = document.getElementById('channels-row');
    if (!row) return;
    const channels = (chData.list || []).slice(0, 4);
    while (channels.length < 4) {
        // Pad with empty placeholders so the chip row never partial-renders
        channels.push({ idx: channels.length, name: '', folder: '', color: '#444', tag_filter: '' });
    }
    const activeIdx = (typeof chData.active_idx === 'number') ? chData.active_idx : -1;

    // Cheap signature so we don't tear down DOM on every poll. Includes
    // editingChannelIdx so opening/closing the editor still triggers a
    // chip repaint (so the open chip can show its highlighted state).
    const sig = JSON.stringify([channels, activeIdx, editingChannelIdx]);
    if (sig === lastChannelsSig) return;
    lastChannelsSig = sig;

    row.innerHTML = channels.map((c, i) => {
        const name = escapeHTML(c.name || `Channel ${i + 1}`);
        const folder = escapeHTML(fmtChannelFolder(c.folder));
        const color = c.color || '#444';
        const tag = c.tag_filter ? `<span class="ch-tag">#${escapeHTML(c.tag_filter)}</span>` : '';
        const isActive = (i === activeIdx);
        const folderCls = c.folder ? 'ch-folder' : 'ch-folder empty';
        return `
            <div class="channel-chip${isActive ? ' active' : ''}"
                 data-ch="${i}" style="--ch-color: ${escapeHTML(color)};">
                <button class="ch-edit" data-edit="${i}" title="Edit">&#9881;</button>
                <span class="ch-num">CH ${i + 1}</span>
                <span class="ch-name">${name}</span>
                <span class="${folderCls}">${folder}</span>
                ${tag}
            </div>
        `;
    }).join('');

    // Wire taps. The gear stops propagation so it doesn't also switch.
    row.querySelectorAll('.channel-chip').forEach(chip => {
        const idx = parseInt(chip.dataset.ch);
        chip.addEventListener('click', () => switchChannel(idx));
    });
    row.querySelectorAll('.ch-edit').forEach(btn => {
        btn.addEventListener('click', (ev) => {
            ev.stopPropagation();
            const idx = parseInt(btn.dataset.edit);
            openChannelEditor(idx, channels[idx]);
        });
    });
}

async function switchChannel(idx) {
    // Optimistic-style status: mirror what the S2 button feels like.
    setStatus(`CH ${idx + 1} switching…`);
    try {
        const r = await fetch('/api/channel/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ idx })
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.error || ('HTTP ' + r.status));
        // Server may return ok:false when the channel has no folder; surface it
        if (data.result && data.result.ok === false) {
            throw new Error(data.result.error || 'switch failed');
        }
        // Library will refresh on the next pollState; chip highlight is
        // driven by state.channels.active_idx, no client toggling needed.
    } catch (e) {
        showError('Channel switch failed: ' + e.message);
    }
}

function openChannelEditor(idx, channel) {
    editingChannelIdx = idx;
    lastChannelsSig = '';   // force chip row repaint on next poll
    const ed = document.getElementById('channel-editor');
    if (!ed) return;
    const ce = (id) => document.getElementById(id);
    ce('ce-num').textContent = String(idx + 1);
    ce('ce-name').value = (channel && channel.name) || `Channel ${idx + 1}`;
    ce('ce-folder').value = (channel && channel.folder) || '';
    ce('ce-tag').value = (channel && channel.tag_filter) || '';
    const curColor = (channel && channel.color) || CHANNEL_COLOR_PALETTE[idx];
    ed.style.setProperty('--ch-color', curColor);

    // Build swatches. Selected one gets the white ring.
    const sw = ce('ce-swatches');
    sw.innerHTML = CHANNEL_COLOR_PALETTE.map(c => {
        const sel = (c.toLowerCase() === String(curColor).toLowerCase()) ? ' selected' : '';
        return `<div class="ce-swatch${sel}" data-color="${c}" style="background: ${c};"></div>`;
    }).join('');
    sw.querySelectorAll('.ce-swatch').forEach(el => {
        el.addEventListener('click', () => {
            sw.querySelectorAll('.ce-swatch').forEach(s => s.classList.remove('selected'));
            el.classList.add('selected');
            ed.style.setProperty('--ch-color', el.dataset.color);
        });
    });

    ed.classList.remove('hidden');
    // Scroll into view so the user actually sees the editor (especially
    // important on phones where the channels panel + editor combined
    // exceeds the viewport).
    try { ed.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); } catch (e) {}
}

function closeChannelEditor() {
    editingChannelIdx = null;
    lastChannelsSig = '';
    const ed = document.getElementById('channel-editor');
    if (ed) ed.classList.add('hidden');
}

async function saveChannelEdit() {
    if (editingChannelIdx === null) return;
    const idx = editingChannelIdx;
    const ce = (id) => document.getElementById(id);
    const name = ce('ce-name').value.trim();
    const folder = ce('ce-folder').value.trim();
    const tag = ce('ce-tag').value.trim();
    const swatch = document.querySelector('.ce-swatch.selected');
    const color = swatch ? swatch.dataset.color : CHANNEL_COLOR_PALETTE[idx];
    try {
        const r = await fetch('/api/channel/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ idx, name, folder, color, tag_filter: tag })
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.error || ('HTTP ' + r.status));
        if (data.result && data.result.ok === false) {
            throw new Error(data.result.error || 'save failed');
        }
        setStatus(`Saved Channel ${idx + 1}`);
        closeChannelEditor();
    } catch (e) {
        showError('Save channel failed: ' + e.message);
    }
}

async function setCurrentFolderForEditingChannel() {
    if (editingChannelIdx === null) return;
    const idx = editingChannelIdx;
    try {
        const r = await fetch('/api/channel/set_current', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ idx })
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.error || ('HTTP ' + r.status));
        const result = data.result || {};
        if (result.ok === false) throw new Error(result.error || 'set failed');
        // Reflect new folder in the open editor immediately so the user
        // sees what was captured. Server returns the updated channel dict.
        const ch = result.channel || {};
        if (ch.folder) document.getElementById('ce-folder').value = ch.folder;
        setStatus(`Channel ${idx + 1}: captured current folder`);
    } catch (e) {
        showError('Capture folder failed: ' + e.message);
    }
}

function wireChannelEditor() {
    const ed = document.getElementById('channel-editor');
    if (!ed || ed.dataset.wired) return;
    ed.dataset.wired = 'true';
    document.getElementById('ce-close').addEventListener('click', closeChannelEditor);
    document.getElementById('ce-cancel').addEventListener('click', closeChannelEditor);
    document.getElementById('ce-save').addEventListener('click', saveChannelEdit);
    document.getElementById('ce-use-current').addEventListener('click', setCurrentFolderForEditingChannel);
}

function wirePadHandlers() {
    // (S2 mirror pads removed with the S2 panel — audit 2026-05-16.
    // Kept the function shell + "clear all" hookup below so DOMContentLoaded
    // still finds it.)
    const clearBtn = document.getElementById('clips-clear-all');
    if (clearBtn && !clearBtn.dataset.wired) {
        clearBtn.dataset.wired = 'true';
        clearBtn.addEventListener('click', async () => {
            if (!confirm('Clear all cue points on this video?')) return;
            try {
                await postAction('clip/clear_current', {});
            } catch (e) {
                showError('Clear failed: ' + e.message);
            }
        });
    }
    // Sort dropdown — drives the same filter pipeline as the chips.
    const sortSel = document.getElementById('clips-sort');
    if (sortSel && !sortSel.dataset.wired) {
        sortSel.dataset.wired = 'true';
        sortSel.value = sortMode;
        sortSel.addEventListener('change', () => {
            sortMode = sortSel.value || 'added';
            _lastClipsRenderSig = '';
            renderClips(_lastClipsRaw, _lastAllTagsList);
        });
    }
}

function wireActiveTagInput() {
    const input = document.getElementById('active-tag-input');
    const clearBtn = document.getElementById('active-tag-clear');
    if (!input || input.dataset.wired) return;
    input.dataset.wired = 'true';
    input.addEventListener('focus', () => { _activeTagInputFocused = true; });
    let lastCommitted = input.value || '';
    let debounceTimer = null;
    const commit = () => {
        const t = (input.value || '').trim();
        if (t === lastCommitted) return;
        lastCommitted = t;
        postAction('active_tag/set', { tag: t }).catch(() => {});
    };
    // Auto-commit 500ms after typing stops — no Enter required.
    input.addEventListener('input', () => {
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(commit, 500);
    });
    input.addEventListener('blur', () => {
        _activeTagInputFocused = false;
        if (debounceTimer) { clearTimeout(debounceTimer); debounceTimer = null; }
        commit();
    });
    input.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter') {
            ev.preventDefault();
            input.blur();  // triggers commit
        } else if (ev.key === 'Escape') {
            ev.preventDefault();
            input.blur();
        }
    });
    if (clearBtn && !clearBtn.dataset.wired) {
        clearBtn.dataset.wired = 'true';
        clearBtn.addEventListener('click', () => {
            input.value = '';
            lastCommitted = '';
            postAction('active_tag/set', { tag: '' }).catch(() => {});
        });
    }
}

// ── Preview panel ───────────────────────────────────────────────────
// Shows a live MJPEG stream of the deck the crossfader is staged on
// (state.crossfade.preview_deck_idx). Reuses the per-deck preview-stream
// infrastructure — /preview/<n>.mjpg is multipart/x-mixed-replace, so
// the <img> swaps frames natively and smoothly. We only (re)point the
// src when the staged deck actually CHANGES — re-setting it every poll
// would restart the MJPEG connection and stutter.
function renderPreviewMonitor(state) {
    const img = document.getElementById('preview-monitor');
    const off = document.getElementById('monitor-offline');
    const label = document.getElementById('preview-deck-label');
    const nameEl = document.getElementById('preview-name');
    if (!img) return;
    // Wire the error handler once — covers "deck shows in state but its
    // preview stream isn't actually running".
    if (!img.dataset.wired) {
        img.dataset.wired = 'true';
        img.addEventListener('error', () => {
            img.classList.add('hidden');
            if (off) off.classList.remove('hidden');
        });
    }
    const xf = state.crossfade || {};
    const decks = state.decks || [];
    const idx = (typeof xf.preview_deck_idx === 'number') ? xf.preview_deck_idx : 0;
    const deck = decks[idx];
    const hasPreview = !!(deck && (deck.filepath || deck.name));
    if (hasPreview) {
        if (img.dataset.deck !== String(idx)) {
            img.src = '/preview/' + idx + '.mjpg';
            img.dataset.deck = String(idx);
        }
        img.classList.remove('hidden');
        if (off) off.classList.add('hidden');
        if (label) label.textContent = 'Deck ' + (idx + 1);
        if (nameEl) nameEl.textContent = fmtBasename(deck.filepath || deck.name || '');
    } else {
        if (img.dataset.deck !== undefined) {
            img.removeAttribute('src');
            delete img.dataset.deck;
        }
        img.classList.add('hidden');
        if (off) off.classList.remove('hidden');
        if (label) label.textContent = 'no deck staged';
        if (nameEl) nameEl.textContent = '';
    }
}

function start() {
    wirePadHandlers();
    wireLibrarySearch();
    wireChannelEditor();
    wireBpmBandPicker();
    wireActiveTagInput();
    wireBankPreview();
    wireAutoBankButton();
    wireTagSearchInput();
    wireOpenersLauncher();
    wireMagicMix();
    wirePickerPanel();
    wireVotesPanel();
    wireClipSearchPanel();
    wireArcBadge();
    wirePhraseTicker();
    setStatus('Connecting...');
    pollState();
    pollTimer = setInterval(pollState, 500); // 2Hz polling
}

// Kick off
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
} else {
    start();
}

// ── Analytics ───────────────────────────────────────────────────────
// Appended-only: wraps renderState() so we don't have to edit it. The
// analytics blob is published on state.analytics by main.py once the
// 2.5s set_analytics() timer is wired; until then we show "Analytics
// loading..." in every block.

// Cache the last rendered analytics signature so we don't tear down
// DOM on every 500ms poll (the top-10 list would otherwise rebuild
// itself, costing scroll position on iPad).
let _lastAnalyticsSig = '';
// Hold onto the most recently polled analytics blob so the Refresh
// button can force a re-render without a network round-trip.
let _lastAnalyticsBlob = null;

function renderAnalytics(an) {
    const blob = an || {};
    _lastAnalyticsBlob = blob;
    const sig = JSON.stringify(blob);
    if (sig === _lastAnalyticsSig) return;
    _lastAnalyticsSig = sig;

    const isEmpty = !blob || Object.keys(blob).length === 0;

    // ── Today counters ────────────────────────────────────────────
    const today = blob.today || {};
    const fmt = (v) => (v === undefined || v === null) ? '--' : String(v);
    const setText = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };
    setText('an-fires',   isEmpty ? '--' : fmt(today.fires_today));
    setText('an-beats',   isEmpty ? '--' : fmt(blob.beats_today));
    setText('an-flips',   isEmpty ? '--' : fmt(blob.flips_today));
    setText('an-alltime', isEmpty ? '--' : fmt(today.all_time_clips));

    // ── Top-10 clips ──────────────────────────────────────────────
    const topList = document.getElementById('analytics-top-list');
    if (topList) {
        const topClips = Array.isArray(blob.top_clips) ? blob.top_clips : [];
        if (isEmpty) {
            topList.innerHTML = '<li class="analytics-empty">Analytics loading...</li>';
        } else if (topClips.length === 0) {
            topList.innerHTML = '<li class="analytics-empty">No plays yet</li>';
        } else {
            topList.innerHTML = topClips.map((c, i) => {
                const name = escapeHTML(c.name || (c.id || '').slice(0, 8) || 'clip');
                const cnt = Number(c.play_count || 0);
                const bpmVal = Number(c.bpm || 0);
                const bpm = bpmVal > 0
                    ? `<span class="atc-bpm">${Math.round(bpmVal)}</span>`
                    : '<span class="atc-bpm"></span>';
                return `<li class="analytics-top-clip">`
                     + `<span class="atc-rank">${i + 1}</span>`
                     + `<span class="atc-name">${name}</span>`
                     + bpm
                     + `<span class="atc-count">${cnt}</span>`
                     + `</li>`;
            }).join('');
        }
    }

    // ── BPM histogram ─────────────────────────────────────────────
    const barsBox = document.getElementById('analytics-bpm-bars');
    if (barsBox) {
        const hist = Array.isArray(blob.bpm_histogram) ? blob.bpm_histogram : [];
        if (isEmpty) {
            barsBox.innerHTML = '<div class="analytics-empty">Analytics loading...</div>';
        } else if (hist.length === 0) {
            barsBox.innerHTML = '<div class="analytics-empty">No BPM data yet</div>';
        } else {
            const maxCount = Math.max(1, ...hist.map(b => Number(b.count) || 0));
            barsBox.innerHTML = hist.map(b => {
                const label = escapeHTML(b.band || '');
                const cnt = Number(b.count || 0);
                const pct = Math.round((cnt / maxCount) * 100);
                const cls = cnt === 0 ? 'analytics-bar empty' : 'analytics-bar';
                return `<div class="analytics-row">`
                     + `<span class="ar-label">${label}</span>`
                     + `<span class="ar-bar-bg"><span class="${cls}" style="width:${pct}%"></span></span>`
                     + `<span class="ar-count">${cnt}</span>`
                     + `</div>`;
            }).join('');
        }
    }

    // ── Disk usage line ───────────────────────────────────────────
    const diskLine = document.getElementById('analytics-disk-line');
    if (diskLine) {
        const disk = blob.disk_usage || {};
        if (isEmpty) {
            diskLine.textContent = 'Analytics loading...';
        } else {
            const proxyMb = Number(disk.proxy_mb || 0).toFixed(1);
            const thumbMb = Number(disk.thumb_mb || 0).toFixed(1);
            const total = Number(disk.total_files_indexed || 0);
            diskLine.innerHTML =
                  `Proxies <span class="adl-num">${proxyMb} MB</span>`
                + `<span class="adl-sep">/</span>`
                + `Thumbs <span class="adl-num">${thumbMb} MB</span>`
                + `<span class="adl-sep">/</span>`
                + `<span class="adl-num">${total}</span> files`;
        }
    }
}

// Wrap renderState() so we tag-along on every poll without editing
// the original function. `function renderState(state)` at top level
// becomes window.renderState (this script is not a module), so we can
// stash + reassign safely. Done at script-load time — before start()
// fires the first pollState().
(function _wireAnalyticsRender() {
    const _origRenderState = window.renderState;
    if (typeof _origRenderState !== 'function') {
        // Defensive — if the rest of the script failed to load, just
        // skip wiring. The panel will keep showing "Analytics loading..."
        return;
    }
    window.renderState = function _wrappedRenderState(state) {
        _origRenderState(state);
        try {
            renderAnalytics(state && state.analytics ? state.analytics : {});
        } catch (e) {
            // Don't let an analytics bug brick the rest of the UI.
            console.error('renderAnalytics failed:', e);
        }
    };
    // Refresh button: re-render from cached blob (no fetch). Useful
    // if a CSS reload nuked the DOM mid-session.
    document.addEventListener('DOMContentLoaded', () => {
        const btn = document.getElementById('analytics-refresh');
        if (btn && !btn.dataset.wired) {
            btn.dataset.wired = 'true';
            btn.addEventListener('click', () => {
                // Force a redraw by clearing the dedupe signature.
                _lastAnalyticsSig = '';
                renderAnalytics(_lastAnalyticsBlob || {});
            });
        }
    });
    // If DOMContentLoaded already fired (script at end of body), wire
    // the button immediately.
    if (document.readyState !== 'loading') {
        const btn = document.getElementById('analytics-refresh');
        if (btn && !btn.dataset.wired) {
            btn.dataset.wired = 'true';
            btn.addEventListener('click', () => {
                _lastAnalyticsSig = '';
                renderAnalytics(_lastAnalyticsBlob || {});
            });
        }
    }
})();

// CDJ Tier-2 phase meter — local 60Hz extrapolation from polled bpm/beat
// data. Runs in its own IIFE so it's independent of the main pollState
// loop. Reads window._phaseState (set in pollState) and writes CSS vars
// + class toggles on the phase meter DOM. No expensive layout — only
// custom-prop assignment + classlist toggle.
(() => {
    const fillEl = document.getElementById('phase-fill');
    const beatEl = document.getElementById('phase-beat');
    if (!fillEl || !beatEl) return;

    let lastBeatIdx = -1;
    let lastPhase = 0;

    function tick() {
        requestAnimationFrame(tick);
        const s = window._phaseState;
        if (!s || !s.bpm || !s.enabled) {
            fillEl.style.setProperty('--phase', '0%');
            fillEl.style.setProperty('--phase-color', '#333');
            beatEl.style.color = '#333';
            return;
        }
        const beatPeriodMs = 60000 / s.bpm;
        const now = performance.now();
        const ageMs = s.beatAgeMs + (now - s.receivedAt);
        // Phase 0..1 inside the current beat. Phase 0 = just hit, phase
        // 1 ~ next beat about to fire.
        const beatsSinceCapture = Math.floor(ageMs / beatPeriodMs);
        const phase = (ageMs % beatPeriodMs) / beatPeriodMs;

        // Derive the bar position (1..4) from total beats since first
        // capture. Reset to 1 if we've drifted (defensive).
        const totalBeats = (s.barIdx || 0) + beatsSinceCapture;
        const barStep = ((totalBeats % 4) + 4) % 4 + 1;  // 1..4

        // Detect a fresh beat — phase just wrapped near 0.
        if (phase < lastPhase - 0.4) {
            // Beat hit. Pulse the counter.
            beatEl.classList.add('beat-pulse');
            setTimeout(() => beatEl.classList.remove('beat-pulse'), 80);
        }
        lastPhase = phase;

        fillEl.style.setProperty('--phase', (phase * 100).toFixed(1) + '%');
        fillEl.style.setProperty('--phase-color', '#ef4444');
        beatEl.textContent = String(barStep);
        if (!beatEl.classList.contains('beat-pulse')) {
            beatEl.style.color = '#ef4444';
        }
    }
    requestAnimationFrame(tick);
})();
