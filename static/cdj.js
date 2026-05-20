/* CDJ Tier-3 v2 — /cdj.html immersive view, VJ-native layout.

   See VJ_ROADMAP.md for the design rationale.

   - Polls /api/state every 500ms.
   - Animates phase meter locally at 60Hz via rAF (smooth between polls).
   - Renders bank grid + bank tabs.
   - LIVE preview = lib_thumbnail of currently-playing file.
   - CUE preview = lib_thumbnail of selected preview deck's file (or
     mjpeg stream when available — currently static thumb fallback).
   - Color-tint per pad cell — pulls palette tag once color_tagger.py
     finishes; until then pads stay default-untinted. The fetch is
     lazy + cached per-filepath so we don't hammer the DB.

   iOS Safari rules per Setpiece lessons:
   - Visible error trap in <head>.
   - cache: 'no-store' on every fetch.
   - All module-state hoisted to top (no temporal-dead-zone). */

"use strict";

// ── shared SHA-256 (matches thumbnails.py lib_thumbnail_hash) ──────────
const HASH_CACHE = new Map();
async function libThumbHash(filepath) {
    if (!filepath) return "";
    if (HASH_CACHE.has(filepath)) return HASH_CACHE.get(filepath);
    const enc = new TextEncoder();
    const buf = await crypto.subtle.digest("SHA-256", enc.encode(filepath));
    const hex = Array.from(new Uint8Array(buf))
        .map(b => b.toString(16).padStart(2, "0"))
        .join("");
    const short = hex.slice(0, 16);
    HASH_CACHE.set(filepath, short);
    return short;
}

// ── color tag cache (palette + hue, fetched lazily per filepath) ───────
// Empty until /api/file/colors lands. Until then all pads show no tint.
const COLOR_CACHE = new Map();   // filepath -> {palette:[3 hex], hue, therm}
const COLOR_INFLIGHT = new Set();
async function fetchColorsFor(filepath) {
    if (!filepath || COLOR_CACHE.has(filepath) || COLOR_INFLIGHT.has(filepath)) {
        return COLOR_CACHE.get(filepath) || null;
    }
    COLOR_INFLIGHT.add(filepath);
    try {
        const r = await fetch(
            "/api/file/colors?path=" + encodeURIComponent(filepath),
            { cache: "no-store" }
        );
        if (r.ok) {
            const d = await r.json();
            if (d && d.palette) {
                COLOR_CACHE.set(filepath, d);
                return d;
            }
        }
    } catch (e) {
        // endpoint might not exist yet; that's fine — pads stay untinted
    } finally {
        COLOR_INFLIGHT.delete(filepath);
    }
    // Cache an empty result so we don't re-attempt every poll
    COLOR_CACHE.set(filepath, null);
    return null;
}

// ── phase-meter state ──────────────────────────────────────────────────
let phaseState = { bpm: 0, beatAgeMs: 0, receivedAt: 0, enabled: false, bar: 0 };
let lastPhase = 0;

// ── DOM ref cache ──────────────────────────────────────────────────────
const els = {};
function el(id) {
    if (!els[id]) els[id] = document.getElementById(id);
    return els[id];
}

// ── poll loop ──────────────────────────────────────────────────────────
async function poll() {
    try {
        // /api/state is the main snapshot. /api/set-arc is fetched in
        // parallel — this lets the set-arc badge work without waiting
        // on a restart to pick up state-injection changes in
        // http_server.py. If set-arc 404s on an older server build,
        // we just skip the badge.
        const [rState, rArc] = await Promise.all([
            fetch("/api/state",   { cache: "no-store" }),
            fetch("/api/set-arc", { cache: "no-store" }).catch(() => null),
        ]);
        if (!rState.ok) throw new Error(`HTTP ${rState.status}`);
        const s = await rState.json();
        if (rArc && rArc.ok && !s.set_arc) {
            try { s.set_arc = await rArc.json(); } catch (_) {}
        }
        el("badge-conn").textContent = "LIVE";
        el("badge-conn").className = "badge badge-conn-on";
        await render(s);
    } catch (e) {
        el("badge-conn").textContent = "OFFLINE";
        el("badge-conn").className = "badge badge-conn-off";
        console.error("poll failed", e);
    }
}

async function render(s) {
    // === Status badges (Row 1) ============================================
    const ar = s.audio_reactive || {};
    const flipMode = (ar.flip_on_beat_mode || ar.flip_mode ||
                      (ar.flip_on_beat ? "BANK" : "OFF"))
        .toString().toUpperCase();
    setBadge(el("badge-react"), ar.enabled, "REACT");
    setBadgeFlip(el("badge-flip"), flipMode);
    setBadge(el("badge-tempo"), !!s.tempo_lock, "TEMPO", "amber");
    setBadge(el("badge-stutter"), !!s.stutter_active, "STUTTER", "amber");
    const ld = s.lyric_drive || {};
    const pinned = !!(ld.pinned_release || ld.pinned_file ||
                      s.pin_release || s.pin_file);
    setBadge(el("badge-pin"), pinned, "PIN", "green");
    // Set-arc badge — phase-specific color when enabled, dim when off.
    // When AUTO mode is on, prepend "A·" to the label and add a thin
    // pulsing white border so the operator knows the system is
    // driving the phase. Tap to cycle phase; long-press handled below.
    const arc = s.set_arc || {};
    const arcBadge = el("badge-arc");
    if (arcBadge) {
        if (arc.enabled) {
            const phaseColors = {
                opening:   {bg: "#1e3a8a", fg: "#bfdbfe"},   // deep blue
                build:     {bg: "#d97706", fg: "#fef3c7"},   // warm amber
                peak:      {bg: "#dc2626", fg: "#fee2e2"},   // red
                breakdown: {bg: "#1e293b", fg: "#94a3b8"},   // dark slate
            };
            const pc = phaseColors[arc.phase] || phaseColors.opening;
            const labelTxt = (arc.label || arc.phase || "").toUpperCase();
            arcBadge.textContent = (arc.auto ? "A·" : "") + "ARC " + labelTxt;
            arcBadge.className = "badge";
            arcBadge.style.background = pc.bg;
            arcBadge.style.color = pc.fg;
            // Auto mode -> brighter glow + thin white outline so it
            // visibly distinguishes from manual mode at a glance.
            arcBadge.style.boxShadow = arc.auto
                ? `0 0 14px ${pc.bg}, 0 0 0 1px rgba(255,255,255,0.6) inset`
                : `0 0 10px ${pc.bg}aa`;
        } else {
            arcBadge.textContent = "ARC";
            arcBadge.className = "badge badge-off";
            arcBadge.style.background = "";
            arcBadge.style.color = "";
            arcBadge.style.boxShadow = "";
        }
    }

    // === Meta strip (Row 2) ==============================================
    const pb = s.playback || {};
    const fname = pb.file ? basename(pb.file) : "No video loaded";
    el("meta-filename").textContent = fname;
    el("meta-position").textContent = fmtTime(pb.position || 0);
    el("meta-duration").textContent = fmtTime(pb.duration || 0);
    const bpm = ar.bpm;
    el("meta-bpm").textContent = bpm ? Math.round(bpm) : "---";
    const speed = pb.speed || s.playback_speed || 1.0;
    el("meta-speed").textContent = `${speed.toFixed(2)}x`;
    el("meta-speed").style.color = (Math.abs(speed - 1.0) > 0.01)
        ? "var(--accent-slip)" : "var(--text-tertiary)";

    // === Preview row (Row 3) =============================================
    // LIVE = currently playing file's library thumbnail.
    if (pb.file) {
        const hash = await libThumbHash(pb.file);
        const url = `/lib_thumbnails/${hash}.jpg`;
        if (el("preview-live-img").dataset.src !== url) {
            el("preview-live-img").dataset.src = url;
            el("preview-live-img").src = url;
        }
    }
    // CUE = preview-deck file's library thumbnail. State exposes
    // decks[]; the preview-deck index is set by FX2 cluster on S2.
    // Tolerate variations: state.preview_deck or state.decks[?].is_preview.
    const decks = s.decks || [];
    let cueFile = null;
    if (typeof s.preview_deck_idx === "number" &&
        decks[s.preview_deck_idx]) {
        cueFile = decks[s.preview_deck_idx].file;
    } else {
        const previewDeck = decks.find(d => d && d.is_preview);
        if (previewDeck) cueFile = previewDeck.file;
    }
    if (cueFile) {
        const hash = await libThumbHash(cueFile);
        const url = `/lib_thumbnails/${hash}.jpg`;
        if (el("preview-cue-img").dataset.src !== url) {
            el("preview-cue-img").dataset.src = url;
            el("preview-cue-img").src = url;
        }
    } else {
        el("preview-cue-img").removeAttribute("src");
        el("preview-cue-img").dataset.src = "";
    }
    // LIVE progress bar (mini)
    const dur = pb.duration || 0;
    const pos = pb.position || 0;
    if (dur > 0) {
        const pct = Math.max(0, Math.min(100, (pos / dur) * 100));
        el("preview-progress-fill").style.width = pct.toFixed(2) + "%";
    }

    // === Phase snapshot for rAF loop =====================================
    if (bpm && bpm > 30 && bpm < 250) {
        phaseState = {
            bpm,
            beatAgeMs: ar.last_beat_age_ms || 0,
            receivedAt: performance.now(),
            enabled: !!ar.enabled,
            bar: phaseState.bar,
        };
    } else {
        phaseState.enabled = false;
    }

    // === Bank grid + tabs (Rows 4 + 5) ===================================
    const banks = s.banks || [];
    const activeBank = banks.find(b => b.active) || banks[0] || {};
    await renderBankGrid(s.bank_clips || [], pb.file);
    renderBankTabs(banks);
}

async function renderBankGrid(clips, currentFile) {
    const grid = el("bank-grid");
    const cells = clips.slice(0, 16);
    while (cells.length < 16) cells.push(null);

    // Pre-compute hashes in parallel.
    const hashes = await Promise.all(cells.map(c =>
        c && c.filepath ? libThumbHash(c.filepath) : Promise.resolve("")
    ));
    // Fire-and-forget color fetches (don't block render).
    cells.forEach(c => { if (c && c.filepath) fetchColorsFor(c.filepath); });

    const html = cells.map((c, i) => {
        if (!c) return `<div class="pad pad-empty"></div>`;
        const hash = hashes[i];
        const src = hash ? `/lib_thumbnails/${hash}.jpg` : "";
        const isActive = c.filepath && c.filepath === currentFile;
        const isPinned = !!c.pinned;
        const tint = paletteToTint(COLOR_CACHE.get(c.filepath));
        let cls = "pad";
        if (isActive) cls += " pad-active";
        if (isPinned) cls += " pad-pinned";
        const styleTint = tint ? ` style="--pad-tint:${tint};"` : "";
        const img = src
            ? `<img class="pad-thumb" src="${src}" alt="">`
            : `<div class="pad-thumb"></div>`;
        const hueDot = tint ? `<span class="pad-hue"></span>` : "";
        return `<div class="${cls}"${styleTint}>${img}<span class="pad-label">${i + 1}</span>${hueDot}</div>`;
    }).join("");
    if (grid.dataset.lastHtml !== html) {
        grid.dataset.lastHtml = html;
        grid.innerHTML = html;
    }
}

function renderBankTabs(banks) {
    const tabs = el("bank-tabs");
    if (!tabs) return;
    const letters = ["A","B","C","D","E","F","G","H"];
    const html = letters.map(L => {
        const b = banks.find(x => x.letter === L) || {};
        const name = (b.name || "").trim();
        const cls = b.active ? "bank-tab active" : "bank-tab";
        return `<div class="${cls}">
            <div class="bank-tab-letter">${L}</div>
            <div class="bank-tab-name">${name || "&middot;"}</div>
        </div>`;
    }).join("");
    if (tabs.dataset.lastHtml !== html) {
        tabs.dataset.lastHtml = html;
        tabs.innerHTML = html;
    }
}

// ── palette → CSS tint color ───────────────────────────────────────────
// Pick the dominant (first) hex from the palette tag's 3-color list.
// Falls back to null if no palette data yet (cell shows no tint).
function paletteToTint(colorData) {
    if (!colorData || !colorData.palette || !colorData.palette.length) return null;
    let hex = colorData.palette[0];
    if (!hex.startsWith("#")) hex = "#" + hex;
    return hex;
}

// ── helpers ────────────────────────────────────────────────────────────
function basename(p) {
    if (!p) return "";
    const m = p.match(/[^\\\/]+$/);
    return m ? m[0] : p;
}
function fmtTime(sec) {
    sec = Math.max(0, Math.floor(sec || 0));
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`;
}
function setBadge(node, on, label, variant = "red") {
    if (!node) return;
    node.textContent = label;
    if (on) {
        const klass = variant === "green" ? "badge-on-green"
                    : variant === "amber" ? "badge-on-amber"
                    : "badge-on";
        node.className = `badge ${klass}`;
    } else {
        node.className = "badge badge-off";
    }
}
function setBadgeFlip(node, mode) {
    if (!node) return;
    node.textContent = `FLIP ${mode}`;
    if (mode === "OFF") {
        node.className = "badge badge-off";
    } else if (mode === "BANK") {
        node.className = "badge badge-on-green";
    } else if (mode === "FOLDER") {
        node.className = "badge badge-on-amber";
    } else {
        node.className = "badge badge-on";
    }
}

// ── 60Hz phase animation ───────────────────────────────────────────────
function phaseTick() {
    requestAnimationFrame(phaseTick);
    const fill = el("phase-bar-fill");
    const counter = el("phase-counter");
    if (!fill || !counter) return;
    const s = phaseState;
    if (!s.bpm || !s.enabled) {
        fill.style.width = "0%";
        fill.classList.add("idle");
        counter.textContent = "·";
        counter.classList.add("idle");
        counter.classList.remove("beat-pulse");
        return;
    }
    fill.classList.remove("idle");
    counter.classList.remove("idle");
    const beatPeriod = 60000 / s.bpm;
    const now = performance.now();
    const ageMs = s.beatAgeMs + (now - s.receivedAt);
    const beatsSince = Math.floor(ageMs / beatPeriod);
    const phase = (ageMs % beatPeriod) / beatPeriod;
    const barStep = (((phaseState.bar + beatsSince) % 4) + 4) % 4 + 1;

    if (phase < lastPhase - 0.4) {
        counter.classList.add("beat-pulse");
        setTimeout(() => counter.classList.remove("beat-pulse"), 80);
    }
    lastPhase = phase;

    fill.style.width = (phase * 100).toFixed(2) + "%";
    counter.textContent = String(barStep);
}

// ── ARC badge interactions: tap = cycle phase, long-press = toggle auto.
// Wire once at boot. The badge element exists immediately from HTML.
(function wireArcBadge() {
    const b = document.getElementById("badge-arc");
    if (!b) return;
    b.style.cursor = "pointer";
    b.style.userSelect = "none";
    b.style.webkitUserSelect = "none";
    let pressTimer = null;
    let longPressed = false;
    const LONG_PRESS_MS = 550;
    const post = (path) => fetch(path, {method: "POST",
        headers: {"Content-Type": "application/json"}, body: "{}"})
        .catch(e => console.error("arc post failed", e));
    const onDown = (ev) => {
        longPressed = false;
        clearTimeout(pressTimer);
        pressTimer = setTimeout(() => {
            longPressed = true;
            post("/api/set-arc/auto");
        }, LONG_PRESS_MS);
    };
    const onUp = (ev) => {
        clearTimeout(pressTimer);
        if (!longPressed) {
            ev.preventDefault();
            post("/api/set-arc/cycle");
        }
        longPressed = false;
    };
    const onCancel = () => { clearTimeout(pressTimer); longPressed = false; };
    // preventDefault on touchend ALWAYS (not just in onUp) so iOS
    // Safari doesn't synth a mouse event ~300ms later that re-fires
    // the handler chain and undoes the long-press toggle.
    b.addEventListener("touchstart", onDown, {passive: true});
    b.addEventListener("touchend",   (ev) => {
        if (ev.cancelable) ev.preventDefault();
        onUp(ev);
    });
    b.addEventListener("touchcancel", onCancel);
    b.addEventListener("mousedown",  onDown);
    b.addEventListener("mouseup",    onUp);
    b.addEventListener("mouseleave", onCancel);
})();

// ── boot ───────────────────────────────────────────────────────────────
poll();
setInterval(poll, 500);
requestAnimationFrame(phaseTick);
