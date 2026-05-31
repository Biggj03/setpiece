// Resolume control panel (standalone). Talks to resolume_panel.py /api/*.
// iOS Safari rules baked in: state hoisted to top, visible error strip,
// POST-only (never cached), throttled fader sends.

var STATE = null;
var ACTIVE_TAB = "mix";
var POLL_MS = 700;
var pollTimer = null;
var blackedOut = false;
var lastSent = {};
var SEND_THROTTLE_MS = 60;
var dragging = {};
var clipLayer = 1;
var gridCols = 5;   // clip-grid columns; operator-adjustable (3-6)
var masterDragging = false;  // suppress poll-sync while riding master

function showErr(m) {
  var b = document.getElementById("errbar");
  if (b) { b.textContent = String(m); b.classList.remove("hidden"); }
}
function clearErr() {
  var b = document.getElementById("errbar");
  if (b) b.classList.add("hidden");
}
window.onerror = function (m, s, ln, c) { showErr("JS " + m + " @" + ln + ":" + c); };
window.addEventListener("unhandledrejection", function (e) {
  showErr("promise: " + (e.reason && e.reason.message ? e.reason.message : e.reason));
});

function api(path, body) {
  return fetch("/api/" + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  }).then(function (r) { return r.json(); });
}
function apiThrottled(key, path, body) {
  var now = Date.now();
  if (lastSent[key] && now - lastSent[key] < SEND_THROTTLE_MS) return;
  lastSent[key] = now;
  api(path, body).catch(function (e) { showErr(path + ": " + e); });
}

function getLayer(idx) {
  if (!STATE || !STATE.layers) return null;
  for (var i = 0; i < STATE.layers.length; i++)
    if (STATE.layers[i].index === idx) return STATE.layers[i];
  return null;
}

// ── status light ──
function paintStatus(s) {
  var dot = document.getElementById("statusdot");
  var txt = document.getElementById("statustext");
  if (s && s.reachable) {
    dot.className = "dot dot-ok";
    txt.textContent = (s.product || "Arena") + (s.bridge_enabled ? "" : " (OSC off)");
  } else {
    dot.className = "dot dot-bad";
    txt.textContent = "Arena offline";
  }
}

// ── MIX: layer faders ──
function renderLayers(s) {
  var wrap = document.getElementById("layers");
  var layers = (s && s.layers) || [];
  if (wrap.children.length !== layers.length) {
    wrap.innerHTML = "";
    layers.forEach(function (L) { wrap.appendChild(buildLayer(L)); });
  }
  layers.forEach(updateLayer);
}
function buildLayer(L) {
  var div = document.createElement("div");
  div.className = "layer"; div.id = "layer-" + L.index;

  var head = document.createElement("div"); head.className = "layer-head";
  var name = document.createElement("span"); name.className = "layer-name";
  name.textContent = L.name || ("Layer " + L.index);
  var meta = document.createElement("span"); meta.className = "layer-meta";
  meta.id = "meta-" + L.index;
  var btns = document.createElement("div"); btns.className = "layer-btns";

  var bypass = document.createElement("button");
  bypass.className = "lbtn"; bypass.id = "bypass-" + L.index; bypass.textContent = "MUTE";
  bypass.onclick = function () {
    var cur = getLayer(L.index);
    var on = !(cur && cur.bypassed);
    api("layer_bypass", { layer: L.index, bypassed: on })
      .catch(function (e) { showErr("bypass: " + e); });
  };
  var clr = document.createElement("button");
  clr.className = "lbtn warn"; clr.textContent = "CLEAR";
  clr.onclick = function () {
    api("layer_clear", { layer: L.index }).catch(function (e) { showErr("clear: " + e); });
  };
  btns.appendChild(bypass); btns.appendChild(clr);
  head.appendChild(name); head.appendChild(meta); head.appendChild(btns);

  var row = document.createElement("div"); row.className = "fader-row";
  var f = document.createElement("input");
  f.type = "range"; f.min = "0"; f.max = "1"; f.step = "0.01"; f.id = "op-" + L.index;
  f.value = (L.opacity == null ? 1 : L.opacity);
  var val = document.createElement("span"); val.className = "fader-val"; val.id = "opval-" + L.index;
  f.addEventListener("input", function () {
    var v = parseFloat(f.value);
    val.textContent = Math.round(v * 100) + "%";
    apiThrottled("op-" + L.index, "layer_opacity", { layer: L.index, opacity: v });
  });
  ["touchstart", "mousedown"].forEach(function (ev) {
    f.addEventListener(ev, function () { dragging[L.index] = true; });
  });
  ["touchend", "mouseup"].forEach(function (ev) {
    f.addEventListener(ev, function () { dragging[L.index] = false; });
  });
  row.appendChild(f); row.appendChild(val);
  div.appendChild(head); div.appendChild(row);
  return div;
}
function updateLayer(L) {
  var div = document.getElementById("layer-" + L.index);
  if (!div) return;
  div.classList.toggle("bypassed", !!L.bypassed);
  div.classList.toggle("has-active", L.active_clip != null);
  var meta = document.getElementById("meta-" + L.index);
  if (meta) meta.textContent = L.loaded + "/" + L.columns +
    (L.active_clip ? "  • col " + L.active_clip : "");
  var bypass = document.getElementById("bypass-" + L.index);
  if (bypass) bypass.classList.toggle("on", !!L.bypassed);
  if (!dragging[L.index]) {
    var f = document.getElementById("op-" + L.index);
    var val = document.getElementById("opval-" + L.index);
    if (f && L.opacity != null) {
      f.value = L.opacity;
      if (val) val.textContent = Math.round(L.opacity * 100) + "%";
    }
  }
}

// ── master output fader ──
function initMaster() {
  var f = document.getElementById("master");
  var val = document.getElementById("master-val");
  if (!f) return;
  f.addEventListener("input", function () {
    var v = parseFloat(f.value);
    if (val) val.textContent = Math.round(v * 100) + "%";
    apiThrottled("master", "master", { level: v });
  });
  ["touchstart", "mousedown"].forEach(function (ev) {
    f.addEventListener(ev, function () { masterDragging = true; });
  });
  ["touchend", "mouseup"].forEach(function (ev) {
    f.addEventListener(ev, function () { masterDragging = false; });
  });
}

function reconcileMaster(snap) {
  // Keep the master fader synced to Arena's real master (panic, Arena
  // restart, or a move in Arena itself), unless the operator is dragging.
  if (masterDragging) return;
  if (!snap || typeof snap.master !== "number") return;
  var f = document.getElementById("master");
  var val = document.getElementById("master-val");
  if (f) {
    f.value = snap.master;
    if (val) val.textContent = Math.round(snap.master * 100) + "%";
  }
}

// ── crossfader ──
function initCrossfader() {
  var xf = document.getElementById("xfade");
  xf.addEventListener("input", function () {
    apiThrottled("xfade", "crossfader", { phase: parseFloat(xf.value) });
  });
}

// ── tempo: tap, beatsnap, resync ──
var tapTimes = [];          // recent tap timestamps (ms)
var tapClearTimer = null;   // clears tapTimes after taps stop (readout resyncs)
var beatsnapPending = null; // index we just set (suppress poll flicker)

function initTempo() {
  var tap = document.getElementById("tap");
  if (tap) tap.addEventListener("click", onTap);
  var resync = document.getElementById("resync");
  if (resync) resync.addEventListener("click", function () {
    api("resync", {}).catch(function (e) { showErr("resync: " + e); });
  });
  document.querySelectorAll(".bsbtn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var idx = parseInt(btn.getAttribute("data-index"), 10);
      beatsnapPending = idx;
      paintBeatsnap(idx);
      api("beatsnap", { index: idx })
        .then(function () { beatsnapPending = null; })
        .catch(function (e) { beatsnapPending = null; showErr("beatsnap: " + e); });
    });
  });
}

function onTap() {
  var now = Date.now();
  // A gap > 2s starts a fresh count (new tempo).
  if (tapTimes.length && now - tapTimes[tapTimes.length - 1] > 2000) {
    tapTimes = [];
  }
  tapTimes.push(now);
  if (tapTimes.length > 8) tapTimes.shift();   // rolling window
  // Once tapping stops, drop the tap history so reconcileTempo resumes
  // syncing the BPM readout from Arena (the review's stale-readout fix).
  if (tapClearTimer) clearTimeout(tapClearTimer);
  tapClearTimer = setTimeout(function () { tapTimes = []; }, 3000);
  if (tapTimes.length >= 2) {
    // Average interval across the window -> BPM.
    var span = tapTimes[tapTimes.length - 1] - tapTimes[0];
    var bpm = 60000 / (span / (tapTimes.length - 1));
    if (bpm >= 20 && bpm <= 500) {
      var rounded = Math.round(bpm * 10) / 10;
      var el = document.getElementById("bpm-val");
      if (el) el.textContent = rounded.toFixed(1) + " BPM";
      api("tempo", { bpm: rounded }).catch(function (e) { showErr("tempo: " + e); });
    }
  }
}

function paintBeatsnap(idx) {
  document.querySelectorAll(".bsbtn").forEach(function (b) {
    b.classList.toggle("on", parseInt(b.getAttribute("data-index"), 10) === idx);
  });
}

function reconcileTempo(snap) {
  if (!snap) return;
  // BPM readout (only when the operator isn't mid-tap).
  if (typeof snap.tempo === "number" && !tapTimes.length) {
    var el = document.getElementById("bpm-val");
    if (el) el.textContent = snap.tempo.toFixed(1) + " BPM";
  }
  // Beatsnap selector reflects Arena unless we just changed it.
  if (beatsnapPending === null && typeof snap.beatsnap === "number") {
    paintBeatsnap(snap.beatsnap);
  }
}

// ── CLIPS: scene (column) row ──
// Firing a column triggers that scene across ALL layers in one tap — a core
// VJ move. Column count is derived from the live snapshot (max columns over
// the layers), so it tracks the comp without a dedicated endpoint.
function renderScenes(s) {
  var row = document.getElementById("scene-row");
  if (!row) return;
  var layers = (s && s.layers) || [];
  var n = 0;
  layers.forEach(function (L) { if (L.columns > n) n = L.columns; });
  // Rebuild only when the count changes (avoids clobbering taps each poll).
  if (row.childElementCount === n) return;
  row.innerHTML = "";
  for (var c = 1; c <= n; c++) {
    (function (col) {
      var b = document.createElement("button");
      b.className = "scene-btn";
      b.textContent = col;
      b.onclick = function () {
        api("fire_column", { column: col })
          .catch(function (e) { showErr("scene: " + e); });
      };
      row.appendChild(b);
    })(c);
  }
}

// ── CLIPS grid ──
function initClipLayerPicker(s) {
  var sel = document.getElementById("clip-layer");
  var layers = (s && s.layers) || [];
  if (sel.children.length === layers.length && layers.length) return;
  sel.innerHTML = "";
  layers.forEach(function (L) {
    var o = document.createElement("option");
    o.value = L.index;
    o.textContent = (L.name || ("Layer " + L.index)) + " (" + L.loaded + ")";
    sel.appendChild(o);
  });
  sel.value = clipLayer;
  sel.onchange = function () { clipLayer = parseInt(sel.value, 10); loadClips(); };
}
function applyGridCols() {
  var grid = document.getElementById("clipgrid");
  if (grid) grid.style.gridTemplateColumns = "repeat(" + gridCols + ", 1fr)";
  document.querySelectorAll(".dbtn").forEach(function (b) {
    b.classList.toggle("on", parseInt(b.getAttribute("data-cols"), 10) === gridCols);
  });
}

function initGridDensity() {
  document.querySelectorAll(".dbtn").forEach(function (b) {
    b.onclick = function () {
      gridCols = parseInt(b.getAttribute("data-cols"), 10);
      applyGridCols();
    };
  });
  applyGridCols();
}

function loadClips() {
  api("clip_names", { layer: clipLayer }).then(function (res) {
    var grid = document.getElementById("clipgrid");
    grid.innerHTML = "";
    applyGridCols();
    var clips = (res && res.clips) || [];
    var L = getLayer(clipLayer);
    var active = L ? L.active_clip : null;
    clips.forEach(function (c) {
      var cell = document.createElement("div");
      cell.className = "cell" + (c.loaded ? "" : " empty") +
        (active === c.column ? " active" : "");
      // Thumbnail (loaded clips with a real thumb id). Lazy-loaded;
      // falls back silently to the text label if the image errors.
      if (c.thumb_id) {
        var img = document.createElement("img");
        img.className = "cell-thumb";
        img.loading = "lazy";
        img.src = "/thumb/" + c.thumb_id;
        img.onerror = function () { this.style.display = "none"; };
        cell.appendChild(img);
      }
      var col = document.createElement("div"); col.className = "cell-col";
      col.textContent = c.column;
      var nm = document.createElement("div"); nm.className = "cell-name";
      nm.textContent = c.loaded ? (c.name || "") : "-";
      cell.appendChild(col); cell.appendChild(nm);
      if (c.loaded) {
        cell.onclick = function () {
          api("connect_clip", { layer: clipLayer, column: c.column })
            .catch(function (e) { showErr("fire: " + e); });
        };
      }
      grid.appendChild(cell);
    });
    if (!clips.length)
      grid.innerHTML = '<div class="cell empty">no clips on this layer</div>';
  }).catch(function (e) { showErr("clip_names: " + e); });
}

// ── FX tab: dynamic effect faders ──
// Effects/params are composition-specific, so the tab DISCOVERS them from
// /api/effects and renders one fader per drivable (ParamRange) param. Each
// fader drives by the param's live id via /api/param, throttled, with a
// per-fader drag guard (mirrors the master fader's masterDragging pattern).
var fxDragging = {};   // param id -> true while held

function initFx() {
  var btn = document.getElementById("fx-refresh");
  if (btn) btn.addEventListener("click", loadEffects);
}

function loadEffects() {
  api("effects", {}).then(function (res) {
    var wrap = document.getElementById("fxlist");
    var effects = (res && res.effects) || [];
    wrap.innerHTML = "";
    if (!effects.length) {
      wrap.innerHTML = '<div class="cell empty">no effects on the composition</div>';
      return;
    }
    effects.forEach(function (eff) {
      var box = document.createElement("div");
      box.className = "fx-effect" + (eff.bypassed ? " bypassed" : "");
      var head = document.createElement("div");
      head.className = "fx-head";
      var nm = document.createElement("span");
      nm.className = "fx-name";
      nm.textContent = eff.display_name || eff.name;
      head.appendChild(nm);
      // Bypass toggle — kills the effect without losing its fader settings.
      // Lit = active, dim = bypassed. Drives the boolean bypass param by id.
      if (eff.bypass_id != null) {
        var tog = document.createElement("button");
        tog.className = "fx-bypass" + (eff.bypassed ? "" : " on");
        tog.textContent = eff.bypassed ? "OFF" : "ON";
        tog.onclick = function () {
          var nowOn = box.classList.contains("bypassed");  // currently bypassed -> turning ON
          var bypass = !nowOn;                             // toggle target bypass state
          // optimistic UI
          box.classList.toggle("bypassed", bypass);
          tog.classList.toggle("on", !bypass);
          tog.textContent = bypass ? "OFF" : "ON";
          api("param", { id: eff.bypass_id, value: bypass })
            .catch(function (e) { showErr("bypass: " + e); });
        };
        head.appendChild(tog);
      }
      box.appendChild(head);
      eff.params.forEach(function (p) {
        box.appendChild(buildFxParam(p));
      });
      wrap.appendChild(box);
    });
  }).catch(function (e) { showErr("effects: " + e); });
}

function buildFxParam(p) {
  var row = document.createElement("div");
  row.className = "fx-param";
  var label = document.createElement("span");
  label.className = "fx-plabel";
  label.textContent = p.name;
  var f = document.createElement("input");
  f.type = "range";
  f.min = (p.min == null ? 0 : p.min);
  f.max = (p.max == null ? 1 : p.max);
  f.step = "0.01";
  f.value = (p.value == null ? 0 : p.value);
  var val = document.createElement("span");
  val.className = "fx-pval";
  val.textContent = fmtFx(p.value);
  f.addEventListener("input", function () {
    var v = parseFloat(f.value);
    val.textContent = fmtFx(v);
    apiThrottled("fx-" + p.id, "param", { id: p.id, value: v });
  });
  ["touchstart", "mousedown"].forEach(function (ev) {
    f.addEventListener(ev, function () { fxDragging[p.id] = true; });
  });
  ["touchend", "mouseup"].forEach(function (ev) {
    f.addEventListener(ev, function () { fxDragging[p.id] = false; });
  });
  row.appendChild(label);
  row.appendChild(f);
  row.appendChild(val);
  return row;
}

function fmtFx(v) {
  if (typeof v !== "number") return "--";
  return (Math.round(v * 100) / 100).toString();
}

// ── tabs ──
function initTabs() {
  var tabs = document.querySelectorAll(".tab");
  tabs.forEach(function (t) {
    t.onclick = function () {
      ACTIVE_TAB = t.getAttribute("data-tab");
      tabs.forEach(function (x) { x.classList.remove("tab-active"); });
      t.classList.add("tab-active");
      document.getElementById("tab-mix").classList.toggle("hidden", ACTIVE_TAB !== "mix");
      document.getElementById("tab-clips").classList.toggle("hidden", ACTIVE_TAB !== "clips");
      document.getElementById("tab-fx").classList.toggle("hidden", ACTIVE_TAB !== "fx");
      if (ACTIVE_TAB === "clips") { initClipLayerPicker(STATE); renderScenes(STATE); loadClips(); }
      if (ACTIVE_TAB === "fx") loadEffects();
    };
  });
}

// ── panic ──
// The button's armed state is the SINGLE SOURCE only at the instant of a
// tap; thereafter it's reconciled from Arena's actual composition master
// each poll (reconcilePanic). So if Arena restarts, or master is moved in
// Arena directly, the button can't lie about whether output is black.
var panicPending = false;   // suppress reconcile flicker mid-request

function setPanicVisual(armed) {
  var btn = document.getElementById("panic");
  if (!btn) return;
  blackedOut = armed;
  btn.classList.toggle("armed", armed);
  btn.innerHTML = armed ? "RESTORE" : "PANIC<br>BLACK";
}

function reconcilePanic(snap) {
  // Don't fight an in-flight panic/restore request.
  if (panicPending) return;
  if (!snap || typeof snap.master !== "number") return;
  // master near 0 => output is black => button should read RESTORE (armed).
  setPanicVisual(snap.master <= 0.001);
}

function initPanic() {
  var btn = document.getElementById("panic");
  btn.onclick = function () {
    var goingDark = !blackedOut;
    panicPending = true;
    setPanicVisual(goingDark);                 // optimistic immediate feedback
    api(goingDark ? "panic" : "restore", {})
      .then(function () { panicPending = false; })
      .catch(function (e) {
        panicPending = false;
        showErr((goingDark ? "panic" : "restore") + ": " + e);
      });
  };
}

// ── poll ──
function poll() {
  api("state", {}).then(function (snap) {
    STATE = snap; clearErr(); paintStatus(snap); renderLayers(snap);
    reconcilePanic(snap);
    if (ACTIVE_TAB === "clips") { initClipLayerPicker(snap); renderScenes(snap); }
  }).catch(function (e) {
    paintStatus({ reachable: false }); showErr("state: " + e);
  });
}
function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  poll();
  pollTimer = setInterval(poll, POLL_MS);
}

document.addEventListener("DOMContentLoaded", function () {
  try { initTabs(); initPanic(); initMaster(); initCrossfader(); initGridDensity(); initFx(); startPolling(); }
  catch (e) { showErr("init: " + e); }
});
