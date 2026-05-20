/* =======================================================================
   Setpiece "Now Playing" overlay — OBS browser source logic

   - Polls GET /api/state every 500ms.
   - Cleans the clip basename for display.
   - Pulses the BPM number on each detected beat (last_beat_age_ms).
   - Fades the whole card out if the server goes away (no stale data).

   Vanilla JS, no build step. Wrapped defensively because OBS's embedded
   browser (CEF) swallows uncaught errors quietly.
   ======================================================================= */
(function () {
  "use strict";

  // ---- tunables -------------------------------------------------------
  var POLL_MS = 500;          // how often we hit /api/state
  var BEAT_LIT_MS = 140;      // a beat younger than this lights the BPM number
  var DISCONNECT_AFTER_MS = 1500; // no successful poll for this long -> fade out

  // ---- element handles (hoisted; grabbed once on load) ----------------
  var elOverlay, elCard, elClipName, elProgressFill, elTimePos, elTimeDur,
      elBpmNumber, elBpmBlock, elOnair;

  var lastGoodAt = 0;         // timestamp of last successful /api/state
  var litUntil = 0;           // BPM stays "lit" until this timestamp
  var connectedOnce = false;

  // ---- helpers --------------------------------------------------------

  // "C:\clips\my-cool_clip.mp4" -> "my cool clip"
  function cleanClipName(path) {
    if (!path) return "—"; // em dash placeholder
    // basename — handle both / and \ separators (Windows + Linux paths).
    var base = String(path).split(/[\\/]/).pop() || "";
    // strip extension
    base = base.replace(/\.[^.]+$/, "");
    // dashes / underscores -> spaces, collapse runs of whitespace
    base = base.replace(/[-_]+/g, " ").replace(/\s+/g, " ").trim();
    return base || "—";
  }

  // seconds -> "M:SS" (or "H:MM:SS" for long clips)
  function fmtTime(secs) {
    if (typeof secs !== "number" || !isFinite(secs) || secs < 0) secs = 0;
    secs = Math.floor(secs);
    var h = Math.floor(secs / 3600);
    var m = Math.floor((secs % 3600) / 60);
    var s = secs % 60;
    if (h > 0) {
      return h + ":" + String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
    }
    return m + ":" + String(s).padStart(2, "0");
  }

  // ---- render ---------------------------------------------------------

  function render(state) {
    var playback = (state && state.playback) || {};
    var audio = (state && state.audio_reactive) || {};

    // --- clip name ---
    elClipName.textContent = cleanClipName(playback.file);

    // --- progress bar + time ---
    var pos = Number(playback.position) || 0;
    var dur = Number(playback.duration) || 0;
    var pct = dur > 0 ? Math.max(0, Math.min(100, (pos / dur) * 100)) : 0;
    elProgressFill.style.width = pct.toFixed(2) + "%";
    elTimePos.textContent = fmtTime(pos);
    elTimeDur.textContent = fmtTime(dur);

    // --- ON AIR / idle state ---
    // Playing -> live red pulse. Stopped -> neutral dim pill.
    if (playback.is_playing) {
      elOverlay.classList.remove("idle");
    } else {
      elOverlay.classList.add("idle");
    }

    // --- BPM ---
    var bpm = Number(audio.bpm) || 0;
    var arEnabled = !!audio.enabled;
    var beatAge = audio.last_beat_age_ms; // may be null

    if (arEnabled && bpm > 0) {
      elBpmBlock.classList.remove("inactive");
      elBpmNumber.textContent = String(Math.round(bpm));
    } else {
      // audio-reactive off or no BPM locked yet
      elBpmBlock.classList.add("inactive");
      elBpmNumber.textContent = arEnabled ? "--" : "--";
    }

    // Beat pulse: when the last beat is fresh, latch "lit" for a short
    // window. We latch (rather than read instantaneously) so the pulse
    // is visible even though we only poll every 500ms.
    if (arEnabled && typeof beatAge === "number" && beatAge >= 0 &&
        beatAge < BEAT_LIT_MS) {
      litUntil = Date.now() + BEAT_LIT_MS;
    }
  }

  // Runs on its own rAF loop so the beat-pulse decays smoothly between
  // polls instead of stepping at 500ms granularity.
  function animationTick() {
    var now = Date.now();

    // BPM lit state
    if (now < litUntil) {
      elBpmNumber.classList.add("lit");
    } else {
      elBpmNumber.classList.remove("lit");
    }

    // Disconnect handling: fade the card out if we haven't heard back.
    if (connectedOnce && (now - lastGoodAt) > DISCONNECT_AFTER_MS) {
      elOverlay.classList.add("disconnected");
    }

    requestAnimationFrame(animationTick);
  }

  // ---- polling --------------------------------------------------------

  function poll() {
    // cache-bust the request itself so CEF never serves a stale snapshot
    fetch("/api/state?_=" + Date.now(), { cache: "no-store" })
      .then(function (resp) {
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        return resp.json();
      })
      .then(function (state) {
        lastGoodAt = Date.now();
        connectedOnce = true;
        elOverlay.classList.remove("disconnected");
        render(state);
      })
      .catch(function (err) {
        // Server down / unreachable. Don't blow up — animationTick()
        // will fade the overlay out once DISCONNECT_AFTER_MS elapses.
        // (No console spam: OBS's log fills fast.)
      });
  }

  // ---- boot -----------------------------------------------------------

  function boot() {
    elOverlay      = document.getElementById("overlay");
    elCard         = document.getElementById("card");
    elClipName     = document.getElementById("clipName");
    elProgressFill = document.getElementById("progressFill");
    elTimePos      = document.getElementById("timePos");
    elTimeDur      = document.getElementById("timeDur");
    elBpmNumber    = document.getElementById("bpmNumber");
    elBpmBlock     = document.getElementById("bpmBlock");
    elOnair        = document.getElementById("onair");

    // Start hidden-ish until the first good poll lands (avoid flashing
    // placeholder dashes on a live stream).
    elOverlay.classList.add("disconnected");

    poll();
    setInterval(poll, POLL_MS);
    requestAnimationFrame(animationTick);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
