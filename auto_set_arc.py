"""
Auto set-arc detection -- watch BPM + flip-rate over a rolling window
and pick the most appropriate phase (opening / build / peak /
breakdown).

WHY THIS EXISTS
---------------
Set-arc was shipped with MANUAL cycling (MK2 ALL button). That works,
but during an intense set the operator has both hands on transport
controls and can forget to advance the phase. Auto-detect closes the
loop: when BPM crosses 135 or flips suddenly burst, the picker
profile flips to PEAK automatically and the iPad badge updates to
show what the system is reading.

HEURISTIC
---------
Two signals:

  BPM (smoothed, from audio_reactive.current_bpm())
  flip rate per minute (from a deque of timestamps for recent flips)

Plus a derived TREND signal: is BPM rising or falling over the last
30s vs the prior 30s? +1 / 0 / -1.

Phase classification cascade:

  PEAK      - BPM >= 135 OR flip_rate > 6/min (drop hit hard)
  BREAKDOWN - BPM trend < 0 (falling) AND flip_rate < 2/min
                OR BPM <= 100 with no flips for 30s
  BUILD     - BPM trend > 0 (rising) AND BPM in [110, 135]
  OPENING   - low BPM (<= 115) AND stable/rising
  (fallthrough) keep current phase (hysteresis)

The hysteresis matters: a stable mid-tempo run shouldn't ping-pong
between BUILD and OPENING every 10s. We only switch when one
of the explicit rules fires.

CONFIG
------
Cooldown: don't switch phases more than once per ~25s. Prevents
oscillation around threshold values.

USAGE
-----
    from auto_set_arc import AutoSetArc
    asa = AutoSetArc()
    # On every manual or auto flip:
    asa.record_flip(time.time())
    # Periodically (every 5-10s):
    new_phase = asa.detect_phase(
        bpm=audio_reactive.current_bpm(),
        current_phase=app._set_arc_phase,
        now=time.time(),
    )
    if new_phase != app._set_arc_phase:
        app._set_arc_phase = new_phase
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import set_arc_thresholds as _sat


class AutoSetArc:
    """Stateful auto-phase detector. One instance per app."""

    # Window over which we count flips for "flip rate."
    FLIP_WINDOW_S = 60.0
    # BPM history sampling. We record one sample per detect_phase()
    # call so the deque grows at the detect-cadence rate.
    BPM_HISTORY_LEN = 60   # ~10 min if detect runs every 10s
    # Cooldown between phase transitions (sec).
    SWITCH_COOLDOWN_S = 25.0
    # Trend window: compare avg BPM in last N samples vs prior N.
    TREND_WINDOW = 6       # ~60s if detect runs every 10s

    # Phase thresholds — sourced from the shared set_arc_thresholds module
    # so the live detector and the offline analyzer can't drift apart. Kept
    # as class attributes (self.PEAK_BPM ...) so the predict_* methods below
    # read unchanged.
    PEAK_BPM = _sat.PEAK_BPM
    BUILD_LO_BPM = _sat.BUILD_LO_BPM
    BUILD_HI_BPM = _sat.BUILD_HI_BPM
    OPENING_HI_BPM = _sat.OPENING_HI_BPM
    BREAKDOWN_LO_BPM = _sat.BREAKDOWN_LO_BPM

    # Flip-rate thresholds (per minute)
    PEAK_FLIP_RATE = _sat.PEAK_FLIP_RATE
    BREAKDOWN_FLIP_RATE = _sat.BREAKDOWN_FLIP_RATE

    def __init__(self):
        self._flip_times: deque[float] = deque(maxlen=200)
        self._bpm_history: deque[float] = deque(maxlen=self.BPM_HISTORY_LEN)
        self._last_switch_at: float = 0.0
        self._last_detected_phase: Optional[str] = None

    # ── input recorders ───────────────────────────────────────────────
    def record_flip(self, ts: Optional[float] = None) -> None:
        """Called on every flip (manual or auto-flip). Cheap; safe to
        call from any thread (deque append is atomic in CPython)."""
        self._flip_times.append(ts if ts is not None else time.time())

    def record_bpm_sample(self, bpm: float) -> None:
        """Pushed by detect_phase() but exposed so callers can prime
        the history (e.g. during long pauses where detect isn't
        called)."""
        if bpm and bpm > 0:
            self._bpm_history.append(float(bpm))

    # ── derived values ────────────────────────────────────────────────
    def flip_rate_per_min(self, now: Optional[float] = None) -> float:
        """Flips per minute over the last FLIP_WINDOW_S."""
        if now is None:
            now = time.time()
        cutoff = now - self.FLIP_WINDOW_S
        # Pop stale entries from the left of the deque.
        while self._flip_times and self._flip_times[0] < cutoff:
            self._flip_times.popleft()
        return len(self._flip_times) * (60.0 / self.FLIP_WINDOW_S)

    def bpm_trend(self) -> int:
        """+1 rising, -1 falling, 0 stable. Compares last TREND_WINDOW
        samples to the TREND_WINDOW samples before them."""
        n = self.TREND_WINDOW
        if len(self._bpm_history) < 2 * n:
            return 0
        recent = list(self._bpm_history)[-n:]
        prior = list(self._bpm_history)[-2 * n:-n]
        r = sum(recent) / len(recent)
        p = sum(prior) / len(prior)
        if r - p > 2.0:
            return +1
        if r - p < -2.0:
            return -1
        return 0

    def bpm_slope_per_sec(self) -> float:
        """Linear-regression slope of recent BPM samples — units BPM/sec.
        Used by predict_phase_at() to extrapolate forward in time.
        Returns 0.0 when history is too short to fit reliably. Assumes
        samples are roughly uniformly spaced at detect_phase()'s cadence
        (~5-10s)."""
        hist = list(self._bpm_history)
        if len(hist) < 3:
            return 0.0
        # Use the last 12 samples (≈ 2 min at 10s detect cadence) so
        # we fit on recent trajectory, not the whole set.
        window = hist[-12:]
        n = len(window)
        # Sample spacing: best estimate is the configured detect cadence.
        # Caller's detect_phase runs every ~5-10s; pick 5s as conservative
        # (overestimating slope is safer than under).
        dt_s = 5.0
        # Simple least-squares slope: sum((x-x̄)(y-ȳ)) / sum((x-x̄)²)
        xs = [i * dt_s for i in range(n)]
        ys = window
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num = sum((x - x_mean) * (y - y_mean)
                  for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs) or 1.0
        return num / den   # BPM per second

    def predict_phase_at(
        self,
        bpm: float,
        seconds_ahead: float,
        current_phase: str,
        now: Optional[float] = None,
    ) -> tuple[str, float, str, float]:
        """Forward-projected phase: "where will we be in N seconds?"

        Differs from predict_next_phase (which is "what should the
        current phase be if we weren't cooldown-gated"). This one
        EXTRAPOLATES BPM using the recent slope, then runs the same
        classification.

        Returns (predicted_phase, confidence, reason_str, projected_bpm).

        Confidence is shrunk vs predict_next_phase by 30% when looking
        ≥30s ahead because slope predictions degrade fast — useful as
        a heads-up hint, not a hard signal."""
        slope = self.bpm_slope_per_sec()  # BPM/sec
        projected_bpm = max(0.0, bpm + slope * float(seconds_ahead))
        # Re-use the existing predictor on the projected BPM. We pass
        # current_phase so its "hold current" fallback returns
        # something sensible.
        ph, base_conf, reason = self.predict_next_phase(
            projected_bpm, current_phase, now=now,
        )
        # Confidence decay: lookahead is less certain the further
        # forward we look. 30s → 0.7×, 60s → 0.5×, 120s → 0.3×.
        decay = max(0.3, 1.0 - 0.012 * float(seconds_ahead))
        conf = round(base_conf * decay, 2)
        reason_full = (
            f"in {int(seconds_ahead)}s: bpm {projected_bpm:.0f} "
            f"(slope {slope*60:+.1f}/min) → {ph}; {reason}"
        )
        return (ph, conf, reason_full, projected_bpm)

    # ── prediction lookahead ─────────────────────────────────────────
    # "What phase is the auto-detector likely to switch to next, given
    # current BPM trajectory?" -- a softer version of the cascade in
    # detect_phase(). Skips the cooldown gate so it always returns
    # the best guess for "RIGHT NOW if I weren't holding off."
    # Returns (predicted_phase, confidence_0_to_1, reason_str).
    def predict_next_phase(
        self,
        bpm: float,
        current_phase: str,
        now: Optional[float] = None,
    ) -> tuple[str, float, str]:
        if now is None:
            now = time.time()
        flip_rate = self.flip_rate_per_min(now)
        trend = self.bpm_trend()

        # Strong PEAK signals
        if bpm >= self.PEAK_BPM:
            return ("peak", 0.95, f"bpm {bpm:.0f} >= {self.PEAK_BPM}")
        if flip_rate > self.PEAK_FLIP_RATE:
            return ("peak", 0.85,
                    f"flip rate {flip_rate:.1f}/min > {self.PEAK_FLIP_RATE}")
        # Approaching PEAK
        if bpm >= (self.PEAK_BPM - 8) and trend > 0:
            return ("peak", 0.6, f"bpm {bpm:.0f} rising toward peak")

        # Strong BREAKDOWN signals
        if (trend < 0 and flip_rate < self.BREAKDOWN_FLIP_RATE):
            return ("breakdown", 0.8,
                    f"trend down + flip rate {flip_rate:.1f}/min low")
        if (bpm > 0 and bpm <= self.BREAKDOWN_LO_BPM
                and flip_rate < 1.0):
            return ("breakdown", 0.85,
                    f"bpm {bpm:.0f} <= {self.BREAKDOWN_LO_BPM} + low flips")
        # Approaching BREAKDOWN
        if trend < 0 and bpm <= self.BREAKDOWN_LO_BPM + 10:
            return ("breakdown", 0.5,
                    f"bpm {bpm:.0f} falling toward breakdown")

        # BUILD: rising in band
        if trend > 0 and self.BUILD_LO_BPM <= bpm <= self.BUILD_HI_BPM:
            return ("build", 0.7,
                    f"trend up in {self.BUILD_LO_BPM}-{self.BUILD_HI_BPM} band")

        # OPENING: low / soft / stable
        if bpm > 0 and bpm <= self.OPENING_HI_BPM and trend >= 0:
            return ("opening", 0.6, f"bpm {bpm:.0f} low + stable/rising")

        # Nothing specific -> stay
        return (current_phase, 0.3, "no strong signal, holding current")

    # ── main detector ────────────────────────────────────────────────
    def detect_phase(
        self,
        bpm: float,
        current_phase: str,
        now: Optional[float] = None,
    ) -> str:
        """Return the phase the auto-detector wants right now. Caller
        is responsible for actually applying it (and tracking that
        the change happened)."""
        if now is None:
            now = time.time()
        self.record_bpm_sample(bpm)

        flip_rate = self.flip_rate_per_min(now)
        trend = self.bpm_trend()

        # Cooldown: don't switch if we switched too recently. Returning
        # current_phase tells the caller "do nothing." We still record
        # the BPM sample above so trend stays accurate.
        if (now - self._last_switch_at) < self.SWITCH_COOLDOWN_S:
            return current_phase

        proposed = self._classify(bpm, flip_rate, trend, current_phase)

        if proposed != current_phase:
            self._last_switch_at = now
            self._last_detected_phase = proposed
        return proposed

    def _classify(
        self,
        bpm: float,
        flip_rate: float,
        trend: int,
        current: str,
    ) -> str:
        """Stateless cascade — delegates to the shared classifier so the
        live detector and the offline analyzer label phases identically.
        Kept as a method (the cascade reads from self's thresholds via the
        shared module) for back-compat with existing callers."""
        return _sat.classify(bpm, flip_rate, trend, current)

    # ── diagnostics ───────────────────────────────────────────────────
    def snapshot(self, now: Optional[float] = None) -> dict:
        """For HTTP / OLED display / logs."""
        if now is None:
            now = time.time()
        return {
            "flip_rate_per_min": round(self.flip_rate_per_min(now), 2),
            "bpm_trend": self.bpm_trend(),
            "bpm_history_len": len(self._bpm_history),
            "flip_history_len": len(self._flip_times),
            "last_switch_age_s": round(now - self._last_switch_at, 1),
            "last_detected_phase": self._last_detected_phase,
        }
