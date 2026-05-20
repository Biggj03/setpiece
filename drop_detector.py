"""
Drop detection for VJ set-craft.

Codifies the rule from "Architectural Paradigms in Live Visual Performance":

    Pros LOCK on one hero visual at the peak; rookies do faster cuts.
    The drop is for *freezing on the icon*, not machine-gunning more flips.

The detector watches the rolling RMS energy ingested by the existing audio
capture loop (audio_reactive.py). When the current frame's energy spikes
past `energy_delta_threshold` times the rolling median over the last
`window_seconds`, it fires `on_drop_detected(intensity)`.

Intentionally tiny / dependency-free:
  - No audio device opened here. We piggyback on the WASAPI loopback that
    audio_reactive.py already runs (one stream is plenty).
  - No SciPy / aubio. Pure stdlib + numpy.
  - Stateless from the caller's perspective: just `ingest_energy(rms)` per
    frame and we'll call back when something dropworthy happens.

Debounce: after a hit we go quiet for `window_seconds * 2`. Drops are by
definition rare events; the hold pattern in main.py needs the suppression
window to actually run out before a NEW drop tries to re-arm hero-hold.
Without the debounce, the giant peak that triggered the drop would keep
re-triggering for as long as it stayed above the rolling median.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover -- audio_reactive needs numpy too
    np = None  # type: ignore


logger = logging.getLogger("drop_detector")


class DropDetector:
    """Spike-vs-median detector over a rolling energy window.

    Designed to be called from the audio capture thread — `ingest_energy`
    is cheap (constant-time deque append + ~one numpy median over a few
    hundred samples).
    """

    def __init__(
        self,
        on_drop_detected: Callable[[float], None],
        window_seconds: float = 2.0,
        energy_delta_threshold: float = 2.5,
        sample_rate_hz: float = 86.0,  # hops-per-sec at 44.1kHz / hop=512
        min_floor_rms: float = 0.005,
    ):
        """
        Args:
            on_drop_detected: Called with the spike intensity (current /
                median ratio) when a drop fires. Runs on the audio capture
                thread — handler MUST marshal to the Qt thread for any UI
                or mpv mutation.
            window_seconds: How much history to keep for the median.
                Default 2s is enough to ride out a typical pre-drop
                build-up (silence + riser) without locking the median to
                a single quiet bar.
            energy_delta_threshold: How many times the rolling median the
                current energy must exceed to count as a drop. Default
                2.5x is "noticeably more energy than the recent average."
                For quiet songs raise the floor instead — see
                `min_floor_rms`.
            sample_rate_hz: Estimated ingest call rate. Used only to size
                the deque. Default 86 Hz matches sr=44100 / hop=512 which
                is what audio_reactive opens. If the actual rate differs
                the window just ends up slightly shorter/longer in
                seconds; not catastrophic.
            min_floor_rms: Median floor for the spike test. Without this,
                a dead-silent section pushes the median to ~0 and ANY
                audio reads as a "drop" (divide-by-tiny). Default 0.005
                is well below normal music levels but above mic noise.
        """
        self.on_drop_detected = on_drop_detected
        self.window_seconds = float(window_seconds)
        self.energy_delta_threshold = float(energy_delta_threshold)
        self.min_floor_rms = float(min_floor_rms)
        # Size the deque for the requested window in seconds. Cap at a
        # sane upper bound so a misconfigured sample_rate can't allocate
        # something huge.
        maxlen = max(8, min(4096, int(round(window_seconds * sample_rate_hz))))
        self._buf: deque[float] = deque(maxlen=maxlen)
        self._last_fire_ts: float = 0.0
        self._enabled: bool = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: bool) -> None:
        """Toggle without losing the rolling buffer (so re-enabling
        doesn't have to wait `window_seconds` to re-prime the median)."""
        self._enabled = bool(on)

    def reset(self) -> None:
        """Drop history. Useful when switching songs / between sets so
        the prior song's quiet tail doesn't make the next song's normal
        intro register as a drop."""
        self._buf.clear()
        self._last_fire_ts = 0.0

    def ingest_energy(self, rms: float) -> None:
        """Feed one audio-frame RMS into the detector.

        Always appends to the rolling buffer (so the median tracks the
        current track), but only evaluates the drop test when enabled
        AND the buffer has enough samples to be meaningful.
        """
        if np is None:
            return
        try:
            rms = float(rms)
        except (TypeError, ValueError):
            return
        if rms != rms or rms < 0:  # NaN / negative guard
            return
        self._buf.append(rms)

        if not self._enabled:
            return

        # Need at least ~1 second of history before testing — otherwise
        # the median is dominated by whatever the loop opened on.
        if len(self._buf) < max(8, self._buf.maxlen // 2):
            return

        now = time.time()
        # Debounce: after a hit, ignore further triggers for 2x the
        # window. A drop holds for a phrase; another drop on top of
        # that is the same drop, not a new one.
        if now - self._last_fire_ts < self.window_seconds * 2.0:
            return

        # numpy.median is fine here — buffer is small (~170 samples by
        # default) so this is microseconds. Floor it so a near-silent
        # section can't push the median to ~0 and turn every breath
        # into a drop.
        median = float(np.median(self._buf))
        if median < self.min_floor_rms:
            median = self.min_floor_rms

        ratio = rms / median
        if ratio >= self.energy_delta_threshold:
            self._last_fire_ts = now
            logger.info(
                "DROP rms=%.4f median=%.4f ratio=%.2fx (thr=%.2fx)",
                rms, median, ratio, self.energy_delta_threshold,
            )
            try:
                self.on_drop_detected(ratio)
            except Exception as e:
                # Same defensive posture as audio_reactive's on_beat:
                # a busted handler must not kill the capture thread.
                logger.warning("on_drop_detected handler raised: %s", e)
