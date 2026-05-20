"""
Audio-reactive beat detection via WASAPI loopback.
Spectral flux onset detection (no aubio dependency).

Also estimates BPM from a rolling window of recent inter-beat intervals
and reports it through ``on_bpm`` so callers can surface it (iPad UI etc).
"""

import threading
import time
from collections import deque
from typing import Optional, Callable

try:
    import numpy as np
    import pyaudiowpatch as pyaudio
    _AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as e:
    _AVAILABLE = False
    _IMPORT_ERROR = str(e)


# How many recent beats to keep when estimating BPM. 8 is plenty to be
# stable but short enough that a tempo change is reflected within ~4 sec.
_BPM_WINDOW = 8


class AudioReactive:
    """Spectral-flux onset detector that fires callbacks on beat detection."""

    def __init__(
        self,
        on_beat: Optional[Callable[[], None]] = None,
        on_bpm: Optional[Callable[[float], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_drop_detected: Optional[Callable[[float], None]] = None,
        sensitivity: float = 1.8,
        min_interval: float = 0.42,
        hop: int = 512,
        device_substring: str = "",
        drop_window_seconds: float = 2.0,
        drop_energy_delta_threshold: float = 2.5,
    ):
        self.on_beat = on_beat
        self.on_bpm = on_bpm
        self.on_error = on_error
        # Kick flux of the most recent beat. Set right before each
        # on_beat() callback so consumers (the picker's downbeat
        # auto-detector) can read per-beat energy without changing
        # the zero-arg on_beat contract. 0.0 = projected/missed beat.
        self.last_beat_flux = 0.0
        self.sensitivity = sensitivity
        self.min_interval = min_interval
        self.hop = hop
        self.device_substring = device_substring
        # Optional drop detector — only constructed if a callback was
        # provided. Piggybacks on this loop's per-frame RMS so we don't
        # open a second WASAPI stream. Caller controls enable/disable
        # via self._drop_detector.set_enabled() at runtime so a setting
        # toggle takes effect without restarting capture.
        self._drop_detector = None
        if on_drop_detected is not None:
            try:
                from drop_detector import DropDetector
                self._drop_detector = DropDetector(
                    on_drop_detected=on_drop_detected,
                    window_seconds=drop_window_seconds,
                    energy_delta_threshold=drop_energy_delta_threshold,
                )
            except Exception as e:
                # Don't kill audio-reactive just because drop_detector
                # import / construction failed. Log and continue.
                import logging
                logging.getLogger("audio_reactive").warning(
                    "DropDetector unavailable: %s", e,
                )
        # _running is a threading.Event (not a plain bool) so the capture
        # thread sees stop() promptly and start() can't double-spawn.
        # (Audit fix C5.)
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_lock = threading.Lock()
        self._last_beat_time = 0.0
        self._beat_history: deque = deque(maxlen=_BPM_WINDOW)
        self._current_bpm: float = 0.0
        # Set by the capture thread when device discovery fails so the
        # caller can surface a nice error to the iPad.
        self._startup_error: Optional[str] = None
        self._startup_done = threading.Event()

    @property
    def current_bpm(self) -> float:
        return self._current_bpm

    def start(self) -> tuple[bool, str]:
        """Start beat detection. Blocks briefly waiting for device discovery.
        Serialized via _start_lock + a prior-thread join so a rapid
        on/off toggle can't leave two WASAPI capture threads alive."""
        if not _AVAILABLE:
            return False, f"Audio libraries unavailable: {_IMPORT_ERROR}"
        with self._start_lock:
            if self._running.is_set():
                return True, "Already running"
            # Join any previous capture thread still winding down before
            # we spawn a fresh one. (Audit fix C5.)
            if self._thread is not None and self._thread.is_alive():
                self._running.clear()
                self._thread.join(timeout=2.0)

            self._startup_error = None
            self._startup_done.clear()
            self._running.set()
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

        # Give the capture thread up to ~1.5s to either open the stream
        # or fail. That way start() can return a useful error string
        # rather than silently lying that everything is fine.
        opened = self._startup_done.wait(timeout=1.5)
        if not opened:
            return True, "Audio-reactive starting..."
        if self._startup_error:
            self._running.clear()
            return False, self._startup_error
        return True, "Audio-reactive started"

    def stop(self):
        """Stop beat detection."""
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._beat_history.clear()
        self._current_bpm = 0.0

    def _record_beat(self, now: float):
        """Record a beat timestamp. BPM is computed by the comb-filter
        autocorrelation in the capture loop now — this just keeps a
        timestamp deque so external code that reads beat_history still
        works."""
        self._beat_history.append(now)

    def _capture_loop(self):
        """Main audio capture and onset detection loop."""
        try:
            import pyaudiowpatch as pyaudio
            import numpy as np
        except ImportError:
            self._startup_error = "Audio libraries unavailable"
            self._startup_done.set()
            return

        pa = pyaudio.PyAudio()
        stream = None
        try:
            # Find loopback device
            device_idx = None
            if self.device_substring:
                # Prefer loopback devices that match the substring.
                try:
                    for info in pa.get_loopback_device_info_generator():
                        if self.device_substring.lower() in info["name"].lower():
                            device_idx = info["index"]
                            break
                except Exception:
                    pass
                # Fall back: scan all devices for a name match.
                if device_idx is None:
                    for i in range(pa.get_device_count()):
                        info = pa.get_device_info_by_index(i)
                        if self.device_substring.lower() in info["name"].lower():
                            device_idx = i
                            break
                if device_idx is None:
                    self._startup_error = (
                        f"No audio device matching '{self.device_substring}'"
                    )
                    self._startup_done.set()
                    if self.on_error:
                        try: self.on_error(self._startup_error)
                        except Exception: pass
                    return
            else:
                try:
                    device_idx = pa.get_default_wasapi_loopback()["index"]
                except Exception as e:
                    self._startup_error = f"No WASAPI loopback device: {e}"
                    self._startup_done.set()
                    if self.on_error:
                        try: self.on_error(self._startup_error)
                        except Exception: pass
                    return

            info = pa.get_device_info_by_index(device_idx)
            sr = int(info["defaultSampleRate"])
            channels = int(info["maxInputChannels"])
            try:
                import logging
                logging.getLogger("audio_reactive").info(
                    "Capture device: idx=%s name=%r sr=%s ch=%s",
                    device_idx, info.get("name", "?"), sr, channels,
                )
            except Exception:
                pass

            try:
                stream = pa.open(
                    input_device_index=device_idx,
                    format=pyaudio.paFloat32,
                    channels=channels,
                    rate=sr,
                    input=True,
                    frames_per_buffer=self.hop,
                )
            except Exception as e:
                self._startup_error = f"Cannot open audio stream: {e}"
                try:
                    import logging
                    logging.getLogger("audio_reactive").error(
                        "pa.open failed for idx=%s: %s", device_idx, e,
                    )
                except Exception:
                    pass
                self._startup_done.set()
                if self.on_error:
                    try: self.on_error(self._startup_error)
                    except Exception: pass
                return

            # Stream is open, signal start() that we're alive.
            self._startup_done.set()
            try:
                import logging
                logging.getLogger("audio_reactive").info(
                    "Stream OPEN: %r @ %sHz / %sch", info.get("name", "?"), sr, channels,
                )
            except Exception:
                pass

            # ── Kick-isolated, comb-filter, flywheel-gated pipeline ──────
            # Per "Real-Time Beat Detection for VJ" research:
            #  1. Sub-band slice bins 1..3 (~86-258 Hz) = kick band only
            #  2. Half-wave-rectified flux on that band (ignore decay)
            #  3. Autocorrelation of recent flux → tempo period (lag in hops)
            #  4. Phase "flywheel": only accept onsets near the projected
            #     beat phase, project autonomously when no peak is found
            #  5. IQR-filtered tempo over a sliding window → BPM
            import logging
            _ar_log = logging.getLogger("audio_reactive")

            # Sub-band bins for kick. With sr=44100, n_fft=512 → bin width
            # = sr/n_fft = 86.13 Hz. Bins 1..3 inclusive cover ~86-258 Hz.
            # Adapt slice if the audio device opens at a different sr.
            n_fft = self.hop
            bin_hz = sr / n_fft
            kick_lo_bin = max(1, int(round(40.0 / bin_hz)))
            kick_hi_bin = max(kick_lo_bin + 1, int(round(260.0 / bin_hz)))

            # Tempo search range: 70-160 BPM in hops. lag = sr/hop / (bpm/60).
            hops_per_sec = sr / self.hop
            min_lag = max(2, int(round(hops_per_sec * 60.0 / 160.0)))  # fastest
            max_lag = int(round(hops_per_sec * 60.0 / 70.0))            # slowest
            buf_len = max_lag * 4  # ~4 measures of history at slowest tempo
            onset_env = np.zeros(buf_len, dtype=np.float32)
            onset_idx = 0

            prev_kick_mag = 0.0
            current_period = 0.0   # in hops; 0 = unknown
            phase_hop = 0          # rolling hop counter (mod buf_len for env)
            next_beat_hop = -1     # absolute hop counter for next predicted beat
            abs_hop = 0
            interval_window: deque = deque(maxlen=12)  # for IQR
            _level_frames = 0
            cold_start_taps = 0  # bootstrap before tempo lock

            _ar_log.info(
                "Pipeline: sub-band bins[%d:%d] (%.0f-%.0f Hz), lag range %d-%d hops "
                "(%.0f-%.0f BPM), buf=%d hops",
                kick_lo_bin, kick_hi_bin + 1, kick_lo_bin * bin_hz, (kick_hi_bin + 1) * bin_hz,
                min_lag, max_lag, hops_per_sec * 60.0 / max_lag,
                hops_per_sec * 60.0 / min_lag, buf_len,
            )

            while self._running.is_set():
                try:
                    data = stream.read(self.hop, exception_on_overflow=False)
                    audio = np.frombuffer(data, dtype=np.float32)

                    # Mono mix if stereo (interleaved frames)
                    if channels > 1 and len(audio) >= channels:
                        audio = audio.reshape(-1, channels).mean(axis=1)

                    # FFT on hanning-windowed frame
                    fft = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))

                    # Kick-band magnitude (sum over low bins)
                    kick_mag = float(np.sum(fft[kick_lo_bin:kick_hi_bin + 1]))
                    # Half-wave rectified flux: only ENERGY INCREASES count
                    flux = max(0.0, kick_mag - prev_kick_mag)
                    prev_kick_mag = kick_mag

                    # Append to rolling onset envelope
                    onset_env[onset_idx] = flux
                    onset_idx = (onset_idx + 1) % buf_len
                    abs_hop += 1

                    # Per-hop RMS — used by drop detection AND by the
                    # 1Hz level dump below. Cheap (one numpy reduction
                    # over <=512 samples) so always compute.
                    rms = float(np.sqrt(np.mean(audio * audio)))
                    # Feed the optional drop detector. It does its own
                    # rolling-median / debounce / threshold; this call
                    # is fire-and-forget.
                    if self._drop_detector is not None:
                        self._drop_detector.ingest_energy(rms)

                    # Periodic status dump (~1 Hz at sr=44100, hop=512)
                    _level_frames += 1
                    if _level_frames >= int(hops_per_sec):
                        _level_frames = 0
                        _ar_log.info(
                            "level rms=%.4f kick_flux=%.1f period=%.1f bpm=%.1f sens=%.2f",
                            rms, flux, current_period, self._current_bpm, self.sensitivity,
                        )

                    # Refresh tempo period via autocorrelation every ~0.25s
                    if abs_hop % max(1, int(hops_per_sec / 4)) == 0:
                        # Reorder buffer so newest is at the end (trailing)
                        env = np.concatenate((onset_env[onset_idx:], onset_env[:onset_idx]))
                        # Subtract mean to suppress DC bias in autocorrelation
                        env = env - env.mean()
                        if np.any(env > 0):
                            best_score = -1.0
                            best_lag = 0
                            # Vectorized autocorrelation across candidate lags
                            for lag in range(min_lag, max_lag + 1):
                                score = float(np.dot(env[lag:], env[:-lag]))
                                if score > best_score:
                                    best_score = score
                                    best_lag = lag
                            # Parabolic interpolation for sub-hop precision
                            if min_lag < best_lag < max_lag and best_score > 0:
                                ym1 = float(np.dot(env[best_lag - 1:], env[:-(best_lag - 1)]))
                                yp1 = float(np.dot(env[best_lag + 1:], env[:-(best_lag + 1)]))
                                denom = (ym1 - 2 * best_score + yp1)
                                if abs(denom) > 1e-9:
                                    offset = 0.5 * (ym1 - yp1) / denom
                                    if -1.0 < offset < 1.0:
                                        current_period = best_lag + offset
                                    else:
                                        current_period = float(best_lag)
                                else:
                                    current_period = float(best_lag)
                            else:
                                current_period = float(best_lag)
                            # Update BPM from period (with IQR smoothing)
                            if current_period > 0:
                                period_sec = current_period * self.hop / sr
                                inst_bpm = 60.0 / period_sec
                                interval_window.append(inst_bpm)
                                if len(interval_window) >= 4:
                                    arr = np.array(interval_window)
                                    q1, q3 = np.percentile(arr, [25, 75])
                                    iqr = q3 - q1
                                    lo = q1 - 1.5 * iqr
                                    hi = q3 + 1.5 * iqr
                                    inliers = arr[(arr >= lo) & (arr <= hi)]
                                    smoothed = float(inliers.mean()) if len(inliers) else float(arr.mean())
                                else:
                                    smoothed = inst_bpm
                                self._current_bpm = smoothed
                                if self.on_bpm:
                                    try:
                                        self.on_bpm(smoothed)
                                    except Exception:
                                        pass

                    # ── Beat detection: phase-aware flywheel ─────────────
                    # Dynamic threshold: sensitivity-scaled mean of the envelope
                    env_thresh = float(onset_env.mean()) * self.sensitivity
                    is_peak = flux > env_thresh and flux > 0

                    if current_period < min_lag:
                        # Bootstrap: no tempo lock yet. Accept any clear peak,
                        # respect a soft minimum interval to avoid ghost notes.
                        if is_peak:
                            now = time.time()
                            if now - self._last_beat_time >= self.min_interval:
                                self._last_beat_time = now
                                self._record_beat(now)
                                cold_start_taps += 1
                                _ar_log.info("BOOTSTRAP beat #%d flux=%.1f thr=%.1f",
                                             cold_start_taps, flux, env_thresh)
                                if self.on_beat:
                                    try:
                                        self.last_beat_flux = float(flux)
                                        self.on_beat()
                                    except Exception as e:
                                        print(f"on_beat handler error: {e}")
                    else:
                        # Flywheel mode: accept peaks near the projected
                        # phase. Window widened from ±15% to ±25% per
                        # test-suite finding — at 120 BPM the narrow
                        # window quantized to ~5 hops and on-grid kicks
                        # were systematically just outside it (callbacks
                        # firing 2x/10s instead of 20x).
                        if next_beat_hop < 0:
                            next_beat_hop = abs_hop + int(round(current_period))
                        window_half = max(3, int(round(current_period * 0.25)))
                        in_window = abs(abs_hop - next_beat_hop) <= window_half
                        if in_window and is_peak:
                            now = time.time()
                            self._last_beat_time = now
                            self._record_beat(now)
                            _ar_log.info("BEAT (locked) flux=%.1f bpm=%.1f phase_err=%d",
                                         flux, self._current_bpm, abs_hop - next_beat_hop)
                            if self.on_beat:
                                try:
                                    self.last_beat_flux = float(flux)
                                    self.on_beat()
                                except Exception as e:
                                    print(f"on_beat handler error: {e}")
                            next_beat_hop = abs_hop + int(round(current_period))
                        elif abs_hop > next_beat_hop + window_half:
                            # No peak found in window — fire the on_beat
                            # callback ANYWAY (a missed-but-projected beat
                            # is still a beat for VJ visual purposes) and
                            # advance the flywheel.
                            _ar_log.debug("BEAT (projected) bpm=%.1f", self._current_bpm)
                            self._record_beat(time.time())
                            if self.on_beat:
                                try:
                                    # Projected beat: no kick peak found,
                                    # so by definition low energy → 0.0.
                                    self.last_beat_flux = 0.0
                                    self.on_beat()
                                except Exception as e:
                                    # Log at WARNING — a real on_beat
                                    # handler bug otherwise kills the
                                    # auto-flip feedback chain forever
                                    # while looking like "auto-flip just
                                    # isn't firing" to the user. Visible
                                    # is better than silent. (Audit fix
                                    # 2026-05-16.)
                                    _ar_log.warning(
                                        "on_beat handler raised: %s", e)
                            next_beat_hop += int(round(current_period))
                except Exception as e:
                    # Recoverable per-hop errors (transient WASAPI hiccup,
                    # numpy edge case) shouldn't kill beat detection — log
                    # and continue. Only break on repeated failures.
                    # (Audit fix M14.)
                    self._consecutive_errors = getattr(self, "_consecutive_errors", 0) + 1
                    _ar_log.warning(
                        "Capture loop hop error (%d consecutive): %s",
                        self._consecutive_errors, e,
                    )
                    if self._consecutive_errors >= 20:
                        _ar_log.error(
                            "Capture loop: 20 consecutive errors — giving up",
                            exc_info=True,
                        )
                        # Surface to main.py so it can clear the iPad's
                        # "audio reactive ON" state instead of lying.
                        if self.on_error:
                            try:
                                self.on_error("Audio capture died — re-enable to retry")
                            except Exception:
                                pass
                        break
                    # Brief backoff so a tight error loop doesn't spin hot.
                    time.sleep(0.02)
                    continue
                else:
                    # Clean hop — reset the consecutive-error counter.
                    self._consecutive_errors = 0

        finally:
            # Make sure start() is never left waiting.
            if not self._startup_done.is_set():
                if not self._startup_error:
                    self._startup_error = "Audio capture exited before startup"
                self._startup_done.set()
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            pa.terminate()
