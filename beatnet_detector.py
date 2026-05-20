"""
BeatNet-based phase-locked beat detector. Drop-in alternative to
``audio_reactive.AudioReactive``.

Why this exists
---------------
Per the research PDF "Architectural Paradigms in Live Visual Performance",
the existing spectral-flux onset detector is REACTIVE (fires after the
energy spike) while modern VJ rigs use PHASE-LOCKED LOOP beat trackers
that PREDICT the next beat. BeatNet (Heydari & Duan, ISMIR 2021) is the
SOTA open-source option: a CRNN feeds beat/downbeat activations into a
particle filter that maintains a phase estimate of the bar.

This wrapper mirrors ``AudioReactive``'s public interface so the rest of
the app (picker, /cdj.html phase meter, lyric_drive, flip-on-beat logic,
S2 controller) sees identical ``on_beat`` callbacks regardless of which
backend is active.

Compatibility shim
------------------
BeatNet depends on ``madmom`` 0.16.1 which predates Python 3.10 and
NumPy 1.24 — both deprecate things madmom uses. The shim at the top of
this module aliases the removed names BEFORE importing BeatNet/madmom so
the 96 sites that use ``np.float``/``collections.MutableSequence``/etc.
keep working without modifying upstream code.

Audio capture
-------------
BeatNet's built-in ``stream`` mode opens default pyaudio mic input — no
good for VJ work where we want WASAPI loopback off the system mix. We
sidestep this by instantiating BeatNet with ``mode='realtime'`` (which
does NOT open pyaudio) and driving the inner pipeline manually:
    LOG_SPECT.process_audio → CRNN model → particle_filter_cascade.process
…feeding it 20 ms hops sourced from pyaudiowpatch loopback (resampled
44100/48000 → 22050 mono, which is what BeatNet was trained on).

Latency: 50 Hz inference, particle filter accumulates ~5 hops (~100 ms)
of context before emitting beats. First inference call takes 1-2 s
because the CRNN weights load lazily.
"""

import logging
import threading
import time
from collections import deque
from typing import Callable, Optional


# ── madmom / NumPy compat shim — MUST run before BeatNet import ────────
import collections
import collections.abc
for _n in ("MutableSequence", "MutableMapping", "Iterable", "Sequence",
           "Mapping", "Callable"):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))
try:
    import numpy as np
    for _n, _t in (("float", float), ("int", int), ("bool", bool),
                   ("complex", complex), ("object", object), ("str", str),
                   ("long", int)):
        if not hasattr(np, _n):
            setattr(np, _n, _t)
except ImportError:
    np = None  # surfaced as _AVAILABLE=False below
# ───────────────────────────────────────────────────────────────────────


try:
    import pyaudiowpatch as pyaudio
    import torch
    from BeatNet.BeatNet import BeatNet
    _AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as e:  # broad: madmom can raise weird things
    _AVAILABLE = False
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"


_BPM_WINDOW = 8
_log = logging.getLogger("beatnet_detector")


class BeatNetDetector:
    """Phase-locked beat detector. Public surface matches AudioReactive.

    Constructor signature is a superset of AudioReactive so callers can
    pass the same kwargs. Unused kwargs (``sensitivity``, ``hop``) are
    accepted and ignored so a one-line backend swap works.
    """

    def __init__(
        self,
        on_beat: Optional[Callable[[], None]] = None,
        on_bpm: Optional[Callable[[float], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        device_substring: str = "",
        min_interval: float = 0.18,
        model: int = 1,
        # Accepted-but-ignored (kept for AudioReactive signature parity
        # so main.py can pass identical kwargs to either backend):
        sensitivity: float = 1.8,
        hop: int = 512,
        on_drop_detected: Optional[Callable[[float], None]] = None,
        drop_window_seconds: float = 2.0,
        drop_energy_delta_threshold: float = 2.5,
    ):
        self.on_beat = on_beat
        self.on_bpm = on_bpm
        self.on_error = on_error
        self.device_substring = device_substring
        self.min_interval = float(min_interval)
        self.model_id = int(model)
        # Kept so callers / GUI sliders don't crash when poking these:
        self.sensitivity = float(sensitivity)
        self.hop = int(hop)

        # Lifecycle
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_lock = threading.Lock()
        self._startup_error: Optional[str] = None
        self._startup_done = threading.Event()

        # State for external readers (CDJ phase meter etc.)
        self._last_beat_time = 0.0
        self._beat_history: deque = deque(maxlen=_BPM_WINDOW)
        self._current_bpm: float = 0.0
        # Estimated beat period in seconds. Updated whenever we see >=2
        # beats; consumers compute phase as
        #   (time.time() - _last_beat_time) / _period_sec, mod 1
        self._period_sec: float = 0.0

        # BeatNet instance (lazy-built in capture thread so model load
        # doesn't block start()).
        self._bn: Optional["BeatNet"] = None

    # ── AudioReactive parity properties ────────────────────────────────

    @property
    def current_bpm(self) -> float:
        return self._current_bpm

    def is_running(self) -> bool:
        return self._running.is_set()

    def get_bpm(self) -> float:
        return self._current_bpm

    def last_beat_age_ms(self) -> int:
        if self._last_beat_time <= 0:
            return -1
        return int((time.time() - self._last_beat_time) * 1000)

    def get_phase(self) -> float:
        """0..1 position within the current beat. Used by /cdj.html
        phase meter. Returns 0.0 if no tempo lock yet."""
        if self._period_sec <= 0 or self._last_beat_time <= 0:
            return 0.0
        elapsed = time.time() - self._last_beat_time
        return float((elapsed / self._period_sec) % 1.0)

    # ── start/stop ─────────────────────────────────────────────────────

    def start(self) -> tuple[bool, str]:
        if not _AVAILABLE:
            return False, f"BeatNet unavailable: {_IMPORT_ERROR}"
        with self._start_lock:
            if self._running.is_set():
                return True, "Already running"
            if self._thread is not None and self._thread.is_alive():
                self._running.clear()
                self._thread.join(timeout=2.0)

            self._startup_error = None
            self._startup_done.clear()
            self._running.set()
            self._thread = threading.Thread(
                target=self._capture_loop, daemon=True,
                name="BeatNetCapture",
            )
            self._thread.start()

        # Wait briefly for device discovery (but not for full model
        # load — that's ~1-2s and shouldn't block the UI thread.
        # AudioReactive waits 1.5s; we match.)
        opened = self._startup_done.wait(timeout=1.5)
        if not opened:
            return True, "BeatNet starting (model loading)..."
        if self._startup_error:
            self._running.clear()
            return False, self._startup_error
        return True, "BeatNet started"

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._beat_history.clear()
        self._current_bpm = 0.0
        self._period_sec = 0.0

    # ── internals ──────────────────────────────────────────────────────

    def _record_beat(self, t: float):
        self._beat_history.append(t)
        # Recompute BPM from inter-beat intervals (median is robust
        # against the occasional missed/extra beat from the particle
        # filter near tempo changes).
        if len(self._beat_history) >= 3:
            ivals = [self._beat_history[i] - self._beat_history[i - 1]
                     for i in range(1, len(self._beat_history))]
            ivals = [iv for iv in ivals if 0.2 <= iv <= 1.5]  # 40-300 BPM
            if ivals:
                med = sorted(ivals)[len(ivals) // 2]
                if med > 0:
                    self._period_sec = med
                    bpm = 60.0 / med
                    self._current_bpm = bpm
                    if self.on_bpm:
                        try:
                            self.on_bpm(bpm)
                        except Exception:
                            pass

    def _find_device(self, pa) -> tuple[Optional[int], Optional[str]]:
        """Locate a WASAPI loopback input matching device_substring.
        Returns (device_index, error_string). One of them is None."""
        if self.device_substring:
            try:
                for info in pa.get_loopback_device_info_generator():
                    if self.device_substring.lower() in info["name"].lower():
                        return int(info["index"]), None
            except Exception:
                pass
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if self.device_substring.lower() in info["name"].lower():
                    return int(info["index"]), None
            return None, (f"No audio device matching "
                          f"'{self.device_substring}'")
        try:
            return int(pa.get_default_wasapi_loopback()["index"]), None
        except Exception as e:
            return None, f"No WASAPI loopback device: {e}"

    def _capture_loop(self):
        """Pulls loopback audio, resamples to 22050 mono, feeds BeatNet
        pipeline frame-by-frame, fires on_beat when the particle filter
        emits new beats. Runs until self._running clears."""
        if not _AVAILABLE:
            self._startup_error = f"BeatNet unavailable: {_IMPORT_ERROR}"
            self._startup_done.set()
            return

        pa = pyaudio.PyAudio()
        stream = None
        try:
            device_idx, err = self._find_device(pa)
            if err:
                self._startup_error = err
                self._startup_done.set()
                if self.on_error:
                    try: self.on_error(err)
                    except Exception: pass
                return

            info = pa.get_device_info_by_index(device_idx)
            sr = int(info["defaultSampleRate"])
            channels = int(info["maxInputChannels"])
            _log.info("BeatNet capture device: idx=%s name=%r sr=%s ch=%s",
                      device_idx, info.get("name", "?"), sr, channels)

            # BeatNet works at 22050 Hz mono, hop = 441 samples (20 ms).
            # Pull enough loopback samples per iteration to produce one
            # 22050-Hz hop after resample. ratio = sr / 22050.
            BN_SR = 22050
            BN_HOP = 441  # 20 ms @ 22050
            ratio = sr / BN_SR
            # Capture chunk size = ceil(BN_HOP * ratio) so each iteration
            # yields >=1 BN hop. Round up generously to avoid jitter.
            cap_chunk = int(round(BN_HOP * ratio))

            try:
                stream = pa.open(
                    input_device_index=device_idx,
                    format=pyaudio.paFloat32,
                    channels=channels,
                    rate=sr,
                    input=True,
                    frames_per_buffer=cap_chunk,
                )
            except Exception as e:
                self._startup_error = f"Cannot open audio stream: {e}"
                _log.error("pa.open failed for idx=%s: %s", device_idx, e)
                self._startup_done.set()
                if self.on_error:
                    try: self.on_error(self._startup_error)
                    except Exception: pass
                return

            # Stream is open — release start() so the GUI doesn't block
            # on the (potentially slow) BeatNet model load.
            self._startup_done.set()
            _log.info("BeatNet stream OPEN: %r @ %sHz / %sch (cap_chunk=%d, "
                      "ratio=%.4f)", info.get("name","?"), sr, channels,
                      cap_chunk, ratio)

            # ── Build BeatNet ──────────────────────────────────────────
            # mode='realtime' so __init__ does NOT open a mic stream;
            # we drive the inner pipeline manually. inference_model='PF'
            # gives us the causal particle filter (the whole point).
            t0 = time.time()
            try:
                self._bn = BeatNet(model=self.model_id, mode='realtime',
                                   inference_model='PF', plot=[],
                                   thread=False)
            except Exception as e:
                _log.exception("BeatNet construct failed")
                if self.on_error:
                    try: self.on_error(f"BeatNet init failed: {e}")
                    except Exception: pass
                return
            _log.info("BeatNet model loaded in %.2fs", time.time() - t0)

            # The pipeline expects a sliding window of
            # (win_length + 2*hop) samples and uses [-2:0] hops as
            # the "current" frame after the first 2 prime hops.
            win = self._bn.log_spec_win_length     # 1410 @ 22050
            hop = self._bn.log_spec_hop_length     # 441  @ 22050
            window_samples = win + 2 * hop          # 2292
            audio_window = np.zeros(window_samples, dtype=np.float32)
            counter = 0
            # Particle filter accumulates beats in .path; we diff
            # against last-seen length to detect newly-emitted beats.
            prev_path_len = 1  # pf.path starts as [[0,0]]
            t_start = time.time()
            consec_errors = 0
            log_every = int(BN_SR / hop)  # ~once per second
            log_counter = 0

            # Resample buffer: stitch leftover samples across iterations
            # so we never drop a partial BN_HOP at chunk boundaries.
            resamp_leftover = np.zeros(0, dtype=np.float32)

            while self._running.is_set():
                try:
                    raw = stream.read(cap_chunk,
                                      exception_on_overflow=False)
                    audio = np.frombuffer(raw, dtype=np.float32)
                    if channels > 1 and len(audio) >= channels:
                        audio = audio.reshape(-1, channels).mean(axis=1)

                    # Cheap linear resample sr → 22050. The CRNN was
                    # trained on librosa-loaded 22050 audio; linear is
                    # imperfect but at 44100→22050 it's literally a
                    # 2-tap decimation and BeatNet handles that fine in
                    # practice. For 48000→22050 librosa would be better,
                    # but invoking librosa.resample per chunk adds 1-2 ms
                    # we don't need for VJ-grade beat sync.
                    if abs(sr - BN_SR) < 1:
                        resampled = audio
                    else:
                        # Linear interp from sr-spaced samples to
                        # BN_SR-spaced samples. Carry leftover so chunk
                        # boundaries don't introduce phase glitches.
                        if len(resamp_leftover):
                            audio = np.concatenate(
                                (resamp_leftover, audio))
                        n_out = int(len(audio) * BN_SR / sr)
                        if n_out < 1:
                            resamp_leftover = audio
                            continue
                        in_idx = (np.arange(n_out) * sr / BN_SR
                                  ).astype(np.int64)
                        in_idx = np.clip(in_idx, 0, len(audio) - 1)
                        resampled = audio[in_idx].astype(np.float32)
                        # Stash tail so next iteration's resample
                        # starts cleanly. Keep ~1 ms of context.
                        keep = int(round(sr * 0.001))
                        resamp_leftover = audio[-keep:] if keep else \
                            np.zeros(0, dtype=np.float32)

                    # Slide BN hops of resampled audio through the
                    # rolling window and run inference per hop.
                    pos = 0
                    while pos + hop <= len(resampled):
                        chunk = resampled[pos:pos + hop]
                        pos += hop
                        audio_window = np.concatenate(
                            (audio_window[hop:], chunk))
                        counter += 1

                        # Same priming as activation_extractor_realtime:
                        # first 2 hops produce a zero pred.
                        if counter < 2:
                            pred = np.zeros([1, 2])
                        else:
                            with torch.no_grad():
                                feats = self._bn.proc.process_audio(
                                    audio_window).T[-1]
                                feats_t = torch.from_numpy(feats)
                                feats_t = (feats_t.unsqueeze(0)
                                                   .unsqueeze(0)
                                                   .to(self._bn.device))
                                out = self._bn.model(feats_t)[0]
                                out = self._bn.model.final_pred(out)
                                out = out.cpu().detach().numpy()
                                pred = np.transpose(out[:2, :])

                        # Particle filter step.
                        try:
                            self._bn.estimator.process(pred)
                        except Exception as e:
                            _log.debug("estimator.process glitch: %s", e)
                            continue

                        # New beat(s) emitted? Particle filter appends
                        # to .path: rows are [time_seconds, beat_in_bar]
                        path = self._bn.estimator.path
                        if len(path) > prev_path_len:
                            new_rows = path[prev_path_len:]
                            prev_path_len = len(path)
                            now = time.time()
                            for row in new_rows:
                                # Debounce vs last fired beat
                                if (now - self._last_beat_time
                                        < self.min_interval):
                                    continue
                                self._last_beat_time = now
                                self._record_beat(now)
                                _log.info("BEAT bpm=%.1f bar=%s "
                                          "elapsed=%.1fs",
                                          self._current_bpm,
                                          int(row[1]) if len(row) > 1 else 0,
                                          now - t_start)
                                if self.on_beat:
                                    try:
                                        self.on_beat()
                                    except Exception as e:
                                        _log.warning(
                                            "on_beat handler raised: %s", e)

                    log_counter += 1
                    if log_counter >= log_every:
                        log_counter = 0
                        _log.info("running: counter=%d path_len=%d bpm=%.1f "
                                  "phase=%.2f",
                                  counter, prev_path_len,
                                  self._current_bpm, self.get_phase())

                    consec_errors = 0

                except Exception as e:
                    consec_errors += 1
                    _log.warning("capture loop error (%d consecutive): %s",
                                 consec_errors, e)
                    if consec_errors >= 20:
                        _log.error("capture loop: giving up after 20 errors",
                                   exc_info=True)
                        if self.on_error:
                            try:
                                self.on_error(
                                    "BeatNet capture died — re-enable "
                                    "to retry")
                            except Exception:
                                pass
                        break
                    time.sleep(0.02)

        finally:
            if not self._startup_done.is_set():
                if not self._startup_error:
                    self._startup_error = (
                        "BeatNet capture exited before startup")
                self._startup_done.set()
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            try:
                pa.terminate()
            except Exception:
                pass
            self._bn = None
