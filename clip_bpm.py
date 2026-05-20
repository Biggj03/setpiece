"""
Per-clip BPM analysis.

Pipeline:
    1. ffmpeg extracts [in_sec, out_sec] of the clip's source as mono
       22050 Hz s16le WAV to a temp file (input-level fast seek for speed).
    2. numpy spectral-flux onset detection finds rough beat candidates
       (same algorithm class as audio_reactive.py, just batch-mode).
    3. Median of inter-onset intervals → BPM = 60 / median.
    4. Sanity-clamp: BPM outside [60, 200] returns None (likely speech /
       silent clip / detection failure).

Caching:
    Each (filepath, in_sec, out_sec) triple gets a sha256 key. The cache
    lives at ~/.setpiece/bpm_cache.json. Re-analysing the same
    clip after restart is free (read-only lookup).

Concurrency:
    A module-level Semaphore caps concurrent analyses at 1 — even with
    90 clips queued, we never spawn 90 ffmpeg processes. Backfill walks
    the list serially.

Defensive:
    Missing ffmpeg/numpy → analyze_clip_bpm() returns None and the
    backfill is a no-op. The caller (clips_db) treats bpm=0 as "unknown"
    so the iPad just doesn't show a badge.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# numpy is the only hard requirement for the math. ffmpeg is checked at
# call time. Both missing → analyse_* returns None, callers carry on.
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False

CACHE_PATH = Path.home() / ".setpiece" / "bpm_cache.json"

# Audio extraction params. 22050Hz mono is plenty for tempo detection
# (Nyquist = 11025Hz, drum transients live well below that) and ~half
# the bytes of 44100Hz, so ffmpeg + numpy both run faster.
SAMPLE_RATE = 22050
HOP = 512                   # ~23ms hop @ 22050Hz — same magnitude as audio_reactive
FRAME = 1024                # FFT window; bigger than hop = overlap

BPM_MIN = 60.0
BPM_MAX = 200.0

# Analysis window. Autocorrelation needs SEVERAL beats of signal to lock —
# a raw 2s clip is hopeless. Tempo is stable over a short span, so we
# extract a wider window starting at the clip's in-point; the answer is
# the clip's tempo, just measured reliably. ffmpeg's -t clamps to the
# source end if there isn't this much left.
MIN_ANALYSIS_SEC = 12.0

# Kick-band. Tempo lives in the kick drum (~40-260 Hz). Slicing the FFT
# to this band BEFORE computing flux rejects hats/snares/vocals that
# swamp a wideband detector — the old detector fired ~6 onsets/sec on
# transients and the median-interval math collapsed to nothing. This is
# the same kick-band + autocorrelation approach audio_reactive.py uses.
KICK_LO_HZ = 40.0
KICK_HI_HZ = 260.0
# Tempo search range as BPM — kept realistic (most music-video content is
# 70-180) to cut autocorrelation octave confusion. Results outside
# [BPM_MIN, BPM_MAX] after octave-folding are rejected.
SEARCH_BPM_MIN = 70.0
SEARCH_BPM_MAX = 180.0

# Subprocess hygiene — don't pop a console window on Windows; portable to Linux.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Cap concurrent analyses across the whole process. 1 is the right answer:
# ffmpeg + numpy are both happy to saturate a core, and the user's
# experience with 90 clips queued is "background hum", not "machine
# unresponsive".
_ANALYSIS_SEMAPHORE = threading.BoundedSemaphore(value=1)

# ffmpeg path is looked up once and cached, same pattern as thumbnails.py.
_ffmpeg_path: Optional[str] = None
_ffmpeg_checked = False
_ffmpeg_lock = threading.Lock()

# Cache I/O lock — small JSON file, but multiple worker threads can race.
_cache_lock = threading.Lock()


def _resolve_ffmpeg() -> Optional[str]:
    """Locate ffmpeg on PATH. Cached. Returns None and warns once if missing."""
    global _ffmpeg_path, _ffmpeg_checked
    with _ffmpeg_lock:
        if _ffmpeg_checked:
            return _ffmpeg_path
        _ffmpeg_checked = True
        _ffmpeg_path = shutil.which("ffmpeg")
        if not _ffmpeg_path:
            logger.warning(
                "ffmpeg not found on PATH; clip BPM analysis disabled. "
                "Install ffmpeg to enable per-clip tempo detection."
            )
        return _ffmpeg_path


# ── Cache (sha256(filepath+in+out) → bpm) ─────────────────────────────


def _cache_key(filepath: str, in_sec: float, out_sec: float) -> str:
    """Stable key for a (file, in, out) triple. Resolve the path so that
    `./vid.mp4` and `/abs/vid.mp4` collide on the same key."""
    raw = f"{Path(filepath).resolve()}|{float(in_sec):.3f}|{float(out_sec):.3f}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"BPM cache read failed: {e}")
        return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug(f"BPM cache write failed: {e}")


def _cache_get(key: str) -> Optional[float]:
    with _cache_lock:
        cache = _load_cache()
        v = cache.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None


def _cache_put(key: str, bpm: float) -> None:
    with _cache_lock:
        cache = _load_cache()
        cache[key] = float(bpm)
        _save_cache(cache)


def _cache_pop(key: str) -> None:
    with _cache_lock:
        cache = _load_cache()
        if key in cache:
            cache.pop(key, None)
            _save_cache(cache)


# ── ffmpeg → WAV bytes ────────────────────────────────────────────────


def _extract_wav_to_tempfile(
    filepath: str, in_sec: float, out_sec: float
) -> Optional[str]:
    """ffmpeg-extract [in, out] of `filepath` as mono 22050Hz s16le WAV.
    Returns the temp path (caller must unlink) or None on failure."""
    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        return None
    src = Path(filepath)
    if not src.exists():
        logger.debug(f"BPM: source missing {filepath}")
        return None
    in_sec = max(0.0, float(in_sec))
    out_sec = float(out_sec)
    # Analyse a wider window than the raw clip (see MIN_ANALYSIS_SEC) —
    # autocorrelation needs several beats and a 2s clip can't give them.
    duration = max(MIN_ANALYSIS_SEC, out_sec - in_sec)

    # NamedTemporaryFile + delete=False so subprocess can write to the
    # path on Windows (which holds the original handle exclusive otherwise).
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="setpiece_bpm_")
    os.close(fd)

    cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error", "-y",
        # Per CLAUDE.md: -ss BEFORE -i = input-level fast seek (key
        # frame snap, dramatically faster than output-level seek).
        "-ss", f"{in_sec:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-vn",                   # drop video
        "-ac", "1",              # mono
        "-ar", str(SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        "-f", "wav",
        tmp,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=20,         # 20s is generous for a 30s clip
            creationflags=_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"BPM ffmpeg timeout for {src.name}")
        try: os.unlink(tmp)
        except Exception: pass
        return None
    except Exception as e:
        logger.warning(f"BPM ffmpeg subprocess failed for {src.name}: {e}")
        try: os.unlink(tmp)
        except Exception: pass
        return None

    if result.returncode != 0 or not Path(tmp).exists() or Path(tmp).stat().st_size == 0:
        err = (result.stderr or b"").decode("utf-8", errors="replace").strip()
        logger.debug(f"BPM ffmpeg rc={result.returncode} for {src.name}: {err}")
        try: os.unlink(tmp)
        except Exception: pass
        return None
    return tmp


def _read_wav_samples(wav_path: str):
    """Read a 16-bit mono WAV into a float32 numpy array in [-1, 1]."""
    if not _NUMPY_AVAILABLE:
        return None
    import wave
    try:
        with wave.open(wav_path, "rb") as wf:
            n = wf.getnframes()
            raw = wf.readframes(n)
    except Exception as e:
        logger.debug(f"BPM wav read failed: {e}")
        return None
    if not raw:
        return None
    # int16 → float32 in [-1, 1]
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


# ── Onset detection + BPM ─────────────────────────────────────────────


def _onset_envelope(samples):
    """Kick-band, half-wave-rectified spectral-flux envelope — one value
    per HOP. This is the continuous signal autocorrelation runs on.

    Why kick-band + half-wave-rectified (vs the old wideband L2 flux):
    the old detector thresholded wideband flux into discrete onsets and
    fired ~6/sec on hats/transients — the median-interval math then
    collapsed because every interval was below the 0.2s floor. Slicing
    to the kick band (~40-260 Hz) and keeping only ENERGY INCREASES
    leaves a clean pulse train that autocorrelates cleanly. Same idea as
    audio_reactive.py's rewrite.
    """
    if samples is None or not _NUMPY_AVAILABLE:
        return None
    n = len(samples)
    if n < FRAME * 4:
        return None
    window = np.hanning(FRAME).astype(np.float32)
    bin_hz = SAMPLE_RATE / FRAME
    lo_bin = max(1, int(round(KICK_LO_HZ / bin_hz)))
    hi_bin = max(lo_bin + 1, int(round(KICK_HI_HZ / bin_hz)))
    n_hops = (n - FRAME) // HOP
    if n_hops < 8:
        return None
    env = np.zeros(n_hops, dtype=np.float32)
    prev_kick = 0.0
    for i in range(n_hops):
        start = i * HOP
        chunk = samples[start:start + FRAME] * window
        mag = np.abs(np.fft.rfft(chunk))
        kick = float(np.sum(mag[lo_bin:hi_bin + 1]))
        # Half-wave rectified flux: only energy INCREASES (kick onsets)
        # count — the decay tail is ignored.
        env[i] = max(0.0, kick - prev_kick)
        prev_kick = kick
    return env


def _bpm_from_envelope(env) -> Optional[float]:
    """Autocorrelate the onset envelope, find the dominant period in the
    SEARCH_BPM range, parabolic-interpolate for sub-hop precision,
    octave-fold into [BPM_MIN, BPM_MAX]. Returns BPM or None."""
    if env is None or not _NUMPY_AVAILABLE or len(env) < 8:
        return None
    env = env - env.mean()
    if not np.any(env > 0):
        return None
    hops_per_sec = SAMPLE_RATE / HOP
    # Candidate autocorrelation lags (in hops) for the search BPM range.
    min_lag = max(2, int(round(hops_per_sec * 60.0 / SEARCH_BPM_MAX)))
    max_lag = int(round(hops_per_sec * 60.0 / SEARCH_BPM_MIN))
    max_lag = min(max_lag, len(env) - 2)
    if max_lag <= min_lag:
        return None
    best_score = -1.0
    best_lag = 0
    for lag in range(min_lag, max_lag + 1):
        score = float(np.dot(env[lag:], env[:-lag]))
        if score > best_score:
            best_score = score
            best_lag = lag
    if best_lag <= 0 or best_score <= 0:
        return None
    # Parabolic interpolation around the peak for sub-hop precision.
    period = float(best_lag)
    if min_lag < best_lag < max_lag:
        ym1 = float(np.dot(env[best_lag - 1:], env[:-(best_lag - 1)]))
        yp1 = float(np.dot(env[best_lag + 1:], env[:-(best_lag + 1)]))
        denom = ym1 - 2.0 * best_score + yp1
        if abs(denom) > 1e-9:
            offset = 0.5 * (ym1 - yp1) / denom
            if -1.0 < offset < 1.0:
                period = best_lag + offset
    period_sec = period * HOP / SAMPLE_RATE
    if period_sec <= 0:
        return None
    bpm = 60.0 / period_sec
    # Octave-fold any harmonic the autocorrelation locked onto.
    for _ in range(4):
        if bpm > BPM_MAX:
            bpm /= 2.0
        elif bpm < BPM_MIN:
            bpm *= 2.0
        else:
            break
    if bpm < BPM_MIN or bpm > BPM_MAX:
        return None
    return float(bpm)


# ── Public API ────────────────────────────────────────────────────────


def analyze_clip_bpm(
    filepath: str, in_sec: float, out_sec: float
) -> Optional[float]:
    """Returns BPM (60..200) or None on any failure / out-of-range result.

    Cached on disk by sha256(filepath + in + out) — re-runs are free.
    """
    if not filepath or out_sec <= in_sec:
        return None
    if not _NUMPY_AVAILABLE:
        return None

    key = _cache_key(filepath, in_sec, out_sec)
    cached = _cache_get(key)
    if cached is not None:
        if BPM_MIN <= cached <= BPM_MAX:
            return cached
        # Stale "None-equivalent" entry (e.g. analysis previously failed).
        # Treat as miss and re-run.

    # Acquire the global semaphore so backfill / single requests / iPad
    # reanalyze taps don't all spawn ffmpeg at once.
    with _ANALYSIS_SEMAPHORE:
        wav_path = _extract_wav_to_tempfile(filepath, in_sec, out_sec)
        if not wav_path:
            return None
        try:
            samples = _read_wav_samples(wav_path)
            env = _onset_envelope(samples)
            bpm = _bpm_from_envelope(env)
        finally:
            try: os.unlink(wav_path)
            except Exception: pass

    if bpm is None:
        logger.debug(f"BPM detection failed for {Path(filepath).name} "
                     f"[{in_sec:.1f}-{out_sec:.1f}] (onsets too sparse / out of range)")
        return None
    _cache_put(key, bpm)
    return bpm


def clear_cache_for_clip(clip: dict) -> None:
    """Drop the cache entry for a clip so the next analyze runs from scratch.
    Used by the iPad's reanalyze button."""
    fp = clip.get("filepath")
    if not fp:
        return
    in_s = float(clip.get("in_seconds") or 0.0)
    out_s = float(clip.get("out_seconds") or 0.0)
    _cache_pop(_cache_key(fp, in_s, out_s))


def analyze_async(
    clip_dict: dict,
    on_done: Callable[[Optional[float]], None],
) -> threading.Thread:
    """Run analysis in a daemon thread; call on_done(bpm_or_none) when done.

    The on_done callback runs on the worker thread — keep it cheap or
    re-marshal to the Qt main thread yourself if it touches widgets.
    """
    fp = clip_dict.get("filepath") or ""
    in_s = float(clip_dict.get("in_seconds") or 0.0)
    out_s = float(clip_dict.get("out_seconds") or 0.0)

    def worker():
        bpm = None
        try:
            bpm = analyze_clip_bpm(fp, in_s, out_s)
        except Exception as e:
            logger.debug(f"analyze_async crashed for {clip_dict.get('id')}: {e}")
        try:
            on_done(bpm)
        except Exception as e:
            logger.debug(f"analyze_async on_done raised: {e}")

    t = threading.Thread(target=worker, name="bpm-analyze", daemon=True)
    t.start()
    return t


def backfill_async(
    clips: list[dict],
    on_bpm: Callable[[str, float], None],
) -> Optional[threading.Thread]:
    """Background analysis of every clip whose `bpm` is 0 or missing.

    `on_bpm(clip_id, bpm)` is invoked once per successfully-analyzed clip.
    Runs serially (the semaphore enforces this anyway) so a 90-clip
    library produces 90 sequential ffmpeg processes, not 90 parallel.

    Returns the worker thread, or None if no work to do / numpy missing.
    """
    if not _NUMPY_AVAILABLE:
        logger.debug("BPM backfill skipped: numpy unavailable")
        return None
    snapshot = [c for c in clips if isinstance(c, dict)]
    pending = [c for c in snapshot
               if not (c.get("bpm") and float(c.get("bpm") or 0) > 0)]
    if not pending:
        return None

    def worker():
        # Defensive — ffmpeg lookup also caches the "missing" state so we
        # don't spam-warn on every backfill.
        if not _resolve_ffmpeg():
            logger.debug("BPM backfill skipped: ffmpeg unavailable")
            return
        logger.info(f"BPM: backfilling {len(pending)} clip(s) in background...")
        ok = 0
        for c in pending:
            cid = c.get("id")
            if not cid:
                continue
            try:
                bpm = analyze_clip_bpm(
                    c.get("filepath") or "",
                    float(c.get("in_seconds") or 0.0),
                    float(c.get("out_seconds") or 0.0),
                )
            except Exception as e:
                logger.debug(f"BPM backfill: analyse failed for {cid}: {e}")
                bpm = None
            if bpm is not None:
                ok += 1
                try:
                    on_bpm(cid, bpm)
                except Exception as e:
                    logger.debug(f"BPM backfill: on_bpm raised for {cid}: {e}")
        logger.info(f"BPM backfill done: {ok}/{len(pending)} succeeded")

    t = threading.Thread(target=worker, name="bpm-backfill", daemon=True)
    t.start()
    return t
