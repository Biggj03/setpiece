"""
Background proxy-cache for 4K video files.

Idea: 4K source files put NVDEC + PCIe under heavy pressure during live
playback (especially with 4 preview streams running concurrently).
Transcode them once to a 1080p H.264 proxy via NVENC, cache the result
keyed by source-path+mtime, and hand back the proxy filepath whenever
the source is requested. Source paths still own the metadata (clips DB,
deck slots) — proxies are an invisible swap at the playback layer only.

Cache dir: ~/.setpiece/proxy/
File naming: <sha1(path|mtime)>.mp4 — collision-free, content-addressed.

Worker model: single background thread + FIFO queue. One ffmpeg job at
a time so we don't compound the GPU pressure we're trying to relieve.
Files already at <=1080p skip transcoding entirely (no benefit).
"""

import hashlib
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".setpiece" / "proxy"
TARGET_H = 1080  # cap height; preserves aspect ratio
TRANSCODE_TIMEOUT_S = 600  # 10 min hard cap for any one job
# ── intro trim ──────────────────────────────────────────────────────────
# Source files carry a ~5s title card the user doesn't want in proxies.
# We trim it at transcode time, but the proxy MUST stay timeline-aligned
# with the original: the app stores clip in/out points in *original-file*
# timestamps and seeks the proxy by them. A naive `-ss 5` would reset the
# proxy's PTS to 0 and offset every clip by 5s. Instead we use
# `-ss {TRIM} -copyts` (and NO -avoid_negative_ts) so the proxy's content
# is original[TRIM:end] but its timestamps remain [TRIM:end] — verified
# with ffprobe: first video frame pts_time == TRIM, format start_time and
# duration both reflect the trim. Set to 0 to disable trimming entirely.
# Overridable via settings.json key "proxy_trim_intro_sec".
PROXY_TRIM_SEC = 5.0
_SETTINGS_PATH = Path.home() / ".setpiece" / "settings.json"


def _load_trim_sec() -> float:
    """Read the intro-trim seconds from settings.json (key
    'proxy_trim_intro_sec'), falling back to the PROXY_TRIM_SEC default.
    Best-effort: missing/corrupt settings file -> default. Clamped to
    >= 0 (a negative trim makes no sense)."""
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        val = data.get("proxy_trim_intro_sec", PROXY_TRIM_SEC)
        return max(0.0, float(val))
    except Exception:
        return max(0.0, float(PROXY_TRIM_SEC))
# Keyframe interval (GOP size) for proxies. CRITICAL: without an explicit
# -g, NVENC with -tune ll produces an INFINITE GOP — one keyframe at the
# start, then nothing but P-frames. That wrecks this app: every seek /
# fire-into-body / loop has to decode from frame 0, and keyframe-only
# preview decode (-skip_frame nokey) gets exactly one frame. -g 15 (~0.6s
# at 24fps) makes proxies seek-friendly AND preview-friendly, for ~5%
# more file size. (Matches the CLAUDE.md NVENC-for-VJ guidance.)
PROXY_GOP = 15
# Hard cap on queue depth. Past this, new queue requests are dropped
# silently — protects against the "fire 16 pads → 16 4K transcodes pile
# up" cascade. The user can always force-build via an explicit action.
MAX_QUEUE_DEPTH = 3
# Disk budget for the proxy cache (Plex-style: keep transcodes around so
# the second play of a file is instant). When the cache grows past this,
# the least-recently-used proxies are evicted down to EVICT_TARGET_BYTES.
# get_proxy() bumps a proxy's mtime on every hit, so "least recently
# used" really means "least recently played". Override via the
# ProxyCache(max_cache_bytes=...) constructor arg / settings.json.
MAX_CACHE_BYTES = 100 * 1024 ** 3            # 100 GB
EVICT_TARGET_FRACTION = 0.90                 # evict down to 90% of the cap
# Windows priority class for ffmpeg subprocesses: BELOW_NORMAL so they
# don't fight mpv / Qt for CPU when the main app is busy. Safe no-op on
# non-Windows (just falls back to 0 = default).
import subprocess as _sp
import sys as _sys
_BELOW_NORMAL = getattr(_sp, "BELOW_NORMAL_PRIORITY_CLASS", 0)
_CREATE_NO_WINDOW = getattr(_sp, "CREATE_NO_WINDOW", 0)
FFMPEG_FLAGS = _BELOW_NORMAL | _CREATE_NO_WINDOW


def _kill_process_tree(proc) -> None:
    """Kill `proc` AND every child it spawned, then reap with wait().

    Windows + chocolatey ffmpeg: `ffmpeg` on PATH is a thin shim that
    launches the real ffmpeg as a CHILD. proc.kill() would terminate
    only the shim and orphan the real (CPU/GPU-heavy) transcode. taskkill
    /T walks the whole tree. Best-effort — never raises."""
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
    except Exception:
        pass
    pid = getattr(proc, "pid", None)
    if pid is not None and _sys.platform == "win32":
        try:
            _sp.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=5,
                creationflags=_CREATE_NO_WINDOW,
            )
        except Exception as e:
            logger.debug(f"proxy taskkill /T pid {pid} failed: {e}")
    try:
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=3.0)
    except Exception:
        logger.warning(f"proxy ffmpeg pid {pid} did not die after kill()")


def _ffmpeg_path() -> str:
    """Resolve ffmpeg binary. Falls back to PATH lookup."""
    return shutil.which("ffmpeg") or "ffmpeg"


def _ffprobe_path() -> str:
    return shutil.which("ffprobe") or "ffprobe"


def _key_for(src: Path) -> str:
    """Stable cache key: sha1 of (resolved path | mtime | trim). Re-edit a
    file → mtime changes → new cache entry; change the intro-trim setting
    → trim changes → new cache entry (old proxies become unreferenced and
    age out via LRU eviction). Old proxies are otherwise left behind until
    you garbage-collect the cache dir manually.

    The trim component is read fresh from settings.json on every call so
    the running app and the standalone batch script always agree on the
    key for a given source + trim setting."""
    trim = _load_trim_sec()
    # Normalise the trim to a stable string so 5 / 5.0 / 5.00 all hash
    # identically (avoids spurious cache misses from float formatting).
    trim_tag = f"{trim:.3f}"
    try:
        st = src.stat()
        raw = f"{src.resolve()}|{int(st.st_mtime)}|{trim_tag}"
    except Exception:
        raw = f"{src}|{trim_tag}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _proxy_path_for(src: Path) -> Path:
    return CACHE_DIR / f"{_key_for(src)}.mp4"


def _probe_height(src: Path) -> Optional[int]:
    """Return the source's *resolution class* — the SHORT side of the
    video frame, i.e. min(width, height) in pixels — or None on failure.

    Using the short side (not the raw stored height) makes the
    "does this need a 1080p proxy?" test orientation-agnostic:
      - landscape 4K  3840x2160 -> 2160  (needs proxy)
      - portrait  4K  2160x3840 -> 2160  (needs proxy)
      - landscape HD  1920x1080 -> 1080  (no proxy)
      - portrait  720  720x1280 ->  720  (no proxy)
    A portrait clip's stored `height` is its LONG side, so the old
    `stream=height`-only probe mis-classified vertical 720p/1080p clips
    as >1080 and transcoded them needlessly. Callers compare the result
    against TARGET_H (1080)."""
    try:
        result = subprocess.run(
            [_ffprobe_path(), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", str(src)],
            capture_output=True, text=True, timeout=15,
            creationflags=_CREATE_NO_WINDOW,
        )
        out = (result.stdout or "").strip()
        if out:
            first = out.splitlines()[0]
            nums = [int(p) for p in first.split(",") if p.strip().isdigit()]
            if len(nums) >= 2:
                return min(nums[0], nums[1])
            if len(nums) == 1:
                return nums[0]
    except Exception as e:
        logger.debug(f"ffprobe dimensions failed for {src}: {e}")
    return None


class ProxyCache:
    """Background-transcode 4K sources to 1080p proxies. Thread-safe."""

    def __init__(self, cache_dir: Path = CACHE_DIR, max_concurrent: int = 1,
                 max_cache_bytes: int = MAX_CACHE_BYTES):
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_cache_bytes = max(1024 ** 3, int(max_cache_bytes))  # floor 1 GB
        # Garbage-collect orphan .tmp.mp4 files from previously-killed
        # transcodes. They have no moov atom so mpv chokes on them.
        try:
            for orphan in self._cache_dir.glob("*.tmp.mp4"):
                try:
                    orphan.unlink()
                    logger.info(f"Cleaned orphan proxy tmp: {orphan.name}")
                except Exception:
                    pass
        except Exception:
            pass
        # Trim the cache to budget at startup in case it grew past the
        # cap in a prior session (or the cap was lowered).
        try:
            self._evict_if_over_cap()
        except Exception as e:
            logger.debug(f"startup cache eviction skipped: {e}")
        # Two queues, two priorities:
        #  _queue       — real-time requests (load_video / fire_deck). Depth
        #                 capped at MAX_QUEUE_DEPTH, drained FIRST.
        #  _batch_queue — whole-library pre-build (prebuild()). Unbounded,
        #                 drained only when _queue is empty. This is the
        #                 "Plex scans + transcodes everything" path.
        self._queue: "queue.Queue[Path]" = queue.Queue()
        self._batch_queue: "queue.Queue[Path]" = queue.Queue()
        self._enqueued: set[str] = set()        # real-time: queued or in flight
        self._batch_enqueued: set[str] = set()  # batch: queued or in flight
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # Paused state: when set, the worker drains the queue without
        # actually transcoding (returns items to the queue periodically)
        # OR we just hold the worker until _paused clears. Implementation
        # below: worker checks _paused before pulling next item.
        self._paused = threading.Event()
        self._current_proc: Optional[subprocess.Popen] = None  # in-flight ffmpeg
        self._current_path: Optional[Path] = None
        # max_concurrent currently fixed at 1 (single worker thread); the
        # constructor arg is here so we can scale later without API churn.
        self._workers: list[threading.Thread] = []
        for i in range(max(1, max_concurrent)):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"ProxyCache-worker-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    # ── public API ──────────────────────────────────────────────────────

    def get_proxy(self, src_path: str) -> Optional[str]:
        """Return the cached proxy's path if it exists and is fresh.
        Returns None if there's no proxy yet (caller plays the original)."""
        if not src_path:
            return None
        try:
            src = Path(src_path)
            if not src.exists():
                return None
            proxy = _proxy_path_for(src)
            if proxy.exists() and proxy.stat().st_size > 0:
                # Bump mtime so LRU eviction treats "recently played" as
                # "recently used" — a proxy you keep firing never gets
                # evicted out from under you.
                try:
                    os.utime(proxy, None)
                except Exception:
                    pass
                return str(proxy)
        except Exception as e:
            logger.debug(f"get_proxy({src_path}) error: {e}")
        return None

    def queue(self, src_path: str) -> bool:
        """Schedule a transcode for src_path. Idempotent — already-queued
        and already-cached paths are skipped. Returns True if newly queued.

        Caps total queue depth at MAX_QUEUE_DEPTH to prevent avalanche
        scenarios (e.g. firing 16 MK2 pads each queueing a 4K transcode)."""
        if not src_path:
            return False
        try:
            src = Path(src_path)
            if not src.exists() or not src.is_file():
                return False
            # Already cached?
            proxy = _proxy_path_for(src)
            if proxy.exists() and proxy.stat().st_size > 0:
                return False
            key = str(src.resolve())
            with self._lock:
                # Already handled by either queue? (batch counts — the
                # background pre-build will get to it.)
                if key in self._enqueued or key in self._batch_enqueued:
                    return False
                if len(self._enqueued) >= MAX_QUEUE_DEPTH:
                    logger.debug(
                        f"Proxy queue full ({len(self._enqueued)}/{MAX_QUEUE_DEPTH}); "
                        f"dropping {src.name}"
                    )
                    return False
                self._enqueued.add(key)
            self._queue.put(src)
            logger.info(f"Proxy queued ({len(self._enqueued)}/{MAX_QUEUE_DEPTH}): {src.name}")
            return True
        except Exception as e:
            logger.debug(f"queue({src_path}) error: {e}")
            return False

    def prebuild(self, folders) -> int:
        """Plex-style whole-library pre-transcode. Walks every folder in
        `folders` (recursively), and for every video file that doesn't
        already have a fresh proxy, queues it on the BATCH queue.

        The batch queue is unbounded and drained at lower priority than
        real-time load/fire requests, so this can chew through the whole
        library in the background without ever delaying a live action.
        Idempotent — files already cached or already queued are skipped.
        Returns the count newly queued."""
        exts = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
        queued = 0
        seen: set[str] = set()
        for folder in (folders or []):
            if not folder:
                continue
            try:
                root = Path(folder)
                if not root.is_dir():
                    continue
                for f in sorted(root.rglob("*")):
                    try:
                        if not f.is_file() or f.suffix.lower() not in exts:
                            continue
                        key = str(f.resolve())
                        if key in seen:
                            continue
                        seen.add(key)
                        proxy = _proxy_path_for(f)
                        if proxy.exists() and proxy.stat().st_size > 0:
                            continue  # already have a fresh proxy
                        with self._lock:
                            if key in self._enqueued or key in self._batch_enqueued:
                                continue
                            self._batch_enqueued.add(key)
                        self._batch_queue.put(f)
                        queued += 1
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"prebuild walk failed for {folder}: {e}")
        if queued:
            logger.info(
                f"Proxy prebuild: queued {queued} file(s) for background "
                f"transcode (batch queue)"
            )
        else:
            logger.info("Proxy prebuild: nothing to do — all files cached")
        return queued

    def stop(self):
        """Signal worker threads to exit. Doesn't kill in-flight ffmpeg."""
        self._stop.set()

    # ── pause / cancel controls ────────────────────────────────────────

    def qsize(self) -> int:
        """Approximate number of files waiting across BOTH queues
        (real-time + batch pre-build), excluding the one in flight."""
        try:
            return self._queue.qsize() + self._batch_queue.qsize()
        except Exception:
            return 0

    def batch_qsize(self) -> int:
        """Files still waiting in the background pre-build queue."""
        try:
            return self._batch_queue.qsize()
        except Exception:
            return 0

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def is_busy(self) -> bool:
        """True if a transcode is currently in flight."""
        with self._lock:
            return self._current_proc is not None

    def current_file_name(self) -> str:
        try:
            with self._lock:
                return self._current_path.name if self._current_path else ""
        except Exception:
            return ""

    def pause(self):
        """Block the worker from starting any NEW transcodes. The one
        currently in flight finishes normally (use cancel_current to
        kill it immediately)."""
        self._paused.set()
        logger.info("ProxyCache: paused")

    def resume(self):
        self._paused.clear()
        logger.info("ProxyCache: resumed")

    # ── disk-budget eviction (Plex-style LRU cache) ────────────────────

    def cache_size_bytes(self) -> int:
        """Total bytes of finished proxy files (.mp4, excludes .tmp.mp4)."""
        total = 0
        try:
            for f in self._cache_dir.glob("*.mp4"):
                if f.name.endswith(".tmp.mp4"):
                    continue
                try:
                    total += f.stat().st_size
                except Exception:
                    pass
        except Exception:
            pass
        return total

    def _evict_if_over_cap(self) -> int:
        """If the cache exceeds the disk budget, delete least-recently-used
        proxies (oldest mtime first — get_proxy bumps mtime on every play)
        until it's back under EVICT_TARGET_FRACTION of the cap. Returns the
        number of files evicted. Best-effort: never raises into the caller."""
        try:
            entries = []
            total = 0
            for f in self._cache_dir.glob("*.mp4"):
                if f.name.endswith(".tmp.mp4"):
                    continue
                try:
                    st = f.stat()
                except Exception:
                    continue
                entries.append((st.st_mtime, st.st_size, f))
                total += st.st_size
            if total <= self._max_cache_bytes:
                return 0
            target = int(self._max_cache_bytes * EVICT_TARGET_FRACTION)
            # Oldest-used first.
            entries.sort(key=lambda e: e[0])
            evicted = 0
            freed = 0
            for _mtime, size, f in entries:
                if total - freed <= target:
                    break
                try:
                    f.unlink()
                    freed += size
                    evicted += 1
                except Exception as e:
                    logger.debug(f"proxy evict failed for {f.name}: {e}")
            if evicted:
                logger.info(
                    f"Proxy cache over budget — evicted {evicted} LRU "
                    f"proxy(ies), freed {freed / 1024**3:.1f} GB "
                    f"(cap {self._max_cache_bytes / 1024**3:.0f} GB)"
                )
            return evicted
        except Exception as e:
            logger.debug(f"_evict_if_over_cap error: {e}")
            return 0

    def cancel_current(self) -> bool:
        """Kill the in-flight ffmpeg process. Returns True if something
        was killed. Reads + kills under the lock so we never race the
        worker swapping _current_proc out from under us. (Audit fix C4.)"""
        with self._lock:
            proc = self._current_proc
            name = self._current_path.name if self._current_path else ""
            if proc is None:
                return False
            try:
                # Tree-kill: the chocolatey ffmpeg shim spawns the real
                # transcoder as a child — proc.kill() alone orphans it.
                _kill_process_tree(proc)
                logger.info(f"ProxyCache: killed in-flight transcode of {name}")
                return True
            except Exception as e:
                logger.debug(f"cancel_current failed: {e}")
                return False

    def cancel_pending(self) -> int:
        """Drop all queued (not-yet-started) transcodes from BOTH queues —
        real-time and batch pre-build. Doesn't kill the in-flight one —
        call cancel_current for that. Returns the number of items dropped."""
        dropped = 0
        for q, enq in ((self._queue, self._enqueued),
                       (self._batch_queue, self._batch_enqueued)):
            try:
                while True:
                    item = q.get_nowait()
                    with self._lock:
                        enq.discard(str(Path(item).resolve()))
                    dropped += 1
            except queue.Empty:
                pass
        if dropped:
            logger.info(f"ProxyCache: dropped {dropped} pending transcodes")
        return dropped

    # ── worker ─────────────────────────────────────────────────────────

    def _worker_loop(self):
        while not self._stop.is_set():
            # Pause gate: hold here without pulling from either queue.
            if self._paused.is_set():
                if self._stop.wait(0.3):
                    return
                continue
            # Priority: drain real-time requests (load_video / fire_deck)
            # FIRST — those are "the user wants this file fast". Only when
            # that queue is empty do we pull from the background batch
            # pre-build queue. This keeps a whole-library prebuild from
            # ever delaying a live load.
            src = None
            try:
                src = self._queue.get_nowait()
            except queue.Empty:
                try:
                    src = self._batch_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
            try:
                self._transcode(src)
            except Exception as e:
                logger.error(f"Proxy worker error on {src}: {e}", exc_info=True)
            finally:
                with self._lock:
                    key = str(src.resolve())
                    # Discard from BOTH sets — a file could have been in
                    # either queue; clearing both is harmless and safe.
                    self._enqueued.discard(key)
                    self._batch_enqueued.discard(key)
                    self._current_proc = None
                    self._current_path = None

    def _transcode(self, src: Path) -> None:
        """ffmpeg → 1080p H.264 NVENC, AAC stereo. Skip if source already
        ≤ TARGET_H (no benefit). Atomic write via .tmp → rename.

        Also defers files >5 GB at >1080p — these are long compilations
        that won't fit the 600s transcode timeout on this rig (every
        attempt times out and orphans an ffmpeg). batch_transcode_recent
        has the same preflight; this brings the prebuild path in sync
        so bank-load fires don't spawn doomed transcodes."""
        try:
            height = _probe_height(src)
        except Exception:
            height = None
        if height is not None and height <= TARGET_H:
            logger.info(f"Proxy skip (already {height}p ≤ {TARGET_H}p): {src.name}")
            return
        # Big-file preflight — match batch_transcode_recent's --big-file-defer-gb.
        try:
            size_gb = src.stat().st_size / 1024 ** 3
        except Exception:
            size_gb = 0.0
        if size_gb > 5.0 and (height or 0) > TARGET_H:
            logger.info(
                f"Proxy defer-upfront ({size_gb:.1f}GB at {height}p > 5GB / >1080p — "
                f"won't fit transcode timeout): {src.name}")
            return

        proxy = _proxy_path_for(src)
        tmp = proxy.with_suffix(".tmp.mp4")
        trim = _load_trim_sec()
        # Timeline-aligned intro trim: -ss {trim} BEFORE -i (fast input
        # seek) + -copyts so the proxy's content starts at original[trim]
        # but its PTS stay at [trim:end] — a seek to original-timestamp T
        # still lands at T in the proxy. Do NOT add -avoid_negative_ts:
        # make_zero would re-zero the PTS and reintroduce the offset.
        trim_args = ["-ss", f"{trim:.3f}", "-copyts"] if trim > 0 else []
        # 2026-05-17 fix: dropped explicit "-c:v h264_cuvid" — it fails
        # on certain 4K H.264 profiles with rc=-22 INVALID_PARAM. Letting
        # ffmpeg auto-pick the decoder under "-hwaccel cuda" is robust
        # across source profiles (mirrors batch_transcode_recent.py:222
        # which had the same fix applied earlier). Added "-surfaces 16"
        # to prevent NVENC's runaway surface negotiation (48 → 4096).
        cmd = [
            _ffmpeg_path(),
            "-y",
            "-hide_banner", "-loglevel", "error",
            *trim_args,
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-i", str(src),
            "-vf", f"scale_cuda=-2:{TARGET_H}",
            "-c:v", "h264_nvenc",
            "-preset", "p1",          # fastest
            "-tune", "ll",            # low latency
            "-surfaces", "16",
            # Frequent keyframes + no B-frames. WITHOUT -g, NVENC -tune ll
            # emits an infinite GOP (1 keyframe, then all P-frames) which
            # makes every seek/fire/loop decode from frame 0 and breaks
            # keyframe-only preview decode. -g 15 ≈ 0.6s keyframe spacing.
            "-g", str(PROXY_GOP), "-bf", "0",
            "-b:v", "6M", "-maxrate", "6M", "-bufsize", "12M",
            "-rc", "vbr",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(tmp),
        ]
        logger.info(f"Proxy transcode start: {src.name}")
        t0 = time.time()
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=FFMPEG_FLAGS,
            )
        except Exception as e:
            logger.warning(f"Proxy ffmpeg spawn failed: {e}")
            return
        # Publish the in-flight proc + path under the lock so cancel_current
        # / is_busy / current_file_name see a consistent pair. (Audit fix C4.)
        with self._lock:
            self._current_proc = proc
            self._current_path = src
        try:
            stdout, stderr = proc.communicate(timeout=TRANSCODE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning(f"Proxy transcode timed out: {src.name}")
            _kill_process_tree(proc)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return
        rc = proc.returncode
        # rc == -9 / -15 etc. means we were killed (cancel_current).
        if rc is not None and rc < 0:
            logger.info(f"Proxy transcode cancelled: {src.name}")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return
        # Mock the old `result` shape for the existing branch below.
        class _R: pass
        result = _R()
        result.returncode = rc
        result.stderr = (stderr or b"").decode("utf-8", "replace")
        if result.returncode != 0:
            logger.warning(
                f"Proxy transcode failed (rc={result.returncode}): {src.name}\n"
                f"stderr tail: {(result.stderr or '')[-400:]}"
            )
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            # Try a software fallback once — older ffmpeg builds can hate
            # the cuda input filter on weird containers.
            self._transcode_sw(src)
            return
        try:
            tmp.replace(proxy)
        except Exception as e:
            logger.warning(f"Proxy rename failed: {e}")
            return
        dt = time.time() - t0
        size_mb = proxy.stat().st_size / (1024 * 1024)
        logger.info(f"Proxy ready: {src.name} → {proxy.name} ({size_mb:.1f} MB, {dt:.1f}s)")
        # Keep the cache within its disk budget (Plex-style LRU eviction).
        self._evict_if_over_cap()

    def _transcode_sw(self, src: Path) -> None:
        """Software fallback when the NVENC pipeline barfs. Slower but
        works on weird container/codec combos.

        Uses Popen + publishes the proc under the lock (Audit fix M2) so
        cancel_current() can actually kill the software ffmpeg too — the
        old subprocess.run() version left _current_proc pointing at the
        dead NVENC proc, so the (potentially minutes-long) sw transcode
        was uncancellable."""
        proxy = _proxy_path_for(src)
        tmp = proxy.with_suffix(".tmp.mp4")
        trim = _load_trim_sec()
        # Same timeline-aligned trim as the NVENC path — see _transcode.
        trim_args = ["-ss", f"{trim:.3f}", "-copyts"] if trim > 0 else []
        cmd = [
            _ffmpeg_path(),
            "-y", "-hide_banner", "-loglevel", "error",
            *trim_args,
            "-i", str(src),
            "-vf", f"scale=-2:{TARGET_H}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            # Match the NVENC path: frequent keyframes, no B-frames, so
            # proxies seek fast and preview-decode cheaply (see PROXY_GOP).
            "-g", str(PROXY_GOP), "-bf", "0",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(tmp),
        ]
        logger.info(f"Proxy transcode (sw fallback): {src.name}")
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=FFMPEG_FLAGS,
            )
        except Exception as e:
            logger.warning(f"Proxy sw ffmpeg spawn failed: {e}")
            return
        # Replace the (now-dead) NVENC proc with the live sw proc so
        # cancel_current / is_busy / current_file_name stay accurate.
        with self._lock:
            self._current_proc = proc
            self._current_path = src
        try:
            _stdout, stderr = proc.communicate(timeout=TRANSCODE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning(f"Proxy sw transcode timed out: {src.name}")
            _kill_process_tree(proc)
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
            return
        rc = proc.returncode
        # rc < 0 → killed by cancel_current.
        if rc is not None and rc < 0:
            logger.info(f"Proxy sw transcode cancelled: {src.name}")
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
            return
        if rc != 0:
            stderr_txt = (stderr or b"").decode("utf-8", "replace")
            logger.warning(
                f"Proxy sw transcode failed (rc={rc}): {src.name}\n"
                f"stderr tail: {stderr_txt[-400:]}"
            )
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
            return
        try:
            tmp.replace(proxy)
        except Exception as e:
            logger.warning(f"Proxy rename (sw) failed: {e}")
            return
        logger.info(f"Proxy ready (sw): {src.name}")
        self._evict_if_over_cap()
