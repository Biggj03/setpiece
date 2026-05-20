"""
Per-deck MJPEG preview pipeline.

Architecture (per the research doc, "Option 2"):
    one ffmpeg subprocess per deck, NVDEC for decoding (free on a GTX 1080 —
    NVDEC has 40+ session capacity), software MJPEG encode at 320x180@5fps
    (~1% CPU per stream), piped to stdout. A daemon reader thread parses
    SOI/EOI byte markers, pushes frames into per-subscriber queues. The
    HTTP layer consumes via subscribe() and writes
    multipart/x-mixed-replace boundaries to Safari, which natively swaps
    the <img> source on each frame — zero client JS needed.

Why ffmpeg subprocess (vs libmpv offscreen, lavfi tile, WebRTC, NDI):
    NVENC on consumer Pascal is capped at 2 simultaneous encode sessions,
    so anything that hardware-encodes the previews collides with the
    primary live encoder. Software MJPEG sidesteps that entirely while
    still hardware-decoding the source. See docs/4DECK_PREVIEW_REPORT.md.

Threading model:
    - main: spawns/kills subprocesses, exposes get_latest_frame / subscribe
    - reader thread (per stream): reads ffmpeg stdout, splits JPEGs,
      fans out to subscribers via per-subscriber queue.Queue(maxsize=2)
    - watchdog: reader notices ffmpeg exit; auto-restarts with backoff
    - HTTP request thread: drains its queue with a small timeout, writes
      to wfile. Disconnect = write raises = unsubscribe + cleanup.

This module imports only stdlib. No mpv/HID/ffmpeg dependency at
import time — `import preview_streams` is always safe.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Generator, Iterable, Optional

logger = logging.getLogger(__name__)

# Hard cap. Matches DECK_COUNT in decks.py — we own one ffmpeg per deck.
MAX_STREAMS = 4

# Tuned for 4-deck previews of mixed 1080p/4K source files. Dropped from
# 320x180@5fps to 240x135@3fps after observing GPU pressure with all 4
# slots loaded with 4K content. ~44% fewer pixels + 40% fewer frames →
# roughly half the per-stream encode + PCIe bandwidth, with negligible
# loss for "is this the right vibe?" thumbnails. Easy to bump back up
# if the rig is underloaded.
FRAME_W = 240
FRAME_H = 135
FRAME_FPS = 3
JPEG_Q = 6

# How long a subscriber waits per get() before checking liveness. Keeps
# the HTTP thread from blocking forever if a stream stalls (shutdown,
# ffmpeg crash, etc).
SUBSCRIBER_TIMEOUT_SEC = 1.5

# A per-subscriber queue depth of 2 means: if the iPad is one frame
# behind, hold one buffered frame; if it falls TWO behind, drop oldest.
# Slow client never back-pressures the producer.
SUBSCRIBER_QUEUE_MAXSIZE = 2

# ffmpeg respawn backoff. Crash → wait → respawn. Resets after a stream
# stays alive >30s.
RESPAWN_BACKOFF_INITIAL = 1.0
RESPAWN_BACKOFF_MAX = 15.0
RESPAWN_HEALTHY_THRESHOLD = 30.0

# Windows: hide the console window that subprocess.Popen would otherwise
# pop for each ffmpeg. No-op on Linux.
_WIN_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _kill_process_tree(proc: Optional["subprocess.Popen"], deck_idx: int = -1) -> None:
    """Kill `proc` AND every child it spawned, then confirm with wait().

    Critical on Windows with a chocolatey-installed ffmpeg: `ffmpeg` on
    PATH is a ~13MB *shim* that launches the real ~200MB ffmpeg as a
    CHILD process. A plain proc.kill() terminates only the shim — the
    real ffmpeg orphans and keeps decoding 4K forever. `taskkill /T`
    walks the whole tree so the real worker actually dies.

    Best-effort: a stuck handle logs but never raises."""
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return  # already dead
    except Exception:
        pass
    pid = getattr(proc, "pid", None)
    if pid is not None and sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=5,
                creationflags=_WIN_NO_WINDOW,
            )
        except Exception as e:
            logger.debug(f"deck {deck_idx}: taskkill /T pid {pid} failed: {e}")
    # Backup path (non-Windows, or taskkill missed it) + reap the handle.
    try:
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=3.0)
    except Exception:
        logger.warning(
            f"deck {deck_idx}: ffmpeg pid {pid} did not die after kill()"
        )


# A sentinel for graceful subscriber shutdown — yielded into the queue
# so the consumer wakes immediately instead of hitting the timeout.
_STOP_SENTINEL = object()


class _Stream:
    """One deck's running ffmpeg + reader thread + subscriber set.

    Internal helper for PreviewStreamManager. Not for direct use.
    """

    def __init__(
        self,
        deck_idx: int,
        filepath: str,
        in_sec: float = 0.0,
        out_sec: Optional[float] = None,
        ffmpeg_path: str = "ffmpeg",
        prefer_hwaccel: bool = True,
    ):
        self.deck_idx = int(deck_idx)
        self.filepath = str(filepath)
        self.in_sec = max(0.0, float(in_sec or 0.0))
        self.out_sec = float(out_sec) if out_sec else None
        self.ffmpeg_path = ffmpeg_path
        self.prefer_hwaccel = prefer_hwaccel

        # Set when stop_stream() is called or the manager shuts down.
        # Reader thread checks it to break out of the read loop without
        # waiting for ffmpeg to finish naturally.
        self._stop_evt = threading.Event()

        # Subscribers: each is a queue.Queue. Producer puts JPEG bytes;
        # if full, pops the oldest first (drop-on-full). Set is guarded
        # by _sub_lock because subscribers can come and go from any HTTP
        # request thread.
        self._sub_lock = threading.Lock()
        self._subscribers: set[queue.Queue] = set()

        # Most recent frame; useful for "give me one JPEG right now"
        # endpoints (poster/preview thumbnail) without subscribing.
        self._latest_lock = threading.Lock()
        self._latest_frame: Optional[bytes] = None

        # Process + reader handle.
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        # Every ffmpeg this stream has ever spawned, guarded by _proc_lock.
        # stop() kills EVERY entry, not just self._proc — closes the race
        # where stop() reads self._proc in the window after _spawn() but
        # before `self._proc = proc`, so a freshly-spawned ffmpeg would
        # otherwise leak as an orphan. Dead procs are pruned on register.
        self._proc_lock = threading.Lock()
        self._all_procs: list[subprocess.Popen] = []

        # Track if NVDEC actually worked or we fell back. Logged once per
        # stream so the user knows whether to expect 1% or 5% CPU.
        self._using_hwaccel = False

        # Spawn timestamp + estimated loop duration — used by
        # get_position() to tell the live player "the preview is currently
        # showing frame T", so 'fire from where I see it' lands close.
        self._spawn_ts: Optional[float] = None
        # If out_sec is set, that's the loop duration. Otherwise we can't
        # know without probing — fallback to assuming the stream advances
        # forever and let the caller deal with overflow.
        self._loop_duration: Optional[float] = (
            (self.out_sec - self.in_sec)
            if (self.out_sec and self.out_sec > self.in_sec)
            else None
        )

    # ── Public-ish API (called from manager) ────────────────────────────

    def start(self) -> None:
        """Spawn ffmpeg + reader. Idempotent if already running."""
        if self._reader and self._reader.is_alive():
            return
        self._stop_evt.clear()
        self._reader = threading.Thread(
            target=self._reader_loop,
            name=f"preview-deck-{self.deck_idx}",
            daemon=True,
        )
        self._reader.start()

    def _register_proc(self, proc: subprocess.Popen) -> None:
        """Track a freshly-spawned ffmpeg so stop() can guarantee it dies.
        Prunes already-dead entries to keep the list bounded."""
        with self._proc_lock:
            self._all_procs = [p for p in self._all_procs if p.poll() is None]
            self._all_procs.append(proc)

    def _kill_proc(self, proc: Optional[subprocess.Popen]) -> None:
        """Kill one ffmpeg (whole process tree) and confirm it's gone.
        Delegates to the module-level _kill_process_tree so the real
        ffmpeg behind a chocolatey shim actually dies."""
        _kill_process_tree(proc, self.deck_idx)

    def stop(self, timeout: float = 2.0) -> None:
        """Signal stop, kill EVERY ffmpeg this stream spawned, wake all
        subscribers, join reader. Safe to call repeatedly."""
        self._stop_evt.set()
        # Kill every ffmpeg we've ever spawned — not just self._proc — so
        # a process spawned in the _spawn()/`self._proc=proc` race window
        # can't survive as an orphan. Confirm each death with wait().
        with self._proc_lock:
            procs = list(self._all_procs)
        for p in procs:
            self._kill_proc(p)
        self._kill_proc(self._proc)  # belt + suspenders
        # Wake any subscribers stuck on queue.get()
        self._broadcast_stop()
        # Join reader (don't wait forever — caller might be in shutdown)
        r = self._reader
        if r is not None:
            r.join(timeout=timeout)
        self._reader = None
        self._proc = None
        with self._proc_lock:
            self._all_procs.clear()

    def get_latest_frame(self) -> Optional[bytes]:
        with self._latest_lock:
            return self._latest_frame

    def subscribe(self) -> "queue.Queue":
        """Register a new subscriber. Returns a Queue the caller drains
        in a loop until it sees _STOP_SENTINEL or its own connection
        breaks. Caller MUST call unsubscribe(q) on cleanup."""
        q: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_QUEUE_MAXSIZE)
        # Pre-seed with the latest frame so the iPad sees something
        # immediately, even if ffmpeg's next frame is ~200ms away.
        seed = self.get_latest_frame()
        if seed:
            try:
                q.put_nowait(seed)
            except queue.Full:
                pass
        with self._sub_lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: "queue.Queue") -> None:
        with self._sub_lock:
            self._subscribers.discard(q)

    def subscriber_count(self) -> int:
        with self._sub_lock:
            return len(self._subscribers)

    def get_position(self) -> Optional[float]:
        """Approximate absolute timeline position (seconds into the source
        file) the preview is currently displaying. Returns None if the
        stream isn't running yet. Used by 'fire from where I see it'."""
        if self._spawn_ts is None:
            return None
        elapsed = time.time() - self._spawn_ts
        if self._loop_duration and self._loop_duration > 0:
            elapsed = elapsed % self._loop_duration
        return self.in_sec + elapsed

    # ── Internals ───────────────────────────────────────────────────────

    def _broadcast_stop(self) -> None:
        with self._sub_lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                # Drop one if full so the sentinel makes it in.
                if q.full():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                q.put_nowait(_STOP_SENTINEL)
            except Exception:
                pass

    def _broadcast_frame(self, frame: bytes) -> None:
        with self._latest_lock:
            self._latest_frame = frame
        with self._sub_lock:
            subs = list(self._subscribers)
        for q in subs:
            # Drop-oldest semantics: if the queue is full the iPad isn't
            # keeping up. We'd rather skip a frame than block ffmpeg.
            try:
                q.put_nowait(frame)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(frame)
                except queue.Full:
                    pass

    def _build_cmd(self, use_hwaccel: bool) -> list[str]:
        """Build the ffmpeg command line.

        Per the PDF research:
            ffmpeg -hwaccel cuda -hwaccel_output_format cuda \
                   -c:v h264_cuvid -i <file> \
                   -vf "scale_cuda=320:180,hwdownload,format=nv12,format=yuv420p" \
                   -r 5 -f image2pipe -vcodec mjpeg -q:v 6 -

        For looping: `-stream_loop -1` BEFORE `-i` is the reliable form
        across ffmpeg builds (image2pipe `-loop -1` is not universal).
        Software fallback drops the cuda flags + uses vanilla scale.
        """
        cmd: list[str] = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",  # back to error after diagnosing ffmpeg exit pattern
            "-nostdin",
            # -re: read the input at its NATIVE realtime rate. Without it
            # ffmpeg races through the file as fast as it can decode —
            # which, combined with -stream_loop, used to expose the trim
            # bug below in seconds, and also wastes CPU. -re paces the
            # whole pipeline to 1x so the loop runs in realtime.
            "-re",
            # Loop the source forever so ffmpeg never exits at EOF.
            "-stream_loop", "-1",
            # DECODE ONLY KEYFRAMES — the single biggest cost lever for
            # previews. `-r 3` only throttles OUTPUT; without this ffmpeg
            # still fully DECODES every input frame and throws most away.
            # For a 240x135 thumbnail, keyframes-only is plenty and cuts
            # decode work ~10-25x. Goes BEFORE -i (it's a decoder option).
            "-skip_frame", "nokey",
        ]
        if use_hwaccel:
            cmd += [
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-c:v", "h264_cuvid",
            ]
        # Fast input-level seek BEFORE -i (the lesson from the Setpiece
        # CLAUDE.md notes: "-ss before -i" is brutally faster than
        # the demuxer-level form).
        if self.in_sec > 0.0:
            cmd += ["-ss", f"{self.in_sec:.3f}"]
        cmd += ["-i", self.filepath]
        # NOTE: there is intentionally NO trim filter and NO -t. The old
        # `trim=0:{duration}` filter was a CRASH-LOOP bug: -stream_loop -1
        # INCREMENTS PTS across loops, so after the first pass the PTS
        # exceeds the trim window, the trim filter EOFs the whole graph,
        # and ffmpeg EXITS — the reader then respawns it, forever. A
        # looping preview just shows the whole file on repeat; honoring a
        # sub-range in/out isn't worth re-introducing that footgun for a
        # 240x135 thumbnail.
        if use_hwaccel:
            vf = f"scale_cuda={FRAME_W}:{FRAME_H},hwdownload,format=nv12,format=yuv420p"
        else:
            vf = f"scale={FRAME_W}:{FRAME_H}"
        cmd += [
            "-an", "-sn",  # no audio, no subs
            "-vf", vf,
            "-r", str(FRAME_FPS),
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", str(JPEG_Q),
            "-",
        ]
        return cmd

    def _spawn(self, use_hwaccel: bool) -> Optional[subprocess.Popen]:
        cmd = self._build_cmd(use_hwaccel)
        logger.debug(f"deck {self.deck_idx} ffmpeg: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                bufsize=0,
                creationflags=_WIN_NO_WINDOW,
            )
            return proc
        except FileNotFoundError:
            # ffmpeg binary missing — caller will log + give up.
            return None
        except Exception as e:
            logger.warning(f"deck {self.deck_idx}: spawn failed: {e}")
            return None

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        """Pump ffmpeg stderr in a daemon thread so its pipe buffer
        doesn't fill (which would back-pressure stdout and stall the
        whole pipeline). We log at debug — only the last line on exit
        gets escalated to warning."""
        def _pump():
            try:
                for line in iter(proc.stderr.readline, b""):
                    if not line:
                        break
                    logger.debug(
                        f"deck {self.deck_idx} ffmpeg-stderr: "
                        f"{line.rstrip().decode('utf-8', 'replace')}"
                    )
            except Exception:
                pass

        t = threading.Thread(
            target=_pump,
            name=f"preview-deck-{self.deck_idx}-stderr",
            daemon=True,
        )
        t.start()

    def _reader_loop(self) -> None:
        """Spawn ffmpeg, parse JPEGs out of stdout, broadcast.
        On crash: backoff + respawn unless _stop_evt is set."""
        backoff = RESPAWN_BACKOFF_INITIAL
        while not self._stop_evt.is_set():
            use_hw = self.prefer_hwaccel
            proc = self._spawn(use_hwaccel=use_hw)
            if proc is None and self.prefer_hwaccel:
                # ffmpeg missing entirely — try one more time without
                # the cuda flags in case it's actually a build that
                # rejects -hwaccel cuda. If still no go, log and bail.
                proc = self._spawn(use_hwaccel=False)
                use_hw = False
            if proc is None:
                logger.error(
                    f"deck {self.deck_idx}: ffmpeg unavailable; "
                    f"giving up on this stream"
                )
                return

            # Register + publish the proc handle IMMEDIATELY — before any
            # logging / stderr-drain — so a concurrent stop() can always
            # find and kill it. _register_proc must happen first so even
            # if stop() reads a stale self._proc, the proc is in _all_procs.
            self._register_proc(proc)
            self._proc = proc
            self._using_hwaccel = use_hw
            # If stop() fired during the spawn, don't even start parsing —
            # just fall through to the finally and kill the proc we just
            # made.
            if self._stop_evt.is_set():
                self._kill_proc(proc)
                break
            logger.info(
                f"deck {self.deck_idx}: ffmpeg started "
                f"(hwaccel={use_hw}, file={Path(self.filepath).name})"
            )
            self._drain_stderr(proc)

            spawn_time = time.time()
            self._spawn_ts = spawn_time
            try:
                self._parse_jpegs(proc)
            except Exception as e:
                logger.debug(f"deck {self.deck_idx}: parser error: {e}")
            finally:
                # Make sure ffmpeg is gone (and CONFIRMED gone) before we
                # respawn — _kill_proc waits on it.
                self._kill_proc(proc)

            if self._stop_evt.is_set():
                break

            # Stayed up for a while → reset backoff. Otherwise grow it.
            uptime = time.time() - spawn_time
            if uptime > RESPAWN_HEALTHY_THRESHOLD:
                backoff = RESPAWN_BACKOFF_INITIAL
            else:
                backoff = min(backoff * 2, RESPAWN_BACKOFF_MAX)
                logger.warning(
                    f"deck {self.deck_idx}: ffmpeg exited after "
                    f"{uptime:.1f}s; respawning in {backoff:.1f}s"
                )

            # If hwaccel was on and we died fast, try software next loop.
            # This is the "no NVIDIA driver / wrong codec" path.
            if use_hw and uptime < 2.0:
                logger.warning(
                    f"deck {self.deck_idx}: hwaccel ffmpeg died fast — "
                    f"falling back to software decode for next attempt"
                )
                self.prefer_hwaccel = False

            # Sleep, but wake immediately on stop().
            self._stop_evt.wait(timeout=backoff)

        # Final cleanup signal to subscribers
        self._broadcast_stop()
        logger.info(f"deck {self.deck_idx}: stream worker exiting")

    def _parse_jpegs(self, proc: subprocess.Popen) -> None:
        """Read ffmpeg stdout in chunks; emit complete JPEGs.

        JPEG framing: every JPEG starts with 0xFF 0xD8 (SOI) and ends
        with 0xFF 0xD9 (EOI). The byte sequence 0xFF 0xD9 can technically
        appear inside a JPEG's entropy-coded data, but ffmpeg's MJPEG
        encoder doesn't emit nested 0xFF 0xD9 outside the actual EOI.
        Conservative: take the first SOI, find the next EOI after it,
        cut there.
        """
        if proc.stdout is None:
            return
        buf = bytearray()
        # Read in modest chunks — too small = syscall-heavy, too large
        # = increased latency at low fps. 16 KB ≈ one frame at this size.
        CHUNK = 16 * 1024
        while not self._stop_evt.is_set():
            try:
                chunk = proc.stdout.read(CHUNK)
            except Exception as e:
                logger.debug(f"deck {self.deck_idx}: read error: {e}")
                break
            if not chunk:
                # ffmpeg closed stdout → exit + respawn upstream
                break
            buf.extend(chunk)

            # Drain as many complete frames as the buffer holds.
            while True:
                soi = buf.find(b"\xff\xd8")
                if soi < 0:
                    # No frame start visible — drop everything (probably
                    # ffmpeg startup garbage / log spill that shouldn't
                    # have hit stdout, but be safe).
                    if len(buf) > 4 * CHUNK:
                        del buf[:-2]
                    break
                # Look for EOI strictly after SOI
                eoi = buf.find(b"\xff\xd9", soi + 2)
                if eoi < 0:
                    # Wait for more bytes. Trim anything before SOI to
                    # bound memory.
                    if soi > 0:
                        del buf[:soi]
                    break
                end = eoi + 2
                frame = bytes(buf[soi:end])
                del buf[:end]
                # Sanity: empty / tiny "frame" = ffmpeg hiccup, skip it
                if len(frame) > 200:
                    self._broadcast_frame(frame)


class PreviewStreamManager:
    """Owns one _Stream per deck. Thread-safe. No Qt dependency.

    Typical lifecycle:
        mgr = PreviewStreamManager()
        mgr.start_stream(0, "/path/to/clip.mp4", in_sec=12.5, out_sec=45.0)
        ... HTTP handler does: q = mgr.subscribe(0); for f in iter_frames(q): ...
        mgr.stop_stream(0)
        mgr.stop_all()  # on app shutdown
    """

    def __init__(self, ffmpeg_path: Optional[str] = None,
                 prefer_hwaccel: bool = False):
        self._lock = threading.RLock()
        self._streams: dict[int, _Stream] = {}
        # Decode path for the 4 preview ffmpegs:
        #   prefer_hwaccel=False (default) → SOFTWARE decode. 4K NVDEC
        #     decode surfaces are fat in VRAM, and four of them on a 4GB
        #     card pushes it into memory pressure ("chugging" even when
        #     GPU compute % looks low). Software decode at 3fps is cheap
        #     on the CPU and keeps the GPU's VRAM free for the live player.
        #   prefer_hwaccel=True → NVDEC. Right only if the GPU has VRAM
        #     headroom to spare (8GB+ card, fewer streams).
        # Pair this with feeding 1080p PROXIES (not 4K originals) as the
        # stream source — see main._start_deck_preview — for the biggest win.
        self._prefer_hwaccel = bool(prefer_hwaccel)
        # Locate ffmpeg up front. None = disabled, all start_stream calls
        # become no-ops (the HTTP layer will 503).
        self._ffmpeg_path: Optional[str] = (
            ffmpeg_path or shutil.which("ffmpeg")
        )
        if not self._ffmpeg_path:
            logger.warning(
                "PreviewStreamManager: ffmpeg not found on PATH — "
                "deck previews disabled. Install ffmpeg or add it to PATH."
            )
        else:
            logger.info(
                "PreviewStreamManager: %s decode for deck previews",
                "hardware (NVDEC)" if self._prefer_hwaccel else "software (CPU)",
            )
        # Set on stop_all() to refuse late start_stream() calls
        self._closed = False

    @property
    def available(self) -> bool:
        """True iff ffmpeg was found and we haven't been shut down."""
        return bool(self._ffmpeg_path) and not self._closed

    def has_stream(self, deck_idx: int) -> bool:
        with self._lock:
            return deck_idx in self._streams

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start_stream(
        self,
        deck_idx: int,
        filepath: str,
        in_sec: float = 0.0,
        out_sec: Optional[float] = None,
    ) -> bool:
        """Spawn (or replace) the ffmpeg pipeline for this deck slot.

        Returns True if a stream is now running for the slot, False
        otherwise (bad index, no ffmpeg, missing file, etc).
        """
        if not (0 <= deck_idx < MAX_STREAMS):
            logger.warning(f"start_stream: bad deck_idx {deck_idx}")
            return False
        if not self.available:
            return False
        if not filepath or not Path(filepath).exists():
            logger.warning(
                f"start_stream(deck={deck_idx}): missing file {filepath!r}"
            )
            return False

        with self._lock:
            old = self._streams.pop(deck_idx, None)
        # Stop the old one OUTSIDE the lock — its reader could be
        # blocked on a syscall, and we don't want to hold up new starts.
        if old is not None:
            old.stop()

        with self._lock:
            if self._closed:
                return False
            stream = _Stream(
                deck_idx=deck_idx,
                filepath=filepath,
                in_sec=in_sec,
                out_sec=out_sec,
                ffmpeg_path=self._ffmpeg_path,
                prefer_hwaccel=self._prefer_hwaccel,
            )
            self._streams[deck_idx] = stream
        stream.start()
        return True

    def stop_stream(self, deck_idx: int) -> None:
        """Stop the stream for this deck (no-op if none running)."""
        with self._lock:
            stream = self._streams.pop(deck_idx, None)
        if stream is not None:
            stream.stop()

    def stop_all(self) -> None:
        """Shutdown — kill every ffmpeg + wake every subscriber.
        After this returns, start_stream() is a no-op."""
        with self._lock:
            streams = list(self._streams.values())
            self._streams.clear()
            self._closed = True
        for s in streams:
            try:
                s.stop()
            except Exception as e:
                logger.debug(f"stop_all: stream {s.deck_idx} stop error: {e}")

    # ── Read API for HTTP handler ───────────────────────────────────────

    def get_latest_frame(self, deck_idx: int) -> Optional[bytes]:
        """Most recently produced JPEG, or None if no stream running yet."""
        with self._lock:
            stream = self._streams.get(deck_idx)
        if stream is None:
            return None
        return stream.get_latest_frame()

    def get_position(self, deck_idx: int) -> Optional[float]:
        """Where the preview for deck N currently is, in absolute file seconds.
        Returns None if no stream is running. Used by fire_deck to land the
        live cut on the same frame the user can see in the iPad preview."""
        with self._lock:
            stream = self._streams.get(deck_idx)
        if stream is None:
            return None
        return stream.get_position()

    def subscribe(self, deck_idx: int) -> Generator[bytes, None, None]:
        """Generator yielding JPEG bytes for the deck's stream until the
        stream is stopped or the consumer breaks out. Each yield is one
        complete JPEG, ready to write into a multipart part.

        The HTTP handler is expected to wrap this in a try/except so
        that a write failure (iPad disconnect) breaks the loop and
        cleanly unsubscribes us.
        """
        with self._lock:
            stream = self._streams.get(deck_idx)
        if stream is None:
            return
        q = stream.subscribe()
        try:
            while True:
                try:
                    item = q.get(timeout=SUBSCRIBER_TIMEOUT_SEC)
                except queue.Empty:
                    # Periodic wake — let the caller decide whether to
                    # keep waiting or bail. Most callers re-loop, but
                    # this lets stop_all() unblock cleanly even if no
                    # frame is ever produced.
                    with self._lock:
                        still_alive = deck_idx in self._streams
                    if not still_alive:
                        return
                    continue
                if item is _STOP_SENTINEL:
                    return
                if not isinstance(item, (bytes, bytearray)):
                    continue
                yield bytes(item)
        finally:
            try:
                stream.unsubscribe(q)
            except Exception:
                pass

    # ── Diagnostics (handy for tests / debug pages) ─────────────────────

    def list_active(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "deck": s.deck_idx,
                    "file": s.filepath,
                    "in_sec": s.in_sec,
                    "out_sec": s.out_sec,
                    "subscribers": s.subscriber_count(),
                    "hwaccel": s._using_hwaccel,
                }
                for s in self._streams.values()
            ]
