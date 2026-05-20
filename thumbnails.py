"""
Thumbnail generation for saved clips.

ffmpeg-based JPEG extraction at the clip's IN-point. Thumbnails live
under ~/.setpiece/thumbnails/<clip_id>.jpg and are served by
http_server.py.

Filmstrips (animated deck previews) live in the same directory as
strip_<hash>.jpg — vertically stacked frames evenly sampled across
[in_sec, out_sec]. CSS scrolls background-position-y to animate.

Defensive: if ffmpeg is missing or any extraction fails, log and move
on. The app must keep working without thumbnails.
"""

import hashlib
import logging
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# 2x retina for a 60x34 (16:9) display slot
THUMB_WIDTH = 120
THUMB_HEIGHT = 68
JPEG_QUALITY = 4  # ffmpeg -q:v scale: 2 (best) .. 31 (worst). 4 ~= JPEG q80.
FFMPEG_TIMEOUT_SEC = 5

# Filmstrip: stack of frames for animated deck previews
FILMSTRIP_FRAMES = 12
FILMSTRIP_FRAME_W = 120
FILMSTRIP_FRAME_H = 68
FILMSTRIP_TIMEOUT_SEC = 12   # longer overall budget for the whole strip job
FILMSTRIP_PER_FRAME_TIMEOUT = 4

THUMBS_DIR = Path.home() / ".setpiece" / "thumbnails"

# Cache the ffmpeg lookup; missing-ffmpeg should warn once, not every call.
_ffmpeg_path: Optional[str] = None
_ffmpeg_checked = False
_ffmpeg_lock = threading.Lock()


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
                "ffmpeg not found on PATH; clip thumbnails disabled. "
                "Install ffmpeg to enable visual previews on the iPad."
            )
        return _ffmpeg_path


def thumbnail_path(clip_id: str) -> Path:
    """Where this clip's thumbnail lives on disk."""
    return THUMBS_DIR / f"{clip_id}.jpg"


def ensure_thumbs_dir() -> None:
    """Idempotent mkdir for the thumbnails directory."""
    try:
        THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create thumbnails dir {THUMBS_DIR}: {e}")


def generate_thumbnail(
    video_filepath: str,
    time_seconds: float,
    output_path: str,
) -> bool:
    """Extract a single frame at `time_seconds` and save as JPEG.

    - Uses input-level fast seek (`-ss` BEFORE `-i`) which is dramatically
      faster than output-level seek for large files.
    - Software JPEG encode (NVENC overhead isn't worth it for 120x68).
    - Returns True on success, False on any failure.
    """
    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        return False

    src = Path(video_filepath)
    if not src.exists():
        logger.debug(f"Thumbnail skipped, source missing: {video_filepath}")
        return False

    out = Path(output_path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.debug(f"Could not create thumbnail parent dir: {e}")
        return False

    # Clamp negative seeks; ffmpeg tolerates 0 fine.
    seek = max(0.0, float(time_seconds))

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{seek:.3f}",   # fast seek BEFORE -i
        "-i", str(src),
        "-frames:v", "1",
        "-vf", f"scale={THUMB_WIDTH}:{THUMB_HEIGHT}:force_original_aspect_ratio=increase,"
                f"crop={THUMB_WIDTH}:{THUMB_HEIGHT}",
        "-q:v", str(JPEG_QUALITY),
        "-f", "image2",
        str(out),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=FFMPEG_TIMEOUT_SEC,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"Thumbnail timeout (>{FFMPEG_TIMEOUT_SEC}s) for {src.name} @ {seek:.2f}s")
        return False
    except Exception as e:
        logger.warning(f"Thumbnail subprocess failed for {src.name}: {e}")
        return False

    if result.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        err = result.stderr.decode("utf-8", errors="replace").strip() if result.stderr else ""
        logger.debug(f"Thumbnail ffmpeg rc={result.returncode} for {src.name}: {err}")
        # Cleanup any 0-byte stub
        try:
            if out.exists() and out.stat().st_size == 0:
                out.unlink()
        except Exception:
            pass
        return False

    return True


def generate_for_clip(clip: dict) -> bool:
    """Convenience: generate a thumbnail for a clip dict, IN-point time."""
    cid = clip.get("id")
    fp = clip.get("filepath")
    t = clip.get("in_seconds", 0.0)
    if not cid or not fp:
        return False
    return generate_thumbnail(fp, float(t), str(thumbnail_path(cid)))


def delete_thumbnail(clip_id: str) -> None:
    """Delete a clip's thumbnail. Silent on missing/failure."""
    if not clip_id:
        return
    try:
        p = thumbnail_path(clip_id)
        if p.exists():
            p.unlink()
    except Exception as e:
        logger.debug(f"Could not delete thumbnail {clip_id}: {e}")


def backfill_async(clips: Iterable[dict]) -> threading.Thread:
    """Start a background thread that generates any missing thumbnails.

    Runs as a daemon so it never blocks shutdown. Idempotent: skips clips
    whose thumbnail already exists. Safe to call on every startup.
    """
    snapshot = list(clips)  # copy now; caller's list may mutate

    def worker():
        try:
            ensure_thumbs_dir()
            missing = [c for c in snapshot
                       if c.get("id") and not thumbnail_path(c["id"]).exists()]
            if not missing:
                return
            logger.info(f"Backfilling {len(missing)} clip thumbnail(s) in background...")
            ok = 0
            for clip in missing:
                if generate_for_clip(clip):
                    ok += 1
            logger.info(f"Thumbnail backfill done: {ok}/{len(missing)} succeeded")
        except Exception as e:
            logger.warning(f"Thumbnail backfill crashed: {e}")

    t = threading.Thread(target=worker, name="thumbnail-backfill", daemon=True)
    t.start()
    return t


# ── Filmstrip (animated deck previews) ─────────────────────────────────

def filmstrip_hash(filepath: str, in_sec: float, out_sec: float) -> str:
    """Stable 16-char hash for a (file, in, out) triple. Same input → same
    hash forever, so the cached strip JPEG can be reused across restarts."""
    raw = f"{Path(filepath).resolve()}|{float(in_sec):.3f}|{float(out_sec):.3f}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def filmstrip_path(strip_hash: str) -> Path:
    """Where this filmstrip lives on disk."""
    return THUMBS_DIR / f"strip_{strip_hash}.jpg"


def _extract_one_frame(video_filepath: str, t_sec: float, out_path: Path) -> bool:
    """Extract a single frame at t_sec into out_path (a JPEG). Used by the
    filmstrip parallel worker. Returns True on success."""
    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        return False
    seek = max(0.0, float(t_sec))
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{seek:.3f}",       # fast input-level seek (per CLAUDE.md)
        "-i", str(video_filepath),
        "-frames:v", "1",
        "-vf", f"scale={FILMSTRIP_FRAME_W}:{FILMSTRIP_FRAME_H}:"
               f"force_original_aspect_ratio=increase,"
               f"crop={FILMSTRIP_FRAME_W}:{FILMSTRIP_FRAME_H}",
        "-q:v", str(JPEG_QUALITY),
        "-f", "image2",
        str(out_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=FILMSTRIP_PER_FRAME_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return False
    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        try:
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink()
        except Exception:
            pass
        return False
    return True


def _stack_frames_vertically(frame_paths: list[Path], output_path: Path) -> bool:
    """Use ffmpeg's vstack to merge N small JPEGs into one tall filmstrip JPEG.
    Returns True on success."""
    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        return False
    if not frame_paths:
        return False
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for fp in frame_paths:
        cmd += ["-i", str(fp)]
    if len(frame_paths) == 1:
        # vstack needs >=2 inputs; for a degenerate single-frame strip,
        # just copy.
        cmd += ["-frames:v", "1", "-q:v", str(JPEG_QUALITY), str(output_path)]
    else:
        cmd += [
            "-filter_complex", f"vstack=inputs={len(frame_paths)}",
            "-frames:v", "1",
            "-q:v", str(JPEG_QUALITY),
            "-f", "image2",
            str(output_path),
        ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=FILMSTRIP_TIMEOUT_SEC,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as e:
        logger.debug(f"vstack failed: {e}")
        return False
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        return False
    return True


def generate_filmstrip(
    video_filepath: str,
    in_sec: float,
    out_sec: float,
    output_path: str,
    frame_count: int = FILMSTRIP_FRAMES,
) -> bool:
    """Build an animated-thumbnail filmstrip for [in_sec, out_sec].

    - Extracts `frame_count` evenly-spaced frames using parallel ffmpeg
      processes (input-level fast seek per file).
    - Stacks them vertically into one JPEG at output_path, total size
      FILMSTRIP_FRAME_W × (FILMSTRIP_FRAME_H × frame_count).
    - Idempotent at the caller level (same hash → same path); will overwrite
      if called again.
    - Returns True on success.
    """
    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        return False
    src = Path(video_filepath)
    if not src.exists():
        logger.debug(f"Filmstrip skipped, source missing: {video_filepath}")
        return False
    out = Path(output_path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.debug(f"Could not create filmstrip parent dir: {e}")
        return False

    in_sec = max(0.0, float(in_sec))
    out_sec = float(out_sec)
    if out_sec <= in_sec:
        # Caller didn't know the duration; fall back to a tiny static strip
        # of the IN frame so the deck still has SOMETHING to show.
        out_sec = in_sec + 0.01
    duration = out_sec - in_sec

    n = max(1, int(frame_count))
    if n == 1:
        times = [in_sec]
    else:
        # Sample at frame midpoints across [in, out]. Skip exactly 0 and
        # exactly out to dodge VFR edge cases that often return blank frames.
        step = duration / n
        times = [in_sec + step * (i + 0.5) for i in range(n)]

    # Stage frames in a temp folder next to the strip
    tmp_dir = out.parent / f"_strip_tmp_{out.stem}"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False

    frame_paths: list[Path] = []
    try:
        # Parallel extraction (per CLAUDE.md note about ffmpeg concat being slow)
        with ThreadPoolExecutor(max_workers=min(4, n)) as pool:
            futures = []
            for i, t in enumerate(times):
                fp = tmp_dir / f"f{i:02d}.jpg"
                frame_paths.append(fp)
                futures.append(pool.submit(_extract_one_frame, str(src), t, fp))
            results = [f.result() for f in futures]

        # Keep only the frames that actually rendered. If too few, bail.
        good = [fp for fp, ok in zip(frame_paths, results) if ok and fp.exists()]
        if not good:
            logger.debug(f"Filmstrip: 0/{n} frames extracted for {src.name}")
            return False
        if len(good) < n:
            logger.debug(f"Filmstrip: only {len(good)}/{n} frames for {src.name}; padding")
            # Pad by repeating the last good frame so the CSS keyframe count
            # matches FILMSTRIP_FRAMES across the UI.
            good = good + [good[-1]] * (n - len(good))

        ok = _stack_frames_vertically(good, out)
        if not ok:
            logger.debug(f"Filmstrip vstack failed for {src.name}")
            return False
        return True
    finally:
        # Clean up scratch frames
        for fp in frame_paths:
            try:
                if fp.exists():
                    fp.unlink()
            except Exception:
                pass
        try:
            tmp_dir.rmdir()
        except Exception:
            pass


def generate_filmstrip_for_deck(deck: dict) -> bool:
    """Convenience: generate a filmstrip given a deck dict with filepath /
    in_sec / out_sec / strip_hash. Returns True on success."""
    fp = deck.get("filepath")
    if not fp:
        return False
    in_s = float(deck.get("in_sec") or 0.0)
    out_s = float(deck.get("out_sec") or 0.0)
    h = deck.get("strip_hash") or filmstrip_hash(fp, in_s, out_s)
    return generate_filmstrip(fp, in_s, out_s, str(filmstrip_path(h)))


def filmstrip_backfill_async(decks: Iterable[dict]) -> threading.Thread:
    """Background thread: generate any missing filmstrips for the given
    deck snapshot. Daemon, idempotent."""
    snapshot = [d for d in decks if d]

    def worker():
        try:
            ensure_thumbs_dir()
            missing = [d for d in snapshot
                       if d.get("strip_hash")
                       and not filmstrip_path(d["strip_hash"]).exists()]
            if not missing:
                return
            logger.info(f"Backfilling {len(missing)} filmstrip(s) in background...")
            ok = 0
            for d in missing:
                if generate_filmstrip_for_deck(d):
                    ok += 1
            logger.info(f"Filmstrip backfill done: {ok}/{len(missing)} succeeded")
        except Exception as e:
            logger.warning(f"Filmstrip backfill crashed: {e}")

    t = threading.Thread(target=worker, name="filmstrip-backfill", daemon=True)
    t.start()
    return t


# Filmstrip ID validation regex used by HTTP route. Plain hex, length 16.
import re as _re
FILMSTRIP_ID_RE = _re.compile(r"^[0-9a-fA-F]{16}$")

# Library-file thumbnail validation. Same hex/length shape as filmstrips,
# but a separate route + filename prefix so they can never collide with
# clip thumbnails (which are uuids) on disk or in URL space.
LIB_THUMB_ID_RE = _re.compile(r"^[0-9a-fA-F]{16}$")

# Library thumbs are extracted at this fraction of the file's duration.
# 5% lands past the typical fade-in / black slug and usually gives a
# decent "title card" frame. Falls back to a fixed offset if duration
# probing fails.
LIB_THUMB_FRACTION = 0.05
LIB_THUMB_FALLBACK_SEC = 5.0
LIB_THUMB_PROBE_TIMEOUT = 3
# Library files larger than this are skipped by the on-demand generator
# AND the backfill — long files turn ffmpeg fast-seek into a slow seek
# and stall the request thread / starve the backfill loop.
LIB_THUMB_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


def lib_thumbnail_hash(filepath: str) -> str:
    """Stable 16-char hash of an absolute filepath, used as the cache key
    for library-file thumbnails. Filename-only (no mtime) so renaming the
    file invalidates the cache; old thumb just becomes orphaned.
    """
    raw = str(Path(filepath).resolve())
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def lib_thumbnail_path(file_hash: str) -> Path:
    """Where this library file's thumbnail lives on disk.
    `lib_` prefix prevents collision with clip thumbs (`<uuid>.jpg`)."""
    return THUMBS_DIR / f"lib_{file_hash}.jpg"


def _probe_duration_seconds(filepath: str) -> float:
    """Best-effort duration probe via ffprobe (if installed).
    Returns 0.0 on any failure — caller should fall back to a fixed
    offset. Cheap (subprocess + tiny output) but still gated by a short
    timeout."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", filepath],
            capture_output=True,
            timeout=LIB_THUMB_PROBE_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return 0.0
    try:
        return float((r.stdout or b"").decode("utf-8", "replace").strip() or 0.0)
    except Exception:
        return 0.0


def generate_library_thumbnail(filepath: str) -> Optional[str]:
    """Extract a frame ~5% into the file (good "title card" usually) and
    save as a 120x68 JPEG under THUMBS_DIR. Returns the output path on
    success, or None on any failure (missing ffmpeg, oversized file,
    extraction error, etc).

    Idempotent: caller is expected to check `lib_thumbnail_path(...).exists()`
    first and skip if cached. We still re-extract if invoked, mostly so
    on-demand HTTP requests for a missing thumb just work.
    """
    if not filepath:
        return None
    src = Path(filepath)
    if not src.exists() or not src.is_file():
        return None
    try:
        size = src.stat().st_size
    except OSError:
        return None
    if size <= 0 or size > LIB_THUMB_MAX_BYTES:
        # Skip absurdly-large files (often broken or hours-long) — fast
        # seek doesn't save us at that scale.
        logger.debug(f"Lib thumb skipped (size={size}): {src.name}")
        return None

    ensure_thumbs_dir()
    out = lib_thumbnail_path(lib_thumbnail_hash(str(src)))

    duration = _probe_duration_seconds(str(src))
    if duration > 0:
        seek = duration * LIB_THUMB_FRACTION
    else:
        seek = LIB_THUMB_FALLBACK_SEC

    if generate_thumbnail(str(src), seek, str(out)):
        return str(out)
    return None
