"""
CLIP semantic embedder.

For every video in the library, sample 5 keyframes (10/30/50/70/90%
timestamps), push them through the CLIP image encoder as one batch,
average the per-frame embeddings, L2-normalize, and store the resulting
512-dim float32 vector in a new SQLite table `clip_embeddings`.

WHY THIS EXISTS
---------------
Tag-based search ("twerk", "dance") only finds clips that already have
the right token in the path or have already been tagged by a specific
detector. CLIP embeddings let us do natural-language search across the
whole library by VISUAL MEANING — no string-match required. Once every
file has a vector, a query like "girl twerking outdoors" is a single
nearest-neighbour lookup in 512-dim space (see clip_search.py).

MODEL
-----
openai/clip-vit-base-patch32 via the open_clip_torch library.
- ~150 MB weights, 512-dim output, 224x224 input.
- ViT-B/32 on a GTX 1650 (4 GB) does ~20-30 img/sec at batch=32, so a
  full library of 5678 files * 5 frames = 28k images is ~10-20 minutes
  of pure GPU time, plus disk decode.
- Falls back to CPU with a printed warning if cuda is unavailable.

STORAGE
-------
NEW table:
    clip_embeddings(
        filepath TEXT PRIMARY KEY,
        vec      BLOB,        -- 512 * float32 = 2048 bytes, L2-normalised
        model    TEXT,        -- e.g. "ViT-B-32/openai"
        updated  REAL
    )

We keep this in the same path_tags.db3 file the rest of the project
already uses; sqlite handles concurrent readers/writers via WAL +
busy_timeout=30s.

IDEMPOTENCY
-----------
A file is skipped if it already has a row in clip_embeddings (and the
model column matches). `--force` deletes the existing row before
re-embedding.

USAGE
-----
    # Quick sanity-test on 5 files
    python clip_tagger.py --limit 5

    # Full library, single worker (GPU is the bottleneck)
    python clip_tagger.py

    # Only one folder
    python clip_tagger.py --root "/path/to/clips"

    # Re-embed already-embedded files
    python clip_tagger.py --force

    # How many embeddings do we have?
    python clip_tagger.py --list
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

APP_STATE = Path.home() / ".setpiece"
DB_PATH = APP_STATE / "path_tags.db3"

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

# ── tuning knobs ───────────────────────────────────────────────────────
N_KEYFRAMES = 5
KEYFRAME_PCTS = (0.10, 0.30, 0.50, 0.70, 0.90)
MAX_FILE_BYTES = 32_000_000_000     # 32 GB sanity cap. Was 1.2 GB
# (an earlier filter) but that excluded ~592 large 4K files
# from semantic search for no real reason: CLIP only samples 5
# keyframes, so embed cost is seek-bound, not size-bound — measured
# 2.3s to embed a 24 GB file. (2026-05-20)
MAX_FRAMES_READ = 6000              # cap cap.read() calls per file
COMMIT_EVERY = 10                   # WAL commits every N processed files

# ── model identifier ───────────────────────────────────────────────────
# open_clip names this model "ViT-B-32" with the "openai" pretrained tag.
# 512-dim image embedding.
MODEL_NAME = "ViT-B-32"
MODEL_PRETRAINED = "openai"
MODEL_TAG = f"{MODEL_NAME}/{MODEL_PRETRAINED}"
EMBED_DIM = 512


# ── DB helpers ─────────────────────────────────────────────────────────


def _open_db() -> sqlite3.Connection:
    """Open the path_tags.db3 with WAL + busy_timeout for coexistence."""
    conn = sqlite3.connect(str(DB_PATH), timeout=60.0)
    # busy_timeout MUST be set before any operation that takes a lock,
    # otherwise the default 0ms wait fires immediately. Set it first.
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _with_retry(fn, attempts: int = 8, base_sleep: float = 1.0):
    """Retry helper for "database is locked" — the parallel pose tagger
    can hold long write transactions which push us past busy_timeout."""
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            last_exc = e
            if "locked" not in str(e).lower():
                raise
            sleep_for = base_sleep * (i + 1)
            logger.warning(
                f"DB locked, retrying in {sleep_for:.1f}s (attempt {i + 1}/{attempts})"
            )
            time.sleep(sleep_for)
    raise last_exc  # type: ignore[misc]


def _ensure_schema(conn: sqlite3.Connection) -> None:
    def _do():
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clip_embeddings (
                filepath TEXT PRIMARY KEY,
                vec      BLOB NOT NULL,
                model    TEXT NOT NULL,
                updated  REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_clip_model ON clip_embeddings(model)"
        )
        conn.commit()

    _with_retry(_do)


# ── keyframe extraction ────────────────────────────────────────────────


def _extract_keyframes(filepath: str) -> list[np.ndarray] | None:
    """Open the video, grab N_KEYFRAMES BGR uint8 frames at the
    configured timestamps. Returns None on decode failure / unusable
    file. Each frame is raw — preprocessing happens on GPU side."""
    try:
        if os.path.getsize(filepath) > MAX_FILE_BYTES:
            return None
    except OSError:
        return None
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0.1 or n_frames <= 0:
            return None
        duration = n_frames / fps
        if duration <= 0.5:
            return None
        frames: list[np.ndarray] = []
        for pct in KEYFRAME_PCTS:
            target_ms = duration * pct * 1000.0
            cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
            ok, frame = cap.read()
            # Some codecs ignore POS_MSEC. Fall back to a frame-index
            # seek as a second try before giving up on this keyframe.
            if not ok or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(n_frames * pct))
                ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        if not frames:
            return None
        return frames
    finally:
        cap.release()


def _preprocess_for_clip(
    frames: list[np.ndarray], preprocess
) -> "torch.Tensor":
    """Run the open_clip preprocess transform on each frame and stack
    them into a single (N, 3, 224, 224) tensor ready for the GPU.
    open_clip's preprocess expects a PIL.Image."""
    import torch
    from PIL import Image

    tensors = []
    for bgr in frames:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensors.append(preprocess(pil))
    return torch.stack(tensors, dim=0)


# ── model loader (cached) ──────────────────────────────────────────────


_MODEL_CACHE: dict = {}


def load_clip_model(device: str | None = None):
    """Load + cache the CLIP image encoder + its preprocess transform.
    Returns (model, preprocess, device_used)."""
    import torch
    import open_clip

    if "model" in _MODEL_CACHE:
        return (
            _MODEL_CACHE["model"],
            _MODEL_CACHE["preprocess"],
            _MODEL_CACHE["device"],
        )

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("WARNING: CUDA not available, falling back to CPU "
                  "(much slower)", file=sys.stderr)
            device = "cpu"

    model, _train_preprocess, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=MODEL_PRETRAINED
    )
    model = model.to(device)
    model.eval()
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["preprocess"] = preprocess
    _MODEL_CACHE["device"] = device
    return model, preprocess, device


def embed_video(filepath: str, model, preprocess, device: str) -> np.ndarray | None:
    """Full per-file pipeline: extract keyframes -> preprocess -> CLIP
    image encode -> average -> L2-normalize. Returns float32 (512,) on
    success, None on failure."""
    import torch

    frames = _extract_keyframes(filepath)
    if not frames:
        return None
    try:
        batch = _preprocess_for_clip(frames, preprocess).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)  # (N, 512), float32
            # Average per-frame embeddings -> single per-clip vector.
            mean = feats.mean(dim=0)
            mean = mean / (mean.norm() + 1e-9)
        vec = mean.detach().cpu().to(torch.float32).numpy()
        if vec.shape[0] != EMBED_DIM:
            logger.warning(
                f"unexpected embedding dim {vec.shape[0]} for {filepath}"
            )
            return None
        return vec.astype(np.float32, copy=False)
    except Exception as e:
        logger.debug(f"embed failed for {filepath}: {e}")
        return None


# ── batch driver ───────────────────────────────────────────────────────


def _candidate_files(
    cur: sqlite3.Cursor, root: str | None, force: bool, limit: int
) -> list[str]:
    if root:
        like_a = root.replace("/", "\\") + "%"
        like_b = root.replace("\\", "/") + "%"
        rows = cur.execute(
            "SELECT filepath FROM files WHERE filepath LIKE ? OR filepath LIKE ? "
            "ORDER BY filepath",
            (like_a, like_b),
        ).fetchall()
    else:
        rows = cur.execute(
            "SELECT filepath FROM files ORDER BY filepath"
        ).fetchall()
    candidates = [r[0] for r in rows]

    if not force:
        already = {
            r[0]
            for r in cur.execute(
                "SELECT filepath FROM clip_embeddings WHERE model = ?",
                (MODEL_TAG,),
            ).fetchall()
        }
        before = len(candidates)
        candidates = [fp for fp in candidates if fp not in already]
        logger.info(
            f"already embedded (skipping): {before - len(candidates)}"
        )

    if limit:
        candidates = candidates[:limit]
    return candidates


def tag_library(
    root: str | None = None,
    force: bool = False,
    limit: int = 0,
    workers: int = 1,
) -> dict:
    if not DB_PATH.is_file():
        logger.error(f"no DB at {DB_PATH}")
        return {"ok": False, "error": "no db"}

    conn = _open_db()
    _ensure_schema(conn)
    cur = conn.cursor()

    if force:
        def _drop():
            return cur.execute(
                "DELETE FROM clip_embeddings WHERE model = ?", (MODEL_TAG,)
            ).rowcount
        n = _with_retry(_drop)
        _with_retry(conn.commit)
        logger.info(f"force: dropped {n} existing embeddings for {MODEL_TAG}")

    candidates = _candidate_files(cur, root, force, limit)
    total = len(candidates)
    logger.info(f"will embed: {total} files (workers={workers})")
    if total == 0:
        conn.close()
        return {"ok": True, "embedded": 0, "failed": 0}

    model, preprocess, device = load_clip_model()
    logger.info(f"loaded {MODEL_TAG} on {device}")

    n_embedded = 0
    n_failed = 0
    t0 = time.time()

    def _process(fp: str):
        # Decode happens on CPU and releases the GIL inside cv2; the
        # GPU encode is serial (single CUDA context). With workers > 1
        # we get overlapping decode with GPU work, but the GPU is the
        # bottleneck so default is 1.
        try:
            if not Path(fp).is_file():
                return fp, None
            return fp, embed_video(fp, model, preprocess, device)
        except Exception as e:
            logger.debug(f"process failed for {fp}: {e}")
            return fp, None

    # workers == 1 is the common path; keep a serial loop so we don't
    # eat thread-pool overhead on the GPU bottleneck case.
    if workers <= 1:
        for i, fp in enumerate(candidates, 1):
            _, vec = _process(fp)
            if vec is None:
                n_failed += 1
                if i % 25 == 0:
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (total - i) / rate / 60 if rate > 0 else 0
                    logger.info(
                        f"  [{i}/{total}] {n_embedded} ok / {n_failed} fail  "
                        f"({rate:.2f}/s, ~{eta:.1f}min remaining)"
                    )
                continue
            def _insert(fp=fp, vec=vec):
                cur.execute(
                    "INSERT OR REPLACE INTO clip_embeddings "
                    "(filepath, vec, model, updated) VALUES (?, ?, ?, ?)",
                    (fp, vec.tobytes(), MODEL_TAG, time.time()),
                )
            _with_retry(_insert)
            n_embedded += 1
            # Encoding-safe print: Windows console default cp1252 chokes
            # on chars like fullwidth-colon U+FF1A in filenames. Force
            # ASCII-replace so the print never crashes the tagger.
            name = Path(fp).name[:80].encode("ascii", "replace").decode("ascii")
            print(f"[{i}/{total}] OK {name}")
            if i % COMMIT_EVERY == 0:
                _with_retry(conn.commit)
            if i % 25 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate / 60 if rate > 0 else 0
                logger.info(
                    f"  [{i}/{total}] {n_embedded} ok / {n_failed} fail  "
                    f"({rate:.2f}/s, ~{eta:.1f}min remaining)"
                )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process, fp) for fp in candidates]
            for i, fut in enumerate(as_completed(futures), 1):
                fp, vec = fut.result()
                if vec is None:
                    n_failed += 1
                    continue
                def _insert(fp=fp, vec=vec):
                    cur.execute(
                        "INSERT OR REPLACE INTO clip_embeddings "
                        "(filepath, vec, model, updated) VALUES (?, ?, ?, ?)",
                        (fp, vec.tobytes(), MODEL_TAG, time.time()),
                    )
                _with_retry(_insert)
                n_embedded += 1
                # Encoding-safe print (Windows cp1252 chokes on
                # fullwidth-colon etc).
                name = Path(fp).name[:80].encode(
                    "ascii", "replace").decode("ascii")
                print(f"[{i}/{total}] OK {name}")
                if i % COMMIT_EVERY == 0:
                    _with_retry(conn.commit)
                if i % 25 == 0:
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (total - i) / rate / 60 if rate > 0 else 0
                    logger.info(
                        f"  [{i}/{total}] {n_embedded} ok / "
                        f"{n_failed} fail ({rate:.2f}/s, "
                        f"~{eta:.1f}min remaining)"
                    )

    _with_retry(conn.commit)
    conn.close()
    elapsed = time.time() - t0
    logger.info("=" * 56)
    logger.info(
        f"DONE: {n_embedded} embedded / {n_failed} failed in {elapsed:.1f}s"
    )
    if n_embedded:
        logger.info(f"  avg {elapsed / max(1, n_embedded):.2f}s/file")
    return {
        "ok": True,
        "embedded": n_embedded,
        "failed": n_failed,
        "elapsed": elapsed,
    }


def list_embeddings() -> None:
    """Print quick stats on existing embeddings."""
    if not DB_PATH.is_file():
        print("no DB at", DB_PATH)
        return
    conn = _open_db()
    _ensure_schema(conn)
    cur = conn.cursor()
    total = cur.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()[0]
    by_model = cur.execute(
        "SELECT model, COUNT(*) FROM clip_embeddings GROUP BY model"
    ).fetchall()
    n_files = cur.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    print(f"clip_embeddings rows: {total}  (library files: {n_files})")
    for m, n in by_model:
        print(f"  {n:6d}  {m}")
    conn.close()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", default=None,
                    help="restrict to files under this folder")
    ap.add_argument("--force", action="store_true",
                    help="drop existing embeddings first, then re-embed")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N files (0 = all)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel decode workers (default 1; GPU is the bottleneck)")
    ap.add_argument("--list", action="store_true",
                    help="just print current embedding counts and exit")
    args = ap.parse_args()

    if args.list:
        list_embeddings()
        return 0

    r = tag_library(
        root=args.root,
        force=args.force,
        limit=args.limit,
        workers=args.workers,
    )
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
