"""
Geometry tagger — classify the visual STRUCTURE of each clip into one
of the four VJ-canonical buckets and store as tags.

WHY THIS EXISTS
---------------
Per VJ_ROADMAP.md / the PDF (page 11): geometry determines layering &
blend modes. A "particles" clip blends differently than a "solid
polygon" clip (additive vs normal). Pros tag by structural form so
they can quickly grab "the right kind of look" for a layer slot.

THE FIVE BUCKETS
----------------
  geometry:particles    - starfields, dots, isolated light points,
                          sparse high-contrast on dark bg. Loves
                          additive blending.
  geometry:linear       - horizontal "liquid-sky" looks, vertical
                          columns, beams. Directs the eye along axes.
  geometry:polygons     - complex 3D rendered cubes, fractals, solid
                          heavy geometry. Dense, dominant. ABSTRACT.
  geometry:alpha-mask   - black-and-white clips used as luma-keys for
                          masking other layers (high contrast, mostly
                          binary, no grey midtones).
  geometry:photographic - real-world footage: figures, faces, dance,
                          performance. Midtone-dominated, high edge
                          density, no clear abstract structure. Set
                          aside from "polygons" so the polygon bucket
                          stays reserved for ACTUAL polygon-style
                          abstract content -- otherwise photographic
                          content collapses into polygons by default
                          and drowns out the abstract signal.

HOW IT CLASSIFIES (lightweight heuristic, no ML)
-----------------
For each clip, sample 3 frames from the body. Per frame compute:

  1. Edge density (Sobel) -> portion of pixels above edge threshold
  2. Dark-pixel fraction  -> portion of pixels under brightness floor
  3. Pixel histogram bimodality -> are the brightness values clustered
     at 0 and 255 (= alpha-mask) or spread continuously (= polygon)?
  4. Edge direction histogram peakiness -> dominant axis (= linear)
     vs isotropic edges (= polygons or particles)
  5. Connected-component-like sparsity proxy: variance of local
     brightness in 16x16 tiles (high variance + sparse edges =
     particles; uniform tiles = mask or solid)

Buckets are assigned by a small rule cascade — no ML model. The PDF
explicitly calls this "machine learning algorithms that scan ingested
video files to automatically assign these tags based on pixel density,
color averages, and edge detection" — pixel/edge heuristics get us 80%
of the way without the ML dep tree.

USAGE
-----
    python geometry_tagger.py --dry-run --limit 10
    python geometry_tagger.py                 # whole library
    python geometry_tagger.py --refresh       # re-tag everything

PERFORMANCE
-----------
~400-700ms per file. ~3927 files = ~30-45 min single-threaded.

DESIGN NOTES
------------
- Sample at 64x64 like the other taggers. Edges and bimodality are
  preserved at that resolution.
- Sobel via numpy convolution (no scipy dep).
- The four buckets are MUTUALLY EXCLUSIVE: each clip gets exactly ONE
  geometry tag. (Unlike motion which gets independent static/dynamic
  + smooth/jumpy + complexity tags.) The picker uses geometry as a
  hard-constraint filter, not a soft signal.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

APP_STATE = Path.home() / ".setpiece"
DB_PATH = APP_STATE / "path_tags.db3"
SAMPLE_PX = 64
N_FRAMES = 3

# Tag prefix
_GEO_PREFIX = "geometry:"

# Sobel kernels (3x3)
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def _conv2d(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """3x3 convolution via shifted sums. No scipy dep."""
    H, W = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    for di in range(3):
        for dj in range(3):
            kij = k[di, dj]
            if kij == 0:
                continue
            sh = np.roll(img, (di - 1, dj - 1), axis=(0, 1))
            out += kij * sh
    return out


def _probe_duration(filepath: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return float(r.stdout.strip() or 0.0)
    except Exception:
        return 0.0


def _sample_grey(filepath: str) -> np.ndarray | None:
    """Pull N_FRAMES greyscale frames spread across body. Returns
    (N, 64, 64) uint8 or None.

    Uses N independent fast-seek ffmpeg calls. The single-call fps=X
    filter approach hangs on long videos (fps=0.003 on a 1hr file
    forces full decode), so we do N small seek-and-extract calls
    instead -- same fix as motion_tagger._sample_frames.
    """
    dur = _probe_duration(filepath)
    if dur <= 1.0:
        return None
    start = dur * 0.15
    end = dur * 0.85
    if end - start < 0.5:
        return None
    timestamps = np.linspace(start, end, N_FRAMES)
    bp = SAMPLE_PX * SAMPLE_PX
    frames: list[np.ndarray] = []
    for ts in timestamps:
        try:
            r = subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error",
                 "-ss", f"{float(ts):.2f}", "-i", filepath,
                 "-frames:v", "1",
                 "-vf", f"scale={SAMPLE_PX}:{SAMPLE_PX},format=gray",
                 "-pix_fmt", "gray", "-f", "rawvideo", "-"],
                capture_output=True, timeout=8,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            raw = r.stdout
        except Exception:
            continue
        if len(raw) >= bp:
            frames.append(np.frombuffer(raw[:bp], dtype=np.uint8)
                          .reshape(SAMPLE_PX, SAMPLE_PX))
    if not frames:
        return None
    return np.stack(frames, axis=0)


def _classify(frames: np.ndarray) -> str:
    """Return the chosen geometry bucket string (without prefix)."""
    if frames is None or len(frames) == 0:
        return "polygons"  # fallback

    # Average frame for stable stats
    avg = frames.mean(axis=0).astype(np.float32)

    # ── feature 1: dark fraction ───────────────────────────────────────
    dark_frac = float((avg < 24).mean())

    # ── feature 2: alpha-mask-ness (bimodal: lots of pure-dark and
    # pure-bright pixels, very few midtones) ──────────────────────────
    hist, _ = np.histogram(avg, bins=8, range=(0, 256))
    dark_count = hist[0]
    bright_count = hist[-1]
    mid_count = hist[2:6].sum()
    total = hist.sum()
    bimodal_score = float((dark_count + bright_count) / max(1, total))
    mid_score = float(mid_count / max(1, total))

    # ── feature 3: edge density via Sobel magnitude ───────────────────
    gx = _conv2d(avg, _SOBEL_X)
    gy = _conv2d(avg, _SOBEL_Y)
    mag = np.sqrt(gx * gx + gy * gy)
    edge_frac = float((mag > 40.0).mean())

    # ── feature 4: edge direction histogram → "linear" if dominated by
    # horizontal or vertical edges, "polygons/particles" if isotropic.
    # Cheap proxy: ratio of |gx| total vs |gy| total. If one axis has
    # 3x more energy, dominant direction.
    sum_gx = float(np.abs(gx).sum())
    sum_gy = float(np.abs(gy).sum())
    if sum_gx + sum_gy > 0:
        axial_ratio = max(sum_gx, sum_gy) / max(1.0, min(sum_gx, sum_gy))
    else:
        axial_ratio = 1.0

    # ── feature 5: tile variance (particles = bright dots on dark =
    # high inter-tile variance + sparse edges; solid polygons = uniform
    # tiles + dense edges) ────────────────────────────────────────────
    tiles = avg.reshape(4, 16, 4, 16).transpose(0, 2, 1, 3).reshape(16, 256)
    tile_means = tiles.mean(axis=1)
    inter_tile_var = float(tile_means.var())

    # ── classification rules (cascade) ─────────────────────────────────
    # 1. Alpha-mask: high bimodal_score + low mid_score (pure b/w)
    if bimodal_score > 0.55 and mid_score < 0.20:
        return "alpha-mask"
    # 2. Linear: strong axial dominance + meaningful edge density
    if axial_ratio > 2.2 and edge_frac > 0.05:
        return "linear"
    # 3. Particles: dark dominant frame + sparse-but-bright high-variance tiles
    if dark_frac > 0.45 and edge_frac < 0.12 and inter_tile_var > 100:
        return "particles"
    # 4. Photographic: midtone-dominated dense-edge real-world footage
    #    (figures / dance / performance). MUST come before the polygons
    #    fallback or photographic content collapses there and drowns
    #    out the genuine-polygons signal. Criteria:
    #      - very few dark pixels (dk < 0.20)  -- not a dark scene
    #      - very few bright/bimodal (bim < 0.15) -- not high-contrast graphic
    #      - heavy midtones (mid > 0.65) -- fleshy/natural color range
    #      - dense edges (edge > 0.30) -- handheld bodies = lots of contours
    if (dark_frac < 0.20 and bimodal_score < 0.15
            and mid_score > 0.65 and edge_frac > 0.30):
        return "photographic"
    # 5. Default: polygons (dense / solid / complex abstract content)
    return "polygons"


def tag_library(
    root: str | None = None,
    dry_run: bool = False,
    refresh: bool = False,
    limit: int = 0,
) -> dict:
    if not DB_PATH.is_file():
        logger.error(f"no DB at {DB_PATH}")
        return {"ok": False, "error": "no db"}

    if not dry_run and not refresh:
        bak = DB_PATH.with_suffix(".db3.bak.geometry_tagger")
        shutil.copyfile(DB_PATH, bak)
        logger.info(f"DB backup -> {bak}")

    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    cur = conn.cursor()
    # WAL mode + busy timeout so we can run concurrently with the
    # symmetry tagger (or main app) without "database is locked".
    try:
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
    except sqlite3.Error:
        pass

    if root:
        like = root.replace("/", "\\") + "%"
        rows = cur.execute(
            "SELECT filepath FROM files WHERE filepath LIKE ? ORDER BY filepath",
            (like,)
        ).fetchall()
    else:
        rows = cur.execute(
            "SELECT filepath FROM files ORDER BY filepath"
        ).fetchall()
    candidates = [r[0] for r in rows]

    if refresh:
        if not dry_run:
            n = cur.execute(
                "DELETE FROM file_tags WHERE tag LIKE 'geometry:%'"
            ).rowcount
            conn.commit()
            logger.info(f"refresh: dropped {n} existing geometry tags")
    else:
        tagged = set(r[0] for r in cur.execute(
            "SELECT DISTINCT filepath FROM file_tags WHERE tag LIKE 'geometry:%'"
        ).fetchall())
        before = len(candidates)
        candidates = [fp for fp in candidates if fp not in tagged]
        logger.info(f"already tagged (skipping): {before - len(candidates)}")

    if limit:
        candidates = candidates[:limit]
    logger.info(f"will process: {len(candidates)} files")

    n_done = 0
    n_failed = 0
    n_inserted = 0
    bucket_counts: dict = {}
    t0 = time.time()
    for i, fp in enumerate(candidates, 1):
        if not Path(fp).is_file():
            n_failed += 1
            continue
        frames = _sample_grey(fp)
        if frames is None:
            n_failed += 1
            continue
        bucket = _classify(frames)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if not dry_run:
            tag = _GEO_PREFIX + bucket
            cur.execute(
                "INSERT OR IGNORE INTO file_tags(filepath, tag) VALUES (?, ?)",
                (fp, tag)
            )
            if cur.rowcount:
                n_inserted += 1
            # Commit per insert -- short lock windows. Same fix as
            # symmetry_tagger so concurrent taggers don't time out
            # contending for the WAL writer lock.
            conn.commit()
        n_done += 1
        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(candidates) - i) / rate / 60 if rate > 0 else 0
            logger.info(
                f"  [{i}/{len(candidates)}] {n_done} ok / {n_failed} fail "
                f"({rate:.1f}/s, ~{eta:.1f}min remaining)"
            )

    if not dry_run:
        conn.commit()
    conn.close()

    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 56)
    logger.info(f"DONE: {n_done} tagged / {n_failed} failed in {elapsed:.1f}s")
    logger.info(f"  tag rows inserted: {n_inserted}")
    print()
    print("=== geometry distribution ===")
    for bucket, n in sorted(bucket_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  geometry:{bucket}")
    return {"ok": True, "tagged": n_done, "failed": n_failed,
            "inserts": n_inserted}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    r = tag_library(root=args.root, dry_run=args.dry_run,
                    refresh=args.refresh, limit=args.limit)
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
