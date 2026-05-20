"""
Symmetry tagger -- classify the COMPOSITIONAL structure of each clip
into one of four VJ-canonical buckets. The 4th and final axis in the
VJ-native tag set (color / motion / geometry / SYMMETRY).

WHY THIS EXISTS
---------------
Per VJ_ROADMAP.md / the PDF: symmetry drives *layer pairing* and
*viewer eye-tracking*. Two mirror-symmetric clips stack into a
hypnotic kaleidoscopic feel; a radial clip on top of a cohesive
subject pulls the eye to the center. Offset clips don't stack well
with each other (eye gets pulled in two directions) but pair great
with cohesive backgrounds.

THE FOUR BUCKETS
----------------
  symmetry:mirror    - left-right or top-bottom reflective. Two halves
                       mirror each other. Kaleidoscopic, ballet
                       framings, perfectly composed shots.
  symmetry:radial    - center-out rotational. Mandalas, tunnels,
                       fractals, spirals. Pulls the eye to center.
  symmetry:cohesive  - single visual subject occupying frame center.
                       Music video performer, dance footage, hero
                       shots. Eye has ONE target.
  symmetry:offset    - asymmetric framing. Multi-subject, abstract
                       collages, crowd shots. Eye gets pulled across.
                       Pairs well *under* cohesive overlays.

HOW IT CLASSIFIES (lightweight heuristic, no ML)
-----------------
Mirrors the geometry_tagger pattern: 3 frames sampled across the body
at 64x64 greyscale, averaged into one frame, then four cheap features:

  1. Mirror score = correlation(avg, fliplr(avg)) and
                    correlation(avg, flipud(avg)). Take the max.
                    High = strong reflective symmetry.
  2. Radial score = correlation(avg, rot90(avg)) and
                    correlation(avg, rot180(avg)). Average.
                    High = symmetric under rotation.
  3. Centroid distance = where is the brightness centroid?
                    Close to frame center = subject is centered.
  4. Centroid concentration = how tight is the brightness around the
                    centroid? Tight = single subject (cohesive).

Cascade:
  - mirror_score > 0.85           -> "mirror"
  - elif radial_score > 0.75      -> "radial"
  - elif centroid_dist < 0.18 and concentration good -> "cohesive"
  - else                          -> "offset"

USAGE
-----
    python symmetry_tagger.py --dry-run --limit 10
    python symmetry_tagger.py                 # whole library
    python symmetry_tagger.py --refresh       # re-tag everything

PERFORMANCE
-----------
~400-700ms per file. ~3927 files = ~30-45 min single-threaded.
(Same envelope as geometry_tagger -- N=3 frames, same dec cost.)
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

_SYM_PREFIX = "symmetry:"


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
    """N_FRAMES greyscale frames across body. Returns (N,64,64) uint8.

    N independent fast-seek calls (same fix as motion_tagger /
    geometry_tagger -- fps=X filter hangs on long videos).
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


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two same-shaped float arrays."""
    af = a.astype(np.float32).ravel()
    bf = b.astype(np.float32).ravel()
    af -= af.mean()
    bf -= bf.mean()
    denom = float(np.sqrt((af * af).sum() * (bf * bf).sum()))
    if denom < 1e-6:
        return 0.0
    return float((af * bf).sum() / denom)


def _scores(frames: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mirror, radial, centroid_dist, concentration)."""
    avg = frames.mean(axis=0).astype(np.float32)
    H, W = avg.shape
    mirror_h = _corr(avg, np.fliplr(avg))
    mirror_v = _corr(avg, np.flipud(avg))
    mirror_score = max(mirror_h, mirror_v)
    rot90 = np.rot90(avg)
    rot180 = np.rot90(avg, 2)
    radial_score = (_corr(avg, rot90) + _corr(avg, rot180)) / 2.0
    mass = np.clip(avg - avg.mean(), 0, None)
    total = float(mass.sum())
    centroid_dist = 1.0
    concentration = 0.0
    if total > 1e-3:
        ys = np.arange(H).reshape(H, 1).astype(np.float32)
        xs = np.arange(W).reshape(1, W).astype(np.float32)
        cy = float((ys * mass).sum() / total)
        cx = float((xs * mass).sum() / total)
        cy_frac = (cy - H / 2) / (H / 2)
        cx_frac = (cx - W / 2) / (W / 2)
        centroid_dist = float(np.hypot(cy_frac, cx_frac))
        dist_sq = ((xs - cx) ** 2 + (ys - cy) ** 2)
        spread = float(np.sqrt((mass * dist_sq).sum() / total)) / (H / 2)
        concentration = max(0.0, 1.0 - spread)
    return mirror_score, radial_score, centroid_dist, concentration


def _debug_scores(frames: np.ndarray) -> str:
    m, r, d, c = _scores(frames)
    return f"mir={m:+.2f} rad={r:+.2f} dist={d:.2f} conc={c:.2f}"


def _classify(frames: np.ndarray) -> str:
    """Return the chosen symmetry bucket string (without prefix)."""
    if frames is None or len(frames) == 0:
        return "offset"

    mirror_score, radial_score, centroid_dist, concentration = _scores(frames)

    # ── classification cascade ─────────────────────────────────────────
    # Thresholds set empirically from --debug pass over a 40-file sample.
    # Real-world music-video corpus: edge-driven mirror corr lives
    # in the 0.3-0.7 band, with genuinely symmetric framings >0.75. Rot
    # correlation rarely exceeds 0.45 outside actual mandalas/tunnels.
    if mirror_score > 0.75:
        return "mirror"
    if radial_score > 0.50:
        return "radial"
    # Cohesive: clearly-centered brightness mass. Concentration bar is
    # loose -- MV footage fills the frame edge-to-edge so even
    # single-subject shots rarely concentrate tightly. Centroid
    # distance is the primary signal.
    if centroid_dist < 0.22 and concentration > 0.20:
        return "cohesive"
    return "offset"


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
        bak = DB_PATH.with_suffix(".db3.bak.symmetry_tagger")
        shutil.copyfile(DB_PATH, bak)
        logger.info(f"DB backup -> {bak}")

    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    cur = conn.cursor()
    # WAL mode + busy timeout so we can run concurrently with the
    # geometry tagger (or main app) without "database is locked".
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
                "DELETE FROM file_tags WHERE tag LIKE 'symmetry:%'"
            ).rowcount
            conn.commit()
            logger.info(f"refresh: dropped {n} existing symmetry tags")
    else:
        tagged = set(r[0] for r in cur.execute(
            "SELECT DISTINCT filepath FROM file_tags WHERE tag LIKE 'symmetry:%'"
        ).fetchall())
        before = len(candidates)
        candidates = [fp for fp in candidates if fp not in tagged]
        logger.info(f"already tagged (skipping): {before - len(candidates)}")

    if limit:
        candidates = candidates[:limit]
    logger.info(f"will process: {len(candidates)} files")

    debug = bool(getattr(tag_library, "_debug", False))

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
        if debug:
            scores = _debug_scores(frames)
            print(f"  {scores}  {Path(fp).name[:60]}")
        bucket = _classify(frames)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if not dry_run:
            tag = _SYM_PREFIX + bucket
            cur.execute(
                "INSERT OR IGNORE INTO file_tags(filepath, tag) VALUES (?, ?)",
                (fp, tag)
            )
            if cur.rowcount:
                n_inserted += 1
            # Commit after every insert. Keeps the write transaction
            # open only during the millisecond-level INSERT, not across
            # the next 1s of ffmpeg sampling -- so concurrent taggers
            # (geometry, app) don't time out fighting for the lock.
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
    print("=== symmetry distribution ===")
    for bucket, n in sorted(bucket_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  symmetry:{bucket}")
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
    ap.add_argument("--debug", action="store_true",
                    help="print per-file scores (mir/rad/dist/conc)")
    args = ap.parse_args()
    tag_library._debug = args.debug
    r = tag_library(root=args.root, dry_run=args.dry_run,
                    refresh=args.refresh, limit=args.limit)
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
