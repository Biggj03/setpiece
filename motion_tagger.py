"""
Motion / energy tagger — analyze each clip's intrinsic motion intensity
and store as VJ-native tags in path_tags.db3.

WHY THIS EXISTS
---------------
Per VJ_ROADMAP.md, the PDF "Architectural Paradigms in Live Visual
Performance" identifies four VJ-native tag axes:
   Color (shipped via color_tagger.py)
   Geometry
   Energy / Movement   <-- this script
   Symmetry

The PDF (page 11) is explicit: "clips must be tagged by their inherent
movement characteristics rather than arbitrary numerical tempos." A
slow-moving massive 3D object during a 174 BPM track creates an
imposing heavy atmosphere — its visual "energy" is independent of the
music's BPM. The 4-phase set arc (Opening / Build / Peak/Drop /
Breakdown) needs this dimension to drive picker behavior.

WHAT IT TAGS
------------
For each video, samples ~10 frames spread across the body of the clip
(skipping the first/last 10% as bumpers), computes:

  1. **Mean frame-to-frame pixel delta** -> motion magnitude
     -> tag: `motion:static` (low) | `motion:dynamic` (high)
        cutoff at the global median.
  2. **Std-dev of pixel deltas across the sample window**
     -> tag: `motion:smooth` (low variance) | `motion:jumpy` (high)
        smooth = consistent continuous motion; jumpy = glitch/cut.
  3. **Spatial complexity** -> edge density via Laplacian variance
     -> tag: `complexity:N` (N = 0..9, quantile-bucketed globally)

ARCHITECTURE
------------
- ffmpeg pulls N frames at evenly-spaced timestamps in ONE invocation
  via select='not(mod(n,K))' filter — single decoder pass per file,
  much faster than N seek-extracts.
- Compares each adjacent frame pair (delta = mean abs pixel diff
  on grayscale-downsampled 64x64).
- Pure numpy + ffmpeg, no scikit-image/opencv deps.

USAGE
-----
    # Dry-run on 10 files
    python motion_tagger.py --dry-run --limit 10

    # Real run, whole library
    python motion_tagger.py

    # Just one folder
    python motion_tagger.py --root "/path/to/clips"

    # Refresh (drop existing motion tags first)
    python motion_tagger.py --refresh

PERFORMANCE
-----------
~300-600ms per file (single ffmpeg decode + N small numpy diffs).
3927 files = ~25-40 min single-threaded.

DESIGN NOTES
------------
- Static vs dynamic threshold is computed at end-of-run from the global
  delta distribution (median split). First-pass assigns interim values;
  second pass computes the split and writes tags. This avoids the
  "what's a high delta?" question — it's "high RELATIVE to your library."
- Same approach for complexity quantiles (deciles 0..9). A "9" clip is
  in the top 10% most-complex of YOUR library, regardless of absolute
  edge count.
- We sample greyscale 64x64 because motion is dominated by luminance
  changes and we don't need color for delta computation. 4x faster
  than full-color sampling.
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

SAMPLE_PX = 64           # 64x64 grey per frame
N_FRAMES = 10            # samples per clip
SKIP_HEAD_PCT = 0.10     # skip first 10% (bumpers)
SKIP_TAIL_PCT = 0.10     # skip last 10% (outros)


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


def _sample_frames(filepath: str) -> np.ndarray | None:
    """Extract N_FRAMES grayscale 64x64 frames spread across the body.
    Returns (N, 64, 64) uint8 or None on failure.

    Uses N independent fast-seek ffmpeg calls (one per timestamp). This
    is more reliable than a single `fps=X` filter call, which on long
    videos forces ffmpeg to decode the full window before yielding
    frames (e.g. on a 1-hour file at fps=0.003 it waits the full hour).
    N small seek-and-extract calls take ~50ms each vs minutes for the
    filter approach.

    Each ffmpeg call uses input-level -ss (before -i) which does a
    fast keyframe seek then decodes just 1 frame. Total N small RPCs.
    """
    dur = _probe_duration(filepath)
    if dur <= 2.0:
        return None
    start = dur * SKIP_HEAD_PCT
    end = dur * (1.0 - SKIP_TAIL_PCT)
    if end - start < 1.0:
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
    if len(frames) < 2:
        return None
    return np.stack(frames, axis=0)


def _frame_metrics(frames: np.ndarray) -> dict:
    """From (N, H, W) grayscale frames, compute the three metrics.
    Returns dict with raw float values; tag-bucketing happens later
    once we have the global distribution."""
    if frames is None or len(frames) < 2:
        return {}
    f = frames.astype(np.int16)  # signed so subtraction doesn't wrap
    # 1. Adjacent-frame deltas: mean abs pixel diff per pair
    deltas = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    mean_delta = float(deltas.mean())
    std_delta = float(deltas.std())
    # 2. Spatial complexity per frame: Laplacian-like (4-neighbor diff).
    # Avg over the N frames so we get a per-clip complexity score.
    # Simple Laplacian: |center - mean(neighbors)| -- approximated as
    # |f - mean filter| via small offset shifts.
    lap_vars = []
    for fr in frames:
        fr_f = fr.astype(np.float32)
        # 4-neighbor "mean" via shifts (cheap, no scipy)
        s = (np.roll(fr_f, 1, 0) + np.roll(fr_f, -1, 0) +
             np.roll(fr_f, 1, 1) + np.roll(fr_f, -1, 1)) / 4.0
        lap = np.abs(fr_f - s)
        lap_vars.append(float(lap.var()))
    mean_complexity = float(np.mean(lap_vars))
    return {
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "complexity": mean_complexity,
    }


def _tag_from_metrics(m: dict, dist: dict) -> list[str]:
    """Given a file's metrics and the global distribution, return tags."""
    out = []
    # static / dynamic: median split on mean_delta
    if m["mean_delta"] < dist["median_delta"]:
        out.append("motion:static")
    else:
        out.append("motion:dynamic")
    # smooth / jumpy: median split on std_delta
    if m["std_delta"] < dist["median_std"]:
        out.append("motion:smooth")
    else:
        out.append("motion:jumpy")
    # complexity: decile bucket 0..9 from global distribution
    c = m["complexity"]
    bucket = int(np.searchsorted(dist["complexity_deciles"], c))
    bucket = max(0, min(9, bucket))
    out.append(f"complexity:{bucket}")
    return out


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
        bak = DB_PATH.with_suffix(".db3.bak.motion_tagger")
        shutil.copyfile(DB_PATH, bak)
        logger.info(f"DB backup -> {bak}")

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

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
                "DELETE FROM file_tags WHERE tag LIKE 'motion:%' "
                "OR tag LIKE 'complexity:%'"
            ).rowcount
            conn.commit()
            logger.info(f"refresh: dropped {n} existing motion/complexity tags")
    else:
        # Skip files already tagged
        before = len(candidates)
        tagged_set = set()
        for (fp,) in cur.execute(
            "SELECT DISTINCT filepath FROM file_tags WHERE tag LIKE 'motion:%'"
        ).fetchall():
            tagged_set.add(fp)
        candidates = [fp for fp in candidates if fp not in tagged_set]
        logger.info(f"already tagged (skipping): {before - len(candidates)}")

    if limit:
        candidates = candidates[:limit]
    logger.info(f"will process: {len(candidates)} files")

    # PASS 1: extract metrics for every file
    metrics: dict[str, dict] = {}
    n_failed = 0
    t0 = time.time()
    for i, fp in enumerate(candidates, 1):
        if not Path(fp).is_file():
            n_failed += 1
            continue
        frames = _sample_frames(fp)
        if frames is None:
            n_failed += 1
            continue
        m = _frame_metrics(frames)
        if not m:
            n_failed += 1
            continue
        metrics[fp] = m
        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(candidates) - i) / rate / 60 if rate > 0 else 0
            logger.info(
                f"  [pass1 {i}/{len(candidates)}] {len(metrics)} ok / "
                f"{n_failed} fail ({rate:.1f}/s, ~{eta:.1f}min remaining)"
            )

    logger.info(f"pass 1 done: {len(metrics)} metrics, {n_failed} failed")
    if not metrics:
        logger.error("no metrics — abort")
        conn.close()
        return {"ok": False, "error": "no metrics"}

    # Compute global distribution
    deltas = np.array([m["mean_delta"] for m in metrics.values()])
    stds = np.array([m["std_delta"] for m in metrics.values()])
    comps = np.array([m["complexity"] for m in metrics.values()])
    dist = {
        "median_delta": float(np.median(deltas)),
        "median_std": float(np.median(stds)),
        "complexity_deciles": [float(np.quantile(comps, q / 10.0)) for q in range(1, 10)],
    }
    logger.info(f"distribution: median_delta={dist['median_delta']:.1f}  "
                f"median_std={dist['median_std']:.1f}")
    logger.info(f"  complexity deciles: "
                f"{[f'{d:.0f}' for d in dist['complexity_deciles']]}")

    # PASS 2: assign tags from distribution + write
    n_inserted = 0
    motion_counts: dict = {}
    complexity_counts: dict = {}
    for fp, m in metrics.items():
        tags = _tag_from_metrics(m, dist)
        for t in tags:
            if t.startswith("motion:"):
                motion_counts[t] = motion_counts.get(t, 0) + 1
            elif t.startswith("complexity:"):
                complexity_counts[t] = complexity_counts.get(t, 0) + 1
        if not dry_run:
            for t in tags:
                cur.execute(
                    "INSERT OR IGNORE INTO file_tags(filepath, tag) VALUES (?, ?)",
                    (fp, t)
                )
                if cur.rowcount:
                    n_inserted += 1
    if not dry_run:
        conn.commit()
    conn.close()

    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 56)
    logger.info(f"DONE: {len(metrics)} tagged / {n_failed} failed in {elapsed:.1f}s")
    logger.info(f"  tag rows inserted: {n_inserted}")
    print()
    print("=== motion split ===")
    for t, n in sorted(motion_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {t}")
    print()
    print("=== complexity distribution ===")
    for t in sorted(complexity_counts.keys()):
        print(f"  {complexity_counts[t]:5d}  {t}")
    return {"ok": True, "tagged": len(metrics), "failed": n_failed,
            "inserts": n_inserted}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="restrict to files under this folder")
    ap.add_argument("--dry-run", action="store_true",
                    help="show plan without writing")
    ap.add_argument("--refresh", action="store_true",
                    help="drop existing motion/complexity tags + retag")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N files (0 = all)")
    args = ap.parse_args()
    r = tag_library(root=args.root, dry_run=args.dry_run,
                    refresh=args.refresh, limit=args.limit)
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
