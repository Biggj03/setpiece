"""
Color tagger — extract dominant color palette per clip and store it
as searchable tags in path_tags.db3.

WHY THIS EXISTS
---------------
Per the VJ research (see VJ_ROADMAP.md), **color is the primary
sorting axis** for pro VJs — they organize libraries laterally
across the light spectrum (red → yellow → green → blue → magenta)
because lighting design + VJing are environmentally synchronized.
The PDF "Architectural Paradigms in Live Visual Performance" calls
color "the fastest retrieval mechanism for a VJ."

We were previously sorting by performer/genre/BPM — useful, but
secondary to color in a live-set context. This script adds the
color dimension.

WHAT IT DOES
------------
For each video in path_tags.db3 without a color: tag:

  1. ffmpeg samples ONE 64x64 RGB frame at 30% into the clip
     (skips title bumpers, lands in meat of clip). Cheap — no
     full decode, no audio extract.
  2. numpy k-means k=3 in LAB color space (LAB = perceptually
     uniform; closer to how a VJ judges "warm vs cool" than RGB).
  3. Stores tags:
       color:warm | color:cool         (binary thermal)
       color:hue:red | orange | yellow | green | cyan | blue
                    | magenta            (6-bucket spectrum)
       palette:<hex1>:<hex2>:<hex3>    (3-color dominant palette)
     Skips any file already tagged (idempotent + resumable).

ARCHITECTURE
------------
- Reuses the same path_tags.db3 schema and write pattern as
  the rest of the ingest pipeline.
- Single ffmpeg child per file, scaled to 64x64 to keep k-means
  cheap (~5ms per cluster on 4096 pixels).
- Safe to run while setpiece is up (SQLite WAL handles
  concurrent reads).

USAGE
-----
    # Dry-run: count + first few proposals
    python color_tagger.py --dry-run

    # Real run on whole library
    python color_tagger.py

    # Just the loose drops in a folder
    python color_tagger.py --root "/path/to/clips"

    # Refresh: re-tag files that already have color tags
    python color_tagger.py --refresh

    # Limit (smoke test)
    python color_tagger.py --limit 10

PERFORMANCE
-----------
~200-400ms per file (ffmpeg single-frame seek + numpy k-means).
~3900 files = ~15 min single-threaded. Cheap enough to re-run.

DESIGN NOTES
------------
- LAB clustering, not RGB. RGB k-means gives perceptually wrong
  centroids on saturated content. LAB centroids align with how the
  brain reads "those clips feel similar."
- 64x64 sample size = 4096 pixels. Plenty for k=3 in LAB; faster
  is overkill (e.g. 32x32 = 1024 is also fine but the saving is
  in ffmpeg-startup, not k-means).
- Hue buckets match the CDJ standard color rotation + the PDF's
  R→Y→G→B→M spectrum. 6 buckets is the sweet spot for human
  glance-parsing (too few = lumps blue + green; too many = no one
  remembers the boundaries).
- Warm/cool split = simple hue cut: 0-90° + 270-360° = warm
  (reds + oranges + magentas), 90-270° = cool (greens + cyans
  + blues + purples). Per PDF / VJ tradition.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
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
SAMPLE_PX = 64  # 64x64 = 4096 pixels per frame -- enough for k=3

# Tag prefixes (so we can skip already-tagged + cleanup if needed)
_THERM_TAGS = ("color:warm", "color:cool")
_HUE_PREFIX = "color:hue:"
_PALETTE_PREFIX = "palette:"

# 6-bucket hue boundaries (deg, inclusive low, exclusive high).
# Matches the CDJ color rotation reading order + the PDF "R→Y→G→B→M".
_HUE_BUCKETS = [
    ("red",     350, 30),   # wraps 350°-360° + 0°-30°
    ("orange",  30, 70),    # orange-yellow
    ("yellow",  70, 90),
    ("green",   90, 170),
    ("cyan",    170, 210),
    ("blue",    210, 270),
    ("magenta", 270, 350),
]


def _ffmpeg_sample_frame(filepath: str, when_pct: float = 0.3) -> np.ndarray | None:
    """Pull one 64x64 RGB frame from `filepath` at `when_pct` into
    its duration. Returns (64,64,3) uint8 array or None on failure.
    Uses two-pass: probe duration with ffprobe, then -ss before -i for
    fast keyframe seek."""
    # 1. Probe duration
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        dur = float(r.stdout.strip() or 0.0)
    except Exception as e:
        logger.debug(f"ffprobe failed: {filepath}: {e}")
        return None
    if dur <= 0:
        return None
    seek = max(0.1, dur * when_pct)

    # 2. Extract single frame, scale, rawvideo to stdout
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error",
             "-ss", f"{seek:.2f}", "-i", filepath,
             "-frames:v", "1",
             "-vf", f"scale={SAMPLE_PX}:{SAMPLE_PX}",
             "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            capture_output=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        raw = r.stdout
    except Exception as e:
        logger.debug(f"ffmpeg failed: {filepath}: {e}")
        return None
    expected = SAMPLE_PX * SAMPLE_PX * 3
    if len(raw) != expected:
        logger.debug(f"frame size mismatch {filepath}: got {len(raw)}, want {expected}")
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape(SAMPLE_PX, SAMPLE_PX, 3)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 → CIE LAB float. Vectorized. (N,3) → (N,3).

    Inline implementation (no scikit-image dep) — LAB conversion is
    a fixed pipeline of well-known matrix ops, no need for the dep.
    """
    rgb_norm = rgb.astype(np.float32) / 255.0
    # sRGB gamma inverse → linear RGB
    mask = rgb_norm > 0.04045
    lin = np.where(mask,
                   ((rgb_norm + 0.055) / 1.055) ** 2.4,
                   rgb_norm / 12.92)
    # linear RGB → XYZ (D65)
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = lin @ m.T
    # normalize by D65 white
    xyz /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    # XYZ → LAB
    eps = (6.0 / 29.0) ** 3
    f = np.where(xyz > eps,
                 np.cbrt(xyz),
                 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _lab_to_rgb_hex(lab: np.ndarray) -> str:
    """Single LAB triplet → '#rrggbb' string. Inverse of _rgb_to_lab."""
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    eps = 6.0 / 29.0
    def finv(t):
        return t**3 if t > eps else 3 * eps * eps * (t - 4.0 / 29.0)
    xyz = np.array([finv(fx), finv(fy), finv(fz)])
    xyz *= np.array([0.95047, 1.00000, 1.08883])
    m_inv = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ])
    lin = xyz @ m_inv.T
    lin = np.clip(lin, 0, 1)
    srgb = np.where(lin > 0.0031308,
                    1.055 * lin ** (1.0 / 2.4) - 0.055,
                    12.92 * lin)
    srgb = (np.clip(srgb, 0, 1) * 255).astype(int)
    return f"#{srgb[0]:02x}{srgb[1]:02x}{srgb[2]:02x}"


def _kmeans_lab(pixels_lab: np.ndarray, k: int = 3, iters: int = 12) -> np.ndarray:
    """Simple k-means in LAB. Returns (k, 3) centroids sorted by
    cluster size descending. Pure numpy — no sklearn dep."""
    rng = np.random.default_rng(42)  # deterministic per-file
    n = pixels_lab.shape[0]
    # init: pick k random distinct pixels
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = pixels_lab[init_idx].copy()
    for _ in range(iters):
        # distances: (n, k)
        d = np.linalg.norm(pixels_lab[:, None, :] - centroids[None, :, :], axis=2)
        labels = d.argmin(axis=1)
        new_centroids = np.array([
            pixels_lab[labels == j].mean(axis=0) if (labels == j).any()
            else centroids[j]
            for j in range(k)
        ])
        if np.allclose(new_centroids, centroids, atol=0.5):
            break
        centroids = new_centroids
    # Sort by cluster size descending
    sizes = np.array([(labels == j).sum() for j in range(k)])
    order = sizes.argsort()[::-1]
    return centroids[order]


def _hue_from_lab(lab: np.ndarray) -> float:
    """LAB a,b → hue in degrees 0..360. Uses atan2(b, a)."""
    a, b = float(lab[1]), float(lab[2])
    h = math.degrees(math.atan2(b, a))
    return h + 360.0 if h < 0 else h


def _hue_to_bucket(deg: float) -> str:
    for name, lo, hi in _HUE_BUCKETS:
        if lo <= hi:
            if lo <= deg < hi:
                return name
        else:
            # wrap-around bucket (red: 350..30)
            if deg >= lo or deg < hi:
                return name
    return "red"  # fallback


def _is_warm(deg: float) -> bool:
    """Warm hues: 0-90 + 270-360 (reds, oranges, yellows, magentas)."""
    return deg < 90.0 or deg >= 270.0


def _is_low_chroma(lab: np.ndarray, thresh: float = 8.0) -> bool:
    """Near-greyscale pixel? a,b close to 0 = no real hue."""
    a, b = float(lab[1]), float(lab[2])
    return (a * a + b * b) ** 0.5 < thresh


def derive_tags(frame_rgb: np.ndarray) -> list[str]:
    """Given a (64,64,3) RGB frame, return the full set of tag strings."""
    pixels = frame_rgb.reshape(-1, 3).astype(np.float32)
    lab = _rgb_to_lab(pixels)
    centroids = _kmeans_lab(lab, k=3)

    # Find the DOMINANT colored centroid (skip near-greyscale unless all 3 are)
    colored = [c for c in centroids if not _is_low_chroma(c)]
    primary = colored[0] if colored else centroids[0]
    hue = _hue_from_lab(primary)
    bucket = _hue_to_bucket(hue)
    therm = "color:warm" if _is_warm(hue) else "color:cool"

    # Build the palette hex string from the 3 centroids (dominant first)
    hexes = [_lab_to_rgb_hex(c) for c in centroids]
    palette_tag = f"{_PALETTE_PREFIX}{':'.join(h.lstrip('#') for h in hexes)}"

    return [therm, f"{_HUE_PREFIX}{bucket}", palette_tag]


def _file_already_tagged(cur: sqlite3.Cursor, filepath: str) -> bool:
    r = cur.execute(
        "SELECT 1 FROM file_tags WHERE filepath=? AND tag LIKE 'color:%' LIMIT 1",
        (filepath,)
    ).fetchone()
    return r is not None


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
        bak = DB_PATH.with_suffix(".db3.bak.color_tagger")
        shutil.copyfile(DB_PATH, bak)
        logger.info(f"DB backup -> {bak}")

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Pull file list
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
    logger.info(f"candidate files: {len(candidates)}")

    if refresh:
        # Drop existing color tags so we re-derive
        if not dry_run:
            n = cur.execute(
                "DELETE FROM file_tags WHERE tag LIKE 'color:%' OR tag LIKE 'palette:%'"
            ).rowcount
            conn.commit()
            logger.info(f"refresh: dropped {n} existing color/palette tags")
    else:
        before = len(candidates)
        candidates = [fp for fp in candidates if not _file_already_tagged(cur, fp)]
        logger.info(f"already tagged (skipping): {before - len(candidates)}")

    if limit:
        candidates = candidates[:limit]
        logger.info(f"limit={limit}; will process {len(candidates)}")

    n_done = 0
    n_failed = 0
    n_inserted = 0
    bucket_counts: dict = {}
    therm_counts: dict = {}
    t0 = time.time()

    for i, fp in enumerate(candidates, 1):
        if not Path(fp).is_file():
            n_failed += 1
            continue
        frame = _ffmpeg_sample_frame(fp)
        if frame is None:
            n_failed += 1
            logger.debug(f"  [{i}] no frame: {Path(fp).name}")
            continue
        tags = derive_tags(frame)
        # stats
        for t in tags:
            if t.startswith(_HUE_PREFIX):
                b = t[len(_HUE_PREFIX):]
                bucket_counts[b] = bucket_counts.get(b, 0) + 1
            if t in _THERM_TAGS:
                therm_counts[t] = therm_counts.get(t, 0) + 1
        if not dry_run:
            for t in tags:
                cur.execute(
                    "INSERT OR IGNORE INTO file_tags(filepath, tag) VALUES (?, ?)",
                    (fp, t)
                )
                if cur.rowcount:
                    n_inserted += 1
            if i % 25 == 0:
                conn.commit()
        n_done += 1
        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            remaining = (len(candidates) - i) / rate if rate > 0 else 0
            logger.info(
                f"  [{i}/{len(candidates)}] {n_done} ok / {n_failed} fail "
                f"({rate:.1f}/s, ~{remaining/60:.1f}min remaining)"
            )

    if not dry_run:
        conn.commit()
    conn.close()

    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 56)
    logger.info(f"DONE: {n_done} tagged / {n_failed} failed in {elapsed:.1f}s")
    logger.info(f"  tag rows inserted: {n_inserted}")
    logger.info("")
    logger.info("=== thermal split ===")
    for t, n in sorted(therm_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {t}")
    print()
    print("=== hue buckets ===")
    for b, n in sorted(bucket_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  color:hue:{b}")
    return {"ok": True, "tagged": n_done, "failed": n_failed,
            "inserts": n_inserted}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="restrict to files under this folder (default: whole DB)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would happen without writing")
    ap.add_argument("--refresh", action="store_true",
                    help="drop existing color tags + retag everything")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N files (0 = all)")
    args = ap.parse_args()
    r = tag_library(root=args.root, dry_run=args.dry_run,
                    refresh=args.refresh, limit=args.limit)
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
