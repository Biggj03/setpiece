"""
Intro / opener tagger — detect the dance/walking/setup segment at the
start of studio-produced (SP) files so we can route them into an
"Openers" bank.

WHY THIS EXISTS
---------------
Per VJ_ROADMAP #10 + user observation 2026-05-17 night: studio-produced
files (many open with a branded intro card) typically open with
**walking / dancing / setup** scenes for 20-90s before the core
content starts. These are gold for VJ work — projection-friendly,
dance-coded, presentable to mixed audiences. The catch: there's no
existing tag identifying them.

ALGORITHM
---------
The intro→body transition is almost always a hard cut, often the
first hard cut in the file. Heuristic:

1. Open the clip with cv2.VideoCapture, downsample to 96x54 grayscale.
2. Sample at 2 fps for the first 120s (`MAX_SCAN_SEC`).
3. For each frame pair, compute the mean abs-delta. This is the
   per-second frame-diff series.
4. Find peaks: any sample whose delta is > 3.5× the local median
   (15-sample window). These are scene cuts.
5. The intro ends at the FIRST cut that occurs in the 15-90s window
   AND is followed by 5s of "stable" content (no further cuts within
   the next 5s). This filters out montage intros where there are
   multiple early cuts — we want the boundary where the file settles
   into the body.
6. If no qualifying cut, mark `intro:_none` sentinel so re-runs skip.
7. Otherwise tag `intro:<seconds>` (e.g. intro:42).

TARGETING
---------
By default scans ALL files >= 60s. Use `--studio-filter studio-name`
(repeatable) to limit to specific studios — much faster and the
heuristic works better on SP files where intros follow a clear pattern.

USAGE
-----
    # Dry-run on 20 files (writes nothing)
    python intro_tagger.py --dry-run --limit 20

    # Scan only two named studios
    python intro_tagger.py --studio-filter studio-one --studio-filter studio-two

    # Full library scan
    python intro_tagger.py

    # Show distribution after a run
    python intro_tagger.py --report

PERFORMANCE
-----------
At 2 fps × 120s = 240 frames per file. Downsampled grayscale, so each
file ≈ 0.5-2 seconds wall time. ~3000 files in 1.5-2 hours.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("cv2 (opencv-python) required. pip install opencv-python")
    sys.exit(1)

logger = logging.getLogger("intro_tagger")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


DB_PATH = Path.home() / ".setpiece" / "path_tags.db3"

# Frame sampling
SAMPLE_FPS = 2.0           # 2 frames per second
DOWNSAMPLE_WIDTH = 96      # × 54 height after aspect-preserve
MAX_SCAN_SEC = 120         # don't bother past 2 min
MIN_FILE_SEC = 60          # skip clips too short to have an intro

# Cut detection
MEDIAN_WINDOW = 15         # samples for local median
CUT_THRESHOLD_MULT = 3.5   # peak must be > 3.5× local median
INTRO_WIN_LO_SEC = 15.0    # don't tag intros shorter than this
INTRO_WIN_HI_SEC = 90.0    # cap intro length
STABILITY_WIN_SEC = 5.0    # post-cut must be stable for this long

# Safety
MAX_FILE_BYTES = 1.2 * 1024 * 1024 * 1024  # skip files > 1.2 GB (cv2 hangs)

# cv2 decode timeout — a corrupt/odd-codec file fails after 20s rather
# than hanging the scan forever. Flat [propId, value, ...] param list
# for the VideoCapture ctor; empty if this cv2 build lacks the props.
_CAP_TIMEOUT_PARAMS = []
for _prop in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
    _pid = getattr(cv2, _prop, None)
    if _pid is not None:
        _CAP_TIMEOUT_PARAMS += [int(_pid), 20000]


def _downsample_gray(frame):
    """Convert BGR frame to 96×54 grayscale uint8."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    new_w = DOWNSAMPLE_WIDTH
    new_h = max(1, int(h * new_w / max(1, w)))
    small = cv2.resize(frame, (new_w, new_h),
                       interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small.astype(np.uint8)


def _sample_diffs(filepath: str) -> np.ndarray | None:
    """Sample frame-pair deltas at SAMPLE_FPS for first MAX_SCAN_SEC.
    Returns (N,) float32 array of mean abs-deltas, or None on failure."""
    try:
        if _CAP_TIMEOUT_PARAMS:
            cap = cv2.VideoCapture(filepath, cv2.CAP_FFMPEG,
                                   _CAP_TIMEOUT_PARAMS)
        else:
            cap = cv2.VideoCapture(filepath)
    except Exception:
        return None
    if not cap.isOpened():
        try: cap.release()
        except Exception: pass
        return None
    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = total_frames / native_fps if native_fps > 0 else 0
        if duration_s < MIN_FILE_SEC:
            return None
        # SEQUENTIAL read with grab-skip — avoids the keyframe-seek
        # penalty of cv2.set(CAP_PROP_POS_FRAMES). On many codecs the
        # seek has to find the nearest I-frame, which for a 120s scan
        # over 240 samples adds up to minutes. Sequential grab() is
        # cheap (decode without copy). We retrieve() only on the kept
        # frame, decode the rest implicitly via grab().
        skip = max(1, int(native_fps / SAMPLE_FPS))
        scan_frames = min(total_frames, int(MAX_SCAN_SEC * native_fps))
        n_samples = scan_frames // skip
        diffs = []
        prev = None
        for i in range(n_samples):
            # Skip (skip-1) frames cheaply via grab()
            for _ in range(skip - 1):
                if not cap.grab():
                    break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            g = _downsample_gray(frame)
            if g is None:
                continue
            if prev is not None:
                d = float(np.mean(np.abs(g.astype(np.int16)
                                          - prev.astype(np.int16))))
                diffs.append(d)
            prev = g
        return np.array(diffs, dtype=np.float32) if diffs else None
    finally:
        try: cap.release()
        except Exception: pass


def _find_intro_end(diffs: np.ndarray) -> float | None:
    """Find intro-end timestamp in seconds, or None if no clear cut.

    Each diff sample represents ~1/SAMPLE_FPS = 0.5s. Index i =
    timestamp at i / SAMPLE_FPS.
    """
    if len(diffs) < int(INTRO_WIN_LO_SEC * SAMPLE_FPS):
        return None
    n = len(diffs)
    # Per-sample local median for cut threshold
    cuts = []
    for i in range(n):
        lo = max(0, i - MEDIAN_WINDOW // 2)
        hi = min(n, i + MEDIAN_WINDOW // 2 + 1)
        local_med = float(np.median(diffs[lo:hi]))
        if local_med < 0.1:
            continue
        if diffs[i] >= local_med * CUT_THRESHOLD_MULT:
            cuts.append(i)
    if not cuts:
        return None
    # Find first cut in the [INTRO_WIN_LO, INTRO_WIN_HI] second window
    # that has STABILITY_WIN_SEC of calm afterward (no further cut within).
    lo_idx = int(INTRO_WIN_LO_SEC * SAMPLE_FPS)
    hi_idx = int(INTRO_WIN_HI_SEC * SAMPLE_FPS)
    stability_samples = int(STABILITY_WIN_SEC * SAMPLE_FPS)
    for c in cuts:
        if c < lo_idx or c > hi_idx:
            continue
        # Check that no other cut happens in next STABILITY_WIN_SEC
        future_cuts = [x for x in cuts
                       if c < x <= c + stability_samples]
        if not future_cuts:
            return c / SAMPLE_FPS
    return None


def _candidate_files(cur, studio_filters: list[str],
                     force: bool, limit: int) -> list[str]:
    if studio_filters:
        # ANY of the studios matches
        clauses = []
        params = []
        for s in studio_filters:
            clauses.append("filepath IN ("
                           "  SELECT filepath FROM file_tags "
                           "  WHERE tag LIKE ?"
                           ")")
            params.append(f"studio:%{s}%")
        sql = ("SELECT filepath FROM files "
               "WHERE " + " OR ".join(clauses) + " ORDER BY filepath")
        rows = cur.execute(sql, params).fetchall()
    else:
        rows = cur.execute(
            "SELECT filepath FROM files ORDER BY filepath"
        ).fetchall()
    candidates = [r[0] for r in rows]

    if not force:
        already = {r[0] for r in cur.execute(
            "SELECT DISTINCT filepath FROM file_tags "
            "WHERE tag LIKE 'intro:%'"
        ).fetchall()}
        before = len(candidates)
        candidates = [fp for fp in candidates if fp not in already]
        logger.info(
            f"already tagged (skipping): {before - len(candidates)}"
        )

    if limit:
        candidates = candidates[:limit]
    return candidates


def tag_library(
    studio_filters: list[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
    limit: int = 0,
) -> dict:
    if not DB_PATH.is_file():
        logger.error(f"no DB at {DB_PATH}")
        return {"ok": False, "error": "no db"}
    if not dry_run:
        bak = DB_PATH.with_suffix(".db3.bak.intro_tagger")
        shutil.copyfile(DB_PATH, bak)
        logger.info(f"DB backup -> {bak}")
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    cur = conn.cursor()

    # Retry wrapper — same pattern as pose_tagger
    def _with_retry(fn, label="write"):
        last_exc = None
        delay = 1.0
        for attempt in range(1, 17):
            try:
                return fn()
            except sqlite3.OperationalError as e:
                last_exc = e
                if "database is locked" not in str(e):
                    raise
                logger.warning(
                    f"DB locked, retrying in {delay:.1f}s "
                    f"(attempt {attempt}/16, {label})"
                )
                time.sleep(delay)
                delay = min(delay * 1.4, 20.0)
        raise last_exc

    candidates = _candidate_files(cur, studio_filters or [], force, limit)
    total = len(candidates)
    logger.info(f"will process: {total} files")
    if total == 0:
        conn.close()
        return {"ok": True, "tagged": 0, "no_intro": 0, "failed": 0}

    n_tagged = 0
    n_no_intro = 0
    n_failed = 0
    intro_lengths = []
    t0 = time.time()
    for i, fp in enumerate(candidates, 1):
        if not Path(fp).is_file():
            n_failed += 1
            continue
        # Safety: skip giant files (cv2 hangs)
        try:
            if Path(fp).stat().st_size > MAX_FILE_BYTES:
                if not dry_run:
                    def _skip(fp=fp):
                        cur.execute(
                            "INSERT OR IGNORE INTO file_tags(filepath, tag) "
                            "VALUES (?, 'intro:_skip_oversize')", (fp,)
                        )
                    _with_retry(_skip, label=f"skip {fp[-30:]}")
                n_failed += 1
                continue
        except OSError:
            n_failed += 1
            continue
        try:
            diffs = _sample_diffs(fp)
        except Exception as e:
            logger.debug(f"sample failed for {fp}: {e}")
            n_failed += 1
            diffs = None
        if diffs is None or len(diffs) < 4:
            # File too short or decode-fail — sentinel and move on
            if not dry_run:
                def _ins(fp=fp):
                    cur.execute(
                        "INSERT OR IGNORE INTO file_tags(filepath, tag) "
                        "VALUES (?, 'intro:_too_short')", (fp,)
                    )
                _with_retry(_ins, label=f"sentinel {fp[-30:]}")
            n_failed += 1
            continue
        intro_end = _find_intro_end(diffs)
        if intro_end is None:
            n_no_intro += 1
            tag_to_write = "intro:_none"
        else:
            n_tagged += 1
            intro_lengths.append(intro_end)
            tag_to_write = f"intro:{int(round(intro_end))}"

        name = Path(fp).name.encode("ascii", "replace").decode("ascii")
        if intro_end is not None:
            print(f"[{i}/{total}] {name[:60]:60s} intro:{int(round(intro_end))}s")
        elif i % 20 == 0:
            print(f"[{i}/{total}] {name[:60]:60s} (no intro)")
        if not dry_run:
            def _write(fp=fp, t=tag_to_write):
                cur.execute(
                    "INSERT OR IGNORE INTO file_tags(filepath, tag) "
                    "VALUES (?, ?)", (fp, t)
                )
            _with_retry(_write, label=f"insert {fp[-30:]}")
        if i % 10 == 0 and not dry_run:
            _with_retry(conn.commit, label=f"commit @ {i}")
        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate / 60 if rate > 0 else 0
            logger.info(
                f"  [{i}/{total}] {n_tagged} tagged / {n_no_intro} no-intro / "
                f"{n_failed} fail  ({rate:.2f}/s, ~{eta:.1f}min remaining)"
            )
    if not dry_run:
        _with_retry(conn.commit, label="final commit")
    conn.close()
    logger.info("=" * 56)
    logger.info(
        f"DONE: {n_tagged} with intro / {n_no_intro} no intro / "
        f"{n_failed} failed in {time.time() - t0:.1f}s"
    )
    if intro_lengths:
        arr = np.array(intro_lengths)
        logger.info(
            f"intro length: min={arr.min():.0f}s p25={np.percentile(arr,25):.0f}s "
            f"median={np.median(arr):.0f}s p75={np.percentile(arr,75):.0f}s max={arr.max():.0f}s"
        )
    return {
        "ok": True,
        "tagged": n_tagged,
        "no_intro": n_no_intro,
        "failed": n_failed,
    }


def report() -> int:
    """Print intro:N distribution."""
    if not DB_PATH.is_file():
        print(f"no DB at {DB_PATH}")
        return 1
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    cur = conn.cursor()
    total = cur.execute(
        "SELECT COUNT(DISTINCT filepath) FROM file_tags "
        "WHERE tag LIKE 'intro:%'"
    ).fetchone()[0]
    sentinel = cur.execute(
        "SELECT COUNT(DISTINCT filepath) FROM file_tags "
        "WHERE tag LIKE 'intro:_%'"
    ).fetchone()[0]
    real = cur.execute(
        "SELECT COUNT(DISTINCT filepath) FROM file_tags "
        "WHERE tag LIKE 'intro:%' AND tag NOT LIKE 'intro:_%'"
    ).fetchone()[0]
    none_n = cur.execute(
        "SELECT COUNT(*) FROM file_tags WHERE tag = 'intro:_none'"
    ).fetchone()[0]
    print(f"intro:* coverage: {total} files")
    print(f"  real intro tags:  {real}")
    print(f"  no-intro:         {none_n}")
    print(f"  sentinels (skip): {sentinel - none_n}")
    print()
    # Length distribution
    rows = cur.execute(
        "SELECT tag, COUNT(*) FROM file_tags "
        "WHERE tag LIKE 'intro:%' AND tag NOT LIKE 'intro:_%' "
        "GROUP BY tag ORDER BY tag"
    ).fetchall()
    if rows:
        print("intro length buckets:")
        buckets = {(15,30):0, (30,45):0, (45,60):0, (60,75):0, (75,90):0}
        for tag, n in rows:
            try:
                sec = int(tag.split(":")[1])
            except (IndexError, ValueError):
                continue
            for lo, hi in buckets:
                if lo <= sec < hi:
                    buckets[(lo, hi)] += n
                    break
        for (lo, hi), n in buckets.items():
            bar = "#" * min(40, n)
            print(f"  {lo:>3}-{hi:<3}s: {n:>4} {bar}")
    conn.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studio-filter", action="append", default=[],
                    help="substring of studio name (repeatable). "
                         "If empty, scans all files.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-tag files that already have intro:*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", action="store_true",
                    help="print intro:N distribution then exit")
    args = ap.parse_args()
    if args.report:
        return report()
    res = tag_library(
        studio_filters=args.studio_filter or None,
        dry_run=args.dry_run,
        force=args.force,
        limit=args.limit,
    )
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
