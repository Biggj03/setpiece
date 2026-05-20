"""
Pose tagger -- emit `position:*` body-orientation tags from video frames
using YOLOv8-pose keypoint detection.

WHY THIS EXISTS
---------------
The bank-overlay system in main.py has a `positions` layer that maps
A-H to position categories (solo / standing / cowgirl / doggy /
missionary / spoon / anal-pos / group-pos). But the tags those
categories filter on (`doggy`, `cowgirl`, `missionary`, etc.) are
sparse in path_tags.db3 because they came from filename tokenization
only -- most clips don't have the position spelled in the filename.

This tagger looks at the PIXELS and classifies body orientation
from human pose, then emits namespaced `position:` tags so the bank
picker (or a future explicit position layer) can pull from a real
signal instead of guessing from filenames.

ALGORITHM
---------
1. Sample 4 keyframes per video at 20/40/60/80% timestamps
   (avoids title-card noise at very start/end).
2. Run YOLOv8n-pose on each keyframe -> 17 COCO keypoints per
   person detected:
      0=nose 1=l_eye 2=r_eye 3=l_ear 4=r_ear
      5=l_shoulder 6=r_shoulder 7=l_elbow 8=r_elbow
      9=l_wrist 10=r_wrist 11=l_hip 12=r_hip
      13=l_knee 14=r_knee 15=l_ankle 16=r_ankle
3. Classify each frame's pose with simple geometric heuristics
   (see _classify_frame). Yields one of:
      doggy / cowgirl / reverse-cowgirl / missionary
      standing / spoon / face-up / unknown
4. Aggregate across frames: if >=3 of the (up to 4) frames agree,
   emit that as the position tag. Otherwise emit `position:mixed`
   if any clear pose detected, else nothing.

HEURISTICS (one-person frames only)
-----------------------------------
- Multiple people detected -> skip (group scene, other tags cover it).
- Pose ambiguous / not enough keypoints visible -> skip that frame.
- doggy:        face not visible (nose/eyes occluded), hips ~level
                with or below shoulders, legs spread, body roughly
                horizontal -> rear-entry.
- cowgirl:      face visible, knees in front of hips (bent), torso
                roughly vertical, hips below shoulders -> rider-on-top.
                If face is NOT visible but knees-bent-vertical pose
                holds, emit reverse-cowgirl.
- missionary:   body roughly horizontal (shoulders+hips on same y),
                knees raised (above hips in image), face visible.
- standing:     all keypoints roughly vertical alignment (large
                vertical spread, narrow horizontal spread); ankles
                clearly below hips clearly below shoulders.
- spoon:        body horizontal, but legs together (knees+ankles
                close), face profile (one ear visible, not both).
                Harder to detect reliably -- often falls to mixed.
- face-up:      torso visible from below, face visible, head higher
                than hips in image but body not vertical (catch-all
                for supine-with-camera-low shots).

THRESHOLDS
----------
Tuned conservative -- prefer NO tag to a wrong tag. Bank picker
backs off cleanly to other layers when position is absent.

PERFORMANCE
-----------
YOLOv8n-pose on a GTX 1650 (4GB) runs ~30-60ms per 640px frame
on GPU. 4 frames per video + ffmpeg keyframe extract overhead
(~200-400ms per file) => ~0.5-1.0s/file. For 5678 files that's
~50-95 minutes wall time. Real-world will be slower because GPU
is shared with mpv/NVENC; budget ~2-3 hours overnight.

Single-worker by design: the GPU IS the bottleneck. Threading
multiple decoders wastes CPU cycles waiting for GPU queue.

USAGE
-----
    # Dry-run on 20 files
    python pose_tagger.py --dry-run --limit 20

    # Real run, whole library
    python pose_tagger.py

    # Just one folder
    python pose_tagger.py --root "/path/to/clips"

    # Force-retag files that already have position: tags
    python pose_tagger.py --force

    # Inspect existing position tags in DB
    python pose_tagger.py --list-positions
"""

from __future__ import annotations

import argparse
import logging
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

# Model file. ultralytics auto-downloads on first use into ./yolov8n-pose.pt
# next to the script (or into ~/.config/Ultralytics depending on version).
MODEL_NAME = "yolov8n-pose.pt"   # ~6MB, fast. Use yolov8s-pose.pt for accuracy.
MODEL_IMGSZ = 640                # YOLO inference resolution
CONF_THRESHOLD = 0.35            # min detection confidence for a person
KEYPOINT_VIS_THRESHOLD = 0.5     # min keypoint confidence to call it "visible"

# Sample 4 frames at 20/40/60/80% of clip duration. Avoids the very
# start/end (title cards / outros that have no person at all).
SAMPLE_PCTS = (0.20, 0.40, 0.60, 0.80)
MIN_AGREEMENT = 3                # need >=3 of 4 frames to agree for a hard tag

MIN_CLIP_SEC = 4.0               # too short = often a gif loop, skip
MAX_FILE_BYTES = 1_500_000_000   # skip files >1.5 GB (decode hangs risk)
FFMPEG_TIMEOUT_SEC = 8           # per-frame ffmpeg extract timeout

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

# COCO keypoint indices for readability.
KP_NOSE, KP_LEYE, KP_REYE, KP_LEAR, KP_REAR = 0, 1, 2, 3, 4
KP_LSHOULDER, KP_RSHOULDER = 5, 6
KP_LELBOW, KP_RELBOW, KP_LWRIST, KP_RWRIST = 7, 8, 9, 10
KP_LHIP, KP_RHIP = 11, 12
KP_LKNEE, KP_RKNEE, KP_LANKLE, KP_RANKLE = 13, 14, 15, 16

# Lazily loaded YOLO model (avoid loading at import time so
# --list-positions and --help don't pay the cost).
_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "ultralytics not installed. Run: pip install ultralytics"
        ) from e
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"loading {MODEL_NAME} on device={device}")
    _MODEL = YOLO(MODEL_NAME)
    # Move to GPU explicitly. ultralytics handles this internally on
    # .predict() but setting up-front lets us catch CUDA OOM early.
    try:
        _MODEL.to(device)
    except Exception as e:
        logger.warning(f"could not move model to {device}: {e}")
    return _MODEL


# ---- frame extraction ------------------------------------------------------


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


def _extract_frame(filepath: str, ts: float) -> np.ndarray | None:
    """Extract a single frame at timestamp `ts` (seconds) as BGR np array.
    Returns None on any failure. Uses input-level -ss for fast keyframe
    seek (same pattern as motion_tagger)."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error",
             "-ss", f"{float(ts):.2f}", "-i", filepath,
             "-frames:v", "1",
             "-vf", f"scale={MODEL_IMGSZ}:{MODEL_IMGSZ}:"
                    f"force_original_aspect_ratio=decrease,"
                    f"pad={MODEL_IMGSZ}:{MODEL_IMGSZ}:(ow-iw)/2:(oh-ih)/2",
             "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
            capture_output=True, timeout=FFMPEG_TIMEOUT_SEC,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return None
    raw = r.stdout
    expected = MODEL_IMGSZ * MODEL_IMGSZ * 3
    if len(raw) < expected:
        return None
    return np.frombuffer(raw[:expected], dtype=np.uint8).reshape(
        MODEL_IMGSZ, MODEL_IMGSZ, 3
    )


def _sample_keyframes(filepath: str) -> list[np.ndarray]:
    """Return up to len(SAMPLE_PCTS) keyframes for this clip."""
    try:
        sz = os.path.getsize(filepath)
        if sz > MAX_FILE_BYTES:
            return []
    except OSError:
        return []
    dur = _probe_duration(filepath)
    if dur < MIN_CLIP_SEC:
        return []
    frames: list[np.ndarray] = []
    for pct in SAMPLE_PCTS:
        ts = dur * pct
        fr = _extract_frame(filepath, ts)
        if fr is not None:
            frames.append(fr)
    return frames


# ---- pose classification ---------------------------------------------------


def _visible(kp_conf: np.ndarray, idx: int) -> bool:
    """Is keypoint `idx` confidently detected?"""
    return bool(kp_conf[idx] >= KEYPOINT_VIS_THRESHOLD)


def _avg(kp_xy: np.ndarray, kp_conf: np.ndarray, indices: list[int]) -> tuple[float, float] | None:
    """Average position of visible keypoints in `indices`. None if none visible."""
    pts = [kp_xy[i] for i in indices if _visible(kp_conf, i)]
    if not pts:
        return None
    arr = np.asarray(pts, dtype=np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _classify_frame(kp_xy: np.ndarray, kp_conf: np.ndarray) -> str:
    """Classify a single pose into a position label.

    kp_xy: (17, 2) pixel coordinates (x, y), origin top-left.
    kp_conf: (17,) keypoint confidences.

    Returns one of: doggy, cowgirl, reverse-cowgirl, missionary,
    standing, spoon, face-up, unknown.
    """
    # Aggregate landmark positions (None if not visible enough).
    shoulders = _avg(kp_xy, kp_conf, [KP_LSHOULDER, KP_RSHOULDER])
    hips = _avg(kp_xy, kp_conf, [KP_LHIP, KP_RHIP])
    knees = _avg(kp_xy, kp_conf, [KP_LKNEE, KP_RKNEE])
    ankles = _avg(kp_xy, kp_conf, [KP_LANKLE, KP_RANKLE])
    # Face visibility: need at least 2 of (nose, eyes) visible.
    face_pts_visible = sum(
        1 for i in (KP_NOSE, KP_LEYE, KP_REYE) if _visible(kp_conf, i)
    )
    face_visible = face_pts_visible >= 2

    # Need shoulders AND hips at minimum to say anything.
    if shoulders is None or hips is None:
        return "unknown"

    sh_x, sh_y = shoulders
    hp_x, hp_y = hips

    # Torso vector (shoulders -> hips). Use it to decide orientation.
    torso_dy = hp_y - sh_y  # +ve = hips below shoulders (image coords)
    torso_dx = hp_x - sh_x
    torso_len = float(np.hypot(torso_dx, torso_dy))
    if torso_len < 20.0:
        # Too compressed to read -- probably a tight close-up.
        return "unknown"

    # Verticality: angle between torso vector and vertical (y-axis).
    # cos(theta) = |dy| / len. cos near 1 -> vertical, near 0 -> horizontal.
    vertical_ness = abs(torso_dy) / torso_len  # 1.0 = perfectly vertical
    horizontal_ness = abs(torso_dx) / torso_len

    # ---- STANDING --------------------------------------------------------
    # Strong vertical torso + ankles below hips below shoulders + face up.
    if (vertical_ness > 0.85 and torso_dy > 0 and ankles is not None
            and ankles[1] > hp_y > sh_y):
        # Also: spread between shoulder y and ankle y should be > 2.5x
        # shoulder-hip dist (full body extended).
        if (ankles[1] - sh_y) > 2.0 * torso_len:
            return "standing"

    # ---- COWGIRL / REVERSE-COWGIRL ---------------------------------------
    # Torso roughly vertical (rider sits up), hips below shoulders,
    # knees in front of (i.e. roughly LEVEL with or slightly above) hips
    # in image. Distinguishing: face visible -> cowgirl, face hidden ->
    # reverse-cowgirl.
    if (vertical_ness > 0.7 and torso_dy > 0 and knees is not None
            and hp_y >= sh_y):
        knees_y = knees[1]
        # Knees should be near hip level (riding crouch) -- within
        # 0.6 * torso_len of hip y.
        if abs(knees_y - hp_y) < 0.6 * torso_len:
            if face_visible:
                return "cowgirl"
            else:
                return "reverse-cowgirl"

    # ---- DOGGY -----------------------------------------------------------
    # Body roughly horizontal (torso vector mostly along x), face NOT
    # visible (back-of-head / occluded), hips ~level with shoulders.
    if horizontal_ness > 0.6 and not face_visible:
        if abs(torso_dy) < 0.6 * torso_len:
            return "doggy"

    # ---- MISSIONARY ------------------------------------------------------
    # Body horizontal, face visible, knees raised (above hips in image
    # -> smaller y value). Person on their back, legs up.
    if horizontal_ness > 0.5 and face_visible and knees is not None:
        if knees[1] < hp_y - 0.2 * torso_len:
            return "missionary"

    # ---- SPOON -----------------------------------------------------------
    # Body horizontal, face partially visible (profile -- one ear or
    # one eye but not both), knees and ankles close together (legs
    # not spread).
    if horizontal_ness > 0.6 and knees is not None and ankles is not None:
        # Profile shot: exactly one ear visible.
        ears_visible = sum(1 for i in (KP_LEAR, KP_REAR) if _visible(kp_conf, i))
        knee_spread = 0.0
        if _visible(kp_conf, KP_LKNEE) and _visible(kp_conf, KP_RKNEE):
            knee_spread = abs(kp_xy[KP_LKNEE][0] - kp_xy[KP_RKNEE][0])
        if ears_visible == 1 and knee_spread < 0.15 * torso_len * 2:
            return "spoon"

    # ---- FACE-UP ---------------------------------------------------------
    # Catch-all for supine poses that didn't fit missionary: face visible,
    # head higher in image than hips, body not vertical, not classifiable
    # otherwise.
    if face_visible and sh_y < hp_y and horizontal_ness > 0.3:
        return "face-up"

    return "unknown"


def _analyze_one_file(model, path: str) -> list[str]:
    """Return list of position: tags this clip should get. Empty if none."""
    frames = _sample_keyframes(path)
    if not frames:
        return []

    # Run YOLO on each frame. Batch in one predict() call for GPU efficiency.
    try:
        results = model.predict(
            frames,
            imgsz=MODEL_IMGSZ,
            conf=CONF_THRESHOLD,
            verbose=False,
            device=None,  # use whatever the model was moved to
        )
    except Exception as e:
        logger.debug(f"YOLO predict failed for {path}: {e}")
        return []

    classifications: list[str] = []
    for res in results:
        kps = getattr(res, "keypoints", None)
        if kps is None:
            classifications.append("unknown")
            continue
        # res.keypoints.xy : tensor (n_persons, 17, 2)
        # res.keypoints.conf : tensor (n_persons, 17)
        try:
            xy = kps.xy.cpu().numpy() if hasattr(kps.xy, "cpu") else np.asarray(kps.xy)
            conf = (kps.conf.cpu().numpy() if hasattr(kps.conf, "cpu")
                    else np.asarray(kps.conf))
        except Exception:
            classifications.append("unknown")
            continue
        if xy is None or len(xy) == 0:
            classifications.append("unknown")
            continue
        # Multiple people -> skip this frame (group scene; let other tags handle it).
        if len(xy) >= 2:
            classifications.append("multi")
            continue
        # Single person.
        person_xy = xy[0]      # (17, 2)
        person_conf = conf[0]  # (17,)
        label = _classify_frame(person_xy, person_conf)
        classifications.append(label)

    # Aggregate. Drop "unknown" + "multi" from the vote.
    votes = [c for c in classifications if c not in ("unknown", "multi")]
    if not votes:
        return []
    # Majority rule with strong-agreement threshold.
    counts: dict[str, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    best_label, best_n = max(counts.items(), key=lambda kv: kv[1])
    if best_n >= MIN_AGREEMENT:
        return [f"position:{best_label}"]
    # Split decision -- some clear poses but no majority. Emit mixed
    # ONLY if we had >=2 clear (non-unknown) classifications, otherwise
    # the signal is too weak to assert anything.
    if len(votes) >= 2:
        return ["position:mixed"]
    return []


# ---- DB / batch driver -----------------------------------------------------


def _candidate_files(cur: sqlite3.Cursor, root: str | None,
                     force: bool, limit: int) -> list[str]:
    if root:
        like = root.replace("/", "\\") + "%"
        rows = cur.execute(
            "SELECT filepath FROM files WHERE filepath LIKE ? ORDER BY filepath",
            (like,),
        ).fetchall()
    else:
        rows = cur.execute(
            "SELECT filepath FROM files ORDER BY filepath"
        ).fetchall()
    candidates = [r[0] for r in rows]

    if not force:
        # Skip files that have ANY position:* tag — including the
        # `position:_scanned` sentinel for no-pose results — so re-runs
        # don't waste GPU re-decoding the same clips.
        already = {r[0] for r in cur.execute(
            "SELECT DISTINCT filepath FROM file_tags "
            "WHERE tag LIKE 'position:%'"
        ).fetchall()}
        before = len(candidates)
        candidates = [fp for fp in candidates if fp not in already]
        logger.info(f"already tagged (skipping): {before - len(candidates)}")

    if limit:
        candidates = candidates[:limit]
    return candidates


def tag_library(
    root: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    max_workers: int = 1,
    limit: int = 0,
) -> dict:
    """Batch-tag the whole library (or a sub-tree).

    Note: max_workers accepted for CLI symmetry with the other taggers but
    ignored -- pose inference is GPU-bound, parallel decoders just thrash."""
    if not DB_PATH.is_file():
        logger.error(f"no DB at {DB_PATH}")
        return {"ok": False, "error": "no db"}

    if not dry_run:
        bak = DB_PATH.with_suffix(".db3.bak.pose_tagger")
        shutil.copyfile(DB_PATH, bak)
        logger.info(f"DB backup -> {bak}")

    # Coexist with live VJ app + other taggers: WAL + 30s busy_timeout.
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    cur = conn.cursor()

    # Retry wrapper for DB writes — survives transient "database is locked"
    # errors when other tagger writers are also active.
    # Mirrors the pattern from clip_tagger.py:_with_retry. Up to 16
    # retries with exponential backoff up to 20s. Reraises on final
    # failure. (Budget raised 2026-05-19 after 8-retry exhausted under
    # fat-transaction contention from other taggers.)
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

    if force and not dry_run:
        n = cur.execute(
            "DELETE FROM file_tags WHERE tag LIKE 'position:%'"
        ).rowcount
        conn.commit()
        logger.info(f"force: dropped {n} existing position:* tags")

    candidates = _candidate_files(cur, root, force, limit)
    total = len(candidates)
    logger.info(f"will process: {total} files (single-worker, GPU-bound)")

    if total == 0:
        conn.close()
        return {"ok": True, "tagged": 0, "silent": 0, "failed": 0,
                "inserts": 0, "counts": {}}

    # Load model now (will download yolov8n-pose.pt on first run).
    try:
        model = _load_model()
    except Exception as e:
        logger.error(f"model load failed: {e}")
        conn.close()
        return {"ok": False, "error": f"model load failed: {e}"}

    tag_counts: dict[str, int] = {}
    n_inserted = 0
    n_tagged_files = 0
    n_skipped = 0
    n_failed = 0
    t0 = time.time()

    for i, fp in enumerate(candidates, 1):
        name = Path(fp).name
        if not Path(fp).is_file():
            n_failed += 1
            continue
        try:
            tags = _analyze_one_file(model, fp)
        except Exception as e:
            logger.debug(f"analysis failed for {fp}: {e}")
            n_failed += 1
            tags = None

        if tags is None:
            n_failed += 1
        elif not tags:
            n_skipped += 1
            # Insert a sentinel so we don't re-scan this file next run.
            # The candidates filter excludes any file with a position:*
            # tag, so position:_scanned is enough to skip permanently.
            if not dry_run:
                def _ins_sentinel(fp=fp):
                    cur.execute(
                        "INSERT OR IGNORE INTO file_tags(filepath, tag) "
                        "VALUES (?, ?)",
                        (fp, "position:_scanned"),
                    )
                    return cur.rowcount
                _with_retry(_ins_sentinel,
                            label=f"sentinel {fp[-30:]}")
        else:
            n_tagged_files += 1
            for t in tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
            if not dry_run:
                for t in tags:
                    def _ins(fp=fp, t=t):
                        cur.execute(
                            "INSERT OR IGNORE INTO file_tags(filepath, tag) "
                            "VALUES (?, ?)",
                            (fp, t),
                        )
                        return cur.rowcount
                    rc = _with_retry(_ins, label=f"insert {fp[-30:]}")
                    if rc:
                        n_inserted += 1

        # Progress: print every tagged file, brief no-tags line every 10th.
        if tags:
            print(f"[{i}/{total}] {name[:70]} -> {' '.join(tags)}")
        elif i % 10 == 0:
            print(f"[{i}/{total}] {name[:60]} -> (no tags)")

        # Commit every 10 files (matches the other taggers).
        if i % 10 == 0 and not dry_run:
            _with_retry(conn.commit, label=f"commit @ {i}")
        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate / 60 if rate > 0 else 0
            logger.info(
                f"  [{i}/{total}] {n_tagged_files} tagged / "
                f"{n_skipped} no-pose / {n_failed} fail  "
                f"({rate:.2f}/s, ~{eta:.1f}min remaining)"
            )

    if not dry_run:
        _with_retry(conn.commit, label="final commit")
    conn.close()

    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 56)
    logger.info(
        f"DONE: {n_tagged_files} got position tags / "
        f"{n_skipped} silent / {n_failed} failed in {elapsed:.1f}s"
    )
    logger.info(f"  tag rows inserted: {n_inserted}")
    print()
    print("=== position tag counts ===")
    for t, n in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {t}")
    return {
        "ok": True,
        "tagged": n_tagged_files,
        "silent": n_skipped,
        "failed": n_failed,
        "inserts": n_inserted,
        "counts": tag_counts,
    }


def list_positions() -> int:
    """Print all position:* tags currently in the DB and their counts."""
    if not DB_PATH.is_file():
        print(f"no DB at {DB_PATH}")
        return 1
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT tag, COUNT(*) FROM file_tags "
        "WHERE tag LIKE 'position:%' "
        "GROUP BY tag ORDER BY COUNT(*) DESC, tag ASC"
    ).fetchall()
    total = cur.execute(
        "SELECT COUNT(DISTINCT filepath) FROM file_tags "
        "WHERE tag LIKE 'position:%'"
    ).fetchone()[0]
    conn.close()
    if not rows:
        print("no position:* tags in DB yet")
        return 0
    print(f"=== position:* tags ({total} distinct files tagged) ===")
    for tag, n in rows:
        print(f"  {n:5d}  {tag}")
    return 0


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
    ap.add_argument("--dry-run", action="store_true",
                    help="analyse + print but do not write to DB")
    ap.add_argument("--force", action="store_true",
                    help="drop existing position:* tags first, then retag")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N files (0 = all)")
    ap.add_argument("--workers", type=int, default=1,
                    help="accepted for CLI symmetry; ignored (GPU-bound)")
    ap.add_argument("--list-positions", action="store_true",
                    help="print position:* tags currently in DB and exit")
    args = ap.parse_args()
    if args.list_positions:
        return list_positions()
    r = tag_library(
        root=args.root,
        dry_run=args.dry_run,
        force=args.force,
        max_workers=args.workers,
        limit=args.limit,
    )
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
