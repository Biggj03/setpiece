"""
vision_tag.py — batch visual auto-tagger for the whole library.

Walks the library, pulls a handful of keyframes from each video, runs the
WD14 ONNX tagger (SmilingWolf's booru-style content tagger) on CPU, and
merges the resulting tags into the same path_tags.db3 -> file_tags table
the iPad filter chips already read. The chips get smarter with zero app
changes.

Why WD14, why local, why CPU
----------------------------
* WD14 is trained on exactly this kind of content (anime + photo), so it
  produces real content tags (hair colour, setting, count of people,
  framing, ...) instead of the colour-histogram guesses ai_tagger.py's
  heuristic backend falls back to.
* Local because the library is adult content — the cloud vision APIs in
  ai_tagger.py would refuse it outright.
* CPU on purpose: the GPU stays 100% free for the live rig + the proxy
  transcode. ~0.2-0.6s/frame on CPU; the whole library is a ~2-hour
  idle job. Uses the onnxruntime / Pillow / numpy the project already
  ships — no new heavy deps.

Storage / safety
----------------
* Tags go into path_tags.db3 -> file_tags(filepath, tag). The chip UI
  reads that table already, so nothing in the app changes.
* We do NOT touch the `files` table. path_tags_v2's scanner only
  DELETE+reinserts file_tags when a file's mtime/size CHANGES; for a
  static media library that never happens, so vision tags persist.
  If a path rescan ever does wipe them (a genuine file edit), just
  re-run this script — the JSON cache makes unchanged files instant.
* Every result is cached to ~/.setpiece/vision_tags.json keyed by
  path + mtime + model, so the script is fully resumable / incremental.
  Ctrl-C any time; re-run to pick up where it stopped.

Usage
-----
    python vision_tag.py                   # library_root from settings.json
    python vision_tag.py "D:\\Recycle Bin"
    python vision_tag.py --dry-run         # scan + plan, no inference
    python vision_tag.py --limit 50        # first 50 only (testing)
    python vision_tag.py --frames 4 --threshold 0.35
    python vision_tag.py --retag           # recompute even cached files
    python vision_tag.py --no-db           # write the JSON cache only
"""

import argparse
import json
import os
import sqlite3
import struct  # noqa: F401  (kept for parity w/ sibling scripts; harmless)
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxy_cache import (  # noqa: E402
    FFMPEG_FLAGS,
    _ffmpeg_path,
    _ffprobe_path,
    _kill_process_tree,
)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

APP_STATE_DIR = Path.home() / ".setpiece"
SETTINGS_FILE = APP_STATE_DIR / "settings.json"
PATH_TAGS_DB = APP_STATE_DIR / "path_tags.db3"
CACHE_FILE = APP_STATE_DIR / "vision_tags.json"
MODEL_DIR = APP_STATE_DIR / "wd14"

# WD14 repos to try, best-first. Each ships model.onnx + selected_tags.csv.
# ViT-v3 first: good quality and ViT is well-behaved on CPU.
MODEL_REPOS = [
    "SmilingWolf/wd-vit-tagger-v3",
    "SmilingWolf/wd-swinv2-tagger-v3",
    "SmilingWolf/wd-convnext-tagger-v3",
    "SmilingWolf/wd-v1-4-convnext-tagger-v2",
    "SmilingWolf/wd-v1-4-moat-tagger-v2",
]

FRAME_EXTRACT_TIMEOUT_S = 30
PROBE_TIMEOUT_S = 15

# WD14 tags that carry no filter value for a VJ library AND no lyric-matching
# value (we keep over-applied tags like 1girl/breasts because they ARE the
# lyric anchors — "tits out" -> breasts. We only drop things that are pure
# capture artefacts, AI-noise meta tags, or overly-specific pose details
# that no song lyric will ever reach for).
_DROP_TAGS = {
    # Capture artefacts / image meta
    "realistic", "photorealistic", "lips", "blurry", "blurry-background",
    "depth-of-field", "motion-blur", "bokeh", "english-text", "text",
    "watermark", "signature", "artist-name", "web-address", "username",
    "censored", "mosaic-censoring", "bar-censor", "logo", "letterboxed",
    "chromatic-aberration", "film-grain", "jpeg-artifacts",
    # Pose details that no lyric maps to
    "closed-eyes", "looking-at-viewer", "looking-down", "looking-up",
    "looking-back", "looking-away", "looking-to-the-side", "half-closed-eyes",
    "rolling-eyes", "one-eye-closed",
    # Background meta — not filter-useful
    "simple-background", "white-background", "black-background",
    "grey-background", "gradient-background", "blurry-foreground",
    "multiple-views", "white-space",
    # AI noise / WD14 quality-meta
    "bad-anatomy", "bad-id", "bad-hands", "bad-eyes",
    "absurdres", "highres", "lowres", "low-quality", "medium-quality",
    "worst-quality", "best-quality", "ugly", "cropped",
    # Overly-specific facial markers that bloat the orphan tail
    "mole", "mole-under-eye", "mole-under-mouth", "mole-on-breast",
    "mole-on-cheek", "birthmark", "facial-mark",
    # Solo-focus duplicates with 'solo'; eyelashes etc. are too granular
    "solo-focus", "eyelashes", "artist-self-insert",
    # Audit-flagged: anime / booru-only tags that misfire on live-action
    # adult content. WD14 inherits these from its anime-trained vocab;
    # adversarial cross-check (audit_tags.py) showed low cross-model
    # agreement AND wrong-context application. Dropped from DB + the
    # _DROP_TAGS list so future tagger runs don't re-add them.
    "loli", "yaoi", "bara", "nose",
}


# Windows: stdout/stderr default to cp1252, which can't encode the
# unicode characters that turn up in some filenames (e.g. fullwidth
# vertical bar U+FF5C). Reconfigure to utf-8 with replace-on-fail so a
# single funky filename can't kill the whole multi-hour run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def log(msg: str) -> None:
    line = f"[vtag {datetime.now():%H:%M:%S}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        # Belt-and-suspenders if the reconfigure above ever fails on a
        # weird build — sanitize and re-emit so the process never dies
        # while just trying to print a status line.
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)


def _canon(p) -> str:
    """Lowercased resolved path — stable dict key across path-spelling."""
    try:
        return str(Path(p).resolve()).lower()
    except Exception:
        return str(p).lower()


# ── settings / cache ───────────────────────────────────────────────────

def _load_settings() -> dict:
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_cache() -> dict:
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=1, sort_keys=True)
        os.replace(tmp, CACHE_FILE)
    except Exception as e:
        log(f"  warn: cache flush failed: {e}")


# ── WD14 model ─────────────────────────────────────────────────────────

def _ensure_model(repo_override: str | None) -> tuple:
    """Download (once) + return (onnx_path, tags_csv_path, repo_id)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub not installed — `pip install huggingface_hub`"
        ) from e
    repos = [repo_override] if repo_override else MODEL_REPOS
    last_err = None
    for repo in repos:
        safe = repo.replace("/", "__")
        dest = MODEL_DIR / safe
        dest.mkdir(parents=True, exist_ok=True)
        try:
            onnx = hf_hub_download(repo, "model.onnx", local_dir=str(dest))
            csv = hf_hub_download(repo, "selected_tags.csv", local_dir=str(dest))
            log(f"model ready: {repo}")
            return Path(onnx), Path(csv), repo
        except Exception as e:
            last_err = e
            log(f"  {repo}: unavailable ({e})")
            continue
    raise RuntimeError(f"could not fetch any WD14 model — last error: {last_err}")


def _load_tags(csv_path: Path) -> list:
    """selected_tags.csv -> list of (name, category) in model-output order.
    category: 0=general, 4=character, 9=rating."""
    import csv as _csv
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = _csv.DictReader(fh)
        for r in reader:
            name = (r.get("name") or "").strip()
            try:
                cat = int(r.get("category") or 0)
            except Exception:
                cat = 0
            rows.append((name, cat))
    return rows


def _load_session(onnx_path: Path):
    """CPU onnxruntime session + (input_name, output_name, target_size)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    # input shape is [N, H, W, 3]; H/W may be ints or symbolic strings.
    shape = inp.shape
    size = 448
    try:
        h = shape[1]
        if isinstance(h, int) and h > 0:
            size = h
    except Exception:
        pass
    return sess, inp.name, out.name, size


def _prepare(pil_img, size: int):
    """WD14 preprocessing: square white-pad -> resize -> BGR -> NHWC f32,
    values left in 0-255 (WD14 expects un-normalised input)."""
    import numpy as np
    from PIL import Image
    img = pil_img.convert("RGB")
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    canvas = canvas.resize((size, size), Image.BICUBIC)
    arr = np.asarray(canvas, dtype=np.float32)      # HWC, RGB, 0-255
    arr = arr[:, :, ::-1]                           # RGB -> BGR
    return np.ascontiguousarray(np.expand_dims(arr, 0))  # 1HWC


def _norm_tag(raw: str) -> str:
    """booru tag -> chip-style token: lowercase, '_'->'-', drop parens."""
    t = raw.strip().lower().replace("_", "-")
    t = t.replace("(", "").replace(")", "").replace("'", "")
    t = "".join(c for c in t if c.isalnum() or c == "-")
    while "--" in t:
        t = t.replace("--", "-")
    return t.strip("-")


# ── frame extraction ───────────────────────────────────────────────────

def _probe_duration(src: Path) -> float:
    try:
        proc = subprocess.Popen(
            [_ffprobe_path(), "-v", "error", "-show_entries",
             "format=duration", "-of", "default=nw=1:nk=1", str(src)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=FFMPEG_FLAGS,
        )
        out, _ = proc.communicate(timeout=PROBE_TIMEOUT_S)
        return max(0.0, float((out or b"").decode("utf-8", "replace").strip()))
    except Exception:
        try:
            _kill_process_tree(proc)  # type: ignore[name-defined]
        except Exception:
            pass
        return 0.0


def _extract_frames(src: Path, n: int, tmpdir: Path) -> list:
    """Pull n keyframes spread across the file, skipping the intro.
    Returns a list of PIL Images (may be shorter than n on partial fails)."""
    from PIL import Image
    dur = _probe_duration(src)
    if dur > 1.0:
        # Evenly spaced inside [12%, 88%] — dodges intro cards + end cards.
        lo, hi = dur * 0.12, dur * 0.88
        if n == 1:
            stamps = [(lo + hi) / 2]
        else:
            step = (hi - lo) / (n - 1)
            stamps = [lo + i * step for i in range(n)]
    else:
        # Duration probe failed — fall back to fixed early offsets.
        stamps = [6.0, 18.0, 35.0, 55.0][:n] or [6.0]

    images = []
    for i, t in enumerate(stamps):
        out = tmpdir / f"f{i}.jpg"
        cmd = [
            _ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{t:.3f}", "-i", str(src),
            "-frames:v", "1", "-q:v", "3", str(out),
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=FFMPEG_FLAGS,
            )
            proc.communicate(timeout=FRAME_EXTRACT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            continue
        except Exception:
            continue
        if out.exists() and out.stat().st_size > 0:
            try:
                with Image.open(out) as im:
                    images.append(im.copy())
            except Exception:
                pass
            try:
                out.unlink()
            except Exception:
                pass
    return images


# ── tagging ────────────────────────────────────────────────────────────

def _tag_images(sess, inp_name, out_name, size, tags, images,
                threshold: float, max_tags: int) -> list:
    """Run WD14 over the frames; aggregate by MAX confidence per tag.
    Returns chip-style tags sorted by confidence desc, capped."""
    import numpy as np
    if not images:
        return []
    best: dict[str, float] = {}
    for im in images:
        batch = _prepare(im, size)
        probs = sess.run([out_name], {inp_name: batch})[0][0]
        for (name, cat), p in zip(tags, probs):
            p = float(p)
            # Ratings (cat 9) want a firmer threshold — they're 1-of-4ish.
            thr = 0.50 if cat == 9 else threshold
            if p < thr:
                continue
            tok = _norm_tag(name)
            if len(tok) < 2 or tok in _DROP_TAGS:
                continue
            if p > best.get(tok, 0.0):
                best[tok] = p
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return [tok for tok, _ in ranked[:max_tags]]


# ── DB merge ───────────────────────────────────────────────────────────

class TagDB:
    """Thin writer over path_tags.db3 -> file_tags. Adds vision tags
    without touching the `files` table (see module docstring)."""

    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path), isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        # file_tags already exists if path_tags ever scanned; create
        # defensively so a never-scanned library still works.
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS file_tags ("
            "filepath TEXT NOT NULL, tag TEXT NOT NULL, "
            "PRIMARY KEY (filepath, tag))"
        )

    def replace_tags(self, filepath: str, old_tags: list, new_tags: list) -> None:
        self.conn.execute("BEGIN")
        try:
            for t in old_tags:
                self.conn.execute(
                    "DELETE FROM file_tags WHERE filepath=? AND tag=?",
                    (filepath, t),
                )
            self.conn.executemany(
                "INSERT OR IGNORE INTO file_tags(filepath, tag) VALUES (?, ?)",
                [(filepath, t) for t in new_tags],
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


# ── main ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default=None,
                    help="folder to scan (default: library_root from settings.json)")
    ap.add_argument("--frames", type=int, default=3,
                    help="keyframes sampled per video (default 3)")
    ap.add_argument("--threshold", type=float, default=0.35,
                    help="WD14 confidence threshold for general tags (default 0.35)")
    ap.add_argument("--max-tags", type=int, default=20,
                    help="max tags kept per video (default 20)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N videos (0 = no limit; for testing)")
    ap.add_argument("--model", default=None,
                    help="override WD14 repo id (default: auto best-first)")
    ap.add_argument("--retag", action="store_true",
                    help="recompute even files already in the cache")
    ap.add_argument("--no-db", action="store_true",
                    help="write the JSON cache only, skip the path_tags.db3 merge")
    ap.add_argument("--dry-run", action="store_true",
                    help="scan + print the plan, run no inference")
    args = ap.parse_args()

    # ── resolve library root ────────────────────────────────────────────
    root_str = args.source or _load_settings().get("library_root")
    if not root_str:
        log("no source given and no library_root in settings.json — aborting")
        return 2
    root = Path(root_str)
    if not root.is_dir():
        log(f"source folder not found: {root}")
        return 2

    # ── scan ────────────────────────────────────────────────────────────
    log(f"scanning {root} recursively ...")
    files = []
    for f in root.rglob("*"):
        try:
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                files.append(f)
        except Exception:
            continue
    files.sort(key=lambda p: str(p).lower())
    log(f"{len(files)} video file(s) found")

    cache = _load_cache()
    model_id = args.model or MODEL_REPOS[0]

    # Partition into already-done vs to-do (resumable).
    todo = []
    cached_ok = 0
    for f in files:
        key = _canon(f)
        try:
            mtime = f.stat().st_mtime
        except Exception:
            continue
        ent = cache.get(key)
        fresh = (
            ent is not None
            and not args.retag
            and abs(float(ent.get("mtime", -1)) - mtime) < 1.0
            and ent.get("model") == model_id
        )
        if fresh:
            cached_ok += 1
        else:
            todo.append((f, key, mtime))
    if args.limit > 0:
        todo = todo[: args.limit]
    log(f"plan: {cached_ok} already cached, {len(todo)} to tag"
        + (f" (capped to --limit {args.limit})" if args.limit else ""))

    if args.dry_run:
        for f, _, _ in todo[:15]:
            log(f"  would tag: {f.name}")
        if len(todo) > 15:
            log(f"  ... +{len(todo) - 15} more")
        log("dry run — nothing transcoded, nothing written")
        return 0
    if not todo:
        log("nothing to do — everything already cached")
        return 0

    # ── load model ──────────────────────────────────────────────────────
    onnx_path, csv_path, repo = _ensure_model(args.model)
    model_id = repo
    tags = _load_tags(csv_path)
    sess, inp_name, out_name, size = _load_session(onnx_path)
    log(f"WD14 loaded: {len(tags)} tags, input {size}x{size}, CPU inference")

    db = None
    if not args.no_db:
        if not PATH_TAGS_DB.exists():
            log(f"warn: {PATH_TAGS_DB} not found — path_tags hasn't scanned "
                "this library yet. Writing cache only; re-run after a scan, "
                "or pass --no-db to silence this.")
        else:
            db = TagDB(PATH_TAGS_DB)
            log(f"merging into {PATH_TAGS_DB}")

    tmpdir = Path(tempfile.mkdtemp(prefix="vtag_"))
    tagged = no_frames = no_tags = errors = 0
    t_start = time.time()
    try:
        for idx, (f, key, mtime) in enumerate(todo, 1):
            try:
                images = _extract_frames(f, args.frames, tmpdir)
                if not images:
                    no_frames += 1
                    log(f"  [{idx}/{len(todo)}] NO FRAMES  {f.name}")
                    continue
                new_tags = _tag_images(
                    sess, inp_name, out_name, size, tags, images,
                    args.threshold, args.max_tags,
                )
                if not new_tags:
                    no_tags += 1
                old_tags = []
                ent = cache.get(key)
                if ent and isinstance(ent.get("tags"), list):
                    old_tags = [str(t) for t in ent["tags"]]
                if db is not None:
                    db.replace_tags(str(f), old_tags, new_tags)
                cache[key] = {
                    "path": str(f),
                    "mtime": mtime,
                    "model": model_id,
                    "tags": new_tags,
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
                tagged += 1
                if idx % 25 == 0 or idx == len(todo):
                    rate = idx / max(0.01, time.time() - t_start)
                    eta = (len(todo) - idx) / max(0.01, rate)
                    log(f"  [{idx}/{len(todo)}] {rate:.1f} vid/s  "
                        f"ETA {eta/60:.0f}m  last: {f.name[:48]} "
                        f"-> {', '.join(new_tags[:6])}")
                if idx % 50 == 0:
                    _save_cache(cache)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                errors += 1
                log(f"  [{idx}/{len(todo)}] ERROR {f.name}: {e}")
    except KeyboardInterrupt:
        log("interrupted — flushing cache + DB ...")
    finally:
        _save_cache(cache)
        if db is not None:
            db.close()
        try:
            for leftover in tmpdir.glob("*"):
                leftover.unlink(missing_ok=True)
            tmpdir.rmdir()
        except Exception:
            pass

    dt = time.time() - t_start
    log("=" * 56)
    log(f"FINISHED in {dt/60:.1f}m — tagged: {tagged}  no-frames: {no_frames}  "
        f"empty-result: {no_tags}  errors: {errors}")
    log(f"cache: {CACHE_FILE} ({len(cache)} entries total)")
    if db is not None:
        log("vision tags merged into path_tags.db3 — restart the app / "
            "rebuild the filter to see the new chips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
