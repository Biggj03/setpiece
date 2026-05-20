"""
One-command ingest pipeline for newly-added videos.

Tagging a library means running several analysis passes:
  1. Path-token scan (folder/filename tokens -> searchable tags)
  2. Visual-attribute taggers (color, geometry, motion, symmetry, pose)
  3. Segment/intro detection
  4. Semantic CLIP embeddings for free-text search

`ingest.py` runs the whole chain. Every step is idempotent + resumable
(each underlying tagger skips work it has already done), so you can run
this after every batch of new clips and only pay for the new files.

USAGE
-----
    # Default: scan the configured library root with the full pipeline
    python ingest.py --root "/path/to/clips"

    # Just one stage (when re-running after a fix)
    python ingest.py --root "/path/to/clips" --only color

    # Skip stages
    python ingest.py --root "/path/to/clips" --skip pose --skip vision

    # Dry-run: show what would happen, don't write
    python ingest.py --root "/path/to/clips" --dry-run

STAGES
------
  index     - path_tags folder/filename token scan
  color     - dominant-color / palette tagging
  geometry  - edge-density / line-direction tagging
  motion    - frame-to-frame energy / movement tagging
  symmetry  - left/right SSIM symmetry tagging
  pose      - coarse subject-placement (pose estimation)
  intro     - opening-segment detection
  vision    - CLIP semantic embeddings (GPU; slowest per-file)

Each stage's "done" state lives in the tag DB, so re-running is cheap.
The index step is INLINE - uses path_tags.PathTagIndex directly so we
get a live progress report instead of shelling out.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Stage names (must be unique + stable; --only and --skip take these).
# Order matters: index first so later taggers see path tokens, vision
# last because the GPU embedding pass is the slowest per file.
STAGES = [
    "index", "color", "geometry", "motion",
    "symmetry", "pose", "intro", "vision",
]

# Per-stage tagger script + how it takes the library root.
#   "root-flag"  -> python <script> --root <root>
#   "root-arg"   -> python <script> <root>
#   "no-root"    -> python <script>          (reads files from the tag DB)
_STAGE_SCRIPTS = {
    "color":    ("color_tagger.py", "root-flag"),
    "geometry": ("geometry_tagger.py", "root-flag"),
    "motion":   ("motion_tagger.py", "root-flag"),
    "symmetry": ("symmetry_tagger.py", "root-flag"),
    "pose":     ("pose_tagger.py", "root-flag"),
    "intro":    ("intro_tagger.py", "no-root"),
    "vision":   ("vision_tag.py", "root-arg"),
}

HERE = Path(__file__).parent
PYTHON = sys.executable


def _run(cmd: list[str], *, label: str, dry_run: bool = False) -> int:
    """Run a subprocess, stream its output, return rc."""
    logger.info("")
    logger.info("=" * 64)
    logger.info(f"  >>> {label}")
    logger.info(f"      cmd: {' '.join(cmd)}")
    logger.info("=" * 64)
    if dry_run:
        logger.info("  [DRY] skipped")
        return 0
    t0 = time.time()
    try:
        rc = subprocess.call(cmd, cwd=str(HERE))
    except KeyboardInterrupt:
        logger.warning(f"  ! {label} interrupted by user")
        raise
    dt = time.time() - t0
    if rc == 0:
        logger.info(f"  OK {label} done in {dt:.1f}s")
    else:
        logger.error(f"  FAIL {label} rc={rc} after {dt:.1f}s")
    return rc


def stage_index(root: str, dry_run: bool) -> int:
    """Inline: walk root, tokenize paths, upsert into path_tags.db3."""
    logger.info("")
    logger.info("=" * 64)
    logger.info(f"  >>> INDEX: path_tags scan of {root}")
    logger.info("=" * 64)
    if dry_run:
        logger.info("  [DRY] skipped")
        return 0
    t0 = time.time()
    try:
        from path_tags import PathTagIndex
    except Exception as e:
        logger.error(f"  could not import PathTagIndex: {e}")
        return 1
    idx = PathTagIndex()
    if not idx.scan_async(root):
        logger.error("  scan_async rejected (already scanning, or bad root)")
        return 1
    # Block here until scan finishes -- caller wants a synchronous pipeline.
    while idx.is_scanning():
        time.sleep(1.0)
    dt = time.time() - t0
    logger.info(f"  OK INDEX done in {dt:.1f}s "
                f"(total files in DB now: {idx.total_files()})")
    return 0


def stage_tagger(stage: str, root: str, dry_run: bool) -> int:
    """Shell out to a tagger script, passing the root as that script
    expects it."""
    script, root_mode = _STAGE_SCRIPTS[stage]
    cmd = [PYTHON, script]
    if root_mode == "root-flag":
        cmd += ["--root", root]
    elif root_mode == "root-arg":
        cmd.append(root)
    # "no-root": tagger reads its file list from the tag DB.
    return _run(cmd, label=f"{stage.upper()}: {script}", dry_run=dry_run)


def run_pipeline(args: argparse.Namespace) -> int:
    root = args.root

    requested = list(STAGES)
    if args.only:
        bad = [s for s in args.only if s not in STAGES]
        if bad:
            logger.error(f"unknown stage(s) in --only: {bad}")
            return 2
        requested = list(args.only)
    if args.skip:
        bad = [s for s in args.skip if s not in STAGES]
        if bad:
            logger.error(f"unknown stage(s) in --skip: {bad}")
            return 2
        requested = [s for s in requested if s not in args.skip]

    logger.info("")
    logger.info("#" + "=" * 62 + "#")
    logger.info(f"# INGEST  root: {root}")
    logger.info(f"# stages: {', '.join(requested)}")
    if args.dry_run:
        logger.info("# [DRY RUN]")
    logger.info("#" + "=" * 62 + "#")

    if not args.dry_run and not Path(root).is_dir():
        logger.error(f"root does not exist: {root}")
        return 1

    t0 = time.time()
    failures = []

    for stage in requested:
        if stage == "index":
            rc = stage_index(root, args.dry_run)
        elif stage in _STAGE_SCRIPTS:
            rc = stage_tagger(stage, root, args.dry_run)
        else:
            logger.warning(f"unknown stage '{stage}' -- skipped")
            continue
        if rc != 0:
            failures.append(stage)
            if not args.continue_on_error:
                logger.error(f"stage {stage} failed and --continue-on-error "
                             f"not set -- aborting pipeline")
                break

    dt = time.time() - t0
    logger.info("")
    logger.info("=" * 64)
    if failures:
        logger.error(f"INGEST FINISHED in {dt:.1f}s with FAILURES: {failures}")
        return 1
    logger.info(f"INGEST COMPLETE in {dt:.1f}s -- all stages OK")
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
    ap.add_argument(
        "--root", required=True,
        help="library root to ingest",
    )
    ap.add_argument(
        "--only", action="append", choices=STAGES, default=None,
        help="run ONLY this stage; repeat the flag for multiple stages "
             "(e.g. --only index --only color)",
    )
    ap.add_argument(
        "--skip", action="append", choices=STAGES, default=None,
        help="skip this stage; repeat the flag for multiple stages "
             "(e.g. --skip pose --skip vision)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="print what would run, don't actually execute",
    )
    ap.add_argument(
        "--continue-on-error", action="store_true",
        help="if a stage fails, keep going (default: abort)",
    )
    args = ap.parse_args()
    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        logger.warning("ingest cancelled (Ctrl-C)")
        return 130


if __name__ == "__main__":
    sys.exit(main())
