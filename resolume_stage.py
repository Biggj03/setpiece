"""
Stage a working set of media files into Resolume Arena's clip grid (REST).

This is the SETUP half that pairs with resolume_out.py's runtime OSC:
Arena can only fire clips its grid already holds, so before a show (or
when the operator switches material) the working set gets loaded into
grid slots over REST and registered filepath -> (layer, column) on the
bridge. After that `bridge.fire_clip(path)` resolves to the right slot,
and resolume_dynamic.DynamicStager uses the same `open_into_slot` seam
to stage off-pool picks on a MISS.

CONTRACT WITH ARENA (proven on Arena 7.25.2):
  Load a file into a clip slot:
    POST /api/v1/composition/layers/{L}/clips/{C}/open
    body (raw text): file:///C:/path.mp4   (3 slashes, %-encoded, fwd slashes)
  Read state:
    GET  /api/v1/composition[/layers/{L}[/clips/{C}]]

GRID MODEL (v1 — single staging layer):
  Lay the working set across columns of ONE layer (default 1). Arena
  does gapless clip->clip on a layer with the layer's transition, and
  auto-grows columns when you `open` a clip past the current count, so
  a 40-clip working set just works on a fresh comp. Multi-layer
  compositing (bed + hero + overlay) reuses this loader per layer.

THE TWIN-SWAP GUARD (gig-safety):
  Arena 7.25.2 cannot `open` some containers it otherwise decodes —
  Matroska (.mkv) is rejected even when the video inside is H.264, and
  VP9 .webm is rejected on codec (both return HTTP 412). Worse, some
  .mkv files LOAD without error but render black — no throw, just a
  dead hero mid-set. The discipline: remux/transcode same-stem .mp4
  twins next to such files at library-prep time, and `arena_safe_path`
  swaps to the twin at staging time. Callers still key their mapping by
  the ORIGINAL path (what the show driver fires); only the file Arena
  loads is swapped.

Stdlib only (urllib). Never raises into the caller on a single-clip
failure — it logs, skips, and stages the rest, returning the slots that
succeeded. Also runnable as a CLI for deck prep:

    python resolume_stage.py path/to/clips --layer 1
"""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_REST = "http://localhost:8080/api/v1"

# Containers Arena 7.25.2 won't `open` (HTTP 412) — keep .mp4 twins of
# these next to the originals at library-prep time:
#   .mkv  -> container reject even for H.264 inside (lossless remux)
#   .webm -> VP9 codec reject (transcode)
ARENA_UNOPENABLE = {".mkv", ".webm"}

# What the CLI considers stageable media when given a folder.
MEDIA_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".wmv", ".mpg", ".mpeg",
              ".mkv", ".webm"}


def file_uri(path: str) -> str:
    """Windows-correct file URI: exactly 3 slashes, forward slashes,
    percent-encoded path. Arena's `open` endpoint takes this as the raw
    POST body."""
    p = str(path).replace("\\", "/")
    return "file:///" + urllib.parse.quote(p)


def arena_safe_path(path) -> str:
    """Resolve a media path to one Arena can actually open.

    If the file's container is in ARENA_UNOPENABLE and a same-stem .mp4
    twin exists beside it, return the twin; otherwise return the input
    unchanged (Arena 412s the bad file, which the stager logs and skips
    — better than silently dropping a clip to black mid-set).
    """
    try:
        p = Path(path)
        if p.suffix.lower() in ARENA_UNOPENABLE:
            twin = p.with_suffix(".mp4")
            if twin.exists():
                logger.debug("Arena: %s -> .mp4 twin for %s",
                             p.suffix, p.name)
                return str(twin)
    except Exception:
        pass
    return str(path)


class ResolumeStager:
    """REST client that loads files into Arena's grid and reports slots."""

    def __init__(self, rest_base: str = DEFAULT_REST, timeout: float = 12.0):
        self._rest = rest_base.rstrip("/")
        self._timeout = float(timeout)

    # -- low-level REST -----------------------------------------------------

    def _get(self, path: str) -> Optional[dict]:
        try:
            with urllib.request.urlopen(f"{self._rest}/{path}",
                                        timeout=self._timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            logger.debug("Arena GET %s failed: %s", path, e)
            return None

    def reachable(self) -> bool:
        """True if Arena's REST API answers (webserver enabled)."""
        p = self._get("product")
        return bool(p and p.get("name"))

    def open_into_slot(self, layer: int, column: int, filepath: str) -> bool:
        """Load one file into (layer, column). Returns True on success.

        Runs the twin-swap guard first; the caller's mapping still uses
        the original path it passed in. This is the seam
        resolume_dynamic.DynamicStager stages through on a MISS.
        """
        uri = file_uri(arena_safe_path(filepath))
        req = urllib.request.Request(
            f"{self._rest}/composition/layers/{int(layer)}"
            f"/clips/{int(column)}/open",
            data=uri.encode("utf-8"), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                # 200/204 both signal success across Arena builds.
                return r.status in (200, 204)
        except urllib.error.HTTPError as e:
            logger.warning("Arena open L%sC%s failed HTTP %s: %s",
                           layer, column, e.code, filepath)
            return False
        except Exception as e:
            logger.warning("Arena open L%sC%s failed: %s", layer, column, e)
            return False

    def clear_slot(self, layer: int, column: int) -> bool:
        """Remove the clip in (layer, column). Returns True on success.

        Uses the PER-CLIP clear endpoint — the layer-level
        `layers/{L}/clear` returns 204 but does not actually remove
        clips (they silently survive until something loads over them);
        per-clip clear genuinely removes. This is the undo for
        open_into_slot, e.g. cleaning up a probe clip after a check.
        """
        req = urllib.request.Request(
            f"{self._rest}/composition/layers/{int(layer)}"
            f"/clips/{int(column)}/clear",
            data=b"", method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                return r.status in (200, 204)
        except Exception as e:
            logger.warning("Arena clear L%sC%s failed: %s", layer, column, e)
            return False

    def slot_loaded(self, layer: int, column: int) -> bool:
        """True if the slot now holds a file (fileinfo present). Use as a
        readback check after open_into_slot — Arena can 204 an `open` and
        still leave the slot empty in rare races."""
        clip = self._get(f"composition/layers/{int(layer)}"
                         f"/clips/{int(column)}")
        if not clip:
            return False
        video = clip.get("video") if isinstance(clip.get("video"), dict) else {}
        fi = video.get("fileinfo")
        return bool(fi and (fi.get("exists") if isinstance(fi, dict) else fi))

    # -- high-level staging -------------------------------------------------

    def stage_working_set(self, filepaths: list, layer: int = 1,
                          start_col: int = 1, verify: bool = False) -> dict:
        """Load `filepaths` onto `layer`, one per column from start_col.

        Returns {filepath: (layer, column)} for the clips that loaded —
        feed it straight to the bridge's clip registry. Skips files that
        fail to load (logged) rather than aborting the whole stage; the
        failed file's column is left behind so a later re-stage of the
        same list lands in the same slots.

        verify=True does a readback GET per clip to confirm fileinfo
        landed (slower; use for a setup check, not a hot re-stage).
        """
        mapping: dict = {}
        col = int(start_col)
        for fp in filepaths:
            if not fp:
                continue
            ok = self.open_into_slot(layer, col, fp)
            if ok and verify:
                ok = self.slot_loaded(layer, col)
            if ok:
                mapping[str(fp)] = (int(layer), col)
            col += 1
        logger.info("Resolume staged %d/%d clips onto layer %d",
                    len(mapping), len([f for f in filepaths if f]), layer)
        return mapping


def collect_media(paths: list) -> list:
    """Expand a mix of files and folders into a sorted list of stageable
    media file paths (folders are scanned one level, by MEDIA_EXTS)."""
    out: list = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(str(f) for f in p.iterdir()
                              if f.is_file()
                              and f.suffix.lower() in MEDIA_EXTS))
        elif p.is_file():
            out.append(str(p))
        else:
            logger.warning("skipping missing path: %s", raw)
    return out


def main(argv) -> int:
    """CLI for deck prep: stage files/folders onto a layer.

    Examples:
        python resolume_stage.py path/to/clips --layer 1
        python resolume_stage.py a.mp4 b.mp4 --layer 2 --start-col 5 --verify
    """
    import argparse
    ap = argparse.ArgumentParser(
        description="Stage media files into Resolume Arena's clip grid.")
    ap.add_argument("paths", nargs="+",
                    help="media files and/or folders (folders scanned "
                         "one level for media)")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--start-col", type=int, default=1)
    ap.add_argument("--rest", default=DEFAULT_REST,
                    help=f"Arena REST base (default {DEFAULT_REST})")
    ap.add_argument("--verify", action="store_true",
                    help="readback-check each slot after open")
    a = ap.parse_args(argv[1:])

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    files = collect_media(a.paths)
    if not files:
        print("nothing to stage (no media files found)")
        return 1
    st = ResolumeStager(rest_base=a.rest)
    if not st.reachable():
        print(f"Arena REST not reachable at {a.rest} — is Arena running "
              "with the webserver enabled?")
        return 2
    mapping = st.stage_working_set(files, layer=a.layer,
                                   start_col=a.start_col, verify=a.verify)
    for fp, (ly, col) in mapping.items():
        print(f"  L{ly}C{col:<3} {Path(fp).name}")
    failed = len(files) - len(mapping)
    print(f"staged {len(mapping)}/{len(files)} clips onto layer {a.layer}"
          + (f" ({failed} failed — see log)" if failed else ""))
    return 0 if mapping else 1


# ---------------------------------------------------------------------------
# Self-test — offline (no Arena needed). Pins the URI rule, the twin-swap
# guard, and the no-Arena graceful degradation.
# ---------------------------------------------------------------------------

def _self_test() -> None:
    # Windows file URI: 3 slashes, fwd slashes, encoded spaces + specials.
    u = file_uri(r"C:\Media Library\Show Reel #2 [Final].mp4")
    assert u.startswith("file:///C"), u
    assert "\\" not in u
    assert "%20" in u            # space encoded
    assert "%23" in u            # '#' encoded
    assert u.count("/") >= 4     # file:/// + path separators

    # Unreachable Arena => reachable() False, staging returns {} (no raise).
    s = ResolumeStager(rest_base="http://127.0.0.1:9/api/v1", timeout=0.3)
    assert s.reachable() is False
    assert s.stage_working_set(["x.mp4", "y.mp4"]) == {}

    # arena_safe_path: .mkv/.webm swap to their .mp4 twin only when the
    # twin exists; everything else passes through unchanged.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        twin_mkv = Path(d) / "Has Twin [x].mkv"
        twin_mp4 = Path(d) / "Has Twin [x].mp4"
        lone_mkv = Path(d) / "No Twin.mkv"
        webm_t = Path(d) / "Webm Twin.webm"
        webm_mp4 = Path(d) / "Webm Twin.mp4"
        for f in (twin_mkv, twin_mp4, lone_mkv, webm_t, webm_mp4):
            f.write_bytes(b"")
        assert arena_safe_path(str(twin_mkv)) == str(twin_mp4)
        assert arena_safe_path(str(webm_t)) == str(webm_mp4)
        assert arena_safe_path(str(lone_mkv)) == str(lone_mkv)
        assert arena_safe_path(str(twin_mp4)) == str(twin_mp4)
        assert arena_safe_path(r"E:\x\song.mov") == r"E:\x\song.mov"

    print("resolume_stage._self_test: OK")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sys.exit(main(sys.argv))
    logging.basicConfig(level=logging.INFO)
    _self_test()
