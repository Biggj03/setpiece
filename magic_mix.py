"""magic_mix.py — "what should I play next" recommender for setpiece.

Pure stdlib. Read-only access to the existing Setpiece data sources. The intent
is that main.py (or a UI panel / Stream Deck button) instantiates one
``MagicMix`` at startup and calls ``suggest_next`` whenever it needs ideas
for the next fire.

Design notes:
    * Tag overlap uses Jaccard on the *union* of two tag sources per file:
      path-derived tags from path_tags.db3 (lots of files have these from
      auto-indexing) AND clip-level tags from clips.json (manual / AI).
    * BPM proximity uses ``capture_bpm`` first (the BPM the clip was
      recorded against), falling back to ``bpm`` (detected). Either being
      0 means "unknown" → neutral 0.5 instead of penalising the candidate.
    * Recency penalty fires whenever the candidate path appears anywhere in
      the recent_fires window. We don't compare by clip id because the
      scratch basket only knows file paths.
    * Variety bonus rewards candidates that bring at least one tag the
      LIVE clip doesn't have. Capped so it can't dominate tag overlap.
    * Star bonus is per-file: if ANY saved clip from that file is starred,
      the file gets the bonus. Files with no saved clips simply miss out.

Path matching quirks (Windows):
    * scratch.json and clips.json store paths with mixed slashes
      (``E:/...`` in some entries, ``E:\\...`` in others).
    * path_tags.db3 stores whatever was indexed — typically backslash form.
    * We normalise on read by lower-casing and using ``os.path.normpath``
      so the three sources can be cross-referenced reliably.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# How long a reload() stays "fresh". The iPad polls /api/magic/suggest
# every ~15s and a full reload() re-reads clips.json + scratch.json +
# the ENTIRE path_tags file_tags table — wasteful to do on every poll.
# The recommender's inputs change slowly, so ~60s staleness is invisible
# for a "what to play next" suggestion. (Audit fix M13.)
RELOAD_TTL_SEC = 60.0


# ---------- weights (tuned per the spec; kept as module constants so the
# UI / a future settings panel can override them without rewriting code) ----

W_TAG_OVERLAP = 0.40
W_BPM = 0.25
W_RECENCY = -0.15  # applied as score += W_RECENCY * penalty_strength
W_VARIETY = 0.20
W_STAR = 0.10

BPM_TOLERANCE = 30.0  # BPM diff at which proximity score hits 0
RECENT_WINDOW = 10  # how many recent fires count for the penalty
VARIETY_PER_NEW_TAG = 0.05  # bonus per new tag, capped at W_VARIETY
RECENCY_PENALTY_STRENGTH = 0.8  # in [0..1]; spec calls for 0.8


def _norm(path: str | None) -> str:
    """Canonicalise a file path for cross-source matching.

    Lower-cased + ``os.path.normpath`` gives us case-insensitive matching
    on Windows and collapses mixed ``/`` and ``\\`` to one form.
    """
    if not path:
        return ""
    try:
        return os.path.normpath(path).lower()
    except (TypeError, ValueError):
        return path.lower()


class MagicMix:
    """Recommend "what should I play next" based on the LIVE clip + history.

    All data is loaded eagerly in __init__. Call ``reload()`` to pick up
    changes the user made through the rest of the app since startup. The
    class never writes to any of the data sources.
    """

    def __init__(
        self,
        clips_db_path: Path,
        scratch_path: Path,
        path_tags_db: Path,
    ) -> None:
        self.clips_db_path = Path(clips_db_path)
        self.scratch_path = Path(scratch_path)
        self.path_tags_db = Path(path_tags_db)

        # per-normalised-path: aggregated info from clips.json
        # {norm_path: {"raw": original_path, "tags": set, "bpm": float,
        #              "capture_bpm": float, "starred": bool}}
        self._clip_index: dict[str, dict[str, Any]] = {}

        # per-normalised-path: tags from path_tags.db3
        self._path_tags_index: dict[str, set[str]] = {}

        # current scratch list (raw paths, preserves the user's casing)
        self._scratch: list[str] = []

        # Wall-clock of the last reload(), for reload_if_stale()'s TTL gate.
        self._last_reload_ts: float = 0.0

        self.reload()

    # ----------------------------- loading -----------------------------

    def reload(self) -> None:
        """Re-read all three data sources. Force a full reload — callers
        on a hot path (the HTTP /api/magic/suggest handler) should use
        reload_if_stale() instead."""
        self._clip_index = self._load_clips(self.clips_db_path)
        self._path_tags_index = self._load_path_tags(self.path_tags_db)
        self._scratch = self._load_scratch(self.scratch_path)
        self._last_reload_ts = time.time()
        logger.info(
            "MagicMix loaded: %d files in clips.json, %d files with path-tags, "
            "%d items in scratch",
            len(self._clip_index),
            len(self._path_tags_index),
            len(self._scratch),
        )

    def reload_if_stale(self, ttl: float = RELOAD_TTL_SEC) -> bool:
        """Reload only if it's been more than `ttl` seconds since the last
        reload. Returns True if a reload actually happened. Use this on
        the HTTP hot path so a 15s poll cadence doesn't trigger a full
        re-read of every data source every time. (Audit fix M13.)"""
        if time.time() - self._last_reload_ts >= ttl:
            self.reload()
            return True
        return False

    @staticmethod
    def _load_clips(path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            logger.warning("clips.json not found at %s", path)
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to read clips.json at %s", path)
            return {}

        index: dict[str, dict[str, Any]] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            fp = entry.get("filepath")
            if not fp:
                continue
            key = _norm(fp)
            slot = index.setdefault(
                key,
                {
                    "raw": fp,
                    "tags": set(),
                    "bpm": 0.0,
                    "capture_bpm": 0.0,
                    "starred": False,
                    "play_count": 0,
                    "last_played_ts": 0.0,
                },
            )
            for t in entry.get("tags") or []:
                if t:
                    slot["tags"].add(str(t).lower())
            # prefer the largest non-zero bpm / capture_bpm seen across
            # all clips of that file
            for field in ("bpm", "capture_bpm"):
                v = entry.get(field) or 0.0
                if v and v > slot[field]:
                    slot[field] = float(v)
            if entry.get("starred"):
                slot["starred"] = True
            slot["play_count"] = max(
                slot["play_count"], int(entry.get("play_count") or 0)
            )
            slot["last_played_ts"] = max(
                slot["last_played_ts"], float(entry.get("last_played_ts") or 0.0)
            )
        return index

    @staticmethod
    def _load_path_tags(db_path: Path) -> dict[str, set[str]]:
        if not db_path.exists():
            logger.warning("path_tags.db3 not found at %s", db_path)
            return {}
        try:
            # read-only — use URI mode so we never accidentally write
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
        except sqlite3.Error:
            logger.exception("Failed to open path_tags.db3 read-only")
            return {}
        index: dict[str, set[str]] = {}
        try:
            for filepath, tag in conn.execute(
                "SELECT filepath, tag FROM file_tags"
            ):
                if not filepath or not tag:
                    continue
                index.setdefault(_norm(filepath), set()).add(str(tag).lower())
        except sqlite3.Error:
            logger.exception("Failed reading file_tags table")
        finally:
            conn.close()
        return index

    @staticmethod
    def _load_scratch(path: Path) -> list[str]:
        if not path.exists():
            logger.warning("scratch.json not found at %s", path)
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to read scratch.json at %s", path)
            return []
        if not isinstance(data, list):
            return []
        return [str(p) for p in data if p]

    # ------------------------- public lookups --------------------------

    @property
    def scratch(self) -> list[str]:
        """Current scratch basket (as loaded; main.py likely overrides
        with its in-memory copy)."""
        return list(self._scratch)

    def tags_for(self, filepath: str) -> set[str]:
        """All tags known about a file: path-derived ∪ clip-level."""
        key = _norm(filepath)
        clip_tags = self._clip_index.get(key, {}).get("tags", set())
        return set(self._path_tags_index.get(key, set())) | set(clip_tags)

    def bpm_for(self, filepath: str) -> float:
        """Best-effort BPM for a file. Prefers capture_bpm, falls back
        to detected bpm. Returns 0.0 if unknown."""
        slot = self._clip_index.get(_norm(filepath))
        if not slot:
            return 0.0
        return float(slot.get("capture_bpm") or slot.get("bpm") or 0.0)

    def is_starred(self, filepath: str) -> bool:
        return bool(self._clip_index.get(_norm(filepath), {}).get("starred"))

    # --------------------------- scoring -------------------------------

    def _score_candidate(
        self,
        candidate: str,
        live_path: str | None,
        live_bpm: float,
        live_tags: set[str],
        recent_norm: set[str],
    ) -> tuple[float, list[str], dict[str, float]]:
        """Return (score, reasons, breakdown) for a single candidate path."""

        reasons: list[str] = []
        breakdown: dict[str, float] = {}

        cand_tags = self.tags_for(candidate)
        cand_bpm = self.bpm_for(candidate)
        cand_starred = self.is_starred(candidate)
        cand_norm = _norm(candidate)

        # 1) Tag overlap (Jaccard)
        if cand_tags and live_tags:
            inter = cand_tags & live_tags
            union = cand_tags | live_tags
            jaccard = len(inter) / len(union) if union else 0.0
        else:
            inter = set()
            jaccard = 0.0
        tag_score = jaccard
        breakdown["tag_overlap"] = tag_score * W_TAG_OVERLAP
        if inter:
            shown = sorted(inter)[:4]
            reasons.append("shared tags: " + ", ".join(shown))

        # 2) BPM proximity
        if cand_bpm <= 0 or live_bpm <= 0:
            bpm_score = 0.5  # neutral — we just don't know
            if cand_bpm > 0 and live_bpm <= 0:
                reasons.append(f"candidate BPM {cand_bpm:.0f}, live unknown")
        else:
            dist = abs(live_bpm - cand_bpm)
            bpm_score = max(0.0, 1.0 - dist / BPM_TOLERANCE)
            if bpm_score >= 0.8:
                reasons.append(
                    f"BPM match ({cand_bpm:.0f} vs {live_bpm:.0f})"
                )
            elif bpm_score <= 0.2:
                reasons.append(
                    f"BPM mismatch ({cand_bpm:.0f} vs {live_bpm:.0f})"
                )
        breakdown["bpm"] = bpm_score * W_BPM

        # 3) Recency penalty
        recency_pen = 0.0
        if cand_norm in recent_norm:
            recency_pen = RECENCY_PENALTY_STRENGTH
            reasons.append("just played — penalised")
        # we want a negative contribution → multiply
        breakdown["recency"] = W_RECENCY * recency_pen

        # 4) Variety bonus — new tags candidate brings to the table
        new_tags = cand_tags - live_tags if live_tags else cand_tags
        if new_tags:
            raw_bonus = min(W_VARIETY, len(new_tags) * VARIETY_PER_NEW_TAG)
            breakdown["variety"] = raw_bonus
            if raw_bonus >= 0.1:
                some = sorted(new_tags)[:3]
                reasons.append("variety: " + ", ".join(some))
        else:
            breakdown["variety"] = 0.0

        # 5) Star bonus
        if cand_starred:
            breakdown["star"] = W_STAR
            reasons.append("starred")
        else:
            breakdown["star"] = 0.0

        total = sum(breakdown.values())
        # clamp to [0, 1]
        total = max(0.0, min(1.0, total))
        return total, reasons, breakdown

    # --------------------------- public API ----------------------------

    def suggest_next(
        self,
        live_path: str | None,
        live_bpm: float = 0.0,
        recent_fires: list[str] | None = None,
        candidates: list[str] | None = None,
        top_n: int = 5,
    ) -> list[dict]:
        """Rank candidates and return the top ``top_n`` suggestions.

        Args:
            live_path: file path currently on the LIVE deck (may be None
                if nothing is fired yet).
            live_bpm: detected BPM from audio_reactive (0 → unknown).
            recent_fires: ordered list of recently fired file paths;
                only the last RECENT_WINDOW entries are used.
            candidates: paths to rank. Defaults to the current scratch
                basket as loaded from scratch.json. Pass main.py's
                in-memory list to use the live state instead.
            top_n: how many results to return.

        Returns:
            List of dicts (already sorted, highest first):
                {"path", "name", "score", "reasons"}
        """
        if candidates is None:
            candidates = self._scratch

        # de-dupe candidates while preserving order, and drop empties
        seen: set[str] = set()
        cands: list[str] = []
        for c in candidates:
            if not c:
                continue
            key = _norm(c)
            if key in seen:
                continue
            seen.add(key)
            cands.append(c)

        if not cands:
            return []

        live_tags = self.tags_for(live_path) if live_path else set()
        recent_norm = {
            _norm(p) for p in (recent_fires or [])[-RECENT_WINDOW:] if p
        }
        # never suggest the file that's already LIVE
        live_norm = _norm(live_path) if live_path else ""

        scored: list[dict] = []
        for cand in cands:
            if _norm(cand) == live_norm:
                continue
            score, reasons, _ = self._score_candidate(
                cand, live_path, live_bpm, live_tags, recent_norm
            )
            scored.append(
                {
                    "path": cand,
                    "name": os.path.basename(cand),
                    "score": round(score, 4),
                    "reasons": reasons,
                }
            )

        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[: max(0, int(top_n))]

    def explain(
        self,
        candidate_path: str,
        live_path: str,
        live_bpm: float,
    ) -> dict:
        """Verbose scoring breakdown for one candidate.

        Useful for "why was this suggested?" tooltips and for debugging
        the weight tuning.
        """
        live_tags = self.tags_for(live_path) if live_path else set()
        cand_tags = self.tags_for(candidate_path)
        cand_bpm = self.bpm_for(candidate_path)
        cand_starred = self.is_starred(candidate_path)

        score, reasons, breakdown = self._score_candidate(
            candidate_path,
            live_path,
            live_bpm,
            live_tags,
            set(),  # explain() ignores recency by design
        )

        return {
            "candidate": candidate_path,
            "candidate_name": os.path.basename(candidate_path),
            "live": live_path,
            "live_bpm": live_bpm,
            "candidate_bpm": cand_bpm,
            "candidate_tags": sorted(cand_tags),
            "live_tags": sorted(live_tags),
            "shared_tags": sorted(cand_tags & live_tags),
            "new_tags": sorted(cand_tags - live_tags),
            "starred": cand_starred,
            "score": round(score, 4),
            "reasons": reasons,
            "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
            "weights": {
                "tag_overlap": W_TAG_OVERLAP,
                "bpm": W_BPM,
                "recency": W_RECENCY,
                "variety": W_VARIETY,
                "star": W_STAR,
            },
        }


# --------------------------- self-test --------------------------------


def _pick_synthetic_live(mm: "MagicMix") -> str | None:
    """Pick something off-scratch with tags so the demo actually scores."""
    # prefer a path that has path-tags AND is NOT in the current scratch
    scratch_norm = {_norm(p) for p in mm.scratch}
    candidates: list[tuple[int, str]] = []
    for norm_key, tags in mm._path_tags_index.items():  # noqa: SLF001
        if not tags or norm_key in scratch_norm:
            continue
        # need to recover a "raw" filepath; clips.json has it sometimes
        raw = mm._clip_index.get(norm_key, {}).get("raw")  # noqa: SLF001
        if raw:
            candidates.append((len(tags), raw))
    if candidates:
        # most-tagged first → richest tag overlap potential
        candidates.sort(reverse=True)
        return candidates[0][1]
    # fall back to the first scratch item (will still work, just less interesting)
    return mm.scratch[0] if mm.scratch else None


def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    home = Path.home() / ".setpiece"
    clips_db = home / "clips.json"
    scratch = home / "scratch.json"
    path_tags = home / "path_tags.db3"

    print(f"Data dir: {home}")
    print(f"  clips.json     exists={clips_db.exists()}")
    print(f"  scratch.json   exists={scratch.exists()}")
    print(f"  path_tags.db3  exists={path_tags.exists()}")

    try:
        mm = MagicMix(clips_db, scratch, path_tags)
    except Exception as e:  # noqa: BLE001
        print(f"MagicMix init failed: {e}")
        return

    live = _pick_synthetic_live(mm)
    if live is None:
        print("No usable LIVE candidate found — nothing in clips/path-tags. "
              "Run the indexer first; magic_mix has nothing to chew on.")
        return

    live_bpm = mm.bpm_for(live) or 128.0
    print()
    print(f"Synthetic LIVE: {os.path.basename(live)}")
    print(f"  bpm={live_bpm:.1f}")
    print(f"  tags={sorted(mm.tags_for(live))}")
    print()

    # simulate recent fires using the next 2 scratch entries (if any)
    recent = mm.scratch[:2]

    suggestions = mm.suggest_next(
        live_path=live,
        live_bpm=live_bpm,
        recent_fires=recent,
        candidates=None,  # use scratch
        top_n=5,
    )

    if not suggestions:
        print("No suggestions — scratch is empty or all entries match LIVE.")
        return

    print(f"Top {len(suggestions)} suggestions:")
    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. [{s['score']:.3f}] {s['name']}")
        for r in s["reasons"]:
            print(f"        - {r}")

    # also exercise explain() on the top result
    top_path = suggestions[0]["path"]
    print()
    print("explain() for the #1 suggestion:")
    detail = mm.explain(top_path, live, live_bpm)
    print(f"  shared_tags : {detail['shared_tags']}")
    print(f"  new_tags    : {detail['new_tags'][:6]}{'...' if len(detail['new_tags'])>6 else ''}")
    print(f"  breakdown   : {detail['breakdown']}")
    print(f"  final score : {detail['score']}")


if __name__ == "__main__":
    _main()
