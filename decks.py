"""
Deck slot management — VJ-style "preview + go-live" launchpad.

Four slots, each holding either a saved clip OR a library file. Each
slot owns an animated filmstrip thumbnail (12 frames sampled across the
clip's duration) for visual preview on the iPad.

Persistence: ~/.setpiece/decks.json. Filmstrips are cached
forever in ~/.setpiece/thumbnails/strip_<hash>.jpg — same
(filepath, in, out) triple → same hash → reused forever.

This module is just data + persistence + filmstrip orchestration.
Wiring decks to the actual player ("fire") lives in main.py.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import thumbnails

logger = logging.getLogger(__name__)

DECK_COUNT = 4

# JSON store, sibling of clips.json / settings.json
DECKS_FILE = Path.home() / ".setpiece" / "decks.json"


def make_deck_entry(
    slot: int,
    source_type: str,           # "clip" | "file"
    source_id: str,             # clip uuid OR resolved filepath for "file"
    filepath: str,
    in_sec: float,
    out_sec: float,
    name: str,
) -> dict:
    """Build a deck dict with a stable strip_hash."""
    in_s = max(0.0, float(in_sec))
    out_s = float(out_sec)
    if out_s <= in_s:
        out_s = in_s + 0.01
    return {
        "slot": int(slot),
        "source_type": source_type,
        "source_id": source_id,
        "filepath": filepath,
        "in_sec": in_s,
        "out_sec": out_s,
        "name": name,
        "strip_hash": thumbnails.filmstrip_hash(filepath, in_s, out_s),
    }


class DeckStore:
    """Thread-safe in-memory deck store with JSON persistence.

    Owns: array of 4 slots. Each slot is None or a deck dict.
    Does NOT own: AppState (mirroring there is the caller's job) or
    the player (firing is the caller's job).
    """

    def __init__(self, filepath: Optional[Path] = None):
        self._path = Path(filepath) if filepath else DECKS_FILE
        self._lock = threading.RLock()
        self._slots: list = [None] * DECK_COUNT
        self._load()

    # ── State access ───────────────────────────────────────────────────

    def get(self, slot: int) -> Optional[dict]:
        with self._lock:
            if 0 <= slot < DECK_COUNT:
                d = self._slots[slot]
                return dict(d) if d else None
            return None

    def all(self) -> list:
        with self._lock:
            return [dict(d) if d else None for d in self._slots]

    # ── Mutations (each persists) ──────────────────────────────────────

    def set(self, slot: int, deck: Optional[dict]) -> None:
        with self._lock:
            if not (0 <= slot < DECK_COUNT):
                return
            self._slots[slot] = dict(deck) if deck else None
            self._save()

    def clear(self, slot: int) -> None:
        self.set(slot, None)

    def replace_all(self, decks: list) -> None:
        with self._lock:
            d = list(decks or [])
            d = (d + [None] * DECK_COUNT)[:DECK_COUNT]
            self._slots = [dict(x) if x else None for x in d]
            self._save()

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    self._slots = (raw + [None] * DECK_COUNT)[:DECK_COUNT]
                    # Sanity: anything that's truthy but not a dict → drop
                    self._slots = [
                        d if isinstance(d, dict) else None for d in self._slots
                    ]
        except Exception as e:
            logger.warning(f"Could not load decks.json: {e}")
            self._slots = [None] * DECK_COUNT

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._slots, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Could not save decks.json: {e}")

    # ── Filmstrip backfill ─────────────────────────────────────────────

    def backfill_filmstrips_async(self) -> threading.Thread:
        """Generate any missing filmstrip JPEGs in a daemon thread.
        Safe to call on every startup."""
        return thumbnails.filmstrip_backfill_async(self.all())

    def regenerate_filmstrip_async(self, slot: int) -> Optional[threading.Thread]:
        """Kick off a one-off filmstrip generation for a single slot.
        Daemon; non-blocking. Returns the thread (or None if slot empty)."""
        d = self.get(slot)
        if not d:
            return None

        def worker():
            try:
                thumbnails.ensure_thumbs_dir()
                ok = thumbnails.generate_filmstrip_for_deck(d)
                if ok:
                    logger.info(f"Filmstrip ready for deck {slot}: {d.get('name')}")
                else:
                    logger.debug(f"Filmstrip failed for deck {slot}: {d.get('name')}")
            except Exception as e:
                logger.warning(f"Filmstrip worker crashed for deck {slot}: {e}")

        t = threading.Thread(
            target=worker, name=f"filmstrip-deck-{slot}", daemon=True,
        )
        t.start()
        return t
