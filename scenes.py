"""
Scene presets — snapshot + restore the full 4-deck layout under a name.

A scene is a saved photo of the deck slots ({slot, source_type, source_id,
filepath, in_sec, out_sec, name, ...}) so you can curate a set, save it,
then snap back to it later (or load a different scene to swap the whole
launchpad in one move).

Persistence: ~/.setpiece/scenes.json. Same shape pattern as
decks.json — easy to inspect / hand-edit if needed.
"""

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCENES_FILE = Path.home() / ".setpiece" / "scenes.json"


class SceneStore:
    """Thread-safe JSON-backed store of named deck-snapshot presets."""

    def __init__(self, path: Path = SCENES_FILE):
        self._path = path
        self._lock = threading.Lock()
        self._scenes: list[dict] = []
        self._load()

    # ── reads ───────────────────────────────────────────────────────────

    def all(self) -> list[dict]:
        with self._lock:
            return [dict(s) for s in self._scenes]

    def get(self, scene_id: str) -> Optional[dict]:
        with self._lock:
            for s in self._scenes:
                if s.get("id") == scene_id:
                    return dict(s)
        return None

    # ── mutations (each persists) ──────────────────────────────────────

    def save(self, name: str, deck_snapshot: list) -> dict:
        """Create a new scene from a deck snapshot.

        deck_snapshot: list of length 4 (None for empty slots, dict
        otherwise) — typically the result of decks_store.all().

        Returns the saved scene dict (with assigned id + timestamp)."""
        clean_name = (name or "").strip() or f"Scene {len(self._scenes) + 1}"
        # Deep-copy each slot dict so mutating the live deck later doesn't
        # poison the saved snapshot.
        snapshot = [dict(d) if isinstance(d, dict) else None for d in (deck_snapshot or [])]
        # Pad / truncate to 4 slots so the load path is uniform.
        while len(snapshot) < 4:
            snapshot.append(None)
        snapshot = snapshot[:4]
        scene = {
            "id": str(uuid.uuid4()),
            "name": clean_name,
            "created_ts": time.time(),
            "decks": snapshot,
        }
        with self._lock:
            self._scenes.append(scene)
            self._save()
        logger.info("Scene saved: %r (%d/4 slots filled)",
                    clean_name, sum(1 for d in snapshot if d))
        return dict(scene)

    def delete(self, scene_id: str) -> bool:
        with self._lock:
            before = len(self._scenes)
            self._scenes = [s for s in self._scenes if s.get("id") != scene_id]
            changed = len(self._scenes) != before
            if changed:
                self._save()
        return changed

    def rename(self, scene_id: str, new_name: str) -> bool:
        clean_name = (new_name or "").strip()
        if not clean_name:
            return False
        with self._lock:
            for s in self._scenes:
                if s.get("id") == scene_id:
                    s["name"] = clean_name
                    self._save()
                    return True
        return False

    # ── persistence ─────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                self._scenes = [s for s in raw if isinstance(s, dict)]
        except Exception as e:
            logger.warning("Could not load scenes from %s: %s", self._path, e)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._scenes, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Could not save scenes to %s: %s", self._path, e)
