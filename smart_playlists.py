"""
Smart playlists — saved compositions of (library folder + path-tag
filter). Tap a playlist chip to switch the library to a specific
folder AND apply a specific tag filter in one move.

Use case: "high-energy peak set" might be
    folder = /path/to/clips/edm
    tags   = ["edm", "drop"]

Saved as a chip in the library header. Tap → applies both.

Persistence: ~/.setpiece/smart_playlists.json. Each entry is
{id, name, library_root, tags[], created_ts}.
"""

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PLAYLISTS_FILE = Path.home() / ".setpiece" / "smart_playlists.json"


class SmartPlaylistStore:
    def __init__(self, path: Path = PLAYLISTS_FILE):
        self._path = path
        self._lock = threading.Lock()
        self._items: list[dict] = []
        self._load()

    def all(self) -> list[dict]:
        with self._lock:
            return [dict(s) for s in self._items]

    def get(self, pid: str) -> Optional[dict]:
        with self._lock:
            for s in self._items:
                if s.get("id") == pid:
                    return dict(s)
        return None

    def save(self, name: str, library_root: str, tags: list) -> dict:
        clean_name = (name or "").strip() or f"Playlist {len(self._items) + 1}"
        entry = {
            "id": str(uuid.uuid4()),
            "name": clean_name,
            "library_root": str(library_root or "").strip(),
            "tags": [str(t).strip() for t in (tags or []) if t and str(t).strip()],
            "created_ts": time.time(),
        }
        with self._lock:
            self._items.append(entry)
            self._save()
        logger.info(
            f"Smart playlist saved: {clean_name!r} (root={entry['library_root']!r}, tags={entry['tags']})"
        )
        return dict(entry)

    def delete(self, pid: str) -> bool:
        with self._lock:
            before = len(self._items)
            self._items = [s for s in self._items if s.get("id") != pid]
            changed = before != len(self._items)
            if changed:
                self._save()
        return changed

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                self._items = [s for s in raw if isinstance(s, dict)]
        except Exception as e:
            logger.warning("Could not load smart playlists: %s", e)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._items, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Could not save smart playlists: %s", e)
