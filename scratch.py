"""
Scratch list — a lightweight curated basket of library files for the
current set / mood. Distinct from:
- working_set folder: a real filesystem hot folder (drag-and-drop in
  Explorer; auto-queues proxy transcodes).
- clips bank: cross-video saved IN/OUT cue points.
- deck slots: the 4 always-loaded preview decks.

Scratch is just an ordered list of file paths you've picked from the
library to keep within reach during a session. Tap a chip → fire to
LIVE. Clear when the set is over.

Persistence: ~/.setpiece/scratch.json (so it survives restarts;
clear button or stop adding things if you'd rather it not).
"""

import json
import logging
import random
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

SCRATCH_FILE = Path.home() / ".setpiece" / "scratch.json"
SCRATCH_SETS_FILE = Path.home() / ".setpiece" / "scratch_sets.json"
MAX_ENTRIES = 64  # plenty for a session, keeps the iPad chip list manageable


class ScratchStore:
    def __init__(self, path: Path = SCRATCH_FILE):
        self._path = path
        self._lock = threading.Lock()
        self._files: list[str] = []
        self._load()

    # ── reads ───────────────────────────────────────────────────────────

    def all(self) -> list[str]:
        with self._lock:
            return list(self._files)

    def contains(self, path: str) -> bool:
        with self._lock:
            return path in self._files

    # ── mutations ──────────────────────────────────────────────────────

    def add(self, path: str) -> bool:
        """Append path to the scratch list. Idempotent — already-present
        paths are no-op. Returns True if newly added."""
        if not path:
            return False
        with self._lock:
            if path in self._files:
                return False
            self._files.append(path)
            # Cap the list — drop oldest if we exceed.
            if len(self._files) > MAX_ENTRIES:
                self._files = self._files[-MAX_ENTRIES:]
            self._save()
        logger.info(f"Scratch + {Path(path).name}")
        return True

    def remove(self, path: str) -> bool:
        with self._lock:
            if path not in self._files:
                return False
            self._files = [p for p in self._files if p != path]
            self._save()
        logger.info(f"Scratch - {Path(path).name}")
        return True

    def clear(self) -> int:
        with self._lock:
            n = len(self._files)
            self._files = []
            self._save()
        logger.info(f"Scratch cleared ({n} entries)")
        return n

    def shuffle(self) -> int:
        """Randomize the order. Returns the new length."""
        with self._lock:
            random.shuffle(self._files)
            self._save()
            return len(self._files)

    def replace_all(self, paths: list) -> int:
        """Replace the basket with `paths` (used by named-set load)."""
        with self._lock:
            cleaned = [str(p) for p in (paths or []) if p]
            self._files = cleaned[:MAX_ENTRIES]
            self._save()
            return len(self._files)

    # ── persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                self._files = [str(p) for p in raw if isinstance(p, str)]
        except Exception as e:
            logger.warning("Could not load scratch from %s: %s", self._path, e)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._files, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Could not save scratch to %s: %s", self._path, e)


class ScratchSetStore:
    """Named snapshots of the scratch list. Each set = {id, name, files[], created_ts}.
    Useful for keeping multiple curated 'sets' (e.g. 'opener', 'peak',
    'cooldown') and flipping between them mid-session."""

    def __init__(self, path: Path = SCRATCH_SETS_FILE):
        self._path = path
        self._lock = threading.Lock()
        self._sets: list[dict] = []
        self._load()

    def all(self) -> list[dict]:
        with self._lock:
            return [dict(s) for s in self._sets]

    def save_set(self, name: str, files: list) -> dict:
        clean_name = (name or "").strip() or f"Set {len(self._sets) + 1}"
        snap = [str(p) for p in (files or []) if p]
        s = {
            "id": str(uuid.uuid4()),
            "name": clean_name,
            "created_ts": time.time(),
            "files": snap,
        }
        with self._lock:
            self._sets.append(s)
            self._save()
        logger.info(f"Scratch set saved: {clean_name!r} ({len(snap)} files)")
        return dict(s)

    def get(self, set_id: str):
        with self._lock:
            for s in self._sets:
                if s.get("id") == set_id:
                    return dict(s)
        return None

    def delete(self, set_id: str) -> bool:
        with self._lock:
            before = len(self._sets)
            self._sets = [s for s in self._sets if s.get("id") != set_id]
            changed = before != len(self._sets)
            if changed:
                self._save()
        return changed

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                self._sets = [s for s in raw if isinstance(s, dict)]
        except Exception as e:
            logger.warning("Could not load scratch sets: %s", e)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._sets, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Could not save scratch sets: %s", e)
