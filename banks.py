"""
Banks — 8 quick-switch slots (A-H) for scratch lists.

Each bank holds a snapshot of the scratch list (full file paths). Tap
a letter chip on the iPad (or eventually a Maschine MK2 Group button)
to swap the live scratch list to that bank's contents.

Differs from `scratch_sets`:
- scratch_sets: unlimited, user-named, displayed as a chip cloud,
  intended for keeping historical curations.
- banks: fixed 8-slot grid (A-H), letter-addressed, intended for
  rapid in-set switching (MK2 Group buttons map 1:1).

Persistence: ~/.setpiece/banks.json. Schema:
{
    "active": "A",
    "slots": {
        "A": {"name": "Opener", "folder": "", "files": ["/path/...", ...]},
        "B": {"name": "Peak",   "folder": "D:/...", "files": [...]},
        ...
    }
}

The optional ``folder`` field locks the bank to a source folder. When
set, ``reroll_banks`` (in main.py) draws random clips from that folder
instead of the theme-tag query — useful when tagging missed a category
(e.g. SoloSoft/Pinup dancing) but the folder itself is well-curated.
Empty string / unset = theme-tag mode (original behavior).
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BANKS_FILE = Path.home() / ".setpiece" / "banks.json"
SLOT_LETTERS = ("A", "B", "C", "D", "E", "F", "G", "H")
MAX_FILES_PER_BANK = 64


class BankStore:
    def __init__(self, path: Path = BANKS_FILE):
        self._path = path
        self._lock = threading.Lock()
        self._active: str = "A"
        # slot letter → {"name": str, "folder": str, "files": [str, ...]}
        # folder = "" means use theme-tag reroll; non-empty = folder-locked.
        self._slots: dict[str, dict] = {
            ltr: {"name": "", "folder": "", "files": []} for ltr in SLOT_LETTERS}
        self._load()

    # ── reads ───────────────────────────────────────────────────────────

    def all_slots(self) -> list[dict]:
        """Return a slim view of all 8 slots for the iPad.
        Each entry: {letter, name, folder, count, active}."""
        with self._lock:
            return [
                {
                    "letter": ltr,
                    "name": (self._slots.get(ltr, {}) or {}).get("name", "") or "",
                    "folder": (self._slots.get(ltr, {}) or {}).get("folder", "") or "",
                    "count": len((self._slots.get(ltr, {}) or {}).get("files", []) or []),
                    "active": ltr == self._active,
                }
                for ltr in SLOT_LETTERS
            ]

    def get(self, letter: str) -> Optional[dict]:
        ltr = (letter or "").upper().strip()
        if ltr not in SLOT_LETTERS:
            return None
        with self._lock:
            slot = self._slots.get(ltr) or {}
            return {
                "letter": ltr,
                "name": slot.get("name", ""),
                "folder": slot.get("folder", "") or "",
                "files": list(slot.get("files") or []),
            }

    def active(self) -> str:
        with self._lock:
            return self._active

    # ── mutations (each persists) ──────────────────────────────────────

    def save_into(self, letter: str, name: str, files: list,
                  set_folder_lock: str | None = None) -> bool:
        """Snapshot `files` into bank `letter` with optional name.

        ``set_folder_lock`` controls the folder-anchor for reroll:
          * ``None`` (default): preserve whatever lock already exists.
            Use for incremental saves where the source should stay.
          * ``""`` (empty string): clear the lock — reroll falls back
            to theme-tag mode. Use for explicit content saves where
            the caller doesn't want anchoring.
          * ``"/library/some-category"`` (a path): set the lock to that
            folder. Use for vertical-press auto-split — reroll then
            re-pulls fresh files FROM that same folder so the user
            stays in the vibe they just picked.

        Bug history (2026-05-18 morning): we used to always preserve.
        Stale locks from old sessions wouldn't update on vertical-press.
        Then we tried always-clear; that broke the natural "stay in
        the folder I just picked" reroll. Now: vertical-press passes
        the new folder explicitly so reroll re-anchors correctly."""
        ltr = (letter or "").upper().strip()
        if ltr not in SLOT_LETTERS:
            return False
        with self._lock:
            existing = self._slots.get(ltr) or {}
            if set_folder_lock is None:
                folder_val = existing.get("folder", "") or ""
            else:
                folder_val = (set_folder_lock or "").strip().replace(
                    "\\", "/")
            self._slots[ltr] = {
                "name": (name or "").strip(),
                "folder": folder_val,
                "files": [str(p) for p in (files or []) if p][:MAX_FILES_PER_BANK],
            }
            self._save()
        msg = f"Bank {ltr} saved: {self._slots[ltr]['name']!r} ({len(self._slots[ltr]['files'])} files)"
        if set_folder_lock is not None and folder_val != (existing.get("folder") or ""):
            if folder_val:
                msg += f"  [folder-lock SET: {folder_val}]"
            else:
                msg += f"  [folder-lock CLEARED: was {existing.get('folder', '')}]"
        logger.info(msg)
        return True

    def save_folder(self, letter: str, folder: str) -> bool:
        """Bind bank `letter` to a source folder. Empty string clears
        the lock (back to theme-tag mode). Doesn't touch the name or
        files — caller can reroll separately if they want fresh picks."""
        ltr = (letter or "").upper().strip()
        if ltr not in SLOT_LETTERS:
            return False
        normalized = (folder or "").strip().replace("\\", "/")
        with self._lock:
            slot = self._slots.get(ltr) or {"name": "", "folder": "", "files": []}
            slot["folder"] = normalized
            self._slots[ltr] = slot
            self._save()
        if normalized:
            logger.info(f"Bank {ltr} folder-locked: {normalized}")
        else:
            logger.info(f"Bank {ltr} folder lock cleared")
        return True

    def clear_slot(self, letter: str) -> bool:
        ltr = (letter or "").upper().strip()
        if ltr not in SLOT_LETTERS:
            return False
        with self._lock:
            self._slots[ltr] = {"name": "", "folder": "", "files": []}
            self._save()
        logger.info(f"Bank {ltr} cleared")
        return True

    def set_active(self, letter: str) -> bool:
        ltr = (letter or "").upper().strip()
        if ltr not in SLOT_LETTERS:
            return False
        with self._lock:
            self._active = ltr
            self._save()
        return True

    # ── persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                active = (raw.get("active") or "A").upper()
                if active in SLOT_LETTERS:
                    self._active = active
                slots = raw.get("slots") or {}
                for ltr in SLOT_LETTERS:
                    s = slots.get(ltr)
                    if isinstance(s, dict):
                        self._slots[ltr] = {
                            "name": str(s.get("name") or ""),
                            "folder": str(s.get("folder") or ""),
                            "files": [str(p) for p in (s.get("files") or []) if p],
                        }
        except Exception as e:
            logger.warning("Could not load banks: %s", e)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"active": self._active, "slots": self._slots}
            self._path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Could not save banks: %s", e)
