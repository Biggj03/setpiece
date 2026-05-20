"""
Channels — saved library-folder presets bound to the S2 FX channel buttons.

Each channel is one of four "tape-deck preset" slots that the user can
configure from the iPad. A channel knows three things:
    name         — short label shown on the iPad chip ("Loops", "Glitch", ...)
    folder       — absolute path the library browser jumps to on switch
    color        — CSS color string used by the iPad chip (the S2 LEDs are
                   amber-only, so this is purely a visual hint on the tablet)
    tag_filter   — optional string passed through to the tags pipeline; the
                   tags/favorites agent owns the actual filter implementation.
                   We just persist it here so the user's setup survives
                   restarts.

Persistence: ~/.setpiece/channels.json. Always exactly 4 entries —
shorter saves get padded with defaults, longer saves get truncated.

This module is pure data + persistence. Wiring channels to the S2 buttons
and library browser lives in main.py (mirrors the decks.py / DeckStore
split — channels.py owns the bytes, main.py owns the behaviour).
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CHANNEL_COUNT = 4

# JSON store, sibling of decks.json / settings.json / clips.json
CHANNELS_FILE = Path.home() / ".setpiece" / "channels.json"

# Default channel colors: red / orange / green / blue. Picked so each
# button is instantly identifiable on the iPad chip row even without
# reading the name. Match the "pinned peripheral" vibe from the user's
# Stream Deck UX notes — color is the page identifier.
DEFAULT_COLORS = ["#ff3355", "#ff9933", "#33cc66", "#3399ff"]

# Default names: just "1".."4". User renames in the iPad editor.
DEFAULT_NAMES = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]


def make_channel(
    idx: int,
    name: str = "",
    folder: str = "",
    color: str = "",
    tag_filter: str = "",
) -> dict:
    """Build a normalized channel dict. All fields stringified, idx clamped."""
    i = max(0, min(CHANNEL_COUNT - 1, int(idx)))
    return {
        "idx": i,
        "name": str(name or DEFAULT_NAMES[i]),
        "folder": str(folder or ""),
        "color": str(color or DEFAULT_COLORS[i]),
        "tag_filter": str(tag_filter or ""),
    }


def default_channels(library_root: str = "") -> list[dict]:
    """Build the initial 4 channels — all pointing at the current library
    root, each with its signature color. First-run default."""
    root = library_root or ""
    return [
        make_channel(i, name=DEFAULT_NAMES[i], folder=root, color=DEFAULT_COLORS[i])
        for i in range(CHANNEL_COUNT)
    ]


def _coerce(entry, idx: int) -> dict:
    """Force a loaded entry into the canonical shape, filling gaps with
    defaults. Tolerates missing keys / bad types — channels.json hand-edits
    shouldn't break startup."""
    if not isinstance(entry, dict):
        return make_channel(idx)
    return make_channel(
        idx=idx,
        name=entry.get("name") or "",
        folder=entry.get("folder") or "",
        color=entry.get("color") or "",
        tag_filter=entry.get("tag_filter") or "",
    )


class ChannelStore:
    """Thread-safe in-memory channel store with JSON persistence.

    Owns: array of exactly 4 channel dicts.
    Does NOT own: AppState (mirroring there is the caller's job) or the
    library browser (jumping is the caller's job).
    """

    def __init__(
        self,
        filepath: Optional[Path] = None,
        default_folder: str = "",
    ):
        self._path = Path(filepath) if filepath else CHANNELS_FILE
        self._lock = threading.RLock()
        self._default_folder = default_folder
        self._channels: list[dict] = default_channels(default_folder)
        self._load()

    # ── State access ───────────────────────────────────────────────────

    def get(self, idx: int) -> Optional[dict]:
        with self._lock:
            if 0 <= idx < CHANNEL_COUNT:
                return dict(self._channels[idx])
            return None

    def all(self) -> list[dict]:
        with self._lock:
            return [dict(c) for c in self._channels]

    # ── Mutations (each persists) ──────────────────────────────────────

    def set(self, idx: int, channel: dict) -> Optional[dict]:
        """Replace channel `idx` outright. Returns the stored dict, or
        None if idx is out of range."""
        with self._lock:
            if not (0 <= idx < CHANNEL_COUNT):
                return None
            self._channels[idx] = _coerce(channel, idx)
            self._save()
            return dict(self._channels[idx])

    def update(self, idx: int, **fields) -> Optional[dict]:
        """Partial update: only the provided fields are touched. Empty
        strings clear a field (UI sends "" to mean "use default")."""
        with self._lock:
            if not (0 <= idx < CHANNEL_COUNT):
                return None
            cur = dict(self._channels[idx])
            for k in ("name", "folder", "color", "tag_filter"):
                if k in fields and fields[k] is not None:
                    cur[k] = str(fields[k])
            # Re-normalize so empty strings reset to defaults
            self._channels[idx] = _coerce(cur, idx)
            self._save()
            return dict(self._channels[idx])

    def replace_all(self, channels: list) -> None:
        """Overwrite all 4 slots. Pads/truncates to exactly 4."""
        with self._lock:
            src = list(channels or [])
            src = (src + [None] * CHANNEL_COUNT)[:CHANNEL_COUNT]
            self._channels = [_coerce(c, i) for i, c in enumerate(src)]
            self._save()

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    src = (raw + [None] * CHANNEL_COUNT)[:CHANNEL_COUNT]
                    self._channels = [_coerce(c, i) for i, c in enumerate(src)]
                    return
        except Exception as e:
            logger.warning(f"Could not load channels.json: {e}")
        # No file (or load failed) — keep the defaults seeded in __init__
        # and persist them so the file exists on first run.
        try:
            self._save()
        except Exception:
            pass

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._channels, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Could not save channels.json: {e}")
