"""
Single source of truth for app state.
Thread-safe via RLock; all writers acquire before mutation.
iPad reads via HTTP polling; PC components are the only writers.
"""

import threading
import time
from typing import Optional


class AppState:
    """Thread-safe app state. iPad polls; PC writes."""

    def __init__(self):
        self._lock = threading.RLock()

        # Playback state
        self.current_file: Optional[str] = None
        self.position: float = 0.0
        self.duration: float = 0.0
        self.is_playing: bool = False
        self.volume: int = 80

        # Clip markers (pending)
        self.pending_in: Optional[float] = None

        # Audio-reactive state
        self.audio_reactive_enabled: bool = False
        self.audio_sensitivity: float = 1.8
        self.last_beat_time: float = 0.0
        self.detected_bpm: float = 0.0
        # Independent toggle: when False, audio-reactive keeps tracking
        # BPM but stops auto-flipping videos. Lets the user lean on the
        # BPM lock while playing manually.
        self.flip_on_beat: bool = True

        # S2 hardware state
        self.s2_connected: bool = False
        self.s2_armed_pads: list[int] = []  # which pads have clips loaded
        self.s2_last_action: str = ""
        self.s2_last_action_time: float = 0.0

        # Library state
        # library_root: hard boundary — browser cannot escape above this.
        # library_folder: current folder being viewed (always inside root).
        # rel_path: library_folder relative to root, with '/' separators.
        # subfolders: sorted (case-insensitive) names within library_folder.
        # files: list of dicts {"name": str, "hash": str, "has_thumb": bool}
        #        — the hash is sha256(abs_filepath)[:16] used as the
        #        thumbnail cache key. Was list[str] historically; the
        #        renderer + cursor helpers tolerate either shape but
        #        main._publish_library always emits dicts now.
        self.library_root: str = ""
        self.library_folder: str = ""
        self.library_rel_path: str = ""
        self.library_files: list = []
        self.library_subfolders: list[str] = []
        # Browse encoder cursor (which library file is currently "selected").
        # -1 = no selection. Driven by S2 browse encoder; iPad highlights it.
        self.library_selected_idx: int = -1

        # Saved clips (preview slots for iPad)
        self.clips: list[dict] = []
        # Sorted union of every tag across all clips. Snapshot-published so
        # the iPad's <datalist> can suggest existing tags without polling
        # a separate endpoint. Owner: clips_db on every mark_out / set_tags.
        self.all_tags: list[str] = []
        # Bank: a global, cross-video clip bank shown on the iPad. Distinct
        # from .clips (which is per-current-video). Sorted starred-first,
        # then by recency. Capped to ~24 entries for UI comfort.
        self.bank_clips: list[dict] = []
        # Scenes (saved deck-layout snapshots). iPad publishes name + id
        # only — full deck contents are loaded via the scene/load action.
        self.scenes: list[dict] = []
        # Scratch list — curated session basket of library file paths.
        # Published as objects so iPad can show name without doing the
        # basename split itself.
        self.scratch: list[dict] = []
        # Saved scratch-sets (named snapshots of the scratch list).
        self.scratch_sets: list[dict] = []
        # Banks — 8-slot quick-switch grid for scratch lists (A-H).
        self.banks: list[dict] = []
        # Bank LAYER OVERLAY active name (2026-05-18). Default empty
        # string means the "default" body-parts/actions layer. Updated
        # by main.py via set_bank_layer() whenever the user cycles
        # layers (MK2 STEP / iPad chip).
        self.bank_active_layer: str = "default"
        self.bank_layer_order: list = ["default"]
        # Smart playlists — saved (library_root + path-tag filter) combos.
        self.smart_playlists: list[dict] = []
        # Analytics blob (top clips, BPM histogram, today counters, disk usage).
        self.analytics: dict = {}
        # Proxy-transcode runtime status (queue depth + in-flight name).
        self.proxy_status: dict = {"paused": False, "queued": 0, "in_flight": ""}
        # Path-derived auto-tag chips. Each entry: {"tag": str, "count": int}.
        # Populated by main.py from the background path-tag index.
        self.path_tags_top: list[dict] = []
        # Active path-tag filter on the library. Empty list = no filter.
        self.path_tag_filter: list[str] = []
        # Sticky active tag: when non-empty, every Mark OUT (and SHIFT+Loop
        # Out save-loop) auto-applies this tag. Lets the user pre-pick a
        # tag once and machine-gun saves without typing per clip.
        self.active_tag: str = ""

        # Deck slots (live VJ launchpad). Fixed length = 4. Each slot is
        # None or a dict: {slot, source_type, source_id, filepath,
        # in_sec, out_sec, name, strip_hash}. PC writer only.
        self.decks: list = [None, None, None, None]

        # Crossfade ("preview deck B") state.
        # preview_file: filepath currently loaded into the secondary mpv
        # track for alpha-blend, or None if no preview is staged.
        # crossfader_position: 0.0 = pure live, 1.0 = pure preview, 0.5 = blend.
        # blend_active: whether the lavfi-complex graph is currently up
        # (False if mpv refused load_preview, lavfi-complex unsupported, etc).
        # preview_deck_idx: which deck slot (0..3) is mirrored as the
        # preview source. By design we pull from deck 0 — keeps the
        # mental model simple — but this field tells the UI which deck
        # the user is currently staging for the crossfade.
        self.preview_file: Optional[str] = None
        self.crossfader_position: float = 0.0
        self.blend_active: bool = False
        self.preview_deck_idx: int = 0

        # Status / errors
        self.last_message: str = ""
        self.last_error: str = ""

        # Jog sensitivity mode (gentle/medium/coarse)
        self.jog_mode: str = "medium"

        # Channels — saved library-folder presets bound to S2 FX channel
        # buttons (fx1_ch1, fx2_ch1, fx1_ch2, fx2_ch2). Always exactly 4.
        # Each entry: {idx, name, folder, color, tag_filter}.
        # active_channel_idx: which one is currently "lit" (-1 = none).
        # ChannelStore (channels.py) is the source of truth on disk;
        # AppState only mirrors for the iPad snapshot.
        self.channels: list[dict] = []
        self.active_channel_idx: int = -1

    def set_jog_mode(self, mode: str):
        with self._lock:
            self.jog_mode = mode

    def snapshot(self) -> dict:
        """Get a JSON-serializable snapshot of current state."""
        with self._lock:
            return {
                "playback": {
                    "file": self.current_file,
                    "position": self.position,
                    "duration": self.duration,
                    "is_playing": self.is_playing,
                    "volume": self.volume,
                },
                "markers": {
                    "pending_in": self.pending_in,
                },
                "audio_reactive": {
                    "enabled": self.audio_reactive_enabled,
                    "flip_on_beat": self.flip_on_beat,
                    "sensitivity": self.audio_sensitivity,
                    "last_beat_age_ms": int((time.time() - self.last_beat_time) * 1000)
                    if self.last_beat_time else None,
                    "bpm": self.detected_bpm,
                },
                "s2": {
                    "connected": self.s2_connected,
                    "armed_pads": self.s2_armed_pads,
                    "last_action": self.s2_last_action,
                    "last_action_age_ms": int((time.time() - self.s2_last_action_time) * 1000)
                    if self.s2_last_action_time else None,
                    "jog_mode": self.jog_mode,
                },
                "library": {
                    "root": self.library_root,
                    "folder": self.library_folder,
                    "rel_path": self.library_rel_path,
                    "files": self.library_files,
                    "subfolders": self.library_subfolders,
                    "at_root": self.library_rel_path in ("", "."),
                    "selected_idx": self.library_selected_idx,
                },
                "clips": self.clips,
                "all_tags": list(self.all_tags),
                "bank_clips": list(self.bank_clips),
                "active_tag": self.active_tag,
                "scenes": list(self.scenes),
                "scratch": list(self.scratch),
                "scratch_sets": list(self.scratch_sets),
                "banks": list(self.banks),
                "bank_layer": {
                    "active": self.bank_active_layer,
                    "order": list(self.bank_layer_order),
                },
                "smart_playlists": list(self.smart_playlists),
                "analytics": dict(self.analytics or {}),
                "proxy_status": dict(self.proxy_status or {}),
                "path_tags": {
                    "top": list(self.path_tags_top),
                    "filter": list(self.path_tag_filter),
                },
                "decks": list(self.decks),
                "crossfade": {
                    "preview_file": self.preview_file,
                    "preview_deck_idx": self.preview_deck_idx,
                    "position": self.crossfader_position,
                    "blend_active": self.blend_active,
                    "live_file": self.current_file,
                },
                "channels": {
                    "list": [dict(c) for c in self.channels],
                    "active_idx": self.active_channel_idx,
                },
                "status": {
                    "message": self.last_message,
                    "error": self.last_error,
                    "timestamp": time.time(),
                },
            }

    def update_playback(self, position: float = None, duration: float = None,
                        is_playing: bool = None, file: str = None):
        """Update playback state (called from player tick)."""
        with self._lock:
            if position is not None:
                self.position = position
            if duration is not None:
                self.duration = duration
            if is_playing is not None:
                self.is_playing = is_playing
            if file is not None:
                self.current_file = file

    def set_pending_in(self, position: Optional[float]):
        """Set/clear pending IN marker."""
        with self._lock:
            self.pending_in = position

    def set_audio_reactive(self, enabled: bool, sensitivity: float = None):
        """Update audio-reactive state."""
        with self._lock:
            self.audio_reactive_enabled = enabled
            if sensitivity is not None:
                self.audio_sensitivity = sensitivity

    def set_flip_on_beat(self, value: bool):
        """Set the auto-flip-on-beat toggle. (Audit fix H1 — gives main.py
        a proper accessor instead of reaching into AppState._lock and
        writing the attribute directly.)"""
        with self._lock:
            self.flip_on_beat = bool(value)

    def toggle_flip_on_beat(self) -> bool:
        """Flip the toggle atomically and return the NEW value. (Audit fix
        H1.) Doing the read+write under one lock acquisition keeps it
        race-free vs. two separate get/set calls."""
        with self._lock:
            self.flip_on_beat = not self.flip_on_beat
            return self.flip_on_beat

    def get_flip_on_beat(self) -> bool:
        """Read the auto-flip-on-beat toggle under the lock."""
        with self._lock:
            return self.flip_on_beat

    def record_beat(self):
        """Record a beat detection event."""
        with self._lock:
            self.last_beat_time = time.time()

    def set_detected_bpm(self, bpm: float):
        """Update detected BPM (called from audio-reactive thread)."""
        with self._lock:
            self.detected_bpm = float(bpm) if bpm and bpm > 0 else 0.0

    def set_s2_connected(self, connected: bool):
        """Update S2 connection state."""
        with self._lock:
            self.s2_connected = connected

    def record_s2_action(self, action: str):
        """Record S2 button press for iPad display."""
        with self._lock:
            self.s2_last_action = action
            self.s2_last_action_time = time.time()

    def set_armed_pads(self, pad_indices: list[int]):
        """Update which pads have clips loaded."""
        with self._lock:
            self.s2_armed_pads = list(pad_indices)

    def set_library(self, folder: str, files: list,
                    subfolders: list[str] = None, rel_path: str = None,
                    root: str = None):
        """Update library folder + file list (and optional navigation fields).

        `files` is now expected to be list[dict] with shape
        {"name": str, "hash": str, "has_thumb": bool}. Legacy callers
        passing list[str] are auto-coerced (hash/has_thumb absent) so
        nothing crashes during a partial migration.
        """
        with self._lock:
            self.library_folder = folder
            coerced = []
            for f in files or []:
                if isinstance(f, dict):
                    coerced.append(dict(f))
                else:
                    # Legacy bare string — keep it queryable but flag it
                    # as un-thumbnailable.
                    coerced.append({"name": str(f), "hash": "", "has_thumb": False})
            self.library_files = coerced
            if subfolders is not None:
                self.library_subfolders = list(subfolders)
            if rel_path is not None:
                self.library_rel_path = rel_path
            if root is not None:
                self.library_root = root

    def set_library_root(self, root: str):
        """Set the hard navigation boundary."""
        with self._lock:
            self.library_root = root

    def set_library_selected_idx(self, idx: int):
        """Move the browse-encoder cursor."""
        with self._lock:
            self.library_selected_idx = int(idx)

    def get_library_snapshot(self) -> dict:
        """Return a consistent snapshot of the library browse state under
        one lock acquisition. (Audit fix H8 — non-Qt threads like the S2
        action worker were reading library_files / library_selected_idx /
        library_folder as separate unlocked attribute accesses, so a
        _publish_library swap between reads could mix generations.)

        Returns {files, selected_idx, folder, root, rel_path}. `files` is
        a shallow copy so the caller can index it safely even if the
        underlying list is replaced afterwards."""
        with self._lock:
            return {
                "files": list(self.library_files or []),
                "selected_idx": self.library_selected_idx,
                "folder": self.library_folder,
                "root": self.library_root,
                "rel_path": self.library_rel_path,
            }

    def move_library_cursor(self, delta: int) -> dict:
        """Atomically move the browse cursor by `delta` (wrapping) and
        return the new snapshot {files, selected_idx, folder, ...}.
        (Audit fix H8.) Doing the len()/index/clamp/store under one lock
        means the cursor can never land out of range because the folder
        changed mid-move."""
        with self._lock:
            files = self.library_files or []
            n = len(files)
            if n == 0:
                self.library_selected_idx = -1
                new_idx = -1
            else:
                cur = self.library_selected_idx
                if cur < 0:
                    cur = 0 if delta >= 0 else n - 1
                    new_idx = cur
                else:
                    new_idx = (cur + delta) % n
                self.library_selected_idx = new_idx
            return {
                "files": list(files),
                "selected_idx": new_idx,
                "folder": self.library_folder,
                "root": self.library_root,
                "rel_path": self.library_rel_path,
            }

    def set_clips(self, clips: list[dict]):
        """Update saved clips list."""
        with self._lock:
            self.clips = list(clips)

    def set_bank_clips(self, clips: list[dict]):
        with self._lock:
            self.bank_clips = list(clips or [])

    def set_active_tag(self, tag: str):
        with self._lock:
            self.active_tag = (tag or "").strip()

    def set_proxy_status(self, status: dict):
        with self._lock:
            self.proxy_status = dict(status or {})

    def set_analytics(self, blob: dict):
        with self._lock:
            self.analytics = dict(blob or {})

    def set_smart_playlists(self, items: list):
        with self._lock:
            self.smart_playlists = [
                {
                    "id": s.get("id"),
                    "name": s.get("name"),
                    "tags": list(s.get("tags") or []),
                    "library_root": s.get("library_root", ""),
                }
                for s in (items or []) if isinstance(s, dict)
            ]

    def set_banks(self, slots: list):
        with self._lock:
            self.banks = list(slots or [])

    def set_bank_layer(self, active: str, order: list = None):
        """Publish the currently active layer name + cycle order to
        the iPad. Called from main.cycle_bank_layer() and on startup
        so the iPad chip shows the right name from page load."""
        with self._lock:
            self.bank_active_layer = (active or "default").strip().lower()
            if order is not None:
                self.bank_layer_order = list(order)

    def set_scratch_sets(self, sets: list):
        with self._lock:
            self.scratch_sets = [
                {
                    "id": s.get("id"),
                    "name": s.get("name"),
                    "count": len(s.get("files") or []),
                }
                for s in (sets or []) if isinstance(s, dict)
            ]

    def set_path_tags_top(self, items: list):
        with self._lock:
            self.path_tags_top = list(items or [])

    def set_path_tag_filter(self, tags: list[str]):
        with self._lock:
            self.path_tag_filter = [t for t in (tags or []) if isinstance(t, str) and t.strip()]

    def set_scratch(self, paths: list[str]):
        # Convert raw paths to chip objects with name + path. Most-recent-
        # first so the latest add appears on the left.
        import os
        with self._lock:
            self.scratch = [
                {"path": p, "name": os.path.basename(p) or p}
                for p in reversed(paths or [])
            ]

    def set_scenes(self, scenes: list[dict]):
        # Slim the published list — iPad only needs id, name, slot count.
        with self._lock:
            self.scenes = [
                {
                    "id": s.get("id"),
                    "name": s.get("name"),
                    "created_ts": s.get("created_ts"),
                    "slot_count": sum(1 for d in (s.get("decks") or []) if d),
                }
                for s in (scenes or [])
                if isinstance(s, dict)
            ]

    def set_all_tags(self, tags: list[str]):
        """Update the global tag union (sorted/deduped by caller; we just
        store a defensive copy)."""
        with self._lock:
            self.all_tags = list(tags or [])

    def set_deck(self, slot: int, deck_data: Optional[dict]):
        """Set or clear deck slot 0..3. Pass None to clear."""
        with self._lock:
            if 0 <= slot < len(self.decks):
                self.decks[slot] = dict(deck_data) if deck_data else None

    def clear_deck(self, slot: int):
        """Empty a deck slot."""
        with self._lock:
            if 0 <= slot < len(self.decks):
                self.decks[slot] = None

    def set_decks(self, decks: list):
        """Replace the full deck array (used on startup)."""
        with self._lock:
            # Pad / truncate to exactly 4 slots
            d = list(decks or [])
            d = (d + [None, None, None, None])[:4]
            self.decks = d

    def set_message(self, msg: str, error: bool = False):
        """Set status message for iPad display."""
        with self._lock:
            if error:
                self.last_error = msg
            else:
                self.last_message = msg

    def set_volume(self, volume: int):
        """Update volume display."""
        with self._lock:
            self.volume = max(0, min(100, volume))

    def set_channels(self, channels: list):
        """Replace the channel list. Coerces to exactly 4 entries —
        anything shorter gets padded with empty dicts so iPad rendering
        never blows up on len() mismatches."""
        with self._lock:
            src = list(channels or [])
            src = (src + [{}] * 4)[:4]
            self.channels = [dict(c) if isinstance(c, dict) else {} for c in src]

    def set_active_channel(self, idx: int):
        """Set the currently-active channel index. Pass -1 (or any
        out-of-range value) to clear. Does not mutate self.channels."""
        with self._lock:
            i = int(idx)
            if 0 <= i < 4:
                self.active_channel_idx = i
            else:
                self.active_channel_idx = -1

    def set_crossfade(self, position: float = None, preview_file: object = ...,
                      blend_active: bool = None, preview_deck_idx: int = None):
        """Update crossfader/preview state. preview_file uses sentinel `...`
        so passing None explicitly clears it (vs not passing it at all)."""
        with self._lock:
            if position is not None:
                self.crossfader_position = max(0.0, min(1.0, float(position)))
            if preview_file is not ...:
                self.preview_file = preview_file
            if blend_active is not None:
                self.blend_active = bool(blend_active)
            if preview_deck_idx is not None and 0 <= preview_deck_idx < 4:
                self.preview_deck_idx = int(preview_deck_idx)
