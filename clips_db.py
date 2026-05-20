"""
Clip database: JSON-backed store for IN/OUT marks on video files.

Schema fields (per clip):
    id              uuid4 string
    name            human-readable
    filepath        absolute video path
    in_seconds      mark IN (float)
    out_seconds     mark OUT (float)
    duration        out - in (float, denormalized)
    tags            list[str]   sorted, lowercase, alpha-only, deduped
    starred         bool        user favourite flag
    play_count      int         number of times bumped via bump_play_count
    last_played_ts  float       unix-seconds, 0.0 = never played
    bpm             float       detected tempo (60..200) or 0.0 = unknown

Backwards compatibility: any clip loaded from a pre-tags clips.json gets
the four new fields populated with safe defaults on _load(). The on-disk
file is rewritten in the new shape on first save.
"""

import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import thumbnails

# clip_bpm is OPTIONAL — analysis depends on numpy + ffmpeg, neither of
# which is strictly required for the rest of the app to boot. Missing
# import → backfill becomes a no-op and bpm stays at 0.0.
try:
    import clip_bpm
    _BPM_IMPORT_ERROR: Optional[str] = None
except Exception as _bpm_err:  # pragma: no cover - environment-dependent
    clip_bpm = None  # type: ignore[assignment]
    _BPM_IMPORT_ERROR = str(_bpm_err)

logger = logging.getLogger(__name__)


# Tags must be lowercase, alpha (a-z) only, no spaces or punctuation.
# Keeps the URL/JSON wire format trivial and matches the user-facing
# convention of one-word genre/role tags ("drop", "build", "outro").
_TAG_RE = re.compile(r"^[a-z]+$")
_MAX_TAG_LEN = 24
_MAX_TAGS_PER_CLIP = 16


def _normalize_tag(raw: str) -> str:
    """Coerce a raw tag input to canonical form. Returns "" if it can't
    be made into a valid tag (e.g. empty after stripping non-alpha)."""
    if not isinstance(raw, str):
        return ""
    # Strip everything that isn't a letter, lowercase the result.
    cleaned = re.sub(r"[^A-Za-z]", "", raw).lower()
    if not cleaned:
        return ""
    cleaned = cleaned[:_MAX_TAG_LEN]
    return cleaned if _TAG_RE.match(cleaned) else ""


def _normalize_tags(raw_tags) -> list[str]:
    """Coerce a list of raw tag strings to canonical form: alpha-only,
    lowercase, deduped, sorted, capped at _MAX_TAGS_PER_CLIP."""
    if not raw_tags:
        return []
    out: set[str] = set()
    for t in raw_tags:
        norm = _normalize_tag(t)
        if norm:
            out.add(norm)
    return sorted(out)[:_MAX_TAGS_PER_CLIP]


class ClipDatabase:
    """Simple JSON-backed clip store. Each clip is an IN/OUT pair on a file."""

    def __init__(self, filepath: str = ""):
        self._path = filepath or str(Path.home() / ".setpiece" / "clips.json")
        self._clips: list[dict] = []
        # RLock guards _clips / _pending_in / _save. mark_out runs on the
        # Qt thread, the clip_bpm backfill thread calls set_bpm → _save,
        # and get_all_clips can be hit from the HTTP thread — without this
        # a backfill write during a json.dumps could tear the file.
        # (Audit fix M11.) RLock (not Lock) because some methods call
        # others that also lock.
        self._lock = threading.RLock()
        # Pending IN-point per file (not persisted until OUT is marked)
        self._pending_in: dict[str, float] = {}
        # Optional listener: notified whenever BPM lands for a clip so
        # main.py can re-publish the clip list to the iPad without
        # waiting for the next mark_in/mark_out cycle.
        self._on_bpm_updated: Optional[callable] = None
        self._load()
        # Backfill any missing thumbnails for previously-saved clips.
        # Runs in a daemon thread so it never blocks startup.
        try:
            thumbnails.ensure_thumbs_dir()
            thumbnails.backfill_async(self._clips)
        except Exception as e:
            logger.debug(f"Thumbnail backfill not started: {e}")
        # Backfill missing BPM in a separate daemon thread. clip_bpm's
        # internal semaphore serialises the ffmpeg work, so even with a
        # 90-clip library this is one analyser at a time.
        if clip_bpm is not None:
            try:
                clip_bpm.backfill_async(
                    self._clips,
                    on_bpm=self._apply_bpm_result,
                )
            except Exception as e:
                logger.debug(f"BPM backfill not started: {e}")
        elif _BPM_IMPORT_ERROR:
            logger.debug(f"BPM analysis disabled: {_BPM_IMPORT_ERROR}")

    def set_on_bpm_updated(self, callback) -> None:
        """Wire an external listener (called as ``cb(clip_id, bpm)``) so
        the app can refresh the iPad clip list when a background BPM
        analysis completes. Safe to call any time; passing None unhooks."""
        self._on_bpm_updated = callback

    def mark_in(self, filepath: str, position: float) -> dict:
        """Stash a pending IN-point for this file."""
        if not filepath:
            return {"ok": False, "message": "no filepath"}
        key = self._key(filepath)
        with self._lock:
            self._pending_in[key] = float(position)
        return {
            "ok": True,
            "message": f"IN @ {self._fmt(position)}",
            "in_seconds": float(position),
        }

    def mark_out(self, filepath: str, position: float, name: str = "",
                 capture_speed: float = 1.0, capture_bpm: float = 0.0) -> dict:
        """Mark OUT and save clip.

        capture_speed: live playback speed at save time (1.0 = normal).
        capture_bpm: detected BPM at save time (0 if unknown).
        Both are restored when the clip is fired so the loop sounds /
        looks like it did when you captured it."""
        if not filepath:
            return {"ok": False, "message": "no filepath"}
        key = self._key(filepath)
        with self._lock:
            in_sec = self._pending_in.get(key)
            if in_sec is None:
                return {"ok": False, "message": "no pending IN for this file"}
            if in_sec >= position:
                return {"ok": False, "message": f"IN ({in_sec:.1f}s) >= OUT ({position:.1f}s)"}

            clip = {
                "id": str(uuid.uuid4()),
                "name": name or f"clip_{len(self._clips)}",
                "filepath": filepath,
                "in_seconds": in_sec,
                "out_seconds": position,
                "duration": position - in_sec,
                # New metadata defaults — also applied retroactively to old
                # clips by _migrate_clip() so the schema is uniform on disk.
                "tags": [],
                "starred": False,
                "play_count": 0,
                "last_played_ts": 0.0,
                # 0.0 = unknown / not yet analysed. clip_bpm.analyze_async fills
                # this in on the next background sweep (kicked off below).
                "bpm": 0.0,
                # Live capture context: playback speed and detected BPM at the
                # moment of save. Restored when the clip is fired so the loop
                # plays back at the same tempo as when you grabbed it.
                "capture_speed": float(capture_speed) if capture_speed and capture_speed > 0 else 1.0,
                "capture_bpm": float(capture_bpm) if capture_bpm and capture_bpm > 0 else 0.0,
            }
            self._clips.append(clip)
            del self._pending_in[key]
            self._save()
        # Generate thumbnail at the IN-point. Defensive — failure is fine.
        try:
            thumbnails.generate_for_clip(clip)
        except Exception as e:
            logger.debug(f"Thumbnail generation failed for {clip.get('id')}: {e}")
        # Kick off BPM analysis for this single clip in a daemon thread.
        # The semaphore inside clip_bpm caps concurrency at 1 across the
        # whole process, so this never floods ffmpeg even when a flurry
        # of clips are saved back-to-back.
        if clip_bpm is not None:
            try:
                clip_bpm.analyze_async(
                    clip,
                    on_done=lambda bpm, cid=clip["id"]: self._apply_bpm_result(cid, bpm),
                )
            except Exception as e:
                logger.debug(f"BPM analyse_async failed for {clip['id']}: {e}")
        return {
            "ok": True,
            "message": f"Clip saved: {clip['name']}",
            "clip": clip,
        }

    def get_clip(self, idx: int) -> Optional[dict]:
        """Get clip by index."""
        with self._lock:
            if 0 <= idx < len(self._clips):
                return self._clips[idx]
            return None

    def get_all_clips(self) -> list[dict]:
        """Get all clips."""
        with self._lock:
            return self._clips.copy()

    def get_clips_for_file(self, filepath: str) -> list[dict]:
        """Get only clips marked on the given file. 'Deck per video' model:
        the pads/iPad show only what's relevant to what's on screen."""
        if not filepath:
            return []
        key = self._key(filepath)
        with self._lock:
            return [c for c in self._clips if self._key(c.get("filepath", "")) == key]

    def clips_by_tag(self, tag: str) -> list[dict]:
        """Return every clip that carries the given tag. Powers the
        iPad "Openers" launcher (tag='opener' / 'transition-rich')
        and any future tag-filtered clip pickers. Sorted starred-first,
        then by last_played desc."""
        tag = (tag or "").strip()
        if not tag:
            return []
        with self._lock:
            matched = [
                dict(c) for c in self._clips
                if tag in (c.get("tags") or [])
            ]
        # Stable sort: starred first, then most-recently-played first.
        matched.sort(
            key=lambda c: (
                0 if c.get("starred") else 1,
                -float(c.get("last_played_ts") or 0.0),
            )
        )
        return matched

    def delete_clip_by_id(self, clip_id: str) -> dict:
        """Delete clip by its uuid. Used by iPad X-button (avoids index shift)."""
        with self._lock:
            for i, c in enumerate(self._clips):
                if c.get("id") == clip_id:
                    return self.delete_clip(i)
        return {"ok": False, "message": f"Clip {clip_id} not found"}

    def clear_clips_for_file(self, filepath: str) -> dict:
        """Drop every clip marked on the given file."""
        if not filepath:
            return {"ok": False, "message": "no filepath"}
        key = self._key(filepath)
        with self._lock:
            before = len(self._clips)
            kept = [c for c in self._clips if self._key(c.get("filepath", "")) != key]
            removed = [c for c in self._clips if self._key(c.get("filepath", "")) == key]
            self._clips = kept
            self._save()
        for c in removed:
            try:
                thumbnails.delete_thumbnail(c.get("id", ""))
            except Exception:
                pass
        return {"ok": True, "message": f"Cleared {before - len(kept)} clip(s)"}

    def delete_clip(self, idx: int) -> dict:
        """Delete clip by index."""
        with self._lock:
            if 0 <= idx < len(self._clips):
                removed = self._clips.pop(idx)
                self._save()
            else:
                return {"ok": False, "message": f"Clip {idx} not found"}
        # Drop the thumbnail too. Silent on failure.
        try:
            thumbnails.delete_thumbnail(removed.get("id", ""))
        except Exception as e:
            logger.debug(f"Thumbnail delete failed for {removed.get('id')}: {e}")
        return {"ok": True, "message": f"Deleted: {removed['name']}", "clip": removed}

    # ── Tag / star / play-count metadata ──────────────────────────────────

    def _find_by_id(self, clip_id: str) -> Optional[dict]:
        with self._lock:
            for c in self._clips:
                if c.get("id") == clip_id:
                    return c
            return None

    def set_starred(self, clip_id: str, value: bool) -> dict:
        """Toggle/set the star flag on a clip. Persists immediately."""
        with self._lock:
            clip = self._find_by_id(clip_id)
            if not clip:
                return {"ok": False, "message": f"Clip {clip_id} not found"}
            clip["starred"] = bool(value)
            self._save()
            return {"ok": True, "clip_id": clip_id, "starred": clip["starred"]}

    def set_tags(self, clip_id: str, tags_list) -> dict:
        """Replace the entire tag list. Tags are normalized: lowercase,
        alpha-only, deduped, sorted, capped at _MAX_TAGS_PER_CLIP."""
        with self._lock:
            clip = self._find_by_id(clip_id)
            if not clip:
                return {"ok": False, "message": f"Clip {clip_id} not found"}
            clip["tags"] = _normalize_tags(tags_list or [])
            self._save()
            return {"ok": True, "clip_id": clip_id, "tags": list(clip["tags"])}

    def add_tag(self, clip_id: str, tag: str) -> dict:
        """Add one tag (idempotent)."""
        with self._lock:
            clip = self._find_by_id(clip_id)
            if not clip:
                return {"ok": False, "message": f"Clip {clip_id} not found"}
            norm = _normalize_tag(tag)
            if not norm:
                return {"ok": False, "message": "invalid tag (a-z only)"}
            existing = set(clip.get("tags") or [])
            existing.add(norm)
            clip["tags"] = sorted(existing)[:_MAX_TAGS_PER_CLIP]
            self._save()
            return {"ok": True, "clip_id": clip_id, "tags": list(clip["tags"])}

    def remove_tag(self, clip_id: str, tag: str) -> dict:
        """Remove one tag (no-op if missing)."""
        with self._lock:
            clip = self._find_by_id(clip_id)
            if not clip:
                return {"ok": False, "message": f"Clip {clip_id} not found"}
            norm = _normalize_tag(tag)
            existing = [t for t in (clip.get("tags") or []) if t != norm]
            clip["tags"] = existing
            self._save()
            return {"ok": True, "clip_id": clip_id, "tags": list(clip["tags"])}

    def bump_play_count(self, clip_id: str) -> dict:
        """Increment play_count + stamp last_played_ts to now. Called from
        the play_clip path so both sort dimensions update on every fire."""
        with self._lock:
            clip = self._find_by_id(clip_id)
            if not clip:
                return {"ok": False, "message": f"Clip {clip_id} not found"}
            clip["play_count"] = int(clip.get("play_count", 0) or 0) + 1
            clip["last_played_ts"] = time.time()
            self._save()
            return {
                "ok": True,
                "clip_id": clip_id,
                "play_count": clip["play_count"],
                "last_played_ts": clip["last_played_ts"],
            }

    # ── BPM ───────────────────────────────────────────────────────────────

    def set_bpm(self, clip_id: str, bpm: float) -> dict:
        """Stamp a detected BPM onto a clip and persist. ``bpm`` outside
        the [60, 200] range is clamped to 0.0 (= unknown) so we don't
        propagate junk values into the iPad sort/filter."""
        with self._lock:
            clip = self._find_by_id(clip_id)
            if not clip:
                return {"ok": False, "message": f"Clip {clip_id} not found"}
            try:
                value = float(bpm)
            except (TypeError, ValueError):
                value = 0.0
            if value < 60.0 or value > 200.0:
                value = 0.0
            clip["bpm"] = value
            self._save()
            return {"ok": True, "clip_id": clip_id, "bpm": value}

    def _apply_bpm_result(self, clip_id: str, bpm) -> None:
        """Background-thread callback target. Stamps the result and
        notifies any external listener so the iPad can refresh."""
        if not clip_id:
            return
        if bpm is None:
            # Analysis ran but produced no usable BPM — leave the field
            # at 0 so a later reanalyze is still allowed.
            return
        try:
            result = self.set_bpm(clip_id, float(bpm))
        except Exception as e:
            logger.debug(f"set_bpm raised for {clip_id}: {e}")
            return
        if result.get("ok") and self._on_bpm_updated:
            try:
                self._on_bpm_updated(clip_id, result["bpm"])
            except Exception as e:
                logger.debug(f"on_bpm_updated raised for {clip_id}: {e}")

    def reanalyze_bpm(self, clip_id: str) -> dict:
        """Drop any cached BPM and re-run analysis in a daemon thread.
        Used by the iPad's "BPM looks wrong" reanalyze button."""
        with self._lock:
            clip = self._find_by_id(clip_id)
            if not clip:
                return {"ok": False, "message": f"Clip {clip_id} not found"}
            if clip_bpm is None:
                return {"ok": False, "message": "BPM analysis unavailable"}
            # Reset to 0 immediately so the iPad shows "analyzing..." on next poll.
            clip["bpm"] = 0.0
            self._save()
        try:
            clip_bpm.clear_cache_for_clip(clip)
            clip_bpm.analyze_async(
                clip,
                on_done=lambda bpm, cid=clip_id: self._apply_bpm_result(cid, bpm),
            )
        except Exception as e:
            return {"ok": False, "message": f"reanalyze failed: {e}"}
        return {"ok": True, "clip_id": clip_id, "analyzing": True}

    def get_all_tags(self) -> list[str]:
        """Sorted, deduplicated union of every tag across all clips.
        Used by the iPad picker datalist so existing tags suggest themselves."""
        seen: set[str] = set()
        with self._lock:
            for c in self._clips:
                for t in c.get("tags") or []:
                    if isinstance(t, str) and t:
                        seen.add(t)
        return sorted(seen)

    # ── Internal ──────────────────────────────────────────────────────────

    def _key(self, filepath: str) -> str:
        """Normalized file path key."""
        return str(Path(filepath).resolve())

    def _fmt(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS.mmm"""
        h = int(seconds) // 3600
        m = (int(seconds) % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    @staticmethod
    def _migrate_clip(clip: dict) -> bool:
        """In-place add the new metadata fields if missing. Returns True
        if anything changed (so _load can decide whether to re-save)."""
        changed = False
        if "tags" not in clip or not isinstance(clip.get("tags"), list):
            clip["tags"] = []
            changed = True
        else:
            # Normalize legacy tag shapes (mixed case, spaces, etc).
            norm = _normalize_tags(clip["tags"])
            if norm != clip["tags"]:
                clip["tags"] = norm
                changed = True
        if "starred" not in clip:
            clip["starred"] = False
            changed = True
        else:
            clip["starred"] = bool(clip.get("starred"))
        if "play_count" not in clip:
            clip["play_count"] = 0
            changed = True
        else:
            try:
                clip["play_count"] = int(clip.get("play_count") or 0)
            except (TypeError, ValueError):
                clip["play_count"] = 0
                changed = True
        if "last_played_ts" not in clip:
            clip["last_played_ts"] = 0.0
            changed = True
        else:
            try:
                clip["last_played_ts"] = float(clip.get("last_played_ts") or 0.0)
            except (TypeError, ValueError):
                clip["last_played_ts"] = 0.0
                changed = True
        # BPM: default 0.0 = unknown / not yet analysed. Out-of-range
        # values are coerced to 0 so a corrupt manual edit doesn't
        # poison the iPad sort / filter.
        if "bpm" not in clip:
            clip["bpm"] = 0.0
            changed = True
        else:
            try:
                bpm_val = float(clip.get("bpm") or 0.0)
            except (TypeError, ValueError):
                bpm_val = 0.0
                changed = True
            if bpm_val and (bpm_val < 60.0 or bpm_val > 200.0):
                bpm_val = 0.0
                changed = True
            if clip.get("bpm") != bpm_val:
                clip["bpm"] = bpm_val
                changed = True
        # capture_speed defaults to 1.0 for legacy clips so they fire at
        # normal speed (matches their prior behaviour). New clips capture
        # the live speed at save time.
        if "capture_speed" not in clip:
            clip["capture_speed"] = 1.0
            changed = True
        else:
            try:
                cs = float(clip.get("capture_speed") or 1.0)
            except (TypeError, ValueError):
                cs = 1.0
                changed = True
            if cs <= 0 or cs > 5.0:
                cs = 1.0
                changed = True
            if clip.get("capture_speed") != cs:
                clip["capture_speed"] = cs
                changed = True
        # capture_bpm: 0 means unknown — fire just respects capture_speed.
        if "capture_bpm" not in clip:
            clip["capture_bpm"] = 0.0
            changed = True
        return changed

    def create_segment(
        self,
        filepath: str,
        in_seconds: float,
        out_seconds: float,
        name: str = "",
        tags: list = None,
        starred: bool = False,
    ) -> dict:
        """Create a clip with EXPLICIT in/out positions (no player-
        state dependency). Designed for batch curation from the iPad
        Mark-clip form and external scripts.

        Differs from mark_in + mark_out (which read the live player
        position): this takes both positions up front and writes the
        clip immediately. Returns the saved clip dict."""
        import uuid as _uuid
        if not filepath:
            return {"ok": False, "error": "no filepath"}
        try:
            in_s = float(in_seconds)
            out_s = float(out_seconds)
        except (TypeError, ValueError):
            return {"ok": False, "error": "in/out must be numbers"}
        if in_s >= out_s:
            return {"ok": False,
                    "error": f"IN ({in_s}s) >= OUT ({out_s}s)"}
        clean_tags = []
        if tags:
            for t in tags:
                t = str(t or "").strip()
                if t and t not in clean_tags:
                    clean_tags.append(t)
        clip = {
            "id": str(_uuid.uuid4()),
            "name": str(name or "").strip() or f"clip_{int(in_s)}-{int(out_s)}",
            "filepath": filepath,
            "in_seconds": in_s,
            "out_seconds": out_s,
            "duration": out_s - in_s,
            "tags": clean_tags,
            "starred": bool(starred),
            "play_count": 0,
            "last_played_ts": 0.0,
            "bpm": 0.0,
            "capture_speed": 1.0,
            "capture_bpm": 0.0,
        }
        with self._lock:
            self._clips.append(clip)
            self._save()
        # Best-effort thumbnail / BPM analysis (mirrors mark_out path).
        try:
            thumbnails.generate_for_clip(clip)
        except Exception as e:
            logger.debug(f"thumb gen failed for new clip: {e}")
        try:
            if clip_bpm is not None:
                clip_bpm.analyze_async(
                    clip,
                    on_done=lambda bpm, cid=clip["id"]:
                        self._apply_bpm_result(cid, bpm),
                )
        except Exception:
            pass
        return {"ok": True, "clip": clip}

    def reload(self) -> dict:
        """Force re-read of clips.json from disk. Used after batch-
        external-edits of the file (e.g. tonight's `python -c '...'`
        scripts that mark clips outside the running app). Without this
        the in-memory copy stays stale forever.

        Returns {ok, before, after, delta} for reporting."""
        with self._lock:
            before = len(self._clips)
            self._clips = []
            self._load()
            after = len(self._clips)
        return {
            "ok": True,
            "before": before,
            "after": after,
            "delta": after - before,
        }

    def _load(self):
        """Load from disk. Migrates pre-tags clips on first read."""
        try:
            if Path(self._path).exists():
                self._clips = json.loads(Path(self._path).read_text(encoding="utf-8"))
        except Exception:
            self._clips = []
        # Migration: backfill new metadata fields onto any pre-existing
        # clips. We only re-save when something actually changed so we
        # don't bump mtime on a clean db.
        any_migrated = False
        for c in self._clips:
            if isinstance(c, dict) and self._migrate_clip(c):
                any_migrated = True
        if any_migrated:
            try:
                self._save()
            except Exception as e:
                logger.debug(f"Migration save failed: {e}")

    def _save(self):
        """Save to disk. Atomic write (tmp + os.replace) so a torn write
        can never leave clips.json half-serialised. Callers already hold
        self._lock (RLock) in every mutating path; we re-acquire here so
        a direct _save() (e.g. from _load migration) is also guarded."""
        try:
            with self._lock:
                payload = json.dumps(self._clips, indent=2, ensure_ascii=False)
            dest = Path(self._path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(str(tmp), str(dest))
        except Exception as e:
            print(f"Failed to save clips: {e}")
