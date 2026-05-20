"""
Analytics — read-only stats over the clips DB + on-disk cache dirs.

Pure stdlib. Does NOT import clips_db / app_state / main; reads the
JSON file at clips_db_path directly so we never need to modify the
owning module. Loaded clips are cached in-memory for 10 seconds so
a burst of poll calls (top_played + bpm_histogram + today_stats +
disk_usage in the same publish cycle) only hits disk once.

Integration sketch (do NOT apply here — this is just a note for main.py):

    # near other component constructions:
    from analytics import Analytics
    self.analytics = Analytics(
        clips_db_path = CONFIG_DIR / "clips.json",
        proxy_dir     = CACHE_DIR / "proxy",
        thumb_dir     = CACHE_DIR / "thumbnails",
    )

    # inside the existing 2.5s publish/refresh QTimer callback:
    self.state.set_analytics({
        "top_clips":     self.analytics.top_played(10),
        "bpm_histogram": self.analytics.bpm_histogram(),
        "disk_usage":    self.analytics.disk_usage(),
        "today":         self.analytics.today_stats(),
        # Counters owned by main (beats / flips happen in audio_reactive
        # + player_mpv, not in clips.json) — pass them through here so
        # the iPad sees one merged blob:
        "beats_today":   self._beats_today_counter,
        "flips_today":   self._flips_today_counter,
    })

    # app_state.py would gain a thin setter:
    #     def set_analytics(self, blob: dict):
    #         with self._lock:
    #             self.analytics = dict(blob or {})
    # and snapshot() would include it as "analytics".
    # (Per the task constraint we do NOT modify app_state.py here —
    #  the iPad client falls back to {} when missing.)
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# BPM histogram buckets. Half-open intervals [lo, hi) except the last
# which is open-ended (160+). Bands match the cue-points filter chip
# bands roughly, but widened to 20 BPM so a 60-clip library still
# produces non-empty bars.
_BPM_BANDS: list[tuple[str, float, float]] = [
    ("60-80",  60.0,  80.0),
    ("80-100", 80.0,  100.0),
    ("100-120", 100.0, 120.0),
    ("120-140", 120.0, 140.0),
    ("140-160", 140.0, 160.0),
    ("160+",   160.0, 10_000.0),
]


class Analytics:
    """Read-only analytics over clips.json + proxy/thumb cache dirs.

    Cheap to construct. Methods are safe to call from the Qt main thread
    (no I/O on the hot path within the 10s cache window). All methods
    return JSON-serialisable primitives — drop straight onto AppState.
    """

    CACHE_TTL_S = 10.0

    def __init__(self, clips_db_path: Path, proxy_dir: Path, thumb_dir: Path):
        self._clips_db_path = Path(clips_db_path)
        self._proxy_dir = Path(proxy_dir)
        self._thumb_dir = Path(thumb_dir)
        # In-memory cache of the parsed clip list. Refreshed when
        # _cache_age() > CACHE_TTL_S.
        self._cached_clips: list[dict] = []
        self._cached_at: float = 0.0

    # ── Public API ────────────────────────────────────────────────────

    def top_played(self, n: int = 10) -> list[dict]:
        """Return the top-N clips by play_count, descending.

        Each row: {id, name, play_count, bpm}. Zero-play clips ARE
        included only as filler if there aren't enough plays to fill N;
        callers that want "only ever-played" can filter on the client.
        """
        clips = self._get_clips()
        # Stable sort by play_count desc, then last_played_ts desc as a
        # tie-breaker so the newest fire wins among equal-count clips.
        ranked = sorted(
            clips,
            key=lambda c: (
                -int(c.get("play_count") or 0),
                -float(c.get("last_played_ts") or 0.0),
            ),
        )
        out: list[dict] = []
        for c in ranked[: max(0, int(n))]:
            out.append({
                "id": str(c.get("id") or ""),
                "name": str(c.get("name") or ""),
                "play_count": int(c.get("play_count") or 0),
                "bpm": float(c.get("bpm") or 0.0),
            })
        return out

    def bpm_histogram(self) -> list[dict]:
        """Return [{band, count}] for the predefined BPM bands.

        Clips with bpm == 0 (unknown / not yet analysed) are skipped
        entirely; we don't want a phantom "0-60" bar dominating the
        histogram while the backfill thread chews through a fresh DB.
        """
        clips = self._get_clips()
        counts = [0] * len(_BPM_BANDS)
        for c in clips:
            try:
                bpm = float(c.get("bpm") or 0.0)
            except (TypeError, ValueError):
                continue
            if bpm <= 0.0:
                continue
            for i, (_label, lo, hi) in enumerate(_BPM_BANDS):
                if lo <= bpm < hi:
                    counts[i] += 1
                    break
        return [
            {"band": label, "count": counts[i]}
            for i, (label, _lo, _hi) in enumerate(_BPM_BANDS)
        ]

    def disk_usage(self) -> dict:
        """Sum bytes of cached proxies + thumbnails. Single-level scan
        each (non-recursive) — matches how setpiece lays out both
        dirs in practice. Returns MB rounded to 1 decimal.

        Shape: {proxy_mb, thumb_mb, total_files_indexed}.
        """
        proxy_bytes, proxy_files = self._dir_bytes(self._proxy_dir)
        thumb_bytes, thumb_files = self._dir_bytes(self._thumb_dir)
        return {
            "proxy_mb": round(proxy_bytes / (1024 * 1024), 1),
            "thumb_mb": round(thumb_bytes / (1024 * 1024), 1),
            "total_files_indexed": int(proxy_files + thumb_files),
        }

    def today_stats(self) -> dict:
        """Counters for the iPad's "today's session" line.

        - fires_today:  number of distinct clip-fires since local midnight
                        (sum of plays whose last_played_ts >= today). NOTE:
                        play_count is total-time, last_played_ts is just the
                        latest. We can't reconstruct "5 plays today" from
                        play_count alone, so this is "clips fired today"
                        (count of clips with last_played_ts >= midnight),
                        which is the metric the user actually cares about.
        - last_24h:     same idea but rolling 24h window.
        - all_time_clips: total clip count in the DB.
        """
        clips = self._get_clips()
        now = time.time()
        midnight = _local_midnight_ts(now)
        last_24h_cutoff = now - 86_400.0

        fires_today = 0
        last_24h = 0
        for c in clips:
            try:
                ts = float(c.get("last_played_ts") or 0.0)
            except (TypeError, ValueError):
                ts = 0.0
            if ts <= 0:
                continue
            if ts >= midnight:
                fires_today += 1
            if ts >= last_24h_cutoff:
                last_24h += 1
        return {
            "fires_today": fires_today,
            "last_24h": last_24h,
            "all_time_clips": len(clips),
        }

    # ── Internals ─────────────────────────────────────────────────────

    def _get_clips(self) -> list[dict]:
        """Return the parsed clip list, refreshing from disk if the
        in-memory cache is older than CACHE_TTL_S. Always returns a
        list (empty on parse failure)."""
        if (time.time() - self._cached_at) <= self.CACHE_TTL_S and self._cached_clips:
            return self._cached_clips
        self._cached_clips = self._read_clips_from_disk()
        self._cached_at = time.time()
        return self._cached_clips

    def _read_clips_from_disk(self) -> list[dict]:
        """Read + parse clips.json. Quiet on any failure — analytics is
        a nice-to-have, not load-bearing for live playback."""
        try:
            if not self._clips_db_path.exists():
                return []
            raw = self._clips_db_path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else []
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"analytics: failed to read {self._clips_db_path}: {e}")
            return []
        if not isinstance(data, list):
            return []
        # Defensive: only keep dict entries; a corrupt edit could leave
        # bare strings or nulls in the array.
        return [c for c in data if isinstance(c, dict)]

    @staticmethod
    def _dir_bytes(path: Path) -> tuple[int, int]:
        """Sum file sizes (non-recursive) in `path`. Returns (bytes,
        file_count). Returns (0, 0) if the dir doesn't exist yet —
        cache dirs are created lazily by their owners on first write."""
        total = 0
        count = 0
        try:
            if not path.exists() or not path.is_dir():
                return 0, 0
            for entry in path.iterdir():
                try:
                    if entry.is_file():
                        total += entry.stat().st_size
                        count += 1
                except OSError:
                    # Permission errors, race-removed files, etc — skip.
                    continue
        except OSError as e:
            logger.debug(f"analytics: dir scan failed for {path}: {e}")
            return 0, 0
        return total, count


def _local_midnight_ts(now: Optional[float] = None) -> float:
    """Unix timestamp for today's local midnight. Used so "today" lines
    up with the user's wall clock, not UTC."""
    t = now if now is not None else time.time()
    dt = datetime.fromtimestamp(t)
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp()
