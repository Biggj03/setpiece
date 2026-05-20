"""Persistent per-day JSONL log of VJ session activity.

Captures every fire, flip, beat (throttled), deck-load, scratch-save, bank
switch and arbitrary custom events. Designed to be called from many threads
(audio-reactive, MK2 reader, Qt main, HTTP server) without raising into the
caller — failures are swallowed and logged.

File layout:
    ~/.setpiece/sessions/YYYY-MM-DD.jsonl

One JSON object per line. Schema (per event):
    ts        ISO-local timestamp, e.g. "2026-05-13T19:42:01.123456"
    ts_unix   float seconds since epoch
    event     event type string ("fire", "flip", "beat", "deck_load", ...)
    ...       event-specific fields

Pure stdlib. No external deps.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# How often record_beat() actually writes a line. 1 of every BEAT_THROTTLE
# beats hits disk. At 130 BPM that's ~4 lines/min instead of 130.
BEAT_THROTTLE = 32


class SessionLog:
    """Append-only JSONL session activity log, one file per local day."""

    def __init__(self, dir_path: Path = Path.home() / ".setpiece" / "sessions"):
        self.dir_path = Path(dir_path)
        try:
            self.dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error("Could not create session log dir %s: %s", self.dir_path, e)

        self._lock = threading.Lock()
        self._beat_counter = 0  # internal — drives BEAT_THROTTLE

        self._current_date: Optional[date] = None
        self._fh = None  # type: Optional[Any]
        self._current_path: Optional[Path] = None

        # Open today's file lazily on first write — but try once now so a
        # caller knows immediately if the dir is unwritable.
        self._ensure_file_open()

    # ------------------------------------------------------------------ file mgmt

    def _file_for_date(self, d: date) -> Path:
        return self.dir_path / f"{d.isoformat()}.jsonl"

    def _ensure_file_open(self) -> bool:
        """Make sure self._fh points at today's file. Returns True on success."""
        today = date.today()
        if self._fh is not None and self._current_date == today:
            return True

        # Close stale handle from a previous day if we crossed midnight.
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception as e:
                logger.warning("Error closing stale session log: %s", e)
            self._fh = None

        path = self._file_for_date(today)
        try:
            # Line-buffered append. We also flush explicitly per record so a
            # crash doesn't lose the last few events.
            self._fh = open(path, "a", encoding="utf-8", buffering=1)
            self._current_date = today
            self._current_path = path
            return True
        except OSError as e:
            logger.error("Could not open session log %s: %s", path, e)
            self._fh = None
            self._current_date = None
            self._current_path = None
            return False

    # ------------------------------------------------------------------ core write

    def record(self, event_type: str, **fields) -> None:
        """Append one JSONL event. Never raises into caller."""
        try:
            now = time.time()
            payload = {
                "ts": datetime.fromtimestamp(now).isoformat(timespec="microseconds"),
                "ts_unix": now,
                "event": event_type,
            }
            # Don't let user fields clobber our metadata.
            for k, v in fields.items():
                if k in payload:
                    continue
                payload[k] = v

            line = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error("session_log: could not serialize event %r: %s", event_type, e)
            return

        with self._lock:
            if not self._ensure_file_open():
                return
            try:
                self._fh.write(line + "\n")
                self._fh.flush()
            except Exception as e:
                logger.error("session_log: write failed: %s", e)

    # ------------------------------------------------------------------ convenience wrappers

    def record_fire(self, source: str, filepath: str, **extra) -> None:
        """Log a fire event.

        source: 'mk2_pad' | 's2_pad' | 'bank' | 'scratch_chip' | 'auto_flip'
        """
        self.record("fire", source=source, filepath=filepath, **extra)

    def record_flip(self, from_path: str, to_path: str, source: str) -> None:
        self.record("flip", from_path=from_path, to_path=to_path, source=source)

    def record_beat(self, bpm: float) -> None:
        """Throttled — only 1 of every BEAT_THROTTLE beats is persisted."""
        # Counter increment is cheap; protect it so two threads can't both
        # hit the "write" slot.
        with self._lock:
            self._beat_counter += 1
            should_write = (self._beat_counter % BEAT_THROTTLE) == 1
            counter_snapshot = self._beat_counter
        if should_write:
            self.record("beat", bpm=bpm, counter=counter_snapshot)

    def record_deck_load(self, slot: int, filepath: str) -> None:
        self.record("deck_load", slot=slot, filepath=filepath)

    def record_save_clip(self, clip_id: str, name: str) -> None:
        self.record("save_clip", clip_id=clip_id, name=name)

    def record_bank_switch(self, letter: str) -> None:
        self.record("bank_switch", letter=letter)

    # ------------------------------------------------------------------ read-side

    def _read_events(self, target_date: Optional[str] = None):
        """Yield event dicts from a given day's file. Skips bad lines."""
        if target_date is None:
            target_date = date.today().isoformat()
        path = self.dir_path / f"{target_date}.jsonl"
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for lineno, raw in enumerate(f, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError as e:
                        logger.warning("session_log: skipping bad line %d in %s: %s",
                                       lineno, path, e)
        except OSError as e:
            logger.error("session_log: could not read %s: %s", path, e)

    def stats_today(self) -> dict:
        """Aggregate counters for today's log."""
        fires = 0
        flips = 0
        beats = 0
        save_clips = 0
        bank_switches = 0
        per_source: Counter[str] = Counter()
        first_ts: Optional[str] = None
        last_ts: Optional[str] = None

        for ev in self._read_events():
            ts = ev.get("ts")
            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

            kind = ev.get("event")
            if kind == "fire":
                fires += 1
                src = ev.get("source")
                if src:
                    per_source[src] += 1
            elif kind == "flip":
                flips += 1
                src = ev.get("source")
                if src:
                    per_source[src] += 1
            elif kind == "beat":
                beats += 1
            elif kind == "save_clip":
                save_clips += 1
            elif kind == "bank_switch":
                bank_switches += 1

        return {
            "fires_total": fires,
            "flips_total": flips,
            "beats_recorded": beats,
            "save_clips": save_clips,
            "bank_switches": bank_switches,
            "per_source": dict(per_source),
            "first_event_ts": first_ts,
            "last_event_ts": last_ts,
        }

    def replay(self, date_str: Optional[str] = None, fmt: str = "console") -> str:
        """Render a human timeline of events for a day.

        date_str: 'YYYY-MM-DD' or None for today.
        fmt: 'console' (plain) or 'markdown'.
        """
        target = date_str or date.today().isoformat()
        events = list(self._read_events(target))

        if fmt == "markdown":
            return self._render_markdown(target, events)
        return self._render_console(target, events)

    @staticmethod
    def _short_ts(ts: str) -> str:
        # ISO -> HH:MM:SS for compactness.
        if not ts:
            return "--:--:--"
        try:
            return datetime.fromisoformat(ts).strftime("%H:%M:%S")
        except ValueError:
            return ts[:8]

    @staticmethod
    def _basename(p: Any) -> str:
        if not p:
            return ""
        try:
            return os.path.basename(str(p))
        except Exception:
            return str(p)

    def _format_event(self, ev: dict) -> str:
        kind = ev.get("event", "?")
        if kind == "fire":
            return f"FIRE     [{ev.get('source','?')}] {self._basename(ev.get('filepath'))}"
        if kind == "flip":
            return (f"FLIP     [{ev.get('source','?')}] "
                    f"{self._basename(ev.get('from_path'))} -> "
                    f"{self._basename(ev.get('to_path'))}")
        if kind == "beat":
            return f"BEAT     bpm={ev.get('bpm','?')} (#{ev.get('counter','?')})"
        if kind == "deck_load":
            return f"DECK     slot={ev.get('slot','?')} {self._basename(ev.get('filepath'))}"
        if kind == "save_clip":
            return f"SAVE     id={ev.get('clip_id','?')} \"{ev.get('name','')}\""
        if kind == "bank_switch":
            return f"BANK     -> {ev.get('letter','?')}"
        # Unknown / custom event: dump remaining fields.
        extras = {k: v for k, v in ev.items() if k not in ("ts", "ts_unix", "event")}
        return f"{kind.upper():<8} {extras}"

    def _render_console(self, target: str, events: list) -> str:
        lines = [f"=== Session Replay {target} ({len(events)} events) ==="]
        if not events:
            lines.append("  (no events)")
            return "\n".join(lines)
        for ev in events:
            lines.append(f"  {self._short_ts(ev.get('ts',''))}  {self._format_event(ev)}")
        lines.append(f"=== {len(events)} events ===")
        return "\n".join(lines)

    def _render_markdown(self, target: str, events: list) -> str:
        lines = [f"# Session Replay — {target}", "", f"_{len(events)} events_", ""]
        if not events:
            lines.append("_(no events)_")
            return "\n".join(lines)
        lines.append("| Time | Event | Detail |")
        lines.append("| ---- | ----- | ------ |")
        for ev in events:
            kind = ev.get("event", "?")
            detail = self._format_event(ev)
            # Strip the leading "KIND " label off detail so the table column is cleaner.
            detail_clean = detail
            prefix_guess = detail.split(None, 1)
            if len(prefix_guess) == 2:
                detail_clean = prefix_guess[1]
            lines.append(f"| {self._short_ts(ev.get('ts',''))} | {kind} | {detail_clean} |")
        return "\n".join(lines)

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Flush and close the current day's file handle. Idempotent."""
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                    self._fh.close()
                except Exception as e:
                    logger.warning("session_log: close error: %s", e)
                finally:
                    self._fh = None
                    self._current_date = None
                    self._current_path = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


# ===================================================================== self-test

def _selftest() -> None:
    import random

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    with tempfile.TemporaryDirectory(prefix="setpiece_sessionlog_") as tmp:
        tmpdir = Path(tmp)
        print(f"[selftest] temp dir: {tmpdir}")

        log = SessionLog(dir_path=tmpdir)

        sources = ["mk2_pad", "s2_pad", "bank", "scratch_chip", "auto_flip"]
        clips = [
            "C:/vj/clip_alpha.mp4",
            "C:/vj/clip_beta.mp4",
            "C:/vj/clip_gamma.mp4",
            "C:/vj/clip_delta.mp4",
        ]

        # Plan: ~50 mixed events
        # 20 fires, 5 flips, 80 beats (=> 80/32 = 3 persisted),
        # 4 deck_loads, 3 save_clips, 2 bank_switches, plus 1 custom event.
        expected_fires = 20
        expected_flips = 5
        expected_beat_writes = 0  # computed below
        expected_save_clips = 3
        expected_bank_switches = 2

        for i in range(expected_fires):
            log.record_fire(random.choice(sources), random.choice(clips), velocity=random.randint(40, 127))

        for i in range(expected_flips):
            a, b = random.sample(clips, 2)
            log.record_flip(a, b, source=random.choice(sources))

        # Beats — only 1 of every BEAT_THROTTLE persists.
        beat_count = 80
        for i in range(beat_count):
            log.record_beat(bpm=128.0 + (i % 3))
        # Mirror the internal modulo logic exactly:
        expected_beat_writes = sum(1 for n in range(1, beat_count + 1) if n % BEAT_THROTTLE == 1)

        for slot, clip in enumerate(clips):
            log.record_deck_load(slot, clip)
        expected_deck_loads = len(clips)

        for i in range(expected_save_clips):
            log.record_save_clip(clip_id=f"clip_{i:03d}", name=f"my mix {i}")

        for letter in ("A", "B"):
            log.record_bank_switch(letter)

        # Custom event via raw record()
        log.record("custom_marker", note="drop incoming", level=0.87)
        expected_custom = 1

        total_expected = (
            expected_fires + expected_flips + expected_beat_writes
            + expected_deck_loads + expected_save_clips
            + expected_bank_switches + expected_custom
        )

        log.close()

        # Verify file exists and count lines.
        files = list(tmpdir.glob("*.jsonl"))
        assert len(files) == 1, f"expected 1 jsonl file, got {len(files)}"
        log_file = files[0]
        with open(log_file, "r", encoding="utf-8") as f:
            actual_lines = sum(1 for _ in f if _.strip())

        print(f"[selftest] expected {total_expected} lines, found {actual_lines}")
        assert actual_lines == total_expected, (
            f"line count mismatch: expected {total_expected}, got {actual_lines}"
        )

        # Re-open to read stats (fresh handle, same dir).
        log2 = SessionLog(dir_path=tmpdir)
        stats = log2.stats_today()
        print("[selftest] stats_today:")
        for k, v in stats.items():
            print(f"    {k}: {v}")

        assert stats["fires_total"] == expected_fires, stats
        assert stats["flips_total"] == expected_flips, stats
        assert stats["beats_recorded"] == expected_beat_writes, stats
        assert stats["save_clips"] == expected_save_clips, stats
        assert stats["bank_switches"] == expected_bank_switches, stats
        assert stats["first_event_ts"] is not None
        assert stats["last_event_ts"] is not None
        # Per-source covers both fires and flips.
        per_source_total = sum(stats["per_source"].values())
        assert per_source_total == expected_fires + expected_flips, stats

        # Replay smoke tests.
        console = log2.replay(fmt="console")
        assert "Session Replay" in console
        assert "FIRE" in console
        assert "BANK" in console
        print("[selftest] console replay first 12 lines:")
        for line in console.splitlines()[:12]:
            print(f"    {line}")

        md = log2.replay(fmt="markdown")
        assert md.startswith("# Session Replay")
        assert "| Time | Event | Detail |" in md
        print(f"[selftest] markdown replay length: {len(md)} chars")

        # Replay for a date with no file -> graceful empty result.
        empty = log2.replay(date_str="1999-01-01")
        assert "no events" in empty
        print("[selftest] empty-day replay OK")

        log2.close()

        # ------ Thread safety smoke test ------
        log3 = SessionLog(dir_path=tmpdir / "threaded")
        N_THREADS = 8
        PER_THREAD = 50

        def worker(tid: int):
            for i in range(PER_THREAD):
                log3.record_fire("mk2_pad", f"thread{tid}_clip{i}.mp4", tid=tid, i=i)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        log3.close()

        log3b = SessionLog(dir_path=tmpdir / "threaded")
        tstats = log3b.stats_today()
        expected_thread_fires = N_THREADS * PER_THREAD
        print(f"[selftest] threaded fires: got {tstats['fires_total']}, expected {expected_thread_fires}")
        assert tstats["fires_total"] == expected_thread_fires, tstats
        log3b.close()

        print()
        print("=" * 60)
        print("SESSION LOG SELF-TEST: PASS")
        print(f"  Events written:    {total_expected}")
        print(f"  Beats throttled:   {beat_count} beats -> {expected_beat_writes} writes (1/{BEAT_THROTTLE})")
        print(f"  Thread test:       {N_THREADS} threads x {PER_THREAD} writes = {expected_thread_fires} OK")
        print(f"  File size:         {log_file.stat().st_size} bytes for {actual_lines} events")
        print(f"  Avg bytes/event:   {log_file.stat().st_size / actual_lines:.1f}")
        print("=" * 60)


if __name__ == "__main__":
    _selftest()
