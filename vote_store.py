"""Per-file vote store. Hardware up/down votes during live VJ rise
to the top (or get suppressed) in the picker. Schema is tiny —
one row per file with a running score + last-touched timestamp.

The picker reads `score(path)` and uses it as a multiplicative
weight on the candidate's selection probability. So +1 vote
nudges that file's odds, several votes really push it up;
downvotes shrink it. Clamped so a heavily-downvoted clip can
still appear (0.1x) and a heavily-upvoted clip caps at 5.0x.

Stored in `~/.setpiece/votes.db3` next to path_tags.db3.
"""

from __future__ import annotations
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_votes (
    filepath TEXT PRIMARY KEY,
    score    INTEGER NOT NULL DEFAULT 0,
    ups      INTEGER NOT NULL DEFAULT 0,
    downs    INTEGER NOT NULL DEFAULT 0,
    updated  REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_votes_score ON file_votes(score DESC);

CREATE TABLE IF NOT EXISTS file_category_votes (
    filepath TEXT NOT NULL,
    letter   TEXT NOT NULL,
    count    INTEGER NOT NULL DEFAULT 0,
    updated  REAL    NOT NULL DEFAULT 0,
    PRIMARY KEY (filepath, letter)
);
CREATE INDEX IF NOT EXISTS idx_catvotes_file ON file_category_votes(filepath);
"""


class VoteStore:
    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self._db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.executescript(_SCHEMA)

    def _bump(self, filepath: str, delta: int) -> tuple[int, int, int]:
        """Adjust score by delta (+1 / -1). Returns (score, ups, downs)
        AFTER the bump."""
        if not filepath:
            return (0, 0, 0)
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT score, ups, downs FROM file_votes WHERE filepath=?",
                (filepath,),
            ).fetchone()
            if row:
                score, ups, downs = row
            else:
                score, ups, downs = 0, 0, 0
            score += delta
            if delta > 0:
                ups += 1
            elif delta < 0:
                downs += 1
            self._conn.execute(
                "INSERT INTO file_votes(filepath, score, ups, downs, updated) "
                "VALUES (?,?,?,?,?) "
                "ON CONFLICT(filepath) DO UPDATE SET "
                "  score=excluded.score, ups=excluded.ups, "
                "  downs=excluded.downs, updated=excluded.updated",
                (filepath, score, ups, downs, now),
            )
        return (score, ups, downs)

    def upvote(self, filepath: str) -> tuple[int, int, int]:
        return self._bump(filepath, +1)

    def downvote(self, filepath: str) -> tuple[int, int, int]:
        return self._bump(filepath, -1)

    def score(self, filepath: str) -> int:
        if not filepath:
            return 0
        with self._lock:
            row = self._conn.execute(
                "SELECT score FROM file_votes WHERE filepath=?",
                (filepath,),
            ).fetchone()
        return int(row[0]) if row else 0

    def stats(self, filepath: str) -> tuple[int, int, int]:
        """(score, ups, downs) for OLED / iPad display."""
        if not filepath:
            return (0, 0, 0)
        with self._lock:
            row = self._conn.execute(
                "SELECT score, ups, downs FROM file_votes WHERE filepath=?",
                (filepath,),
            ).fetchone()
        if not row:
            return (0, 0, 0)
        return (int(row[0]), int(row[1]), int(row[2]))

    def picker_weight(self, filepath: str) -> float:
        """Multiplicative weight for the picker. score 0 -> 1.0;
        +N -> 1 + 0.3N capped at 5.0; -N -> max(0.1, 1 - 0.2N)."""
        s = self.score(filepath)
        if s == 0:
            return 1.0
        if s > 0:
            return min(5.0, 1.0 + 0.3 * s)
        return max(0.1, 1.0 + 0.2 * s)  # s is negative

    def top_voted(self, limit: int = 20) -> list[tuple[str, int]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT filepath, score FROM file_votes "
                "WHERE score > 0 ORDER BY score DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [(r[0], int(r[1])) for r in rows]

    def bottom_voted(self, limit: int = 20) -> list[tuple[str, int]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT filepath, score FROM file_votes "
                "WHERE score < 0 ORDER BY score ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [(r[0], int(r[1])) for r in rows]

    def total_voted(self) -> int:
        with self._lock:
            r = self._conn.execute(
                "SELECT COUNT(*) FROM file_votes"
            ).fetchone()
        return int(r[0] or 0)

    # ─── Category corrections (fuzzy reclassification) ──────────────
    # When the user holds GROUP letter X while a clip plays, that's a
    # vote: "this file is really category X." Cumulative — one wrong
    # press = mild nudge, multiple consistent corrections = override.

    def correct_category(
        self, filepath: str, letter: str
    ) -> tuple[int, dict]:
        """Record a category correction. Returns (this letter's count,
        full dict of all letter counts for this file)."""
        if not filepath or not letter:
            return (0, {})
        letter = letter.upper()
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT count FROM file_category_votes "
                "WHERE filepath=? AND letter=?",
                (filepath, letter),
            ).fetchone()
            count = (int(row[0]) if row else 0) + 1
            self._conn.execute(
                "INSERT INTO file_category_votes(filepath, letter, count, updated) "
                "VALUES (?,?,?,?) "
                "ON CONFLICT(filepath, letter) DO UPDATE SET "
                "  count=excluded.count, updated=excluded.updated",
                (filepath, letter, count, now),
            )
            all_rows = self._conn.execute(
                "SELECT letter, count FROM file_category_votes "
                "WHERE filepath=?",
                (filepath,),
            ).fetchall()
        return (count, {r[0]: int(r[1]) for r in all_rows})

    def category_corrections(self, filepath: str) -> dict:
        """All category-correction counts for a file. Empty if none."""
        if not filepath:
            return {}
        with self._lock:
            rows = self._conn.execute(
                "SELECT letter, count FROM file_category_votes "
                "WHERE filepath=?",
                (filepath,),
            ).fetchall()
        return {r[0]: int(r[1]) for r in rows}
