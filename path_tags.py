"""
Path-derived auto-tag index.

Walk a library tree once, tokenize every file's path (folder names +
filename), persist (filepath → tag) pairs to SQLite. Surfaces a
controlled vocabulary of tags drawn from how the user already organized
their content — no manual tagging required.

DB layout (single file, ~/.setpiece/path_tags.db3):

    files(filepath PRIMARY KEY, mtime, size, indexed_ts)
    file_tags(filepath, tag, PRIMARY KEY (filepath, tag))
    INDEX on file_tags(tag)

Tokenizer rules (intentionally conservative — better to miss than to
flood the chip strip with noise):

    - split on non-alphanumeric
    - lowercase
    - drop tokens <3 chars
    - drop pure-digit tokens
    - drop long hex blobs (URL-ish IDs like 6459caa05c91d)
    - drop resolution/extension noise (p1080, mp4, 720p, …)
    - drop English stop words and generic web cruft
"""

import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DB_FILE = Path.home() / ".setpiece" / "path_tags.db3"

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

# Tokens we always drop. Lowercase.
_NOISE = {
    # Extensions + container-style noise
    "mp4", "mkv", "mov", "avi", "webm", "m4v",
    # Resolution markers
    "p1080", "p2160", "p720", "p480", "p360",
    "1080p", "2160p", "720p", "480p", "360p", "4k",
    # Common English stop words
    "the", "and", "for", "with", "from", "this", "that", "you",
    "are", "was", "but", "not", "all", "any", "one", "out", "off",
    "vol", "feat", "ft", "official",
    # Web cruft
    "com", "www", "video", "music", "watch", "stream",
    # Common filename annotations we don't want as tags
    "edit", "remix", "extended", "version", "uncensored", "hd",
}
# Tokens that consist entirely of hex/digits longer than 7 chars look
# like content IDs — drop them by regex (faster than seeding a big set).
_HEX_BLOB = re.compile(r"^[0-9a-f]{7,}$")
_SPLIT = re.compile(r"[^a-zA-Z0-9]+")


def tokenize_path(filepath: str, library_root: Optional[str] = None) -> set[str]:
    """Extract searchable tokens from a path. If library_root is given,
    we tokenize relative to it (so the root-folder name itself doesn't
    become a dominant tag on every file)."""
    p = Path(filepath)
    if library_root:
        try:
            rel = p.relative_to(library_root)
            parts = list(rel.parts)
        except Exception:
            parts = [p.name]
    else:
        # Last 4 segments — captures the file + a few enclosing folders
        # without bringing in root-of-drive noise like "E:\".
        parts = list(p.parts)[-4:]
    text = " ".join(str(s) for s in parts)
    tokens: set[str] = set()
    for t in _SPLIT.split(text):
        t = t.lower().strip()
        if len(t) < 3:
            continue
        if t in _NOISE:
            continue
        if t.isdigit():
            continue
        if _HEX_BLOB.match(t):
            continue
        tokens.add(t)
    return tokens


class PathTagIndex:
    """SQLite-backed file → tag index. Thread-safe via a single
    connection guarded by a lock; queries are short."""

    def __init__(self, db_path: Path = DB_FILE):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False, isolation_level=None,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._scan_thread: Optional[threading.Thread] = None
        self._scanning = False

    # ── schema ─────────────────────────────────────────────────────────

    def _init_schema(self):
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    filepath TEXT PRIMARY KEY,
                    mtime REAL,
                    size INTEGER,
                    indexed_ts REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS file_tags (
                    filepath TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (filepath, tag)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tag ON file_tags(tag)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_filepath ON file_tags(filepath)")

    # ── public API ─────────────────────────────────────────────────────

    def is_scanning(self) -> bool:
        return self._scanning

    def scan_async(self, root: str) -> bool:
        """Start a background scan of `root`. Returns False if a scan is
        already in flight."""
        if self._scanning:
            return False
        if not root or not Path(root).is_dir():
            return False
        self._scanning = True
        self._scan_thread = threading.Thread(
            target=self._scan, args=(root,),
            name="PathTagIndex-scan", daemon=True,
        )
        self._scan_thread.start()
        return True

    def top_tags(self, limit: int = 50) -> list[tuple[str, int]]:
        """Return (tag, file_count) pairs sorted by count desc."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT tag, COUNT(*) AS c FROM file_tags "
                "GROUP BY tag ORDER BY c DESC, tag ASC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [(r[0], int(r[1])) for r in rows]

    def files_with_tag(self, tag: str) -> set[str]:
        """All filepaths tagged with `tag`."""
        if not tag:
            return set()
        with self._lock:
            rows = self._conn.execute(
                "SELECT filepath FROM file_tags WHERE tag = ?",
                (tag.lower().strip(),),
            ).fetchall()
        return {r[0] for r in rows}

    def files_with_all_tags(self, tags: Iterable[str]) -> set[str]:
        """Intersection — files that have ALL the given tags."""
        tags = [t.lower().strip() for t in tags if t and t.strip()]
        if not tags:
            return set()
        # Inner-join one table per tag would work but is awkward at the
        # SQL layer; intersect Python sets instead — fast at our scale.
        result: Optional[set[str]] = None
        for tag in tags:
            files = self.files_with_tag(tag)
            if result is None:
                result = files
            else:
                result &= files
            if not result:
                return set()
        return result or set()

    def tags_for_file(self, filepath: str) -> set[str]:
        if not filepath:
            return set()
        with self._lock:
            rows = self._conn.execute(
                "SELECT tag FROM file_tags WHERE filepath = ?",
                (filepath,),
            ).fetchall()
        return {r[0] for r in rows}

    def total_files(self) -> int:
        with self._lock:
            r = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()
        return int(r[0] or 0)

    # ── background scan ────────────────────────────────────────────────

    def _scan(self, root: str):
        try:
            root_p = Path(root)
            t0 = time.time()
            scanned = 0
            inserted = 0
            updated = 0
            unchanged = 0
            removed = 0
            # Walk all video files under root.
            current_paths: set[str] = set()
            for path in root_p.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in VIDEO_EXTS:
                    continue
                scanned += 1
                sp = str(path)
                current_paths.add(sp)
                try:
                    st = path.stat()
                    mtime = float(st.st_mtime)
                    size = int(st.st_size)
                except Exception:
                    continue
                with self._lock:
                    row = self._conn.execute(
                        "SELECT mtime, size FROM files WHERE filepath = ?",
                        (sp,),
                    ).fetchone()
                if row is not None:
                    prev_mtime, prev_size = float(row[0] or 0), int(row[1] or 0)
                    if abs(prev_mtime - mtime) < 1.0 and prev_size == size:
                        unchanged += 1
                        continue
                # Tokenize + insert. CRITICAL: only delete path-derived
                # tags (no colon = path token). Enricher tags use a
                # namespace prefix (`performer:`, `studio:`, `bpm:`,
                # `dialog:`, `color:`, `palette:`, `cast:`, `quality:`,
                # `res:`, `dur:`) — those are HARD-EARNED data that
                # would be wiped if we did a blanket DELETE here. Bug
                # was: auto_organize.py file moves triggered this scan
                # and nuked thousands of enricher tags. The
                # auto_organize.py tag-preservation hack worked around
                # it; this is the upstream fix so any source of rescan
                # is safe.
                tokens = tokenize_path(sp, str(root_p))
                with self._lock:
                    self._conn.execute("BEGIN")
                    # Only delete path tokens (no ':' = path-derived).
                    # Enricher tags (with ':') survive across rescans.
                    self._conn.execute(
                        "DELETE FROM file_tags WHERE filepath = ? "
                        "AND tag NOT LIKE '%:%'",
                        (sp,),
                    )
                    self._conn.executemany(
                        "INSERT OR IGNORE INTO file_tags(filepath, tag) "
                        "VALUES (?, ?)",
                        [(sp, t) for t in tokens],
                    )
                    self._conn.execute(
                        "INSERT OR REPLACE INTO files(filepath, mtime, size, indexed_ts) "
                        "VALUES (?, ?, ?, ?)",
                        (sp, mtime, size, time.time()),
                    )
                    self._conn.execute("COMMIT")
                if row is None:
                    inserted += 1
                else:
                    updated += 1
            # Drop entries for files that no longer exist under root.
            # (Only files that previously sat inside this root — don't
            # clobber files indexed from a different scan root.)
            with self._lock:
                rows = self._conn.execute(
                    "SELECT filepath FROM files WHERE filepath LIKE ?",
                    (str(root_p) + "%",),
                ).fetchall()
            stale = [r[0] for r in rows if r[0] not in current_paths]
            if stale:
                with self._lock:
                    self._conn.execute("BEGIN")
                    for sp in stale:
                        self._conn.execute("DELETE FROM file_tags WHERE filepath = ?", (sp,))
                        self._conn.execute("DELETE FROM files WHERE filepath = ?", (sp,))
                    self._conn.execute("COMMIT")
                removed = len(stale)
            dt = time.time() - t0
            logger.info(
                "PathTagIndex scan done in %.1fs: %d scanned (+%d ins, ~%d upd, =%d unchanged, -%d removed)",
                dt, scanned, inserted, updated, unchanged, removed,
            )
        except Exception as e:
            logger.error(f"PathTagIndex scan failed: {e}", exc_info=True)
        finally:
            self._scanning = False
