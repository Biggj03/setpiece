"""
Path-derived auto-tag index — v2.

Drop-in replacement for ``path_tags.py``. Same public API
(``PathTagIndex`` with ``scan_async``, ``top_tags``, ``files_with_tag``,
``files_with_all_tags``, ``tags_for_file``, ``total_files``,
``is_scanning``) so callers don't change.

What's new in v2 vs v1
----------------------

1. **ph<hex> content-ID filter.** v1's hex regex was anchored on pure
   ``^[0-9a-f]{7,}$`` and let site content-IDs like ``ph6038162117963`` slide
   through 23 times in the test sample. v2 also strips any token that
   is one or two alpha chars followed by 6+ hex (catches ``ph``,
   ``vid``, ``id``, plus future variants).

2. **Expanded ``_NOISE`` set** based on a real audit of the library:

   - Video-hosting site names that leaked in as dominant tags
     when files are named after their download source.
   - Stop-word-ish singletons that survived v1's 3-char floor
     (``about``, ``even``, ``before``, ``please``, ``please``, ``when``,
     ``your``, ``youre``, ``their``, ``got``, ``goes``, ``how``, ``let``,
     ``now``, ``over``, ``please``, ``wants``, ``made``, ``makes``,
     ``gets``, ``part``, ``top``, ``only``, ``even``).
   - Roman-numeral fragments (``iii``, ``vol``, ``vols``).
   - Common typos seen in the library (``increadible``, ``exersice``,
     ``youre``, ``dont``).

3. **Multi-word phrase preservation.** A small whitelist of known
   genre/style bigrams (``music video``, ``pole dance``,
   ``slow motion``, ``time lapse``, ``truth or dare``) is detected
   before single-token split and emitted as joined tokens like
   ``slow-motion``. Limits flooding the chip strip with loose tokens.

4. **Simple stem folding.** A built-in singular/plural folder collapses
   ``parties → party``, ``melodies → melody``, ``loops → loop`` etc.
   Pure suffix rules, no ML.

5. **Optional ``min_token_global_frequency``.** If set, on every
   ``top_tags()`` call (or via ``rebuild_filter()``) we hide tokens
   that show up in fewer than N files across the library — kills typos
   and one-off proper nouns without deleting them from storage.
   Default off (=1) so behavior matches v1 unless you opt in.

Schema is unchanged from v1, so the existing DB file works without
migration.
"""

import logging
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DB_FILE = Path.home() / ".setpiece" / "path_tags.db3"

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv"}


def canonical_path(p) -> str:
    """Canonical, comparison-safe form of a filesystem path.

    On Windows this lowercases the drive/dirs and normalises ``/`` vs
    ``\\`` and ``..`` segments; on POSIX it just abspath+normpaths.
    Idempotent — ``canonical_path(canonical_path(x)) == canonical_path(x)``.

    (Audit fix H7 / L4.) The path-tag DB stores ``str(path)`` from an
    ``rglob`` walk whose exact spelling depends on the ``root`` string
    handed to the scan, while ``_publish_library`` builds query paths a
    different way. Run BOTH sides through this helper before comparing
    and the case/separator mismatch that made tag-filtered libraries
    show zero files disappears. Does NOT touch the filesystem (no
    ``resolve()``), so it's cheap and safe on missing files."""
    try:
        return os.path.normcase(os.path.normpath(os.path.abspath(str(p))))
    except Exception:
        return os.path.normcase(os.path.normpath(str(p)))

# Tokens we always drop. Lowercase.
_NOISE: set[str] = {
    # ── v1 carry-overs ─────────────────────────────────────────────────
    # Extensions + container-style noise
    "mp4", "mkv", "mov", "avi", "webm", "m4v",
    # Resolution markers
    "p1080", "p2160", "p720", "p480", "p360",
    "1080p", "2160p", "720p", "480p", "360p",
    "1080", "2160", "720", "480", "360",  # bare numbers also (digit filter would catch these but be explicit)
    "4k", "8k", "uhd",
    # Common English stop words
    "the", "and", "for", "with", "from", "this", "that", "you",
    "are", "was", "but", "not", "all", "any", "one", "out", "off",
    "vol", "feat", "ft", "official",
    # Web cruft
    "com", "www", "video", "music", "watch", "stream",
    # Common filename annotations
    "edit", "remix", "extended", "version", "uncensored", "hd",

    # ── v2 additions, audited against real library ─────────────────────
    # Video-hosting site names leak in as dominant tags when files
    # are named after their source. Add the sites your own library
    # came from here so they drop out of the chip strip.
    # Extra stop-words seen as singletons / low-value bigrams
    "about", "after", "back", "before", "better", "best",
    "can", "dont", "don", "even", "ever", "every",
    "following", "got", "get", "gets", "goes", "going",
    "her", "his", "him", "she", "they", "them", "their",
    "how", "let", "lets", "like", "made", "makes", "make",
    "now", "only", "over", "please",
    "ride", "rides", "riding",  # too generic in this library
    "wants", "want", "wakes", "when", "where", "what",
    "who", "why", "will", "your", "youre", "yours",
    "into", "onto", "upon", "than", "then", "still",
    "just", "very", "more", "much", "some", "such",
    "also", "only", "even", "yet",
    # Generic structural words
    "part", "parts", "top", "name", "names", "thing", "things",
    "way", "ways", "time", "times", "day", "days",
    # Roman-numerals / volume markers
    "iii", "iiii", "vii", "viii", "vols",
    # Typos found in this library — fold/drop
    "increadible",  # → incredible (typo)
    "exersice",     # → exercise (typo)
    "sht",          # truncation of "shit" — too short/generic anyway
    # Generic verbs that flooded as singletons
    "fall", "fell", "got", "hold", "held",
    "look", "looks", "looked", "looking",
    "say", "said", "see", "saw", "seen",
    "take", "took", "taken", "taking",

    # ── v3: codec / release-group / container cruft ────────────────────
    # Scene-style filenames carry tokens like
    # "Title.720p.HEVC.x265.PRT.mp4" — codec / container / release-group
    # markers that dominate the chip list but describe no CONTENT.
    "hevc", "avc", "x264", "x265", "h264", "h265", "av1",
    "aac", "ac3", "eac3", "dts", "flac", "opus", "mp3",
    "prt", "wrb", "scene", "rip", "webrip", "web", "bdrip", "dvdrip",
    "hdrip", "brrip", "cam", "ts", "internal", "proper", "repack",
    "vids", "vid", "clip", "clips", "full", "complete", "compilation",
    "p", "s", "e",  # stray single letters from "S01E02"-style splits
    "fps", "khz", "kbps", "bit", "8bit", "10bit",
}


# Pure hex blob ≥7 chars (kept from v1)
_HEX_BLOB = re.compile(r"^[0-9a-f]{7,}$")
# v2: short alpha prefix + 6+ hex chars (catches ph<hex>, vid<hex>, id<hex>)
_PREFIXED_HEX_ID = re.compile(r"^[a-z]{1,4}[0-9a-f]{6,}$")
# Looks like an MD5/UUID fragment with separators stripped already
_LONG_ALPHANUM_ID = re.compile(r"^[a-z0-9]{10,}$")  # last-resort: random-looking long mixed strings
# v3: numeric-with-unit junk that bloats the orphan tail —
# "100s", "352k", "34jj", "48o", "100th", "2hrs", "36fff", "361m6", etc.
# This was on track to add ~2000 orphan tags from filenames mentioning
# resolution-adjacent / file-size-adjacent strings.
_NUM_UNIT = re.compile(r"^\d+[a-z]+\d*$")
# v3: short alphanumeric hash fragments embedded in filenames like
# "Sunset Loop [5v83gu].mp4" or "Neon Grid [f09ro5].mp4" —
# 4-9 char mixed letter+digit that isn't a known short word. WD14 tags
# like "1girl"/"2girls" come in through vision_tag.py directly (they
# bypass this filter), so dropping mixed alphanum from FILENAME tokens
# doesn't lose them.
_SHORT_HASHISH = re.compile(r"^(?=.*\d)(?=.*[a-z])[a-z0-9]{4,9}$")

_SPLIT = re.compile(r"[^a-zA-Z0-9]+")

# ── Known multi-word phrases we want preserved as single chips ─────────
# Order matters: longest match first. Stored space-joined, emitted with
# hyphens so they look natural in the chip strip ("real-life-hentai").
_PHRASES: list[tuple[str, ...]] = [
    ("music", "video"),
    ("pole", "dance"),
    ("pole", "dancing"),
    ("slow", "motion"),
    ("time", "lapse"),
    ("truth", "or", "dare"),
    ("dance", "party"),
    ("dance", "off"),
]
# Sort longest first so "real life hentai" wins over "real life"
_PHRASES.sort(key=lambda p: -len(p))


# Trivial singular/plural folder. Conservative — only well-known forms.
# Longest suffix first; we stop at the first match.
_PLURAL_RULES = [
    ("dollz", "doll"),
    ("dolls", "doll"),
    ("babes", "babe"),
    ("toys", "toy"),
    ("boys", "boy"),
    ("girls", "girl"),
    ("loops", "loop"),
    ("ies", "y"),           # parties → party, melodies → melody
]


# Words that look plural but aren't, or that we don't want folded.
_NO_FOLD = {
    "christmas", "kiss", "miss", "boss", "glass", "class", "press",
    "dress", "stress", "address", "across", "less",
    "bass",  # specific to this library
    "this", "his", "yes", "bus", "plus", "us",
}


def _fold_stem(tok: str) -> str:
    """Conservative stem folder. Returns a canonical form for known
    plurals; leaves unrecognized words alone."""
    if len(tok) <= 4:
        return tok
    if tok in _NO_FOLD:
        return tok
    for suf, repl in _PLURAL_RULES:
        # Allow the rule to fire even when the suffix == whole word.
        # Require the *result* to be at least 3 chars.
        if tok.endswith(suf) and len(tok) >= len(suf):
            stem = tok[: -len(suf)] + repl
            if len(stem) >= 3:
                return stem
    # Generic -s drop, but only if the result is still 3+ chars and the
    # word doesn't end in "ss" / "us" / "is" / "as" (would mangle
    # "miss", "christmas", "bus")
    if (
        tok.endswith("s")
        and not tok.endswith("ss")
        and not tok.endswith("us")
        and not tok.endswith("is")
        and not tok.endswith("as")
        and not tok.endswith("ys")
        and len(tok) > 4
    ):
        return tok[:-1]
    return tok


def _drop_token(tok: str) -> bool:
    """Return True if ``tok`` should be discarded."""
    if len(tok) < 3:
        return True
    if tok in _NOISE:
        return True
    if tok.isdigit():
        return True
    if _HEX_BLOB.match(tok):
        return True
    if _PREFIXED_HEX_ID.match(tok):
        return True
    # Final heuristic: long mixed alphanumeric token that doesn't look
    # like an English word. Cheap check: if it has digits AND letters
    # AND is ≥10 chars, treat as ID-ish.
    if len(tok) >= 10 and any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
        return True
    # Numeric-with-unit junk: 100s, 352k, 34jj, 48o, 100th, 2hrs etc.
    if _NUM_UNIT.match(tok):
        return True
    # Short hash-looking mixed alphanum 4-9 chars. Safe to drop for path
    # tokens because WD14 tags (1girl/2girls/etc.) add through vision_tag,
    # not through this filter.
    if _SHORT_HASHISH.match(tok):
        return True
    return False


def _extract_phrases(words: list[str]) -> tuple[set[str], list[str]]:
    """Scan ``words`` (already lowercased) for known phrases.
    Returns (phrase_tags, leftover_words). The leftover list still
    contains the constituent words — phrases are *additive* chips, not
    replacements, since the user might also want to filter on the
    individual component (e.g. ``hentai`` alone)."""
    phrase_tags: set[str] = set()
    n = len(words)
    i = 0
    while i < n:
        matched = False
        for phrase in _PHRASES:
            plen = len(phrase)
            if i + plen <= n and tuple(words[i : i + plen]) == phrase:
                phrase_tags.add("-".join(phrase))
                i += plen
                matched = True
                break
        if not matched:
            i += 1
    return phrase_tags, words


def _folder_tag(name: str) -> Optional[str]:
    """Turn a FOLDER name into ONE canonical tag. Performer / studio /
    category folders ('Abigaiil.Morris', 'BBW', 'Big.Tit.Babes') are the
    richest signal in a well-organised library, but the normal per-token
    split shreds them into useless fragments ('abigaiil' + 'morris').
    This keeps the whole folder name as a single tag: lowercased,
    separators (. _ space) collapsed to '-'. Returns None for names that
    aren't useful as a tag (empty, too long, all-digits, pure noise)."""
    if not name:
        return None
    s = name.strip().lower()
    for suf in (".com", ".net", ".org", ".tv"):  # strip site suffix
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = re.sub(r"[\s._]+", "-", s)        # separators → hyphen
    s = re.sub(r"[^a-z0-9-]", "", s)      # drop anything else
    s = re.sub(r"-{2,}", "-", s).strip("-")
    # len < 3 rejects 2-char abbreviation folders ('LH', 'EB', 'SQ') that
    # are meaningless as chips.
    if not s or len(s) < 3 or len(s) > 40 or s.isdigit():
        return None
    if s.count("-") > 4:                 # 5+ segments = probably a title, not a tag
        return None
    if s in _NOISE:
        return None
    return s


def tokenize_path(filepath: str, library_root: Optional[str] = None) -> set[str]:
    """Extract searchable tokens from a path. Behaviour matches v1's
    contract: returns a ``set[str]`` of lowercased, denoised tags.

    Folder parts ALSO each yield one whole-folder tag (see _folder_tag),
    additive on top of the per-token split, so 'Abigaiil.Morris' gives
    both the precise 'abigaiil-morris' chip and the fuzzy 'abigaiil'."""
    p = Path(filepath)
    if library_root:
        try:
            rel = p.relative_to(library_root)
            parts = list(rel.parts)
        except Exception:
            parts = [p.name]
    else:
        parts = list(p.parts)[-4:]
    if not parts:
        return set()
    folder_parts = parts[:-1]
    filename = parts[-1]

    # Per-token split + phrase detection + denoise — on the FILENAME ONLY.
    # Filenames are descriptive sentences, so splitting them into tokens
    # is right. Folder names are single concepts (performer / studio /
    # category) — running the splitter over THOSE just produces fragment
    # noise ('big', 'tit', 'morris', 'lh'); folders go through
    # _folder_tag below instead, which keeps each as one clean tag.
    raw_words = [w.lower().strip() for w in _SPLIT.split(filename) if w.strip()]
    phrase_tags, _ = _extract_phrases(raw_words)
    tokens: set[str] = set(phrase_tags)
    for w in raw_words:
        if _drop_token(w):
            continue
        folded = _fold_stem(w)
        if _drop_token(folded):
            continue
        tokens.add(folded)

    # Whole-folder tags — each directory component above the filename
    # becomes ONE canonical tag (performer / studio / category).
    for fp in folder_parts:
        ft = _folder_tag(str(fp))
        if ft:
            tokens.add(ft)
    return tokens


class PathTagIndex:
    """SQLite-backed file → tag index. API-compatible with v1.

    New optional knob: ``min_token_global_frequency``. When set to N>1,
    ``top_tags()`` filters out any tag that appears in fewer than N
    files across the library. Useful to hide typos and one-off proper
    nouns without rebuilding the DB. Default 1 (off)."""

    def __init__(
        self,
        db_path: Path = DB_FILE,
        min_token_global_frequency: int = 1,
    ):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False, isolation_level=None,
            timeout=30.0,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # busy_timeout: wait up to 30s for a competing writer (action_
        # tagger at workers=4, intro_tagger, etc.) to release the lock
        # rather than failing the whole library scan instantly with
        # "database is locked". Added 2026-05-19 after a scan lost the
        # race and 149 newly-added videos went un-indexed.
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._init_schema()
        self._scan_thread: Optional[threading.Thread] = None
        self._scanning = False
        # Guards the _scanning check-and-set so two near-simultaneous
        # scan_async calls (library_scan + _kick_path_tag_scan racing)
        # can't both pass the guard and spawn two _scan threads issuing
        # BEGIN on the same connection. (Audit fix M12.)
        self._scan_lock = threading.Lock()
        self._min_freq = max(1, int(min_token_global_frequency))

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
        with self._scan_lock:
            return self._scanning

    def scan_async(self, root: str) -> bool:
        # Validate the root BEFORE taking the lock (no need to serialise
        # a filesystem stat).
        if not root or not Path(root).is_dir():
            return False
        # Atomic check-and-set so only one scan thread ever runs at a
        # time (Audit fix M12). Spawn the thread INSIDE the lock so a
        # second caller can't slip in between the flag set and start().
        with self._scan_lock:
            if self._scanning:
                return False
            self._scanning = True
            self._scan_thread = threading.Thread(
                target=self._scan, args=(root,),
                name="PathTagIndex-scan", daemon=True,
            )
            self._scan_thread.start()
        return True

    def top_tags(self, limit: int = 50) -> list[tuple[str, int]]:
        """Return (tag, file_count) pairs sorted by count desc. If
        ``min_token_global_frequency`` was set >1 on the instance,
        low-frequency tags are hidden."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT tag, COUNT(*) AS c FROM file_tags "
                "GROUP BY tag HAVING c >= ? "
                "ORDER BY c DESC, tag ASC LIMIT ?",
                (self._min_freq, int(limit)),
            ).fetchall()
        return [(r[0], int(r[1])) for r in rows]

    def search_tags(self, query: str, limit: int = 30) -> list[tuple[str, int]]:
        """Substring-match tags by query (case-insensitive), return
        (tag, count) pairs sorted by count desc. Used by the iPad's
        tag-search input to surface buried tags (e.g. specific
        performer names) that don't make the top-20 chip strip.

        When the primary substring search finds very few hits AND
        the query is long enough to suggest a typo-tolerant lookup
        would help, falls back to a prefix-of-query search. Example:
        searching "abigail" returns ~2 hits in the source data, but
        the corpus has "abigaiil-morris" (typo'd source filename
        with 19 files). The prefix fallback ("abig%") catches it."""
        q = (query or "").strip().lower()
        if not q:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT tag, COUNT(*) AS c FROM file_tags "
                "WHERE tag LIKE ? "
                "GROUP BY tag HAVING c >= ? "
                "ORDER BY c DESC, tag ASC LIMIT ?",
                (f"%{q}%", self._min_freq, int(limit)),
            ).fetchall()
        primary = [(r[0], int(r[1])) for r in rows]

        # Prefix-fallback when primary sparse + query has >= 4 chars
        # to play with. Searches for tags containing q[:4] -- catches
        # source-data typos that lengthen / re-spell the canonical
        # word.
        total_primary_hits = sum(c for _, c in primary)
        if total_primary_hits < 5 and len(q) >= 4:
            prefix = q[:4]
            if prefix != q:   # else we'd just re-run the same query
                with self._lock:
                    p_rows = self._conn.execute(
                        "SELECT tag, COUNT(*) AS c FROM file_tags "
                        "WHERE tag LIKE ? "
                        "GROUP BY tag HAVING c >= ? "
                        "ORDER BY c DESC, tag ASC LIMIT ?",
                        (f"%{prefix}%", self._min_freq, int(limit)),
                    ).fetchall()
                # Merge primary + prefix, dedupe by tag, keep highest
                # count, re-sort by count desc.
                merged: dict[str, int] = {t: c for t, c in primary}
                for t, c in p_rows:
                    c = int(c)
                    if t not in merged or c > merged[t]:
                        merged[t] = c
                primary = sorted(
                    merged.items(),
                    key=lambda x: (-x[1], x[0]),
                )[:limit]
        return primary

    def similar_files(
        self,
        primary_files: list[str],
        limit: int = 80,
        exclude_tags_starting_with: tuple[str, ...] = (
            "color:", "symmetry:", "geometry:", "motion:",
            "complexity:", "quality:", "res:", "dur:",
        ),
    ) -> list[tuple[str, int]]:
        """Find files most similar to the given primary set, by
        weighted tag-overlap. Returns (filepath, score) sorted by
        score desc. Primary files are excluded from the result.

        Score formula: for each candidate file, score = sum over its
        tags of (count of that tag across primary files). So a file
        that shares many high-frequency primary tags scores higher
        than one sharing rare or generic tags.

        Generic tags (color:warm, motion:dynamic, etc.) are EXCLUDED
        from the scoring because they don't carry similarity signal --
        every file has color/motion tags so they'd dominate.

        Used by bank_auto_split / tag_to_banks to pad each bank
        with the *closest* related content, not just any co-tagged
        file."""
        if not primary_files:
            return []
        primary_set = set(primary_files)
        with self._lock:
            # 1. Build the "primary tag profile" -- weighted by
            #    frequency across primary files. Excludes generic tags.
            placeholders = ",".join("?" * len(primary_files))
            rows = self._conn.execute(
                f"SELECT tag, COUNT(*) AS c FROM file_tags "
                f"WHERE filepath IN ({placeholders}) "
                f"GROUP BY tag ORDER BY c DESC",
                list(primary_files),
            ).fetchall()
            profile: dict[str, int] = {}
            for tag, c in rows:
                if any(tag.startswith(p) for p in exclude_tags_starting_with):
                    continue
                profile[tag] = int(c)
            if not profile:
                return []
            # 2. Pull every candidate file that has ANY profile tag.
            #    Score = sum of profile[tag] over the candidate's tags
            #    that are in the profile. Aggregated server-side via
            #    a JOIN -- much faster than per-file scoring in Python.
            tag_placeholders = ",".join("?" * len(profile))
            # CASE/SUM with profile weights inlined into the SQL
            # via parameter binding. SQLite has no map type so we
            # build a literal "score" addition per tag.
            # For 50 tags this is fine; for very large profiles we'd
            # batch.
            score_expr = " + ".join(
                f"(CASE WHEN tag = ? THEN ? ELSE 0 END)"
                for _ in profile
            )
            params: list = []
            for tag, weight in profile.items():
                params.append(tag)
                params.append(weight)
            cand_rows = self._conn.execute(
                f"SELECT filepath, SUM({score_expr}) AS score "
                f"FROM file_tags "
                f"WHERE tag IN ({tag_placeholders}) "
                f"GROUP BY filepath "
                f"HAVING score > 0 "
                f"ORDER BY score DESC LIMIT ?",
                params + list(profile.keys()) + [int(limit) + len(primary_set)],
            ).fetchall()
        # 3. Drop primary files; keep top `limit`.
        out: list[tuple[str, int]] = []
        for fp, score in cand_rows:
            if fp in primary_set:
                continue
            out.append((fp, int(score)))
            if len(out) >= limit:
                break
        return out

    def co_occurring_tags(
        self,
        filepaths: list[str],
        limit: int = 12,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, int]]:
        """Return tags that appear most frequently across the given
        files, descending by count. Used to find "related" tags for
        bank-padding: pass the files for tag X, get back the tags
        that often co-occur with X. Caller can then pull files for
        those tags as the pad pool."""
        if not filepaths:
            return []
        exclude = exclude or set()
        with self._lock:
            placeholders = ",".join("?" * len(filepaths))
            rows = self._conn.execute(
                f"SELECT tag, COUNT(*) AS c FROM file_tags "
                f"WHERE filepath IN ({placeholders}) "
                f"GROUP BY tag ORDER BY c DESC, tag ASC LIMIT ?",
                list(filepaths) + [int(limit) * 3],
            ).fetchall()
        # Filter out excluded tags + the technical color/symmetry/
        # geometry/motion namespaces (too generic to make good
        # "related" filters). Same with the tiny tags (<2 files).
        result: list[tuple[str, int]] = []
        for tag, c in rows:
            if tag in exclude:
                continue
            if any(tag.startswith(p) for p in (
                "color:", "symmetry:", "geometry:", "motion:",
                "complexity:", "quality:", "res:", "dur:",
            )):
                continue
            if int(c) < 2:
                continue
            result.append((tag, int(c)))
            if len(result) >= limit:
                break
        return result

    def files_with_tag(self, tag: str) -> set[str]:
        if not tag:
            return set()
        with self._lock:
            rows = self._conn.execute(
                "SELECT filepath FROM file_tags WHERE tag = ?",
                (tag.lower().strip(),),
            ).fetchall()
        return {r[0] for r in rows}

    def files_with_all_tags(self, tags: Iterable[str]) -> set[str]:
        tags = [t.lower().strip() for t in tags if t and t.strip()]
        if not tags:
            return set()
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

    def _write_txn(self, ops, label: str = "write") -> bool:
        """Run ops(conn) inside a BEGIN/COMMIT, retrying the whole
        transaction on 'database is locked'. `ops` is a callable taking
        the sqlite connection; it must NOT issue BEGIN/COMMIT itself.

        Why: the library scan competes with the taggers (parallel
        decode workers, each committing) for the single SQLite writer
        slot. Without retry, one lost race aborted the ENTIRE scan and
        newly-added files silently went un-indexed. 12 attempts with
        exponential backoff (~50s budget) on top of the 30s
        busy_timeout means the scan rides through contention instead
        of giving up. (Added 2026-05-19.)"""
        delay = 0.5
        last_exc = None
        for attempt in range(1, 13):
            try:
                with self._lock:
                    self._conn.execute("BEGIN")
                    try:
                        ops(self._conn)
                        self._conn.execute("COMMIT")
                    except Exception:
                        try:
                            self._conn.execute("ROLLBACK")
                        except Exception:
                            pass
                        raise
                return True
            except sqlite3.OperationalError as e:
                last_exc = e
                if "locked" not in str(e).lower():
                    raise
                if attempt == 12:
                    break
                time.sleep(delay)
                delay = min(delay * 1.5, 8.0)
        logger.warning(
            f"PathTagIndex(v2): {label} gave up after 12 retries: "
            f"{last_exc}"
        )
        return False

    def _scan(self, root: str):
        try:
            root_p = Path(root)
            t0 = time.time()
            scanned = 0
            inserted = 0
            updated = 0
            unchanged = 0
            removed = 0
            current_paths: set[str] = set()
            for path in root_p.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in VIDEO_EXTS:
                    continue
                # Skip housekeeping folders that live inside the library
                # root but aren't real content:
                #   _dedup_recycle/  — holding folder for dedup_apply.py
                #                      safe-moved dupes (reversible)
                #   _proxy_cache/    — transcoded playback proxies
                # Without this, a re-scan re-indexes dedup'd files under
                # their mangled holding-folder paths, undoing the dedup.
                parts_lower = {p.lower() for p in path.parts}
                if "_dedup_recycle" in parts_lower or \
                   "_proxy_cache" in parts_lower:
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
                tokens = tokenize_path(sp, str(root_p))

                def _index_one(conn, sp=sp, tokens=tokens,
                               mtime=mtime, size=size):
                    conn.execute(
                        "DELETE FROM file_tags WHERE filepath = ?", (sp,)
                    )
                    conn.executemany(
                        "INSERT INTO file_tags(filepath, tag) VALUES (?, ?)",
                        [(sp, t) for t in tokens],
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO files(filepath, mtime, size, indexed_ts) "
                        "VALUES (?, ?, ?, ?)",
                        (sp, mtime, size, time.time()),
                    )

                if not self._write_txn(_index_one,
                                       label=f"index {sp[-40:]}"):
                    # Couldn't write this file even after retries —
                    # skip it; the next scan will pick it up. Don't
                    # abort the whole scan.
                    continue
                if row is None:
                    inserted += 1
                else:
                    updated += 1
            with self._lock:
                rows = self._conn.execute(
                    "SELECT filepath FROM files WHERE filepath LIKE ?",
                    (str(root_p) + "%",),
                ).fetchall()
            stale = [r[0] for r in rows if r[0] not in current_paths]
            if stale:
                def _drop_stale(conn, stale=stale):
                    for sp in stale:
                        conn.execute(
                            "DELETE FROM file_tags WHERE filepath = ?", (sp,))
                        conn.execute(
                            "DELETE FROM files WHERE filepath = ?", (sp,))
                if self._write_txn(_drop_stale, label="drop stale rows"):
                    removed = len(stale)
            dt = time.time() - t0
            logger.info(
                "PathTagIndex(v2) scan done in %.1fs: %d scanned (+%d ins, ~%d upd, =%d unchanged, -%d removed)",
                dt, scanned, inserted, updated, unchanged, removed,
            )
        except Exception as e:
            logger.error(f"PathTagIndex(v2) scan failed: {e}", exc_info=True)
        finally:
            with self._scan_lock:
                self._scanning = False


# ── self-test: before/after diff on the real library ───────────────────

if __name__ == "__main__":
    import collections
    import os
    import sys

    try:
        import path_tags as v1
    except ImportError:
        print("Could not import path_tags (v1) for comparison.")
        sys.exit(1)

    roots = sys.argv[1:]
    if not roots:
        print("Usage: python path_tags_v2.py <library-root> [more-roots...]")
        sys.exit(1)

    paths: list[tuple[str, str]] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith((".mp4", ".mkv", ".mov", ".avi", ".webm")):
                    paths.append((os.path.join(dirpath, fn), root))

    print(f"Sampling {len(paths)} real files from the library\n")

    v1_freq: collections.Counter[str] = collections.Counter()
    v2_freq: collections.Counter[str] = collections.Counter()
    for fp, root in paths:
        v1_freq.update(v1.tokenize_path(fp, root))
        v2_freq.update(tokenize_path(fp, root))

    v1_tokens = set(v1_freq.keys())
    v2_tokens = set(v2_freq.keys())
    dropped = sorted(v1_tokens - v2_tokens)
    added = sorted(v2_tokens - v1_tokens)

    print(f"v1: {len(v1_tokens)} unique tokens")
    print(f"v2: {len(v2_tokens)} unique tokens")
    print(f"  -> {len(dropped)} dropped, {len(added)} added by v2\n")

    print("=== TOP 20 — v1 ===")
    for t, c in v1_freq.most_common(20):
        print(f"  {c:4d}  {t}")
    print("\n=== TOP 20 — v2 ===")
    for t, c in v2_freq.most_common(20):
        print(f"  {c:4d}  {t}")

    print("\n=== Tokens v2 dropped (sample of 40) ===")
    for t in dropped[:40]:
        print(" ", t)

    print("\n=== Tokens v2 added (multi-word phrases + folded stems) ===")
    for t in added:
        print(" ", t)

    # Spot-check a few interesting filenames.
    print("\n=== Per-file before/after on 5 examples ===")
    interesting = [
        p for p, _ in paths
        if "real-life-hentai" in p or "ph6" in p or "pole" in p.lower()
    ][:5]
    def _safe(s: str) -> str:
        # Strip chars Windows cp1252 console can't render.
        return s.encode("ascii", "replace").decode("ascii")

    for fp in interesting:
        root = next((r for p, r in paths if p == fp), "")
        v1t = v1.tokenize_path(fp, root)
        v2t = tokenize_path(fp, root)
        print(f"\n  FILE: {_safe(Path(fp).name)}")
        print(f"    v1: {sorted(v1t)}")
        print(f"    v2: {sorted(v2t)}")
        print(f"    +v2 only: {sorted(v2t - v1t)}")
        print(f"    -v1 only: {sorted(v1t - v2t)}")
