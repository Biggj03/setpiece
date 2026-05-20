"""
EDM-phrase tagger — for each clip in path_tags.db3, check if its
dialog tags contain any iconic EDM/dance/pop phrase from the
curated corpus (`edm_phrases.txt`). If yes, add `phrase:<slug>` tags.

WHY THIS EXISTS
---------------
User insight (2026-05-17): "they still play alot of the lyrical drops
from the last 20 years the pool of likely phrases would have to be
quite small ... on any given set the chance it says 'Drop the Bass'
or 'You are my clarity' is like 1 in 5."

Phrase-tagged clips become HERO-VISUAL candidates: when a known
phrase fires in the live music (or in a video's own dialog), the
picker can prioritize the clip whose dialog ALSO contained that
phrase. This creates a tight visual <-> audio loop for the most
predictable / iconic moments in a DJ set.

WHAT IT DOES
------------
1. Loads phrases from `edm_phrases.txt` (one per line; # comments OK;
   `phrase | artist : track` format also accepted, artist part ignored).
2. Slugifies each phrase: lowercase, drop punctuation, hyphenate.
3. For each clip in DB with dialog tags: check if ALL words of the
   phrase appear as dialog: tags on that clip (order-insensitive).
4. If hit: INSERT `phrase:<full-slug>` tag.

WHY THIS WORKS
--------------
- Whisper Phase A tagged ~13k dialog tags across 2.3k files. Each
  dialog tag is one transcribed word. For a phrase like "drop the
  bass" to be detected in a clip, the clip's dialog tags must include
  "drop", "the", and "bass". This is a coarse but precise filter —
  false positives are rare because all three words must coexist.
- Order isn't checked (Whisper transcription is per-word + order
  isn't always reliable for sung vocals). The probability of a clip
  having ALL of [drop, the, bass] without ACTUALLY containing the
  phrase is low.

USAGE
-----
    # Dry-run: see what would tag
    python phrase_tagger.py --dry-run

    # Real run, default corpus path
    python phrase_tagger.py

    # Custom corpus path
    python phrase_tagger.py --corpus my_phrases.txt

    # Refresh (drop existing phrase: tags + retag)
    python phrase_tagger.py --refresh

PERFORMANCE
-----------
Pure SQL + Python set ops. ~1 second per 1000 clips. No ffmpeg, no
audio analysis. Cheap to re-run after corpus updates.
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

APP_STATE = Path.home() / ".setpiece"
DB_PATH = APP_STATE / "path_tags.db3"
DEFAULT_CORPUS = Path(__file__).parent / "edm_phrases.txt"

# Dialog stop-words we should NOT count as "must match" (they're
# present in tons of clips and would make any phrase containing them
# match too liberally). Phrase-matching checks that ONLY the CONTENT
# words (not stop-words) coexist as dialog tags on a clip.
#
# Larger than a typical English stop-word list because dialog tags
# come from voice-over -- words like 'here', 'come', 'now',
# 'baby', 'girl', 'boy' etc. appear in 30-90% of clips. If a phrase
# reduces to ONLY these after filtering, it becomes a noise match.
_STOP_WORDS = frozenset([
    # Articles + prepositions
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at",
    "by", "with", "from", "for", "as", "into", "onto", "over",
    # Pronouns
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "his",
    "she", "her", "it", "its", "they", "them", "their",
    # Aux + common verbs
    "is", "am", "are", "was", "were", "be", "been", "do", "does",
    "did", "have", "has", "had", "will", "would", "could", "should",
    "can", "may", "might",
    # Common adverbs / quantifiers
    "so", "very", "really", "just", "now", "then", "here", "there",
    "all", "any", "some", "no", "not", "more", "most", "much",
    "this", "that", "these", "those",
    # Dialog high-frequency words (block to reduce false positives)
    "yes", "yeah", "oh", "ah", "ok", "okay", "good", "baby",
    "girl", "boy", "come", "go", "down", "up",
    "out", "want", "feel", "see", "make", "take", "get",
    "ready", "let", "love", "say",
])

_SLUG_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def _slugify(phrase: str) -> str:
    """'Drop the Bass' -> 'drop-the-bass'"""
    s = phrase.lower().strip()
    s = _SLUG_RE.sub(" ", s)
    s = _WS_RE.sub("-", s).strip("-")
    return s


def _content_words(phrase: str) -> set[str]:
    """Words from the phrase that must appear in dialog tags.
    Stop-words excluded — we don't require 'the' to appear for
    'drop the bass' to match (Whisper often drops articles)."""
    out = set()
    for w in _WS_RE.split(phrase.lower().strip()):
        w = _SLUG_RE.sub("", w)
        if not w or len(w) < 2 or w in _STOP_WORDS:
            continue
        out.add(w)
    return out


def load_corpus(path: Path) -> list[tuple[str, str, set[str]]]:
    """Return list of (full_phrase, slug, content_words_set).
    Skips comments + blanks + dupes."""
    if not path.is_file():
        logger.error(f"corpus not found: {path}")
        return []
    seen_slugs: set[str] = set()
    out = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            # Strip optional "phrase | artist : track" suffix
            phrase = line.split("|", 1)[0].strip()
            if not phrase:
                continue
            slug = _slugify(phrase)
            if not slug or slug in seen_slugs:
                continue
            words = _content_words(phrase)
            # Require AT LEAST 2 distinct content words. A 1-word phrase
            # after stop-filtering matches anything with that word in
            # dialog -- way too noisy. Phrases like "yeah" / "hands up"
            # (which reduces to {hands}) get dropped here.
            if len(words) < 2:
                continue
            seen_slugs.add(slug)
            out.append((phrase, slug, words))
    return out


def tag_library(
    corpus_path: Path,
    dry_run: bool = False,
    refresh: bool = False,
) -> dict:
    if not DB_PATH.is_file():
        logger.error(f"no DB at {DB_PATH}")
        return {"ok": False, "error": "no db"}

    phrases = load_corpus(corpus_path)
    logger.info(f"loaded {len(phrases)} phrases from {corpus_path.name}")
    if not phrases:
        return {"ok": False, "error": "no phrases"}

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    if refresh and not dry_run:
        n = cur.execute(
            "DELETE FROM file_tags WHERE tag LIKE 'phrase:%'"
        ).rowcount
        conn.commit()
        logger.info(f"refresh: dropped {n} existing phrase: tags")

    # Build {filepath -> set(dialog_words)} for every file with dialog tags.
    # One query, single pass. Cheap.
    rows = cur.execute(
        "SELECT filepath, SUBSTR(tag, 8) FROM file_tags "
        "WHERE tag LIKE 'dialog:%'"
    ).fetchall()
    dialog_by_file: dict[str, set[str]] = {}
    for fp, word in rows:
        dialog_by_file.setdefault(fp, set()).add(word.lower())
    logger.info(f"files with dialog tags: {len(dialog_by_file)}")

    # For each file, check each phrase
    n_inserted = 0
    n_files_tagged = 0
    hit_counts: Counter = Counter()
    files_with_phrases: set[str] = set()

    for fp, dwords in dialog_by_file.items():
        for full, slug, content_words in phrases:
            # Phrase matches if ALL content words of the phrase appear
            # in the file's dialog tag set.
            if content_words and content_words.issubset(dwords):
                tag = f"phrase:{slug}"
                hit_counts[slug] += 1
                files_with_phrases.add(fp)
                if not dry_run:
                    cur.execute(
                        "INSERT OR IGNORE INTO file_tags(filepath, tag) "
                        "VALUES (?, ?)", (fp, tag)
                    )
                    if cur.rowcount:
                        n_inserted += 1

    if not dry_run:
        conn.commit()
    conn.close()

    n_files_tagged = len(files_with_phrases)
    logger.info("")
    logger.info("=" * 56)
    logger.info(f"DONE  files tagged: {n_files_tagged} / inserts: {n_inserted}")
    logger.info(f"  total phrase hits: {sum(hit_counts.values())}")
    logger.info(f"  unique phrases hit: {len(hit_counts)} / {len(phrases)} in corpus")

    print()
    print("=== top 20 phrases by hit count ===")
    for slug, n in hit_counts.most_common(20):
        print(f"  {n:5d}  phrase:{slug}")
    if len(hit_counts) > 20:
        print(f"  ... ({len(hit_counts) - 20} more phrases also hit)")

    print()
    print("=== phrases in corpus that NEVER matched (consider tightening) ===")
    matched_slugs = set(hit_counts.keys())
    never = [s for _, s, _ in phrases if s not in matched_slugs]
    for s in never[:15]:
        print(f"  - {s}")
    if len(never) > 15:
        print(f"  ... ({len(never) - 15} more never matched)")

    return {"ok": True, "files_tagged": n_files_tagged,
            "inserts": n_inserted, "phrases_hit": len(hit_counts)}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                    help=f"phrase corpus file (default: {DEFAULT_CORPUS.name})")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would happen without writing")
    ap.add_argument("--refresh", action="store_true",
                    help="drop existing phrase: tags + retag")
    args = ap.parse_args()
    r = tag_library(corpus_path=args.corpus, dry_run=args.dry_run,
                    refresh=args.refresh)
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
