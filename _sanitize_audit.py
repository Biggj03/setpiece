"""Sanitization audit for the open-source release.

Scans a project tree for content-domain terms, site/studio names, performer
names, personal identifiers, and bake-in bank labels. Produces a markdown
report so we know exactly what to scrub before a public commit.

This script is tooling, not part of the app. It is intentionally GENERIC:
the in-source denylist below holds only category/domain terms that are safe
to publish (they describe what to look for, not the private content itself).

LOCAL EXTENSION TERM-LIST  (the important part)
-----------------------------------------------
A term-list audit can only catch terms it knows. The in-source denylist
necessarily misses content-specific names — a performer or subfolder in the
testbed library — because writing those names into this public script would
itself be the leak it's trying to prevent.

So the audit also reads an OPTIONAL local file, `_sanitize_audit.extra-terms.txt`,
sitting beside this script. One term per line; blank lines and `#` comments
ignored; each term may be suffixed with `|word` for a word-boundary match
(default is case-insensitive substring). That file is GITIGNORED — it holds
the user's real testbed subfolder names and must never be committed. The
populate helper below generates it from a content-source directory.

    python _sanitize_audit.py [ROOT]                  # scan (auto-loads extension)
    python _sanitize_audit.py [ROOT] -o out.md        # custom report path
    python _sanitize_audit.py --no-extra [ROOT]       # ignore the extension
    python _sanitize_audit.py --populate "D:/library"  # (re)write extension list
                                                       #   from that dir's subfolder
                                                       #   names, then exit

This keeps the public script harmless while the local extension catches
content-specific leaks the generic list can't see.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# --- denylist -------------------------------------------------------------
# mode: "sub" = case-insensitive substring; "word" = \bword\b boundary match.
# Categories order the report; "domain" hits are the most urgent.
#
# GENERIC ONLY. Do not add real performer names or testbed subfolder names
# here — those go in the gitignored _sanitize_audit.extra-terms.txt so this
# file stays publishable.

DENYLIST: list[tuple[str, str, str]] = [
    # (term, category, mode)
    # content domain — unambiguous, scrub on sight
    ("pmv",            "domain",     "word"),
    ("porn",           "domain",     "sub"),
    ("nsfw",           "domain",     "sub"),
    ("xxx",            "domain",     "word"),
    ("2257",           "domain",     "sub"),
    ("onlyfans",       "domain",     "sub"),
    ("fansly",         "domain",     "sub"),
    ("creampie",       "domain",     "sub"),
    # site / studio names
    ("brazzers",       "site",       "sub"),
    ("lilhumpers",     "site",       "sub"),
    ("plumperpass",    "site",       "sub"),
    ("moneytalks",     "site",       "sub"),
    ("wasteland",      "site",       "sub"),
    ("faphouse",       "site",       "sub"),
    ("xhamster",       "site",       "sub"),
    ("pmvhaven",       "site",       "sub"),
    # performer names (word-boundary — real names false-positive easily)
    ("roxy raye",      "performer",  "sub"),
    ("lila",           "performer",  "word"),
    # personal / machine identifiers — must not ship
    ("earh c-137",     "identity",   "sub"),
    ("earh",           "identity",   "word"),
    # internal codenames worth a look
    ("38g",            "codename",   "word"),
    # testbed library path roots (the folder these clips live under)
    ("recycle bin",    "path",       "sub"),
    # bank labels baked into code/config (review, may be legit English words)
    ("boobs",          "bank-label", "word"),
    ("hardcore",       "bank-label", "word"),
    ("ass",            "bank-label", "word"),
    ("cock",           "bank-label", "word"),
    ("fuck",           "bank-label", "word"),
]

# Name of the local, gitignored extension term-list (beside this script).
EXTRA_TERMS_FILE = "_sanitize_audit.extra-terms.txt"

# --- file handling --------------------------------------------------------

SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".benchmarks",
    "hidapi-tmp", "node_modules", ".venv", "venv", ".venv-analyze", "env",
}

# never grepped (binary / media); existence still noted if name matches
BINARY_EXTS = {
    ".dll", ".pt", ".pth", ".zip", ".pyc", ".pyd", ".so", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".webp", ".pdf",
    ".db3", ".db", ".sqlite", ".sqlite3", ".ttf", ".woff", ".woff2",
    ".mp4", ".mkv", ".webm", ".avi", ".wmv", ".mov", ".mp3", ".wav",
    ".npy", ".npz", ".bin",
}

# data/log artifacts — list for wholesale-exclusion review, don't term-scan
ARTIFACT_EXTS = {".log", ".db3", ".db", ".sqlite", ".sqlite3"}

MAX_GREP_BYTES = 8 * 1024 * 1024  # files larger than this are reported, not grepped


def load_extra_terms(script_dir: Path) -> list[tuple[str, str, str]]:
    """Read the gitignored extension term-list, if present. Format: one term
    per line; `#` comments + blanks ignored; optional `|word` suffix selects
    word-boundary mode (default substring). All extension terms get the
    "extra" category so the report shows them distinctly. Returns [] if the
    file is absent — the audit then runs on the generic denylist alone."""
    f = script_dir / EXTRA_TERMS_FILE
    if not f.is_file():
        return []
    out: list[tuple[str, str, str]] = []
    for raw in f.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        mode = "sub"
        if line.endswith("|word"):
            line = line[: -len("|word")].strip()
            mode = "word"
        if line:
            out.append((line.lower(), "extra", mode))
    return out


def compile_patterns(denylist) -> list[tuple[re.Pattern, str, str]]:
    out = []
    for term, cat, mode in denylist:
        if mode == "word":
            pat = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        else:
            pat = re.compile(re.escape(term), re.IGNORECASE)
        out.append((pat, term, cat))
    return out


def populate_extra_terms(content_dir: Path, script_dir: Path) -> int:
    """Walk `content_dir` and write each immediate-and-nested SUBFOLDER NAME
    (not files, not full paths) into the gitignored extension term-list, so
    the audit learns the testbed's performer/category folder names. Run
    locally; the output file is gitignored and must never be committed.

    Writes per-folder names only — never a content/file listing."""
    if not content_dir.is_dir():
        print(f"--populate: not a directory: {content_dir}", file=sys.stderr)
        return 2
    names = _extract_content_tokens(content_dir)
    out_path = script_dir / EXTRA_TERMS_FILE
    header = [
        "# Sanitization audit — LOCAL extension term-list. GITIGNORED.",
        "# Auto-generated DISTINCTIVE content tokens from a content-source",
        "# dir's subfolder names (performer / studio / category words).",
        f"# Source: {content_dir}",
        "# One term per line; '#' comments + blanks ignored. All terms are",
        "# word-boundary matched (|word) so common substrings don't false-fire.",
        "# Tokens are filtered against an English-stopword + release-cruft set,",
        "# but REVIEW before trusting: delete any dictionary words that slip",
        "# through (they'd flag innocent source comments). NEVER COMMIT — these",
        "# names are the sensitive content.",
        "",
    ]
    body = [f"{t}|word" for t in sorted(names, key=str.lower)]
    out_path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    print(f"--populate: wrote {len(body)} content tokens to {out_path.name}")
    print("  REVIEW the file + prune any dictionary words before relying on it.")
    print("  (gitignored — verify with: git check-ignore "
          f"{EXTRA_TERMS_FILE})")
    return 0


# Tokens that are release-encoding cruft or too generic to be content
# signal — dropped from the auto-extracted extension list. Not exhaustive;
# the populate output is meant to be reviewed, not blindly trusted.
_POPULATE_STOP = {
    # articles / glue
    "and", "the", "of", "to", "in", "on", "for", "with", "by", "at", "a",
    # release/codec cruft
    "xxx", "web", "dl", "webrip", "bdrip", "split", "scenes", "scene",
    "mp4", "mkv", "avi", "wmv", "hevc", "x265", "x264", "h264", "avc",
    "aac", "prt", "kleenex", "rarbg", "720p", "1080p", "480p", "540p",
    "2160p", "4k", "uhd", "hd", "sd", "cd1", "cd2", "pt1", "pt2", "vol",
    "part", "mv", "scr", "rip", "p", "v", "x", "remux", "internal",
    # very common english nouns/adjectives that appear in folder names AND
    # in innocent source comments — drop so they don't false-fire.
    "dream", "dreams", "true", "video", "videos", "mature", "blonde",
    "amateur", "compilation", "collection", "full", "extended", "edit",
    "new", "best", "mix", "set", "live", "show", "party", "music",
}


def _extract_content_tokens(content_dir: Path) -> set[str]:
    """Tokenize subfolder names into distinctive content words (performer /
    studio / category names), dropping stopwords, release cruft, pure
    numbers, and short/very-long tokens. Returns the surviving token set.

    Deliberately conservative on length (>=5) so 4-char common words like
    'anal'/'teen' that double as folder noise don't dominate; the operator
    can hand-add any short, distinctive term to the file afterwards."""
    toks: set[str] = set()
    for p in content_dir.rglob("*"):
        if not p.is_dir():
            continue
        for raw in re.split(r"[\s._\-\[\]()]+", p.name):
            t = raw.strip().lower()
            if (len(t) >= 5 and len(t) <= 40 and t.isalpha()
                    and t not in _POPULATE_STOP):
                toks.add(t)
    return toks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".", help="project root to scan")
    ap.add_argument("-o", "--out", default="_sanitize_audit_report.md")
    ap.add_argument("--no-extra", action="store_true",
                    help="ignore the local extension term-list")
    ap.add_argument("--populate", metavar="CONTENT_DIR",
                    help="(re)write the extension term-list from this "
                         "content-source dir's subfolder names, then exit")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.populate:
        return populate_extra_terms(Path(args.populate).resolve(), script_dir)

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    denylist = list(DENYLIST)
    extra = [] if args.no_extra else load_extra_terms(script_dir)
    denylist += extra
    patterns = compile_patterns(denylist)
    self_name = Path(__file__).name

    content_hits: dict[Path, list[tuple[int, str, str, str]]] = {}
    name_hits: list[tuple[Path, str, str]] = []
    artifacts: list[tuple[Path, int]] = []
    large_skipped: list[tuple[Path, int]] = []
    term_counts: dict[str, int] = {t: 0 for t, _, _ in denylist}
    files_scanned = 0

    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if not path.is_file():
            continue
        # Never scan the audit's own files (script, report, extension list).
        if path.name in (self_name, args.out, EXTRA_TERMS_FILE):
            continue

        rel = path.relative_to(root)
        ext = path.suffix.lower()

        for pat, term, cat in patterns:
            if pat.search(path.name):
                name_hits.append((rel, term, cat))
                term_counts[term] += 1

        if ext in ARTIFACT_EXTS:
            artifacts.append((rel, path.stat().st_size))
            continue
        if ext in BINARY_EXTS:
            continue

        size = path.stat().st_size
        if size > MAX_GREP_BYTES:
            large_skipped.append((rel, size))
            continue

        files_scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for lineno, line in enumerate(text.splitlines(), 1):
            for pat, term, cat in patterns:
                if pat.search(line):
                    trimmed = line.strip()[:160]
                    content_hits.setdefault(path, []).append(
                        (lineno, term, cat, trimmed))
                    term_counts[term] += 1

    write_report(root, Path(args.out), files_scanned, content_hits,
                 name_hits, artifacts, large_skipped, term_counts, denylist,
                 len(extra))
    print(f"scanned {files_scanned} text files under {root}")
    print(f"  denylist: {len(DENYLIST)} built-in + {len(extra)} extension terms")
    print(f"  {len(content_hits)} files with term hits")
    print(f"  {len(name_hits)} path/name hits")
    print(f"  {len(artifacts)} data/log artifacts flagged for exclusion review")
    print(f"report written to {args.out}")
    return 0


def write_report(root, out_path, files_scanned, content_hits, name_hits,
                  artifacts, large_skipped, term_counts, denylist,
                  n_extra) -> None:
    CAT_ORDER = ["domain", "site", "performer", "identity", "codename",
                 "path", "extra", "bank-label"]
    CAT_LABEL = {
        "domain": "Content-domain terms (urgent)",
        "site": "Site / studio names",
        "performer": "Performer names",
        "identity": "Personal / machine identifiers",
        "codename": "Internal codenames",
        "path": "Testbed library paths",
        "extra": "Local extension terms (testbed subfolder names)",
        "bank-label": "Bank labels (review — may be legit English)",
    }
    lines: list[str] = []
    a = lines.append

    total_content = sum(len(v) for v in content_hits.values())
    a("# Sanitization Audit Report")
    a("")
    a(f"- Root scanned: `{root}`")
    a(f"- Text files scanned: {files_scanned}")
    a(f"- Denylist: {len(denylist) - n_extra} built-in + {n_extra} "
      "local-extension terms")
    a(f"- Files with term hits: {len(content_hits)}  ({total_content} hits)")
    a(f"- Path/name hits: {len(name_hits)}")
    a(f"- Data/log artifacts to review for exclusion: {len(artifacts)}")
    a(f"- Large files not grepped: {len(large_skipped)}")
    a("")
    a("> Generated by `_sanitize_audit.py`. This report itself contains the "
      "flagged terms — do not ship it.")
    a("")

    a("## Hit count by term")
    a("")
    a("| Term | Category | Hits |")
    a("|---|---|---|")
    by_cat: dict[str, list[str]] = {}
    for term, cat, _ in denylist:
        by_cat.setdefault(cat, []).append(term)
    for cat in CAT_ORDER:
        for term in by_cat.get(cat, []):
            c = term_counts.get(term, 0)
            mark = "" if c == 0 else " **"
            end = "" if c == 0 else "**"
            # For extension terms, show the term only if it actually hit —
            # the list can be long and the report must not become a content
            # dump of every subfolder name.
            if cat == "extra" and c == 0:
                continue
            a(f"| `{term}` | {cat} |{mark}{c}{end} |")
    a("")

    a("## Path / filename hits")
    a("")
    if name_hits:
        a("Files whose path or name contains a flagged term — rename or remove.")
        a("")
        a("| Path | Term | Category |")
        a("|---|---|---|")
        for rel, term, cat in sorted(name_hits):
            a(f"| `{rel}` | `{term}` | {cat} |")
    else:
        a("_None._")
    a("")

    a("## Data / log artifacts — exclude wholesale")
    a("")
    a("Generated against the private library. These should NOT be copied into "
      "the public repo regardless of content (they hold real filenames).")
    a("")
    if artifacts:
        a("| Path | Size (KB) |")
        a("|---|---|")
        for rel, size in sorted(artifacts, key=lambda x: -x[1]):
            a(f"| `{rel}` | {size // 1024:,} |")
    else:
        a("_None._")
    a("")

    if large_skipped:
        a("## Large files not grepped")
        a("")
        a("Too large to term-scan — decide on exclusion manually.")
        a("")
        a("| Path | Size (KB) |")
        a("|---|---|")
        for rel, size in sorted(large_skipped, key=lambda x: -x[1]):
            a(f"| `{rel}` | {size // 1024:,} |")
        a("")

    a("## Content hits (term matches inside files)")
    a("")
    if not content_hits:
        a("_No term hits in scanned text files._")
        a("")
    else:
        for cat in CAT_ORDER:
            files_in_cat = []
            for path, hits in content_hits.items():
                if any(h[2] == cat for h in hits):
                    files_in_cat.append(path)
            if not files_in_cat:
                continue
            a(f"### {CAT_LABEL[cat]}")
            a("")
            for path in sorted(files_in_cat, key=lambda p: str(p)):
                rel = path.relative_to(root)
                cat_hits = [h for h in content_hits[path] if h[2] == cat]
                a(f"**`{rel}`** — {len(cat_hits)} hit(s)")
                a("")
                a("| Line | Term | Context |")
                a("|---|---|---|")
                for lineno, term, _, trimmed in cat_hits:
                    safe = trimmed.replace("|", "\\|").replace("`", "'")
                    a(f"| {lineno} | `{term}` | `{safe}` |")
                a("")

    a("## Suggested order of operations")
    a("")
    a("1. Start the public repo from a **fresh `git init`** — never filter the "
       "private history.")
    a("2. Exclude every path in *Data/log artifacts* and the session/handoff "
       "`.md` files.")
    a("3. Resolve *Content-domain* + *Site* + *Performer* + *Extension* hits.")
    a("4. Rename files flagged in *Path/filename hits*.")
    a("5. Replace *Personal/machine identifiers* (paths, usernames) with "
       "placeholders or env/config lookups.")
    a("6. Manually review *Bank label* hits — keep neutral English uses, "
       "neutralise baked-in category defaults.")
    a("7. Re-run with the local extension list until ALL categories are zero.")
    a("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
