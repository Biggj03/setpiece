"""Generic tagger hang watchdog.

Runs forever, checking every CHECK_INTERVAL_S seconds whether the
supervised tagger has made log progress. If the tagger goes silent
past STALL_THRESHOLD_S it is treated as hung: the watchdog kills it,
identifies the file at the head of the remaining queue (the likely
culprit), sentinel-tags that file so the next run skips it, and
restarts the tagger.

This is the self-healing pattern that lets multi-hour, multi-thousand
file tagging runs complete unattended — one pathological file can never
wedge the whole run.

USAGE
-----
    python tagger_watchdog.py --tagger color_tagger.py \\
        --root /path/to/clips --tag-prefix color:

    # extra args after `--` are forwarded verbatim to the tagger
    python tagger_watchdog.py --tagger pose_tagger.py \\
        --root /path/to/clips --tag-prefix pose: -- --workers 4

ARGUMENTS
---------
    --tagger      tagger script to supervise (e.g. color_tagger.py)
    --root        library root passed to the tagger as --root
    --tag-prefix  the tag namespace the tagger writes (e.g. "color:").
                  Used to find the next un-done file and to sentinel-tag
                  a hanging file so the restarted run skips it.

Side effects:
    - Kills tagger PIDs (children of `python <tagger> ...`)
    - Inserts a `<prefix>_skip_decode_hang` sentinel into the tag DB
    - Spawns new tagger subprocesses

Stops:
    - Ctrl-C
    - When the running tagger writes a "DONE:" line (finished naturally)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import psutil


REPO_DIR = Path(__file__).parent
DB_PATH = Path.home() / ".setpiece" / "path_tags.db3"

CHECK_INTERVAL_S = 60          # how often to check
STALL_THRESHOLD_S = 12 * 60    # 12 min silent = stalled.
# A deliberately generous threshold. An earlier 4-min threshold was
# killing legitimately-slow 4K-HEVC files mid-analysis and CAUSING
# restart thrash. 12 min comfortably tolerates the slowest legit file
# while still catching a genuine catastrophic hang.
MAX_CONSECUTIVE_RESTARTS = 20  # safety: stop if we restart 20× in a row


def _log(restart_log: Path, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [watchdog] {msg}"
    # Windows consoles may be cp1252 — coerce non-ASCII to `?` rather
    # than crashing on UnicodeEncodeError.
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"))
    try:
        with open(restart_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _find_tagger_pids(script: str) -> list[int]:
    """PIDs of running python procs whose cmdline mentions the tagger."""
    pids = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            joined = " ".join(p.info.get("cmdline") or [])
            if script in joined:
                pids.append(p.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return pids


def _kill_pid(restart_log: Path, pid: int) -> None:
    try:
        p = psutil.Process(pid)
        p.kill()
        p.wait(timeout=5)
        _log(restart_log, f"killed PID {pid}")
    except Exception as e:
        _log(restart_log, f"kill PID {pid} failed: {e}")


def _log_mtime(log_path: Path) -> float:
    try:
        return log_path.stat().st_mtime
    except Exception:
        return 0.0


def _rotate_log(restart_log: Path, log_path: Path) -> None:
    """Move a completed prior run's log aside before this session starts.

    Without this, a stale 'DONE:' line from the last finished run makes
    the done-check fire on the very first check — the watchdog exits
    'cleanly' without ever starting the tagger and the backlog silently
    never gets processed."""
    try:
        if log_path.is_file() and log_path.stat().st_size > 0:
            prev = log_path.with_name(log_path.name + ".prev")
            try:
                if prev.exists():
                    prev.unlink()
            except Exception:
                pass
            log_path.replace(prev)
            _log(restart_log, f"rotated stale tagger log -> {prev.name}")
    except Exception as e:
        _log(restart_log, f"tagger-log rotate failed: {e}")


def _log_done(log_path: Path) -> bool:
    """True if the tagger wrote a DONE: line (completed naturally)."""
    try:
        with open(log_path, "rb") as f:
            try:
                f.seek(-4096, os.SEEK_END)
            except OSError:
                f.seek(0)
            tail = f.read().decode("utf-8", errors="replace")
        return "DONE:" in tail
    except Exception:
        return False


def _next_candidate_file(restart_log: Path, root: str,
                         tag_prefix: str) -> str | None:
    """Find the first file under `root` that has no tag in the tagger's
    namespace yet — the most likely file currently hanging the tagger."""
    try:
        c = sqlite3.connect(str(DB_PATH), timeout=30.0)
        c.execute("PRAGMA busy_timeout=30000")
        already = {
            r[0] for r in c.execute(
                "SELECT DISTINCT filepath FROM file_tags WHERE tag LIKE ?",
                (tag_prefix + "%",),
            ).fetchall()
        }
        rows = c.execute(
            "SELECT filepath FROM files WHERE filepath LIKE ? "
            "ORDER BY filepath",
            (root + "%",),
        ).fetchall()
        c.close()
        for (fp,) in rows:
            if fp not in already:
                return fp
        return None
    except Exception as e:
        _log(restart_log, f"candidate lookup failed: {e}")
        return None


def _mark_bad(restart_log: Path, filepath: str, tag_prefix: str) -> bool:
    """Sentinel-tag a hanging file so the restarted tagger skips it."""
    try:
        c = sqlite3.connect(str(DB_PATH), timeout=30.0)
        c.execute("PRAGMA busy_timeout=30000")
        c.execute(
            "INSERT OR IGNORE INTO file_tags(filepath, tag) VALUES (?, ?)",
            (filepath, tag_prefix + "_skip_decode_hang"),
        )
        c.commit()
        c.close()
        _log(restart_log, f"bad-marked: {filepath[-60:]}")
        return True
    except Exception as e:
        _log(restart_log, f"mark-bad failed: {e}")
        return False


def _start_tagger(restart_log: Path, cmd: list[str],
                  log_path: Path) -> int | None:
    """Launch the tagger as a detached subprocess, append-mode log.

    CREATE_NO_WINDOW (Windows only — a safe no-op elsewhere) keeps the
    restart silent so a stall-restart doesn't flash a console window
    during a live set."""
    try:
        f = open(log_path, "ab")
        creationflags = 0
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            cmd,
            stdout=f, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(REPO_DIR),
            creationflags=creationflags,
        )
        _log(restart_log, f"started tagger PID {proc.pid}")
        return proc.pid
    except Exception as e:
        _log(restart_log, f"start tagger failed: {e}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--tagger", required=True,
                    help="tagger script to supervise (e.g. color_tagger.py)")
    ap.add_argument("--root", required=True,
                    help="library root passed to the tagger")
    ap.add_argument("--tag-prefix", required=True,
                    help="tag namespace the tagger writes (e.g. 'color:')")
    ap.add_argument("extra", nargs="*",
                    help="extra args forwarded to the tagger (after --)")
    args = ap.parse_args()

    script = args.tagger
    logs_dir = REPO_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / (Path(script).stem + ".log")
    restart_log = logs_dir / "tagger_watchdog.log"
    cmd = [sys.executable, "-u", str(REPO_DIR / script),
           "--root", args.root] + list(args.extra)

    _log(restart_log, f"watchdog starting — supervising {script}")
    _rotate_log(restart_log, log_path)
    _log(restart_log,
         f"check_interval={CHECK_INTERVAL_S}s "
         f"stall_threshold={STALL_THRESHOLD_S}s "
         f"max_consecutive_restarts={MAX_CONSECUTIVE_RESTARTS}")

    consecutive_restarts = 0
    last_mtime = _log_mtime(log_path)
    last_progress_time = time.time()

    while True:
        try:
            time.sleep(CHECK_INTERVAL_S)

            if _log_done(log_path):
                _log(restart_log, "tagger DONE: line detected — exiting cleanly")
                return 0

            pids = _find_tagger_pids(script)
            mtime = _log_mtime(log_path)
            now = time.time()

            if mtime > last_mtime:
                last_mtime = mtime
                last_progress_time = now
                consecutive_restarts = 0
                continue

            silence_s = now - last_progress_time

            if not pids:
                _log(restart_log, "no tagger PID found — starting one")
                if _start_tagger(restart_log, cmd, log_path):
                    consecutive_restarts += 1
                continue

            if silence_s < STALL_THRESHOLD_S:
                continue

            # STALLED
            _log(restart_log,
                 f"STALL detected: log silent {silence_s:.0f}s "
                 f"(threshold {STALL_THRESHOLD_S}s)")

            # Identify the likely hanger BEFORE killing (DB access is
            # contention-free while the tagger is dead).
            hanger = _next_candidate_file(restart_log, args.root,
                                          args.tag_prefix)
            if not hanger:
                _log(restart_log,
                     "could not identify next candidate file — "
                     "restarting anyway")

            for pid in pids:
                _kill_pid(restart_log, pid)
            time.sleep(2)

            if hanger:
                _mark_bad(restart_log, hanger, args.tag_prefix)

            consecutive_restarts += 1
            if consecutive_restarts > MAX_CONSECUTIVE_RESTARTS:
                _log(restart_log,
                     f"safety stop: {consecutive_restarts} restarts in a "
                     "row, something is fundamentally wrong")
                return 1

            _start_tagger(restart_log, cmd, log_path)
            last_mtime = _log_mtime(log_path)
            last_progress_time = time.time()

        except KeyboardInterrupt:
            _log(restart_log, "Ctrl-C — exiting")
            return 0
        except Exception as e:
            _log(restart_log, f"main loop error: {e}")
            time.sleep(CHECK_INTERVAL_S)


if __name__ == "__main__":
    sys.exit(main())
