"""
Working-set hot folder.

Watches a single folder (default ~/Setpiece-Working/) at low frequency and
fires a callback whenever a video file appears or disappears. Use case:
drag-and-drop a handful of files into one place and have the app pre-
proxy + surface them for the night's set without manual library work.

No dependencies beyond the stdlib — polling at 1 Hz is way more than
enough for human drag-and-drop and trivially Windows-friendly.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path.home() / "Setpiece-Working"
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
POLL_INTERVAL_S = 1.0


class WorkingSetWatcher:
    """Single-folder polling watcher. Fires on_added(path) and
    on_removed(path) callbacks on the watcher thread."""

    def __init__(
        self,
        folder: Path = DEFAULT_DIR,
        on_added: Optional[Callable[[str], None]] = None,
        on_removed: Optional[Callable[[str], None]] = None,
    ):
        self._folder = folder
        self._folder.mkdir(parents=True, exist_ok=True)
        self.on_added = on_added
        self.on_removed = on_removed
        self._known: set[str] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="WorkingSetWatcher", daemon=True,
        )
        self._thread.start()
        logger.info(f"Working-set watcher: {self._folder}")

    @property
    def folder(self) -> Path:
        return self._folder

    def list_files(self) -> list[Path]:
        """Return current list of video files in the folder, sorted by mtime
        descending (newest first)."""
        try:
            files = [
                p for p in self._folder.iterdir()
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS
            ]
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return files
        except Exception as e:
            logger.debug(f"working-set list error: {e}")
            return []

    def stop(self):
        self._stop.set()

    # ── internal ───────────────────────────────────────────────────────

    def _loop(self):
        # Seed _known with what's already there so we don't fire on_added
        # for the folder's existing contents at startup.
        try:
            self._known = {str(p.resolve()) for p in self.list_files()}
        except Exception:
            pass
        while not self._stop.is_set():
            try:
                cur = {str(p.resolve()) for p in self.list_files()}
                added = cur - self._known
                removed = self._known - cur
                for path in sorted(added):
                    logger.info(f"Working-set ADD: {Path(path).name}")
                    if self.on_added:
                        try:
                            self.on_added(path)
                        except Exception as e:
                            logger.debug(f"on_added handler error: {e}")
                for path in sorted(removed):
                    logger.info(f"Working-set REMOVE: {Path(path).name}")
                    if self.on_removed:
                        try:
                            self.on_removed(path)
                        except Exception as e:
                            logger.debug(f"on_removed handler error: {e}")
                self._known = cur
            except Exception as e:
                logger.debug(f"Working-set poll error: {e}")
            self._stop.wait(POLL_INTERVAL_S)
