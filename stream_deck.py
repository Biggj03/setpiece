"""
Elgato Stream Deck (Mk1, 15-key) controller for setpiece.

Five color-coded pages in Setpiece's "Pinned Peripheral" UX:
  MAIN     (purple) — transport: PLAY/PAUSE, FLIP, FLIP BACK, MARK IN/OUT,
                      and direct fire of decks 0..3.
  LIBRARY  (blue)   — paginated library-file launcher with thumbnails.
                      Tap = seek browse cursor + fire to LIVE.
  CLIPS    (green)  — saved clips for the CURRENT video (per-video model).
                      Tap = play_clip(idx).
  DECKS    (orange) — preview-deck slots 1..4 + crossfade indicator. Tap a
                      deck = fire_deck(N) (go live).
  VJ       (pink)   — performance toggles: audio-reactive, jog mode,
                      fullscreen, reset speed, BPM display, beat indicator.

Pinned perimeter (identical physical placement on every page):
  key 0  HOME           glows in current page color; cycles to next page
  key 2  page indicator page name + index, glows in page color
  key 4  PAGE NEXT      cycles forward (or in-page sub-pagination on
                        LIBRARY / CLIPS when more than one sub-page exists)
  key 10 BLACKOUT       app.pause()
  key 14 GO LIVE        fires the most recently assigned deck slot

Layout (Mk1 — 5x3, top-left = 0, row-major):
     0  1  2  3  4
     5  6  7  8  9
    10 11 12 13 14
Centre 10 keys (1, 5-9, 11-13) are page-specific dynamic content.

Threading model (strict — see CLAUDE.md "Stream Deck threading rules"):
  - main:          construction, registration, start()/stop()
  - render worker: THE ONLY thread that calls device.set_key_image /
                   set_brightness. Drains _render_queue and uses
                   `with deck:` to batch atomically.
  - watchdog:      polls deck.connected() every 1.5s; opens on first
                   detect, reopens after disconnect, queues a full
                   redraw post-reconnect.
  - state-poll:    every 1s polls clip count / library cursor / decks /
                   bpm / beat — re-queues a render only when a value
                   relevant to the current page changed.
  - HID callback:  deck.set_key_callback() runs on the USB HID poll
                   thread. NEVER calls set_key_image. Just records the
                   press timestamp and enqueues a render request.

Why the queue-not-callback rule matters (cost ~2 hours to learn):
  Pushing 233KB (15 keys x ~15KB JPEG) of pixel data from inside the HID
  callback starves the polling thread and deadlocks the device. Queue +
  worker + `with deck:` is the only safe pattern. Bare `except: pass` on
  set_key_image is also a footgun — it hides transport errors so dead
  handles get poked forever after a disconnect; we log and recycle.

Device caveat (this user's deck):
  Output reports are intermittently rejected at the driver level; even
  a clean open() can fail. We monkey-patch _reset_key_stream to no-op
  so open() at least succeeds when the device is in a partially-good
  state. Recovery is a USB unplug/replug. See
  memory/stream_deck_output_report_bug.md.

Hidapi backend: the LibUSB transport needs hidapi.dll next to the script
on Windows. If absent, start() returns success and the controller idles
silently — the watchdog will pick the device up the moment the dll is
dropped in (or, in practice, after the user fixes it and restarts).
"""

import colorsys
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Optional dependencies: import-guard so main.py never crashes if either is
# missing. start() will report a clean reason and idle.
try:
    from StreamDeck.DeviceManager import DeviceManager
    from StreamDeck.ImageHelpers import PILHelper
    _SD_AVAILABLE = True
    _SD_IMPORT_ERR: Optional[str] = None
except Exception as e:  # pragma: no cover - environment-dependent
    _SD_AVAILABLE = False
    _SD_IMPORT_ERR = str(e)

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont
    _PIL_AVAILABLE = True
    _PIL_IMPORT_ERR: Optional[str] = None
except Exception as e:  # pragma: no cover
    _PIL_AVAILABLE = False
    _PIL_IMPORT_ERR = str(e)


# ── Layout constants ────────────────────────────────────────────────────────

# Stream Deck Mk1 = 5x3 = 15 keys, 72x72 px native.
KEY_HOME       = 0
KEY_PAGE_IND   = 2
KEY_PAGE_NEXT  = 4
KEY_BLACKOUT   = 10
KEY_GO_LIVE    = 14
PERIMETER_KEYS = (KEY_HOME, KEY_PAGE_IND, KEY_PAGE_NEXT, KEY_BLACKOUT, KEY_GO_LIVE)

# Centre slots — left-to-right, top-to-bottom. 10 dynamic keys per page.
CENTRE_SLOTS = [k for k in range(15) if k not in PERIMETER_KEYS]

# Page order for the HOME-cycle (matches the docstring listing).
PAGE_ORDER = ["main", "library", "clips", "decks", "vj"]

# Each page has a signature hue. Used for HOME, PAGE NEXT and the page
# indicator badge so a glance tells you exactly where you are even mid-set.
PAGE_COLOR = {
    "main":    (123,  31, 162),  # #7B1FA2 purple
    "library": ( 25, 118, 210),  # #1976D2 blue
    "clips":   ( 56, 142,  60),  # #388E3C green
    "decks":   (245, 124,   0),  # #F57C00 orange
    "vj":      (194,  24,  91),  # #C2185B pink
}
PAGE_TITLE = {
    "main":    "MAIN",
    "library": "LIB",
    "clips":   "CLIP",
    "decks":   "DECK",
    "vj":      "VJ",
}

# Fixed accent colors used across pages
COL_BG_DARK    = ( 24,  24,  28)
COL_BG_EMPTY   = ( 12,  12,  16)
COL_RED        = (200,  40,  40)   # GO LIVE / blackout-armed
COL_AMBER      = (255, 170,  30)   # pause / armed-action
COL_GREEN_HOT  = ( 40, 200, 100)   # mark-in / play
COL_BLUE_HOT   = ( 50, 150, 255)   # flip / cool
COL_PINK_HOT   = (230,  60, 180)   # mark-out / vj
COL_GRAY       = ( 90,  90, 100)

# Where library-file thumbnails live (matches thumbnails.lib_thumbnail_path).
THUMBS_DIR = Path.home() / ".setpiece" / "thumbnails"

# Long-press SHIFT modifier — held this long means "fire" instead of
# "load to next preview deck" on the LIBRARY page. (Tap = fire too;
# long-press currently behaves the same — kept hookable for future use.)
LONG_PRESS_SECONDS = 0.5

# Number of preview decks. Matches AppState.decks length (4).
NUM_PREVIEW_DECKS = 4

# Beat indicator: how recent must last_beat_time be to count as "live"?
BEAT_FRESH_SECONDS = 0.18

# State poll period (seconds). Slow enough to be cheap, fast enough that
# the BPM display + beat indicator feel responsive.
STATE_POLL_SECONDS = 0.5

# Watchdog poll period (seconds).
WATCHDOG_POLL_SECONDS = 1.5

# Render coalesce window (seconds) — drain stacked requests before render.
COALESCE_WINDOW_SECONDS = 0.04


# ── Helpers (font, color, image) ────────────────────────────────────────────

_FONT_CACHE: dict = {}


def _load_font(size: int):
    """Try a real TTF; cache by size. Falls back to PIL default."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf",   # Segoe UI Bold (Windows)
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            if os.path.exists(path):
                f = ImageFont.truetype(path, size)
                _FONT_CACHE[size] = f
                return f
        except Exception:
            continue
    f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def _text_size(draw, text: str, font) -> tuple[int, int]:
    """Wrapper around textbbox that tolerates legacy PIL fallback shapes."""
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            return (len(text) * 7, 12)


def _multi_text_size(draw, text: str, font) -> tuple[int, int]:
    try:
        l, t, r, b = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=1)
        return (r - l, b - t)
    except Exception:
        return _text_size(draw, text.split("\n")[0], font)


def _brighten_rgb(rgb: tuple[int, int, int], amount: int = 50) -> tuple[int, int, int]:
    return tuple(min(255, c + amount) for c in rgb)


def _readable_fg(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Pick black or white text based on bg luminance."""
    r, g, b = bg
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (10, 10, 10) if lum > 160 else (255, 255, 255)


# ── Button descriptor ───────────────────────────────────────────────────────

class _Button:
    """A single rendered key.

    `kind` controls how the renderer composes the image:
      - "label":  solid bg color + multi-line text label
      - "thumb":  background image (PIL.Image) + optional badge + label
      - "empty":  flat dim placeholder

    `action` is a callable invoked on release (taking no args). None = inert.
    `sig`    is a hashable content signature used by the image cache so the
             same content does not re-rasterise across frames.
    """

    __slots__ = ("kind", "label", "bg", "fg", "img", "badge",
                 "action", "long_action", "sig")

    def __init__(self, kind: str = "empty", *,
                 label: str = "",
                 bg: tuple[int, int, int] = COL_BG_DARK,
                 fg: Optional[tuple[int, int, int]] = None,
                 img: Optional["Image.Image"] = None,
                 badge: str = "",
                 action: Optional[Callable[[], None]] = None,
                 long_action: Optional[Callable[[], None]] = None,
                 sig: Optional[tuple] = None):
        self.kind = kind
        self.label = label
        self.bg = bg
        self.fg = fg if fg is not None else _readable_fg(bg)
        self.img = img
        self.badge = badge
        self.action = action
        self.long_action = long_action
        # Default content signature — overridable for thumb keys whose
        # composition depends on more than label/bg.
        self.sig = sig if sig is not None else (kind, label, bg, badge)


# ── Controller ──────────────────────────────────────────────────────────────

class StreamDeckController:
    """Stream Deck Mk1 controller — five-page Pinned Peripheral UX.

    Lifecycle:
        sd = StreamDeckController(app)
        sd.start()         # safe even with no device plugged in
        ...
        sd.stop()          # blanks keys and joins threads

    The controller never blocks the app: every device interaction is
    funnelled through the render queue, every callback returns immediately,
    and missing methods on `app` are logged once and skipped.
    """

    def __init__(self, app):
        # Reference back to the main VJPracticeApp.
        self._app = app

        # Device + lifecycle
        self.deck = None
        self._device_lock = threading.Lock()
        self._want_running = False
        self._available_reason: Optional[str] = None

        # Page state
        self._current_page: str = self._restore_page()
        self._library_subpage: int = 0
        self._clips_subpage: int = 0

        # Round-robin assignment for "fire to next deck" on LIBRARY page,
        # plus tracking of the most recently assigned deck for GO LIVE.
        self._next_deck_slot: int = 0
        self._last_assigned_slot: int = 0

        # Cached render data so the state-poll thread can detect changes
        # without thrashing the queue. All access is single-threaded
        # (only the poll loop reads/writes).
        self._last_clip_count_for_file: int = -1
        self._last_clip_file: str = ""
        self._last_library_files_sig: tuple = ()
        self._last_library_selected: int = -2
        self._last_decks_sig: tuple = ()
        self._last_bpm: float = -1.0
        self._last_audio_state: bool = False
        self._last_jog_mode: str = ""
        self._last_pending_in: float = -2.0
        self._last_beat_lit: bool = False

        # Per-key press timestamps (long-press detection). One finger at a
        # time on a Stream Deck — single dict is fine.
        self._press_time_ns: dict[int, int] = {}

        # Currently mapped page contents — set by _build_page() and read
        # by the HID callback to dispatch a press. Lock-guarded because
        # the render worker rebuilds this list while the HID thread reads.
        self._page_buttons: list[Optional[_Button]] = [None] * 15
        self._page_lock = threading.Lock()

        # Render queue. Items:
        #   ("full",)             — repaint every key
        #   ("page", page_name)   — switch page then repaint everything
        #   ("flash", key, until) — temporary brighten on press
        # Drop-oldest under saturation: state, not history.
        self._render_queue: queue.Queue = queue.Queue(maxsize=128)

        # Threads
        self._render_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._state_poll_thread: Optional[threading.Thread] = None

        # Image cache: key (key_idx, content-signature) -> native bytes.
        # Keeps re-renders cheap; each key keeps at most one cached image.
        # A second dict (key_idx -> sig) lets us evict the prior entry
        # when the key's content changes.
        self._image_cache: dict[tuple, bytes] = {}
        self._cache_sig_for_key: dict[int, tuple] = {}
        self._cache_lock = threading.Lock()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> tuple[bool, str]:
        """Start all threads. Returns (ok, message).

        ok=True even if no device is plugged in or hidapi.dll is missing —
        the watchdog will pick the device up later. Returns ok=False only
        when the python deps themselves are unavailable (import error).
        """
        if not _SD_AVAILABLE:
            self._available_reason = f"streamdeck library missing: {_SD_IMPORT_ERR}"
            logger.warning(self._available_reason)
            return False, self._available_reason
        if not _PIL_AVAILABLE:
            self._available_reason = f"Pillow missing: {_PIL_IMPORT_ERR}"
            logger.warning(self._available_reason)
            return False, self._available_reason

        # Friendly hidapi.dll check on Windows. Watchdog retries forever
        # so dropping the dll in later still works without restart.
        if sys.platform == "win32":
            script_dir = Path(__file__).parent
            if not (script_dir / "hidapi.dll").exists():
                logger.warning(
                    "stream_deck: hidapi.dll not found next to script "
                    f"({script_dir}). Stream Deck will not enumerate until "
                    "you drop hidapi.dll there. (See CLAUDE.md.)"
                )

        self._want_running = True

        self._render_thread = threading.Thread(
            target=self._render_loop, name="streamdeck-render", daemon=True
        )
        self._render_thread.start()

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="streamdeck-watchdog", daemon=True
        )
        self._watchdog_thread.start()

        self._state_poll_thread = threading.Thread(
            target=self._state_poll_loop, name="streamdeck-statepoll", daemon=True
        )
        self._state_poll_thread.start()

        return True, "Stream Deck controller started"

    def stop(self):
        """Stop the controller; blank keys before close."""
        self._want_running = False
        with self._device_lock:
            d = self.deck
        if d is not None:
            try:
                with d:
                    d.reset()
                    d.close()
            except Exception as e:
                logger.debug(f"stream_deck close failed: {e}")
        self.deck = None

        for t in (self._render_thread, self._watchdog_thread, self._state_poll_thread):
            if t and t.is_alive():
                t.join(timeout=2.0)

    # ── Settings persistence ─────────────────────────────────────────────

    def _restore_page(self) -> str:
        """Pull last-used page from app.config (settings.json). Default main."""
        try:
            cfg = getattr(self._app, "config", None) or {}
            page = cfg.get("stream_deck_page", "main")
            if page in PAGE_ORDER:
                return page
        except Exception:
            pass
        return "main"

    def _persist_page(self):
        """Write current page back into settings.json. Best-effort."""
        try:
            self._app.config["stream_deck_page"] = self._current_page
            saver = getattr(self._app, "_save_config", None)
            if callable(saver):
                saver()
        except Exception as e:
            logger.debug(f"stream_deck: persist page failed: {e}")

    # ── Watchdog ─────────────────────────────────────────────────────────

    def _watchdog_loop(self):
        """Poll for the device every WATCHDOG_POLL_SECONDS. Opens on first
        detect; reopens after disconnect. Schedules a full redraw on every
        successful (re)open."""
        ticks = max(1, int(WATCHDOG_POLL_SECONDS / 0.1))
        while self._want_running:
            with self._device_lock:
                d = self.deck

            if d is None:
                if self._try_open():
                    self._enqueue_render(("full",))
            else:
                try:
                    alive = bool(d.connected())
                except Exception:
                    alive = False
                if not alive:
                    logger.warning("stream_deck: device disappeared")
                    self._close_device()

            for _ in range(ticks):
                if not self._want_running:
                    return
                time.sleep(0.1)

    def _try_open(self) -> bool:
        """Enumerate, open the first deck, register HID callback. Returns
        True iff a device is now open and ready."""
        try:
            decks = DeviceManager().enumerate()
        except Exception as e:
            # Most common: hidapi.dll missing. Already warned in start().
            logger.debug(f"stream_deck enumerate failed: {e}")
            return False
        if not decks:
            return False

        d = decks[0]
        # Monkey-patch _reset_key_stream to no-op for THIS instance only.
        # On this user's deck/firmware, the 1024-byte output report
        # _reset_key_stream sends throws "Failed to write out report (-1)".
        # Skipping it lets open() succeed; subsequent set_key_image writes
        # work fine when the device is in a good state. Diagnosed by raw
        # hidapi tests — feature reports OK, output reports intermittently
        # rejected during reset specifically. KEEP THIS PATCH.
        try:
            d._reset_key_stream = lambda: None
        except Exception:
            pass

        try:
            d.open()
            try:
                d.reset()
            except Exception as reset_err:
                logger.warning(f"stream_deck: reset failed (continuing): {reset_err}")
            d.set_brightness(70)
            d.set_key_callback(self._on_key_event)
        except Exception as e:
            logger.error(f"stream_deck open failed: {e}")
            try:
                d.close()
            except Exception:
                pass
            return False

        with self._device_lock:
            self.deck = d
        # Reset per-connection ephemeral state.
        self._press_time_ns.clear()
        with self._cache_lock:
            self._image_cache.clear()
            self._cache_sig_for_key.clear()
        # Force the next state-poll comparison to detect everything fresh.
        self._last_clip_count_for_file = -1
        self._last_library_files_sig = ()
        self._last_decks_sig = ()
        self._last_bpm = -1.0
        self._last_jog_mode = ""

        logger.info(
            f"stream_deck: opened {d.deck_type()} "
            f"({d.key_count()} keys, {d.key_image_format()['size']})"
        )
        return True

    def _close_device(self):
        with self._device_lock:
            d = self.deck
            self.deck = None
        if d is None:
            return
        try:
            with d:
                d.reset()
                d.close()
        except Exception as e:
            logger.debug(f"stream_deck close on disconnect failed: {e}")

    # ── HID callback (RUNS ON USB POLL THREAD — DO NOT TOUCH IMAGES) ────

    def _on_key_event(self, deck, key_idx: int, pressed: bool):
        """Called by the streamdeck library on every key press/release.

        Threading rule: we are on the USB HID polling thread. Calling
        set_key_image here would deadlock the device. Just record state
        and enqueue work; the render worker handles all drawing.
        """
        try:
            if pressed:
                self._press_time_ns[key_idx] = time.perf_counter_ns()
                # Quick visual ack — the render worker brightens the key.
                self._enqueue_render(("flash", key_idx,
                                      time.perf_counter_ns() + 110_000_000))
                return

            press_ns = self._press_time_ns.pop(key_idx, None)
            if press_ns is None:
                return
            held_s = (time.perf_counter_ns() - press_ns) / 1e9
            self._dispatch_press(key_idx, long_press=held_s >= LONG_PRESS_SECONDS)
        except Exception as e:
            logger.error(f"stream_deck key event handler crashed: {e}", exc_info=True)

    def _dispatch_press(self, key_idx: int, long_press: bool):
        """Map a key release to its action. Runs on HID poll thread —
        keep it cheap; never touch images here."""
        with self._page_lock:
            btn = self._page_buttons[key_idx] if 0 <= key_idx < 15 else None
        if btn is None:
            return
        action = btn.long_action if (long_press and btn.long_action) else btn.action
        if action is None:
            return
        try:
            action()
        except Exception as e:
            logger.error(f"stream_deck action error on key {key_idx}: {e}",
                         exc_info=True)

    # ── State polling ────────────────────────────────────────────────────

    def _state_poll_loop(self):
        """Re-queue a render when state relevant to the current page changes.

        Cheap: pulls a few attrs from app/state, computes signature tuples,
        compares to last seen. Only queues when something actually moved.
        """
        ticks = max(1, int(STATE_POLL_SECONDS / 0.1))
        while self._want_running:
            try:
                self._poll_once()
            except Exception as e:
                logger.debug(f"stream_deck state-poll error: {e}")
            for _ in range(ticks):
                if not self._want_running:
                    return
                time.sleep(0.1)

    def _poll_once(self):
        page = self._current_page
        state = getattr(self._app, "state", None)

        changed = False

        # Beat indicator: only matters on VJ page.
        if page == "vj" and state is not None:
            now = time.time()
            beat_lit = (state.last_beat_time
                        and (now - state.last_beat_time) <= BEAT_FRESH_SECONDS)
            if bool(beat_lit) != self._last_beat_lit:
                self._last_beat_lit = bool(beat_lit)
                changed = True
            new_bpm = float(getattr(state, "detected_bpm", 0.0) or 0.0)
            if abs(new_bpm - self._last_bpm) > 0.5:
                self._last_bpm = new_bpm
                changed = True
            new_audio = bool(getattr(state, "audio_reactive_enabled", False))
            if new_audio != self._last_audio_state:
                self._last_audio_state = new_audio
                changed = True
            new_jog = str(getattr(state, "jog_mode", ""))
            if new_jog != self._last_jog_mode:
                self._last_jog_mode = new_jog
                changed = True

        # MAIN: pending_in for the MARK IN LED.
        if page == "main" and state is not None:
            pi = getattr(state, "pending_in", None)
            piv = float(pi) if pi is not None else -1.0
            if abs(piv - self._last_pending_in) > 0.05:
                self._last_pending_in = piv
                changed = True

        # CLIPS: clip count for current file.
        if page == "clips":
            db = getattr(self._app, "clips_db", None)
            cur = self._current_video_path()
            if db is not None:
                try:
                    n = len(db.get_clips_for_file(cur)) if cur else 0
                except Exception:
                    n = 0
            else:
                n = 0
            if n != self._last_clip_count_for_file or cur != self._last_clip_file:
                self._last_clip_count_for_file = n
                self._last_clip_file = cur or ""
                # Snap subpage in if it would be empty
                pages = max(1, (n + 8 - 1) // 8)
                if self._clips_subpage >= pages:
                    self._clips_subpage = 0
                changed = True

        # LIBRARY: file list signature + cursor position.
        if page == "library" and state is not None:
            files = list(getattr(state, "library_files", []) or [])
            sig = tuple((f.get("hash") or f.get("name", "")) if isinstance(f, dict) else str(f)
                        for f in files)
            if sig != self._last_library_files_sig:
                self._last_library_files_sig = sig
                pages = max(1, (len(sig) + 8 - 1) // 8)
                if self._library_subpage >= pages:
                    self._library_subpage = 0
                changed = True
            sel = int(getattr(state, "library_selected_idx", -1))
            if sel != self._last_library_selected:
                self._last_library_selected = sel
                changed = True

        # DECKS: occupied slots (name only — the visual just shows label
        # + slot number; thumbnails are filmstrips and not used here).
        if page == "decks" and state is not None:
            decks = list(getattr(state, "decks", []) or [])
            sig = tuple((d.get("name") if isinstance(d, dict) else None) for d in decks)
            xfade = float(getattr(state, "crossfader_position", 0.0) or 0.0)
            full_sig = sig + (round(xfade, 2),)
            if full_sig != self._last_decks_sig:
                self._last_decks_sig = full_sig
                changed = True

        if changed:
            self._enqueue_render(("full",))

    def _current_video_path(self) -> str:
        player = getattr(self._app, "player", None)
        return getattr(player, "current_file", None) or ""

    # ── Render queue + worker ────────────────────────────────────────────

    def _enqueue_render(self, item: tuple):
        try:
            self._render_queue.put_nowait(item)
        except queue.Full:
            # State, not history. Drop oldest, keep newest.
            try:
                self._render_queue.get_nowait()
                self._render_queue.put_nowait(item)
            except Exception:
                pass

    def _render_loop(self):
        """Drain render requests and push images. THE ONLY thread that
        calls set_key_image / set_brightness / etc.

        Coalesces tightly-spaced requests by draining everything pending
        within COALESCE_WINDOW_SECONDS of the first dequeue, then doing
        one render. A flash overlay is short-lived and self-expires.
        """
        flash_until: dict[int, int] = {}
        while self._want_running:
            try:
                item = self._render_queue.get(timeout=0.2)
            except queue.Empty:
                # No new requests — but expire any active flash overlays.
                if flash_until:
                    now_ns = time.perf_counter_ns()
                    expired = [k for k, t in flash_until.items() if now_ns >= t]
                    if expired:
                        for k in expired:
                            del flash_until[k]
                        self._do_render(brighten_keys=set(flash_until.keys()))
                continue

            # Coalesce: drain anything piled up behind this one. Last
            # "page" item wins; "full" survives; any "flash" merges.
            items = [item]
            deadline = time.perf_counter() + COALESCE_WINDOW_SECONDS
            while time.perf_counter() < deadline:
                try:
                    items.append(self._render_queue.get_nowait())
                except queue.Empty:
                    break

            new_page: Optional[str] = None
            for it in items:
                if it[0] == "page":
                    new_page = it[1]
                elif it[0] == "flash":
                    flash_until[it[1]] = max(flash_until.get(it[1], 0), it[2])

            if new_page and new_page != self._current_page:
                self._current_page = new_page
                self._persist_page()
                # Page change invalidates the image cache (signatures shift).
                with self._cache_lock:
                    self._image_cache.clear()
                    self._cache_sig_for_key.clear()

            self._do_render(brighten_keys=set(flash_until.keys()))

    def _do_render(self, brighten_keys: Optional[set] = None):
        """Build the current page's button list, then push each key."""
        with self._device_lock:
            d = self.deck
        if d is None:
            return

        try:
            buttons = self._build_page(self._current_page)
        except Exception as e:
            logger.warning(f"stream_deck page build failed: {e}", exc_info=True)
            return

        # Publish so the HID dispatcher can resolve presses.
        with self._page_lock:
            self._page_buttons = buttons

        brighten_keys = brighten_keys or set()

        try:
            with d:
                size = d.key_image_format()["size"]
                for k in range(d.key_count()):
                    btn = buttons[k] if k < len(buttons) else None
                    if btn is None:
                        btn = _Button("empty")
                    bright = k in brighten_keys
                    sig = btn.sig + (("b",) if bright else ())
                    cached = self._cache_get(k, sig)
                    if cached is None:
                        img = self._render_button(d, btn, size, bright)
                        try:
                            cached = PILHelper.to_native_format(d, img)
                        except Exception as e:
                            logger.warning(f"stream_deck native conv failed key {k}: {e}")
                            continue
                        self._cache_put(k, sig, cached)
                    try:
                        d.set_key_image(k, cached)
                    except Exception as e:
                        # Transport hiccup. Don't tear down — watchdog will
                        # recycle if it's truly gone. Log and move on.
                        logger.warning(f"stream_deck set_key_image({k}) failed: {e}")
        except Exception as e:
            logger.warning(f"stream_deck render error: {e}", exc_info=True)

    def _cache_get(self, key_idx: int, sig: tuple) -> Optional[bytes]:
        with self._cache_lock:
            existing_sig = self._cache_sig_for_key.get(key_idx)
            if existing_sig != sig:
                # Different content for this slot — drop the stale entry.
                if existing_sig is not None:
                    self._image_cache.pop((key_idx, existing_sig), None)
                return None
            return self._image_cache.get((key_idx, sig))

    def _cache_put(self, key_idx: int, sig: tuple, native: bytes):
        with self._cache_lock:
            old = self._cache_sig_for_key.get(key_idx)
            if old is not None and old != sig:
                self._image_cache.pop((key_idx, old), None)
            self._cache_sig_for_key[key_idx] = sig
            self._image_cache[(key_idx, sig)] = native

    # ── Image building ───────────────────────────────────────────────────

    def _render_button(self, d, btn: _Button, size: tuple[int, int], bright: bool):
        """Compose a single key image from a _Button descriptor."""
        w, h = size

        if btn.kind == "empty":
            bg = COL_BG_EMPTY if not bright else _brighten_rgb(COL_BG_EMPTY, 80)
            return Image.new("RGB", (w, h), bg)

        if btn.kind == "thumb" and btn.img is not None:
            base = self._fit_cover(btn.img, (w, h))
            if bright:
                base = ImageEnhance.Brightness(base).enhance(1.4)
            draw = ImageDraw.Draw(base)
            if btn.label:
                self._draw_caption(draw, btn.label, w, h, btn.fg)
            if btn.badge:
                self._draw_badge(draw, btn.badge, w, h)
            return base

        # Fall through: kind == "label" (or "thumb" without an image).
        bg = btn.bg
        if bright:
            bg = _brighten_rgb(bg, 60)
        img = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(img)
        self._draw_label(draw, btn.label, w, h, btn.fg)
        if btn.badge:
            self._draw_badge(draw, btn.badge, w, h)
        return img

    def _draw_label(self, draw, text: str, w: int, h: int,
                    fg: tuple[int, int, int]):
        if not text:
            return
        # Pick a size that fits the widest line.
        lines = text.split("\n")
        size = 18 if len(lines) == 1 else 15
        while size >= 9:
            font = _load_font(size)
            widest = 0
            for ln in lines:
                tw, _ = _text_size(draw, ln, font)
                widest = max(widest, tw)
            if widest <= w - 6:
                break
            size -= 2
        font = _load_font(size)
        tw, th = _multi_text_size(draw, text, font)
        x = (w - tw) // 2
        y = (h - th) // 2 - 2
        try:
            draw.multiline_text((x, y), text, fill=fg, font=font,
                                align="center", spacing=1)
        except Exception:
            draw.text((x, y), lines[0], fill=fg, font=font)

    def _draw_caption(self, draw, text: str, w: int, h: int,
                      fg: tuple[int, int, int]):
        """Bottom strip caption over a thumbnail."""
        if not text:
            return
        font = _load_font(11)
        # Truncate to fit a single line
        s = text
        while s and _text_size(draw, s, font)[0] > w - 6:
            s = s[:-1]
        if not s:
            return
        tw, th = _text_size(draw, s, font)
        # Dark band behind the caption
        band_h = th + 4
        draw.rectangle([0, h - band_h, w, h], fill=(0, 0, 0))
        draw.text(((w - tw) // 2, h - band_h + 1), s,
                  fill=(255, 255, 255), font=font)

    def _draw_badge(self, draw, badge: str, w: int, h: int):
        """Slot-number badge in the top-left corner (matches iPad UI)."""
        font = _load_font(13)
        tw, th = _text_size(draw, badge, font)
        pad = 2
        x1, y1 = 0, 0
        x2, y2 = tw + pad * 3, th + pad * 3
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        draw.text((x1 + pad + 1, y1 + pad - 1), badge,
                  fill=(255, 255, 255), font=font)

    def _fit_cover(self, img: "Image.Image", target: tuple[int, int]) -> "Image.Image":
        """Resize to cover `target`, centre-cropped. Keeps aspect ratio."""
        tw, th = target
        sw, sh = img.size
        if sw == 0 or sh == 0:
            return Image.new("RGB", target, COL_BG_DARK)
        scale = max(tw / sw, th / sh)
        nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
        img = img.convert("RGB").resize((nw, nh), Image.LANCZOS)
        left = (nw - tw) // 2
        top = (nh - th) // 2
        return img.crop((left, top, left + tw, top + th))

    # ── Page builders ────────────────────────────────────────────────────

    def _build_page(self, page: str) -> list[_Button]:
        """Return a 15-element list of _Button for the given page."""
        buttons: list[_Button] = [_Button("empty") for _ in range(15)]

        # Pinned perimeter (identical placement on every page).
        page_color = PAGE_COLOR.get(page, COL_GRAY)
        title = PAGE_TITLE.get(page, page.upper())

        buttons[KEY_HOME] = _Button(
            "label", label="HOME", bg=page_color,
            action=self._action_cycle_page,
            sig=("home", page),
        )
        buttons[KEY_PAGE_IND] = _Button(
            "label", label=f"{title}\n{PAGE_ORDER.index(page) + 1}/{len(PAGE_ORDER)}",
            bg=page_color, action=self._action_cycle_page,
            sig=("ind", page),
        )
        buttons[KEY_PAGE_NEXT] = _Button(
            "label", label="PAGE>", bg=page_color,
            action=self._action_page_next,
            sig=("next", page, self._library_subpage if page == "library"
                 else self._clips_subpage if page == "clips" else 0),
        )
        buttons[KEY_BLACKOUT] = _Button(
            "label", label="BLACK", bg=COL_BG_DARK, fg=(220, 220, 220),
            action=self._action_blackout,
            sig=("black",),
        )
        buttons[KEY_GO_LIVE] = _Button(
            "label", label=f"LIVE\nD{self._last_assigned_slot + 1}",
            bg=COL_RED, action=self._action_go_live,
            sig=("live", self._last_assigned_slot),
        )

        # Page-specific centre.
        if page == "main":
            self._fill_main(buttons)
        elif page == "library":
            self._fill_library(buttons)
        elif page == "clips":
            self._fill_clips(buttons)
        elif page == "decks":
            self._fill_decks(buttons)
        elif page == "vj":
            self._fill_vj(buttons)
        # Unknown pages just leave the centre empty.

        return buttons

    # ── Page: MAIN ───────────────────────────────────────────────────────

    def _fill_main(self, buttons: list[_Button]):
        """Transport + 4 deck-fire keys."""
        state = getattr(self._app, "state", None)
        is_playing = bool(getattr(state, "is_playing", False)) if state else False
        play_label = "PAUSE" if is_playing else "PLAY"
        play_bg = COL_AMBER if is_playing else COL_GREEN_HOT

        pending = self._last_pending_in
        in_label = "MARK\nIN" if pending < 0 else f"IN\n{pending:.0f}s"
        in_bg = COL_GREEN_HOT if pending >= 0 else COL_BG_DARK

        # Top row centre (key 1, 3) — FLIP BACK / FLIP
        buttons[1] = _Button("label", label="<FLIP", bg=COL_BLUE_HOT,
                             action=self._call("flip_back"),
                             sig=("flipb",))
        buttons[3] = _Button("label", label="FLIP>", bg=COL_BLUE_HOT,
                             action=self._call("flip"),
                             sig=("flipf",))

        # Middle row (key 5-9) — MARK IN / PLAY-PAUSE / MARK OUT / fire D1, D2
        buttons[5] = _Button("label", label=in_label, bg=in_bg,
                             action=self._call("mark_in"),
                             sig=("min", round(pending, 1)))
        buttons[6] = _Button("label", label=play_label, bg=play_bg,
                             action=self._call("toggle_play"),
                             sig=("pp", is_playing))
        buttons[7] = _Button("label", label="MARK\nOUT", bg=COL_PINK_HOT,
                             action=self._call("mark_out"),
                             sig=("mout",))
        buttons[8] = self._deck_fire_button(0)
        buttons[9] = self._deck_fire_button(1)

        # Bottom row (key 11-13) — fire D3, D4, RESTART
        buttons[11] = self._deck_fire_button(2)
        buttons[12] = self._deck_fire_button(3)
        buttons[13] = _Button("label", label="RSTRT", bg=COL_BG_DARK,
                              action=self._call("restart_video"),
                              sig=("rstrt",))

    def _deck_fire_button(self, slot: int) -> _Button:
        """Compact fire-deck key labelled with the deck's clip name."""
        name = ""
        try:
            decks = getattr(self._app.state, "decks", []) or []
            if 0 <= slot < len(decks) and decks[slot]:
                name = (decks[slot].get("name") or "")[:8]
        except Exception:
            pass
        if name:
            label = f"D{slot + 1}\n{name}"
            bg = COL_GRAY
        else:
            label = f"D{slot + 1}\nempty"
            bg = COL_BG_DARK
        def fire():
            self._fire_deck(slot)
        return _Button("label", label=label, bg=bg, action=fire,
                       sig=("dfire", slot, name))

    # ── Page: LIBRARY ────────────────────────────────────────────────────

    def _fill_library(self, buttons: list[_Button]):
        """8 paginated thumbnail keys + sub-page indicator on PAGE NEXT."""
        state = getattr(self._app, "state", None)
        files = list(getattr(state, "library_files", []) or []) if state else []
        per_page = 8
        sub_pages = max(1, (len(files) + per_page - 1) // per_page)
        sub = self._library_subpage % sub_pages

        # Override PAGE NEXT to cycle library sub-pages instead of switching
        # top-level pages, but keep next-top-level on long-press.
        page_color = PAGE_COLOR["library"]
        buttons[KEY_PAGE_NEXT] = _Button(
            "label", label=f"SUB\n{sub + 1}/{sub_pages}",
            bg=page_color,
            action=self._action_library_subpage_next,
            long_action=self._action_page_next,
            sig=("libnext", sub, sub_pages),
        )

        start = sub * per_page
        for i, key in enumerate(CENTRE_SLOTS[:per_page]):
            file_idx = start + i
            if file_idx >= len(files):
                buttons[key] = _Button("empty",
                                       sig=("libempty", key))
                continue
            entry = files[file_idx]
            if isinstance(entry, dict):
                name = entry.get("name", "") or ""
                fhash = entry.get("hash", "") or ""
            else:
                name = str(entry)
                fhash = ""
            thumb = self._load_lib_thumb(fhash) if fhash else None
            short = name.rsplit(".", 1)[0][:14]
            badge = str(file_idx + 1)
            def make_action(idx=file_idx):
                def act():
                    self._library_fire(idx)
                return act
            sig = ("libfile", file_idx, fhash, bool(thumb))
            if thumb is not None:
                buttons[key] = _Button(
                    "thumb", label=short, img=thumb, badge=badge,
                    action=make_action(), sig=sig,
                )
            else:
                # No thumb on disk yet — colored placeholder with the name.
                hue_seed = sum(ord(c) for c in name) if name else file_idx
                hue = (hue_seed * 0.137) % 1.0
                r, g, b = colorsys.hsv_to_rgb(hue, 0.45, 0.55)
                bg = (int(r * 255), int(g * 255), int(b * 255))
                buttons[key] = _Button(
                    "label", label=short, bg=bg, badge=badge,
                    action=make_action(), sig=sig,
                )

        # Centre slots 8-9 (key 12, 13) overflow when per_page < len(CENTRE_SLOTS).
        # We use 8 slots; the remaining two (key 12, 13) are empty placeholders
        # so the user has visual breathing room next to BLACKOUT / GO LIVE.
        for key in CENTRE_SLOTS[per_page:]:
            buttons[key] = _Button("empty", sig=("libempty", key))

    def _load_lib_thumb(self, file_hash: str) -> Optional["Image.Image"]:
        """Read a library thumbnail JPEG; returns None on miss/error."""
        if not file_hash:
            return None
        path = THUMBS_DIR / f"lib_{file_hash}.jpg"
        if not path.exists():
            return None
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.debug(f"stream_deck lib thumb load failed for {file_hash}: {e}")
            return None

    def _library_fire(self, file_idx: int):
        """Tap a library key: select that file in app state, then fire it
        to LIVE via app.fire_selected_library_file()."""
        state = getattr(self._app, "state", None)
        if state is None:
            return
        try:
            state.set_library_selected_idx(file_idx)
        except Exception as e:
            logger.warning(f"stream_deck: set_library_selected_idx failed: {e}")
            return
        self._call("fire_selected_library_file")()

    # ── Page: CLIPS ──────────────────────────────────────────────────────

    def _fill_clips(self, buttons: list[_Button]):
        """8 paginated clips for the current video."""
        db = getattr(self._app, "clips_db", None)
        cur = self._current_video_path()
        clips = []
        if db is not None and cur:
            try:
                clips = list(db.get_clips_for_file(cur))
            except Exception:
                clips = []

        per_page = 8
        sub_pages = max(1, (len(clips) + per_page - 1) // per_page)
        sub = self._clips_subpage % sub_pages

        page_color = PAGE_COLOR["clips"]
        buttons[KEY_PAGE_NEXT] = _Button(
            "label", label=f"SUB\n{sub + 1}/{sub_pages}",
            bg=page_color,
            action=self._action_clips_subpage_next,
            long_action=self._action_page_next,
            sig=("clipnext", sub, sub_pages),
        )

        start = sub * per_page
        for i, key in enumerate(CENTRE_SLOTS[:per_page]):
            clip_idx = start + i
            if clip_idx >= len(clips):
                buttons[key] = _Button("empty", sig=("clipempty", key))
                continue
            clip = clips[clip_idx]
            cid = clip.get("id", "")
            name = (clip.get("name") or "clip")[:12]
            badge = str(clip_idx + 1)
            thumb = self._load_clip_thumb(cid)
            def make_action(idx=clip_idx):
                def act():
                    self._call_with("play_clip", idx)
                return act
            sig = ("clip", cid, bool(thumb), name)
            if thumb is not None:
                buttons[key] = _Button(
                    "thumb", label=name, img=thumb, badge=badge,
                    action=make_action(), sig=sig,
                )
            else:
                buttons[key] = _Button(
                    "label", label=name, bg=PAGE_COLOR["clips"],
                    badge=badge, action=make_action(), sig=sig,
                )

        for key in CENTRE_SLOTS[per_page:]:
            buttons[key] = _Button("empty", sig=("clipempty", key))

    def _load_clip_thumb(self, clip_id: str) -> Optional["Image.Image"]:
        if not clip_id:
            return None
        path = THUMBS_DIR / f"{clip_id}.jpg"
        if not path.exists():
            return None
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.debug(f"stream_deck clip thumb load failed for {clip_id}: {e}")
            return None

    # ── Page: DECKS ──────────────────────────────────────────────────────

    def _fill_decks(self, buttons: list[_Button]):
        """4 deck slots + crossfade visualisation."""
        state = getattr(self._app, "state", None)
        decks = list(getattr(state, "decks", []) or []) if state else []

        # Use the four corners of the centre block for slots 1..4.
        slot_keys = [1, 3, 11, 13]
        for slot, key in enumerate(slot_keys):
            entry = decks[slot] if slot < len(decks) else None
            name = ""
            if isinstance(entry, dict):
                name = (entry.get("name") or "")[:10]
            label = f"D{slot + 1}\n{name}" if name else f"D{slot + 1}\nempty"
            bg = PAGE_COLOR["decks"] if name else COL_BG_DARK
            def make_action(s=slot):
                def act():
                    self._fire_deck(s)
                return act
            buttons[key] = _Button(
                "label", label=label, bg=bg,
                action=make_action(),
                sig=("deck", slot, name),
            )

        # Crossfader bar — render in the middle column (5, 6, 7, 8, 9).
        xfade = 0.0
        try:
            xfade = float(getattr(state, "crossfader_position", 0.0) or 0.0)
        except Exception:
            xfade = 0.0
        # 5 keys span 0..1 — pick the one closest to xfade and brighten it.
        xfade_idx = max(0, min(4, int(round(xfade * 4))))
        xfade_keys = [5, 6, 7, 8, 9]
        for i, key in enumerate(xfade_keys):
            if i == xfade_idx:
                buttons[key] = _Button(
                    "label", label="X", bg=PAGE_COLOR["decks"],
                    fg=(0, 0, 0),
                    sig=("xfdot", i),
                )
            elif i == 0 or i == 4:
                lab = "L" if i == 0 else "P"
                buttons[key] = _Button(
                    "label", label=lab, bg=COL_BG_DARK, fg=COL_GRAY,
                    sig=("xfend", i),
                )
            else:
                buttons[key] = _Button(
                    "label", label="-", bg=COL_BG_DARK, fg=COL_GRAY,
                    sig=("xfdash", i),
                )

        # Centre slots 12 sits between deck D3 and D4 — leave it empty.
        # (Already populated above with deck slots; nothing else to do.)

    # ── Page: VJ ─────────────────────────────────────────────────────────

    def _fill_vj(self, buttons: list[_Button]):
        """Performance toggles + readouts."""
        state = getattr(self._app, "state", None)
        audio_on = bool(getattr(state, "audio_reactive_enabled", False)) if state else False
        bpm = float(getattr(state, "detected_bpm", 0.0) or 0.0) if state else 0.0
        jog = str(getattr(state, "jog_mode", "")) if state else ""

        # Audio reactive toggle
        if audio_on:
            ar_label = "AUDIO\nON"
            ar_bg = COL_GREEN_HOT
            ar_action = self._call("audio_reactive_stop")
        else:
            ar_label = "AUDIO\nOFF"
            ar_bg = COL_BG_DARK
            ar_action = self._call("audio_reactive_start")
        buttons[1] = _Button("label", label=ar_label, bg=ar_bg,
                             action=ar_action,
                             sig=("ar", audio_on))

        # Cycle jog sensitivity
        buttons[3] = _Button("label", label=f"JOG\n{(jog or '?')[:6]}",
                             bg=COL_BG_DARK,
                             action=self._action_cycle_jog,
                             sig=("jog", jog))

        # BPM display (read-only)
        bpm_text = "--" if bpm <= 0 else f"{bpm:.0f}"
        buttons[5] = _Button("label", label=f"BPM\n{bpm_text}",
                             bg=COL_BG_DARK,
                             sig=("bpm", round(bpm, 1)))

        # Beat indicator — bright pink when fresh, dim when stale.
        if self._last_beat_lit:
            beat_bg = COL_PINK_HOT
            beat_label = "* BEAT"
        else:
            beat_bg = (60, 30, 50)
            beat_label = "beat"
        buttons[6] = _Button("label", label=beat_label, bg=beat_bg,
                             sig=("beat", self._last_beat_lit))

        # Toggle fullscreen
        buttons[7] = _Button("label", label="FULL\nSCRN",
                             bg=COL_BG_DARK,
                             action=self._call("toggle_fullscreen"),
                             sig=("fs",))

        # Reset speed → 1.0x
        buttons[8] = _Button("label", label="SPEED\n1.0x",
                             bg=COL_BG_DARK,
                             action=self._call("reset_speed"),
                             sig=("rs",))

        # Manual flip (handy on this page during a set)
        buttons[9] = _Button("label", label="FLIP", bg=COL_BLUE_HOT,
                             action=self._call("flip"),
                             sig=("vjflip",))

        # Bottom row centre (11, 12, 13) — RESTART, MARK IN, MARK OUT
        buttons[11] = _Button("label", label="RSTRT", bg=COL_BG_DARK,
                              action=self._call("restart_video"),
                              sig=("vjrstrt",))
        buttons[12] = _Button("label", label="MARK\nIN", bg=COL_GREEN_HOT,
                              action=self._call("mark_in"),
                              sig=("vjin",))
        buttons[13] = _Button("label", label="MARK\nOUT", bg=COL_PINK_HOT,
                              action=self._call("mark_out"),
                              sig=("vjout",))

    # ── Action helpers ───────────────────────────────────────────────────

    def _call(self, method_name: str) -> Callable[[], None]:
        """Return a zero-arg callable that invokes app.<method_name>(). Logs
        once if the method is missing instead of crashing the dispatcher."""
        def fn():
            m = getattr(self._app, method_name, None)
            if m is None:
                logger.warning(f"stream_deck: app missing method '{method_name}'")
                return
            try:
                m()
            except Exception as e:
                logger.error(f"stream_deck: {method_name}() raised: {e}",
                             exc_info=True)
        return fn

    def _call_with(self, method_name: str, *args):
        """Like _call but for methods that take args."""
        m = getattr(self._app, method_name, None)
        if m is None:
            logger.warning(f"stream_deck: app missing method '{method_name}'")
            return
        try:
            m(*args)
        except Exception as e:
            logger.error(f"stream_deck: {method_name}({args}) raised: {e}",
                         exc_info=True)

    def _fire_deck(self, slot: int):
        """fire_deck(slot) AND remember it for GO LIVE."""
        if not (0 <= slot < NUM_PREVIEW_DECKS):
            return
        self._last_assigned_slot = slot
        self._call_with("fire_deck", slot)
        # Re-render so GO LIVE label updates.
        self._enqueue_render(("full",))

    # ── Action callbacks (bound to perimeter keys) ───────────────────────

    def _action_cycle_page(self):
        idx = PAGE_ORDER.index(self._current_page) if self._current_page in PAGE_ORDER else 0
        nxt = PAGE_ORDER[(idx + 1) % len(PAGE_ORDER)]
        self._enqueue_render(("page", nxt))

    def _action_page_next(self):
        # Same behaviour as HOME-cycle by default. Library / Clips override
        # this on PAGE NEXT to do sub-paging instead, but their long-press
        # falls back here so the top-level cycle is always reachable.
        self._action_cycle_page()

    def _action_library_subpage_next(self):
        state = getattr(self._app, "state", None)
        files = list(getattr(state, "library_files", []) or []) if state else []
        sub_pages = max(1, (len(files) + 8 - 1) // 8)
        self._library_subpage = (self._library_subpage + 1) % sub_pages
        self._enqueue_render(("full",))

    def _action_cycle_jog(self):
        """VJ page: bump the jog-sensitivity baseline (gentle/medium/coarse)."""
        self._call("cycle_jog_sensitivity")()
        # Re-render so the JOG label updates promptly.
        self._enqueue_render(("full",))

    def _action_clips_subpage_next(self):
        db = getattr(self._app, "clips_db", None)
        cur = self._current_video_path()
        clips = []
        if db is not None and cur:
            try:
                clips = list(db.get_clips_for_file(cur))
            except Exception:
                pass
        sub_pages = max(1, (len(clips) + 8 - 1) // 8)
        self._clips_subpage = (self._clips_subpage + 1) % sub_pages
        self._enqueue_render(("full",))

    def _action_blackout(self):
        self._call("pause")()

    def _action_go_live(self):
        self._call_with("fire_deck", self._last_assigned_slot)
