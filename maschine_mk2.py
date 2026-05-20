"""
Maschine MK2 HID trigger integration.

Goal: use the 16 velocity-sensitive pads as a bigger trigger bank (vs
the S2's 8 pads), without using the deprecated Maschine 2 software.

# Why this is harder than the S2
Native Instruments' background daemon (NIHA — `NIHardwareAgent`) holds
an EXCLUSIVE USB lock on the MK2 whenever it's running. Opening the
device while NIHA is alive will fail with a Windows "device in use" or
"access denied" style error. To use the MK2 with this app you have to:

    sc stop NIHardwareAgent
    # or kill it via Task Manager (NIHardwareAgent.exe)
    # or uninstall Maschine 2 entirely if you never use it.

If NIHA respawns, set it to Manual start in services.msc.

# What we read
The MK2 exposes a HID interface (Windows usage page 0xFF01 or similar).
USB IDs: VID 0x17CC, PID 0x1140 (community-confirmed for MK2). We open
that HID handle directly with cython-hidapi and read input reports.

Pad and button bytes live at fixed offsets inside the report. The
exact layout varies by firmware revision, so this module spends its
first life as a discovery rig: every distinct report is logged so we
can read the bytes off the log and pin down offsets for this specific
device. Once offsets are known, we parse pad velocities and fire
on_pad_press(idx, velocity) callbacks.

Internal pad index 0..15 → physical pad number per NI's encoding:
    row    = idx // 4
    col    = idx %  4
    pad_no = (3 - row) * 4 + col + 1  # 1..16, bottom-left = 1
"""

import logging
import threading
import time
from typing import Callable, Optional

try:
    import hid  # cython-hidapi
    _HID_AVAILABLE = True
    _HID_ERROR = None
except Exception as e:
    _HID_AVAILABLE = False
    _HID_ERROR = e
    hid = None  # type: ignore

logger = logging.getLogger(__name__)

VID = 0x17CC          # Native Instruments
PID_MK2 = 0x1140      # Maschine MK2 (community-confirmed)
# A few other observed PIDs people see for MK2 silicon revs — try all
# in order. If a different one matches we still open it.
PID_CANDIDATES = [0x1140, 0x1200, 0x1110]

# Pad pressure thresholds on the 0..4095 (12-bit) range used by report
# 0x20. Resting noise floor is typically <100; a deliberate strike is
# 500+. PRESS rising-edge triggers the callback; pressure must fall
# back below RELEASE before another press will fire.
PRESS_THRESHOLD = 250
RELEASE_THRESHOLD = 80
# Don't refire on the same pad inside this window — protects against
# the noise blips at the threshold boundary.
PAD_DEBOUNCE_MS = 60

# Report IDs (community spec from cabl + open-maschine):
REPORT_PAD_PRESSURE = 0x20  # 65 bytes: ID + 16 pairs of pad pressure
REPORT_BUTTONS_ENCS = 0x01  # 26 bytes: button mask + encoder values
REPORT_PAD_LEDS = 0x80      # 49 bytes: ID + 16 pads × 3 RGB
# Report 0x81: Group A-H (each has TWO RGB dies, so 6 bytes per group)
# + lower transport row monochrome LEDs.
# Verified against shaduzlabs/cabl + hansfbaier/open-maschine.
REPORT_GROUP_LEDS = 0x81    # 57 bytes: ID + 8 groups × 6 RGB + 8 transport mono
GROUP_LEDS_LEN = 57
# Report 0x82: monochrome utility buttons (Control, Step, Browse,
# Sampling, navigation, display row, transport-secondary, etc.) —
# 32 bytes including ID. Not used yet but reserved.
REPORT_MONO_LEDS = 0x82
MONO_LEDS_LEN = 32

# Convert a hardware pad LABEL (1..16, bottom-left = 1) to the firmware's
# LOGICAL pad index (0..15, top-left = 0). Same encoding both directions
# for input (pad strike index → label) and output (label → LED slot).
def label_to_logical(label: int) -> int:
    if not (1 <= label <= 16):
        return -1
    z = label - 1
    row = 3 - z // 4
    col = z % 4
    return row * 4 + col


def _physical_pad_num(idx: int) -> int:
    """Translate internal 0..15 → physical 1..16 (bottom-left = 1).
    Per the Maschine 2.4 protocol research."""
    if not (0 <= idx <= 15):
        return -1
    r = idx // 4
    c = idx % 4
    return (3 - r) * 4 + c + 1


class MaschineMK2:
    """Reads HID reports from the Maschine MK2 and fires pad/button events.

    First-life behaviour: log every distinct report so we can pin down
    the exact byte offsets for this specific firmware revision. Once
    `pad_offsets` is filled in (either via discovery or as a constant),
    pad velocities are parsed and on_pad_press / on_pad_release fire."""

    def __init__(
        self,
        on_pad_press: Optional[Callable[[int, int], None]] = None,
        on_pad_release: Optional[Callable[[int], None]] = None,
        on_button_press: Optional[Callable[[int], None]] = None,
        on_button_release: Optional[Callable[[int], None]] = None,
        on_encoder_delta: Optional[Callable[[int, int], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        discovery_mode: bool = True,
    ):
        self.on_pad_press = on_pad_press
        self.on_pad_release = on_pad_release
        # on_button_press fires with the BIT INDEX of the changed button
        # in report 0x01's mask (0..47, since the mask is up to 6 bytes).
        # The main app maps bit→action; this module just emits raw edges.
        self.on_button_press = on_button_press
        self.on_button_release = on_button_release
        # on_encoder_delta(encoder_idx, delta_signed_int) — discovered
        # 2026-05-17: idx 0 = master encoder (byte[8] low nibble).
        # Other encoders TBD when wired.
        self.on_encoder_delta = on_encoder_delta
        self.on_error = on_error
        self.discovery_mode = discovery_mode
        # Last seen button mask — used to compute rising/falling edges.
        self._last_button_mask: int = 0
        # Last seen master encoder counter (4-bit, byte[8] low nibble).
        self._last_master_enc: int = 0

        self._device = None
        self._product = ""
        self._opened_pid = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        # Serialises every WRITE to the shared HID handle. The pad-LED
        # report (set_pad_colors_by_label) and the OLED frame writer
        # (MK2Display._write_chunks) both grab this lock so a 49-byte LED
        # report can never land mid-way through an 8-chunk OLED frame.
        # (Audit fix M15.) The reader thread only ever READS, so it does
        # not need the lock. MK2Display is handed this same lock in start().
        self._dev_write_lock = threading.Lock()
        # Per-pad state (idx → {"down": bool, "last_press_ms": int})
        self._pad_state = [
            {"down": False, "last_press_ms": 0, "last_velocity": 0}
            for _ in range(16)
        ]
        # Pad velocity offsets inside the input report. To be confirmed
        # during discovery. Common community reports place 16 pads at
        # consecutive byte pairs (uint16 LE pressure) starting around
        # offset 1 of a particular report id, but firmware revs vary.
        # Set to None → discovery mode (just log reports).
        self.pad_offsets: Optional[list[int]] = None
        # Distinct reports we've seen (for discovery logging).
        self._seen_report_sigs: set[bytes] = set()

    # ── public API ─────────────────────────────────────────────────────

    def start(self) -> tuple[bool, str]:
        """Open the device + start the reader thread. Returns (ok, msg)."""
        if not _HID_AVAILABLE:
            return False, f"hidapi not available: {_HID_ERROR}"
        if self._running:
            return True, "already running"
        device, pid = self._open()
        if device is None:
            msg = (
                "Maschine MK2 not found OR locked by NIHardwareAgent. "
                "Stop the NIHA service (services.msc → "
                "NIHardwareAgent → Stop) and retry."
            )
            if self.on_error:
                try: self.on_error(msg)
                except Exception: pass
            return False, msg
        self._device = device
        self._opened_pid = pid
        try:
            self._device.set_nonblocking(True)
        except Exception:
            pass
        try:
            self._product = self._device.get_product_string() or "Maschine MK2"
        except Exception:
            self._product = "Maschine MK2"
        logger.info(
            "MK2 opened: %r (VID 0x%04x PID 0x%04x)",
            self._product, VID, pid,
        )
        # Initialise the OLED renderer with the SHARED device handle AND
        # the shared device-write lock, so writes from the LED path and
        # the OLED path serialise cleanly against each other.
        self._oled = None
        self._oled_status = None
        # Hold-until timestamp for temporary OLED overrides (bank-switch
        # flash, etc.). push_oled_status() respects this — it won't
        # overwrite the left framebuffer until the hold expires, so a
        # 3-second confirmation flash actually stays on screen instead
        # of being painted over by the next regular status refresh.
        self._oled_left_hold_until: float = 0.0
        try:
            from maschine_mk2_oled import MK2Display, StatusRenderer
            self._oled = MK2Display(self._device, write_lock=self._dev_write_lock)
            self._oled_status = StatusRenderer()
            logger.info("MK2 OLED renderer attached")
        except Exception as e:
            logger.warning(f"MK2 OLED unavailable (input still works): {e}")
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop, name="MaschineMK2-reader", daemon=True,
        )
        self._thread.start()
        return True, f"MK2 connected: {self._product}"

    def push_oled_status(self, live_filename: str = "", bpm: float = 0.0,
                          scratch_count: int = 0, pad_rgb: list = None,
                          active_bank: str = "A",
                          bank_name: str = "",
                          page_name: str = "",
                          last_vertical: str = "",
                          browse_mode: bool = False,
                          browse_items: list = None,
                          browse_cursor: int = 0,
                          browse_header: str = "",
                          verticals: list = None,
                          active_page_idx: int = 0) -> None:
        """Render + push a status frame to BOTH OLED screens. Safe to
        call from the Qt thread; the underlying MK2Display serialises
        writes against the LED path. No-op if OLED init failed.

        Respects ``_oled_left_hold_until`` for the LEFT framebuffer —
        if a flash override is currently active, the left screen
        isn't repainted (so the override persists for its hold
        window). The RIGHT screen always refreshes since pad LEDs
        are time-sensitive context.

        ``page_name`` + ``last_vertical`` (optional, default "") drive
        the new bottom strip on the LEFT OLED that shows the active
        MK2 vertical-page name + the most recently fired vertical.
        Both empty → renders the legacy layout unchanged."""
        if not self._oled or not self._oled_status:
            return
        import time as _time
        try:
            # Skip left repaint while a hold-flash is active.
            if _time.time() >= self._oled_left_hold_until:
                fb_left = self._oled_status.render_left(
                    live_filename=live_filename,
                    bpm=float(bpm or 0.0),
                    scratch_count=int(scratch_count or 0),
                    bank_letter=str(active_bank or "A"),
                    bank_name=str(bank_name or ""),
                    page_name=str(page_name or ""),
                    last_vertical=str(last_vertical or ""),
                )
                self._oled.set_framebuffer(0, fb_left)
            if browse_mode:
                fb_right = self._oled_status.render_right_browse(
                    items=list(browse_items or []),
                    cursor_idx=int(browse_cursor or 0),
                    folder_label=str(browse_header or ""),
                )
            else:
                fb_right = self._oled_status.render_right(
                    pad_states=list(pad_rgb or []),
                    active_bank=str(active_bank or "A"),
                    page_name=str(page_name or ""),
                    last_vertical=str(last_vertical or ""),
                    verticals=list(verticals or []),
                    active_page_idx=int(active_page_idx or 0),
                )
            self._oled.set_framebuffer(1, fb_right)
        except Exception as e:
            logger.debug(f"push_oled_status failed: {e}")

    def play_bank_load_sweep(self, theme_color: tuple,
                              on_done=None) -> None:
        """Column-fill sweep animation across the 4x4 pad grid in the
        bank's theme color. ~300ms total. Each column lights up left
        to right then the whole grid holds for a beat before the
        optional ``on_done`` callback runs (typically used to restore
        normal scratch pad colors after the animation).

        Runs on a daemon thread so it doesn't block the Qt thread.
        The callback fires on the animation thread — caller is
        responsible for marshaling back to Qt if needed."""
        if not self._device:
            if on_done:
                try:
                    on_done()
                except Exception:
                    pass
            return

        def _run():
            import time as _time
            # 4 frames, left-to-right column fill
            # Labels: bottom-left=1, top-right=16. Col 0 = labels 1,5,9,13.
            for cols_lit in (1, 2, 3, 4):
                colors = {}
                for col in range(cols_lit):
                    for row in range(4):
                        label = col + 1 + (row * 4)
                        colors[label] = theme_color
                self.set_pad_colors_by_label(colors)
                _time.sleep(0.05)
            # Hold the full-grid theme color for a beat so the flash
            # actually registers visually before scratch colors return.
            _time.sleep(0.12)
            if on_done:
                try:
                    on_done()
                except Exception as e:
                    logger.debug(f"sweep on_done failed: {e}")

        threading.Thread(target=_run,
                         name="MK2-bank-sweep", daemon=True).start()

    def push_oled_reclassify_flash(self, letter: str, name: str,
                                   count: int,
                                   hold_seconds: float = 1.5) -> None:
        """Flash on hold-GROUP-X reclassify gesture. Shows
        "RECLASSIFY -> X" header + clip name + correction count."""
        if not self._oled or not self._oled_status:
            return
        import time as _time
        try:
            fb = self._oled_status.render_reclassify_flash(
                letter=letter, name=name, count=count,
            )
            self._oled.set_framebuffer(0, fb)
            self._oled_left_hold_until = (
                _time.time() + float(hold_seconds))
        except Exception as e:
            logger.debug(f"push_oled_reclassify_flash failed: {e}")

    def push_oled_vote_flash(self, direction: str, score: int,
                             ups: int, downs: int,
                             clip_name: str = "",
                             hold_seconds: float = 1.2) -> None:
        """Flash a +1/-1 vote confirmation on the LEFT OLED for
        ``hold_seconds`` seconds. Called by ``mk2_vote_up`` /
        ``mk2_vote_down``. direction is '+' or '-'."""
        if not self._oled or not self._oled_status:
            return
        import time as _time
        try:
            fb = self._oled_status.render_vote_flash(
                direction=direction, score=score,
                ups=ups, downs=downs,
                clip_name=clip_name,
            )
            self._oled.set_framebuffer(0, fb)
            self._oled_left_hold_until = (
                _time.time() + float(hold_seconds))
        except Exception as e:
            logger.debug(f"push_oled_vote_flash failed: {e}")

    def push_oled_layer_flash(self, layer_name: str,
                              hold_seconds: float = 2.0) -> None:
        """Flash LAYER + name on the LEFT OLED for ``hold_seconds``
        seconds. Called by ``cycle_bank_layer`` whenever the user
        rotates between bank layers (default/positions/pov/mood).
        Hardware confirmation that the cycle registered."""
        if not self._oled or not self._oled_status:
            return
        import time as _time
        try:
            fb = self._oled_status.render_layer_flash(layer_name)
            self._oled.set_framebuffer(0, fb)
            self._oled_left_hold_until = (
                _time.time() + float(hold_seconds))
        except Exception as e:
            logger.debug(f"push_oled_layer_flash failed: {e}")

    def push_oled_clip_to_banks_flash(self, query: str, matches: int,
                                       distribution: dict,
                                       elapsed_ms: int = 0,
                                       hold_seconds: float = 2.5) -> None:
        """Flash on bank/clip_to_banks fire from iPad. Shows the
        query + per-letter distribution + match count on the LEFT
        OLED. Lets the operator confirm the routing from across the
        room without having to look at the tablet."""
        if not self._oled or not self._oled_status:
            return
        import time as _time
        try:
            fb = self._oled_status.render_clip_to_banks_flash(
                query=query, matches=matches,
                distribution=distribution, elapsed_ms=elapsed_ms,
            )
            self._oled.set_framebuffer(0, fb)
            self._oled_left_hold_until = (
                _time.time() + float(hold_seconds))
        except Exception as e:
            logger.debug(f"push_oled_clip_to_banks_flash failed: {e}")

    def push_oled_bank_flash(self, letter: str, name: str,
                              count: int, first_files: list = None,
                              hold_seconds: float = 3.0) -> None:
        """Flash a bank-switch confirmation on the LEFT OLED for
        ``hold_seconds``. Called by ``bank_load`` whenever the user
        taps a bank letter — gives hardware feedback that the switch
        registered. After the hold expires, the normal status display
        takes over again on the next refresh."""
        if not self._oled or not self._oled_status:
            return
        import time as _time
        try:
            fb = self._oled_status.render_bank_flash(
                letter=letter, name=name, count=count,
                first_files=first_files,
            )
            self._oled.set_framebuffer(0, fb)
            self._oled_left_hold_until = _time.time() + float(hold_seconds)
        except Exception as e:
            logger.debug(f"push_oled_bank_flash failed: {e}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.5)
        try:
            if self._device:
                # Blank pad LEDs on shutdown so the controller doesn't
                # sit there glowing after the app closes.
                try:
                    self.set_pad_colors_by_label({})
                except Exception:
                    pass
                # Blank OLEDs too.
                try:
                    if self._oled:
                        self._oled.blank_both()
                except Exception:
                    pass
                self._device.close()
        except Exception:
            pass
        self._device = None

    # Cached state for Report 0x81 so we can update groups + transport
    # independently and re-send the combined report. Initial values =
    # all off; main.py paints both as soon as devices are up.
    _group_colors_cache: dict = None  # type: ignore
    _transport_brightness_cache: dict = None  # type: ignore
    # Byte-index map for the lower transport row in Report 0x81 (after
    # the 8 groups × 6 bytes = bytes 1-48 used by groups; transport
    # starts at byte 49 / index 48 of the data payload).
    _TRANSPORT_LEDS = {
        "restart": 48,
        "left":    49,
        "right":   50,
        "grid":    51,
        "play":    52,
        "rec":     53,
        "erase":   54,
        "shift":   55,
    }

    def _send_report_81(self) -> bool:
        """Compose Report 0x81 from cached group + transport state and
        ship to device. Called whenever either updates."""
        if not self._device:
            return False
        if self._group_colors_cache is None:
            self._group_colors_cache = {}
        if self._transport_brightness_cache is None:
            self._transport_brightness_cache = {}
        data = bytearray(GROUP_LEDS_LEN)
        data[0] = REPORT_GROUP_LEDS
        for i, letter in enumerate("ABCDEFGH"):
            r, g, b = self._group_colors_cache.get(letter, (0, 0, 0))
            r = max(0, min(255, int(r)))
            g = max(0, min(255, int(g)))
            b = max(0, min(255, int(b)))
            off = 1 + i * 6
            data[off + 0] = r
            data[off + 1] = g
            data[off + 2] = b
            data[off + 3] = r
            data[off + 4] = g
            data[off + 5] = b
        # Transport row: monochrome brightness 0-255
        for name, idx in self._TRANSPORT_LEDS.items():
            v = self._transport_brightness_cache.get(name, 0)
            data[1 + idx] = max(0, min(255, int(v)))
        try:
            with self._dev_write_lock:
                self._device.write(bytes(data))
            return True
        except Exception as e:
            logger.debug(f"_send_report_81 write failed: {e}")
            return False

    def set_group_leds(self, colors_by_letter: dict) -> bool:
        """Light the 8 Group A-H buttons. ``colors_by_letter`` maps
        ``'A'``..``'H'`` to ``(r, g, b)`` tuples 0..255. Missing letters
        are blanked. Each group has TWO RGB dies internally; we set
        both to the same color for a solid look.

        Verified against shaduzlabs/cabl + hansfbaier/open-maschine."""
        if self._group_colors_cache is None:
            self._group_colors_cache = {}
        # Replace cache contents (caller passes the full state they
        # want — missing letters become off, matching the original API)
        self._group_colors_cache = dict(colors_by_letter or {})
        return self._send_report_81()

    def set_transport_leds(self, brightness_by_name: dict) -> bool:
        """Light the lower transport row LEDs. Mono brightness 0-255 per
        named button: restart / left / right / grid / play / rec /
        erase / shift. Names not in the dict KEEP their previous
        brightness (incremental update, unlike set_group_leds).

        Lets ``main.py`` adjust just PLAY (e.g. on play/pause) without
        having to re-state all 8 transport bytes."""
        if self._transport_brightness_cache is None:
            self._transport_brightness_cache = {}
        for name, v in (brightness_by_name or {}).items():
            if name in self._TRANSPORT_LEDS:
                self._transport_brightness_cache[name] = int(v)
        return self._send_report_81()

    # ── mono utility LEDs (report 0x82) ──────────────────────────────
    # Lights up the monochrome utility buttons: CONTROL, STEP, BROWSE,
    # SAMPLING, ALL, AUTO WR, the page arrows, master section, etc.
    # Per-byte layout is firmware-dependent — bytes 1..31 each control
    # one button's brightness (0..255). The exact byte→button mapping
    # is discovered empirically; see MK2_BUTTON_MAP.md for the table.
    _mono_brightness_cache: dict = None  # {offset: brightness}

    def set_mono_leds(self, brightness_by_offset: dict,
                       merge: bool = True) -> bool:
        """Write report 0x82 to set monochrome utility-button LEDs.

        Args:
            brightness_by_offset: {byte_offset_1to31: brightness_0_255}
            merge: if True (default), missing offsets keep their last
                   cached value. If False, missing offsets go to 0.

        Returns True on successful write. Safe to call rapidly — the
        device-write lock serialises against OLED + pad-LED writes."""
        if not self._device:
            return False
        if self._mono_brightness_cache is None:
            self._mono_brightness_cache = {}
        if merge:
            self._mono_brightness_cache.update(brightness_by_offset)
        else:
            self._mono_brightness_cache = dict(brightness_by_offset)
        data = bytearray(MONO_LEDS_LEN)
        data[0] = REPORT_MONO_LEDS
        for off, v in self._mono_brightness_cache.items():
            if 1 <= off < MONO_LEDS_LEN:
                data[off] = max(0, min(255, int(v)))
        try:
            with self._dev_write_lock:
                self._device.write(bytes(data))
            return True
        except Exception as e:
            logger.debug(f"set_mono_leds write failed: {e}")
            return False

    def mono_leds_all_on(self, brightness: int = 200) -> bool:
        """Discovery helper: light EVERY mono LED at given brightness.
        Use to identify which physical buttons have backlight LEDs.
        Pair with mono_leds_all_off() to reset."""
        all_on = {off: brightness for off in range(1, MONO_LEDS_LEN)}
        return self.set_mono_leds(all_on, merge=False)

    def mono_leds_all_off(self) -> bool:
        """Discovery helper: turn off every mono LED."""
        return self.set_mono_leds({}, merge=False)

    def set_pad_colors_by_label(self, label_colors: dict) -> bool:
        """Light the 16 pads. `label_colors` maps hardware label (1..16)
        → (r, g, b) tuple, 0..255 each. Missing labels are blanked.
        Returns True if the write succeeded.

        Pads are stored in the LED report in LOGICAL order (top-left
        = idx 0), so we translate hardware labels through label_to_logical."""
        if not self._device:
            return False
        data = bytearray(49)
        data[0] = REPORT_PAD_LEDS
        for label in range(1, 17):
            color = label_colors.get(label, (0, 0, 0))
            logical = label_to_logical(label)
            if logical < 0:
                continue
            off = 1 + logical * 3
            data[off]     = max(0, min(255, int(color[0])))
            data[off + 1] = max(0, min(255, int(color[1])))
            data[off + 2] = max(0, min(255, int(color[2])))
        try:
            # Serialise against OLED frame writes — see _dev_write_lock.
            with self._dev_write_lock:
                self._device.write(bytes(data))
            return True
        except Exception as e:
            logger.debug(f"set_pad_colors write failed: {e}")
            return False

    # ── internals ──────────────────────────────────────────────────────

    def _open(self) -> tuple:
        # Try each candidate PID. Returns (device, pid) on success, (None, None).
        for pid in PID_CANDIDATES:
            try:
                dev = hid.device()
                dev.open(VID, pid)
                return dev, pid
            except Exception as e:
                logger.debug(f"MK2 open VID=0x{VID:04x} PID=0x{pid:04x} failed: {e}")
                continue
        # Also try a generic enumerate scan in case the PID differs.
        try:
            for info in hid.enumerate(VID, 0):
                pid = int(info.get("product_id") or 0)
                prod = info.get("product_string") or ""
                if "maschine" in prod.lower() and "mk2" in prod.lower():
                    try:
                        dev = hid.device()
                        dev.open(VID, pid)
                        logger.info(f"MK2 found by enumerate: PID=0x{pid:04x} product={prod!r}")
                        return dev, pid
                    except Exception as e:
                        logger.debug(f"MK2 enumerate-open failed PID=0x{pid:04x}: {e}")
        except Exception as e:
            logger.debug(f"MK2 enumerate scan failed: {e}")
        return None, None

    def _reader_loop(self):
        while self._running:
            try:
                # 64 bytes is plenty for MK2 input reports; cython-hidapi
                # blocks for the timeout if non-blocking was unsuccessful.
                data = self._device.read(64, timeout_ms=80)
            except Exception as e:
                logger.warning(f"MK2 read error: {e}")
                time.sleep(0.05)
                continue
            if not data:
                continue
            try:
                self._handle_report(bytes(data))
            except Exception as e:
                logger.error(f"MK2 report handler error: {e}", exc_info=True)

    def _handle_report(self, raw: bytes):
        if not raw:
            return
        rid = raw[0]
        if rid == REPORT_PAD_PRESSURE:
            self._handle_pad_report(raw)
        elif rid == REPORT_BUTTONS_ENCS:
            self._handle_buttons_report(raw)

    def _handle_buttons_report(self, raw: bytes):
        """Report 0x01: 26 bytes. Bytes 1..6 form a 48-bit button mask
        (LSB first). Bytes 7+ carry encoder values (absolute counters
        that wrap 0..255; client diffs to get delta).

        Buttons + encoders are reported in the SAME report — a rotation
        of an encoder fires this handler with the same button mask but
        different bytes 7+. So we must NOT early-return on
        `mask == self._last_button_mask` — we still need to check the
        encoder bytes for changes.
        """
        # Build the mask out of however many bytes we have (be safe).
        mask = 0
        for i in range(min(6, len(raw) - 1)):
            mask |= raw[1 + i] << (i * 8)
        # ── Button diff (rising/falling edges) ──────────────────────
        if mask != self._last_button_mask:
            changed = mask ^ self._last_button_mask
            for bit in range(48):
                if not (changed & (1 << bit)):
                    continue
                now_pressed = bool(mask & (1 << bit))
                if now_pressed:
                    logger.info(f"MK2 button bit {bit} press")
                    if self.on_button_press:
                        try:
                            self.on_button_press(bit)
                        except Exception as e:
                            logger.error(f"on_button_press handler error: {e}")
                else:
                    logger.debug(f"MK2 button bit {bit} release")
                    if self.on_button_release:
                        try:
                            self.on_button_release(bit)
                        except Exception:
                            pass
            self._last_button_mask = mask
        # ── Encoder diff (bytes 7+) ────────────────────────────────
        # Discovered 2026-05-17 night via diagnostic logging:
        #   byte[8] low nibble = MASTER encoder (4-bit unsigned counter,
        #   wraps 0↔15, +1 per CW detent, -1 per CCW). Upper nibble was
        #   always 0 in observation — probably another encoder/state, TBD.
        # Other encoders (display strip 0-3) live at other byte positions;
        # discover same way (rotate them, watch the diagnostic before
        # this final parser landed).
        if len(raw) > 8:
            new_master = raw[8] & 0x0F
            old_master = getattr(self, "_last_master_enc", new_master)
            if new_master != old_master:
                # 4-bit wrap-aware signed delta.
                d = (new_master - old_master) & 0x0F
                if d >= 8:
                    d -= 16
                self._last_master_enc = new_master
                logger.debug(f"MK2 master encoder delta={d:+d} "
                             f"(byte[8]: {old_master} → {new_master})")
                if self.on_encoder_delta:
                    try:
                        # Encoder index 0 = master (reserved name).
                        # Future: byte position → encoder name map.
                        self.on_encoder_delta(0, d)
                    except Exception as e:
                        logger.debug(f"on_encoder_delta error: {e}")

    # Encoder callback: (encoder_idx, delta_signed_int) → caller decides
    # what each encoder does. idx 0 = master encoder. Set externally.
    on_encoder_delta = None
    _last_master_enc: int = 0

    def _handle_pad_report(self, raw: bytes):
        """Report 0x20: 16 pad pairs, each pair = (low_byte, high_byte).
        Within a pair: pad_index = high>>4 (logical 0..15),
        pressure = ((high & 0x0F) << 8) | low (12-bit, 0..4095).

        Logical indices map to physical labels via _physical_pad_num
        (bottom-left = 1)."""
        now_ms = int(time.time() * 1000)
        # Each report carries ALL 16 pads' current pressures, addressed
        # by the high byte's upper nibble (not by position in the array).
        for i in range(16):
            off = 1 + i * 2
            if off + 1 >= len(raw):
                break
            low = raw[off]
            high = raw[off + 1]
            pad_idx = (high & 0xF0) >> 4
            pressure = ((high & 0x0F) << 8) | low
            if not (0 <= pad_idx <= 15):
                continue
            state = self._pad_state[pad_idx]
            state["last_velocity"] = pressure
            if not state["down"]:
                if pressure >= PRESS_THRESHOLD and (now_ms - state["last_press_ms"]) >= PAD_DEBOUNCE_MS:
                    state["down"] = True
                    state["last_press_ms"] = now_ms
                    pad_no = _physical_pad_num(pad_idx)
                    logger.info(f"MK2 PAD {pad_no} press p={pressure}")
                    if self.on_pad_press:
                        try:
                            self.on_pad_press(pad_no, pressure)
                        except Exception as e:
                            logger.error(f"on_pad_press handler error: {e}")
            else:
                if pressure <= RELEASE_THRESHOLD:
                    state["down"] = False
                    pad_no = _physical_pad_num(pad_idx)
                    logger.debug(f"MK2 PAD {pad_no} release")
                    if self.on_pad_release:
                        try:
                            self.on_pad_release(pad_no)
                        except Exception:
                            pass
