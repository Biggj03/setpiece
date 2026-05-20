"""
Traktor Kontrol S2 MK1 HID controller with callback-based dispatch.

Adapted from Setpiece traktor_ctrl.py — same hardware parsing & threading model,
but instead of POSTing to an HTTP API, dispatches to Python callbacks
registered by the caller (main.py / AppState).

Thread model (strict — do not deviate):
  - reader thread:  parses USB HID input, enqueues actions/continuous events.
                    NEVER calls user callbacks; never calls device.write().
  - writer thread:  THE ONLY caller of device.write(). Fed by self._write_queue.
                    cython-hidapi has no internal write serialization.
  - action worker:  drains _action_queue; invokes user-registered callbacks
                    (both button presses AND continuous control updates).
                    Keeps slow callbacks (mpv API, file I/O) off the reader.
  - LED render:     20Hz loop; composes _led_buffer + flash overlays + state
                    LEDs; enqueues a single 63-byte write per dirty tick.
  - device watchdog: polls .connected() every 1.5s; reopens on disconnect;
                    fires on_connect/on_disconnect callbacks on edges.

Why this matters:
  cython-hidapi is NOT thread-safe on Windows. Multiple concurrent writes
  cause USB transfer failures. All writes funnel through ONE thread + queue.

  The previous MVP called continuous callbacks (e.g. set_volume → mpv) from
  the HID reader thread. Touching the jog wheel emits report_02 packets at
  ~50Hz, each of which re-fired the volume/crossfader callbacks (noise on
  the unmoved faders flipped LSBs). mpv calls under that thread blocked the
  reader, USB transfers backed up, and audio dropped. Fixed by routing all
  continuous events through _action_queue → action_worker_thread.
"""

import logging
import queue
import struct
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import hid
    _AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as e:
    _AVAILABLE = False
    _IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)

# ─── S2 USB / protocol constants ────────────────────────────────────────────

VENDOR_ID = 0x17CC
PRODUCT_ID = 0x1100

LED_REPORT = 0x80
LED_MAX = 0x1F
LED_DIM = 0x06
LED_LIVE = 0x14
LED_OFF = 0x00

# LED offsets within the 62-byte LED payload (from Setpiece traktor_ctrl.py)
LED_A_PEAK         = 0x01
LED_A_LOOP_IN      = 0x02
LED_A_PLAY         = 0x03
LED_A_SYNC         = 0x04
LED_A_LOOP_OUT     = 0x05
LED_A_RESET        = 0x06
LED_A_CUE          = 0x07
LED_A_SHIFT        = 0x08
LED_A_PAD4_GREEN   = 0x09
LED_A_PAD3_GREEN   = 0x0A
LED_A_PAD2_GREEN   = 0x0B
LED_A_PAD1_GREEN   = 0x0C
LED_A_PAD4_BLUE    = 0x0D
LED_A_PAD3_BLUE    = 0x0E
LED_A_PAD2_BLUE    = 0x0F
LED_A_PAD1_BLUE    = 0x10
LED_B_VU_START     = 0x11   # 0x11..0x14 (4 segments)
LED_A_VU_START     = 0x15   # 0x15..0x18 (4 segments)
LED_FX1_PARAM3     = 0x19
LED_FX1_PARAM2     = 0x1A
LED_FX1_PARAM1     = 0x1B
LED_FX1_FOCUS      = 0x1C
LED_B_PFL          = 0x1D
LED_B_LOADED       = 0x1E
LED_A_LOADED       = 0x1F
LED_A_PFL          = 0x20
LED_B_LOOP_OUT     = 0x21
LED_B_LOOP_IN      = 0x22
LED_B_PLAY         = 0x23
LED_B_SYNC         = 0x24
LED_B_PEAK         = 0x25
LED_B_RESET        = 0x26
LED_B_CUE          = 0x27
LED_B_SHIFT        = 0x28
LED_B_PAD4_GREEN   = 0x29
LED_B_PAD3_GREEN   = 0x2A
LED_B_PAD2_GREEN   = 0x2B
LED_B_PAD1_GREEN   = 0x2C
LED_B_PAD4_BLUE    = 0x2D
LED_B_PAD3_BLUE    = 0x2E
LED_B_PAD2_BLUE    = 0x2F
LED_B_PAD1_BLUE    = 0x30
LED_WARNING        = 0x33
LED_B_SAMPLES      = 0x34
LED_A_SAMPLES      = 0x35
LED_FX2_PARAM3     = 0x36
LED_FX2_PARAM2     = 0x37
LED_FX2_PARAM1     = 0x38
LED_FX2_FOCUS      = 0x39
LED_FX2_CH2        = 0x3A
LED_FX1_CH2        = 0x3B
LED_FX2_CH1        = 0x3C
LED_FX1_CH1        = 0x3D

LED_A_PADS_BLUE  = [LED_A_PAD1_BLUE, LED_A_PAD2_BLUE, LED_A_PAD3_BLUE, LED_A_PAD4_BLUE]
LED_B_PADS_BLUE  = [LED_B_PAD1_BLUE, LED_B_PAD2_BLUE, LED_B_PAD3_BLUE, LED_B_PAD4_BLUE]
LED_A_PADS_GREEN = [LED_A_PAD1_GREEN, LED_A_PAD2_GREEN, LED_A_PAD3_GREEN, LED_A_PAD4_GREEN]
LED_B_PADS_GREEN = [LED_B_PAD1_GREEN, LED_B_PAD2_GREEN, LED_B_PAD3_GREEN, LED_B_PAD4_GREEN]

# Public LED-name → offset map (for set_led_state)
LED_NAME_MAP: Dict[str, int] = {
    "a_peak": LED_A_PEAK, "a_loop_in": LED_A_LOOP_IN, "a_play": LED_A_PLAY,
    "a_sync": LED_A_SYNC, "a_loop_out": LED_A_LOOP_OUT, "a_reset": LED_A_RESET,
    "a_cue": LED_A_CUE, "a_shift": LED_A_SHIFT,
    "a_pad1_green": LED_A_PAD1_GREEN, "a_pad2_green": LED_A_PAD2_GREEN,
    "a_pad3_green": LED_A_PAD3_GREEN, "a_pad4_green": LED_A_PAD4_GREEN,
    "a_pad1_blue": LED_A_PAD1_BLUE, "a_pad2_blue": LED_A_PAD2_BLUE,
    "a_pad3_blue": LED_A_PAD3_BLUE, "a_pad4_blue": LED_A_PAD4_BLUE,
    "fx1_param1": LED_FX1_PARAM1, "fx1_param2": LED_FX1_PARAM2,
    "fx1_param3": LED_FX1_PARAM3, "fx1_focus": LED_FX1_FOCUS,
    "b_pfl": LED_B_PFL, "b_loaded": LED_B_LOADED, "a_loaded": LED_A_LOADED,
    "a_pfl": LED_A_PFL,
    "b_loop_out": LED_B_LOOP_OUT, "b_loop_in": LED_B_LOOP_IN,
    "b_play": LED_B_PLAY, "b_sync": LED_B_SYNC, "b_peak": LED_B_PEAK,
    "b_reset": LED_B_RESET, "b_cue": LED_B_CUE, "b_shift": LED_B_SHIFT,
    "b_pad1_green": LED_B_PAD1_GREEN, "b_pad2_green": LED_B_PAD2_GREEN,
    "b_pad3_green": LED_B_PAD3_GREEN, "b_pad4_green": LED_B_PAD4_GREEN,
    "b_pad1_blue": LED_B_PAD1_BLUE, "b_pad2_blue": LED_B_PAD2_BLUE,
    "b_pad3_blue": LED_B_PAD3_BLUE, "b_pad4_blue": LED_B_PAD4_BLUE,
    "warning": LED_WARNING, "b_samples": LED_B_SAMPLES, "a_samples": LED_A_SAMPLES,
    "fx2_param1": LED_FX2_PARAM1, "fx2_param2": LED_FX2_PARAM2,
    "fx2_param3": LED_FX2_PARAM3, "fx2_focus": LED_FX2_FOCUS,
    "fx2_ch2": LED_FX2_CH2, "fx1_ch2": LED_FX1_CH2,
    "fx2_ch1": LED_FX2_CH1, "fx1_ch1": LED_FX1_CH1,
}

# Mapping used by flash overlays (button-press feedback)
_LED_FOR_BUTTON: Dict[str, int] = {
    "a_play": LED_A_PLAY, "a_cue": LED_A_CUE, "a_sync": LED_A_SYNC,
    "a_loop_in": LED_A_LOOP_IN, "a_loop_out": LED_A_LOOP_OUT,
    "a_reset": LED_A_RESET, "a_pfl": LED_A_PFL,
    "a_load": LED_A_LOADED, "a_samples": LED_A_SAMPLES,
    "b_play": LED_B_PLAY, "b_cue": LED_B_CUE, "b_sync": LED_B_SYNC,
    "b_loop_in": LED_B_LOOP_IN, "b_loop_out": LED_B_LOOP_OUT,
    "b_reset": LED_B_RESET, "b_pfl": LED_B_PFL,
    "b_load": LED_B_LOADED, "b_samples": LED_B_SAMPLES,
    "fx1_focus": LED_FX1_FOCUS, "fx2_focus": LED_FX2_FOCUS,
    "fx1_param1_btn": LED_FX1_PARAM1, "fx1_param2_btn": LED_FX1_PARAM2,
    "fx1_param3_btn": LED_FX1_PARAM3,
    "fx2_param1_btn": LED_FX2_PARAM1, "fx2_param2_btn": LED_FX2_PARAM2,
    "fx2_param3_btn": LED_FX2_PARAM3,
    "fx1_ch1": LED_FX1_CH1, "fx1_ch2": LED_FX1_CH2,
    "fx2_ch1": LED_FX2_CH1, "fx2_ch2": LED_FX2_CH2,
    "a_pad1": LED_A_PAD1_BLUE, "a_pad2": LED_A_PAD2_BLUE,
    "a_pad3": LED_A_PAD3_BLUE, "a_pad4": LED_A_PAD4_BLUE,
    "b_pad1": LED_B_PAD1_BLUE, "b_pad2": LED_B_PAD2_BLUE,
    "b_pad3": LED_B_PAD3_BLUE, "b_pad4": LED_B_PAD4_BLUE,
}


# ─── HID report parsers ─────────────────────────────────────────────────────

def _parse_report_01(data) -> Optional[dict]:
    """Buttons + jog wheel positions.

    Jog format on S2 MK1 (empirically determined):
      byte 1 = jog A position (uint8, wraps at 256). Bytes 2-4 are a host
               timestamp counter that monotonically increases — NOT part of
               the jog position. Reading bytes 1-4 as <I gave a value
               dominated by the timestamp and broke direction.
      byte 5 = jog B position (uint8, wraps at 256). Bytes 6-8 same story.
    """
    if len(data) < 32 or data[0] != 0x01:
        return None
    jog_a = data[1]
    jog_b = data[5]
    b9, bA, bB, bC, bD, bE = data[9], data[10], data[11], data[12], data[13], data[14]
    return {
        'jog_a': jog_a, 'jog_b': jog_b,
        # Deck A main buttons (byte 0x0D)
        'a_shift': bool(bD & 0x80), 'a_sync': bool(bD & 0x40),
        'a_cue':   bool(bD & 0x20), 'a_play': bool(bD & 0x10),
        'a_pad1':  bool(bD & 0x08), 'a_pad2': bool(bD & 0x04),
        'a_pad3':  bool(bD & 0x02), 'a_pad4': bool(bD & 0x01),
        # Deck B main buttons (byte 0x0C)
        'b_shift': bool(bC & 0x80), 'b_sync': bool(bC & 0x40),
        'b_cue':   bool(bC & 0x20), 'b_play': bool(bC & 0x10),
        'b_pad1':  bool(bC & 0x08), 'b_pad2': bool(bC & 0x04),
        'b_pad3':  bool(bC & 0x02), 'b_pad4': bool(bC & 0x01),
        # Deck A secondary + FX1 (byte 0x09)
        'a_pfl':          bool(b9 & 0x80), 'a_loop_in':      bool(b9 & 0x40),
        'a_loop_out':     bool(b9 & 0x20), 'a_reset':        bool(b9 & 0x10),
        'fx1_focus':      bool(b9 & 0x08),
        'fx1_param1_btn': bool(b9 & 0x04),
        'fx1_param2_btn': bool(b9 & 0x02),
        'fx1_param3_btn': bool(b9 & 0x01),
        # FX2 + FX channel enables (byte 0x0A)
        'fx2_focus':      bool(bA & 0x80),
        'fx2_param1_btn': bool(bA & 0x40),
        'fx2_param2_btn': bool(bA & 0x20),
        'fx2_param3_btn': bool(bA & 0x10),
        'fx1_ch2':        bool(bA & 0x08), 'fx2_ch2': bool(bA & 0x04),
        'fx1_ch1':        bool(bA & 0x02), 'fx2_ch1': bool(bA & 0x01),
        # Deck B secondary + load/samples (byte 0x0B)
        'b_pfl':      bool(bB & 0x80), 'b_loop_in':  bool(bB & 0x40),
        'b_loop_out': bool(bB & 0x20), 'b_reset':    bool(bB & 0x10),
        'a_load':     bool(bB & 0x08), 'b_load':     bool(bB & 0x04),
        'a_samples':  bool(bB & 0x02), 'b_samples':  bool(bB & 0x01),
        # Encoder presses (byte 0x0E)
        'a_gain_enc_press':  bool(bE & 0x01),
        'a_left_enc_press':  bool(bE & 0x02),
        'a_right_enc_press': bool(bE & 0x04),
        'browse_enc_press':  bool(bE & 0x08),
        'b_gain_enc_press':  bool(bE & 0x10),
        'b_left_enc_press':  bool(bE & 0x20),
        'b_right_enc_press': bool(bE & 0x40),
    }


def _parse_report_02(data) -> Optional[dict]:
    """Continuous controls (16-bit faders/knobs) + 4-bit relative encoders + jog touch."""
    # Highest offset accessed is u16(0x31) which reads bytes 49 and 50 — need
    # at least 51 bytes. Real S2 sends 64-byte reports so this is purely a
    # safety guard for malformed/truncated input. Was `< 32` which let
    # short-but-not-tiny buffers slip through and crash with struct.error.
    if len(data) < 51 or data[0] != 0x02:
        return None

    def u16(off):
        return struct.unpack_from('<H', bytes(data), off)[0]

    return {
        'a_gain_enc_4bit':  data[1] & 0x0F,
        'a_left_enc_4bit':  (data[1] >> 4) & 0x0F,
        'a_right_enc_4bit': data[2] & 0x0F,
        'browse_enc_4bit':  (data[2] >> 4) & 0x0F,
        'b_gain_enc_4bit':  data[3] & 0x0F,
        'b_left_enc_4bit':  (data[3] >> 4) & 0x0F,
        'b_right_enc_4bit': data[4] & 0x0F,
        'a_jog_touch':   u16(0x0D),
        'a_tempo':       u16(0x0F),
        'a_eq_hi':       u16(0x11),
        'b_jog_touch':   u16(0x1D),
        'b_tempo':       u16(0x1F),
        'b_eq_hi':       u16(0x21),
        'b_eq_mid':      u16(0x23),
        'a_eq_mid':      u16(0x25),
        'a_eq_low':      u16(0x27),
        'b_eq_low':      u16(0x29),
        'a_volume':      u16(0x2B),
        'b_volume':      u16(0x2D),
        'crossfader':    u16(0x2F),
        'headphone_mix': u16(0x31),
    }


# Names of fields in report_02 that are 16-bit absolute faders/knobs (need >>9 to 7-bit)
_CONT_16BIT_NAMES = (
    'a_tempo', 'a_eq_hi', 'b_tempo', 'b_eq_hi', 'b_eq_mid', 'a_eq_mid',
    'a_eq_low', 'b_eq_low', 'a_volume', 'b_volume', 'crossfader',
    'headphone_mix',
)

_ENCODER_NAMES = (
    'a_gain_enc_4bit', 'a_left_enc_4bit', 'a_right_enc_4bit',
    'browse_enc_4bit', 'b_gain_enc_4bit', 'b_left_enc_4bit',
    'b_right_enc_4bit',
)

# All button-style names dispatched on rising edges (excludes shifts as modifiers)
_BUTTON_NAMES = (
    # Deck A
    'a_play', 'a_cue', 'a_sync', 'a_pfl', 'a_loop_in', 'a_loop_out',
    'a_reset', 'a_load', 'a_samples',
    'a_pad1', 'a_pad2', 'a_pad3', 'a_pad4',
    # Deck B
    'b_play', 'b_cue', 'b_sync', 'b_pfl', 'b_loop_in', 'b_loop_out',
    'b_reset', 'b_load', 'b_samples',
    'b_pad1', 'b_pad2', 'b_pad3', 'b_pad4',
    # FX
    'fx1_focus', 'fx1_param1_btn', 'fx1_param2_btn', 'fx1_param3_btn',
    'fx2_focus', 'fx2_param1_btn', 'fx2_param2_btn', 'fx2_param3_btn',
    'fx1_ch1', 'fx1_ch2', 'fx2_ch1', 'fx2_ch2',
    # Encoder presses
    'a_gain_enc_press', 'a_left_enc_press', 'a_right_enc_press',
    'browse_enc_press',
    'b_gain_enc_press', 'b_left_enc_press', 'b_right_enc_press',
)

# Threshold above which a jog_touch reading counts as "user is touching".
# Calibrated empirically in Setpiece; ~0x4000 of 0xFFFF works reliably.
_JOG_TOUCH_THRESHOLD = 0x0C40  # was 0x4000 — observed sensor maxes ~0x0E6E on this user's MK1; rest noise ~0x0BFD


# ─── Controller ─────────────────────────────────────────────────────────────

class S2Controller:
    """Traktor S2 MK1 HID controller with callback dispatch.

    Public API:
        start() / stop()
        on_action(name, callback)              -- button presses
        on_continuous(name, callback)          -- faders, knobs, jog velocity
        on_encoder(name, callback)             -- 4-bit relative encoders (delta)
        on_connect(callback) / on_disconnect(callback)
        flash_led(name, duration_ms)           -- transient press feedback
        set_led_state(name, brightness)        -- persistent state LED
        clear_led_state(name)
        set_pads_armed(deck, indices)          -- highlight loaded clip pads
        set_audio_reactive_indicator(active)
    """

    def __init__(self):
        self.device: Optional["hid.device"] = None
        self._want_running = False
        self.connected = False

        # Threads
        self._reader_thread: Optional[threading.Thread] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._action_worker_thread: Optional[threading.Thread] = None
        self._led_render_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._device_lock = threading.Lock()

        # Callbacks
        self._action_callbacks: Dict[str, Callable] = {}
        self._continuous_callbacks: Dict[str, Callable] = {}
        self._encoder_callbacks: Dict[str, Callable] = {}
        self._on_connect_cb: Optional[Callable[[], None]] = None
        self._on_disconnect_cb: Optional[Callable[[], None]] = None

        # Queues + LED buffers
        self._write_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=64)
        self._action_queue: "queue.Queue[dict]" = queue.Queue(maxsize=256)
        self._led_lock = threading.Lock()
        self._led_state = bytearray(62)         # caller-driven persistent state
        self._overlay_until: Dict[str, int] = {}  # button → expires_ns
        self._overlay_lock = threading.Lock()
        self._led_dirty = threading.Event()

        # Input state tracking
        self._last_buttons: Dict[str, bool] = {}
        self._last_continuous: Dict[str, int] = {}    # 7-bit values
        self._last_encoders: Dict[str, int] = {}      # 4-bit raw
        self._last_jog_pos: Dict[str, int] = {}       # 'a'/'b' → uint32
        self._last_jog_t_ns: Dict[str, int] = {}
        self._jog_touched: Dict[str, bool] = {"a": False, "b": False}
        self._reader_heartbeat_ns: int = 0
        # Per-fader (lo, hi) for raw-uint16 → 0-127 scaling. Populated
        # from feature report 0xD0 at open + auto-widened at runtime.
        # _handle_continuous writes it from the reader thread;
        # _save_fader_calibration iterates it from the Qt thread on stop().
        # Guard with _cal_lock so a snapshot during shutdown can't see a
        # half-updated dict (Audit fix H3).
        self._fader_calibration: Dict[str, tuple] = {}
        self._cal_lock = threading.Lock()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> Tuple[bool, str]:
        if not _AVAILABLE:
            return False, f"hidapi not available: {_IMPORT_ERROR}"

        self._want_running = True

        # Writer FIRST (must be ready for any LED write)
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="s2-writer", daemon=True
        )
        self._writer_thread.start()

        # Action worker
        self._action_worker_thread = threading.Thread(
            target=self._action_worker_loop, name="s2-actionwk", daemon=True
        )
        self._action_worker_thread.start()

        # LED renderer
        self._led_render_thread = threading.Thread(
            target=self._led_render_loop, name="s2-ledrender", daemon=True
        )
        self._led_render_thread.start()

        # Device watchdog (also opens device for the first time)
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="s2-watchdog", daemon=True
        )
        self._watchdog_thread.start()

        # Try synchronous open so caller knows status immediately
        opened = self._open_device()
        msg = "S2 connected" if opened else "Waiting for Traktor S2..."
        logger.info(msg)
        return True, msg

    def stop(self):
        self._want_running = False
        # Save calibration BEFORE closing — _close_device tears down the
        # state we'd need to recompute it.
        try:
            self._save_fader_calibration()
        except Exception:
            pass
        self._close_device()
        for t in (self._reader_thread, self._writer_thread,
                  self._action_worker_thread, self._led_render_thread,
                  self._watchdog_thread):
            if t and t.is_alive():
                t.join(timeout=2.0)

    # ── Public registration API ──────────────────────────────────────────

    def on_action(self, action_name: str, callback: Callable):
        """Register callback for a button press (rising edge).

        Callback signature: () -> None. SHIFT combos are dispatched to
        '<name>_shift' if registered, else fall back to base name.
        """
        self._action_callbacks[action_name] = callback

    def on_continuous(self, control_name: str, callback: Callable):
        """Register callback for a continuous control (fader / knob / jog velocity).

        Callback signature: (value: int) -> None.
        For 16-bit faders/knobs: value is 0..127 (7-bit, debounced).
        For jog velocity (a_jog_velocity / b_jog_velocity): signed int,
            positive = forward, magnitude ~= ticks per ~10ms.
        For jog touch (a_jog_touch / b_jog_touch): 0 or 1.
        """
        self._continuous_callbacks[control_name] = callback

    def on_encoder(self, encoder_name: str, callback: Callable):
        """Register callback for a 4-bit relative encoder.

        Callback signature: (delta: int) -> None.  Positive = clockwise.
        Names: a_gain_enc_4bit, a_left_enc_4bit, a_right_enc_4bit,
               browse_enc_4bit, b_gain_enc_4bit, b_left_enc_4bit,
               b_right_enc_4bit.
        """
        self._encoder_callbacks[encoder_name] = callback

    def on_connect(self, callback: Callable[[], None]):
        """Called (from watchdog thread) on every successful connect."""
        self._on_connect_cb = callback

    def on_disconnect(self, callback: Callable[[], None]):
        """Called (from watchdog thread) on every disconnect."""
        self._on_disconnect_cb = callback

    # ── Public LED API ───────────────────────────────────────────────────

    def flash_led(self, button_name: str, duration_ms: int = 120):
        """Transient bright flash for press feedback. Thread-safe."""
        if button_name not in _LED_FOR_BUTTON:
            return
        expires_ns = time.perf_counter_ns() + duration_ms * 1_000_000
        with self._overlay_lock:
            cur = self._overlay_until.get(button_name, 0)
            if expires_ns > cur:
                self._overlay_until[button_name] = expires_ns
        self._led_dirty.set()

    def set_led_state(self, led_name: str, brightness: int):
        """Set a persistent LED brightness (0..LED_MAX). Survives across renders."""
        off = LED_NAME_MAP.get(led_name)
        if off is None:
            logger.debug(f"set_led_state: unknown LED '{led_name}'")
            return
        b = max(0, min(LED_MAX, int(brightness)))
        with self._led_lock:
            self._led_state[off] = b
        self._led_dirty.set()

    def clear_led_state(self, led_name: str):
        self.set_led_state(led_name, 0)

    def set_pads_armed(self, deck: str, armed_indices: List[int]):
        """Light the BLUE pad LEDs for armed clip slots on a deck.

        deck: 'a' or 'b'
        armed_indices: list of 0..3 ints (per-deck pad indices) to light.
        """
        deck = deck.lower()
        if deck not in ("a", "b"):
            return
        pads = LED_A_PADS_BLUE if deck == "a" else LED_B_PADS_BLUE
        with self._led_lock:
            for i, off in enumerate(pads):
                self._led_state[off] = LED_LIVE if i in armed_indices else 0
        self._led_dirty.set()

    def set_audio_reactive_indicator(self, active: bool):
        """Light A_LOAD LED to indicate audio-reactive mode is on."""
        with self._led_lock:
            self._led_state[LED_A_LOADED] = LED_LIVE if active else LED_DIM
            self._led_state[LED_A_SAMPLES] = LED_DIM if active else LED_LIVE
        self._led_dirty.set()

    # ── Device open/close ────────────────────────────────────────────────

    def _open_device(self) -> bool:
        with self._device_lock:
            if self.device is not None:
                return True
            try:
                d = hid.device()
                d.open(VENDOR_ID, PRODUCT_ID)
                d.set_nonblocking(True)
                self.device = d
            except Exception as e:
                logger.debug(f"S2 open failed: {e}")
                self.device = None
                return False

        logger.info("S2 device opened")
        self.connected = True
        # Fetch per-device fader calibration (Mixxx pattern). Without this
        # the faders look 3-bit (only 8 distinct positions). With it, full
        # ~12-bit ADC range is recovered.
        self._fetch_fader_calibration()
        # Reset all input-state tracking so the first reports after reconnect
        # produce fresh edges (and a stale 32-bit jog pos doesn't generate a
        # wild velocity delta).
        self._last_jog_pos.clear()
        self._last_jog_t_ns.clear()
        self._last_buttons.clear()
        self._last_continuous.clear()
        self._last_encoders.clear()
        self._jog_touched = {"a": False, "b": False}
        # Boot animation + initial paint
        self._led_boot_animation()
        self._led_dirty.set()

        # Start reader thread
        if not self._reader_thread or not self._reader_thread.is_alive():
            self._reader_thread = threading.Thread(
                target=self._reader_loop, name="s2-reader", daemon=True
            )
            self._reader_thread.start()

        # Notify caller
        if self._on_connect_cb:
            try:
                self._on_connect_cb()
            except Exception as e:
                logger.error(f"on_connect callback failed: {e}")
        return True

    def _close_device(self):
        with self._device_lock:
            d = self.device
            self.device = None
        if not d:
            return
        # Best-effort: blank LEDs first
        try:
            d.write(bytes([LED_REPORT]) + bytes(62))
        except Exception:
            pass
        try:
            d.close()
        except Exception:
            pass

        was_connected = self.connected
        self.connected = False
        if was_connected and self._on_disconnect_cb:
            try:
                self._on_disconnect_cb()
            except Exception as e:
                logger.error(f"on_disconnect callback failed: {e}")

    # ── Watchdog ─────────────────────────────────────────────────────────

    def _watchdog_loop(self):
        """Polls device.connected() every 1.5s; reopens on disconnect.
        Also detects silent reader-thread death via heartbeat."""
        while self._want_running:
            time.sleep(1.5)
            if not self._want_running:
                break
            with self._device_lock:
                d = self.device
            if d is None:
                if self._open_device():
                    logger.info("S2 reconnected")
                continue

            # Device is open — check it's actually still connected
            try:
                still = bool(d.connected())  # cython-hidapi exposes this
            except Exception:
                # Older hidapi may lack .connected(); fall back to heartbeat
                still = True

            if not still:
                logger.warning("S2 disappeared (device.connected() == False)")
                self._close_device()
                continue

            # Heartbeat: reader updates _reader_heartbeat_ns each iteration.
            # If stale > 3s while device is supposedly open, recycle it.
            if self._reader_heartbeat_ns:
                stale_ns = time.perf_counter_ns() - self._reader_heartbeat_ns
                if stale_ns > 3_000_000_000:
                    logger.warning(
                        f"reader heartbeat stale {stale_ns/1e9:.1f}s — recycling device"
                    )
                    self._close_device()

    # ── Reader thread ────────────────────────────────────────────────────

    def _reader_loop(self):
        """Read HID reports and dispatch parsed events.

        Critical: do NOT call user callbacks here. Enqueue everything onto
        _action_queue so a slow callback can't starve USB reads.
        """
        while self._want_running:
            with self._device_lock:
                d = self.device
            if d is None:
                return

            self._reader_heartbeat_ns = time.perf_counter_ns()
            try:
                data = d.read(64)
            except OSError as e:
                logger.warning(f"reader OSError (device gone?): {e}")
                self._close_device()
                return
            except Exception as e:
                logger.warning(f"reader transient error: {e}")
                time.sleep(0.05)
                continue

            if not data:
                # Non-blocking read returned nothing; sleep briefly to yield
                time.sleep(0.002)
                continue

            try:
                rid = data[0]
                if rid == 0x01:
                    state = _parse_report_01(data)
                    if state:
                        self._handle_buttons(state)
                        self._handle_jogs(state)
                elif rid == 0x02:
                    state = _parse_report_02(data)
                    if state:
                        self._handle_jog_touch(state)
                        self._handle_continuous(state)
                        self._handle_encoders(state)
            except Exception as e:
                logger.warning(f"reader handler error: {e}", exc_info=True)
                continue

    # ── Button handling (with SHIFT modifier) ────────────────────────────

    def _handle_buttons(self, state: dict):
        shift_a = state.get("a_shift", False)
        shift_b = state.get("b_shift", False)
        self._last_buttons["a_shift"] = shift_a
        self._last_buttons["b_shift"] = shift_b

        for name in _BUTTON_NAMES:
            cur = bool(state.get(name, False))
            prev = self._last_buttons.get(name, False)
            if cur and not prev:
                # Determine which SHIFT applies
                if name.startswith("a_"):
                    held = shift_a
                elif name.startswith("b_"):
                    held = shift_b
                else:  # fx1_*, fx2_*, browse_* — either SHIFT works
                    held = shift_a or shift_b
                self._enqueue_button(name, held)
            self._last_buttons[name] = cur

        # SHIFT held may need to repaint the SHIFT LEDs
        self._led_dirty.set()

    def _enqueue_button(self, name: str, shift_held: bool):
        """Push a button-press action onto the worker queue. Flash LED inline
        (just sets an overlay flag — the writer is the only one touching USB)."""
        self._flash_button_led(name)
        item = {
            "type": "button",
            "name": name,
            "shift": shift_held,
            "ts_ns": time.perf_counter_ns(),
        }
        self._enqueue_action(item)

    def _flash_button_led(self, name: str, duration_ms: int = 120):
        if name not in _LED_FOR_BUTTON:
            return
        expires_ns = time.perf_counter_ns() + duration_ms * 1_000_000
        with self._overlay_lock:
            cur = self._overlay_until.get(name, 0)
            if expires_ns > cur:
                self._overlay_until[name] = expires_ns
        self._led_dirty.set()

    # ── Jog wheel handling ───────────────────────────────────────────────

    def _handle_jog_touch(self, state: dict):
        """Detect jog-touch edges from report_02 (uint16 force value)."""
        for deck in ("a", "b"):
            raw = state.get(f"{deck}_jog_touch")
            if raw is None:
                continue
            new_touched = raw > _JOG_TOUCH_THRESHOLD
            if new_touched != self._jog_touched[deck]:
                logger.debug(f"jog touch edge deck={deck} touched={new_touched}")
                self._jog_touched[deck] = new_touched
                self._enqueue_continuous(f"{deck}_jog_touch", 1 if new_touched else 0)

    def _handle_jogs(self, state: dict):
        """Compute angular velocity from 32-bit position deltas; enqueue as
        continuous events under 'a_jog_velocity' / 'b_jog_velocity'.

        Wraparound: jog_a / jog_b are uint32 that wrap at 0x100000000. A delta
        outside [-2^31, +2^31) means we wrapped — adjust accordingly.

        Velocity unit: ticks per ~10ms (signed int). Positive = forward.
        Stationary jogs send the same position repeatedly → delta=0 → no emit.
        """
        now_ns = time.perf_counter_ns()
        for deck in ("a", "b"):
            cur = state.get(f"jog_{deck}")
            if cur is None:
                continue
            prev = self._last_jog_pos.get(deck)
            self._last_jog_pos[deck] = cur
            if prev is None:
                self._last_jog_t_ns[deck] = now_ns
                continue
            # uint8 wraparound: delta in [-128, +127]
            delta = (cur - prev) & 0xFF
            if delta > 127:
                delta -= 256
            if delta == 0:
                continue
            prev_t = self._last_jog_t_ns.get(deck, now_ns)
            self._last_jog_t_ns[deck] = now_ns
            dt_ms = max(1, (now_ns - prev_t) // 1_000_000)
            # Velocity = ticks per 10ms. Single-tick at 10ms = 1.
            # Capped at ±500 (very fast spin = ~50 ticks/10ms).
            velocity = int(delta * 10 / dt_ms)
            velocity = max(-500, min(500, velocity))
            self._enqueue_continuous(f"{deck}_jog_velocity", velocity)

    # ── Continuous (faders/knobs) handling ───────────────────────────────

    def _scale_fader(self, name: str, v16: int) -> int:
        """Map raw 16-bit ADC reading → 7-bit (0-127) using per-device
        calibration if available. Falls back to >>9 when uncalibrated.
        Without calibration, S2 faders look 3-bit (0-7 only)."""
        cal = self._fader_calibration.get(name) if self._fader_calibration else None
        if cal:
            lo, hi = cal
            if hi > lo:
                v = (v16 - lo) * 127 // (hi - lo)
                return max(0, min(127, v))
        return v16 >> 9  # legacy fallback

    _CAL_FILE = Path.home() / ".setpiece" / "fader_calibration.json"

    def _fetch_fader_calibration(self) -> None:
        """Load persisted per-fader (lo, hi) calibration if available;
        otherwise seed with a typical 12-bit range. Live observation
        widens the range and we save back to disk on close."""
        DEFAULT_LO, DEFAULT_HI = 16, 4080
        loaded: Dict[str, tuple] = {}
        try:
            if self._CAL_FILE.exists():
                import json as _json
                raw = _json.loads(self._CAL_FILE.read_text(encoding="utf-8"))
                for k, v in raw.items():
                    if isinstance(v, list) and len(v) == 2:
                        loaded[k] = (int(v[0]), int(v[1]))
                logger.info(f"fader calibration loaded from disk: {len(loaded)} entries")
        except Exception as e:
            logger.info(f"fader calibration load failed (using defaults): {e}")
        fresh = {
            name: loaded.get(name, (DEFAULT_LO, DEFAULT_HI))
            for name in _CONT_16BIT_NAMES
            if name not in ("a_jog_touch", "b_jog_touch")
        }
        # Swap the reference under the lock so a reconnect can't race the
        # reader thread's auto-cal write (Audit fix H3).
        with self._cal_lock:
            self._fader_calibration = fresh
        if not loaded:
            logger.info(
                f"fader calibration seeded with default [{DEFAULT_LO}..{DEFAULT_HI}]; "
                "do a full sweep of each fader once for it to converge"
            )

    def _save_fader_calibration(self) -> None:
        """Persist learned fader ranges so they survive restarts. Best-effort."""
        try:
            import json as _json
            import os as _os
            self._CAL_FILE.parent.mkdir(parents=True, exist_ok=True)
            # Snapshot under the lock so the reader thread can't mutate a
            # tuple mid-iteration (Audit fix H3).
            with self._cal_lock:
                snapshot = dict(self._fader_calibration)
            payload = {k: [lo, hi] for k, (lo, hi) in snapshot.items()}
            # Atomic write: tmp + os.replace so a crash mid-write can't
            # leave a truncated calibration file.
            tmp = self._CAL_FILE.with_suffix(".json.tmp")
            tmp.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
            _os.replace(str(tmp), str(self._CAL_FILE))
            logger.info(f"fader calibration saved ({len(payload)} entries)")
        except Exception as e:
            logger.debug(f"fader calibration save failed: {e}")

    def _handle_continuous(self, state: dict):
        """Threshold-debounced 7-bit deltas for absolute faders/knobs.
        Routes through the action queue — never calls callbacks inline."""
        for name in _CONT_16BIT_NAMES:
            v16 = state.get(name)
            if v16 is None:
                continue
            # Online auto-cal: widen the observed range every report so
            # the next _scale_fader call can map it correctly. Mutate
            # under _cal_lock so _save_fader_calibration's snapshot is
            # never torn (Audit fix H3).
            cal = self._fader_calibration.get(name)
            if cal is not None:
                lo, hi = cal
                if v16 < lo: lo = v16
                if v16 > hi: hi = v16
                if (lo, hi) != cal:
                    with self._cal_lock:
                        self._fader_calibration[name] = (lo, hi)
            v7 = self._scale_fader(name, v16)
            prev = self._last_continuous.get(name)
            if prev is None or v7 != prev:
                self._last_continuous[name] = v7
                self._enqueue_continuous(name, v7)

    def _enqueue_continuous(self, name: str, value: int):
        """Push a continuous-control update onto the action worker queue."""
        # Skip the enqueue if no callback registered — saves queue churn for
        # high-rate sources like jog_velocity / unmapped faders. The worker
        # would just no-op anyway, but jog can fire at ~50Hz and we don't want
        # to fill the queue with noise that delays real button presses.
        if name not in self._continuous_callbacks:
            return
        item = {
            "type": "continuous",
            "name": name,
            "value": value,
            "ts_ns": time.perf_counter_ns(),
        }
        self._enqueue_action(item)

    # ── Encoder (4-bit relative) handling ────────────────────────────────

    def _handle_encoders(self, state: dict):
        for name in _ENCODER_NAMES:
            cur = state.get(name)
            if cur is None:
                continue
            prev = self._last_encoders.get(name)
            self._last_encoders[name] = cur
            if prev is None:
                continue
            delta = (cur - prev) & 0x0F
            if delta == 0:
                continue
            if delta > 8:
                delta -= 16
            if name not in self._encoder_callbacks:
                continue
            item = {
                "type": "encoder",
                "name": name,
                "delta": delta,
                "ts_ns": time.perf_counter_ns(),
            }
            self._enqueue_action(item)

    # ── Action queue ─────────────────────────────────────────────────────

    def _enqueue_action(self, item: dict):
        try:
            self._action_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        # Queue full. Make room — but NEVER sacrifice a button press for a
        # continuous/encoder update (Audit fix M18). A frantic crossfader +
        # jog can flood the 256-deep queue; without this, a real button
        # press could be the "oldest" item and get evicted by fader noise.
        try:
            oldest = self._action_queue.get_nowait()
        except queue.Empty:
            # Raced empty between the Full and the get — just retry once.
            try:
                self._action_queue.put_nowait(item)
            except Exception:
                pass
            return
        new_is_button = item.get("type") == "button"
        old_is_button = oldest.get("type") == "button"
        if old_is_button and not new_is_button:
            # The oldest item is a button and the new one is just a
            # continuous/encoder update — drop the NEW one, keep the
            # button (re-enqueue it).
            try:
                self._action_queue.put_nowait(oldest)
            except Exception:
                pass
            logger.warning(
                f"action queue full — dropped continuous {item.get('name')} "
                f"to keep button {oldest.get('name')}"
            )
            return
        # oldest is droppable (continuous/encoder) OR both are buttons —
        # drop the oldest, enqueue the newest.
        logger.warning(
            f"action queue full — dropped {oldest.get('type')} {oldest.get('name')}"
        )
        try:
            self._action_queue.put_nowait(item)
        except Exception:
            pass

    def _action_worker_loop(self):
        """Drains action queue, calls user callbacks. Single thread → callbacks
        never run concurrently, simplifying caller code (no extra locks needed)."""
        while self._want_running:
            try:
                item = self._action_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            kind = item.get("type")
            name = item.get("name", "")
            try:
                if kind == "button":
                    shift_held = item.get("shift", False)
                    cb = None
                    if shift_held:
                        cb = self._action_callbacks.get(f"{name}_shift")
                    if cb is None:
                        cb = self._action_callbacks.get(name)
                    if cb is not None:
                        cb()
                        logger.info(f"button: {name}{' [SHIFT]' if shift_held else ''}")
                    else:
                        logger.debug(f"unmapped button: {name}")
                elif kind == "continuous":
                    cb = self._continuous_callbacks.get(name)
                    if cb is not None:
                        cb(item["value"])
                elif kind == "encoder":
                    cb = self._encoder_callbacks.get(name)
                    if cb is not None:
                        cb(item["delta"])
            except Exception as e:
                logger.error(f"callback error ({kind}:{name}): {e}", exc_info=True)

    # ── LED render thread ────────────────────────────────────────────────

    def _led_render_loop(self):
        """20Hz: composes _led_state + flash overlays + SHIFT, pushes a single
        write to _write_queue. Skips ticks that don't change anything (dirty flag)."""
        period = 0.05  # 20Hz
        while self._want_running:
            self._led_dirty.wait(timeout=period)
            if not self._want_running:
                break
            now_ns = time.perf_counter_ns()
            # Expire overlays first; if any expired, force a final paint
            with self._overlay_lock:
                expired = [k for k, t in self._overlay_until.items() if now_ns >= t]
                for k in expired:
                    del self._overlay_until[k]
                active_overlays = list(self._overlay_until.keys())

            still_dirty = self._led_dirty.is_set() or expired or active_overlays
            if not still_dirty:
                continue
            self._led_dirty.clear()

            # Build composite buffer: caller state + overlays
            with self._led_lock:
                buf = bytearray(self._led_state)
            for name in active_overlays:
                off = _LED_FOR_BUTTON.get(name)
                if off is not None:
                    buf[off] = LED_MAX

            # SHIFT held → bright SHIFT LEDs (live read of last_buttons)
            buf[LED_A_SHIFT] = LED_MAX if self._last_buttons.get("a_shift") else LED_DIM
            buf[LED_B_SHIFT] = LED_MAX if self._last_buttons.get("b_shift") else LED_DIM

            # Render the next tick if overlays are still active
            if active_overlays:
                self._led_dirty.set()

            try:
                self._write_queue.put_nowait(bytes([LED_REPORT]) + bytes(buf))
            except queue.Full:
                # LED state is "current snapshot" — drop oldest, push newest
                try:
                    self._write_queue.get_nowait()
                    self._write_queue.put_nowait(bytes([LED_REPORT]) + bytes(buf))
                except Exception:
                    pass

    def _led_boot_animation(self):
        """Quick proof-of-life sweep on connect. Direct enqueue."""
        try:
            buf = bytearray(62)
            for off in LED_A_PADS_GREEN + LED_B_PADS_GREEN:
                buf[off] = LED_LIVE
            self._write_queue.put_nowait(bytes([LED_REPORT]) + bytes(buf))
            time.sleep(0.10)
            buf = bytearray(62)
            for off in LED_A_PADS_BLUE + LED_B_PADS_BLUE:
                buf[off] = LED_LIVE
            self._write_queue.put_nowait(bytes([LED_REPORT]) + bytes(buf))
            time.sleep(0.10)
        except Exception as e:
            logger.debug(f"boot animation failed: {e}")

    # ── Writer thread ────────────────────────────────────────────────────

    def _writer_loop(self):
        """THE ONLY thread that calls device.write(). cython-hidapi has no
        internal write serialization on Windows — concurrent writes corrupt
        USB transfers. All LED writes funnel here via _write_queue."""
        while self._want_running:
            try:
                payload = self._write_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._device_lock:
                d = self.device
            if d is None:
                continue
            try:
                d.write(payload)
            except Exception as e:
                logger.warning(f"HID write failed: {e}; closing device")
                self._close_device()  # watchdog will reopen


# ─── Module-level helpers ───────────────────────────────────────────────────

def run_in_thread() -> Optional[S2Controller]:
    """Convenience: build, start, and return a controller (or None on failure)."""
    ctrl = S2Controller()
    ok, msg = ctrl.start()
    print(f"Traktor S2: {msg}")
    return ctrl if ok else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ctrl = S2Controller()
    ok, msg = ctrl.start()
    print(msg)
    if not ok:
        raise SystemExit(1)
    print("Press Ctrl-C to exit. (For full event tracing run test_s2_full.py)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ctrl.stop()
        print("Stopped.")
