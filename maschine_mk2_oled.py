"""
Maschine MK2 OLED renderer.

The MK2 has two 256x64 monochrome OLED screens. Each frame is a 2048-byte
1bpp framebuffer (256 * 64 / 8 = 2048). Frames are pushed to the device
over its already-open HID handle in 8 chunks of 256 bytes, each prefixed
by a 9-byte header:

    [0xE0 | display_index, 0x00, 0x00, chunk_index * 8,
     0x00, 0x20, 0x00, 0x08, 0x00]

Display 0 = LEFT (header byte = 0xE0), display 1 = RIGHT (0xE1).
Each framebuffer byte must be BIT-REVERSED before transmission so the
LCD's MSB-first orientation matches the natural left-to-right pixel
order in our internal buffer.

This module does NOT open the HID device. The Maschine MK2 input reader
(`maschine_mk2.MaschineMK2`) already holds the device handle exclusively
(NIHA must be stopped). We share that handle and synchronise writes via
an internal lock so the input poller and the display writer never
interleave a chunked frame.

Integration sketch (do NOT wire — just for reference):

    # inside maschine_mk2.MaschineMK2 after self._device is opened ───
    from maschine_mk2_oled import MK2Display, StatusRenderer, TextRenderer

    self._display = MK2Display(self._device)
    self._status = StatusRenderer()

    # push a frame from anywhere on the main app thread:
    fb_left = self._status.render_left(
        live_filename="midnight_run_v3.mp4",
        bpm=128.5,
        scratch_count=3,
    )
    self._display.set_framebuffer(0, fb_left)

    # or a quick blank on shutdown:
    self._display.blank_both()

The renderer classes are pure-Python (PIL + numpy). PIL is already a
dependency via `thumbnails.py`, so no new pip installs.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable, Optional

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
    _PIL_ERROR: Optional[Exception] = None
except Exception as e:  # pragma: no cover - PIL is a project dep
    _PIL_AVAILABLE = False
    _PIL_ERROR = e
    Image = ImageDraw = ImageFont = None  # type: ignore

logger = logging.getLogger(__name__)

# ── protocol constants ────────────────────────────────────────────────
DISPLAY_WIDTH = 256
DISPLAY_HEIGHT = 64
FRAMEBUFFER_BYTES = DISPLAY_WIDTH * DISPLAY_HEIGHT // 8  # 2048
CHUNK_COUNT = 8
CHUNK_PAYLOAD_BYTES = FRAMEBUFFER_BYTES // CHUNK_COUNT   # 256
HEADER_LEN = 9
# Pre-compute the per-byte bit-reversal table once at import time. Doing
# the bit-reverse 2048 times in a generator expression on every frame is
# painfully slow; a lookup table runs in microseconds.
_BIT_REVERSE_LUT = bytes(
    int(f"{b:08b}"[::-1], 2) for b in range(256)
)
# Numpy view of the same table — used for vectorised reversal of a whole
# framebuffer in one shot (orders of magnitude faster than a python loop).
_BIT_REVERSE_NP = np.frombuffer(_BIT_REVERSE_LUT, dtype=np.uint8)


def _bit_reverse_framebuffer(fb: bytes | bytearray) -> bytes:
    """Return a copy of `fb` with every byte's bits reversed (MSB↔LSB)."""
    arr = np.frombuffer(bytes(fb), dtype=np.uint8)
    return _BIT_REVERSE_NP[arr].tobytes()


# ── 1-bpp packing helpers ─────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
#  PACKING MODE — tune these two if the screens show "jumbled" output
# ─────────────────────────────────────────────────────────────────────
# Small monochrome OLED/LCD controllers come in two byte layouts:
#   - "raster":  8 horizontally-adjacent pixels per byte, rows top→bottom
#   - "page":    8 VERTICALLY-stacked pixels per byte, 8-px-tall pages
#                left→right, top→bottom (the SSD1306 convention)
# Both produce 2048 bytes for 256x64, so a wrong guess looks like
# "recognizable garbage." If the OLEDs are jumbled, flip these:
#   The 4 combos to try (in rough likelihood order):
#     1. PACK_MODE="page",   BIT_REVERSE=False   ← new default
#     2. PACK_MODE="page",   BIT_REVERSE=True
#     3. PACK_MODE="raster", BIT_REVERSE=False
#     4. PACK_MODE="raster", BIT_REVERSE=True    ← original (was jumbled)
# Confirmed via the "morse dashes" symptom: page-mode packs vertical
# 8px columns that the display painted as horizontal runs → dashes.
# Display reads RASTER. Original raster+reverse looked "jumbled" because
# of the bit-reverse, so the correct combo is raster + NO reverse.
PACK_MODE = "raster"        # "page" | "raster"
BIT_REVERSE = False         # apply per-byte MSB↔LSB flip before send


def _pack_raster(bits: np.ndarray) -> bytes:
    """8 horizontally-adjacent pixels per byte, MSB = leftmost. Rows
    top→bottom. Output = H*W/8 bytes."""
    return np.packbits(bits, axis=-1).tobytes()


def _pack_page(bits: np.ndarray) -> bytes:
    """8 vertically-stacked pixels per byte (bit 0 = topmost). Pages are
    8px tall, ordered top→bottom; within a page, columns left→right.
    For 256x64 → 8 pages x 256 cols = 2048 bytes. SSD1306-style."""
    h, w = bits.shape
    if h % 8 != 0:
        # Pad to a multiple of 8 rows so the reshape is clean.
        pad = 8 - (h % 8)
        bits = np.vstack([bits, np.zeros((pad, w), dtype=np.uint8)])
        h += pad
    pages = bits.reshape(h // 8, 8, w)          # (n_pages, 8, w)
    weights = (1 << np.arange(8)).astype(np.uint16).reshape(1, 8, 1)
    packed = (pages * weights).sum(axis=1).astype(np.uint8)  # (n_pages, w)
    return packed.tobytes()


def pack_1bpp(pixels: np.ndarray) -> bytes:
    """Pack a (H, W) numpy uint8 array (0/1 or 0/255) into a 1bpp buffer.
    Layout chosen by the module-level PACK_MODE flag. Output length
    = H*W/8 bytes."""
    if pixels.dtype != np.uint8:
        pixels = pixels.astype(np.uint8)
    bits = (pixels > 0).astype(np.uint8)
    if PACK_MODE == "page":
        return _pack_page(bits)
    return _pack_raster(bits)


# ─────────────────────────────────────────────────────────────────────
#  MK2Display — talks to the HID device
# ─────────────────────────────────────────────────────────────────────
class MK2Display:
    """Pushes 1bpp framebuffers to the two Maschine MK2 OLED screens.

    Thread-safe: a single lock serialises every multi-chunk write so the
    8 chunks of a frame are never interrupted by a write to the other
    display (or any other caller). The input poller in `MaschineMK2`
    only ever READS from the device, so the lock is sufficient.
    """

    def __init__(self, device, write_lock: Optional[threading.Lock] = None):
        if device is None:
            raise ValueError("MK2Display requires an opened hidapi device")
        self._device = device
        # If the owner (MaschineMK2) passes in its device-write lock we
        # share it, so the 49-byte pad-LED report can never interleave a
        # chunk of an 8-chunk OLED frame. (Audit fix M15.) When called
        # standalone (tests) we fall back to a private lock.
        self._lock = write_lock if write_lock is not None else threading.Lock()

    # ── public API ────────────────────────────────────────────────────
    def set_framebuffer(self, display_idx: int, fb: bytes | bytearray) -> None:
        """Push 2048 raw 1bpp bytes to display 0 (left) or 1 (right).

        Bit-reversal is applied internally; pass the buffer with the
        natural orientation produced by `pack_1bpp`.
        """
        if display_idx not in (0, 1):
            raise ValueError(f"display_idx must be 0 or 1, got {display_idx!r}")
        if len(fb) != FRAMEBUFFER_BYTES:
            raise ValueError(
                f"framebuffer must be {FRAMEBUFFER_BYTES} bytes, "
                f"got {len(fb)}"
            )
        # Bit-reverse only if the controller wants MSB-first (toggle via
        # the module-level BIT_REVERSE flag — see the packing-mode notes).
        out_fb = _bit_reverse_framebuffer(fb) if BIT_REVERSE else bytes(fb)
        self._write_chunks(display_idx, out_fb)

    def clear(self, display_idx: int) -> None:
        """Push an all-zero (blank) framebuffer to one display."""
        self.set_framebuffer(display_idx, b"\x00" * FRAMEBUFFER_BYTES)

    def blank_both(self) -> None:
        """Clear both screens."""
        self.clear(0)
        self.clear(1)

    # ── plumbing ──────────────────────────────────────────────────────
    def _write_chunks(self, display_idx: int, reversed_fb: bytes) -> None:
        header_byte0 = 0xE0 | (display_idx & 0x01)
        with self._lock:
            for chunk in range(CHUNK_COUNT):
                header = bytes([
                    header_byte0,
                    0x00, 0x00,
                    chunk * 8,
                    0x00, 0x20, 0x00, 0x08, 0x00,
                ])
                start = chunk * CHUNK_PAYLOAD_BYTES
                payload = reversed_fb[start : start + CHUNK_PAYLOAD_BYTES]
                try:
                    self._device.write(header + payload)
                except Exception as e:
                    # Don't blow up the caller — log and bail on this
                    # frame. Subsequent frames will retry from scratch.
                    logger.warning(
                        "MK2 display %d chunk %d write failed: %s",
                        display_idx, chunk, e,
                    )
                    return


# ─────────────────────────────────────────────────────────────────────
#  TextRenderer — draws lines of text into a framebuffer
# ─────────────────────────────────────────────────────────────────────
class TextRenderer:
    """Renders one or more text lines into a 256x64 1bpp framebuffer.

    Uses PIL's default bitmap font (no TTF, no extra files) so it works
    out-of-the-box. PIL is already a project dependency via thumbnails.
    """

    def __init__(self, width: int = DISPLAY_WIDTH, height: int = DISPLAY_HEIGHT):
        if not _PIL_AVAILABLE:
            raise RuntimeError(f"PIL not available: {_PIL_ERROR}")
        self.width = width
        self.height = height
        # PIL's default font is a 6x11 bitmap; small enough that we
        # can fit ~5 lines on a 64-px screen.
        self._font = ImageFont.load_default()
        self._line_h = self._estimate_line_height()

    def _estimate_line_height(self) -> int:
        # Bbox of an ascender + descender pair gives the true line height.
        try:
            bbox = self._font.getbbox("Ay|g")
            h = bbox[3] - bbox[1]
        except Exception:
            h = 11  # safe default for PIL's bitmap font
        return max(8, h + 1)

    def render_lines(
        self,
        lines: Iterable[str],
        align: str = "left",
    ) -> bytes:
        """Render lines into a 2048-byte framebuffer.

        Lines that are too wide are clipped (not wrapped — wrapping
        decisions belong to the caller, who knows the semantic content).
        Excess lines beyond what fits vertically are dropped.

        `align`: "left" | "center" | "right".
        """
        img = Image.new("1", (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)

        max_lines = max(1, self.height // self._line_h)
        y = 0
        for i, raw in enumerate(lines):
            if i >= max_lines:
                break
            text = "" if raw is None else str(raw)
            # measure
            try:
                bbox = draw.textbbox((0, 0), text, font=self._font)
                tw = bbox[2] - bbox[0]
            except Exception:
                tw = len(text) * 6
            if align == "center":
                x = max(0, (self.width - tw) // 2)
            elif align == "right":
                x = max(0, self.width - tw)
            else:
                x = 0
            draw.text((x, y), text, fill=1, font=self._font)
            y += self._line_h

        # Convert PIL "1" mode image → numpy (H, W) of 0/255, then pack.
        arr = np.array(img, dtype=np.uint8)
        return pack_1bpp(arr)


# ─────────────────────────────────────────────────────────────────────
#  PadMapRenderer — 4x4 grid of pad states
# ─────────────────────────────────────────────────────────────────────
class PadMapRenderer:
    """Renders a 4x4 pad grid showing which pads are lit.

    Input: list of 16 (r, g, b) tuples in PAD-LABEL order 1..16
    (bottom-left = 1, matching maschine_mk2._physical_pad_num).
    """

    LIT_THRESHOLD = 30  # r+g+b sum; under this = considered "off"

    def __init__(
        self,
        width: int = DISPLAY_WIDTH,
        height: int = DISPLAY_HEIGHT,
        margin: int = 4,
        cell_gap: int = 4,
    ):
        if not _PIL_AVAILABLE:
            raise RuntimeError(f"PIL not available: {_PIL_ERROR}")
        self.width = width
        self.height = height
        self.margin = margin
        self.cell_gap = cell_gap

    def render(self, pad_states: list[tuple[int, int, int]]) -> bytes:
        if len(pad_states) != 16:
            raise ValueError(f"expected 16 pad states, got {len(pad_states)}")
        img = Image.new("1", (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)

        # Carve a 4x4 grid out of (width - 2*margin) x (height - 2*margin).
        # Make the cells square based on the smaller dimension so the
        # whole map looks clean on a 256x64 screen.
        avail_w = self.width - 2 * self.margin
        avail_h = self.height - 2 * self.margin
        cell_w = (avail_w - 3 * self.cell_gap) // 4
        cell_h = (avail_h - 3 * self.cell_gap) // 4
        cell = min(cell_w, cell_h)
        grid_w = 4 * cell + 3 * self.cell_gap
        grid_h = 4 * cell + 3 * self.cell_gap
        # center the grid horizontally; pin it to the top vertically so
        # the right side has room for the bank label / other glyphs.
        ox = (self.width - grid_w) // 2
        oy = (self.height - grid_h) // 2

        for label in range(1, 17):
            # label 1 = bottom-left → row 3 (from top), col 0
            z = label - 1
            row_from_bottom = z // 4
            col = z % 4
            row_from_top = 3 - row_from_bottom
            x0 = ox + col * (cell + self.cell_gap)
            y0 = oy + row_from_top * (cell + self.cell_gap)
            x1 = x0 + cell - 1
            y1 = y0 + cell - 1
            r, g, b = pad_states[z]
            lit = (r + g + b) > self.LIT_THRESHOLD
            if lit:
                draw.rectangle((x0, y0, x1, y1), fill=1, outline=1)
            else:
                draw.rectangle((x0, y0, x1, y1), fill=0, outline=1)

        arr = np.array(img, dtype=np.uint8)
        return pack_1bpp(arr)


# ─────────────────────────────────────────────────────────────────────
#  ThumbnailRenderer — JPEG → dithered 1bpp framebuffer
# ─────────────────────────────────────────────────────────────────────
class ThumbnailRenderer:
    """Loads a clip thumbnail JPEG and renders it as a dithered 1bpp frame.

    Clip thumbnails live at ``~/.setpiece/thumbnails/<id>.jpg``.
    The source is scaled to fit 256x64 *preserving aspect ratio* and
    letterboxed (centered on a black field), then converted to 1-bit via
    Floyd-Steinberg error diffusion so photographic content reads well on
    a monochrome OLED.

    Stateless — safe to call from any thread. Each call returns a fresh
    2048-byte framebuffer. If the JPEG is missing or unreadable, a blank
    framebuffer is returned (logged at WARNING) so the caller never has to
    special-case a bad path mid-set.
    """

    def __init__(self, width: int = DISPLAY_WIDTH, height: int = DISPLAY_HEIGHT):
        if not _PIL_AVAILABLE:
            raise RuntimeError(f"PIL not available: {_PIL_ERROR}")
        self.width = width
        self.height = height

    def render(self, jpeg_path: str) -> bytes:
        """Render `jpeg_path` into a letterboxed, dithered 2048-byte frame."""
        try:
            img = self._load_fitted(jpeg_path)
        except FileNotFoundError:
            logger.warning("thumbnail not found: %s", jpeg_path)
            return b"\x00" * FRAMEBUFFER_BYTES
        except Exception as e:
            logger.warning("thumbnail load failed for %s: %s", jpeg_path, e)
            return b"\x00" * FRAMEBUFFER_BYTES
        arr = np.array(img, dtype=np.uint8)
        return pack_1bpp(arr)

    def render_fitted_image(self, jpeg_path: str):
        """Return the letterboxed PIL "1" image (for compositing callers).

        Used by StatusRenderer.render_left_with_thumb so it can paste the
        thumbnail onto a larger canvas without re-doing the dither.
        Returns ``None`` on any load error.
        """
        try:
            return self._load_fitted(jpeg_path)
        except FileNotFoundError:
            logger.warning("thumbnail not found: %s", jpeg_path)
            return None
        except Exception as e:
            logger.warning("thumbnail load failed for %s: %s", jpeg_path, e)
            return None

    def _load_fitted(self, jpeg_path: str):
        """Load → grayscale → aspect-fit → letterbox → Floyd-Steinberg 1-bit."""
        # PIL raises FileNotFoundError itself, but be explicit for clarity.
        import os
        if not os.path.isfile(jpeg_path):
            raise FileNotFoundError(jpeg_path)
        with Image.open(jpeg_path) as src:
            gray = src.convert("L")
        sw, sh = gray.size
        if sw <= 0 or sh <= 0:
            raise ValueError(f"degenerate thumbnail size {gray.size}")
        # Aspect-preserving fit inside width x height.
        scale = min(self.width / sw, self.height / sh)
        tw = max(1, min(self.width, int(round(sw * scale))))
        th = max(1, min(self.height, int(round(sh * scale))))
        resized = gray.resize((tw, th), Image.LANCZOS)
        # Letterbox: paste centered on a black L canvas, THEN dither the
        # whole canvas so the black bars stay crisp black.
        canvas = Image.new("L", (self.width, self.height), 0)
        ox = (self.width - tw) // 2
        oy = (self.height - th) // 2
        canvas.paste(resized, (ox, oy))
        # PIL's convert("1") uses Floyd-Steinberg error diffusion by default.
        return canvas.convert("1")


# ─────────────────────────────────────────────────────────────────────
#  ActivityRenderer — rolling scrolling bar-graph visualizer
# ─────────────────────────────────────────────────────────────────────
class ActivityRenderer:
    """Rolling visualizer: a scrolling bar graph of recent 0..1 samples.

    Feed it values with ``push_sample(v)`` (audio level, beat intensity,
    whatever) and call ``render()`` to get a 2048-byte framebuffer showing
    the most recent samples as vertical bars across the 256px width.

    Stateful — keeps an internal ring buffer. All state access is guarded
    by a lock so ``push_sample`` (audio thread) and ``render`` (display
    thread) can run concurrently.
    """

    def __init__(
        self,
        width: int = DISPLAY_WIDTH,
        height: int = DISPLAY_HEIGHT,
        capacity: int = 64,
    ):
        if not _PIL_AVAILABLE:
            raise RuntimeError(f"PIL not available: {_PIL_ERROR}")
        self.width = width
        self.height = height
        self.capacity = max(1, capacity)
        self._lock = threading.Lock()
        # Ring buffer of floats, newest at the end after _ordered().
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._pos = 0          # write index
        self._count = 0        # how many valid samples so far

    def push_sample(self, value: float) -> None:
        """Append one 0..1 sample (clamped) to the ring buffer."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = 0.0
        if v != v:  # NaN guard
            v = 0.0
        v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
        with self._lock:
            self._buf[self._pos] = v
            self._pos = (self._pos + 1) % self.capacity
            if self._count < self.capacity:
                self._count += 1

    def reset(self) -> None:
        """Zero the ring buffer."""
        with self._lock:
            self._buf[:] = 0.0
            self._pos = 0
            self._count = 0

    def _ordered(self) -> np.ndarray:
        """Return samples oldest→newest (only the valid ones)."""
        with self._lock:
            if self._count < self.capacity:
                return self._buf[:self._count].copy()
            # full buffer: unwrap the ring starting at the write head
            return np.concatenate(
                (self._buf[self._pos:], self._buf[:self._pos])
            ).copy()

    def render(self) -> bytes:
        """Render the recent samples as a scrolling bar graph."""
        samples = self._ordered()
        img = Image.new("1", (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)
        # Baseline rule along the bottom.
        draw.line((0, self.height - 1, self.width - 1, self.height - 1), fill=1)
        n = len(samples)
        if n > 0:
            # Each sample gets an even slice of the width; bars are drawn
            # newest-on-the-right so it visually scrolls left as you push.
            slot = self.width / float(self.capacity)
            # Right-align: the newest sample sits at the far right.
            start_slot = self.capacity - n
            for i, v in enumerate(samples):
                slot_idx = start_slot + i
                x0 = int(round(slot_idx * slot))
                x1 = int(round((slot_idx + 1) * slot)) - 1
                if x1 < x0:
                    x1 = x0
                bar_h = int(round(v * (self.height - 2)))
                y1 = self.height - 2
                y0 = y1 - bar_h
                if bar_h <= 0:
                    # still draw a 1px nub so silence is visible
                    draw.point((x0, y1), fill=1)
                else:
                    draw.rectangle((x0, y0, x1, y1), fill=1)
        arr = np.array(img, dtype=np.uint8)
        return pack_1bpp(arr)


# ─────────────────────────────────────────────────────────────────────
#  BeatPulseRenderer — decaying filled-circle beat indicator
# ─────────────────────────────────────────────────────────────────────
class BeatPulseRenderer:
    """A big filled circle that flashes full on ``trigger()`` and decays.

    Call ``trigger()`` on each beat; call ``render()`` every display frame.
    The circle radius (and a thin outer ring) is scaled by a decay value
    that falls from 1.0 toward 0.0 over ``decay_seconds`` of wall-clock
    time. Great for the right OLED during audio-reactive sets.

    Stateful — the decay clock and trigger time are guarded by a lock.
    """

    def __init__(
        self,
        width: int = DISPLAY_WIDTH,
        height: int = DISPLAY_HEIGHT,
        decay_seconds: float = 0.35,
        min_scale: float = 0.18,
    ):
        if not _PIL_AVAILABLE:
            raise RuntimeError(f"PIL not available: {_PIL_ERROR}")
        self.width = width
        self.height = height
        self.decay_seconds = max(0.01, float(decay_seconds))
        # Circle never fully vanishes — keeps a faint resting dot so the
        # screen doesn't look dead between beats.
        self.min_scale = max(0.0, min(1.0, float(min_scale)))
        self._lock = threading.Lock()
        # Trigger timestamp far enough in the past that we start at rest.
        self._last_trigger = -1e9

    def trigger(self) -> None:
        """Reset the pulse to full brightness/size (call on each beat)."""
        import time
        with self._lock:
            self._last_trigger = time.monotonic()

    def _current_scale(self) -> float:
        """Decay value in [min_scale, 1.0] based on time since trigger."""
        import time
        with self._lock:
            last = self._last_trigger
        elapsed = time.monotonic() - last
        if elapsed <= 0.0:
            frac = 1.0
        elif elapsed >= self.decay_seconds:
            frac = 0.0
        else:
            # Ease-out: quadratic decay reads punchier than linear.
            lin = 1.0 - (elapsed / self.decay_seconds)
            frac = lin * lin
        return self.min_scale + (1.0 - self.min_scale) * frac

    def render(self) -> bytes:
        """Render the current decayed circle into a 2048-byte frame."""
        scale = self._current_scale()
        img = Image.new("1", (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)
        cx = self.width / 2.0
        cy = self.height / 2.0
        # Max radius fits the SHORTER axis (the 64px height) with a margin.
        max_r = (min(self.width, self.height) / 2.0) - 2.0
        r = max(1.0, max_r * scale)
        bbox = (cx - r, cy - r, cx + r, cy + r)
        draw.ellipse(bbox, fill=1)
        # Outer ring at full max radius as a static frame so there's always
        # a reference for how "big" a full hit is.
        ring = (cx - max_r, cy - max_r, cx + max_r, cy + max_r)
        draw.ellipse(ring, outline=1)
        arr = np.array(img, dtype=np.uint8)
        return pack_1bpp(arr)


# ─────────────────────────────────────────────────────────────────────
#  StatusRenderer — high-level convenience for both screens
# ─────────────────────────────────────────────────────────────────────
class StatusRenderer:
    """High-level scene compositor for live VJ readouts."""

    def __init__(self):
        if not _PIL_AVAILABLE:
            raise RuntimeError(f"PIL not available: {_PIL_ERROR}")
        self._text = TextRenderer()
        self._pads = PadMapRenderer()
        self._thumb = ThumbnailRenderer()
        # MK2's Gen-1 OLED screens are low contrast + low resolution —
        # readability from a glancing distance requires LARGE, bold
        # glyphs. We load three font sizes:
        #   _huge_font  — bank letter (single char fills most of the screen)
        #   _big_font   — bank name (medium-large)
        #   _med_font   — BPM and other secondary glanceable info
        # All bold weight if available; falls through to PIL default if
        # no TTFs found.
        self._huge_font = self._load_font([
            ("C:\\Windows\\Fonts\\arialbd.ttf", 56),
            ("C:\\Windows\\Fonts\\consolab.ttf", 56),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56),
        ])
        self._big_font = self._load_font([
            ("C:\\Windows\\Fonts\\arialbd.ttf", 26),
            ("C:\\Windows\\Fonts\\consolab.ttf", 26),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26),
        ])
        self._med_font = self._load_font([
            ("C:\\Windows\\Fonts\\arialbd.ttf", 16),
            ("C:\\Windows\\Fonts\\consolab.ttf", 16),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16),
        ])

    @staticmethod
    def _load_font(candidates):
        """Try each (path, size) in order. Return first that loads, else
        PIL bitmap default. Never raises."""
        for path, size in candidates:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        return ImageFont.load_default()

    # ── LEFT screen: BANK-dominant readout for peripheral glance ─────
    def render_left(
        self,
        live_filename: str = "",
        bpm: float = 0.0,
        scratch_count: int = 0,
        bank_letter: str = "A",
        bank_name: str = "",
        page_name: str = "",
        last_vertical: str = "",
    ) -> bytes:
        """User's directive 2026-05-16: drop the small-text status
        clutter (live filename, scratch count) since that's already
        on the main display + iPad. OLED is peripheral-glance only,
        so make ONE thing huge and obvious — the active bank letter —
        with name + BPM as secondary info.

        Layout (256x64):
            ┌──────────┬────────────────────────────────┐
            │          │  Dance                         │ ← bank name, big
            │    D     │                                │
            │   HUGE   │  128.5 BPM                     │ ← BPM, medium
            └──────────┴────────────────────────────────┘

        2026-05-17: optional `page_name` + `last_vertical` (both pre-
        formatted by caller) add a thin divider at y=44 and two small
        text lines underneath, showing the active MK2 vertical-page
        and the most recently fired vertical slot. Default-empty →
        renders the original layout with no regression.
        """
        img = Image.new("1", (DISPLAY_WIDTH, DISPLAY_HEIGHT), 0)
        draw = ImageDraw.Draw(img)

        # Verticals strip eats the bottom 20px; compress the upper
        # layout into the top 44px when active. Otherwise keep the
        # original positions exactly.
        has_verticals = bool(page_name) or bool(last_vertical)

        # ── Left column: HUGE bank letter (ORIGINAL LAYOUT) ─────────
        # User feedback 2026-05-17: don't compress the left OLED for
        # verticals. Page info moved to RIGHT OLED instead. Left OLED
        # restored to original always-on layout (huge letter + name + BPM).
        letter = (bank_letter or "A").upper()[:1]
        try:
            bb = draw.textbbox((0, 0), letter, font=self._huge_font)
            lw = bb[2] - bb[0]
            lh = bb[3] - bb[1]
        except Exception:
            lw = lh = 48
        col_w = 80
        try:
            ly = max(0, (DISPLAY_HEIGHT - lh) // 2 - bb[1])
        except Exception:
            ly = 4
        lx = max(0, (col_w - lw) // 2)
        draw.text((lx, ly), letter, fill=1, font=self._huge_font)

        # ── Right area: bank name + BPM stacked ──────────────────────
        right_x = col_w + 6
        name = bank_name or ""
        name_max_w = DISPLAY_WIDTH - right_x - 4
        name_fit = self._fit_text(draw, self._big_font, name, name_max_w)
        draw.text((right_x, 4), name_fit, fill=1, font=self._big_font)
        bpm_text = f"{bpm:.1f} BPM" if bpm and bpm > 0 else "-- BPM"
        draw.text((right_x, 38), bpm_text, fill=1, font=self._med_font)
        # page_name / last_vertical are accepted but no longer used
        # here — they live on the right OLED now (render_right).

        arr = np.array(img, dtype=np.uint8)
        return pack_1bpp(arr)

    # ── LEFT screen variant: bank-switch confirmation flash ─────────
    def render_bank_flash(
        self,
        letter: str,
        name: str,
        count: int = 0,
        first_files: Optional[list] = None,
    ) -> bytes:
        """Bank-switch flash variant of render_left. Same layout — but
        the LETTER block uses INVERTED video (white background, black
        letter) so the user sees an immediate visual "this just
        changed" cue. After the 3s hold expires, the normal
        non-inverted display takes over.

        ``count`` and ``first_files`` kept in the signature for
        backward compat but no longer rendered (user feedback: too
        small + too cluttered for the Gen-1 OLED resolution; main
        display has the file list anyway)."""
        img = Image.new("1", (DISPLAY_WIDTH, DISPLAY_HEIGHT), 0)
        draw = ImageDraw.Draw(img)

        # ── Left column: HUGE letter, INVERTED ───────────────────────
        letter_ch = (letter or "A").upper()[:1]
        col_w = 80
        # White background for the left column
        draw.rectangle((0, 0, col_w - 1, DISPLAY_HEIGHT - 1), fill=1)
        try:
            bb = draw.textbbox((0, 0), letter_ch, font=self._huge_font)
            lw = bb[2] - bb[0]
            lh = bb[3] - bb[1]
            ly = max(0, (DISPLAY_HEIGHT - lh) // 2 - bb[1])
        except Exception:
            lw = lh = 48
            ly = 4
        lx = max(0, (col_w - lw) // 2)
        # Black letter on white background = inverted
        draw.text((lx, ly), letter_ch, fill=0, font=self._huge_font)

        # ── Right area: bank name big ────────────────────────────────
        right_x = col_w + 6
        name_text = (name or "").strip() or "(unnamed)"
        name_max_w = DISPLAY_WIDTH - right_x - 4
        name_fit = self._fit_text(draw, self._big_font, name_text, name_max_w)
        draw.text((right_x, 4), name_fit, fill=1, font=self._big_font)
        # "BANK X" label below, medium font
        draw.text((right_x, 38), f"BANK {letter_ch}",
                  fill=1, font=self._med_font)

        arr = np.array(img, dtype=np.uint8)
        return pack_1bpp(arr)

    def render_vote_flash(self, direction: str, score: int,
                          ups: int, downs: int,
                          clip_name: str = "") -> bytes:
        """Brief vote-confirmation flash on the LEFT OLED. Shown for
        ~1.2s after master `<`/`>` press. direction is '+' or '-'.
        Layout: HUGE +1 / -1 on left, current score + tally on right,
        truncated clip name across the bottom. Distinct from bank
        flash by using a colored letter style (no inversion)."""
        img = Image.new("1", (DISPLAY_WIDTH, DISPLAY_HEIGHT), 0)
        draw = ImageDraw.Draw(img)
        # Left: HUGE direction sign
        col_w = 80
        dir_ch = "+1" if direction == "+" else "-1"
        try:
            bb = draw.textbbox((0, 0), dir_ch, font=self._huge_font)
            lw = bb[2] - bb[0]
            lh = bb[3] - bb[1]
            ly = max(0, (DISPLAY_HEIGHT - lh) // 2 - bb[1])
        except Exception:
            lw = lh = 48
            ly = 4
        lx = max(0, (col_w - lw) // 2)
        draw.text((lx, ly), dir_ch, fill=1, font=self._huge_font)
        # Right: current score + tally
        right_x = col_w + 6
        score_text = f"SCORE {score:+d}"
        draw.text((right_x, 4), score_text, fill=1, font=self._big_font)
        tally = f"^{ups}  v{downs}"
        draw.text((right_x, 30), tally, fill=1, font=self._med_font)
        # Bottom row: clip name truncated. Use med_font (small_font
        # doesn't exist on this OLED renderer); name fitted to width.
        name_text = (clip_name or "").strip()
        if name_text:
            name_fit = self._fit_text(
                draw, self._med_font, name_text, DISPLAY_WIDTH - 8)
            draw.text(
                (4, DISPLAY_HEIGHT - 16),
                name_fit,
                fill=1, font=self._med_font,
            )
        arr = np.array(img, dtype=np.uint8)
        return pack_1bpp(arr)

    def render_layer_flash(self, layer_name: str) -> bytes:
        """Flash on layer-cycle (MK2 STEP button or iPad chip tap).
        "LAYER" header + layer name. Uses BIG font (26pt) not HUGE
        (56pt) for the name — huge spilled the bottom of the 38px
        zone below the 26px header on the 64-tall OLED.
        Falls back to normal display on next refresh."""
        img = Image.new("1", (DISPLAY_WIDTH, DISPLAY_HEIGHT), 0)
        draw = ImageDraw.Draw(img)

        # Smaller header strip so the name has more room.
        header_h = 20
        draw.rectangle((0, 0, DISPLAY_WIDTH - 1, header_h - 1), fill=1)
        header_txt = "LAYER"
        try:
            bb = draw.textbbox((0, 0), header_txt, font=self._med_font)
            tw = bb[2] - bb[0]
        except Exception:
            tw = 50
        tx = max(0, (DISPLAY_WIDTH - tw) // 2)
        draw.text((tx, 1), header_txt, fill=0, font=self._med_font)

        # Layer name in BIG font (26pt) — fits cleanly in the 44px
        # below the 20px header on the 64-tall screen.
        name_text = (layer_name or "default").upper()
        name_max_w = DISPLAY_WIDTH - 12
        try:
            name_fit = self._fit_text(
                draw, self._big_font, name_text, name_max_w)
        except Exception:
            name_fit = name_text
        try:
            bb = draw.textbbox((0, 0), name_fit, font=self._big_font)
            nw = bb[2] - bb[0]
            # Use bb[1] offset so we don't cut top OR bottom for fonts
            # whose bbox doesn't start at y=0.
            ny_offset = -bb[1]
        except Exception:
            nw = len(name_fit) * 14
            ny_offset = 0
        nx = max(0, (DISPLAY_WIDTH - nw) // 2)
        # Vertically center inside the area below the header (44px).
        body_top = header_h
        body_h = DISPLAY_HEIGHT - header_h
        # Big font is ~26-30px tall; center it visually with a small
        # bias up since fonts tend to leave more space below baseline.
        ny = body_top + (body_h - 28) // 2 + ny_offset
        draw.text((nx, ny), name_fit, fill=1, font=self._big_font)

        arr = np.array(img, dtype=np.uint8)
        return pack_1bpp(arr)

    def render_reclassify_flash(self, letter: str, name: str,
                                count: int) -> bytes:
        """Flash on Hold-GROUP-X reclassify gesture. Layout:
        full-width header "RECLASSIFY → X", then the truncated
        clip name + correction-count chip below. ~1.5s hold."""
        img = Image.new("1", (DISPLAY_WIDTH, DISPLAY_HEIGHT), 0)
        draw = ImageDraw.Draw(img)
        # Inverted header strip with the gesture summary.
        header_h = 22
        draw.rectangle((0, 0, DISPLAY_WIDTH - 1, header_h - 1), fill=1)
        L = (letter or "?").upper()[:1]
        header_txt = f"RECLASSIFY -> {L}"
        try:
            bb = draw.textbbox((0, 0), header_txt, font=self._med_font)
            tw = bb[2] - bb[0]
        except Exception:
            tw = 200
        tx = max(2, (DISPLAY_WIDTH - tw) // 2)
        draw.text((tx, 2), header_txt, fill=0, font=self._med_font)
        # Body: clip name (fitted) on first line, count badge below.
        n_text = (name or "").encode(
            "ascii", "replace").decode("ascii")[:42]
        try:
            n_fit = self._fit_text(
                draw, self._med_font, n_text, DISPLAY_WIDTH - 8)
        except Exception:
            n_fit = n_text
        draw.text((4, header_h + 4), n_fit, fill=1, font=self._med_font)
        # Count badge: "x3" right-side
        count_text = f"x{count}"
        try:
            bb = draw.textbbox((0, 0), count_text, font=self._big_font)
            cw = bb[2] - bb[0]
        except Exception:
            cw = 30
        draw.text(
            (DISPLAY_WIDTH - cw - 6, header_h + 18),
            count_text, fill=1, font=self._big_font,
        )
        arr = np.array(img, dtype=np.uint8)
        return pack_1bpp(arr)

    def render_clip_to_banks_flash(self, query: str, matches: int,
                                    distribution: dict,
                                    elapsed_ms: int = 0) -> bytes:
        """Flash on bank/clip_to_banks fire. Layout:
        full-width inverted header "→BANKS  <query>",
        body: 8 letter pills with counts, e.g. "A·12 B·8 C·15 ..."
        plus a small footer "<matches> hits / <ms>ms".
        ~2.5s hold."""
        img = Image.new("1", (DISPLAY_WIDTH, DISPLAY_HEIGHT), 0)
        draw = ImageDraw.Draw(img)
        # Inverted header
        header_h = 20
        draw.rectangle((0, 0, DISPLAY_WIDTH - 1, header_h - 1), fill=1)
        q_safe = (query or "?").encode(
            "ascii", "replace").decode("ascii")[:28]
        header_txt = f"->BANKS  {q_safe}"
        try:
            header_txt = self._fit_text(
                draw, self._med_font, header_txt, DISPLAY_WIDTH - 8)
        except Exception:
            pass
        draw.text((4, 2), header_txt, fill=0, font=self._med_font)
        # Body: 8 letter chips on two rows of 4 each
        body_y = header_h + 4
        row_h = 16
        col_w = DISPLAY_WIDTH // 4
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        for i, ltr in enumerate(letters):
            row = i // 4
            col = i % 4
            x = col * col_w + 4
            y = body_y + row * row_h
            n = (distribution or {}).get(ltr, -1)
            if n is None or n < 0:
                txt = f"{ltr}·-"
            else:
                txt = f"{ltr}·{n}"
            draw.text((x, y), txt, fill=1, font=self._med_font)
        # Footer: hits + ms
        footer_y = body_y + 2 * row_h + 2
        if footer_y < DISPLAY_HEIGHT - 8:
            footer_txt = f"{matches} hits / {elapsed_ms}ms"
            draw.text((4, footer_y), footer_txt,
                      fill=1, font=self._med_font)
        arr = np.array(img, dtype=np.uint8)
        return pack_1bpp(arr)

    # ── LEFT screen variant: thumbnail preview + now-playing text ────
    def render_left_with_thumb(
        self,
        jpeg_path: str,
        live_filename: str,
        bpm: float,
    ) -> bytes:
        """Composite a clip thumbnail (left third) + text (right 2/3).

        Layout on the 256x64 left screen:

            ┌──────────┬────────────────────────────────┐
            │          │ LIVE  midnight_run_v3.mp4      │
            │  thumb   ├────────────────────────────────┤
            │  ~80px   │  128.5  BPM                    │
            └──────────┴────────────────────────────────┘

        The thumbnail is dithered/letterboxed by ThumbnailRenderer and
        pasted into the left column; if it can't be loaded the column is
        left black and the text simply uses the full width feel.
        """
        img = Image.new("1", (DISPLAY_WIDTH, DISPLAY_HEIGHT), 0)
        draw = ImageDraw.Draw(img)
        small = self._text._font

        # Left column ~ 1/3 of the width.
        col_w = DISPLAY_WIDTH // 3  # 85
        # Render the thumbnail into its own col_w x HEIGHT mini-frame so it
        # letterboxes within the column, then paste it in.
        thumb_img = None
        try:
            mini = ThumbnailRenderer(col_w, DISPLAY_HEIGHT)
            thumb_img = mini.render_fitted_image(jpeg_path)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("thumb column render failed: %s", e)
            thumb_img = None
        if thumb_img is not None:
            img.paste(thumb_img, (0, 0))
        # Divider rule between thumbnail and text.
        draw.line((col_w, 0, col_w, DISPLAY_HEIGHT - 1), fill=1)

        text_x = col_w + 4
        text_w = DISPLAY_WIDTH - text_x - 2

        # Top: LIVE tag + filename, ellipsized to the text column width.
        tag = "LIVE "
        tag_px = 6 * len(tag)
        name = self._fit_text(draw, small, live_filename, max(8, text_w - tag_px))
        draw.text((text_x, 1), tag + name, fill=1, font=small)
        draw.line((col_w + 1, 14, DISPLAY_WIDTH - 1, 14), fill=1)

        # Big BPM block in the text column.
        bpm_text = f"{bpm:5.1f}"
        try:
            bb = draw.textbbox((0, 0), bpm_text, font=self._big_font)
            bpm_h = bb[3] - bb[1]
        except Exception:
            bpm_h = 22
        bpm_y = 16 + max(0, (DISPLAY_HEIGHT - 16 - bpm_h) // 2 - 2)
        draw.text((text_x, bpm_y), bpm_text, fill=1, font=self._big_font)
        draw.text((text_x, DISPLAY_HEIGHT - 10), "BPM", fill=1, font=small)

        arr = np.array(img, dtype=np.uint8)
        return pack_1bpp(arr)

    # ── RIGHT screen: pad map + bank letter ──────────────────────────
    def render_right(
        self,
        pad_states: list[tuple[int, int, int]],
        active_bank: str = "A",
        page_name: str = "",
        last_vertical: str = "",
        verticals: list = None,
        active_page_idx: int = 0,
    ) -> bytes:
        """Right OLED: page-centric view (replaces pad-grid since the
        16 pads have their own RGB LEDs — the OLED pad map was
        redundant info competing for screen real-estate).

        Layout when page is active:
            ┌──────────────────────────────────────────────────┐
            │  PG 3                                            │ y=2  (small)
            │                                                  │
            │     TOP PERF A                                   │ y=14 (HUGE)
            │                                                  │
            │  ↑ Marilyn Mayson                                │ y=46 (medium)
            └──────────────────────────────────────────────────┘

        When no page is set (page_name empty), fall back to the
        original pad-grid + bank-letter layout so the screen isn't
        blank for users who haven't opted into pages."""

        # Fallback to original pad-grid layout when no page active
        if not page_name:
            img = Image.new("1", (DISPLAY_WIDTH, DISPLAY_HEIGHT), 0)
            draw = ImageDraw.Draw(img)
            self._draw_pad_grid(draw, pad_states)
            bank = (active_bank or "?")[:1]
            try:
                bb = draw.textbbox((0, 0), bank, font=self._big_font)
                bw = bb[2] - bb[0]
                bh = bb[3] - bb[1]
            except Exception:
                bw, bh = 16, 22
            pad = 2
            bx1 = DISPLAY_WIDTH - 2
            by0 = 2
            bx0 = bx1 - bw - 2 * pad
            by1 = by0 + bh + 2 * pad
            draw.rectangle((bx0, by0, bx1, by1), fill=0, outline=1)
            draw.text((bx0 + pad, by0 + pad - 2), bank, fill=1,
                      font=self._big_font)
            arr = np.array(img, dtype=np.uint8)
            return pack_1bpp(arr)

        # ── Page-active layout ───────────────────────────────────────
        # New 2026-05-18: when verticals list is provided, render the
        # vertical folder labels in a 2-col x 4-row grid so the
        # operator can read what each vertical button fires WITHOUT
        # looking at the iPad. Page name shrinks to a small header.
        # Falls back to the legacy big-page-name layout when no
        # verticals provided (back-compat).
        img = Image.new("1", (DISPLAY_WIDTH, DISPLAY_HEIGHT), 0)
        draw = ImageDraw.Draw(img)
        small = self._text._font  # PIL default ~6x11

        # Small bank indicator top-right (always shown).
        bank = (active_bank or "?")[:1]
        try:
            bb = draw.textbbox((0, 0), bank, font=self._med_font)
            bw = bb[2] - bb[0]
        except Exception:
            bw = 10
        draw.text((DISPLAY_WIDTH - bw - 4, 1), bank, fill=1,
                  font=self._med_font)

        if verticals and len(verticals) > 0:
            # ── Compact 8-slot folder list layout ──
            # Header: "P{idx+1} {PAGE NAME}" small font, top-left.
            page_idx = int(active_page_idx) + 1
            header = f"P{page_idx} {(page_name or '').upper()}"
            # Truncate header so it doesn't overlap the bank indicator
            # (~38 char budget at 6px font = ~228px).
            max_hdr = DISPLAY_WIDTH - bw - 12
            header_fit = self._fit_text(draw, small, header[:42],
                                        max_hdr)
            draw.text((2, 1), header_fit, fill=1, font=small)

            # Divider line under header
            draw.line((0, 12, DISPLAY_WIDTH - 1, 12), fill=1)

            # 2-col x 4-row grid. Each cell ~128px wide x 12px tall.
            # Left col: slots 0-3; right col: slots 4-7.
            COL_W = DISPLAY_WIDTH // 2  # 128px
            ROW_H = 12
            TOP_Y = 14
            CHARS_PER_CELL = 17   # tight at 6px/char in 128px - "N "
            for slot in range(8):
                col = slot // 4         # 0 or 1
                row = slot % 4          # 0..3
                x = col * COL_W + 2
                y = TOP_Y + row * ROW_H
                v = verticals[slot] if slot < len(verticals) else {}
                label = (v.get("label") if isinstance(v, dict)
                         else str(v or ""))
                label = (label or "")
                # Render: "N name" with name truncated.
                slot_num = slot + 1
                if label:
                    text = f"{slot_num} {label[:CHARS_PER_CELL]}"
                else:
                    text = f"{slot_num} ·"   # middle dot = empty
                # Truncate so we don't bleed into next column.
                text = self._fit_text(draw, small, text,
                                      COL_W - 4)
                draw.text((x, y), text, fill=1, font=small)
        else:
            # Legacy big-page-name layout (back-compat for callers
            # that don't pass verticals).
            draw.text((2, 1), "PAGE", fill=1, font=small)
            name_text = (page_name or "").upper()
            chosen_font = self._huge_font
            for font in (self._huge_font, self._big_font,
                         self._med_font):
                try:
                    bb = draw.textbbox((0, 0), name_text, font=font)
                    tw = bb[2] - bb[0]
                except Exception:
                    tw = 100
                if tw <= DISPLAY_WIDTH - 8:
                    chosen_font = font
                    break
            name_fit = self._fit_text(draw, chosen_font, name_text,
                                      DISPLAY_WIDTH - 8)
            try:
                bb2 = draw.textbbox((0, 0), name_fit, font=chosen_font)
                tw2 = bb2[2] - bb2[0]
                th2 = bb2[3] - bb2[1]
                ty = max(12, (DISPLAY_HEIGHT - th2) // 2 - 6)
                tx = max(4, (DISPLAY_WIDTH - tw2) // 2)
            except Exception:
                tx, ty = 8, 14
            draw.text((tx, ty), name_fit, fill=1, font=chosen_font)
            if last_vertical:
                last_text = f"^ {last_vertical}"
                last_fit = self._fit_text(draw, self._med_font,
                                          last_text, DISPLAY_WIDTH - 8)
                draw.text((4, DISPLAY_HEIGHT - 14), last_fit, fill=1,
                          font=self._med_font)

        arr = np.array(img, dtype=np.uint8)
        return pack_1bpp(arr)

    # ── RIGHT screen: BROWSE-mode library nav ─────────────────────────
    def render_right_browse(
        self,
        items: list,
        cursor_idx: int = 0,
        folder_label: str = "",
        total_count: int = 0,
    ) -> bytes:
        """Browse-mode right OLED: a scrolling file list anchored on the
        cursor. ~5 visible rows with the selected row highlighted via
        inverted video (white bar, black text).

        Layout (256x64):
            ┌──────────────────────────────────────────────────┐
            │ FOLDER NAME            (3/127)                   │ y=1  small
            │ ─────────────────────────────────                 │ y=12 divider
            │   03  Charlotte Sartre, Kristy Black.mp4         │ y=14
            │ █ 04  Blanche Bradburry.mp4                      │ y=24 SELECTED
            │   05  Anna De Ville, Brittany Bardot...          │ y=34
            │   06  Linda Sweet, Rebecca Black.mp4             │ y=44
            │   07  ...                                        │ y=54
            └──────────────────────────────────────────────────┘

        Args:
          items: list[str] of display names (already pre-formatted, no
                 path). Caller picks the slice; we just render what we
                 get.
          cursor_idx: index INTO items[] of the selected row (NOT the
                 global library index). Caller does the windowing
                 math so this renderer stays dumb.
          folder_label: shown top-left, ellipsized to fit.
          total_count: shown top-right as "(globalIdx+1/total)" — pass
                 the GLOBAL position so the user sees their place in
                 the whole library, not just the visible window.
        """
        img = Image.new("1", (DISPLAY_WIDTH, DISPLAY_HEIGHT), 0)
        draw = ImageDraw.Draw(img)
        small = self._text._font  # PIL 6x11 default

        # ── Header row: folder name + position counter ───────────────
        folder_txt = (folder_label or "")[:36]
        draw.text((2, 1), folder_txt, fill=1, font=small)
        if total_count > 0:
            # Show GLOBAL position. cursor_idx is local-to-items[],
            # but the caller passes the global index in total_count
            # via tuple convention below -- simpler to accept a tuple.
            pass
        # (positional counter rendering moved into caller's
        # folder_label string -- cleaner separation.)

        # ── Divider line ─────────────────────────────────────────────
        draw.line((0, 12, DISPLAY_WIDTH - 1, 12), fill=1)

        # ── List rows ────────────────────────────────────────────────
        # PIL default font line ~11px. 5 rows from y=14 to y=64 with
        # row_h=10 fits perfectly.
        row_h = 10
        top_y = 14
        max_rows = (DISPLAY_HEIGHT - top_y) // row_h
        # Truncate to what fits.
        visible = items[:max_rows] if items else []
        for i, name in enumerate(visible):
            y = top_y + i * row_h
            is_cursor = (i == cursor_idx)
            text = str(name or "")[:42]
            if is_cursor:
                # Inverted highlight bar across the full width.
                draw.rectangle(
                    (0, y - 1, DISPLAY_WIDTH - 1, y + row_h - 2),
                    fill=1,
                )
                draw.text((4, y), text, fill=0, font=small)
            else:
                draw.text((4, y), text, fill=1, font=small)

        arr = np.array(img, dtype=np.uint8)
        return pack_1bpp(arr)

    # ── helpers ──────────────────────────────────────────────────────
    def _draw_pad_grid(self, draw, pad_states):
        # Same geometry as PadMapRenderer.render but draws onto an
        # existing canvas so we can composite a bank label over it.
        margin = self._pads.margin
        cell_gap = self._pads.cell_gap
        avail_w = DISPLAY_WIDTH - 2 * margin
        avail_h = DISPLAY_HEIGHT - 2 * margin
        cell_w = (avail_w - 3 * cell_gap) // 4
        cell_h = (avail_h - 3 * cell_gap) // 4
        cell = min(cell_w, cell_h)
        grid_w = 4 * cell + 3 * cell_gap
        grid_h = 4 * cell + 3 * cell_gap
        # Bias the grid LEFT so the bank label fits cleanly top-right.
        ox = margin
        oy = (DISPLAY_HEIGHT - grid_h) // 2

        for label in range(1, 17):
            z = label - 1
            row_from_bottom = z // 4
            col = z % 4
            row_from_top = 3 - row_from_bottom
            x0 = ox + col * (cell + cell_gap)
            y0 = oy + row_from_top * (cell + cell_gap)
            x1 = x0 + cell - 1
            y1 = y0 + cell - 1
            r, g, b = pad_states[z]
            lit = (r + g + b) > PadMapRenderer.LIT_THRESHOLD
            if lit:
                draw.rectangle((x0, y0, x1, y1), fill=1, outline=1)
            else:
                draw.rectangle((x0, y0, x1, y1), fill=0, outline=1)

    def _fit_text(self, draw, font, text: str, max_px: int) -> str:
        """Truncate text with an ellipsis so it fits within max_px."""
        if not text:
            return ""
        try:
            bb = draw.textbbox((0, 0), text, font=font)
            w = bb[2] - bb[0]
        except Exception:
            return text
        if w <= max_px:
            return text
        # Binary-chop until it fits, append an ellipsis.
        lo, hi = 0, len(text)
        ell = "..."
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = text[:mid] + ell
            try:
                bb = draw.textbbox((0, 0), candidate, font=font)
                cw = bb[2] - bb[0]
            except Exception:
                cw = max_px + 1
            if cw <= max_px:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + ell


# ─────────────────────────────────────────────────────────────────────
#  Smoke-test entry point
# ─────────────────────────────────────────────────────────────────────
def _self_test() -> None:
    """Sanity check the bit-reversal + packing without touching hardware."""
    assert _BIT_REVERSE_LUT[0x80] == 0x01
    assert _BIT_REVERSE_LUT[0x01] == 0x80
    assert _BIT_REVERSE_LUT[0xAA] == 0x55
    # round-trip
    src = bytes(range(256))
    rev = _bit_reverse_framebuffer(src)
    rev2 = _bit_reverse_framebuffer(rev)
    assert rev2 == src, "bit-reverse not involutive"

    # text renderer produces correct byte count
    tr = TextRenderer()
    fb = tr.render_lines(["hello", "maschine"])
    assert len(fb) == FRAMEBUFFER_BYTES, f"text fb wrong size: {len(fb)}"

    # pad renderer
    pads = [(0, 0, 0)] * 16
    pads[0] = (255, 0, 0)
    pads[5] = (0, 255, 0)
    pr = PadMapRenderer()
    fb = pr.render(pads)
    assert len(fb) == FRAMEBUFFER_BYTES, f"pad fb wrong size: {len(fb)}"

    # status renderer
    sr = StatusRenderer()
    fb_l = sr.render_left("midnight_run_v3_final_final.mp4", 128.5, 7)
    fb_r = sr.render_right(pads, active_bank="A")
    assert len(fb_l) == FRAMEBUFFER_BYTES
    assert len(fb_r) == FRAMEBUFFER_BYTES

    # ── thumbnail renderer ───────────────────────────────────────────
    # Build a throwaway JPEG so we exercise the real load → dither path.
    import os
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="mk2oled_selftest_")
    jpg_path = os.path.join(tmp_dir, "fake_clip.jpg")
    # A non-square gradient so letterboxing + aspect-fit actually engage.
    grad = np.tile(
        np.linspace(0, 255, 320, dtype=np.uint8), (180, 1)
    )
    Image.fromarray(grad, mode="L").save(jpg_path, "JPEG")
    thumb = ThumbnailRenderer()
    fb_t = thumb.render(jpg_path)
    assert len(fb_t) == FRAMEBUFFER_BYTES, f"thumb fb wrong size: {len(fb_t)}"
    # Missing file → blank framebuffer, still exactly 2048 bytes.
    fb_missing = thumb.render(os.path.join(tmp_dir, "does_not_exist.jpg"))
    assert len(fb_missing) == FRAMEBUFFER_BYTES
    assert fb_missing == b"\x00" * FRAMEBUFFER_BYTES, "missing thumb not blank"

    # composited left screen with thumbnail
    fb_lt = sr.render_left_with_thumb(jpg_path, "midnight_run_v3.mp4", 128.5)
    assert len(fb_lt) == FRAMEBUFFER_BYTES, f"left+thumb fb wrong size: {len(fb_lt)}"

    # ── activity renderer ────────────────────────────────────────────
    act = ActivityRenderer()
    fb_a_empty = act.render()
    assert len(fb_a_empty) == FRAMEBUFFER_BYTES, "empty activity fb wrong size"
    # Push more than capacity to exercise the ring-buffer wrap.
    for i in range(150):
        act.push_sample((i % 20) / 19.0)
    fb_a = act.render()
    assert len(fb_a) == FRAMEBUFFER_BYTES, f"activity fb wrong size: {len(fb_a)}"
    act.push_sample(float("nan"))   # NaN guard
    act.push_sample(5.0)            # clamp guard
    assert len(act.render()) == FRAMEBUFFER_BYTES

    # ── beat pulse renderer ──────────────────────────────────────────
    bp = BeatPulseRenderer()
    fb_p_rest = bp.render()         # at rest (never triggered)
    assert len(fb_p_rest) == FRAMEBUFFER_BYTES, "rest pulse fb wrong size"
    bp.trigger()
    fb_p_hit = bp.render()         # full-size right after trigger
    assert len(fb_p_hit) == FRAMEBUFFER_BYTES, f"pulse fb wrong size: {len(fb_p_hit)}"

    # tidy up the temp JPEG
    try:
        os.remove(jpg_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print("maschine_mk2_oled self-test OK")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _self_test()
