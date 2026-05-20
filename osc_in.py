"""
OSC listener for the Setpiece VJ rig.

Counterpart to osc_out.py -- decodes incoming OSC 1.0 messages from
UDP and dispatches to registered handlers. NO external OSC library;
the protocol is simple enough (~50 lines) that we keep it stdlib-only
to match osc_out.py and avoid a pip dep.

Designed for the stem-separation daemon pipeline:
    stem_daemon.py  --OSC over UDP-->  OSCListener  -->  handler(s)

Multiple handlers per address pattern are supported. Address matching
is exact (no OSC pattern globbing) -- keep it simple and explicit.

Usage:
    listener = OSCListener(host="127.0.0.1", port=7400)
    listener.on("/stem/drums/onset", lambda strength: ...)
    listener.start()  # spawns daemon thread
    ...
    listener.stop()

Thread-safety:
    on() is safe to call any time (handlers dict is replaced atomically).
    handler callbacks fire on the listener daemon thread -- handlers
    must be quick / threadsafe / marshal to Qt themselves as needed.
"""
from __future__ import annotations

import logging
import socket
import struct
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ── OSC 1.0 decoding ─────────────────────────────────────────────────

def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    """Read an OSC string starting at `offset`. Returns (string, new_offset).
    OSC strings are null-terminated then padded to 4-byte boundary."""
    end = data.find(b"\x00", offset)
    if end < 0:
        raise ValueError("OSC string: missing null terminator")
    s = data[offset:end].decode("utf-8", errors="replace")
    # Advance past the null + padding to next 4-byte boundary
    end += 1
    pad = (4 - (end - offset) % 4) % 4
    return s, end + pad


def _read_int32(data: bytes, offset: int) -> tuple[int, int]:
    val = struct.unpack(">i", data[offset:offset + 4])[0]
    return val, offset + 4


def _read_float32(data: bytes, offset: int) -> tuple[float, int]:
    val = struct.unpack(">f", data[offset:offset + 4])[0]
    return val, offset + 4


def decode_message(data: bytes) -> tuple[str, list]:
    """Parse a single OSC message. Returns (address, [args]).
    Raises ValueError on malformed packets."""
    address, off = _read_string(data, 0)
    type_tags, off = _read_string(data, off)
    if not type_tags.startswith(","):
        raise ValueError(f"OSC type-tags missing comma: {type_tags!r}")
    args: list = []
    for tag in type_tags[1:]:
        if tag == "i":
            v, off = _read_int32(data, off)
            args.append(v)
        elif tag == "f":
            v, off = _read_float32(data, off)
            args.append(v)
        elif tag == "s":
            v, off = _read_string(data, off)
            args.append(v)
        else:
            # Unknown type: skip the rest -- safer than guessing widths.
            logger.debug(f"OSC: unsupported type tag {tag!r}, "
                         f"skipping rest of message")
            break
    return address, args


# ── Listener ─────────────────────────────────────────────────────────

class OSCListener:
    """UDP OSC listener with address->handler dispatch.

    Bind to a host:port, register handlers via on(address, callback),
    start() the daemon thread. Each incoming packet is decoded and
    dispatched to any handler whose registered address matches exactly.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7400):
        self.host = host
        self.port = port
        self._handlers: dict[str, list[Callable]] = {}
        self._lock = threading.Lock()
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    def on(self, address: str, handler: Callable) -> None:
        """Register handler(s args*) for `address`. Multiple handlers
        per address are supported; called in registration order."""
        with self._lock:
            self._handlers.setdefault(address, []).append(handler)

    def start(self) -> bool:
        """Bind socket + spawn listener thread. Returns False on bind
        failure (port in use, etc.). Idempotent."""
        if self._running.is_set():
            return True
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self.host, self.port))
            self._sock.settimeout(0.25)
        except OSError as e:
            logger.error(f"OSCListener bind failed {self.host}:{self.port}: {e}")
            self._sock = None
            return False
        self._running.set()
        self._thread = threading.Thread(
            target=self._loop, name="osc-listener", daemon=True
        )
        self._thread.start()
        logger.info(f"OSCListener: listening on {self.host}:{self.port}")
        return True

    def stop(self) -> None:
        self._running.clear()
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        self._sock = None
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None

    def is_running(self) -> bool:
        return self._running.is_set()

    def _loop(self) -> None:
        """Receive packets until stop(); decode + dispatch each."""
        while self._running.is_set() and self._sock is not None:
            try:
                data, _addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                continue
            try:
                address, args = decode_message(data)
            except Exception as e:
                logger.debug(f"OSCListener: decode failed: {e}")
                continue
            with self._lock:
                handlers = list(self._handlers.get(address, []))
            for h in handlers:
                try:
                    h(*args)
                except Exception as e:
                    logger.warning(
                        f"OSCListener: handler for {address} raised: {e}",
                        exc_info=True,
                    )
