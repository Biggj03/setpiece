"""
OSC broadcaster for the Setpiece VJ rig.

Broadcasts live rig state (BPM, beats, current clip, fires, crossfade
position) as OSC 1.0 messages over UDP so external software can sync to
it: Resolume, TouchDesigner, QLC+, lighting consoles, other VJ tools.

NO external OSC library. Minimal OSC 1.0 encoding is implemented here --
it's just an address string + type-tag string + args, each chunk
null-padded to a 4-byte boundary, big-endian. See ``_encode_message``.

Thread-safety: ``send_*`` is called from both the audio-reactive thread
and the Qt thread. A small ``threading.Lock`` guards socket re-pointing
(``set_target``) against in-flight sends. The ``sendto`` call itself is
atomic for the small (<<64KB, in practice <100 byte) datagrams we emit,
so the lock is held only briefly and never across blocking work.

UDP send failures never raise into the caller -- they're logged at debug
and swallowed. A dead lighting console must not crash the VJ rig.
"""

import logging
import socket
import struct
import threading

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OSC 1.0 encoding (from scratch -- stdlib only)
# ---------------------------------------------------------------------------

def _pad4(data: bytes) -> bytes:
    """Null-pad ``data`` up to the next multiple of 4 bytes."""
    rem = len(data) % 4
    if rem:
        data += b"\x00" * (4 - rem)
    return data


def _encode_string(s: str) -> bytes:
    """OSC string: UTF-8 bytes + null terminator, null-padded to 4 bytes.

    Note the null terminator is always added *before* padding, so a string
    whose UTF-8 length is already a multiple of 4 still gets 4 null bytes.
    """
    return _pad4(s.encode("utf-8") + b"\x00")


def _encode_int32(i: int) -> bytes:
    """OSC int32: 4-byte big-endian signed integer."""
    return struct.pack(">i", int(i))


def _encode_float32(f: float) -> bytes:
    """OSC float32: 4-byte big-endian IEEE 754."""
    return struct.pack(">f", float(f))


def _encode_message(address: str, *args) -> bytes:
    """Build a complete OSC message.

    Supported arg types: int -> 'i', float -> 'f', str -> 's'.
    Layout: <address string><,typetags string><arg blobs...>
    """
    type_tags = ","
    blobs = []
    for a in args:
        if isinstance(a, bool):
            # bool is a subclass of int -- guard so True/False don't sneak
            # through as int32. We don't emit OSC T/F, so reject loudly.
            raise TypeError("OSC bool args not supported by this encoder")
        elif isinstance(a, int):
            type_tags += "i"
            blobs.append(_encode_int32(a))
        elif isinstance(a, float):
            type_tags += "f"
            blobs.append(_encode_float32(a))
        elif isinstance(a, str):
            type_tags += "s"
            blobs.append(_encode_string(a))
        else:
            raise TypeError(f"unsupported OSC arg type: {type(a)!r}")
    return _encode_string(address) + _encode_string(type_tags) + b"".join(blobs)


# ---------------------------------------------------------------------------
# Broadcaster
# ---------------------------------------------------------------------------

class OSCBroadcaster:
    """Fire-and-forget OSC/UDP broadcaster for Setpiece rig state.

    When ``enabled`` is False every ``send_*`` is a cheap no-op (one bool
    check, no socket touch, no encoding work) so it can stay wired into
    the hot audio path with zero cost when nobody's listening.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9000,
                 enabled: bool = False):
        self._host = host
        self._port = int(port)
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._sock = None
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError as e:  # pragma: no cover - socket creation rarely fails
            logger.debug("OSC socket creation failed: %s", e)
            self._sock = None
        logger.info("OSCBroadcaster init host=%s port=%d enabled=%s",
                    self._host, self._port, self._enabled)

    # -- internal send ------------------------------------------------------

    def _send(self, address: str, *args) -> None:
        """Encode + sendto. Never raises; failures logged at debug."""
        if not self._enabled:
            return
        try:
            packet = _encode_message(address, *args)
        except (TypeError, struct.error) as e:
            logger.debug("OSC encode failed for %s: %s", address, e)
            return
        with self._lock:
            sock = self._sock
            host = self._host
            port = self._port
        if sock is None:
            return
        try:
            sock.sendto(packet, (host, port))
        except OSError as e:
            logger.debug("OSC sendto %s:%d failed: %s", host, port, e)

    # -- public send_* ------------------------------------------------------

    def send_bpm(self, bpm: float) -> None:
        """/setpiece/bpm <float>  -- current estimated tempo."""
        self._send("/setpiece/bpm", float(bpm))

    def send_beat(self, beat_index: int, bpm: float) -> None:
        """/setpiece/beat <int> <float>  -- fire on every detected beat."""
        self._send("/setpiece/beat", int(beat_index), float(bpm))

    def send_clip_change(self, filepath: str, name: str) -> None:
        """/setpiece/clip <string name>  -- a new clip was loaded.

        ``filepath`` is accepted for caller convenience / future use but
        only ``name`` is broadcast (keeps the packet small and tidy for
        downstream UIs).
        """
        self._send("/setpiece/clip", str(name))

    def send_fire(self, source: str, name: str) -> None:
        """/setpiece/fire <string source> <string name>  -- a clip/bank fired."""
        self._send("/setpiece/fire", str(source), str(name))

    def send_crossfade(self, position: float) -> None:
        """/setpiece/xfade <float 0..1>  -- crossfader position."""
        # Clamp -- downstream lighting maths hates out-of-range values.
        p = float(position)
        if p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0
        self._send("/setpiece/xfade", p)

    # -- control ------------------------------------------------------------

    def set_target(self, host: str, port: int) -> None:
        """Re-point the broadcaster live (e.g. user edits settings)."""
        with self._lock:
            self._host = str(host)
            self._port = int(port)
        logger.info("OSC target -> %s:%d", host, port)

    def set_enabled(self, on: bool) -> None:
        """Turn broadcasting on/off. Off => all send_* are no-ops."""
        self._enabled = bool(on)
        logger.info("OSC broadcasting %s", "enabled" if self._enabled else "disabled")

    def close(self) -> None:
        """Close the UDP socket. Safe to call multiple times."""
        with self._lock:
            sock = self._sock
            self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError as e:  # pragma: no cover
                logger.debug("OSC socket close failed: %s", e)
        logger.info("OSCBroadcaster closed")


# ---------------------------------------------------------------------------
# Self-test -- byte-for-byte verification against the OSC 1.0 spec
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Assert encoder output matches hand-computed OSC 1.0 bytes."""

    # --- /setpiece/bpm <float 120.0> -----------------------------------------
    # address "/setpiece/bpm" = 10 chars. +null = 11 bytes. pad to 12:
    #   2f 63 69 6e 65 71 2f 62 70 6d 00 00
    # typetag ",f" = 2 chars. +null = 3 bytes. pad to 4:
    #   2c 66 00 00
    # float32 120.0 big-endian = 42 f0 00 00
    expected_bpm = (
        b"\x2f\x63\x69\x6e\x65\x71\x2f\x62\x70\x6d\x00\x00"
        b"\x2c\x66\x00\x00"
        b"\x42\xf0\x00\x00"
    )
    got_bpm = _encode_message("/setpiece/bpm", 120.0)
    assert got_bpm == expected_bpm, (
        f"/setpiece/bpm mismatch:\n  got={got_bpm!r}\n  exp={expected_bpm!r}"
    )
    assert len(got_bpm) % 4 == 0, "message not 4-byte aligned"

    # --- string padding edge case: 4-char string still gets 4 null bytes --
    # "test" is exactly 4 bytes; OSC requires a null terminator, so the
    # encoded form is 8 bytes (4 data + 4 nulls), NOT 4.
    assert _encode_string("test") == b"\x74\x65\x73\x74\x00\x00\x00\x00", (
        "4-char string padding wrong"
    )
    # "abc" -> 3 data + 1 null = 4, already aligned.
    assert _encode_string("abc") == b"\x61\x62\x63\x00", "3-char string wrong"
    # "" -> 0 data + 1 null -> pad to 4.
    assert _encode_string("") == b"\x00\x00\x00\x00", "empty string wrong"

    # --- int32 ------------------------------------------------------------
    assert _encode_int32(1) == b"\x00\x00\x00\x01", "int32 1 wrong"
    assert _encode_int32(256) == b"\x00\x00\x01\x00", "int32 256 wrong"
    assert _encode_int32(-1) == b"\xff\xff\xff\xff", "int32 -1 wrong"

    # --- float32 ----------------------------------------------------------
    assert _encode_float32(1.0) == b"\x3f\x80\x00\x00", "float32 1.0 wrong"
    assert _encode_float32(0.0) == b"\x00\x00\x00\x00", "float32 0.0 wrong"

    # --- /setpiece/beat <int 4> <float 128.0> --------------------------------
    # address "/setpiece/beat" = 11 chars +null = 12, already aligned.
    # typetag ",if" = 3 chars +null = 4.
    # int32 4 = 00 00 00 04 ; float32 128.0 = 43 00 00 00
    expected_beat = (
        b"\x2f\x63\x69\x6e\x65\x71\x2f\x62\x65\x61\x74\x00"
        b"\x2c\x69\x66\x00"
        b"\x00\x00\x00\x04"
        b"\x43\x00\x00\x00"
    )
    got_beat = _encode_message("/setpiece/beat", 4, 128.0)
    assert got_beat == expected_beat, (
        f"/setpiece/beat mismatch:\n  got={got_beat!r}\n  exp={expected_beat!r}"
    )

    # --- /setpiece/fire <string> <string> ------------------------------------
    # address "/setpiece/fire" = 11 +null = 12. typetag ",ss" = 3 +null = 4.
    # "deck" = 4 data +4 null = 8. "vid" = 3 +1 null = 4.
    expected_fire = (
        b"\x2f\x63\x69\x6e\x65\x71\x2f\x66\x69\x72\x65\x00"
        b"\x2c\x73\x73\x00"
        b"\x64\x65\x63\x6b\x00\x00\x00\x00"
        b"\x76\x69\x64\x00"
    )
    got_fire = _encode_message("/setpiece/fire", "deck", "vid")
    assert got_fire == expected_fire, (
        f"/setpiece/fire mismatch:\n  got={got_fire!r}\n  exp={expected_fire!r}"
    )

    # --- bool must be rejected (it's an int subclass) ---------------------
    try:
        _encode_message("/x", True)
    except TypeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("bool arg should have raised TypeError")

    # --- broadcaster: disabled => no-op, no raise even with no listener ---
    b = OSCBroadcaster(host="127.0.0.1", port=9000, enabled=False)
    b.send_bpm(120.0)          # no-op
    b.send_beat(1, 120.0)      # no-op
    b.set_enabled(True)
    b.send_bpm(120.0)          # actually sends to 127.0.0.1:9000 (nobody home, fine)
    b.send_beat(1, 120.0)
    b.send_clip_change("/path/to/clip.mp4", "clip.mp4")
    b.send_fire("deck", "intro")
    b.send_crossfade(0.5)
    b.send_crossfade(2.0)      # clamps to 1.0, must not raise
    b.set_target("127.0.0.1", 9001)
    b.close()
    b.close()                  # double close must be safe

    print("osc_out._self_test: OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _self_test()
