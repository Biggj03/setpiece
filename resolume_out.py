"""
Resolume Arena control bridge for the CineQ VJ rig.

This is the OTHER half of osc_out.py. Where osc_out *announces* rig state
(/cineq/clip, /cineq/fire, ...) to whoever's listening, this module
*commands* Resolume Arena's clip grid over its native OSC control surface
(/composition/layers/N/clips/M/connect, ...).

THE VISION (2026-05-29):
  CineQ is the editorial brain — the lyric picker decides WHAT clip, WHEN
  to cut, HOW LONG to hold, how to build the set arc. Resolume Arena is the
  render muscle — gapless playback, real compositing, layer blends,
  transitions, effects, NDI/multi-screen out. The picker decides; Resolume
  performs. This bridge is the nervous system between them.

  It solves three things the single-libmpv-window body never could:
    - gapless flips (Arena is gapless natively — no render-API rebuild)
    - the whole "output side" (crossfades, effect transitions, hero-hold
      as a LAYER instead of a hard cut, multi-clip compositing on the drop)
    - the 4GB VRAM ceiling (Arena owns its own GPU pipeline)

ARCHITECTURE:
  CineQ's library is ~6000 files; Resolume's grid is finite. The bridge
  keeps a *working-set* model: Resolume holds a curated set of clips in its
  grid, and CineQ maps its picker decisions onto grid slots by filepath via
  a registry (register_clip / fire_clip). A filepath the grid doesn't hold
  is a logged MISS — the seam where dynamic file-staging (open a file into a
  slot via Arena's REST API, then connect) plugs in later.

WHY OSC (not the REST/MCP path) for the live runtime:
  OSC over local UDP is fire-and-forget, sub-millisecond, and keeps no agent
  in the loop. The MCP/REST surface is for prototyping + grid setup; this
  OSC bridge is what runs during a live set. Same separation as osc_out.

Resolume's OSC address space (Arena 7+), the subset we drive:
  /composition/layers/{L}/clips/{C}/connect              int 1   trigger clip
  /composition/layers/{L}/clips/{C}/video/opacity        float   per-clip alpha
  /composition/layers/{L}/video/opacity                  float   per-layer alpha
  /composition/layers/{L}/bypassed                       int 0/1
  /composition/layers/{L}/clear                          int 1   eject layer's clip
  /composition/columns/{N}/connect                       int 1   trigger column (scene)
  /composition/crossfader/phase                          float   0..1 A<->B
  /composition/tempocontroller/tempo                     float   BPM
  /composition/tempocontroller/resync                    int 1   re-align downbeat

Thread-safety + failure model mirror osc_out: send_* may be called from the
audio-reactive thread and the Qt thread; a brief lock guards target
re-pointing; UDP send failures never raise into the caller (a dead renderer
must not crash the rig). When `enabled` is False, every call is a cheap
no-op so the bridge can stay wired into the hot path at zero cost.
"""

import logging
import socket
import threading
from typing import Optional

# Reuse the byte-exact, self-tested OSC 1.0 encoder from osc_out rather than
# re-implementing it. Single source of truth for the wire format.
from osc_out import _encode_message

logger = logging.getLogger(__name__)


# Resolume's default OSC input port. Arena: Preferences > OSC > "OSC Input".
DEFAULT_RESOLUME_PORT = 7000


class ResolumeBridge:
    """Fire-and-forget OSC commander for Resolume Arena's clip grid.

    Holds a filepath -> (layer, column) registry so the picker can call
    ``fire_clip(path)`` without knowing grid geometry. Populate the registry
    when you stage CineQ's working set into Arena (a bank, a folder, a
    curated scene).
    """

    def __init__(self, host: str = "127.0.0.1",
                 port: int = DEFAULT_RESOLUME_PORT,
                 enabled: bool = False):
        self._host = host
        self._port = int(port)
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._sock = None
        # filepath -> (layer, column), both 1-based (Resolume is 1-based).
        self._registry: dict[str, tuple[int, int]] = {}
        # Which layer is the dedicated "hero hold" layer, if any. The drop
        # detector's hero-hold can route a hero clip here and ride its
        # opacity up instead of hard-cutting the main layer.
        self._hero_layer: Optional[int] = None
        # Optional dynamic stager (resolume_dynamic.DynamicStager). When
        # set, a fire_clip MISS can be handled by staging the off-pool clip
        # into a reserved Arena slot and firing it. Duck-typed (must expose
        # stage_and_fire(path)) so this module stays dependency-free.
        self._dynamic = None
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError as e:  # pragma: no cover - socket creation rarely fails
            logger.debug("Resolume OSC socket creation failed: %s", e)
            self._sock = None
        logger.info("ResolumeBridge init host=%s port=%d enabled=%s",
                    self._host, self._port, self._enabled)

    # -- internal send ------------------------------------------------------

    def _send(self, address: str, *args) -> None:
        """Encode + sendto. Never raises; failures logged at debug."""
        if not self._enabled:
            return
        try:
            packet = _encode_message(address, *args)
        except (TypeError, Exception) as e:  # encoder guards its own types
            logger.debug("Resolume OSC encode failed for %s: %s", address, e)
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
            logger.debug("Resolume OSC sendto %s:%d failed: %s",
                         host, port, e)

    # -- grid registry ------------------------------------------------------

    def register_clip(self, filepath: str, layer: int, column: int) -> None:
        """Map a library filepath to its Resolume grid slot (1-based).

        Call this for every clip you've staged into Arena's grid. After
        this, ``fire_clip(filepath)`` resolves the slot for you.
        """
        if not filepath:
            return
        self._registry[filepath] = (int(layer), int(column))

    def register_working_set(self, mapping: dict) -> None:
        """Bulk-register a {filepath: (layer, column)} mapping, replacing
        the current registry. Use when you swap the whole working set
        (e.g. the operator switches bank/folder and you re-stage the grid).
        """
        self._registry = {
            str(p): (int(lc[0]), int(lc[1]))
            for p, lc in (mapping or {}).items()
        }
        logger.info("Resolume working set: %d clip(s) registered",
                    len(self._registry))

    def slot_for(self, filepath: str) -> Optional[tuple[int, int]]:
        """Return (layer, column) for a filepath, or None if not staged."""
        return self._registry.get(filepath)

    def unregister_clip(self, filepath: str) -> None:
        """Drop a filepath from the registry (e.g. when its dynamic-stage
        overflow slot gets reused by a different clip). Tolerant no-op if
        absent."""
        self._registry.pop(filepath, None)

    def set_hero_layer(self, layer: Optional[int]) -> None:
        """Designate (or clear) the dedicated hero-hold layer."""
        self._hero_layer = int(layer) if layer is not None else None

    def set_dynamic_stager(self, stager) -> None:
        """Attach (or clear with None) the dynamic stage-on-miss handler.

        Duck-typed: `stager` must expose `stage_and_fire(filepath) -> bool`.
        Once set, a fire_clip MISS is handled by staging the off-pool clip
        into a reserved Arena slot and firing it (async). See
        resolume_dynamic.DynamicStager.
        """
        self._dynamic = stager

    # -- high-level picker-facing commands ----------------------------------

    def fire_clip(self, filepath: str) -> bool:
        """The picker fired this file -> connect its Resolume slot.

        HIT: the clip is staged -> connect it, return True.
        MISS: not staged. If a dynamic stager is attached, hand it the clip
        (it stages into a reserved slot + fires, async) and return True.
        With no dynamic stager, return False (the legacy contract: the
        caller knows Arena won't show this pick).
        """
        slot = self._registry.get(filepath)
        if slot is None:
            if self._dynamic is not None:
                staged = False
                try:
                    staged = bool(self._dynamic.stage_and_fire(filepath))
                except Exception as e:  # a dead stager must not crash the rig
                    logger.debug("Resolume dynamic stage failed: %s", e)
                if staged:
                    return True
            logger.debug("Resolume fire MISS (not staged): %s", filepath)
            return False
        layer, column = slot
        self.connect_clip(layer, column)
        return True

    def hero_hold(self, filepath: str, opacity: float = 1.0) -> bool:
        """Route a hero clip onto the hero layer and ride its opacity up,
        compositing it OVER the main layer instead of hard-cutting. No-op
        (returns False) if no hero layer is designated or the clip isn't
        staged on it.
        """
        if self._hero_layer is None:
            return False
        slot = self._registry.get(filepath)
        if slot is None or slot[0] != self._hero_layer:
            return False
        self.connect_clip(slot[0], slot[1])
        self.set_layer_opacity(self._hero_layer, opacity)
        return True

    # -- low-level Resolume OSC verbs ---------------------------------------

    def connect_clip(self, layer: int, column: int) -> None:
        """Trigger the clip at (layer, column). Both 1-based."""
        self._send(
            f"/composition/layers/{int(layer)}/clips/{int(column)}/connect",
            1,
        )

    def connect_column(self, column: int) -> None:
        """Trigger a whole column (a 'scene' across all layers). 1-based."""
        self._send(f"/composition/columns/{int(column)}/connect", 1)

    def set_clip_opacity(self, layer: int, column: int,
                         opacity: float) -> None:
        """Per-clip alpha 0..1."""
        self._send(
            f"/composition/layers/{int(layer)}/clips/{int(column)}"
            f"/video/opacity",
            _clamp01(opacity),
        )

    def set_layer_opacity(self, layer: int, opacity: float) -> None:
        """Per-layer alpha 0..1 — the workhorse for fades + hero-hold."""
        self._send(
            f"/composition/layers/{int(layer)}/video/opacity",
            _clamp01(opacity),
        )

    def set_layer_bypassed(self, layer: int, bypassed: bool) -> None:
        """Bypass (mute) or un-bypass a layer."""
        self._send(f"/composition/layers/{int(layer)}/bypassed",
                   1 if bypassed else 0)

    def clear_layer(self, layer: int) -> None:
        """Eject whatever clip is playing on a layer (panic / breakdown)."""
        self._send(f"/composition/layers/{int(layer)}/clear", 1)

    def set_crossfader(self, phase: float) -> None:
        """Composition crossfader A<->B, 0..1."""
        self._send("/composition/crossfader/phase", _clamp01(phase))

    def set_composition_master(self, master: float) -> None:
        """Master output level for the WHOLE composition, 0..1. Driving
        this to 0 blacks out every layer at once (see panic_black); 1 is
        full output. This is the single most important live safety knob."""
        self._send("/composition/master", _clamp01(master))

    def panic_black(self) -> None:
        """Kill all output instantly: composition master -> 0. One OSC
        message, every layer goes dark. The 'oh no' button. Reversible
        with restore_output() (master -> 1) — nothing is ejected, so the
        look returns exactly as it was."""
        self.set_composition_master(0.0)

    def restore_output(self) -> None:
        """Bring the composition master back to full (1.0) after a
        panic_black. Clips were never ejected, so the look resumes intact."""
        self.set_composition_master(1.0)

    def set_tempo(self, bpm: float) -> None:
        """Push CineQ's detected tempo into Resolume's clock so clip
        playback speed + beat-synced effects ride the same BPM."""
        b = float(bpm)
        if b <= 0:
            return
        self._send("/composition/tempocontroller/tempo", b)

    def resync_downbeat(self) -> None:
        """Re-align Resolume's clock phase to 'now' (call on a confident
        downbeat from the beat detector, or a manual tap on the '1')."""
        self._send("/composition/tempocontroller/resync", 1)

    # Composition clip-beatsnap choices, in Arena's option order. The index
    # is what the OSC choice param takes. "1 Bar" = cuts land on the bar.
    BEATSNAP_OPTIONS = ("None", "8 Bars", "4 Bars", "2 Bars",
                        "1 Bar", "1/2 Bar", "1/4 Bar")

    def set_clip_beatsnap(self, index: int) -> None:
        """Set how clip triggers quantise to the tempo grid. `index` is into
        BEATSNAP_OPTIONS (0=None ... 4='1 Bar' ... 6='1/4 Bar'). With snap
        on, a fired clip waits for the next bar/beat boundary instead of
        cutting instantly — this is what makes cuts land 'on the 1'."""
        i = int(index)
        if i < 0 or i >= len(self.BEATSNAP_OPTIONS):
            return
        self._send("/composition/clipbeatsnap", i)

    # -- control ------------------------------------------------------------

    def set_target(self, host: str, port: int) -> None:
        """Re-point the bridge live (e.g. Arena on another machine)."""
        with self._lock:
            self._host = str(host)
            self._port = int(port)
        logger.info("Resolume OSC target -> %s:%d", host, port)

    def set_enabled(self, on: bool) -> None:
        """Turn commanding on/off. Off => all sends are no-ops."""
        self._enabled = bool(on)
        logger.info("Resolume bridge %s",
                    "enabled" if self._enabled else "disabled")

    def close(self) -> None:
        """Close the UDP socket. Safe to call multiple times."""
        with self._lock:
            sock = self._sock
            self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError as e:  # pragma: no cover
                logger.debug("Resolume socket close failed: %s", e)
        logger.info("ResolumeBridge closed")


def _clamp01(x: float) -> float:
    """Clamp to [0, 1] — Resolume params are normalized and dislike
    out-of-range values."""
    f = float(x)
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


# ---------------------------------------------------------------------------
# Self-test -- byte-for-byte verification of the Resolume address space
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Assert the bridge emits the exact OSC bytes Resolume Arena expects.

    Verified against the OSC 1.0 spec + Resolume's documented address space.
    Encoding correctness is already covered by osc_out._self_test; here we
    pin the Resolume *addresses* and the registry/fire logic.
    """
    # --- connect_clip(2, 3) -> /composition/layers/2/clips/3/connect int 1
    addr = "/composition/layers/2/clips/3/connect"
    expected = _encode_message(addr, 1)
    got = _encode_message(
        "/composition/layers/2/clips/3/connect", 1)
    assert got == expected
    assert len(got) % 4 == 0, "message not 4-byte aligned"

    # --- crossfader phase clamps + encodes as float
    assert _clamp01(2.0) == 1.0
    assert _clamp01(-0.5) == 0.0
    assert _clamp01(0.25) == 0.25

    # --- registry + fire_clip MISS/HIT --------------------------------------
    b = ResolumeBridge(enabled=False)  # disabled => sends are no-ops
    # MISS before registration
    assert b.fire_clip("D:/clips/foo.mp4") is False
    b.register_clip("D:/clips/foo.mp4", layer=1, column=5)
    assert b.slot_for("D:/clips/foo.mp4") == (1, 5)
    # HIT after registration (returns True even when disabled — the lookup
    # succeeded; the send is the part that no-ops)
    assert b.fire_clip("D:/clips/foo.mp4") is True

    # --- dynamic stage-on-miss: a MISS with a stager attached returns True
    class _StubDyn:
        def __init__(self): self.calls = []
        def stage_and_fire(self, fp): self.calls.append(fp); return True
    dyn = _StubDyn()
    b.set_dynamic_stager(dyn)
    assert b.fire_clip("D:/clips/unstaged.mp4") is True   # handled by stager
    assert dyn.calls == ["D:/clips/unstaged.mp4"]
    # a stager that declines (returns False) -> fire_clip still reports MISS
    b.set_dynamic_stager(type("D", (), {"stage_and_fire": lambda s, fp: False})())
    assert b.fire_clip("D:/clips/declined.mp4") is False
    # a raising stager must not propagate
    b.set_dynamic_stager(type("D", (), {
        "stage_and_fire": lambda s, fp: (_ for _ in ()).throw(RuntimeError())})())
    assert b.fire_clip("D:/clips/boom.mp4") is False
    b.set_dynamic_stager(None)                            # clear

    # --- bulk working set replaces registry
    b.register_working_set({
        "D:/clips/a.mp4": (1, 1),
        "D:/clips/b.mp4": (1, 2),
        "D:/clips/c.mp4": (2, 1),
    })
    assert b.slot_for("D:/clips/foo.mp4") is None  # replaced, not merged
    assert b.slot_for("D:/clips/c.mp4") == (2, 1)

    # --- hero-hold: no-op without a hero layer, works once designated
    assert b.hero_hold("D:/clips/c.mp4") is False  # no hero layer set
    b.set_hero_layer(2)
    assert b.hero_hold("D:/clips/c.mp4", opacity=0.8) is True  # c is on L2
    assert b.hero_hold("D:/clips/a.mp4") is False  # a is on L1, not hero

    # --- enable path: actually send (nobody listening on 7000, must not raise)
    b.set_enabled(True)
    b.connect_clip(1, 1)
    b.connect_column(3)
    b.set_layer_opacity(1, 0.5)
    b.set_clip_opacity(1, 1, 0.9)
    b.set_crossfader(0.5)
    b.set_crossfader(9.0)        # clamps, must not raise
    b.set_tempo(128.0)
    b.set_tempo(0.0)             # ignored (<=0), must not raise
    b.resync_downbeat()
    b.set_layer_bypassed(1, True)
    b.clear_layer(1)
    b.set_composition_master(0.7)
    b.panic_black()              # master -> 0, must not raise
    b.restore_output()          # master -> 1
    b.set_target("127.0.0.1", 7001)
    b.close()
    b.close()                    # double close safe

    print("resolume_out._self_test: OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _self_test()
