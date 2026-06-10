"""
Dynamic stage-on-miss for the Setpiece -> Resolume bridge.

THE GAP THIS CLOSES:
  The show driver fires from the WHOLE library; Arena's grid only holds
  the working set staged for the current stretch of the show
  (resolume_stage). A fire that's in the staged pool lights Arena; one
  that ISN'T is a MISS (resolume_out.ResolumeBridge.fire_clip returns
  False) and nothing renders. So Arena shows only a SUBSET of the
  driver's choices.

  DynamicStager closes that seam: on a MISS, load that one off-pool clip
  into a reserved Arena slot over REST (~150ms), register it on the
  bridge, then connect it over OSC. Now every pick reaches Arena.

DESIGN (decided with the operator, 2026-05-30):
  - ASYNC. The REST open is ~150ms — far too slow for the audio/Qt fire
    path. stage_and_fire() hands the work to a single worker thread and
    returns instantly; the off-pool clip appears ~a beat late, never
    stuttering the rig.
  - COALESCING worker (pending depth 1). A flurry of misses can outpace
    REST. Rather than queue them all and fall behind, we keep only the
    NEWEST pending miss — by the time a stale one would load, the show
    has already moved on, so showing the latest is correct.
  - BOUNDED RING. Dynamic clips live in a fixed band of N columns (default
    8) on the staging layer, starting just past the staged pool's
    high-water column. Slots recycle oldest-first, so a multi-hour set
    never balloons Arena's grid. The band never overlaps the staged pool.

  Same discipline as resolume_out/osc_out: no-op when disabled, never
  raises into the caller (a dead renderer must not crash the rig).

GRID LAYOUT (v1 single-layer):
    columns:  1 .............. P | P+1 ... P+N
              [ staged song pool ] [ dynamic ring ]
  P = pool high-water (set per working set via set_pool_high_water). The
  ring is P+1 .. P+N on the same staging layer (default 1).
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_RING_SIZE = 8
DEFAULT_LAYER = 1


class DynamicStager:
    """Loads off-pool clips into a bounded ring of Arena slots on a MISS,
    then fires them. Wraps a persistent REST stager + the OSC bridge.

    Wire-up: created once when the working set first stages (so it shares the
    live ResolumeStager and knows the pool's column high-water), then
    handed to the bridge as its miss handler. fire path:

        bridge.fire_clip(path)  -> False (MISS)
        bridge.stage_on_miss(path) -> DynamicStager.stage_and_fire(path)
    """

    def __init__(self, stager, bridge, layer: int = DEFAULT_LAYER,
                 ring_size: int = DEFAULT_RING_SIZE,
                 pool_high_water: int = 0):
        self._stager = stager          # ResolumeStager (REST)
        self._bridge = bridge          # ResolumeBridge (OSC + registry)
        self._layer = int(layer)
        self._ring_size = max(1, int(ring_size))
        # Highest column the staged pool occupies. Ring lives after it.
        self._pool_high_water = max(0, int(pool_high_water))

        # Ring state — mutated ONLY by the worker thread, so no lock here.
        self._cursor = 0                          # next ring index to use
        self._slot_path: dict[int, str] = {}      # ring column -> filepath
        self._path_slot: dict[str, int] = {}      # filepath -> ring column

        # Coalescing worker. _pending holds the newest unprocessed path
        # (depth 1); guarded by _lock. _wake signals the worker.
        self._lock = threading.Lock()
        self._pending: Optional[str] = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

    # -- configuration ------------------------------------------------------

    def set_pool_high_water(self, col: int) -> None:
        """Set the staged pool's high-water column. The dynamic ring sits
        at col+1 .. col+ring_size, so it never collides with the pool.
        Call this each time a new working set stages its pool. Resets the
        ring (the previous set's dynamic clips are stale)."""
        self._pool_high_water = max(0, int(col))
        self._cursor = 0
        self._slot_path.clear()
        self._path_slot.clear()
        logger.debug("DynamicStager: pool high-water=%d, ring cols %d..%d",
                     self._pool_high_water, self._ring_base() + 1,
                     self._ring_base() + self._ring_size)

    def _ring_base(self) -> int:
        """Column just before the first ring slot."""
        return self._pool_high_water

    def ring_columns(self) -> list:
        """The reserved dynamic columns (1-based), for diagnostics/tests."""
        base = self._ring_base()
        return [base + 1 + i for i in range(self._ring_size)]

    # -- public fire path ---------------------------------------------------

    def stage_and_fire(self, filepath: str) -> bool:
        """Queue an off-pool clip to be staged + fired (async, non-blocking).

        Returns True if accepted for staging, False if it can't (no stager/
        bridge, empty path). Actual REST/OSC happens on the worker thread;
        this returns immediately so the hot path never blocks on the
        ~150ms REST open."""
        if not filepath or self._stager is None or self._bridge is None:
            return False
        self._ensure_worker()
        with self._lock:
            self._pending = filepath
        self._wake.set()
        return True

    # -- worker -------------------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._run, name="resolume-dynamic", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            while True:
                with self._lock:
                    path = self._pending
                    self._pending = None
                if path is None:
                    break
                try:
                    self._stage_and_fire_sync(path)
                except Exception as e:  # never let the worker die
                    logger.debug("DynamicStager worker error: %s", e)

    def stop(self) -> None:
        """Stop the worker thread. Safe to call multiple times."""
        self._stop.set()
        self._wake.set()

    # -- the actual work (also the synchronous entry point for tests) -------

    def _stage_and_fire_sync(self, filepath: str) -> Optional[tuple]:
        """Stage one clip into a ring slot and fire it. Returns (layer,
        column) on success, None on failure. Runs on the worker thread;
        also called directly by the self-test + live-verify (no thread)."""
        # Already live in the ring? Just re-fire it; no REST needed.
        existing = self._path_slot.get(filepath)
        if existing is not None:
            self._bridge.connect_clip(self._layer, existing)
            logger.debug("DynamicStager: re-fire ring clip L%dC%d %s",
                         self._layer, existing, filepath)
            return (self._layer, existing)

        col = self._claim_slot()
        ok = self._stager.open_into_slot(self._layer, col, filepath)
        if not ok:
            # Free the slot we claimed; the open failed so it holds nothing
            # new (the previous occupant, if any, was already evicted from
            # our maps in _claim_slot — accept the small inconsistency).
            logger.warning("DynamicStager: open failed L%dC%d %s",
                           self._layer, col, filepath)
            return None
        self._slot_path[col] = filepath
        self._path_slot[filepath] = col
        self._bridge.register_clip(filepath, self._layer, col)
        self._bridge.connect_clip(self._layer, col)
        logger.info("DynamicStager: staged+fired off-pool clip L%dC%d %s",
                    self._layer, col, filepath)
        return (self._layer, col)

    def _claim_slot(self) -> int:
        """Return the next ring column (1-based), recycling oldest. Evicts
        whatever path currently held that slot from the maps."""
        base = self._ring_base()
        col = base + 1 + (self._cursor % self._ring_size)
        self._cursor += 1
        old = self._slot_path.pop(col, None)
        if old is not None:
            self._path_slot.pop(old, None)
            # Also drop the evicted clip from the BRIDGE registry, else a
            # later fire of it would HIT a column that now holds a different
            # clip and show the wrong thing. Tolerant if the bridge predates
            # unregister_clip.
            try:
                self._bridge.unregister_clip(old)
            except AttributeError:
                pass
        return col


# ---------------------------------------------------------------------------
# Self-test -- offline, no Arena. Fakes the stager + bridge to pin the ring
# recycle / high-water / no-collision / re-fire logic.
# ---------------------------------------------------------------------------

class _FakeStager:
    def __init__(self, fail_paths=None):
        self.opens = []                       # (layer, col, path)
        self._fail = set(fail_paths or [])

    def open_into_slot(self, layer, col, path):
        self.opens.append((layer, col, path))
        return path not in self._fail


class _FakeBridge:
    def __init__(self):
        self.registered = {}                  # path -> (layer, col)
        self.connects = []                    # (layer, col)

    def register_clip(self, path, layer, col):
        self.registered[path] = (layer, col)

    def connect_clip(self, layer, col):
        self.connects.append((layer, col))


def _self_test() -> None:
    # --- ring sits AFTER the pool high-water, no collision ----------------
    st, br = _FakeStager(), _FakeBridge()
    d = DynamicStager(st, br, layer=1, ring_size=8, pool_high_water=36)
    assert d.ring_columns() == [37, 38, 39, 40, 41, 42, 43, 44], d.ring_columns()

    # --- first off-pool clip lands in the first ring column ---------------
    assert d._stage_and_fire_sync("D:/a.mp4") == (1, 37)
    assert br.registered["D:/a.mp4"] == (1, 37)
    assert br.connects[-1] == (1, 37)

    # --- distinct clips march across the ring -----------------------------
    for i, name in enumerate(["b", "c", "d", "e", "f", "g", "h"], start=1):
        layer, col = d._stage_and_fire_sync(f"D:/{name}.mp4")
        assert (layer, col) == (1, 37 + i), (name, layer, col)

    # ring is now full (37..44 hold a..h). Next clip recycles col 37.
    layer, col = d._stage_and_fire_sync("D:/i.mp4")
    assert (layer, col) == (1, 37), (layer, col)
    # the evicted path 'a' is gone from the maps
    assert "D:/a.mp4" not in d._path_slot
    assert d._slot_path[37] == "D:/i.mp4"

    # --- re-firing a clip still resident in the ring: no new REST open ----
    opens_before = len(st.opens)
    layer, col = d._stage_and_fire_sync("D:/i.mp4")
    assert (layer, col) == (1, 37)
    assert len(st.opens) == opens_before, "re-fire must not re-open"
    assert br.connects[-1] == (1, 37)

    # --- set_pool_high_water resets the ring + repositions it -------------
    d.set_pool_high_water(12)
    assert d.ring_columns() == [13, 14, 15, 16, 17, 18, 19, 20]
    assert d._path_slot == {} and d._slot_path == {}
    assert d._stage_and_fire_sync("D:/x.mp4") == (1, 13)

    # --- a failed open returns None, does not register/connect ------------
    st2, br2 = _FakeStager(fail_paths={"D:/bad.mp4"}), _FakeBridge()
    d2 = DynamicStager(st2, br2, ring_size=4, pool_high_water=0)
    assert d2.ring_columns() == [1, 2, 3, 4]
    assert d2._stage_and_fire_sync("D:/bad.mp4") is None
    assert "D:/bad.mp4" not in br2.registered
    assert br2.connects == []

    # --- guards: no stager/bridge => stage_and_fire is a no-op ------------
    assert DynamicStager(None, br).stage_and_fire("D:/z.mp4") is False
    assert DynamicStager(st, None).stage_and_fire("D:/z.mp4") is False
    assert d.stage_and_fire("") is False

    # --- async path: stage_and_fire queues + worker drains it -------------
    import time
    st3, br3 = _FakeStager(), _FakeBridge()
    d3 = DynamicStager(st3, br3, ring_size=4, pool_high_water=0)
    assert d3.stage_and_fire("D:/async.mp4") is True
    for _ in range(50):                       # up to ~0.5s for the worker
        if br3.connects:
            break
        time.sleep(0.01)
    d3.stop()
    assert br3.connects, "worker never fired the queued clip"
    assert br3.registered.get("D:/async.mp4") == (1, 1)

    print("resolume_dynamic._self_test: OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _self_test()
