"""Tests for resolume_selfcheck: a dict-backed fake Arena world where the
fake bridge MUTATES the state the fake reader reads back, so the check
orchestration, the restore discipline, and the probe cleanup are all
verified end-to-end without a live Arena. The live wire formats are
pinned elsewhere (test_resolume_stage stub server, osc_out self-test);
what this file proves is that the selfcheck draws correct conclusions
and leaves the operator's comp exactly as it found it."""
import pytest

import resolume_selfcheck as sc


# ── the fake Arena world ────────────────────────────────────────────────

OPACITY_PID = 901


class World:
    def __init__(self):
        self.crossfader = 0.25
        self.tempo = 120.0
        self.master = 0.8
        self.opacity = 1.0                  # layer 1 video opacity
        self.columns = 3                    # layer 1 existing columns
        self.loaded = {1, 2}                # operator's loaded slots (layer 1)
        self.active_clip = 2                # the clip the operator has live
        self.reachable = True
        # fault injection
        self.tempo_broken = False           # Arena "ignores" tempo OSC
        self.clear_broken = False


class FakeState:
    def __init__(self, world):
        self.w = world

    def preflight(self):
        ok = self.w.reachable
        return {"ok": ok, "checks": [
            {"name": "Arena REST (webserver :8080)", "ok": ok,
             "detail": "", "critical": True}]}

    def snapshot(self, max_clip_probe: int = 0):
        if not self.w.reachable:
            return {"reachable": False, "layers": []}
        return {
            "reachable": True, "product": "Fake Arena",
            "crossfader": self.w.crossfader, "tempo": self.w.tempo,
            "master": self.w.master,
            "layers": [{"index": 1, "columns": self.w.columns,
                        "loaded": len(self.w.loaded),
                        "opacity": self.w.opacity,
                        "active_clip": self.w.active_clip}],
        }

    def layer_opacity_param(self, layer):
        if not self.w.reachable:
            return None
        return (OPACITY_PID, self.w.opacity)

    def set_param_by_id(self, pid, value):
        if pid == OPACITY_PID:
            self.w.opacity = float(value)
            return True
        return False


class FakeBridge:
    def __init__(self, world):
        self.w = world
        self.resyncs = 0
        self.closed = False
        self.connects = []

    def set_crossfader(self, phase):
        self.w.crossfader = float(phase)

    def set_tempo(self, bpm):
        if not self.w.tempo_broken:
            self.w.tempo = float(bpm)

    def panic_black(self):
        self.w.master = 0.0

    def restore_output(self):
        self.w.master = 1.0

    def set_composition_master(self, v):
        self.w.master = float(v)

    def resync_downbeat(self):
        self.resyncs += 1

    def connect_clip(self, layer, col):
        self.connects.append((layer, col))

    def register_clip(self, path, layer, col):
        pass

    def close(self):
        self.closed = True


class FakeStager:
    def __init__(self, world):
        self.w = world
        self.opens = []                     # (layer, col, path)
        self.clears = []                    # (layer, col)
        self.lie_unloaded = False           # readback falsely reports empty

    def open_into_slot(self, layer, col, path):
        self.opens.append((layer, col, path))
        self.w.loaded.add(col)
        return True

    def slot_loaded(self, layer, col):
        if self.lie_unloaded:
            return False
        return col in self.w.loaded

    def clear_slot(self, layer, col):
        self.clears.append((layer, col))
        if self.w.clear_broken:
            return False
        self.w.loaded.discard(col)
        return True


def _run(world, probes=("p1.mp4", "p2.mp4"), **kw):
    return sc.run(state=FakeState(world), bridge=FakeBridge(world),
                  stager=FakeStager(world), probes=list(probes),
                  settle=0.0, **kw)


def _areas(check):
    return {area: ok for (area, ok, _) in check.results}


# ── the green path ──────────────────────────────────────────────────────

def test_all_green_on_healthy_world():
    c = _run(World())
    assert _areas(c) == {"Pre-flight": True, "Bridge OSC": True,
                         "Stager REST": True, "Dynamic ring": True,
                         "Param by-id": True}
    assert c.ok() is True
    assert "ALL GREEN" in c.report()


def test_everything_touched_is_restored():
    w = World()
    _run(w)
    # The operator's comp comes back exactly as it was.
    assert w.crossfader == 0.25
    assert w.tempo == 120.0
    assert w.master == 0.8
    assert w.opacity == 1.0


def test_restore_runs_even_when_readback_raises_mid_blackout():
    # The nightmare: panic_black() lands, then the next snapshot raises.
    # The check must still restore master/tempo/crossfader — a readiness
    # tool that can leave output BLACK on an exception fails its own bar.
    w = World()

    class DyingState(FakeState):
        def __init__(self, world):
            super().__init__(world)
            self.calls = 0

        def snapshot(self, max_clip_probe: int = 0):
            self.calls += 1
            if self.calls == 4:        # snap0, cf, tempo, then DIE on black
                raise OSError("connection dropped mid-check")
            return super().snapshot(max_clip_probe)

    c = sc.run(state=DyingState(w), bridge=FakeBridge(w),
               stager=FakeStager(w), probes=["p1.mp4", "p2.mp4"], settle=0.0)
    assert _areas(c)["Bridge OSC"] is False     # honestly reported
    assert w.master == 0.8                       # ...but output came back
    assert w.tempo == 120.0
    assert w.crossfader == 0.25


def test_operator_clip_is_reconnected_after_probe_fire():
    # The dynamic-ring check FIRES a probe clip, ejecting whatever the
    # operator had playing on the layer. The selfcheck must re-connect
    # the original clip afterward — the last connect on the layer is the
    # operator's clip, not the probe.
    w = World()
    br = FakeBridge(w)
    sc.run(state=FakeState(w), bridge=br, stager=FakeStager(w),
           probes=["p1.mp4", "p2.mp4"], settle=0.0)
    assert br.connects[-1] == (1, 2)


def test_no_reconnect_when_nothing_was_playing():
    w = World()
    w.active_clip = None
    br = FakeBridge(w)
    c = sc.run(state=FakeState(w), bridge=br, stager=FakeStager(w),
               probes=["p1.mp4", "p2.mp4"], settle=0.0)
    # The only connect is the probe fire from inside DynamicStager; we
    # must NOT "restore" a clip that wasn't playing — and the check still
    # passes (a None active clip must not crash the reconnect path).
    assert all(col > 3 for (_, col) in br.connects)
    assert _areas(c)["Dynamic ring"] is True


def test_probe_clips_are_cleaned_up_and_never_touch_content():
    w = World()
    c = _run(w)
    assert c.ok()
    # No probe clip left behind...
    assert w.loaded == {1, 2}
    # ...and no probe ever landed on (or before) the operator's columns.
    # (Verified via the stager's recorded opens in a fresh run.)
    st = FakeStager(World())
    sc.run(state=FakeState(st.w), bridge=FakeBridge(st.w), stager=st,
           probes=["p1.mp4", "p2.mp4"], settle=0.0)
    assert st.opens and all(col > 3 for (_, col, _) in st.opens)


# ── failure modes draw the right conclusions ────────────────────────────

def test_broken_tempo_fails_only_the_osc_check():
    w = World()
    w.tempo_broken = True
    c = _run(w)
    a = _areas(c)
    assert a["Bridge OSC"] is False
    assert a["Stager REST"] and a["Dynamic ring"] and a["Param by-id"]
    assert c.ok() is False


def test_broken_clear_fails_both_probe_checks_and_warns_of_leak():
    # Cleanup is VERIFIED, not asserted: a broken clear fails both probe
    # areas and the report says a probe clip may be left behind (no
    # silent "(cleaned up)" claims).
    w = World()
    w.clear_broken = True
    c = _run(w)
    a = _areas(c)
    assert a["Stager REST"] is False
    assert a["Dynamic ring"] is False
    details = " ".join(d for (_, _, d) in c.results)
    assert "PROBE CLIP" in details


def test_clear_attempted_even_when_readback_lies_empty():
    # The probe-leak hole: open succeeds, but the slot_loaded readback
    # reports empty (transient REST flake / load still in flight). The
    # check must STILL attempt the clear — clearing an empty slot is
    # harmless; leaking a probe clip is not.
    w = World()
    st = FakeStager(w)
    st.lie_unloaded = True
    c = sc.run(state=FakeState(w), bridge=FakeBridge(w), stager=st,
               probes=["p1.mp4", "p2.mp4"], settle=0.0)
    # The SCRATCH column's clear specifically (the ring check issues its
    # own clear at col 5, which must not satisfy this assertion).
    assert (1, 4) in st.clears            # cleanup ran despite readback
    assert _areas(c)["Stager REST"] is False   # honestly failed


def test_injected_bridge_is_not_closed_by_run():
    # run() must never kill a caller's live bridge: ResolumeBridge.close
    # drops the socket and later OSC sends become silent no-ops.
    w = World()
    br = FakeBridge(w)
    sc.run(state=FakeState(w), bridge=br, stager=FakeStager(w),
           probes=["p1.mp4", "p2.mp4"], settle=0.0)
    assert br.closed is False


def test_unreachable_arena_short_circuits():
    w = World()
    w.reachable = False
    c = _run(w)
    # One actionable failure, not five identical ones.
    assert [area for (area, ok, _) in c.results] == ["Pre-flight",
                                                     "Arena REST"]
    assert c.ok() is False
    detail = c.results[-1][2]
    assert "Webserver" in detail


def test_no_probe_media_fails_with_the_fix_in_the_message():
    c = _run(World(), probes=())
    a = _areas(c)
    assert a["Stager REST"] is False and a["Dynamic ring"] is False
    detail = dict((area, d) for (area, _, d) in c.results)["Stager REST"]
    assert "make_samples" in detail


# ── report / aggregation semantics ──────────────────────────────────────

def test_check_aggregation_and_report_markers():
    c = sc.Check()
    assert c.ok() is False               # empty = not green
    c.add("A", True, "fine")
    c.add("B", False, "broke")
    assert c.ok() is False
    rep = c.report()
    assert "[OK ] A: fine" in rep and "[XX ] B: broke" in rep
    assert "NOT READY" in rep


def test_real_layer_opacity_param_order_and_degradation(monkeypatch):
    # The REAL ResolumeState.layer_opacity_param (the fakes above override
    # it, so this pins the actual implementation): (id, value) tuple ORDER
    # and None on missing/partial shapes.
    from resolume_state import ResolumeState
    st = ResolumeState(rest_base="http://127.0.0.1:9/api/v1", timeout=0.1)
    monkeypatch.setattr(
        st, "_get",
        lambda p: {"video": {"opacity": {"id": 42, "value": 0.7}}})
    assert st.layer_opacity_param(1) == (42, 0.7)
    monkeypatch.setattr(st, "_get", lambda p: {"video": {}})
    assert st.layer_opacity_param(1) is None
    monkeypatch.setattr(st, "_get", lambda p: None)
    assert st.layer_opacity_param(1) is None


def test_pick_probes_returns_sorted_mp4s_or_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "SAMPLES_DIR", tmp_path / "nope")
    assert sc.pick_probes() == []
    d = tmp_path / "samples"
    d.mkdir()
    (d / "b.mp4").write_bytes(b"")
    (d / "a.mp4").write_bytes(b"")
    (d / "x.txt").write_bytes(b"")
    monkeypatch.setattr(sc, "SAMPLES_DIR", d)
    assert sc.pick_probes() == [str(d / "a.mp4"), str(d / "b.mp4")]
