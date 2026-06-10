"""
Gig-readiness self-check: one command verifies the WHOLE Setpiece ->
Arena bridge against a live Arena. Run it before a show, or after any
change to the bridge modules, to catch regressions across the stack.

Goes beyond resolume_state.preflight() (which checks the pipeline is
*reachable*): this exercises each shipped subsystem END-TO-END and
reports pass/fail per area:

  1. Pre-flight     — REST up, Arena 7+, composition has content
  2. Bridge OSC     — crossfader / tempo (normalized) / blackout+restore,
                      each verified by REST readback
  3. Stager REST    — probe clip opens into a scratch slot, readback
                      confirms, per-clip clear removes it again
  4. Dynamic ring   — an off-pool probe stages + fires through
                      DynamicStager's miss path, then is cleaned up
  5. Param by-id    — a live param (layer opacity) set + readback +
                      restore via the by-id PUT mechanism

WHAT IT RESTORES: crossfader, tempo, composition master, and layer
opacity return to their pre-check values — restores run in `finally`
blocks, so an exception mid-check cannot leave output black or the
tempo wrong. Probe clips are removed with the per-clip clear (the
layer-level clear silently leaves clips behind; per-clip genuinely
removes) and removal is verified by readback; a clip that was playing
on the probe layer is re-connected after the probe fire. The probe
clips come from samples/ (run `python make_samples.py` once to
generate them) or pass --probe explicitly.

WHAT IT CANNOT RESTORE (known, disclosed):
  - Opening probes past the layer's column high-water permanently GROWS
    the composition's column count (Arena auto-grows; its REST API has
    no column-remove). Each run leaves ~2 empty trailing columns that
    persist if the comp is saved afterward. Empty and harmless, but not
    "restored".
  - resync_downbeat() realigns the beat-clock's downbeat phase. The
    tempo VALUE is restored; the phase nudge is not undoable.
  - OSC restores are fire-and-forget UDP — a dropped restore packet is
    not detected (the check itself reads state back over REST, but the
    final restores are not re-verified).
  - The re-connected operator clip restarts from its trigger point, not
    from the playhead position it was at.

RUN IT BEFORE DOORS, NOT MID-SET: the check is visibly disruptive while
it runs (~3s) — output blacks out and comes back, the probe clip
flashes on the probe layer, tempo jumps and returns.

Exit 0 = ALL GREEN (gig-ready), 1 = something regressed, 2 = could not
even reach Arena.

USAGE
    python resolume_selfcheck.py
    python resolume_selfcheck.py --layer 2 --probe path/to/a.mp4
    python resolume_selfcheck.py --rest http://192.168.4.50:8080/api/v1 \
                                 --osc-host 192.168.4.50
"""

import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SETTLE = 0.3   # seconds for Arena to apply an OSC verb before readback
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"


class Check:
    """Accumulates per-area results and renders the gig-readiness report."""

    def __init__(self):
        self.results = []          # (area, ok, detail)

    def add(self, area: str, ok: bool, detail: str) -> None:
        self.results.append((str(area), bool(ok), str(detail)))

    def ok(self) -> bool:
        return bool(self.results) and all(r[1] for r in self.results)

    def report(self) -> str:
        out = ["=== Setpiece -> Arena self-check ==="]
        for area, ok, detail in self.results:
            out.append(f"  [{'OK ' if ok else 'XX '}] {area}: {detail}")
        out.append("")
        out.append("  >>> ALL GREEN - gig-ready." if self.ok()
                   else "  >>> NOT READY - see failures above.")
        return "\n".join(out)


def pick_probes(n: int = 2) -> list:
    """Pick n probe clips from the generated samples/ dir (sorted, so runs
    are deterministic). Returns [] if none exist — the caller reports that
    as a failure with the fix (run make_samples.py)."""
    if not SAMPLES_DIR.is_dir():
        return []
    clips = sorted(str(f) for f in SAMPLES_DIR.iterdir()
                   if f.is_file() and f.suffix.lower() == ".mp4")
    return clips[:n]


def _layer_columns(snap: dict, layer: int) -> int:
    """Column count of `layer` from a snapshot(), 0 if unknown."""
    for ly in snap.get("layers") or []:
        if ly.get("index") == layer:
            return int(ly.get("columns") or 0)
    return 0


def run(state=None, bridge=None, stager=None, probes=None,
        layer: int = 1, settle: float = DEFAULT_SETTLE) -> Check:
    """Run every check; returns the Check. All clients injectable for
    tests; None constructs the real ones against localhost defaults."""
    if state is None:
        from resolume_state import ResolumeState
        state = ResolumeState()
    own_bridge = bridge is None
    if bridge is None:
        from resolume_out import ResolumeBridge
        bridge = ResolumeBridge(enabled=True)
    if stager is None:
        from resolume_stage import ResolumeStager
        stager = ResolumeStager()
    if probes is None:
        probes = pick_probes()

    c = Check()

    # 1) Pre-flight ----------------------------------------------------------
    try:
        rep = state.preflight()
        bad = ",".join(x["name"] for x in rep["checks"] if not x["ok"])
        c.add("Pre-flight", rep["ok"], "GO" if rep["ok"] else f"NO-GO ({bad})")
    except Exception as e:
        c.add("Pre-flight", False, f"raised: {e}")

    snap0 = {}
    try:
        snap0 = state.snapshot()
    except Exception:
        pass
    if not snap0.get("reachable"):
        # Nothing else can be exercised, and 4 more identical failures
        # would just bury the actionable one.
        c.add("Arena REST", False,
              "unreachable - start Arena + enable Preferences > Webserver")
        return c

    # 2) Bridge OSC verbs, each verified by REST readback ---------------------
    cf0 = snap0.get("crossfader")
    tempo0 = snap0.get("tempo")
    master0 = snap0.get("master")
    try:
        bridge.set_crossfader(1.0)
        time.sleep(settle)
        cf = state.snapshot().get("crossfader")

        bridge.set_tempo(130.0)
        time.sleep(settle)
        tempo = state.snapshot().get("tempo")

        bridge.panic_black()
        time.sleep(settle)
        mblack = state.snapshot().get("master")
        bridge.restore_output()
        time.sleep(settle)
        mrestore = state.snapshot().get("master")
        bridge.resync_downbeat()      # fire-and-forget; crash-checks the verb

        ok = (cf is not None and abs(cf - 1.0) < 0.05
              and tempo is not None and abs(tempo - 130.0) < 1.0
              and mblack is not None and mblack < 0.05
              and mrestore is not None and mrestore > 0.95)
        c.add("Bridge OSC", ok,
              f"xfade={cf} tempo={tempo} black={mblack} restore={mrestore}")
    except Exception as e:
        c.add("Bridge OSC", False, f"raised: {e}")
    finally:
        # Restore the operator's state EVEN IF a readback raised mid-check —
        # a gig-readiness tool must never leave output black or the tempo
        # wrong because an exception skipped the restore path.
        try:
            if cf0 is not None:
                bridge.set_crossfader(float(cf0))
            if tempo0 is not None:
                bridge.set_tempo(float(tempo0))
            if master0 is not None:
                bridge.set_composition_master(float(master0))
        except Exception as e:
            c.add("Bridge OSC restore", False, f"restore raised: {e}")

    # Scratch columns live past the layer's current high-water so the
    # probe never lands on (or clears) operator content.
    scratch_base = _layer_columns(snap0, layer)
    # If a clip is playing on the probe layer, the dynamic-ring fire will
    # eject it — remember it so it can be re-connected afterward.
    active0 = None
    for ly in snap0.get("layers") or []:
        if ly.get("index") == layer:
            active0 = ly.get("active_clip")
            break

    # 3) Stager REST: open -> readback -> per-clip clear ----------------------
    if not probes:
        c.add("Stager REST", False,
              "no probe media - run `python make_samples.py` or pass --probe")
        c.add("Dynamic ring", False, "skipped (no probe media)")
    else:
        col = scratch_base + 1
        opened = landed = cleared = gone = False
        err = None
        try:
            opened = stager.open_into_slot(layer, col, probes[0])
            landed = opened and stager.slot_loaded(layer, col)
        except Exception as e:
            err = e
        finally:
            # Clear whenever the open went through — even if the readback
            # failed or raised, the clip may be in the grid. Clearing an
            # empty slot is harmless; leaking a probe clip is not.
            if opened:
                try:
                    cleared = stager.clear_slot(layer, col)
                    gone = cleared and not stager.slot_loaded(layer, col)
                except Exception:
                    pass
        leak = " PROBE CLIP MAY BE LEFT @ scratch column" \
            if (opened and not gone) else ""
        if err is not None:
            c.add("Stager REST", False, f"raised: {err}.{leak}")
        else:
            c.add("Stager REST", landed and cleared and gone,
                  f"open+readback+clear @ L{layer}C{col} (open={opened} "
                  f"landed={landed} clear={cleared} removed={gone}){leak}")

        # 4) Dynamic stage-on-miss through the ring ---------------------------
        from resolume_dynamic import DynamicStager
        probe2 = probes[1] if len(probes) > 1 else probes[0]
        res = None
        ring_landed = ring_gone = False
        err = None
        try:
            dyn = DynamicStager(stager, bridge, layer=layer, ring_size=2,
                                pool_high_water=col)   # ring past the probe col
            res = dyn._stage_and_fire_sync(probe2)     # synchronous on purpose
            ring_landed = res is not None and stager.slot_loaded(*res)
        except Exception as e:
            err = e
        finally:
            if res is not None:
                # Verified cleanup (clear + removal readback), drop the probe
                # from the bridge registry, and re-connect whatever the
                # operator had playing (the probe fire ejected it).
                try:
                    ring_gone = (stager.clear_slot(*res)
                                 and not stager.slot_loaded(*res))
                except Exception:
                    pass
                try:
                    bridge.unregister_clip(probe2)
                except AttributeError:
                    pass
                if active0 is not None:
                    bridge.connect_clip(layer, int(active0))
        if err is not None:
            ring_leak = (" PROBE CLIP MAY BE LEFT IN RING"
                         if (res is not None and not ring_gone) else "")
            c.add("Dynamic ring", False, f"raised: {err}.{ring_leak}")
        elif res is None:
            c.add("Dynamic ring", False, "probe did not stage+land")
        else:
            c.add("Dynamic ring", ring_landed and ring_gone,
                  f"off-pool probe -> L{res[0]}C{res[1]} "
                  f"(landed={ring_landed} cleaned={ring_gone})"
                  + ("" if ring_gone else " PROBE CLIP MAY BE LEFT IN RING"))

    # 5) Param by-id: set + readback + restore --------------------------------
    got = None
    try:
        got = state.layer_opacity_param(layer)
    except Exception:
        pass
    if not got:
        c.add("Param by-id", False, f"layer {layer} opacity param unreadable")
    else:
        pid, op0 = got
        target = 0.5 if abs(op0 - 0.5) > 0.1 else 0.75
        after = None
        err = None
        restored = False
        try:
            state.set_param_by_id(pid, target)
            time.sleep(settle)
            after = state.layer_opacity_param(layer)
        except Exception as e:
            err = e
        finally:
            # Restore even if the readback raised; a failed restore is its
            # own loud failure (the operator's opacity is wrong).
            try:
                restored = bool(state.set_param_by_id(pid, op0))
            except Exception:
                pass
        if err is not None:
            c.add("Param by-id", False,
                  f"raised: {err} (opacity restore "
                  f"{'ok' if restored else 'FAILED'})")
        else:
            landed = after is not None and abs(after[1] - target) < 0.05
            read = f"{after[1]:.2f}" if after else "?"
            c.add("Param by-id", landed and restored,
                  f"opacity {op0:.2f} -> {target:.2f} (read {read}) "
                  f"restore={'ok' if restored else 'FAILED'}")

    if own_bridge:
        # Only close a bridge this run() constructed. Closing an injected
        # live bridge would silently kill the caller's OSC sends (close
        # drops the socket; later sends become debug-level no-ops).
        try:
            bridge.close()
        except Exception:
            pass
    return c


def main(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Gig-readiness self-check for the Setpiece -> Arena "
                    "bridge (exit 0 = gig-ready).")
    ap.add_argument("--rest", default=None,
                    help="Arena REST base (default localhost:8080)")
    ap.add_argument("--osc-host", default="127.0.0.1")
    ap.add_argument("--osc-port", type=int, default=7000)
    ap.add_argument("--layer", type=int, default=1,
                    help="layer for probe staging (default 1; scratch "
                         "columns are appended past its content)")
    ap.add_argument("--probe", action="append", default=None,
                    help="probe clip path (repeatable; default: samples/)")
    ap.add_argument("--settle", type=float, default=DEFAULT_SETTLE,
                    help="seconds to let Arena apply a verb before readback")
    a = ap.parse_args(argv[1:])

    logging.basicConfig(level=logging.WARNING)
    from resolume_out import ResolumeBridge
    from resolume_stage import ResolumeStager
    from resolume_state import ResolumeState

    kw = {"rest_base": a.rest} if a.rest else {}
    state = ResolumeState(**kw)
    stager = ResolumeStager(**kw)
    bridge = ResolumeBridge(host=a.osc_host, port=a.osc_port, enabled=True)

    c = run(state=state, bridge=bridge, stager=stager, probes=a.probe,
            layer=a.layer, settle=a.settle)
    try:
        bridge.close()                # main owns this bridge
    except Exception:
        pass
    sys.stdout.buffer.write((c.report() + "\n").encode("utf-8", "replace"))
    if any(area == "Arena REST" and not ok
           for (area, ok, _) in c.results):
        return 2                      # could not even reach Arena
    return 0 if c.ok() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
