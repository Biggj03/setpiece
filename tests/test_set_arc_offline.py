"""Tests for the offline set-arc analyzer + the shared threshold module.

No librosa, no audio: a fake segmenter feeds canned sections, so the
classify-reuse / trend / sidecar / phase_at contract is fully covered and
CI-safe. The librosa segmenter itself is integration-tested at the rig.
"""
import pytest

import set_arc_thresholds as sat
import set_arc_offline as off


# ── shared vocabulary: offline and live agree ──────────────────────────

def test_offline_classify_matches_live_detector():
    # The whole point of the shared module: AutoSetArc._classify and the
    # offline path must label identically for the same inputs.
    auto = pytest.importorskip("auto_set_arc")
    a = auto.AutoSetArc()
    mism = 0
    for bpm in range(60, 170, 2):
        for trend in (-1, 0, 1):
            for cur in sat.PHASES:
                if a._classify(bpm, 0.0, trend, cur) != sat.classify(bpm, 0.0, trend, cur):
                    mism += 1
    assert mism == 0


def test_phase_taxonomy_is_the_four_phases():
    assert sat.PHASES == ("opening", "build", "peak", "breakdown")


# ── the opening-vs-breakdown distinction (the bug this caught) ──────────

def test_low_bpm_at_track_start_is_opening_not_breakdown():
    # A soft intro: low BPM, no operator (offline flip_rate=0), no prior
    # higher phase. Must read OPENING — a breakdown is a descent FROM
    # somewhere, never the very start.
    assert sat.classify(100, 0.0, 0, "opening") == "opening"
    assert sat.classify(95, 0.0, 0, "opening") == "opening"


def test_low_bpm_after_a_climb_is_breakdown():
    # Same low BPM, but we've already been to build/peak -> it IS a descent.
    assert sat.classify(100, 0.0, 0, "peak") == "breakdown"
    assert sat.classify(100, 0.0, 0, "build") == "breakdown"


def test_falling_trend_is_breakdown_regardless_of_context():
    # Rule 2a: an actual BPM drop reads breakdown even from opening.
    assert sat.classify(120, 0.0, -1, "opening") == "breakdown"


# ── trend derivation (offline, section-to-section) ─────────────────────

def test_trend_uses_deadband():
    assert off._trend(None, 120) == 0          # first section
    assert off._trend(120, 125) == 1           # clear rise (>2)
    assert off._trend(120, 115) == -1          # clear fall (<-2)
    assert off._trend(120, 121) == 0           # within deadband -> stable


# ── analyze_sections: the testable core ────────────────────────────────

def test_analyze_sections_labels_an_arc():
    # A plausible set shape: soft open -> build -> peak -> come down.
    raw = [
        (0, 30, 100),    # opening (low, stable, track start)
        (30, 60, 120),   # build (rising into band)
        (60, 90, 140),   # peak (>=135)
        (90, 120, 105),  # breakdown (falling from peak)
    ]
    phases = [s["phase"] for s in off.analyze_sections(raw)]
    assert phases == ["opening", "build", "peak", "breakdown"]


def test_analyze_sections_shape_is_debuggable():
    secs = off.analyze_sections([(0, 30, 120)])
    assert set(secs[0]) >= {"start", "end", "bpm", "trend",
                            "phase", "confidence", "reason"}
    assert secs[0]["confidence"] == 1.0


def test_analyze_sections_hysteresis_holds_phase():
    # Rising into build, then a near-flat section that fires NO rule ->
    # hysteresis holds the prior phase. (Offline, any real drop trips the
    # falling-trend breakdown rule, so hysteresis shows on a rising/flat
    # path: 130 -> 131 is +1 BPM = within deadband = trend 0, no rule.)
    raw = [(0, 30, 120), (30, 60, 130), (60, 90, 131)]
    phases = [s["phase"] for s in off.analyze_sections(raw)]
    assert phases == ["opening", "build", "build"]  # last 'build' held


# ── sidecar round-trip + live-consumption reader ───────────────────────

def test_analyze_write_load_phase_at(tmp_path):
    track = tmp_path / "song.mp3"
    track.write_bytes(b"not really audio")  # only the name/path is used
    fake = lambda p: [(0, 30, 100), (30, 60, 140), (60, 90, 105)]
    data = off.analyze_track(str(track), segmenter=fake)
    sc = off.write_sidecar(str(track), data)
    assert sc.name == "song.mp3.arc.json"

    pt = off.load_phase_track(sc)
    assert pt["version"] == off.SIDECAR_VERSION
    assert pt["track"] == "song.mp3"
    # phase_at picks the section covering the position
    assert off.phase_at(pt, 10) == "opening"    # 100bpm start -> opening
    assert off.phase_at(pt, 45) == "peak"       # 140bpm -> peak
    assert off.phase_at(pt, 75) == "breakdown"  # 105bpm down from peak
    # past the end -> hold last known phase, not None
    assert off.phase_at(pt, 999) == "breakdown"


def test_phase_at_tolerates_missing_track():
    assert off.phase_at(None, 10) is None
    assert off.phase_at({}, 10) is None
    assert off.phase_at({"sections": []}, 10) is None


def test_load_phase_track_missing_file_returns_none(tmp_path):
    assert off.load_phase_track(tmp_path / "nope.arc.json") is None


def test_sidecar_for_naming():
    assert off.sidecar_for("/x/y/track.wav").name == "track.wav.arc.json"
