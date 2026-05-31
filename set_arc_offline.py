"""
Offline set-arc analyzer: read a whole track ahead of time and write a
phase track (timestamp -> phase) into a sidecar JSON.

THE SYNERGY (offline feeds live)
--------------------------------
The live detector (`auto_set_arc.AutoSetArc`) reacts to BPM + operator
flip-rate in real time — great for unknown tracks, but it can only ever
guess at "what's coming" from recent slope. Offline, you have the WHOLE
track: segment it, read each section's tempo, and label phases with
ground truth. Write that to `<track>.arc.json`. Then the live detector,
on a known track, can read the precomputed phase for the current playback
position instead of guessing:

    pt = load_phase_track(sidecar_for(track))
    phase = phase_at(pt, position_s) or auto.detect_phase(bpm, current)
    #        ^ ground truth on known tracks   ^ live fallback otherwise

Both stay: offline = ground truth on known tracks; live = unknown tracks
plus the operator-behaviour signal (flip rate) offline can't see.

SHARED VOCABULARY
-----------------
Phase labels + thresholds come from `set_arc_thresholds`, the SAME module
the live detector uses. A section tagged "peak" here is the exact "peak"
the live picker biases toward — no drift between the two sources.

SEGMENTER SEAM
--------------
`analyze_track(path, segmenter=...)` takes a pluggable segmenter:
    segmenter(path) -> [(start_s, end_s, avg_bpm), ...]
The default is `librosa_segmenter` (opt-in dep, see requirements-optional).
Swap in madmom, or a fake for tests, without touching the classify/sidecar
logic. Offline passes flip_rate=0 (a track has no operator); trend is
derived section-to-section from the BPM deltas.

Stdlib only at import time — librosa is imported lazily inside the
segmenter, so this module loads (and its non-audio logic tests) without it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import set_arc_thresholds as sat

SIDECAR_VERSION = 1
# Match the live detector's bpm_trend deadband (auto_set_arc.bpm_trend):
# a section is "rising"/"falling" only if it moves more than this vs the
# previous section. Keeps offline trend consistent with live trend.
TREND_DEADBAND_BPM = 2.0


def sidecar_for(track_path) -> Path:
    """Sidecar path for a track: '<track>.arc.json' beside the file."""
    p = Path(track_path)
    return p.with_suffix(p.suffix + ".arc.json")


def _trend(prev_bpm: Optional[float], bpm: float) -> int:
    """+1 rising / 0 stable / -1 falling vs the previous section, using the
    same deadband as the live detector. None prev (first section) = 0."""
    if prev_bpm is None or prev_bpm <= 0 or bpm <= 0:
        return 0
    d = bpm - prev_bpm
    if d > TREND_DEADBAND_BPM:
        return 1
    if d < -TREND_DEADBAND_BPM:
        return -1
    return 0


def analyze_sections(sections) -> list:
    """Turn raw segments [(start, end, bpm), ...] into labelled phase
    sections, walking left-to-right so each section's `current` (hysteresis
    fallback) is the previous section's phase. flip_rate is 0 offline.

    Returns [{start, end, bpm, trend, phase, confidence, reason}, ...].
    Pure data — no audio, no I/O — so it's the fully-testable core."""
    out = []
    prev_bpm = None
    current = "opening"  # tracks start soft until evidence says otherwise
    for (start, end, bpm) in sections:
        bpm = float(bpm)
        trend = _trend(prev_bpm, bpm)
        phase = sat.classify(bpm, 0.0, trend, current)
        out.append({
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "bpm": round(bpm, 1),
            "trend": trend,
            "phase": phase,
            # Offline confidence is 1.0: this is MEASURED from the audio,
            # not predicted forward like the live slope extrapolation.
            "confidence": 1.0,
            "reason": f"bpm {bpm:.0f} trend {trend:+d} -> {phase}",
        })
        prev_bpm = bpm
        current = phase
    return out


def librosa_segmenter(path) -> list:
    """Default segmenter: librosa beat/tempo + agglomerative segmentation.

    UNVERIFIED HERE: librosa isn't installed in this dev env and there's no
    audio to test on, so this is best-effort structure pending a live run.
    The TESTED contract is the seam + analyze_sections + sidecar round-trip
    (see tests, which use a fake segmenter). Install librosa via
    requirements-optional.txt to use this for real.

    Returns [(start_s, end_s, avg_bpm), ...].
    """
    try:
        import librosa
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "librosa not installed — `pip install -r requirements-optional.txt` "
            "to use the default offline segmenter, or pass your own "
            "segmenter to analyze_track()."
        ) from e

    y, sr = librosa.load(path, mono=True)
    # Onset envelope -> dynamic tempo over time + segment boundaries.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Per-frame tempo (aggregate=None gives a tempo estimate per frame).
    dtempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr,
                                   aggregate=None)
    # Segment the track on timbre (MFCC) into ~8 chunks, then average the
    # local tempo within each chunk.
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    n_seg = 8
    bounds = librosa.segment.agglomerative(mfcc, n_seg)
    bound_times = librosa.frames_to_time(bounds, sr=sr)
    total = librosa.get_duration(y=y, sr=sr)
    edges = [0.0] + list(bound_times) + [total]
    edges = sorted(set(round(e, 3) for e in edges if 0 <= e <= total))
    tempo_times = librosa.times_like(dtempo, sr=sr)
    sections = []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        if e - s < 1.0:
            continue
        mask = (tempo_times >= s) & (tempo_times < e)
        seg_tempo = float(np.mean(dtempo[mask])) if mask.any() else 0.0
        sections.append((s, e, seg_tempo))
    return sections


def analyze_track(path, segmenter: Optional[Callable] = None) -> dict:
    """Full analysis -> sidecar dict (not yet written). Uses `segmenter`
    (default librosa) to get raw sections, then labels them."""
    seg = segmenter or librosa_segmenter
    raw = seg(path)
    return {
        "version": SIDECAR_VERSION,
        "source": "set_arc_offline",
        "track": Path(path).name,
        "sections": analyze_sections(raw),
    }


def write_sidecar(track_path, data: dict) -> Path:
    """Write the analysis dict to '<track>.arc.json'. Returns the path."""
    sc = sidecar_for(track_path)
    sc.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return sc


def load_phase_track(sidecar_path) -> Optional[dict]:
    """Load a sidecar, or None if absent/unreadable (never raises)."""
    try:
        return json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    except Exception:
        return None


def phase_at(phase_track: Optional[dict], position_s: float) -> Optional[str]:
    """The precomputed phase for a playback position, or None if there's no
    track / no section covers the position. This is the live-consumption
    contract: AutoSetArc calls this first, falls back to live detect on None.

    Tolerant of bad input (None track, missing sections) — returns None."""
    if not phase_track:
        return None
    pos = float(position_s)
    last = None
    for sec in phase_track.get("sections") or []:
        try:
            if sec["start"] <= pos < sec["end"]:
                return sec["phase"]
            if pos >= sec["end"]:
                last = sec["phase"]  # remember most recent passed section
        except (KeyError, TypeError):
            continue
    # Past the last section's end (e.g. trailing silence) -> hold the last
    # known phase rather than snapping to None mid-outro.
    return last


def main(argv) -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(argv) < 2:
        print("usage: python set_arc_offline.py <track.(mp3|wav|...)> "
              "[more tracks...]")
        print("Writes <track>.arc.json beside each (needs librosa: "
              "pip install -r requirements-optional.txt).")
        return 1
    rc = 0
    for track in argv[1:]:
        try:
            data = analyze_track(track)
            sc = write_sidecar(track, data)
            n = len(data["sections"])
            phases = " ".join(s["phase"] for s in data["sections"])
            print(f"OK  {Path(track).name}: {n} sections -> {sc.name}")
            print(f"    arc: {phases}")
        except Exception as e:
            print(f"FAIL {Path(track).name}: {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
