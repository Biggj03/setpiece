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

# Octave-correction guard — see fix_bpm_octaves(). Folds suspiciously-low
# section BPMs ×2 when the rest of the track sits substantially higher,
# catching autocorrelation half-time lock on bass-heavy material.
HALFTIME_LOW_BPM = 90.0      # below this is "suspect"
HALFTIME_MEDIAN_HI = 130.0   # ... if the median sits above this
HALFTIME_ACCEPT_BAND = 10.0  # doubled value must land within ±this of median

# Per-frame tempo clamp — see clamp_frame_tempos(). Frames survive within
# this band of the global tempo estimate OR of its half/double octaves;
# the gaps BETWEEN those bands are where the classic pulse-ratio noise
# lives (2/3=0.67 and 3/4=0.75 below the anchor, 4/3=1.33 and 3/2=1.50
# above — at any octave). 0.8..1.25 is symmetric in log space, and the
# octave bands it induces (0.4..0.625, 0.8..1.25, 1.6..2.5 of the anchor)
# never overlap and never admit those ratios.
DTEMPO_CLAMP_LO = 0.8
DTEMPO_CLAMP_HI = 1.25


def sidecar_for(track_path) -> Path:
    """Sidecar path for a track: '<track>.arc.json' beside the file."""
    p = Path(track_path)
    return p.with_suffix(p.suffix + ".arc.json")


def _median(xs: list) -> float:
    """Stdlib-only median (avoids importing statistics for one call)."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (float(s[mid - 1]) + float(s[mid])) / 2.0


def fix_bpm_octaves(sections, log=None):
    """Octave-correct guard against the autocorrelation half-time lock.

    Symptom we're catching: a 152-BPM bass track gets one 76-BPM section
    because the BPM extractor latched onto the half-beat peak instead of
    the downbeat. That one bad read drags the offline analyzer into
    labelling the section as a breakdown.

    Heuristic:
      - Compute the median BPM across all sections in the track.
      - For each section reading < 90 BPM while the median is > 130 BPM,
        double it (×2). If the doubled value is within ±10 of the median,
        accept the fold. Otherwise leave the original AND log the
        suspicious section — better to flag than silently corrupt a
        borderline legitimate slow section.

    Does NOT touch sections in tracks whose median is already low: a
    genuinely slow track shouldn't have its sub-90 readings folded.

    Args:
      sections: [(start, end, bpm), ...] raw segmenter output.
      log: optional callable(str) for "fold accepted" / "fold rejected"
           breadcrumbs. Defaults to logging.getLogger(__name__).info.
    Returns: new [(start, end, bpm), ...] with octave-corrected BPMs.
    """
    if log is None:
        import logging
        log = logging.getLogger(__name__).info

    bpms = [float(b) for (_, _, b) in sections if b and float(b) > 0]
    if len(bpms) < 2:
        return list(sections)  # nothing to compare against
    median = _median(bpms)
    if median <= HALFTIME_MEDIAN_HI:
        return list(sections)  # track is genuinely slow; don't meddle

    out = []
    for (s, e, b) in sections:
        b = float(b)
        if 0 < b < HALFTIME_LOW_BPM:
            doubled = b * 2.0
            if abs(doubled - median) <= HALFTIME_ACCEPT_BAND:
                log(f"octave fold: {b:.1f} -> {doubled:.1f} "
                    f"(median {median:.1f})")
                out.append((s, e, doubled))
                continue
            log(f"suspicious low BPM kept: {b:.1f} "
                f"(median {median:.1f}, doubled would be {doubled:.1f})")
        out.append((s, e, b))
    return out


def clamp_frame_tempos(values, global_bpm: Optional[float],
                       lo: float = DTEMPO_CLAMP_LO,
                       hi: float = DTEMPO_CLAMP_HI) -> list:
    """Drop per-frame tempo estimates that sit at a NON-OCTAVE ratio to
    the track's global tempo estimate.

    Frames survive inside [a*lo, a*hi] of any octave anchor a in
    {global/2, global, global*2}; the gaps between those bands hold the
    classic pulse-ratio noise (2/3, 3/4, 4/3, 3/2 — at any octave) that
    librosa's per-frame estimator latches onto on polyrhythmic bass
    material. The motivating failure (2026-06-01 backend check): ~80% of
    one section's frames latched a sub-pulse at ~0.7x of a 152-BPM bass
    track, the section MEAN dragged to 115, and a peak-energy stretch
    read "breakdown".

    Octave-RELATED deviation is deliberately kept, at original values:
    half- and double-time frames are real musical content (a slow
    groove's defining dynamic; fast genres' tempo switching), and
    keeping the whole octave grid also makes the clamp insensitive to
    the ANCHOR itself octave-locking — observed both ways during 2026-06
    verification (the anchor doubles on a ~69 BPM trip-hop groove, and
    halves on ~180 BPM material). Whole-section octave calls remain
    downstream in fix_bpm_octaves.

    Survivor semantics, pinned deliberately: any octave-coherent
    evidence outvotes any amount of non-octave noise (the motivating
    case WAS an 80/20 noise majority). Only when NO frame is
    octave-coherent does the raw input pass through unchanged, leaving
    the judgment to downstream logic.

    KNOWN LIMITATION (measured, 2026-06-09 A/B round): material whose
    tempo content is genuinely bimodal at a NON-octave ratio to the
    anchor — e.g. a DJ-mix slice switching between ~100 and ~160 BPM
    against a ~103 anchor (a ~sqrt(2) ratio, mid-gap in any band scheme
    that excludes 4/3 and 3/2) — gets its off-anchor mode dropped even
    though it is real music. A side effect of the one-outvotes-all
    semantics: a section whose only octave-coherent frames are tempogram
    bleed from a neighboring section reports the neighbor's tempo. Six
    of seven A/B genres improved or held; one such hyperpop slice
    degraded. There is deliberately NO auto-guard: no track-level
    coherence statistic separates helped from hurt tracks (the least
    anchor-coherent track in the A/B was the biggest win). For known
    octave-ambiguous material, disable clamping at the segmenter seam
    (librosa_segmenter(path, clamp=False)).

    Args:
      values: per-frame tempo estimates for one section (any iterable).
      global_bpm: track-level tempo estimate anchoring the octave grid,
        or None/0 to disable clamping.
    Returns: list of surviving frame tempos (floats), never empty if
      `values` wasn't.
    """
    vals = [float(v) for v in values]
    g = float(global_bpm or 0.0)
    if g <= 0 or not vals:
        return vals
    anchors = (g / 2.0, g, g * 2.0)
    kept = [v for v in vals
            if any(a * lo <= v <= a * hi for a in anchors)]
    return kept if kept else vals


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


def librosa_segmenter(path, clamp: bool = True) -> list:
    """Default segmenter: librosa beat/tempo + agglomerative segmentation.

    The base pipeline was verified on real audio 2026-06-01 (dedicated
    numpy<2 venv — see requirements-optional.txt for why); the per-frame
    tempo clamp was added after that and A/B-verified against the
    pre-clamp baseline across seven real tracks on 2026-06-09: improved
    or unchanged on six (trap, dnb, bass house, techno, deep dubstep,
    trip-hop), degraded on one octave-ambiguous hyperpop DJ-mix slice —
    see the KNOWN LIMITATION note on clamp_frame_tempos.
    The clamp drops per-frame estimates at non-octave ratios to the
    track-level tempo before section-averaging — librosa's per-frame
    estimator latches onto sub-pulses on polyrhythmic bass material, and
    one bimodal stretch is enough to mislabel a peak section as a
    breakdown. Pass clamp=False for octave-ambiguous material (wrap it:
    analyze_track(p, segmenter=lambda q: librosa_segmenter(q, clamp=False))).

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
    # Track-level tempo anchor for the per-frame clamp: the same
    # estimator as dtempo, aggregated over the whole envelope instead of
    # per-frame — which is what keeps it stable where individual frames
    # wander onto a sub-pulse. (Its octave can still be wrong on very
    # fast/slow material; the clamp's octave grid tolerates that.)
    # global_bpm = 0 disables clamping in clamp_frame_tempos.
    global_bpm = 0.0
    if clamp:
        tempo_g = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        global_bpm = float(np.atleast_1d(tempo_g)[0])
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
        if mask.any():
            kept = clamp_frame_tempos(dtempo[mask], global_bpm)
            seg_tempo = sum(kept) / len(kept)
        else:
            seg_tempo = 0.0
        sections.append((s, e, seg_tempo))
    return sections


def analyze_track(path, segmenter: Optional[Callable] = None) -> dict:
    """Full analysis -> sidecar dict (not yet written). Uses `segmenter`
    (default librosa) to get raw sections, runs the octave-correct BPM
    guard (catches autocorrelation half-time lock on bass material),
    then labels them."""
    seg = segmenter or librosa_segmenter
    raw = seg(path)
    corrected = fix_bpm_octaves(raw)
    return {
        "version": SIDECAR_VERSION,
        "source": "set_arc_offline",
        "track": Path(path).name,
        "sections": analyze_sections(corrected),
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
