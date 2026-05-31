"""
Shared set-arc vocabulary: phase names + the BPM/flip thresholds that
classify a moment of a set into a phase.

WHY THIS MODULE EXISTS
----------------------
Two things speak "set arc": the LIVE detector (`auto_set_arc.AutoSetArc`,
watching BPM + operator flip-rate in real time) and the OFFLINE analyzer
(`set_arc_offline`, reading a whole track's audio ahead of time). They MUST
agree on what "peak" means — otherwise a track tagged "peak" offline would
get treated as "build" live, and the two drift into two rigs.

So the calibrated constants live here, once, and both sides import them.
Change a threshold here and both the live picker bias and the offline
phase track move together.

PHASES
------
The 4-phase macro arc, in intensity order:
    opening -> build -> peak -> breakdown
(breakdown is the come-down/resolution, not "lowest intensity start".)

THRESHOLDS
----------
Calibrated to this library. BPM bands + flip-rate cutoffs. The live
detector adds streaming state (trend windows, cooldown) on top; the
offline analyzer adds segmentation. Neither redefines these numbers.
"""

# Phase taxonomy — the shared vocabulary. Order is intensity/progression.
PHASES = ("opening", "build", "peak", "breakdown")

# Phase thresholds (BPM)
PEAK_BPM = 135
BUILD_LO_BPM = 110
BUILD_HI_BPM = 135
OPENING_HI_BPM = 115
BREAKDOWN_LO_BPM = 100

# Flip-rate thresholds (per minute) — operator-behaviour signal, live only.
# The offline analyzer passes flip_rate=0 (a track has no operator), so
# these only ever fire live; kept here so both sides read one definition.
PEAK_FLIP_RATE = 6.0
BREAKDOWN_FLIP_RATE = 2.0


def classify(bpm, flip_rate, trend, current):
    """Stateless phase cascade — the single source of truth for "given
    these signals, what phase is this?". Both the live detector and the
    offline analyzer call THIS, so a section labelled 'peak' offline is the
    same 'peak' the live picker biases toward.

    Args:
      bpm:        smoothed/section-average BPM (0 if unknown).
      flip_rate:  operator flips per minute. Live signal; offline passes 0.
      trend:      +1 rising / 0 stable / -1 falling.
      current:    the current phase, returned as hysteresis fallback when
                  no rule fires (live wants "hold"; offline can pass the
                  previous section's label, or "opening" at track start).

    Returns: one of PHASES (or `current` on fallthrough).
    """
    # 1. PEAK: hard BPM or a flip burst (operator hammering >).
    if bpm >= PEAK_BPM or flip_rate > PEAK_FLIP_RATE:
        return "peak"

    # 2. BREAKDOWN: a DESCENT, not just "low + quiet".
    #   (a) BPM actively falling with low flip activity, OR
    #   (b) a sustained very-low-BPM stretch — but only once we've already
    #       climbed somewhere (current != opening). Without the guard this
    #       mislabels a soft OPENING as a breakdown: a track that opens at
    #       <=100 BPM has trend 0 and (offline) flip_rate 0, which would
    #       otherwise fire (b) on every quiet intro. A breakdown is reached
    #       by coming DOWN from build/peak, never at the very start.
    if trend < 0 and flip_rate < BREAKDOWN_FLIP_RATE:
        return "breakdown"
    if (bpm > 0 and bpm <= BREAKDOWN_LO_BPM and flip_rate < 1.0
            and current != "opening"):
        return "breakdown"

    # 3. BUILD: rising BPM in the build band.
    if trend > 0 and BUILD_LO_BPM <= bpm <= BUILD_HI_BPM:
        return "build"

    # 4. OPENING: low / soft BPM that isn't actively falling.
    if bpm > 0 and bpm <= OPENING_HI_BPM and trend >= 0:
        return "opening"

    # 5. Hysteresis: nothing explicit fired -- keep current.
    return current
