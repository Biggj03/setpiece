"""
Set-arc mode — codify the PDF's 4-phase visual-narrative theory into
picker behavior.

WHY THIS EXISTS
---------------
Per "Architectural Paradigms in Live Visual Performance" (the user's
research PDF, page 5-7), a professional VJ set follows a macro arc:

  OPENING   → cool monochrome, centered, soft, slow, low complexity
  BUILD     → off-center, dual-tone high contrast, escalating motion
  PEAK/DROP → edge-heavy, asymmetric, angular, rapid, full spectrum
  BREAKDOWN → vast negative space, dark monochrome, minimal motion

The picker currently makes *micro* decisions (BPM match + dialog +
performer cohesion). Set-arc adds a *macro* layer: during phase X,
boost candidate clips whose color/motion/geometry tags match phase X's
profile. Operator drives the phase manually (cycles via MK2 button or
HTTP endpoint); auto-detection is a future ship.

ARCHITECTURE
------------
- Phase list + profile definitions live here (data-only, no Qt deps).
- App holds `_set_arc_phase` as runtime state; persisted to settings.
- `phase_boost_for(path, phase, color_data, motion_data)` returns a
  weight multiplier. Picker calls it for each candidate.
- Compounds with phrase-boost + hero-boost (they all multiply).
"""

from __future__ import annotations

from typing import Optional

# Phase identifiers — stable strings used in settings + UI
PHASE_OPENING = "opening"
PHASE_BUILD = "build"
PHASE_PEAK = "peak"
PHASE_BREAKDOWN = "breakdown"
PHASES = [PHASE_OPENING, PHASE_BUILD, PHASE_PEAK, PHASE_BREAKDOWN]

# Per-phase profile. Each profile is a dict of:
#   prefer_thermal:  "warm" | "cool" | None     (matches color:warm/cool)
#   prefer_hues:     list of hue strings        (color:hue:<x>)
#   prefer_motion:   list of motion: tag values (motion:static / dynamic / smooth / jumpy)
#   prefer_complexity_range: (lo, hi) inclusive (complexity:0..9)
#   prefer_geometry: list of geometry: tag values
#   boost_multiplier_match:  weight × this when ALL criteria match
#   boost_multiplier_partial: weight × this when SOME criteria match
#   penalty_no_match:        weight × this when NO criteria match (kept >0)
#
# The boost is multiplicative + intentionally MODEST (1.5-3.0x range)
# so phase-arc nudges the picker without locking out variety. Bigger
# multipliers would defeat the whole point of weighted-random picks.
PHASE_PROFILES: dict[str, dict] = {
    PHASE_OPENING: {
        "label": "OPENING",
        "prefer_thermal": "cool",
        "prefer_hues": ["blue", "cyan", "magenta"],
        "prefer_motion": ["static", "smooth"],
        "prefer_complexity_range": (0, 4),
        "prefer_geometry": ["particles", "linear"],
        # Symmetry: openings call for centered, calm subjects — single
        # focal point. Avoid chaotic offset framings.
        "prefer_symmetry": ["cohesive", "mirror"],
        # Transition-rich: prefer show-format content where the file
        # has substantial dancing/dialog/setup segments. Tagged via
        # studio_transition_tagger.py.
        "prefer_transition_rich": True,
        # bg-friendly: clips that work as projection-background layers.
        # User-curated tag (operator marks during clip-mark workflow).
        "prefer_bg_friendly": True,
        # dialog: clean talking content. Good for OPENING setup vibe
        # before energy ramps up.
        "prefer_dialog": True,
        "boost_match_full": 2.5,
        "boost_match_partial": 1.6,
        "penalty_no_match": 0.7,
    },
    PHASE_BUILD: {
        "label": "BUILD",
        "prefer_thermal": None,    # both warm and cool OK during build
        "prefer_hues": ["orange", "red", "blue"],
        "prefer_motion": ["dynamic", "smooth"],
        "prefer_complexity_range": (3, 7),
        "prefer_geometry": ["polygons", "linear"],
        # Symmetry: build phase escalates -- offset framings push the
        # eye around, layered mirror clips raise tension.
        "prefer_symmetry": ["mirror", "offset"],
        "boost_match_full": 2.0,
        "boost_match_partial": 1.4,
        "penalty_no_match": 0.85,
    },
    PHASE_PEAK: {
        "label": "PEAK / DROP",
        "prefer_thermal": "warm",
        "prefer_hues": ["red", "orange", "yellow", "magenta"],
        "prefer_motion": ["dynamic", "jumpy"],
        "prefer_complexity_range": (6, 9),
        "prefer_geometry": ["polygons", "particles"],
        # Symmetry: peak wants RADIAL (mandalas/tunnels pull the eye
        # to center as the drop hits) + chaotic offset.
        "prefer_symmetry": ["radial", "offset"],
        "boost_match_full": 3.0,
        "boost_match_partial": 1.8,
        "penalty_no_match": 0.6,
    },
    PHASE_BREAKDOWN: {
        "label": "BREAKDOWN",
        "prefer_thermal": "cool",
        "prefer_hues": ["blue", "cyan"],
        "prefer_motion": ["static", "smooth"],
        "prefer_complexity_range": (0, 3),
        "prefer_geometry": ["particles", "alpha-mask"],
        # Symmetry: breakdown = single hypnotic focal point. Cohesive
        # subjects + radial mandalas hold the eye still.
        "prefer_symmetry": ["cohesive", "radial"],
        # Transition-rich: same logic as OPENING -- show-format clips
        # have the calmer non-sex segments that fit a breakdown.
        "prefer_transition_rich": True,
        # bg-friendly: layer-friendly content fits breakdown
        # contemplation moments.
        "prefer_bg_friendly": True,
        # dialog: reflective talking moments fit breakdown character.
        "prefer_dialog": True,
        "boost_match_full": 2.5,
        "boost_match_partial": 1.4,
        "penalty_no_match": 0.55,
    },
}


def next_phase(current: str) -> str:
    """OPENING → BUILD → PEAK → BREAKDOWN → OPENING (cycle)."""
    try:
        i = PHASES.index(current)
    except ValueError:
        return PHASE_OPENING
    return PHASES[(i + 1) % len(PHASES)]


def prev_phase(current: str) -> str:
    try:
        i = PHASES.index(current)
    except ValueError:
        return PHASE_OPENING
    return PHASES[(i - 1) % len(PHASES)]


def label_for(phase: str) -> str:
    p = PHASE_PROFILES.get(phase)
    return p["label"] if p else phase.upper()


def score_clip(
    phase: str,
    *,
    thermal: Optional[str] = None,           # "warm" | "cool" | None
    hue: Optional[str] = None,               # "red" | "blue" | ... | None
    motion_tags: Optional[set[str]] = None,  # e.g. {"static", "smooth"}
    complexity: Optional[int] = None,        # 0..9 or None
    geometry: Optional[str] = None,          # "particles" | ... | None
    symmetry: Optional[str] = None,          # "mirror"|"radial"|"cohesive"|"offset"
    transition_rich: bool = False,           # True if file tagged transition-rich
    bg_friendly: bool = False,               # True if file tagged bg-friendly
    dialog: bool = False,                    # True if file tagged dialog
) -> float:
    """Return the multiplier for a candidate clip under the given phase.

    Counts how many criteria match. Returns:
      boost_match_full     if 4+ criteria match
      boost_match_partial  if 2-3 criteria match
      1.0                  if exactly 1 criterion matches
      penalty_no_match     if 0 criteria match

    Untagged dimensions don't count as matches OR non-matches — they're
    neutral. So clips with sparse tag coverage just get fewer
    opportunities to match, not penalized for missing tags."""
    profile = PHASE_PROFILES.get(phase)
    if not profile:
        return 1.0

    matches = 0
    examined = 0   # number of dimensions we actually got data for

    # Thermal
    if profile.get("prefer_thermal") and thermal:
        examined += 1
        if thermal == profile["prefer_thermal"]:
            matches += 1

    # Hue
    if profile.get("prefer_hues") and hue:
        examined += 1
        if hue in profile["prefer_hues"]:
            matches += 1

    # Motion (any overlap)
    if profile.get("prefer_motion") and motion_tags:
        examined += 1
        if set(profile["prefer_motion"]) & motion_tags:
            matches += 1

    # Complexity range
    if profile.get("prefer_complexity_range") and complexity is not None:
        examined += 1
        lo, hi = profile["prefer_complexity_range"]
        if lo <= complexity <= hi:
            matches += 1

    # Geometry
    if profile.get("prefer_geometry") and geometry:
        examined += 1
        if geometry in profile["prefer_geometry"]:
            matches += 1

    # Symmetry (mirror/radial/cohesive/offset)
    if profile.get("prefer_symmetry") and symmetry:
        examined += 1
        if symmetry in profile["prefer_symmetry"]:
            matches += 1

    # Transition-rich (show-format clip flag). Only counts when the
    # phase profile asks for it (OPENING / BREAKDOWN today). For
    # PEAK/BUILD this dimension is neutral -- transition-rich files
    # still play, they just don't get extra boost.
    if profile.get("prefer_transition_rich"):
        examined += 1
        if transition_rich:
            matches += 1

    # bg-friendly (operator-tagged: works as projection background)
    if profile.get("prefer_bg_friendly"):
        examined += 1
        if bg_friendly:
            matches += 1

    # dialog (operator-tagged: clean talking content)
    if profile.get("prefer_dialog"):
        examined += 1
        if dialog:
            matches += 1

    if examined == 0:
        return 1.0   # nothing to judge on, neutral

    if matches >= 4:
        return profile["boost_match_full"]
    if matches >= 2:
        return profile["boost_match_partial"]
    if matches >= 1:
        return 1.0   # one weak match — neutral
    return profile["penalty_no_match"]
