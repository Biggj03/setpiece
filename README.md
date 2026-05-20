# Setpiece

**A live VJ engine that auto-cuts a tagged video library to live music —
building a structured visual set with beat-quantised cuts, match-cut
continuity, and a tension arc, instead of a random shuffle.**

Setpiece plays a library of video clips fullscreen, in time with live
music, and makes the editorial decisions a human VJ would normally make
by hand: *which clip, when to cut, how long to hold, how to build tension
across a set.*

You point it at a folder of video clips (music videos, concert footage,
abstract loops, motion graphics — typical VJ source material). It analyses
them once, then performs: listening to audio, cutting on the beat, and
shaping the visuals into a coherent arc rather than a random shuffle.

It is a **single desktop process**. No cloud, no account, no streaming
dependency. A hardware controller and/or a phone/tablet act as the control
surface. It runs entirely on one machine.

> **Status: pre-alpha (Alpha Phase 1).** The core engine, taggers and
> control UI are stress-tested and stable. Some platform-specific code is
> still Windows-first; a Linux port is in progress. See
> [Status](#status) below for an honest breakdown.

---

## Why it works the way it does

Setpiece is built on one core idea:

> **DJ paradigms do not translate to VJ work — a VJ tool should be built
> on VJ-native primitives instead.**

| DJ paradigm (rejected) | VJ-native primitive (adopted) |
|---|---|
| A/B decks + crossfader | A **clip grid** — many discrete clips, picked from |
| One continuous track | **Clip State** as the atomic unit (file + in/out + parameters) |
| Genre / BPM / artist tags | A **visual-attribute taxonomy** (color → geometry → energy → symmetry) |
| Flat metronome cutting | A **set arc** — macro tension structure across the whole performance |

## What it does

- **Editorial picker** — instead of random shuffle, a stack of independently
  toggleable layers: beat-quantised cut density, match-cut continuity
  (kinetic carry-over between clips), motif callbacks, and phase-aware
  cohesion.
- **Set arc** — a performance is modelled as a 4-phase macro structure
  (Opening → Build → Peak → Breakdown). Drive it manually or let the
  auto-arc watcher advance phases from BPM and cut-rate. At a peak the
  engine holds one hero clip instead of machine-gunning cuts.
- **Automatic visual-attribute tagging** — offline taggers analyse the
  library once along VJ-native axes (color, geometry, energy/movement,
  symmetry, subject placement, intro detection, CLIP embeddings). No manual
  tagging required.
- **Audio reactivity** — beat / downbeat / drop detection drives the cuts.
- **Categorical banks** — 8 operator-defined banks (A–H), fillable from a
  folder, a free-text vibe search, or a saved preset.
- **Control surfaces** — a HID/MIDI hardware controller and a
  mobile-optimised web UI served over the LAN.

See [`ALPHA1_CONCEPT.md`](ALPHA1_CONCEPT.md) for the full design rationale.

## Quick start

```sh
pip install -r requirements.txt
python main.py
```

You also need **libmpv** and **ffmpeg** on the system — see
[`SETUP.md`](SETUP.md) for the full install (including the optional
heavy/ML dependencies and how to tag a library).

On first launch, point Setpiece at a folder of video clips. Tag the
library with `python ingest.py --root /path/to/clips`, then perform.

## Documentation

- [`SETUP.md`](SETUP.md) — install, dependencies, tagging a library
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the engine is built
- [`docs/CONTROLS.md`](docs/CONTROLS.md) — keyboard, web UI, hardware

## Status

**Solid:**
- The editorial picker stack — tested live.
- The visual-attribute taggers — full-library coverage.
- The self-healing tagger architecture — unattended multi-hour runs.
- Set-arc + hero-hold — manual and automatic.
- The control surface + web UI — daily-driver stable.

**Known limits:**
- Stem-isolated audio reactivity has ~2s latency (accurate, not instant).
- Beat *phase* tracking rides on spectral flux; a dedicated PLL is opt-in.
- Tested on a single modest NVIDIA GPU.
- Some code is still Windows-specific (windowless subprocess flags, HID
  backend); a Linux port is underway and the platform code is isolated.

## License

MIT — see [`LICENSE`](LICENSE).
