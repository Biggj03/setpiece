# Setpiece — Alpha Phase 1 Concept

> **Status:** pre-alpha. This document is the clean-room description of what
> the prototype proved, written as the foundation for a sanitized open-source
> release.
> **Audience:** future contributors, and anyone evaluating the project on GitHub.

---

## 1. What it is

Setpiece is a **live VJ engine**: it plays a library of video clips fullscreen,
in time with live music, and makes the editorial decisions a human VJ would
normally make by hand — *which clip, when to cut, how long to hold, how to
build tension across a set*.

You point it at a folder of video clips (music videos, concert footage,
abstract loops, motion graphics — typical VJ source material). It analyses
them once, then performs: listening to audio, cutting on the beat, and
shaping the visuals into a coherent arc rather than a random shuffle.

It is a **single desktop process**. No cloud, no account, no streaming
dependency. A hardware controller and/or a phone/tablet act as the control
surface. It runs entirely on one machine.

---

## 2. The thesis (why it works the way it does)

The prototype was built to test one core idea:

> **DJ paradigms do not translate to VJ work — and a VJ tool should be built
> on VJ-native primitives instead.**

Three concrete consequences, each validated by the prototype:

| DJ paradigm (rejected) | VJ-native primitive (adopted) |
|---|---|
| A/B decks + crossfader | A **clip grid** — many discrete clips, picked from |
| One continuous track | **Clip State** as the atomic unit (file + in/out + parameters) |
| Genre / BPM / artist tags | A **visual-attribute taxonomy** (color → geometry → energy → symmetry) |
| Flat metronome cutting | A **set arc** — macro tension structure across the whole performance |

Everything below is an expression of those four choices.

---

## 3. Architecture at a glance

```
                ┌──────────────────────────────────────────┐
                │  Setpiece (single PyQt6 process)          │
                │                                           │
   audio in ───►│  Audio engine ──► beat / downbeat / drop  │
                │        │                                  │
                │        ▼                                  │
   controller ─►│  Picker (editorial engine) ──► Clip State │──► fullscreen
   (HID/MIDI)   │        ▲                                  │    video out
                │        │                                  │
                │  Tag database (SQLite)                    │
                │        ▲                                  │
                │  HTTP control server :8765 ◄──────────────┼──► phone/tablet
                └────────┼──────────────────────────────────┘
                         │
            Tagger daemons (run offline, separate processes)
            color · geometry · motion · symmetry · pose · cut-detect · embeddings
```

- **Playback:** embedded libmpv, hardware-decoded (`d3d11va` / NVDEC).
- **Control server:** a threading HTTP server serving a mobile-optimised web UI;
  long-poll / streaming endpoints require true multi-threading.
- **Tag store:** SQLite (WAL mode) holding files, tags, and clip embeddings.
- **Taggers:** independent offline processes that analyse the library and write
  tags back. They are *not* part of the live performance path.

---

## 4. Concepts proved

This is the substance of the prototype — the things that were built, stress-
tested over long live sessions, and shown to work.

### 4.1 The discrete clip-state model

The atomic unit is a **Clip State**: a source file plus an in/out range plus
playback parameters. Clips are discrete and duplicable — the same source file
can appear many times with different in/out points. This was deliberately
chosen over a "hot-cues per file" model, and the choice held up: discrete clip
states are simple to reason about, cheap to pick from, and compose cleanly with
everything else.

### 4.2 The editorial picker stack

The picker is the heart of the engine. Instead of random shuffle, it runs a
stack of independent, individually toggleable editorial layers:

| Layer | What it proved |
|---|---|
| **Phrase cut-density** | Cuts quantise onto the *detected musical downbeat* rather than a flat interval. Measured ~55% of cuts landing on the "1" vs ~25% for random timing. |
| **Match-cut continuity** | The next clip is weighted by kinetic carry-over (motion, complexity, geometry, symmetry) from the current clip — no jarring whiplash between cuts. Soft bias, nothing hard-excluded. |
| **Motif callbacks** | A clip held as a "hero" at a peak gets registered; a later peak can re-deploy it as a deliberate callback. A cooldown stops it echo-chambering. |
| **Phase-aware cohesion** | The cohesion anchor (the clip the picker clusters around) is seeded according to the current set phase. |

Each layer is config-flagged and independently killable — proving the stack is
modular, not a monolith.

### 4.3 The set arc

A performance is modelled as a 4-phase macro structure:

| Phase | Tension | Visual character |
|---|---|---|
| **Opening** | Minimal | Centered, symmetric, soft, cool tones |
| **Build** | Escalating | Off-center, fractured, high-contrast |
| **Peak** | Maximum | Edge-heavy, angular, full-spectrum |
| **Breakdown** | Resolution | Negative space, minimal, dark tones |

The phase can be driven **manually** (controller button) or **automatically**
(an auto-arc watcher reads BPM and cut-rate and advances the phase). The picker
biases clip selection toward each phase's profile. A **lookahead** predicts the
next phase change a few seconds out, so the UI can hint "→ PEAK in 30s".

**Hero-hold on drop:** at a peak entry (or a detected audio drop), the engine
*suppresses cutting* and locks onto a single hero clip for ~16 beats. This was
a key finding — at the climax, professional VJs hold one strong visual; they do
*not* machine-gun cuts. The prototype now does the same.

### 4.4 Automatic visual-attribute tagging

The library is analysed once by a set of offline taggers, each extracting one
VJ-native axis. None of them depend on human-authored metadata:

| Axis | Method |
|---|---|
| **Color** | Sample a frame, k-means (k=3) in LAB space → warm/cool + hue buckets + 3 hex swatches |
| **Geometry** | Edge density + line-direction histogram → particles / linear / polygons / masks |
| **Energy / movement** | Frame-to-frame motion variance → static/dynamic, smooth/jumpy, complexity 0–9 |
| **Symmetry** | Left-half vs right-half SSIM → cohesive vs offset/asymmetric |
| **Subject position** | Pose estimation (YOLOv8-pose) → coarse subject placement |
| **Segment / intro detection** | Per-second frame-diff peaks → detects where an opening segment ends |
| **Semantic embeddings** | CLIP image embeddings → free-text "find me clips like this" search |

The picker consumes these axes directly. The proven point: **a video library
can be made VJ-usable with zero manual tagging** — the visual attributes that
matter for live performance are all machine-extractable.

### 4.5 A self-healing tagger subsystem

Analysing thousands of clips overnight is hostile territory — codecs hang,
decoders stall, one bad file can wedge a run. The prototype proved a robust
pattern:

- **Subprocess isolation** — each file is analysed in a killable child process
  with a hard timeout. A pathological file is sentinel-tagged and skipped, never
  stalling the run.
- **Watchdog supervision** — a supervisor restarts a tagger if it stalls (with a
  deliberately *generous* threshold; an over-tight one caused restart thrash).
- **Chained execution** — taggers hand off to each other automatically
  (tagger A finishes → tagger B launches) with safety caps.
- **Single-writer discipline** — never two processes writing the tag DB at once.

Result: multi-hour, multi-thousand-file tagging runs completing unattended with
zero restarts.

### 4.6 Audio reactivity

- **Onset detection** started as a ~40-line numpy spectral-flux implementation —
  deliberately, because it avoids a fragile native dependency and does the job.
- It **evolved to stem-isolated onsets**: a neural source-separation model
  (Open-Unmix) isolates the drum stem, so onset detection fires on actual drums
  and not on bass synths. Runs several times faster than realtime on a modest
  GPU; ~2s latency, traded for accuracy.
- **Downbeat detection** buckets per-beat kick energy mod-4 to locate the "1",
  feeding the phrase-cut layer.
- **Drop detection** flags energy spikes to trigger hero-hold.

### 4.7 Categorical banks + semantic recall

The control surface exposes **8 categorical banks** (A–H). Banks are
**user-defined categories** — the engine fills them, the operator labels them.
Banks can be built from:

- a folder (point a bank set at a directory),
- a **free-text vibe** ("type a phrase → semantic search → top matches split
  across the 8 banks"),
- a saved preset (star a productive vibe, recall it later as a one-tap chip),
- the auto-detected opening segments (an "Openers" bank).

### 4.8 Hardware + tablet control

- **HID / MIDI control surfaces** — grid-and-encoder style controllers were
  integrated as the primary live surface, with a **fully-lit, state-driven
  feedback** model (every key's color/brightness reflects current state) and a
  "pinned peripheral" layout (global controls in fixed physical positions, the
  same on every page).
- **Phone / tablet web UI** — a mobile-optimised control panel served over the
  LAN: now-playing, cue markers, audio-reactive status, crossfade, **live video
  previews of staged clips**, library browser, banks. The app runs without it
  but the tablet is the comfortable cockpit.

---

## 5. What is proven vs. what is still aspirational

Honest status, because an alpha should be honest.

**Solid:**
- The editorial picker stack — tested live, "cuts feel good".
- The visual-attribute taggers — full-library coverage achieved.
- The self-healing tagger architecture — unattended multi-hour runs.
- Set-arc + hero-hold — manual and automatic both working.
- The control surface + tablet UI — daily-driver stable.

**Jank / known limits:**
- Stem-isolated audio has **~2s latency** — accurate, but not instantaneous.
  True sub-20ms reactivity (beat-phase PLL) is researched but not built.
- Beat *phase* tracking still rides on spectral flux; a dedicated PLL is
  deferred until a live set proves it's needed.
- Single-GPU bound; tested on one modest NVIDIA card.
- Currently **Windows-specific** in places (windowless subprocess flags, HID
  backend) — a Linux port is planned and the platform-specific code is already
  isolated behind safe no-ops.
- One specific Stream Deck unit has a firmware quirk; that integration ships
  **disabled by default**.

---

## 6. Alpha Phase 1 — release scope

What the first sanitized public release should contain:

**In scope:**
- The single-process engine: playback, picker, set-arc, audio reactivity.
- The tagger subsystem with all visual-attribute taggers.
- The HTTP control server + web control UI.
- One reference control-surface integration.
- Generic example config + a small bundled sample-clip set (CC-licensed or
  self-produced footage only).
- `README`, `SETUP`, `ARCHITECTURE`, `CONTROLS` docs.

**Out of scope for Phase 1:**
- Any networked / multi-user / streaming features.
- Any voting / room / sync-watch functionality.
- The neural stem-separation daemon (heavy dependency — make it opt-in later).
- Live-set telemetry / analytics.

---

## 7. Tech stack

- Python 3 + PyQt6 (desktop GUI)
- libmpv (embedded video playback, hardware-decoded)
- ffmpeg / ffprobe (analysis + frame extraction)
- SQLite (tag + embedding store, WAL mode)
- OpenCV, NumPy (tagger image analysis)
- PyTorch — CLIP embeddings, YOLOv8-pose, Open-Unmix stem separation (opt-in)
- A threading HTTP server + vanilla HTML/CSS/JS for the tablet control UI
- HID / MIDI for hardware control surfaces

Target platforms: Windows today, Linux (Debian-class) port planned — the
platform-specific code is already isolated.

---

## 8. The one-sentence pitch (for the GitHub description)

*Setpiece is an open-source live VJ engine that auto-cuts a tagged video
library to live music — building a structured visual set with beat-quantised
cuts, match-cut continuity, and a tension arc, instead of a random shuffle.*
