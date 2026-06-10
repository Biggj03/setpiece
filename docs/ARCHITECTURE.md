# Architecture

Setpiece is a single PyQt6 desktop process. Everything below runs in that
one process except the taggers, which are independent offline tools.

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
            color · geometry · motion · symmetry · pose · intro · embeddings
```

## Runtime pieces

- **Playback** — embedded libmpv, hardware-decoded (`d3d11va` / NVDEC).
  See `player_mpv.py`.
- **Picker** — the editorial engine in `main.py`. Runs a stack of
  independently toggleable layers (phrase cut-density, match-cut
  continuity, motif callbacks, phase-aware cohesion) over the tag
  database to choose the next clip.
- **Audio engine** — `audio_reactive.py` does onset / beat / downbeat /
  drop detection. The default detector is a compact NumPy spectral-flux
  implementation; `beatnet_detector.py` is an opt-in PLL upgrade and
  `stem_daemon.py` an opt-in neural drum-stem path.
- **Set arc** — `set_arc.py` / `auto_set_arc.py` model the performance as
  a 4-phase macro structure and bias the picker per phase. The phase
  vocabulary and thresholds live in `set_arc_thresholds.py`, shared with
  `set_arc_offline.py` — an offline analyzer that segments a known track
  (librosa, optional dep), classifies each section with the *same*
  `classify()` the live detector uses, and writes a `<track>.arc.json`
  sidecar. The consumption API ships with it (`load_phase_track()` /
  `phase_at()`, position-based lookup with live-detection fallback), but
  the auto-arc watcher does not call it yet — wiring the live side to
  prefer sidecar ground truth on known tracks is roadmap.
- **Tag store** — SQLite in WAL mode at `~/.setpiece/path_tags.db3`,
  holding files, tags and clip embeddings. Accessed via `path_tags.py` /
  `path_tags_v2.py`.
- **Control server** — `http_server.py` is a *threading* HTTP server
  (`ThreadingHTTPServer`) serving the mobile web UI on port 8765. It must
  be multi-threaded: the preview-stream and long-poll endpoints each hold
  a connection open, and a single-threaded server would deadlock.
- **Control surfaces** — `s2_controller.py`, `maschine_mk2.py` and
  `stream_deck.py` integrate HID/MIDI hardware. `osc_in.py` / `osc_out.py`
  bridge OSC.

## Resolume bridge (optional render engine)

Setpiece can hand rendering to **Resolume Arena** for live shows —
gapless playback, real compositing, multi-screen out. The bridge is a
set of small standalone stdlib-only modules, tested offline against
fakes and stub servers (no Arena needed for the suite), running in
their own process — deliberately decoupled from the main engine so a
crash on either side never takes down the other.

| Module | Role |
|---|---|
| `gig.py` | One-command launcher: panel + pre-flight + iPad URL |
| `resolume_panel.py` | Touch panel web server (MIX / CLIPS / FX tabs) |
| `resolume_out.py` | OSC commander — the live fire path |
| `resolume_state.py` | REST reader — state snapshots, param control |
| `resolume_stage.py` | REST loader — stage working sets into the grid |
| `resolume_dynamic.py` | Stage-on-miss: off-grid picks load into a bounded ring |
| `resolume_selfcheck.py` | Gig-readiness check (exit 0 = ready) |

Two transports, split by job: **OSC** (UDP, fire-and-forget,
sub-millisecond) carries everything on the live path — clip fires,
opacity, crossfader, tempo, blackout. **REST** (HTTP) handles setup and
readback — loading clips into grid slots, state snapshots, parameter
discovery. Nothing on the hot path ever blocks on HTTP.

The grid is managed as a **working set**: a curated subset of the
library staged into Arena's columns, with a filepath → (layer, column)
registry so the show driver fires by path without knowing grid
geometry. A fired path the grid doesn't hold is a *miss*, which
`resolume_dynamic` resolves by loading that one clip into a bounded
ring of recycled columns over REST (~150 ms, on a worker thread) and
firing it — so every pick reaches Arena (a flurry of misses coalesces
to the newest) without the grid growing unbounded over a multi-hour
set.

Two gig-safety guards live here: staging resolves containers Arena
can't open (`.mkv`, `.webm` — including ones that load but render
black) to same-stem `.mp4` twins, and the selfcheck exercises every
subsystem end-to-end against the live Arena before doors, restoring
the values it changes and removing its probe clips (its header
discloses the small residue it can't undo, e.g. Arena's column
auto-growth).

## The clip-state model

The atomic unit is a **Clip State**: a source file, an in/out range, and
playback parameters. Clip states are discrete and duplicable — the same
source file can appear many times with different in/out points. This was
chosen over a "hot-cues per file" model because discrete states are
simple to reason about, cheap to pick from, and compose cleanly with the
banks, scenes and picker.

## Taggers

The taggers in the ingest pipeline are **independent offline processes**.
They analyse the library once and write tags back to the SQLite store;
they are *not* part of the live performance path. Each extracts one
VJ-native axis:

| Tagger | Axis |
|---|---|
| `color_tagger` | warm/cool + hue buckets + palette swatches |
| `geometry_tagger` | edge density / line direction → particles / linear / polygons / masks |
| `motion_tagger` | frame-to-frame motion variance → static/dynamic, complexity 0–9 |
| `symmetry_tagger` | left/right SSIM → cohesive vs offset |
| `pose_tagger` | pose estimation → coarse subject placement |
| `intro_tagger` | per-second frame-diff peaks → opening-segment detection |
| `vision_tag` | CLIP image embeddings → free-text semantic search |

### Self-healing tagger subsystem

Analysing thousands of clips unattended is hostile territory — codecs
hang, decoders stall, one bad file can wedge a run. The taggers use:

- **Subprocess isolation** — each file is analysed in a killable child
  process with a hard timeout. A pathological file is sentinel-tagged and
  skipped, never stalling the run.
- **Watchdog supervision** — `tagger_watchdog.py` restarts a tagger if it
  stalls, with a deliberately generous threshold.
- **Chained execution** — taggers hand off to each other automatically.
- **Single-writer discipline** — never two processes writing the tag DB
  at once.
