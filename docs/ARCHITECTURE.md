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
  a 4-phase macro structure and bias the picker per phase.
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
