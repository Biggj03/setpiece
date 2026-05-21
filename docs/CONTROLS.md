# Controls

Setpiece can be driven three ways — they all act on the same engine and
can be used together: the desktop keyboard, the web UI, and a hardware
control surface.

## Keyboard (desktop window)

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `I` | Mark in-point on the current clip |
| `O` | Mark out-point on the current clip |
| `B` | Flip — cut to the next clip |
| `F` | Toggle fullscreen |
| `Esc` | Exit fullscreen |
| `Shift+S` | Toggle the scanlines shader overlay |
| `Shift+V` | Toggle the vignette shader overlay |

Shader overlays are mpv user-shaders in `shaders/*.glsl`. They stack —
add more by dropping a `.glsl` into that folder and binding a key in
`main.py`.

## Web UI (phone / tablet)

The HTTP control server serves a mobile-optimised web UI on port 8765.
By default it is **localhost-only** — open `http://127.0.0.1:8765` in a
browser on the machine running Setpiece. To control it from a phone or
tablet, launch with `python main.py --lan` (or set `"lan_access": true`
in settings.json), then open `http://<setpiece-machine>:8765` on any
device on the same LAN. The server has no authentication, so only
enable LAN access on a network you trust.

It exposes now-playing, cue markers, audio-reactive status, crossfade,
live video previews of staged clips, a library browser, and the banks.
The engine runs fine without it — the tablet is just the comfortable
cockpit. Additional surfaces are served at `/cdj.html` and
`/mk2_layout.html`.

The UI is plain HTML/CSS/JS in `static/`. The server uses
cache-busting headers on every response (iOS Safari otherwise serves
stale pages indefinitely).

## Hardware control surface

Setpiece integrates HID/MIDI grid-and-encoder controllers as the primary
live surface. The model:

- **Fully-lit, state-driven feedback** — every key's colour and brightness
  reflects current engine state, so the surface is readable at a glance
  under stage light.
- **Pinned-peripheral layout** — global controls (home / blackout / set
  phase) sit in fixed physical positions, identical on every page. Centre
  keys are dynamic per page.
- **Page colour-coding** — each page has a signature colour so the active
  page is unmistakable.

The reference integration targets a Maschine MK2-class pad controller
(`maschine_mk2.py`) and a Traktor S2-class deck (`s2_controller.py`).
A controller is optional — the app runs on keyboard + web UI alone.

## Banks

Eight categorical banks (A–H) are the operator's quick-switch palette.
Banks are **operator-defined categories** — the engine fills them, you
label them. Fill a bank set from:

- a folder (point a bank set at a directory),
- a free-text vibe (type a phrase → semantic search → top matches split
  across the eight banks),
- a saved preset (star a productive vibe, recall it later),
- the auto-detected opening segments (an "Openers" bank).

The default bank categories are keyed to the visual-attribute axes the
taggers produce (colour, energy, geometry, symmetry). They are starting
points only — relabel and retag them to taste via `bank_layers` /
`bank_categories` in `~/.setpiece/settings.json`.
