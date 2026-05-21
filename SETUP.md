# Setup

Setpiece is a single Python desktop process. This covers installing it,
the system libraries it needs, and getting a video library tagged.

## 1. Python dependencies

Python 3.10+ is required.

```sh
pip install -r requirements.txt
```

This installs the core engine + visual-attribute taggers (PyQt6,
python-mpv, NumPy, OpenCV, Pillow, hidapi).

### Optional / heavy dependencies

The ML-heavy features are opt-in. None are needed to run the engine —
each feature self-disables if its dependency is missing.

```sh
pip install -r requirements-optional.txt
```

| Dependency | Unlocks |
|---|---|
| torch + open-clip-torch | CLIP embeddings — the `vision` tagger and free-text clip search |
| ultralytics | YOLOv8-pose — the subject-placement tagger |
| openunmix | neural drum-stem isolation for stem-accurate onsets |
| BeatNet | PLL beat tracking (higher accuracy than the default detector) |
| anthropic | the optional cloud Vision tagger |
| streamdeck | Elgato Stream Deck integration (disabled by default) |

## 2. System libraries

These are **not** pip packages — install them with your OS package manager.

### libmpv

Video playback uses embedded libmpv (hardware-decoded).

- **Windows:** install mpv and ensure `libmpv-2.dll` is on `PATH`
  (`choco install mpv`).
- **Debian / Ubuntu:** `apt install libmpv2`

### ffmpeg

The taggers use `ffmpeg` / `ffprobe` for frame extraction and analysis.
Install it and ensure both are on `PATH`.

- **Windows:** `choco install ffmpeg`
- **Debian / Ubuntu:** `apt install ffmpeg`

### Model weights

The pose tagger needs `yolov8n-pose.pt`. It is fetched automatically by
`ultralytics` on first run; no manual download required.

## 3. Run

```sh
python main.py
```

On first launch there is no library configured. Point Setpiece at a
folder of video clips — either through the desktop window or by POSTing
to `/api/library/scan` from the web UI. The choice is saved to
`~/.setpiece/settings.json`.

The web control UI is served at `http://<this-machine>:8765` — open it
on a phone or tablet on the same LAN.

## 4. Tag the library

Before the picker can make editorial decisions it needs the library
analysed. Run the ingest pipeline:

```sh
python ingest.py --root /path/to/clips
```

This runs every tagger stage (path tokens, color, geometry, motion,
symmetry, pose, intro detection, CLIP embeddings). Every stage is
idempotent and resumable, so re-run it after adding new clips and you
only pay for the new files.

Run a single stage with `--only`, or skip stages with `--skip`:

```sh
python ingest.py --root /path/to/clips --only color
python ingest.py --root /path/to/clips --skip pose --skip vision
```

Tags are stored in a SQLite database at `~/.setpiece/path_tags.db3`.

## Security / threat model

Setpiece is a **single-operator tool for a trusted LAN**, and the
control server is built that way:

- The HTTP control server on port 8765 has **no authentication**. Anyone
  who can reach the port can control playback, banks and the library.
- It allows cross-origin requests (`Access-Control-Allow-Origin: *`), so
  a web page open in a browser on the same network can also reach it.

This is fine for the intended use — your own machine, your own LAN, at a
gig. **Do not expose port 8765 to the open internet** or run Setpiece on
an untrusted network. If you need that, put it behind a reverse proxy
with auth. File-serving routes (thumbnails, filmstrips, static assets)
are validated against path traversal; the open surface is the control
API itself, by design.

## Platform notes

Setpiece is developed on Windows today; a Debian-class Linux port is in
progress. Platform-specific code (windowless subprocess flags, the HID
backend, WASAPI audio loopback) is isolated behind safe fallbacks, but
expect rough edges on Linux until the port lands.
