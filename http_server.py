"""
HTTP server for iPad remote control.
ThreadingHTTPServer (not HTTPServer) — single-threaded blocks SSE/long-polling.
iOS Safari cache-busting headers on every response.

The /preview/<n>.mjpg routes hold their thread open for the lifetime of
the iPad <img> tag (multipart/x-mixed-replace). That's the entire
reason this is the threading variant — single-threaded would mean the
first <img> tag connection would deadlock /api/state polling forever.
"""

import json
import logging
import re
import socket
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, unquote, urlparse

from app_state import AppState
import thumbnails

# clip ids are uuid4 strings; lock the route down so we never read
# arbitrary paths off disk.
_CLIP_ID_RE = re.compile(r"^[0-9a-fA-F-]{8,64}$")

# Preview MJPEG route: /preview/<deck_idx>.mjpg
_PREVIEW_RE = re.compile(r"^/preview/(\d+)\.mjpg$")
# Boundary string used in multipart/x-mixed-replace. Picking "frame"
# matches the Hacker News / OpenCV / common Safari-tested convention.
_MJPEG_BOUNDARY = b"frame"

# Hard cap on POST body size. The control endpoints take tiny JSON
# payloads; anything larger is rejected with 413 before being read into
# memory (this is a threaded server — unbounded reads are a DoS vector).
_MAX_POST_BYTES = 256 * 1024

logger = logging.getLogger(__name__)

# Embedded static dir (for HTML/JS)
STATIC_DIR = Path(__file__).parent / "static"


# Read-only handle to the path-tags SQLite DB (color/palette tags).
_PATH_TAGS_DB = Path.home() / ".setpiece" / "path_tags.db3"


class VJRequestHandler(BaseHTTPRequestHandler):
    """Handles iPad requests. Serves HTML, JSON state, and accepts commands."""

    state: AppState = None
    callbacks: dict = {}
    # Optional handle to the PreviewStreamManager. None when disabled
    # (mock server, ffmpeg missing, etc) — handlers degrade to 503.
    preview_manager: object = None

    def log_message(self, format, *args):
        # Quieter logs (no per-request stdout spam)
        logger.debug(f"{self.address_string()} - {format % args}")

    def _send_no_cache(self, status: int = 200, content_type: str = "application/json"):
        """Send headers with iOS Safari cache-busting."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/" or path == "/index.html":
            self._serve_static("control.html", "text/html; charset=utf-8")
        elif path == "/control.css":
            self._serve_static("control.css", "text/css; charset=utf-8")
        elif path == "/control.js":
            self._serve_static("control.js", "application/javascript; charset=utf-8")
        elif path == "/magic.css":
            self._serve_static("magic.css", "text/css; charset=utf-8")
        elif path == "/magic.js":
            self._serve_static("magic.js", "application/javascript; charset=utf-8")
        elif path == "/overlay.html":
            self._serve_static("overlay.html", "text/html; charset=utf-8")
        elif path == "/cdj.html" or path == "/cdj":
            self._serve_static("cdj.html", "text/html; charset=utf-8")
        elif path == "/cdj.css":
            self._serve_static("cdj.css", "text/css; charset=utf-8")
        elif path == "/cdj.js":
            self._serve_static("cdj.js", "application/javascript; charset=utf-8")
        elif path == "/mk2_layout.html" or path == "/mk2":
            self._serve_static("mk2_layout.html", "text/html; charset=utf-8")
        elif path == "/overlay.css":
            self._serve_static("overlay.css", "text/css; charset=utf-8")
        elif path == "/overlay.js":
            self._serve_static("overlay.js", "application/javascript; charset=utf-8")
        elif path == "/api/state":
            self._serve_state()
        elif path == "/api/file/colors":
            self._serve_file_colors()
        elif path == "/api/set-arc":
            self._serve_set_arc()
        elif path.startswith("/thumbnails/") and path.endswith(".jpg"):
            self._serve_thumbnail(path[len("/thumbnails/"):-len(".jpg")])
        elif path.startswith("/filmstrips/") and path.endswith(".jpg"):
            self._serve_filmstrip(path[len("/filmstrips/"):-len(".jpg")])
        elif path.startswith("/lib_thumbnails/") and path.endswith(".jpg"):
            self._serve_lib_thumbnail(path[len("/lib_thumbnails/"):-len(".jpg")])
        elif _PREVIEW_RE.match(path):
            m = _PREVIEW_RE.match(path)
            self._serve_preview_mjpeg(int(m.group(1)))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        # CORS preflight for cross-origin control clients. This
        # server is LAN-only and exposes no auth.
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except (TypeError, ValueError):
            self._send_no_cache(400)
            self.wfile.write(b'{"ok":false,"error":"bad Content-Length"}')
            return
        if length > _MAX_POST_BYTES:
            self._send_no_cache(413)
            self.wfile.write(b'{"ok":false,"error":"request body too large"}')
            return
        try:
            body = self.rfile.read(length).decode("utf-8") if length else ""
        except Exception:
            self._send_no_cache(400)
            self.wfile.write(b'{"ok":false,"error":"could not read body"}')
            return
        if body:
            try:
                data = json.loads(body)
            except ValueError:
                # json.JSONDecodeError is a ValueError subclass.
                self._send_no_cache(400)
                self.wfile.write(b'{"ok":false,"error":"malformed JSON body"}')
                return
            if not isinstance(data, dict):
                self._send_no_cache(400)
                self.wfile.write(
                    b'{"ok":false,"error":"JSON body must be an object"}')
                return
        else:
            data = {}

        # Dispatch to registered callback
        action = path.lstrip("/").replace("api/", "")
        # Log iPad-facing bank/set-arc/tag actions so we can debug
        # client-side "failed" reports. Skipping high-frequency stuff
        # (play, pause, flip) to keep the log readable.
        if any(action.startswith(p) for p in (
            "bank/", "set-arc", "path_tags/", "mk2/"
        )):
            preview = (json.dumps(data)[:120] + "...") if data else "{}"
            logger.info(f"[ipad-action] POST /api/{action}  body={preview}")
        if action in self.callbacks:
            try:
                result = self.callbacks[action](data)
                self._send_no_cache(200)
                self.wfile.write(json.dumps({"ok": True, "result": result}).encode("utf-8"))
            except Exception as e:
                logger.error(f"Action {action} failed: {e}", exc_info=True)
                self._send_no_cache(500)
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
        else:
            logger.warning(f"[ipad-action] UNKNOWN action: {action!r}")
            self._send_no_cache(404)
            self.wfile.write(json.dumps({"ok": False, "error": f"unknown action: {action}"}).encode("utf-8"))

    def _serve_static(self, filename: str, content_type: str):
        """Serve a file from the static directory."""
        path = STATIC_DIR / filename
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(f"Not found: {filename}".encode("utf-8"))
            return
        try:
            content = path.read_bytes()
            # iOS Safari ignores no-cache when assets are referenced by
            # static URL — rewrite the HTML to inject ?v=<mtime> on the
            # CSS/JS so any save instantly invalidates the cached copies.
            # Bust both control.* and cdj.* (each HTML page has its own
            # sibling assets) -- previously only control.* was busted
            # so cdj.html's badge-arc render was stale on iPad.
            if filename.endswith(".html"):
                def _mtime(name: str) -> int:
                    try:
                        return int((STATIC_DIR / name).stat().st_mtime)
                    except Exception:
                        return 0
                text = content.decode("utf-8")
                for asset in ("control.css", "control.js",
                              "cdj.css", "cdj.js"):
                    v = _mtime(asset)
                    if v:
                        # Match both href= (css) and src= (js) refs.
                        text = text.replace(f'href="/{asset}"',
                                            f'href="/{asset}?v={v}"')
                        text = text.replace(f'src="/{asset}"',
                                            f'src="/{asset}?v={v}"')
                content = text.encode("utf-8")
            self._send_no_cache(200, content_type)
            self.wfile.write(content)
        except Exception as e:
            logger.error(f"Error serving {filename}: {e}")
            self.send_response(500)
            self.end_headers()

    def _serve_thumbnail(self, clip_id: str):
        """Serve the JPEG thumbnail for a clip. 404 silently if missing."""
        # Defensive: only allow uuid-shaped ids; nothing with slashes/dots.
        if not _CLIP_ID_RE.match(clip_id):
            self.send_response(404)
            self.end_headers()
            return
        path = thumbnails.thumbnail_path(clip_id)
        if not path.exists():
            # No body — JS will fall back to placeholder styling.
            self.send_response(404)
            self.end_headers()
            return
        try:
            data = path.read_bytes()
        except Exception as e:
            logger.debug(f"Thumbnail read failed for {clip_id}: {e}")
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        # Thumbnails never change (clip_id is a uuid). Cache hard, but
        # respect the URL query string the client uses to bust on rebuild.
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            # iPad navigated away mid-write; not actionable.
            pass

    def _serve_filmstrip(self, strip_hash: str):
        """Serve the JPEG filmstrip for a deck slot. 404 silently if missing."""
        if not thumbnails.FILMSTRIP_ID_RE.match(strip_hash):
            self.send_response(404)
            self.end_headers()
            return
        path = thumbnails.filmstrip_path(strip_hash)
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        try:
            data = path.read_bytes()
        except Exception as e:
            logger.debug(f"Filmstrip read failed for {strip_hash}: {e}")
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        # Filmstrip hash is content-addressed → cache forever.
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass

    def _serve_lib_thumbnail(self, file_hash: str):
        """Serve a JPEG thumbnail for a library file. Generates on demand
        if missing. 404 silently on any failure (bad hash, file gone,
        ffmpeg missing, etc) — the iPad UI keeps the placeholder."""
        if not thumbnails.LIB_THUMB_ID_RE.match(file_hash):
            self.send_response(404)
            self.end_headers()
            return
        path = thumbnails.lib_thumbnail_path(file_hash)
        if not path.exists():
            # Try on-demand generation. We need the source filepath; ask
            # AppState to resolve hash -> abs path within the current
            # library folder. This is a single shot per missing thumb,
            # not a hot loop, so doing it in the request thread is fine.
            src = self._resolve_lib_hash_to_path(file_hash)
            if src:
                try:
                    thumbnails.generate_library_thumbnail(src)
                except Exception as e:
                    logger.debug(f"On-demand lib thumb gen failed: {e}")
            if not path.exists():
                self.send_response(404)
                self.end_headers()
                return
        try:
            data = path.read_bytes()
        except Exception as e:
            logger.debug(f"Lib thumb read failed for {file_hash}: {e}")
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        # Hash is content-addressed (rename = new hash), so cache hard.
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass

    def _resolve_lib_hash_to_path(self, file_hash: str) -> Optional[str]:
        """Look up a library file's absolute path from its 16-char hash.
        Walks the CURRENT library folder's file list (already published by
        main.py via AppState.set_library). Cheap — list is ~hundreds at
        worst. Returns None if no match in the current folder."""
        if not self.state:
            return None
        snap_files = list(getattr(self.state, "library_files", []) or [])
        folder = getattr(self.state, "library_folder", "") or ""
        if not folder:
            return None
        for entry in snap_files:
            # Defensive: support both legacy str entries and new dict entries.
            if isinstance(entry, dict):
                h = entry.get("hash")
                name = entry.get("name") or ""
            else:
                # Legacy: compute hash on the fly to find a match.
                name = str(entry)
                h = thumbnails.lib_thumbnail_hash(str(Path(folder) / name))
            if h == file_hash and name:
                return str(Path(folder) / name)
        return None

    def _serve_preview_mjpeg(self, deck_idx: int):
        """Stream the deck's MJPEG preview as multipart/x-mixed-replace.

        Safari natively swaps an <img>'s pixels on each multipart part
        — no JS needed on the client. The connection is held open for
        the full lifetime of the <img> tag; we exit when the iPad
        navigates away (write to wfile raises) or the deck stream is
        stopped (subscribe() generator returns).

        Because this thread blocks for the lifetime of the connection,
        every request gets its own thread (ThreadingHTTPServer). The
        single-threaded HTTPServer would deadlock /api/state polling
        the moment the first <img> tag attached.
        """
        if not (0 <= deck_idx < 4):
            self.send_response(404)
            self.end_headers()
            return
        mgr = self.preview_manager
        if mgr is None or not getattr(mgr, "available", False):
            # Preview pipeline disabled (no ffmpeg, mock server, etc).
            # 503 lets the iPad's onerror swap to placeholder cleanly.
            self.send_response(503)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        # If the deck has no stream running, also 503 — same iPad
        # fallback. Avoids opening a long-poll for an empty slot.
        if not mgr.has_stream(deck_idx):
            self.send_response(503)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return

        # Send headers. Note: NO Content-Length (stream is open-ended).
        # The boundary string here MUST match what we write between
        # parts below (RFC 1341 / browser convention).
        try:
            self.send_response(200)
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY.decode()}",
            )
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            # Keep iOS from buffering the stream.
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
        except Exception:
            # iPad already gone before we got headers out.
            return

        # Push frames until the consumer breaks. subscribe() is a
        # generator that auto-cleans on GeneratorExit / return.
        try:
            for frame in mgr.subscribe(deck_idx):
                if not frame:
                    continue
                # RFC 1341 / browser-tested form. The leading \r\n is
                # part of the previous part's terminator, conventionally
                # included so the very first boundary is well-formed
                # even though it lacks one.
                header = (
                    b"\r\n--" + _MJPEG_BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode("ascii") + b"\r\n\r\n"
                )
                try:
                    self.wfile.write(header)
                    self.wfile.write(frame)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    # iPad disconnected — generator's finally clause
                    # will unsubscribe us. Just exit the loop quietly.
                    break
                except Exception as e:
                    logger.debug(f"preview mjpeg write error (deck {deck_idx}): {e}")
                    break
        except Exception as e:
            # subscribe() shouldn't raise, but be defensive.
            logger.debug(f"preview mjpeg loop error (deck {deck_idx}): {e}")

    def _serve_set_arc(self):
        """GET /api/set-arc -- returns {enabled, auto, phase, label,
        phases, detect}. POST /api/set-arc/auto toggles auto mode.
        Cycling happens via MK2 ALL button (bit 14) or future
        SHIFT+ALL for backwards cycle."""
        app = getattr(VJRequestHandler, "app", None)
        out = {"ok": True, "enabled": False, "auto": False,
               "phase": "opening", "label": "OPENING",
               "phases": ["opening", "build", "peak", "breakdown"],
               "detect": None}
        try:
            from set_arc import label_for, PHASES
            out["phases"] = PHASES
            if app is not None:
                out["enabled"] = bool(getattr(app, "_set_arc_enabled", False))
                out["auto"] = bool(getattr(app, "_set_arc_auto", False))
                out["phase"] = str(getattr(app, "_set_arc_phase", "opening"))
                asa = getattr(app, "_auto_set_arc", None)
                if asa is not None:
                    try:
                        out["detect"] = asa.snapshot()
                    except Exception:
                        pass
                    # Prediction lookahead: what would the auto-detector
                    # pick if we asked it RIGHT NOW (ignoring cooldown)?
                    # Gives the iPad a "→ PEAK ↗" hint badge even when
                    # auto mode is OFF.
                    try:
                        ar = getattr(app, "audio_reactive", None)
                        # current_bpm is a method on one backend and a
                        # float attribute on another (BeatNet). Handle both.
                        _cb = getattr(ar, "current_bpm", None) if ar else None
                        bpm = float((_cb() if callable(_cb) else _cb) or 0.0)
                        cur = out["phase"]
                        pred_phase, pred_conf, pred_reason = (
                            asa.predict_next_phase(bpm, cur)
                        )
                        out["predicted_phase"] = pred_phase
                        out["predicted_confidence"] = round(pred_conf, 2)
                        out["predicted_reason"] = pred_reason
                        out["predicted_is_change"] = (
                            pred_phase != cur and pred_conf > 0.5
                        )
                    except Exception as e:
                        logger.warning(
                            f"predict_next_phase failed: "
                            f"{type(e).__name__}: {e}", exc_info=True)
            out["label"] = label_for(out["phase"])
        except Exception as e:
            logger.warning(
                f"set-arc endpoint failed: {type(e).__name__}: {e}",
                exc_info=True)
        self._send_no_cache(200)
        self.wfile.write(json.dumps(out).encode("utf-8"))

    def _serve_set_arc_auto_toggle(self):
        """POST /api/set-arc/auto -- toggle auto-detect on/off."""
        app = getattr(VJRequestHandler, "app", None)
        out = {"ok": False, "auto": False}
        if app is not None and hasattr(app, "set_arc_auto_toggle"):
            try:
                out["auto"] = bool(app.set_arc_auto_toggle())
                out["ok"] = True
            except Exception as e:
                logger.debug(f"set-arc auto toggle failed: {e}")
                out["error"] = str(e)
        self._send_no_cache(200)
        self.wfile.write(json.dumps(out).encode("utf-8"))

    def _serve_file_colors(self):
        """GET /api/file/colors?path=<urlencoded-filepath>

        Returns the color/palette tags for a single file. Used by
        /cdj.html bank pads to tint their borders by clip palette.

        Response:
          { "ok": true,
            "palette": ["#aabbcc", "#ddeeff", "#001122"],  # dominant first
            "hue":     "red"|"orange"|...|"magenta"|null,
            "therm":   "warm"|"cool"|null,
            "tagged":  true|false }
          When tagged=false (color_tagger hasn't run on this file yet)
          all the color fields are null. JS treats that as "no tint".
        """
        try:
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
        except Exception:
            qs = ""
        params = parse_qs(qs, keep_blank_values=False)
        fp = (params.get("path") or [""])[0].strip()
        if not fp:
            self._send_no_cache(400)
            self.wfile.write(b'{"ok":false,"error":"missing path"}')
            return

        # Pull every color:/palette: tag for this filepath. Cheap.
        out = {
            "ok": True, "palette": None, "hue": None,
            "therm": None, "tagged": False,
        }
        try:
            # Read-only mode tolerates WAL siblings if a tagger
            # is still writing the DB.
            conn = sqlite3.connect(
                f"file:{_PATH_TAGS_DB}?mode=ro", uri=True, timeout=2.0,
            )
            try:
                rows = conn.execute(
                    "SELECT tag FROM file_tags WHERE filepath = ? AND ("
                    "tag LIKE 'color:%' OR tag LIKE 'palette:%')",
                    (fp,)
                ).fetchall()
            finally:
                conn.close()
            for (tag,) in rows:
                if tag == "color:warm":
                    out["therm"] = "warm"; out["tagged"] = True
                elif tag == "color:cool":
                    out["therm"] = "cool"; out["tagged"] = True
                elif tag.startswith("color:hue:"):
                    out["hue"] = tag[len("color:hue:"):]; out["tagged"] = True
                elif tag.startswith("palette:"):
                    # Stored as "palette:rrggbb:rrggbb:rrggbb"
                    parts = tag[len("palette:"):].split(":")
                    hexes = []
                    for p in parts[:3]:
                        p = p.strip()
                        if p and len(p) == 6:
                            hexes.append("#" + p)
                    if hexes:
                        out["palette"] = hexes
                        out["tagged"] = True
        except Exception as e:
            logger.debug(f"_serve_file_colors db error: {e}")

        self._send_no_cache(200)
        self.wfile.write(json.dumps(out).encode("utf-8"))

    def _serve_state(self):
        """Return current AppState as JSON."""
        if not self.state:
            self._send_no_cache(500)
            self.wfile.write(b'{"error": "no state"}')
            return
        snap = self.state.snapshot()
        # Inject set-arc state so /cdj.html can show a phase badge
        # without doing a second HTTP poll. Optional — falls back to
        # disabled/opening if the app ref isn't wired yet.
        app = getattr(VJRequestHandler, "app", None)
        if app is not None:
            try:
                from set_arc import label_for
                phase = str(getattr(app, "_set_arc_phase", "opening"))
                sa = {
                    "enabled": bool(getattr(app, "_set_arc_enabled", False)),
                    "auto": bool(getattr(app, "_set_arc_auto", False)),
                    "phase": phase,
                    "label": label_for(phase),
                }
                # Include the prediction lookahead so the iPad badge
                # can show "→ PEAK ↗" without an extra fetch.
                asa = getattr(app, "_auto_set_arc", None)
                if asa is not None:
                    try:
                        ar = getattr(app, "audio_reactive", None)
                        # current_bpm is a method on one backend and a
                        # float attribute on another (BeatNet). Handle both.
                        _cb = getattr(ar, "current_bpm", None) if ar else None
                        bpm = float((_cb() if callable(_cb) else _cb) or 0.0)
                        pp, pc, _pr = asa.predict_next_phase(bpm, phase)
                        sa["predicted_phase"] = pp
                        sa["predicted_confidence"] = round(pc, 2)
                        sa["predicted_is_change"] = (
                            pp != phase and pc > 0.5
                        )
                        # Forward-projected predictions: 30s and 60s
                        # ahead, using BPM slope extrapolation. Lets
                        # the iPad show "next: BUILD in 30s" hints.
                        try:
                            p30, c30, _r30, b30 = asa.predict_phase_at(
                                bpm, 30.0, phase
                            )
                            p60, c60, _r60, b60 = asa.predict_phase_at(
                                bpm, 60.0, phase
                            )
                            sa["lookahead"] = {
                                "slope_bpm_per_min": round(
                                    asa.bpm_slope_per_sec() * 60.0, 1
                                ),
                                "t30": {
                                    "phase": p30,
                                    "confidence": c30,
                                    "bpm": round(b30, 1),
                                    "is_change": (
                                        p30 != phase and c30 > 0.4
                                    ),
                                },
                                "t60": {
                                    "phase": p60,
                                    "confidence": c60,
                                    "bpm": round(b60, 1),
                                    "is_change": (
                                        p60 != phase and c60 > 0.4
                                    ),
                                },
                            }
                        except Exception:
                            pass
                    except Exception:
                        pass
                snap["set_arc"] = sa
            except Exception:
                snap["set_arc"] = {"enabled": False, "phase": "opening",
                                    "label": "OPENING"}
            # Stem-listener snapshot (when stem_daemon.py is feeding OSC).
            # Quiet failure: leaves snap["stems"] undefined if app doesn't
            # have the listener wired or it's disabled.
            try:
                if hasattr(app, "stem_status"):
                    snap["stems"] = app.stem_status()
            except Exception:
                pass
            # Mode dashboard — every operator-facing mode in one block
            # so the iPad can show a "what mode am I in" panel.
            try:
                if hasattr(app, "mode_summary"):
                    snap["modes"] = app.mode_summary()
            except Exception:
                pass
        self._send_no_cache(200)
        self.wfile.write(
            json.dumps(snap, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )


class HTTPServerThread:
    """Wraps ThreadingHTTPServer in a background thread."""

    def __init__(self, state: AppState, port: int = 8765,
                 preview_manager: object = None, lan: bool = False):
        self.state = state
        self.port = port
        # lan=False (default) binds 127.0.0.1 — only this machine can
        # reach the control server. lan=True binds 0.0.0.0 so a phone /
        # tablet on the LAN can connect. The server has no auth, so LAN
        # exposure is strictly opt-in (see SETUP.md security note).
        self.lan = lan
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.callbacks: dict = {}
        self.preview_manager = preview_manager

    def set_preview_manager(self, preview_manager: object):
        """Wire the preview manager after construction. Useful when the
        HTTP server is built before previews are ready (or vice versa)."""
        self.preview_manager = preview_manager
        VJRequestHandler.preview_manager = preview_manager

    def set_app_ref(self, app: object):
        """Wire the main VJApp instance so endpoints can read live
        runtime state (e.g. set_arc phase, drop-suppression timer).
        Optional -- endpoints fall back to safe defaults if not set."""
        VJRequestHandler.app = app

    def register(self, action: str, callback: Callable):
        """Register a callback for POST /api/<action>."""
        self.callbacks[action] = callback

    def start(self):
        """Start the HTTP server."""
        # Inject state and callbacks into handler class
        VJRequestHandler.state = self.state
        VJRequestHandler.callbacks = self.callbacks
        VJRequestHandler.preview_manager = self.preview_manager

        host = "0.0.0.0" if self.lan else "127.0.0.1"
        self.server = ThreadingHTTPServer((host, self.port), VJRequestHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        if self.lan:
            ip = self._get_local_ip()
            logger.info(f"HTTP server: http://{ip}:{self.port} "
                        f"(LAN mode — reachable from other devices)")
        else:
            ip = "127.0.0.1"
            logger.info(f"HTTP server: http://127.0.0.1:{self.port} "
                        f"(localhost only — pass --lan to allow phone/tablet)")
        return ip

    def stop(self):
        """Stop the server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    def _get_local_ip(self) -> str:
        """Best-effort local IP discovery."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
