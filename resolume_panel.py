"""
Standalone iPad control panel for Resolume Arena.

WHY STANDALONE (not wired into main.py):
  CineQ's main.py is a ~12.5k-line live-prototyped god-object. Bolting a
  control surface into it means merge pain while it's actively evolving,
  and couples the panel's uptime to the whole rig. Instead this is its
  OWN process: it imports only the clean, self-tested Resolume bridge
  modules and talks straight to Arena. If CineQ crashes mid-set, the iPad
  still controls Arena from here. That decoupling is a bar-show win.

WHAT IT IS:
  A tiny stdlib (ThreadingHTTPServer) web server that:
    - serves the touch panel (resolume_panel/ : index.html, style.css, app.js)
    - exposes POST /api/* endpoints that drive Arena via:
        resolume_out.ResolumeBridge   (OSC 7000 -- commands: fire, opacity,
                                        crossfader, panic-to-black)
        resolume_state.ResolumeState  (REST 8080 -- reads: layers, master,
                                        the status light, the clip grid)

USAGE:
    python resolume_panel.py                 # binds 0.0.0.0:8770
    python resolume_panel.py --port 8770 --arena-host 127.0.0.1 --osc-port 7000
  Then on the iPad (same network): http://<this-machine-ip>:8770/

  ThreadingHTTPServer (never HTTPServer): the panel polls /api/state a few
  times a second from possibly several tabs; a single-threaded server would
  serialize and stutter.

BAR-RIG NOTE: for venue stability, run the iPad over a wired link (USB-C/
Lightning -> Ethernet adapter -> a small travel router the laptop is also
on) instead of venue Wi-Fi. OSC/HTTP don't care if the link is wired; they
care that it doesn't drop. Same panel, far fewer dropouts.
"""

import argparse
import json
import logging
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from resolume_out import ResolumeBridge
from resolume_state import ResolumeState

logger = logging.getLogger("resolume_panel")

PANEL_DIR = Path(__file__).parent / "resolume_panel"

# Max POST body we'll read — the panel only sends tiny JSON.
_MAX_BODY = 64 * 1024

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}


class PanelHandler(BaseHTTPRequestHandler):
    # Injected by serve():
    bridge: ResolumeBridge = None
    state: ResolumeState = None

    def log_message(self, fmt, *args):
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # -- helpers ------------------------------------------------------------

    def _send(self, code, body=b"", content_type="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # iOS Safari caches aggressively + ignores no-cache sometimes;
        # belt-and-suspenders on every response.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client navigated away mid-write; not actionable

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def _read_json(self):
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
        except (TypeError, ValueError):
            return {}
        if n <= 0 or n > _MAX_BODY:
            return {}
        try:
            raw = self.rfile.read(n).decode("utf-8")
            d = json.loads(raw) if raw else {}
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    # -- GET: static files --------------------------------------------------

    def do_GET(self):
        path = self.path.split("?", 1)[0]

        # Thumbnail proxy: /thumb/<clipId> -> Arena's PNG. The iPad browser
        # can't reliably reach Arena's :8080 cross-origin, so we fetch it
        # server-side and relay the bytes (cached hard — thumbs rarely
        # change, and the panel cache-busts with ?u=<last_update> when they
        # do).
        if path.startswith("/thumb/"):
            cid = path[len("/thumb/"):]
            try:
                ct, data = self.state.thumbnail_by_id(int(cid))
            except (TypeError, ValueError):
                ct, data = None, None
            if data:
                self.send_response(200)
                self.send_header("Content-Type", ct or "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(404, b"no thumbnail", "text/plain")
            return

        if path == "/":
            path = "/index.html"
        fname = path.lstrip("/")
        # Confine to the panel dir (no traversal).
        target = (PANEL_DIR / fname).resolve()
        try:
            target.relative_to(PANEL_DIR.resolve())
        except ValueError:
            self._send(403, b"forbidden", "text/plain")
            return
        if not target.is_file():
            self._send(404, b"not found", "text/plain")
            return
        ctype = _CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self._send(200, target.read_bytes(), ctype)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # -- POST: control API --------------------------------------------------

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        d = self._read_json()
        b, s = self.bridge, self.state
        try:
            if path == "/api/state":
                snap = s.snapshot()
                snap["bridge_enabled"] = bool(getattr(b, "_enabled", False))
                self._json(snap)

            elif path == "/api/clip_names":
                layer = int(d.get("layer", 1) or 1)
                self._json({"ok": True, "layer": layer,
                            "clips": s.clip_names(layer)})

            elif path == "/api/panic":
                b.panic_black()
                self._json({"ok": True, "blacked_out": True})

            elif path == "/api/restore":
                b.restore_output()
                self._json({"ok": True, "blacked_out": False})

            elif path == "/api/layer_opacity":
                layer = int(d.get("layer", 1))
                op = float(d.get("opacity", 1.0))
                b.set_layer_opacity(layer, op)
                self._json({"ok": True, "layer": layer, "opacity": op})

            elif path == "/api/layer_bypass":
                layer = int(d.get("layer", 1))
                byp = bool(d.get("bypassed", True))
                b.set_layer_bypassed(layer, byp)
                self._json({"ok": True, "layer": layer, "bypassed": byp})

            elif path == "/api/layer_clear":
                layer = int(d.get("layer", 1))
                b.clear_layer(layer)
                self._json({"ok": True, "layer": layer, "cleared": True})

            elif path == "/api/crossfader":
                phase = float(d.get("phase", 0.0))
                b.set_crossfader(phase)
                self._json({"ok": True, "phase": phase})

            elif path == "/api/master":
                # Composition master 0..1 — smooth full-output ride between
                # tracks. Pairs with panic (instant 0): this is the gradual
                # version. Panic reconciles against the same master value.
                level = float(d.get("level", 1.0))
                b.set_composition_master(level)
                self._json({"ok": True, "level": level})

            elif path == "/api/connect_clip":
                layer = int(d.get("layer", 1))
                col = int(d.get("column", 1))
                b.connect_clip(layer, col)
                self._json({"ok": True, "layer": layer, "column": col})

            elif path == "/api/fire_column":
                col = int(d.get("column", 1))
                b.connect_column(col)
                self._json({"ok": True, "column": col})

            else:
                self._json({"ok": False, "error": f"unknown: {path}"}, 404)
        except (TypeError, ValueError) as e:
            self._json({"ok": False, "error": f"bad args: {e}"}, 400)
        except Exception as e:
            logger.exception("panel endpoint %s failed", path)
            self._json({"ok": False, "error": str(e)}, 500)


def _local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _all_ipv4():
    """Every IPv4 address bound on this host, so the operator can pick the
    one the iPad is actually on. _local_ip() returns only the default-route
    address, which may be a VPN/secondary adapter the iPad can't reach
    (learned the hard way: it printed a 10.x VPN IP while the iPad was on
    192.168.x)."""
    addrs = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            addrs.add(info[4][0])
    except Exception:
        pass
    addrs.add(_local_ip())
    # Drop loopback + link-local; never the iPad's route.
    return sorted({a for a in addrs
                   if not a.startswith("127.")
                   and not a.startswith("169.254.")})


def serve(port=8770, arena_host="127.0.0.1", osc_port=7000,
          rest_base="http://localhost:8080/api/v1", bind="0.0.0.0"):
    PanelHandler.bridge = ResolumeBridge(host=arena_host, port=osc_port,
                                         enabled=True)
    PanelHandler.state = ResolumeState(rest_base=rest_base)

    httpd = ThreadingHTTPServer((bind, port), PanelHandler)
    print("=" * 60)
    print(" Resolume control panel")
    print(f"   open on this machine : http://127.0.0.1:{port}/")
    print("   open on iPad/phone   : try one of these (same Wi-Fi/LAN):")
    for a in _all_ipv4():
        tag = "  <-- LAN (most likely)" if a.startswith("192.168.") else ""
        print(f"        http://{a}:{port}/{tag}")
    print(f"   commanding Arena OSC : {arena_host}:{osc_port}")
    print(f"   reading Arena REST   : {rest_base}")
    print("-" * 60)
    # Pre-flight: a plain green/red readout so you know the rig is wired
    # BEFORE the first track, not dark mid-set.
    print(" PRE-FLIGHT")
    pf = PanelHandler.state.preflight()
    for c in pf["checks"]:
        mark = "[ OK ]" if c["ok"] else ("[FAIL]" if c["critical"] else "[warn]")
        print(f"   {mark} {c['name']}: {c['detail']}")
    if pf["ok"]:
        print("   --> READY. Panel is live; open the URL above on the iPad.")
    else:
        print("   --> NOT READY. Start Arena, enable Preferences > Webserver")
        print("       (:8080) and OSC input (:7000), then restart this panel.")
        print("       (Panel still serves; the status light shows offline.)")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down panel")
    finally:
        httpd.server_close()
        PanelHandler.bridge.close()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Standalone Resolume iPad panel")
    ap.add_argument("--port", type=int, default=8770,
                    help="panel web port (default 8770)")
    ap.add_argument("--arena-host", default="127.0.0.1",
                    help="Arena OSC host (default 127.0.0.1)")
    ap.add_argument("--osc-port", type=int, default=7000,
                    help="Arena OSC input port (default 7000)")
    ap.add_argument("--rest-base", default="http://localhost:8080/api/v1",
                    help="Arena REST base URL")
    ap.add_argument("--bind", default="0.0.0.0",
                    help="interface to bind (default 0.0.0.0 = all)")
    a = ap.parse_args()
    serve(port=a.port, arena_host=a.arena_host, osc_port=a.osc_port,
          rest_base=a.rest_base, bind=a.bind)


if __name__ == "__main__":
    main()
