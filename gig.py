"""
gig.py — one-command launcher for the Resolume control rig.

This is the "walk up and go" entry point. Double-click gig.bat (Windows)
or run ./gig.sh (Linux), and it:
  1. starts the standalone Resolume panel on a fixed port,
  2. prints the iPad URL(s) with the LAN address flagged,
  3. opens the control page in this machine's browser (best-effort),
  4. runs the pre-flight check so you see Arena's status before the set.

No PID juggling, no port hunting — close the window to stop. Everything
is standalone (imports the panel + bridge modules only); it never touches
the main CineQ app, so it's safe to run alongside the rig or by itself.

USAGE
    python gig.py                 # port 8770, opens browser
    python gig.py --port 8772     # custom port
    python gig.py --no-browser    # don't auto-open (headless / second screen)
    python gig.py --arena-host 192.168.4.50   # Arena on another machine
"""

import argparse
import sys
import threading
import webbrowser

import resolume_panel

DEFAULT_PORT = 8770


def _open_browser_when_up(port: float, delay: float = 1.2):
    """Open the local control page shortly after the server starts. Runs in
    a daemon thread so it never blocks shutdown; best-effort (a headless box
    with no browser just no-ops)."""
    def _go():
        try:
            webbrowser.open(f"http://127.0.0.1:{int(port)}/")
        except Exception:
            pass
    t = threading.Timer(delay, _go)
    t.daemon = True
    t.start()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="One-command launcher for the Resolume control rig.")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"panel web port (default {DEFAULT_PORT})")
    ap.add_argument("--arena-host", default="127.0.0.1",
                    help="Arena OSC host (default 127.0.0.1)")
    ap.add_argument("--osc-port", type=int, default=7000,
                    help="Arena OSC input port (default 7000)")
    ap.add_argument("--rest-base", default="http://localhost:8080/api/v1",
                    help="Arena REST base URL")
    ap.add_argument("--bind", default="0.0.0.0",
                    help="interface to bind (default 0.0.0.0 = all)")
    ap.add_argument("--no-browser", action="store_true",
                    help="don't auto-open the control page on this machine")
    a = ap.parse_args()

    if not a.no_browser:
        _open_browser_when_up(a.port)

    try:
        resolume_panel.serve(port=a.port, arena_host=a.arena_host,
                             osc_port=a.osc_port, rest_base=a.rest_base,
                             bind=a.bind)
    except OSError as e:
        # Almost always "port already in use" — a panel is probably already
        # running. Tell the operator plainly instead of dumping a traceback.
        print("=" * 60)
        print(f" Could not start on port {a.port}: {e}")
        print(" A panel may already be running on that port. Either open")
        print(f"   http://127.0.0.1:{a.port}/  (it's likely already up)")
        print(" or relaunch with a different --port.")
        print("=" * 60)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
