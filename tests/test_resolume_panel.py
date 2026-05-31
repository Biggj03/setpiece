"""Tests for the standalone Resolume control panel (resolume_panel.py).

The panel is the shareable product's iPad control surface, so it needs
real coverage — not just the inline self-tests on the bridge modules.

These run WITHOUT Resolume Arena: we inject fake bridge + state objects
into PanelHandler (the same class attributes serve() sets at runtime),
start a real ThreadingHTTPServer on a free port, and drive the HTTP
routes. That mirrors tests/test_http_server.py and keeps the suite
CI-safe (no hardware, no network peer).
"""
import json
import socket
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer

import resolume_panel
import resolume_state
import resolume_out


# ── fakes ──────────────────────────────────────────────────────────────

class FakeBridge:
    """Records the high-level commands the panel issues."""
    def __init__(self):
        self._enabled = True
        self.calls = []

    def panic_black(self): self.calls.append(("panic",))
    def restore_output(self): self.calls.append(("restore",))
    def set_composition_master(self, v): self.calls.append(("master", v))
    def set_layer_opacity(self, layer, op): self.calls.append(("opacity", layer, op))
    def set_layer_bypassed(self, layer, b): self.calls.append(("bypass", layer, b))
    def clear_layer(self, layer): self.calls.append(("clear", layer))
    def set_crossfader(self, p): self.calls.append(("xfade", p))
    def connect_clip(self, layer, col): self.calls.append(("connect", layer, col))
    def connect_column(self, col): self.calls.append(("column", col))


class FakeState:
    """Returns canned snapshots so the panel has data without Arena."""
    SNAP = {
        "reachable": True, "product": "Arena 7", "master": 1.0,
        "crossfader": 0.0, "tempo": 128.0,
        "layers": [{"index": 1, "name": "L1", "opacity": 0.5,
                    "bypassed": False, "solo": False, "loaded": 3,
                    "columns": 8, "active_clip": 2}],
    }

    def snapshot(self): return dict(self.SNAP)
    def clip_names(self, layer):
        return [{"column": 1, "name": "clip-a", "loaded": True, "thumb_id": 111},
                {"column": 2, "name": None, "loaded": False, "thumb_id": None}]
    def thumbnail_by_id(self, cid):
        return ("image/png", b"\x89PNG\r\n\x1a\n") if cid == 111 else (None, None)


# ── server harness ─────────────────────────────────────────────────────

def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _Server:
    """Context manager: a live panel server on localhost with fakes wired."""
    def __init__(self, bridge, state):
        resolume_panel.PanelHandler.bridge = bridge
        resolume_panel.PanelHandler.state = state
        self.port = _free_port()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port),
                                         resolume_panel.PanelHandler)

    def __enter__(self):
        self.t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, obj=None):
        data = json.dumps(obj or {}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode())

    def get(self, path):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}{path}", timeout=5) as r:
            return r.status, r.read()


# ── route dispatch tests ───────────────────────────────────────────────

def test_state_endpoint_returns_snapshot_with_bridge_flag():
    with _Server(FakeBridge(), FakeState()) as s:
        code, body = s.post("/api/state")
    assert code == 200
    assert body["reachable"] is True
    assert body["bridge_enabled"] is True       # injected by the handler
    assert len(body["layers"]) == 1


def test_panic_and_restore_call_bridge():
    b = FakeBridge()
    with _Server(b, FakeState()) as s:
        s.post("/api/panic")
        s.post("/api/restore")
    assert ("panic",) in b.calls
    assert ("restore",) in b.calls


def test_master_endpoint_passes_level_through():
    b = FakeBridge()
    with _Server(b, FakeState()) as s:
        code, body = s.post("/api/master", {"level": 0.42})
    assert code == 200 and body["ok"] is True
    assert ("master", 0.42) in b.calls


def test_layer_opacity_dispatches():
    b = FakeBridge()
    with _Server(b, FakeState()) as s:
        s.post("/api/layer_opacity", {"layer": 2, "opacity": 0.7})
    assert ("opacity", 2, 0.7) in b.calls


def test_connect_clip_dispatches():
    b = FakeBridge()
    with _Server(b, FakeState()) as s:
        s.post("/api/connect_clip", {"layer": 1, "column": 5})
    assert ("connect", 1, 5) in b.calls


def test_clip_names_returns_grid():
    with _Server(FakeBridge(), FakeState()) as s:
        code, body = s.post("/api/clip_names", {"layer": 1})
    assert code == 200
    cols = [c["column"] for c in body["clips"]]
    assert cols == [1, 2]


def test_unknown_post_route_404s():
    with _Server(FakeBridge(), FakeState()) as s:
        try:
            s.post("/api/does_not_exist")
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404


def test_get_serves_index_html():
    with _Server(FakeBridge(), FakeState()) as s:
        code, body = s.get("/")
    assert code == 200
    assert b"<!DOCTYPE html>" in body or b"<html" in body


def test_thumbnail_proxy_relays_png_bytes():
    with _Server(FakeBridge(), FakeState()) as s:
        code, body = s.get("/thumb/111")
    assert code == 200
    assert body.startswith(b"\x89PNG")


def test_thumbnail_proxy_404s_when_absent():
    with _Server(FakeBridge(), FakeState()) as s:
        try:
            s.get("/thumb/999")
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404


# ── pure unit tests (no server) ────────────────────────────────────────

def test_all_ipv4_excludes_loopback_and_linklocal():
    addrs = resolume_panel._all_ipv4()
    assert isinstance(addrs, list)
    assert all(not a.startswith("127.") for a in addrs)
    assert all(not a.startswith("169.254.") for a in addrs)


def test_state_param_unwraps_resolume_shapes():
    # Arena wraps most params as {"value": X}; _param must unwrap or pass through.
    assert resolume_state._param({"master": {"value": 0.7}}, "master") == 0.7
    assert resolume_state._param({"master": 0.7}, "master") == 0.7
    assert resolume_state._param({}, "master") is None


def test_state_snapshot_degrades_gracefully_on_dead_arena():
    # Pointed at a dead port: reachable False, empty layers, no raise.
    st = resolume_state.ResolumeState(rest_base="http://127.0.0.1:9/api/v1",
                                      timeout=0.3)
    snap = st.snapshot()
    assert snap["reachable"] is False
    assert snap["layers"] == []


def test_bridge_panic_restore_are_noop_safe_when_disabled():
    # A disabled bridge must never raise (dead-renderer discipline).
    b = resolume_out.ResolumeBridge(enabled=False)
    b.panic_black()
    b.restore_output()
    b.set_composition_master(0.5)
    b.close()
