"""Tests for resolume_stage: the file-URI rule, the twin-swap guard, the
staging mapping, and the actual REST wire contract (against a local stub
server standing in for Arena — no Arena needed)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import resolume_stage as rs


# ── file URI: the 3-slash Windows rule ──────────────────────────────────

def test_file_uri_three_slashes_forward_encoded():
    u = rs.file_uri(r"C:\Media Library\Show Reel #2 [Final].mp4")
    assert u.startswith("file:///C")
    assert "\\" not in u
    assert "%20" in u and "%23" in u   # space + '#' encoded


def test_file_uri_windows_drive_form_matches_live_proven_rule():
    # The exact shape proven live against Arena 7.25.2 (285-clip stage):
    # 3 slashes + the path with backslashes flipped and %-encoding applied
    # (the drive colon encodes to %3A — Arena accepts it).
    assert rs.file_uri(r"C:\Clips\a.mp4") == "file:///C%3A/Clips/a.mp4"


# ── twin-swap guard (the .mkv silent-black-frame protection) ────────────

def test_arena_safe_path_swaps_to_existing_twin(tmp_path):
    mkv = tmp_path / "Hero Clip.mkv"
    mp4 = tmp_path / "Hero Clip.mp4"
    mkv.write_bytes(b"")
    mp4.write_bytes(b"")
    assert rs.arena_safe_path(str(mkv)) == str(mp4)


def test_arena_safe_path_keeps_unopenable_without_twin(tmp_path):
    lone = tmp_path / "No Twin.webm"
    lone.write_bytes(b"")
    # No twin -> pass through; Arena will 412 it and the stager logs+skips,
    # which beats silently swapping to a file that doesn't exist.
    assert rs.arena_safe_path(str(lone)) == str(lone)


def test_arena_safe_path_leaves_safe_containers_alone(tmp_path):
    mp4 = tmp_path / "Fine.mp4"
    mp4.write_bytes(b"")
    assert rs.arena_safe_path(str(mp4)) == str(mp4)
    assert rs.arena_safe_path(r"E:\x\song.mov") == r"E:\x\song.mov"


# ── stage_working_set mapping (stubbed transport) ───────────────────────

class _ScriptedStager(rs.ResolumeStager):
    """open_into_slot scripted by filename; records calls."""

    def __init__(self, fail=()):
        super().__init__(rest_base="http://127.0.0.1:9/api/v1", timeout=0.1)
        self.opens = []
        self._fail = set(fail)

    def open_into_slot(self, layer, column, filepath):
        self.opens.append((layer, column, filepath))
        return filepath not in self._fail


def test_stage_working_set_maps_columns_in_order():
    st = _ScriptedStager()
    m = st.stage_working_set(["a.mp4", "b.mp4", "c.mp4"], layer=2,
                             start_col=5)
    assert m == {"a.mp4": (2, 5), "b.mp4": (2, 6), "c.mp4": (2, 7)}


def test_stage_working_set_failed_file_keeps_its_column():
    # A failed open is skipped from the mapping but its column is NOT
    # reused — re-staging the same list must land files in the same slots.
    st = _ScriptedStager(fail={"bad.mp4"})
    m = st.stage_working_set(["a.mp4", "bad.mp4", "c.mp4"])
    assert m == {"a.mp4": (1, 1), "c.mp4": (1, 3)}


def test_stage_working_set_skips_empty_entries():
    # Empty/None entries are degenerate input, not failed loads: they are
    # dropped WITHOUT consuming a column (unlike a failed load, which
    # keeps its column for re-stage stability).
    st = _ScriptedStager()
    m = st.stage_working_set(["a.mp4", "", None, "b.mp4"])
    assert m == {"a.mp4": (1, 1), "b.mp4": (1, 2)}
    assert len(st.opens) == 2


# ── the REST wire contract, against a stub Arena ────────────────────────

class _StubArena(BaseHTTPRequestHandler):
    requests: list = []          # (method, path, body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode("utf-8")
        type(self).requests.append(("POST", self.path, body))
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        type(self).requests.append(("GET", self.path, ""))
        payload: dict = {}
        if self.path.endswith("/product"):
            payload = {"name": "Stub Arena"}
        elif "/clips/" in self.path:
            payload = {"video": {"fileinfo": {"exists": True}}}
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):    # keep pytest output clean
        pass


@pytest.fixture()
def stub_arena():
    _StubArena.requests = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _StubArena)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/api/v1"
    srv.shutdown()


def test_open_into_slot_posts_file_uri_to_open_endpoint(stub_arena):
    st = rs.ResolumeStager(rest_base=stub_arena, timeout=2.0)
    ok = st.open_into_slot(3, 7, r"C:\Clips\My Clip.mp4")
    assert ok is True
    method, path, body = _StubArena.requests[-1]
    assert (method, path) == ("POST",
                              "/api/v1/composition/layers/3/clips/7/open")
    assert body == "file:///C%3A/Clips/My%20Clip.mp4"


def test_open_into_slot_sends_twin_but_caller_keys_original(stub_arena,
                                                            tmp_path):
    mkv = tmp_path / "Hero.mkv"
    (tmp_path / "Hero.mp4").write_bytes(b"")
    mkv.write_bytes(b"")
    st = rs.ResolumeStager(rest_base=stub_arena, timeout=2.0)
    m = st.stage_working_set([str(mkv)])
    # Arena was sent the .mp4 twin...
    _, _, body = _StubArena.requests[-1]
    assert body.endswith("Hero.mp4")
    # ...but the mapping is keyed by the ORIGINAL path the caller passed.
    assert m == {str(mkv): (1, 1)}


def test_reachable_and_verify_readback(stub_arena):
    st = rs.ResolumeStager(rest_base=stub_arena, timeout=2.0)
    assert st.reachable() is True
    m = st.stage_working_set(["a.mp4"], verify=True)
    assert m == {"a.mp4": (1, 1)}
    # verify=True must have issued a readback GET on the slot.
    gets = [p for (meth, p, _) in _StubArena.requests if meth == "GET"]
    assert any("/composition/layers/1/clips/1" in p for p in gets)


def test_unreachable_arena_degrades_gracefully():
    st = rs.ResolumeStager(rest_base="http://127.0.0.1:9/api/v1",
                           timeout=0.2)
    assert st.reachable() is False
    assert st.stage_working_set(["a.mp4"]) == {}


# ── CLI helpers ─────────────────────────────────────────────────────────

def test_collect_media_scans_folders_sorted_and_filtered(tmp_path):
    (tmp_path / "b.mp4").write_bytes(b"")
    (tmp_path / "a.mov").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    (tmp_path / "sub").mkdir()      # one-level scan: subdirs not descended
    got = rs.collect_media([str(tmp_path)])
    assert got == [str(tmp_path / "a.mov"), str(tmp_path / "b.mp4")]


def test_collect_media_mixes_files_and_folders(tmp_path):
    f = tmp_path / "x.mp4"
    f.write_bytes(b"")
    d = tmp_path / "dir"
    d.mkdir()
    (d / "y.mp4").write_bytes(b"")
    got = rs.collect_media([str(f), str(d), str(tmp_path / "missing.mp4")])
    assert got == [str(f), str(d / "y.mp4")]


# ── the DynamicStager seam: the graduated stager satisfies it ───────────

def test_stager_satisfies_dynamic_stager_seam():
    import resolume_dynamic
    sig_ok = callable(getattr(rs.ResolumeStager, "open_into_slot", None))
    assert sig_ok
    # DynamicStager accepts it as its REST stager without complaint and
    # routes a miss through open_into_slot (scripted here to avoid REST).
    st = _ScriptedStager()

    class _Bridge:
        def __init__(self):
            self.registered, self.connects = {}, []

        def register_clip(self, path, layer, col):
            self.registered[path] = (layer, col)

        def connect_clip(self, layer, col):
            self.connects.append((layer, col))

    br = _Bridge()
    dyn = resolume_dynamic.DynamicStager(st, br, layer=1, ring_size=4,
                                         pool_high_water=10)
    assert dyn._stage_and_fire_sync("off_pool.mp4") == (1, 11)
    assert st.opens == [(1, 11, "off_pool.mp4")]
    assert br.registered["off_pool.mp4"] == (1, 11)
