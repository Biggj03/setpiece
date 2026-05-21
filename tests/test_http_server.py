"""Integration tests for the HTTP control server.

Starts a real `HTTPServerThread` on an ephemeral localhost port and
exercises it over a socket — covering action dispatch, the POST body
size cap (413), and malformed-JSON rejection (400).
"""
import http.client
import json

import pytest

from app_state import AppState
from http_server import HTTPServerThread, _MAX_POST_BYTES


@pytest.fixture
def server():
    """A running localhost server with one registered `ping` action."""
    srv = HTTPServerThread(AppState(), port=0, lan=False)
    srv.register("ping", lambda d: {"ok": True, "pong": d.get("x")})
    srv.start()
    port = srv.server.server_address[1]
    yield port
    srv.stop()


def _post(port, path, body, content_length=None):
    """POST to the server. `content_length`, if given, overrides the
    header independently of the bytes actually sent."""
    raw = body if isinstance(body, bytes) else body.encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.putrequest("POST", path)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length",
                       str(content_length if content_length is not None
                           else len(raw)))
        conn.endheaders()
        conn.send(raw)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def test_valid_post_dispatches_to_callback(server):
    status, data = _post(server, "/api/ping", '{"x": 7}')
    assert status == 200
    # The server wraps a callback's return value as {"ok", "result"}.
    payload = json.loads(data)
    assert payload["ok"] is True
    assert payload["result"]["pong"] == 7


def test_malformed_json_returns_400(server):
    status, _ = _post(server, "/api/ping", "{not valid json")
    assert status == 400


def test_non_object_json_body_returns_400(server):
    status, _ = _post(server, "/api/ping", "[1, 2, 3]")
    assert status == 400


def test_oversized_body_returns_413(server):
    # Claim a body far over the cap. The server must reject from the
    # Content-Length header alone, before reading anything into memory —
    # so we send only two bytes but advertise a huge length.
    status, _ = _post(server, "/api/ping", "{}",
                      content_length=_MAX_POST_BYTES + 5_000_000)
    assert status == 413
