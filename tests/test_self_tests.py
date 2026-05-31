"""Bring the modules' inline `_self_test()` blocks under pytest/CI.

osc_out, resolume_out, and resolume_dynamic each ship a substantial
`_self_test()` (OSC 1.0 byte-exact encoding, the bridge registry +
dynamic stage-on-miss ring, etc.) guarded by `if __name__ == "__main__"`.
Those are real verification — osc_out alone has 21 asserts — but pytest
never ran them, so CI wasn't gating the OSC wire format or the ring logic.
This runs them as tests (they raise AssertionError on failure), with their
chatty stdout swallowed so pytest output stays clean.

These are all Arena-free: osc_out is pure byte math, resolume_out sends
UDP into the void (no listener needed, must not raise), resolume_dynamic
uses in-module fakes.
"""
import contextlib
import io

import osc_out
import resolume_out
import resolume_dynamic


def _run_quietly(fn):
    """Call a self-test, swallowing its prints; let assertions propagate."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()


def test_osc_out_self_test():
    _run_quietly(osc_out._self_test)


def test_resolume_out_self_test():
    _run_quietly(resolume_out._self_test)


def test_resolume_dynamic_self_test():
    _run_quietly(resolume_dynamic._self_test)


# ── granular OSC wire-format invariants ────────────────────────────────
# The whole Resolume OSC bridge rides on _encode_message producing valid
# OSC 1.0. These assert the structural contract directly, so a wire-format
# regression names itself instead of failing deep inside a self_test.

def test_osc_message_is_4byte_aligned():
    # OSC 1.0 requires every message be a multiple of 4 bytes.
    for args in [(), (1,), (0.5,), ("hi",), (1, 0.5)]:
        msg = osc_out._encode_message("/composition/master", *args)
        assert isinstance(msg, bytes)
        assert len(msg) % 4 == 0, ("not 4-aligned for args %r" % (args,))


def test_osc_message_starts_with_address():
    msg = osc_out._encode_message("/composition/tempocontroller/resync", 1)
    assert msg.startswith(b"/composition/tempocontroller/resync\x00")


def test_osc_typetag_reflects_arg_types():
    # A float arg -> ',f' type tag; an int arg -> ',i'. (OSC type tag
    # string starts with a comma.)
    assert b",f" in osc_out._encode_message("/x", 0.5)
    assert b",i" in osc_out._encode_message("/x", 1)


def test_osc_address_is_null_terminated_and_padded():
    # Address "/x" (2 bytes) + null = 3, padded to 4. The 4th byte is a
    # pad null, and the type-tag section follows on a 4-byte boundary.
    msg = osc_out._encode_message("/x", 1)
    assert msg[:4] == b"/x\x00\x00"
