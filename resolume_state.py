"""
Read-side view of Resolume Arena for the iPad control panel.

resolume_out.py COMMANDS Arena (OSC, fire-and-forget). resolume_stage.py
LOADS files (REST POST). This is the third piece: it READS Arena's live
state over REST so the panel can show real layer names, opacities, bypass
flags, the composition master, and a simple "is Arena reachable" light.

Why a separate reader: OSC is one-way (we never hear back), so the only
truthful source for "what is Arena actually doing right now" is the REST
API. The panel polls a compact snapshot from here a few times a second.

Stdlib only (urllib). Never raises into the caller; on any failure the
snapshot reports reachable=False and empty layers, so the panel degrades
to a clear "Arena offline" state instead of erroring.
"""

import json
import logging
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_REST = "http://localhost:8080/api/v1"


def _param(d: dict, key: str):
    """Resolume wraps most params as {'value': X, ...}. Return the value,
    tolerating both wrapped and bare forms (shape varies by build/param)."""
    v = d.get(key)
    if isinstance(v, dict):
        return v.get("value")
    return v


class ResolumeState:
    """REST reader that produces a compact, panel-friendly snapshot."""

    def __init__(self, rest_base: str = DEFAULT_REST, timeout: float = 2.0):
        # Short default timeout: this is polled live, a hung GET must not
        # stall the panel. The stager (file loads) uses a longer one.
        self._rest = rest_base.rstrip("/")
        self._timeout = float(timeout)

    # Composition clip-beatsnap choices, in Arena's option order. The list
    # index IS the value Arena's parameter takes ("1 Bar" = cuts land on the
    # bar). Used by set_clip_beatsnap.
    BEATSNAP_OPTIONS = ("None", "8 Bars", "4 Bars", "2 Bars",
                        "1 Bar", "1/2 Bar", "1/4 Bar")

    def _get(self, path: str) -> Optional[dict]:
        try:
            with urllib.request.urlopen(f"{self._rest}/{path}",
                                        timeout=self._timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            logger.debug("Arena GET %s failed: %s", path, e)
            return None

    def _put_param_by_id(self, param_id: int, value) -> bool:
        """Set an Arena parameter by its id: PUT /parameter/by-id/{id}
        body {"value": <v>}. Returns True on 2xx.

        This is how Resolume's REST API writes params — by id, NOT by the
        composition path (those GET fine but 404 on PUT), and NOT over OSC
        (no OSC encoding of a ParamChoice like clipbeatsnap moved Arena).
        The id comes from the GET snapshot. Verified live on Arena 7.25.2:
        PUT value=0 -> 'None', value=4 -> '1 Bar', value=6 -> '1/4 Bar'."""
        try:
            req = urllib.request.Request(
                f"{self._rest}/parameter/by-id/{int(param_id)}",
                data=json.dumps({"value": value}).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="PUT")
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                return 200 <= r.status < 300
        except Exception as e:
            logger.debug("Arena PUT param %s failed: %s", param_id, e)
            return False

    def find_effect_param(self, effect_name, param_name, layer=None):
        """Resolve the live id of an effect parameter by NAME.

        Effect param ids are composition-specific (they change per .avc), so
        they must never be hardcoded — resolve them at runtime from the live
        snapshot, then drive by id. Looks at the composition's video-effects
        rack by default, or layer N's rack when `layer` is given. Matches
        effect + param by display name (case-insensitive). Returns the int
        id, or None if not found / Arena unreachable.

        Foundation for arc->effects: name the effect+param once, drive it by
        id every frame."""
        if layer is None:
            host = self._get("composition")
        else:
            host = self._get(f"composition/layers/{int(layer)}")
        if not host:
            return None
        video = host.get("video") if isinstance(host.get("video"), dict) else {}
        for eff in (video.get("effects") or []):
            en = _param(eff, "name") or _param(eff, "display_name")
            if not en or en.lower() != effect_name.lower():
                continue
            params = eff.get("params") if isinstance(eff.get("params"), dict) else {}
            for pname, pinfo in params.items():
                if pname.lower() == param_name.lower() and isinstance(pinfo, dict):
                    return pinfo.get("id")
        return None

    def list_effects(self, layer=None) -> list:
        """Discover the live effect rack so the panel can render a fader per
        drivable param. Returns:
          [{"name": str, "display_name": str, "bypassed": bool,
            "params": [{"name": str, "id": int, "value": float,
                        "min": float, "max": float}, ...]}, ...]

        Only ParamRange params are returned (the 0..1 ones a fader drives) —
        ParamChoice/ParamState aren't sliders. Composition rack by default,
        layer N's rack when given. Empty list on a dead Arena (never raises).
        Effects/params are composition-specific, so the panel must discover
        them live rather than hardcode — this is that discovery."""
        if layer is None:
            host = self._get("composition")
        else:
            host = self._get(f"composition/layers/{int(layer)}")
        if not host:
            return []
        video = host.get("video") if isinstance(host.get("video"), dict) else {}
        out = []
        for eff in (video.get("effects") or []):
            params = eff.get("params") if isinstance(eff.get("params"), dict) else {}
            drivable = []
            for pname, pinfo in params.items():
                if not isinstance(pinfo, dict):
                    continue
                if pinfo.get("valuetype") != "ParamRange":
                    continue
                pid = pinfo.get("id")
                if pid is None:
                    continue
                drivable.append({
                    "name": pname,
                    "id": pid,
                    "value": pinfo.get("value"),
                    "min": pinfo.get("min", 0.0),
                    "max": pinfo.get("max", 1.0),
                })
            if not drivable:
                continue
            out.append({
                "name": _param(eff, "name"),
                "display_name": _param(eff, "display_name") or _param(eff, "name"),
                "bypassed": bool(_param(eff, "bypassed")),
                "params": drivable,
            })
        return out

    def set_param_by_id(self, param_id, value) -> bool:
        """Public: set any Arena parameter by its (live) id. Thin wrapper on
        the by-id PUT so the panel's FX faders can drive a param directly
        once list_effects has given them its id — no per-drag name re-resolve.
        Returns True on success, False otherwise. Never raises."""
        if param_id is None:
            return False
        try:
            pid = int(param_id)
        except (TypeError, ValueError):
            return False
        return self._put_param_by_id(pid, value)

    def set_effect_param(self, effect_name, param_name, value, layer=None):
        """Set an effect parameter by (effect name, param name) -> resolve
        live id -> PUT by id. `value` is whatever the param takes (0..1 for a
        ParamRange like Hue Rotate / blur amount). Returns True on success,
        False if the param can't be found or Arena is unreachable. Never
        raises. Verified live on Arena 7.25.2 (HueRotate 'Hue Rotate' 0<->0.5)."""
        pid = self.find_effect_param(effect_name, param_name, layer=layer)
        if pid is None:
            return False
        return self._put_param_by_id(pid, value)

    def set_clip_beatsnap(self, index: int) -> bool:
        """Set how clip triggers quantise to the tempo grid. `index` is into
        BEATSNAP_OPTIONS (0=None ... 4='1 Bar' ... 6='1/4 Bar'). With snap
        on, a fired clip waits for the next bar/beat boundary instead of
        cutting instantly — what makes cuts land 'on the 1'.

        Reads the live clipbeatsnap parameter id, then writes by id over
        REST. Returns True on success, False on bad index / unreachable
        Arena (never raises)."""
        i = int(index)
        if i < 0 or i >= len(self.BEATSNAP_OPTIONS):
            return False
        comp = self._get("composition")
        bs = (comp or {}).get("clipbeatsnap")
        pid = bs.get("id") if isinstance(bs, dict) else None
        if pid is None:
            return False
        return self._put_param_by_id(pid, i)

    def _get_bytes(self, path: str, timeout: float = None):
        """Raw GET -> (content_type, bytes) or (None, None) on failure.
        Used to proxy Arena's PNG clip thumbnails to the panel."""
        try:
            with urllib.request.urlopen(
                    f"{self._rest}/{path}",
                    timeout=(timeout or self._timeout)) as r:
                return r.headers.get("Content-Type", "image/png"), r.read()
        except Exception as e:
            logger.debug("Arena GET(bytes) %s failed: %s", path, e)
            return None, None

    def thumbnail_by_id(self, clip_id):
        """Fetch a clip's thumbnail PNG by its Arena clip id. Returns
        (content_type, bytes) or (None, None). The id comes from the
        snapshot/clip_names 'thumb_id' field."""
        # Arena path: /composition/clips/by-id/<id>/thumbnail
        return self._get_bytes(f"composition/clips/by-id/{int(clip_id)}/thumbnail")

    def reachable(self) -> bool:
        """True if Arena's REST API answers (webserver on + Arena up)."""
        p = self._get("product")
        return bool(p and p.get("name"))

    def preflight(self) -> dict:
        """Pre-set readiness check. Returns a structured report the
        launcher prints as a green/red checklist before a gig, so the
        operator knows the rig is wired BEFORE the first track instead of
        discovering it dark mid-set.

        Shape:
          {
            "ok": bool,                 # all critical checks passed
            "checks": [
              {"name": str, "ok": bool, "detail": str, "critical": bool},
              ...
            ],
          }
        """
        checks = []

        prod = self._get("product")
        rest_ok = bool(prod and prod.get("name"))
        pname = (prod or {}).get("name", "") if rest_ok else ""
        checks.append({
            "name": "Arena REST (webserver :8080)",
            "ok": rest_ok,
            "detail": pname if rest_ok else "no answer — enable Arena "
                      "Preferences > Webserver",
            "critical": True,
        })

        # Version gate: the OSC address space we drive is Arena 7+. Arena 5
        # silently discards our packets, so flag an old version loudly.
        ver_ok = False
        ver_detail = "unknown (Arena unreachable)"
        if rest_ok:
            major = prod.get("major")
            try:
                major = int(major)
            except (TypeError, ValueError):
                major = None
            if major is not None:
                ver_ok = major >= 7
                ver_detail = (f"v{major} OK" if ver_ok
                              else f"v{major} — needs Arena 7+ (OSC scheme "
                                   "differs on older versions)")
            else:
                # Some builds don't expose major; don't hard-fail on it.
                ver_ok = True
                ver_detail = "version field absent; assuming 7+"
        checks.append({
            "name": "Arena version >= 7",
            "ok": ver_ok,
            "detail": ver_detail,
            "critical": False,
        })

        # Composition + staged content: how many clips actually loaded.
        comp = self._get("composition") if rest_ok else None
        layers = (comp or {}).get("layers") or []
        loaded = 0
        for layer in layers:
            for c in (layer.get("clips") or []):
                cv = c.get("video") if isinstance(c.get("video"), dict) else {}
                if cv.get("fileinfo"):
                    loaded += 1
        content_ok = loaded > 0
        checks.append({
            "name": "Composition has clips",
            "ok": content_ok,
            "detail": (f"{loaded} clip(s) loaded across {len(layers)} layer(s)"
                       if content_ok else
                       "0 clips loaded — stage content into Arena's grid"),
            "critical": False,
        })

        ok = all(c["ok"] for c in checks if c["critical"])
        return {"ok": ok, "checks": checks}

    def snapshot(self, max_clip_probe: int = 0) -> dict:
        """Compact live state for the panel. Always returns a dict; on a
        dead Arena returns {'reachable': False, 'layers': [], ...}.

        Shape:
          {
            "reachable": bool,
            "product": "Arena 7" | "",
            "master": float|None,          # composition master 0..1
            "crossfader": float|None,      # crossfader phase 0..1
            "tempo": float|None,           # BPM
            "layers": [
              {"index": 1, "name": "...", "opacity": 0.5,
               "bypassed": bool, "solo": bool,
               "loaded": 37,               # clips with a file
               "columns": 38,
               "active_clip": int|None},   # connected column, if any
              ...
            ],
          }
        """
        out = {
            "reachable": False, "product": "", "master": None,
            "crossfader": None, "tempo": None, "beatsnap": None, "layers": [],
        }
        comp = self._get("composition")
        if not comp:
            return out
        out["reachable"] = True
        prod = self._get("product") or {}
        out["product"] = prod.get("name") or ""
        out["master"] = _param(comp, "master")
        xf = comp.get("crossfader") or {}
        out["crossfader"] = _param(xf, "phase") if isinstance(xf, dict) else None
        tc = comp.get("tempocontroller") or {}
        out["tempo"] = _param(tc, "tempo") if isinstance(tc, dict) else None
        # Clip beatsnap is a choice param: report its index so the panel's
        # selector can reflect Arena's real setting.
        bs = comp.get("clipbeatsnap")
        out["beatsnap"] = bs.get("index") if isinstance(bs, dict) else None

        for i, layer in enumerate(comp.get("layers") or [], start=1):
            video = layer.get("video") if isinstance(layer.get("video"), dict) else {}
            clips = layer.get("clips") or []
            loaded = 0
            active = None
            for ci, c in enumerate(clips, start=1):
                cv = c.get("video") if isinstance(c.get("video"), dict) else {}
                if cv.get("fileinfo"):
                    loaded += 1
                conn = c.get("connected")
                cval = conn.get("value") if isinstance(conn, dict) else conn
                if cval in ("Connected", "Connected & previewing"):
                    active = ci
            out["layers"].append({
                "index": i,
                "name": _param(layer, "name"),
                "opacity": _param(video, "opacity"),
                "bypassed": bool(_param(layer, "bypassed")),
                "solo": bool(_param(layer, "solo")),
                "loaded": loaded,
                "columns": len(clips),
                "active_clip": active,
            })
        return out

    def clip_names(self, layer: int) -> list:
        """Return [{'column': N, 'name': str, 'loaded': bool}] for one
        layer — feeds the panel's CLIPS grid. Empty list on failure."""
        l = self._get(f"composition/layers/{int(layer)}")
        if not l:
            return []
        out = []
        for ci, c in enumerate(l.get("clips") or [], start=1):
            cv = c.get("video") if isinstance(c.get("video"), dict) else {}
            th = c.get("thumbnail") if isinstance(c.get("thumbnail"), dict) else {}
            loaded = bool(cv.get("fileinfo"))
            # Only offer a thumbnail id for loaded clips with a real (non
            # default) thumbnail — saves the panel fetching empty-slot PNGs.
            thumb_id = None
            if loaded and th.get("id") and not th.get("is_default"):
                thumb_id = th.get("id")
            out.append({
                "column": ci,
                "name": _param(c, "name"),
                "loaded": loaded,
                "thumb_id": thumb_id,
            })
        return out


# ---------------------------------------------------------------------------
# Self-test -- offline (no Arena needed) pins the param-unwrap + graceful
# degradation. The live snapshot is exercised by the panel verify script.
# ---------------------------------------------------------------------------

def _self_test() -> None:
    # _param unwraps both wrapped and bare forms.
    assert _param({"master": {"value": 0.7}}, "master") == 0.7
    assert _param({"master": 0.7}, "master") == 0.7
    assert _param({}, "master") is None

    # Unreachable Arena => reachable False, snapshot degrades, no raise.
    s = ResolumeState(rest_base="http://127.0.0.1:9/api/v1", timeout=0.3)
    assert s.reachable() is False
    snap = s.snapshot()
    assert snap["reachable"] is False
    assert snap["layers"] == []
    assert s.clip_names(1) == []

    print("resolume_state._self_test: OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _self_test()
