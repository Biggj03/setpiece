"""
Video player wrapper using libmpv via python-mpv.

Drop-in replacement for player.py (QMediaPlayer backend).
In main.py:    from player import VideoPlayer  ->  from player_mpv import VideoPlayer

Why this backend:
  - Frame-accurate seek + true frame stepping
  - Hardware decode on NVIDIA / Intel / AMD via d3d11va / nvdec / dxva2
  - Doesn't rely on Windows Media Foundation, which has been flaky in QMediaPlayer

Required runtime:
  - python-mpv         (pip install python-mpv)
  - libmpv-2.dll       MUST be in this directory, %PATH%, or System32.
                       This module prepends its own directory to %PATH%
                       BEFORE `import mpv`, so dropping libmpv-2.dll
                       next to player_mpv.py is the simplest install.

Get libmpv-2.dll from:
  https://github.com/shinchiro/mpv-winbuild-cmake/releases  (mpv-dev-x86_64-*.7z)
  or
  https://sourceforge.net/projects/mpv-player-windows/files/libmpv/
"""

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# ── DLL bootstrap (must run BEFORE `import mpv`) ───────────────────────────
# python-mpv resolves libmpv-2.dll via Windows' standard DLL search, which
# checks %PATH%. Prepending this file's dir means the user can just drop
# libmpv-2.dll next to player_mpv.py and it Just Works.
_HERE = Path(__file__).resolve().parent
if sys.platform == "win32":
    os.environ["PATH"] = str(_HERE) + os.pathsep + os.environ.get("PATH", "")
    # Python 3.8+: also register as a DLL search dir so dependent DLLs resolve.
    try:
        os.add_dll_directory(str(_HERE))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

try:
    import mpv  # noqa: E402
    _MPV_AVAILABLE = True
    _MPV_ERROR: Optional[str] = None
except (ImportError, OSError) as e:
    _MPV_AVAILABLE = False
    _MPV_ERROR = str(e)
    mpv = None  # type: ignore[assignment]

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QWidget  # noqa: E402

logger = logging.getLogger(__name__)


def _safe_log(level: str, component: str, message: str) -> None:
    """mpv log_handler — strips non-cp1252 chars so Windows console doesn't choke."""
    msg = message.strip()
    try:
        logger.log(_MPV_LEVELS.get(level, logging.INFO), "mpv[%s] %s", component, msg)
    except UnicodeEncodeError:
        logger.log(
            _MPV_LEVELS.get(level, logging.INFO),
            "mpv[%s] %s",
            component,
            msg.encode("ascii", "replace").decode(),
        )


_MPV_LEVELS = {
    "fatal": logging.CRITICAL,
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "info": logging.INFO,
    "v": logging.DEBUG,
    "debug": logging.DEBUG,
    "trace": logging.DEBUG,
}


class VideoPlayer:
    """Embedded libmpv player.

    Pass any QWidget — typically the same QVideoWidget main.py creates for
    the QMediaPlayer backend. mpv just needs a native HWND, which we force
    via WA_NativeWindow.

    The widget MUST be visible (or be inside a visible parent) before this
    constructor runs, so winId() returns a real HWND. main.py shows the
    window before calling _init_player(), so this works as-is.
    """

    def __init__(self, widget: QWidget):
        if not _MPV_AVAILABLE:
            raise RuntimeError(
                f"python-mpv / libmpv unavailable: {_MPV_ERROR}\n"
                "Install: pip install python-mpv  AND  drop libmpv-2.dll "
                f"into {_HERE}"
            )

        self.widget = widget
        self.current_file: Optional[str] = None
        # Crossfade ("preview deck B") state. Same libmpv instance is fed
        # a second video+audio track via video-add/audio-add, then both are
        # mixed live by an FFmpeg lavfi-complex graph. Opacity 0 = pure
        # live, opacity 1 = pure preview. See set_blend() / load_preview().
        self.preview_file: Optional[str] = None
        self._blend_active: bool = False
        self._blend_opacity: float = 0.0  # 0..1 — preview's contribution
        # NOTE: a real threading.RLock used to guard the multi-step player
        # mutations here (audit fix H5). It caused a HARD HANG —
        # crossfade_blend ran on the S2 action-worker thread, took the lock,
        # and rebuilt mpv's lavfi-complex graph (which needs the Qt thread)
        # while the Qt thread was itself blocked waiting for the same lock.
        # Even short of a true deadlock it stalled the S2 worker, so jog +
        # every other S2 input backed up behind it.
        #
        # crossfade_blend is now MARSHALLED onto the Qt thread (see
        # main._crossfade_blend_apply), so every player mutation runs
        # single-threaded on the Qt loop and no lock is needed. nullcontext
        # keeps the existing `with self._mutation_lock:` call sites valid
        # without re-indenting six methods — it's a no-op.
        self._mutation_lock = contextlib.nullcontext()

        # Force a real HWND. Without this, winId() may return a parent's
        # handle and mpv renders into the wrong region (or a black void).
        widget.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)

        wid = int(widget.winId())
        logger.info("Embedding libmpv into winId=%d", wid)

        try:
            self.player = mpv.MPV(
                wid=str(wid),
                vo="gpu",               # gpu-next is flaky on some NVIDIA drivers; gpu is rock-solid
                hwdec="auto-safe",      # NVDEC / DXVA2 / D3D11VA on NVIDIA
                ytdl=False,             # No youtube-dl integration
                hr_seek="yes",          # Frame-accurate seeking
                keep_open="yes",        # Don't unload on EOF (so seek/flip works)
                osc=False,              # No mpv on-screen-controller
                input_default_bindings=False,
                input_vo_keyboard=False,  # Qt owns keyboard input
                log_handler=_safe_log,
                loglevel="info",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create MPV instance: {e}") from e

        self.player.volume = 80
        try:
            ver = self.player.mpv_version
            hwdec_active = self.player.hwdec_current or "(not yet active)"
            logger.info("Player initialized: %s, hwdec=%s, vo=gpu-next", ver, hwdec_active)
        except Exception:
            pass

    def set_audio_device_by_substring(self, substring: str) -> bool:
        """Pin mpv's audio output to a device whose name or description
        contains `substring` (case-insensitive). Returns True if matched.

        Use case: route video audio to the S2's Monitor output (headphone
        cue circuit) so it doesn't bleed into the system loopback the
        beat detector listens to."""
        if not substring:
            return False
        sub = substring.strip().lower()
        try:
            devices = list(self.player.audio_device_list or [])
        except Exception as e:
            logger.warning("audio_device_list unavailable: %s", e)
            return False
        for d in devices:
            name = str(d.get("name") or "")
            desc = str(d.get("description") or "")
            if sub in name.lower() or sub in desc.lower():
                try:
                    self.player.audio_device = name
                    logger.info("Audio device pinned: %r (desc=%r)", name, desc)
                    return True
                except Exception as e:
                    logger.warning("Failed to set audio_device=%r: %s", name, e)
                    return False
        sample = ", ".join(f"{(d.get('description') or d.get('name'))!r}" for d in devices[:8])
        logger.warning(
            "No audio device matching %r (sampled: %s%s)",
            substring, sample, "..." if len(devices) > 8 else "",
        )
        return False

    # ── playback ──────────────────────────────────────────────────────────

    def load(self, filepath: str, autoplay: bool = True) -> None:
        """Load a video file. Auto-plays by default (matches player.py).

        Loading a new main file tears down any active blend — the secondary
        tracks (vid2/aid2) get cleared by play(), so the lavfi graph would
        reference nonexistent inputs. Caller can re-attach a preview after.
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(filepath)
        # Hold the mutation lock for the whole load sequence (Audit fix
        # H5) so a crossfade_blend on the worker thread can't interleave
        # its video-add + lavfi rebuild between our clear and our play().
        with self._mutation_lock:
            # ALWAYS clear the lavfi graph + tear down any external tracks
            # before play(). The Python-side flags (_blend_active /
            # preview_file) can drift out of sync with mpv's actual graph
            # state — e.g. after a crossfade prep that didn't fully tear
            # down — and the stale "vid2" reference then breaks every
            # subsequent load with "Pad vid2 is not connected to anything"
            # and a black main display. Clearing is cheap; do it unconditionally.
            try:
                self.player.lavfi_complex = ""
            except Exception:
                pass
            if self._blend_active or self.preview_file:
                self._detach_preview()
            # Clear any A-B loop from a previously-fired clip — otherwise the
            # new video inherits the loop bounds and ping-pongs at random spots.
            # Use command() form rather than direct property assignment because
            # ab_loop_a = "no" can crash some python-mpv versions on Windows.
            for prop in ("ab-loop-a", "ab-loop-b"):
                try:
                    self.player.command("set", prop, "no")
                except Exception:
                    pass
            # play() loads the file. Set pause AFTER if autoplay=False.
            self.player.play(filepath)
            self.current_file = filepath
            if autoplay:
                self.player.pause = False
            else:
                self.player.pause = True
        logger.info("Loaded: %s (autoplay=%s)", filepath, autoplay)

    def play(self) -> None:
        self.player.pause = False

    def pause(self) -> None:
        self.player.pause = True

    def seek(self, seconds: float) -> None:
        try:
            self.player.seek(seconds, "absolute", "exact")
        except Exception as e:
            logger.debug("seek(%s) failed: %s", seconds, e)

    def seek_relative(self, delta_seconds: float, exact: bool = False) -> None:
        """Relative seek. Defaults to keyframe (fast — needed for jog scrub
        without audio buffer underrun). Pass exact=True for frame-accurate."""
        precision = "exact" if exact else "keyframes"
        try:
            self.player.seek(delta_seconds, "relative", precision)
        except Exception as e:
            logger.debug("seek_relative(%s) failed: %s", delta_seconds, e)

    def frame_step(self, forward: bool = True) -> None:
        """True single-frame step (mpv's killer feature for VJ scrubbing)."""
        try:
            if forward:
                self.player.frame_step()
            else:
                self.player.frame_back_step()
        except Exception as e:
            logger.debug("frame_step failed: %s", e)

    # ── queries ───────────────────────────────────────────────────────────

    def get_position(self) -> float:
        try:
            return float(self.player.time_pos or 0.0)
        except Exception:
            return 0.0

    def get_duration(self) -> float:
        try:
            return float(self.player.duration or 0.0)
        except Exception:
            return 0.0

    def is_playing(self) -> bool:
        try:
            return not bool(self.player.pause)
        except Exception:
            return False

    # ── volume ────────────────────────────────────────────────────────────

    def set_volume(self, level: int) -> None:
        try:
            self.player.volume = max(0, min(100, int(level)))
        except Exception as e:
            logger.debug("set_volume failed: %s", e)

    def get_volume(self) -> int:
        try:
            return int(self.player.volume or 0)
        except Exception:
            return 0

    # ── crossfade blend (preview deck B) ──────────────────────────────────
    #
    # Strategy: ONE libmpv instance, the live file plays as the main track,
    # a secondary "preview" file is attached via `video-add` + `audio-add`.
    # An FFmpeg lavfi-complex graph mixes the two outputs in real time:
    #
    #   [vid1][vid2]blend=all_mode=normal:all_opacity=X[vo]
    #   [aid1][aid2]amix=inputs=2:weights=A B[ao]
    #
    # X = preview's video opacity (0..1).
    # A,B = audio weights (live, preview). amix sums then divides by sum
    # of weights, so pass weights="(1-X) X" for an equal-power-ish fade
    # (we use a sqrt curve in main.py before calling set_blend).
    #
    # Why one instance instead of two: avoids two HWNDs fighting for the
    # same QVideoWidget, two d3d11/NVDEC contexts, and split audio output
    # devices. lavfi-complex was added to libmpv specifically for this.
    # The graph string can be rebuilt at ~700Hz on this hardware (~1.4ms
    # per swap) so per-tick crossfader updates are cheap.

    def load_preview(self, filepath: Optional[str]) -> bool:
        """Attach a 2nd video+audio track for crossfade blending.

        Pass None (or "") to detach the preview entirely. Returns True if
        the preview is now attached, False on detach or any failure.
        Idempotent — calling with the currently-loaded preview is a no-op.
        """
        if not filepath:
            self._detach_preview()
            return False
        if filepath == self.preview_file:
            return True  # already loaded
        if not Path(filepath).exists():
            logger.warning("load_preview: file missing: %s", filepath)
            self._detach_preview()
            return False
        # Whole attach sequence under the mutation lock (Audit fix H5) —
        # video-add + graph install must not interleave a load() from
        # the Qt thread.
        with self._mutation_lock:
            # Drop the old graph (and old vid2/aid2) before adding new ones.
            self._detach_preview()
            try:
                # Quirk: `video-add <file>` ALSO loads any audio tracks from
                # that file (and vice versa for audio-add). Calling both adds
                # the file twice → duplicate vid3/aid3 tracks, which breaks
                # the lavfi-complex graph (`[vid2]` becomes ambiguous). One
                # call is enough.
                self.player.command("video-add", filepath, "auto")
                self.preview_file = filepath
                # Install graph at opacity=0 so the preview is silent + invisible
                # until the crossfader is moved.
                self._blend_opacity = 0.0
                self._apply_blend_graph(0.0)
                self._blend_active = True
                logger.info("Preview attached: %s", filepath)
                return True
            except Exception as e:
                # Log the offending path too — fullwidth/Unicode chars in the
                # filename (e.g. ＂) sometimes trip mpv's command parser with
                # MPV_ERROR_COMMAND (-12).
                logger.error("load_preview failed for %r: %s", filepath, e)
                self._detach_preview()
                return False

    def _detach_preview(self) -> None:
        """Pull the lavfi graph + 2nd video/audio tracks. Safe to call any
        time — also walks the live track list to catch external tracks
        we lost track of (e.g. a crossfade attach that crashed mid-setup
        and left orphaned vid2/aid2 in mpv but cleared our flags).

        Reentrant under the mutation lock (Audit fix H5) — load() and
        load_preview() already hold it when they call in here."""
        with self._mutation_lock:
            try:
                self.player.lavfi_complex = ""
            except Exception:
                pass
            # Remove ALL externally-added tracks (those with `external-filename`
            # set). The built-in tracks have id=1 and no external-filename.
            # We iterate by ID rather than calling video-remove naked because
            # the bare command needs an explicit id (mpv error -12 otherwise).
            try:
                for t in list(self.player.track_list or []):
                    if not t.get("external-filename"):
                        continue
                    tid = t.get("id")
                    if tid is None:
                        continue
                    ttype = t.get("type")
                    cmd = "video-remove" if ttype == "video" else "audio-remove"
                    try:
                        self.player.command(cmd, str(tid))
                    except Exception:
                        pass
            except Exception:
                pass
            self._blend_active = False
            self._blend_opacity = 0.0
            self.preview_file = None

    def set_blend(self, opacity: float) -> None:
        """Update the crossfade opacity. 0 = pure live, 1 = pure preview.

        Requires load_preview() to have succeeded first; otherwise it's a
        cheap no-op (so the crossfader callback can fire freely).
        """
        # Take the mutation lock so a graph rebuild can't interleave a
        # load()/_detach_preview() from another thread (Audit fix H5).
        with self._mutation_lock:
            if not self._blend_active:
                return
            op = max(0.0, min(1.0, float(opacity)))
            # Skip rebuilds for sub-1% changes — saves ~30 graph rebuilds per
            # full sweep, which adds up if the fader is jittery.
            if abs(op - self._blend_opacity) < 0.005:
                return
            self._blend_opacity = op
            try:
                self._apply_blend_graph(op)
            except Exception as e:
                logger.debug("set_blend(%.3f) failed: %s", op, e)

    def _apply_blend_graph(self, op: float) -> None:
        """Build + push the lavfi-complex string for opacity `op`.

        The blend filter requires both inputs to have matching size +
        format. User clips are mixed-resolution (1920x1080 + 1280x720
        is the common case), so we pre-scale both streams to a common
        1920x1080 canvas with letterbox/pillar-box (`force_original_
        aspect_ratio=decrease` + pad). Pre-scaling fixes the silent
        graph-config failure that drops video to black mid-blend.
        """
        # CROSSFADE AUDIO FIX (Option A — see CROSSFADE_AUDIO_FIX.md):
        # the preview file's audio starts at t=0 while LIVE is mid-track,
        # so blending the two audio streams sounds like two unsynced
        # songs colliding. Fix: keep audio 100% LIVE for the whole blend
        # — the video still crossfades, only the audio holds. The one
        # hard audio cut lands on auto-promote, when the screen is
        # already full-preview, so the visual masks it.
        # amix is left in the graph (vs. dropping it) so a future
        # beat-synced crossfade can just restore op-driven weights here.
        a_live = 1.0
        a_prev = 0.001
        scale = (
            "scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p"
        )
        # NOTE on track ordering: mpv's `video-add` makes the ADDED file
        # become aid1 (newest = first), pushing the original live track
        # to aid2. So the audio weights are swapped relative to video:
        # video uses [vid1]+[vid2] where vid2 is preview (op rising = more
        # vid2), but audio uses [aid1]+[aid2] where aid1 is preview. So
        # the preview's audio weight goes on aid1, live's on aid2.
        graph = (
            f"[vid1]{scale}[v1];"
            f"[vid2]{scale}[v2];"
            f"[v1][v2]blend=all_mode=normal:all_opacity={op:.3f}[vo];"
            f"[aid1][aid2]amix=inputs=2:weights={a_prev:.3f} {a_live:.3f}[ao]"
        )
        # Always pushed from inside load_preview()/set_blend(), both of
        # which hold _mutation_lock — re-acquire (RLock) so a direct call
        # is still safe (Audit fix H5).
        with self._mutation_lock:
            self.player.lavfi_complex = graph

    @property
    def blend_active(self) -> bool:
        return self._blend_active

    @property
    def blend_opacity(self) -> float:
        return self._blend_opacity

    def promote_preview(self) -> Optional[str]:
        """Swap preview → live. The current preview file becomes the new
        main; the old main is discarded. Returns the new live filepath or
        None if there was no preview.

        Implementation: we just `load()` the preview filepath fresh as the
        main track. Cleaner than hot-swapping inside the existing graph
        (which would require muxing logic). User loses position in the
        preview file (restarts at 0) — acceptable for a "go live" promote.
        """
        # Read preview_file + load() atomically under the mutation lock
        # (Audit fix H5) so a concurrent _detach_preview / load_video
        # can't null preview_file between the read and the load.
        with self._mutation_lock:
            prev = self.preview_file
            if not prev:
                return None
            # load() already tears down the blend graph for us
            self.load(prev, autoplay=True)
            return prev

    # ── lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        if getattr(self, "player", None) is None:
            return
        try:
            self.player.terminate()
        except Exception:
            try:
                self.player.quit()
            except Exception:
                pass
        self.player = None  # type: ignore[assignment]
