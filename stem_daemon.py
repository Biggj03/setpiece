"""
Stem-separation daemon for the setpiece VJ rig.

Phase 1 (SHIPPED 2026-05-18 overnight):
    Standalone process. Pulls audio from WASAPI loopback via
    pyaudiowpatch. Runs band-limited spectral flux on the KICK band
    (50-150 Hz) as a placeholder for HS-TasNet drum-stem onsets.
    Emits OSC `/stem/drums/onset <strength>` messages on UDP to the
    main app's OSC listener.

Phase 2 (NEXT SESSION):
    Swap the band-limited flux for HS-TasNet (lucidrains MIT impl).
    Same OSC contract -- only the analysis block changes. See
    STEM_SEPARATION_RESEARCH.md for the integration plan.

Why a separate process:
- Crash isolation: model OOM doesn't kill the player.
- GPU memory: HS-TasNet's CUDA context separate from mpv NVDEC.
- Hot-swappable: replace this daemon without touching the main app.

USAGE
-----
    python stem_daemon.py                  # default: loopback -> 127.0.0.1:7400
    python stem_daemon.py --port 7400
    python stem_daemon.py --device-index 7 # specific WASAPI input
    python stem_daemon.py --list-devices   # show all loopback inputs
    python stem_daemon.py --verbose        # log every onset (loud)

OSC OUTPUT
----------
    /stem/drums/onset <strength_float>   on each detected kick onset

The main app's OSC listener (`osc_in.OSCListener`) receives these and
the registered handler (in main.py) records the onset for picker /
flip-probability boosting.
"""
from __future__ import annotations

import argparse
import logging
import signal
import struct
import sys
import time
from collections import deque
from socket import socket as _socket, AF_INET, SOCK_DGRAM

import numpy as np

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    print("ERROR: pyaudiowpatch not installed. pip install pyaudiowpatch",
          file=sys.stderr)
    sys.exit(2)

logger = logging.getLogger(__name__)


# ── OSC encoding (stdlib only, mirrors osc_out.py) ──────────────────

def _pad4(b: bytes) -> bytes:
    rem = len(b) % 4
    return b + (b"\x00" * (4 - rem) if rem else b"")


def _encode_string(s: str) -> bytes:
    return _pad4(s.encode("utf-8") + b"\x00")


def _encode_float32(f: float) -> bytes:
    return struct.pack(">f", float(f))


def _encode_message(address: str, *args) -> bytes:
    type_tags = ","
    blobs = []
    for a in args:
        if isinstance(a, float):
            type_tags += "f"
            blobs.append(_encode_float32(a))
        elif isinstance(a, int):
            type_tags += "i"
            blobs.append(struct.pack(">i", int(a)))
        elif isinstance(a, str):
            type_tags += "s"
            blobs.append(_encode_string(a))
    return _encode_string(address) + _encode_string(type_tags) + b"".join(blobs)


# ── Audio analysis ──────────────────────────────────────────────────

class StemDaemon:
    """Audio loopback -> per-stem onset -> OSC out."""

    HOP_SAMPLES = 512        # ~11.6 ms at 44.1k -- short for low latency
    KICK_LO_HZ = 50          # drum kick fundamental band
    KICK_HI_HZ = 150
    # Onset threshold: median of recent flux multiplied by this.
    # Tuned conservatively to avoid double-fires; HS-TasNet would
    # give cleaner signal but the band-limited path needs a higher
    # bar.
    ONSET_THRESHOLD_MULT = 2.5
    # Refractory period: no two onsets closer than this (sec). 100ms
    # = max 600 BPM single-kick rate, plenty of headroom for any
    # real EDM track.
    REFRACTORY_S = 0.10
    # Rolling buffer for median flux (~2s at 11.6ms hops)
    BUF_LEN = 180

    # Open-Unmix integration (opt-in via --use-unmix).
    # When enabled: collect 2s of audio, run umxhq drums-only on GPU,
    # then compute flux on the ISOLATED drums waveform instead of raw
    # audio. Latency floor ~2s (window-bound) but accuracy massively
    # improves — won't false-fire on bass synths or low-freq pads.
    UNMIX_WINDOW_SEC = 2.0       # 2-second processing windows
    UNMIX_HOP_SEC = 1.0          # slide by 1s = 50% overlap (no missed onsets at edges)

    def __init__(self, host: str, port: int, device_index: int = -1,
                 verbose: bool = False, use_unmix: bool = False):
        self.host = host
        self.port = port
        self.use_unmix = use_unmix
        self._unmix_model = None
        self._unmix_device = None
        self.device_index = device_index
        self.verbose = verbose
        self._sock = _socket(AF_INET, SOCK_DGRAM)
        self._running = True
        self._onset_count = 0
        self._last_onset_at = 0.0

    def _find_loopback_device(self, pa: "pyaudio.PyAudio") -> int:
        """Find the default WASAPI loopback device (system output mirror)."""
        if self.device_index >= 0:
            return self.device_index
        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            raise RuntimeError("WASAPI host API not available (not Windows?)")
        # Find the default output, then its loopback variant
        default_out_idx = wasapi_info["defaultOutputDevice"]
        default_out = pa.get_device_info_by_index(default_out_idx)
        target_name = default_out["name"]
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if (info.get("hostApi") == wasapi_info["index"]
                    and info.get("isLoopbackDevice", False)
                    and target_name in info["name"]):
                logger.info(
                    f"loopback device: [{i}] {info['name']!r} "
                    f"sr={int(info['defaultSampleRate'])} "
                    f"ch={int(info['maxInputChannels'])}"
                )
                return i
        # Fallback: any loopback device
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("isLoopbackDevice", False):
                logger.info(
                    f"loopback (fallback): [{i}] {info['name']!r}"
                )
                return i
        raise RuntimeError("no WASAPI loopback device found")

    def list_devices(self) -> None:
        pa = pyaudio.PyAudio()
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    loop = " (loopback)" if info.get("isLoopbackDevice") else ""
                    print(f"  [{i:>2}] {info['name']}{loop}  "
                          f"sr={int(info['defaultSampleRate'])} "
                          f"ch={int(info['maxInputChannels'])}")
        finally:
            pa.terminate()

    def _send_onset(self, strength: float) -> None:
        msg = _encode_message("/stem/drums/onset", float(strength))
        try:
            self._sock.sendto(msg, (self.host, self.port))
            self._onset_count += 1
            if self.verbose:
                logger.info(f"[onset] drums  strength={strength:.3f}  "
                            f"total={self._onset_count}")
        except OSError as e:
            logger.debug(f"OSC send failed: {e}")

    def _load_unmix(self) -> bool:
        """Lazy-load Open-Unmix drums-only model on GPU. Returns True
        on success, False if any dependency / device check fails (in
        which case caller falls back to spectral-flux path)."""
        try:
            import torch
            from openunmix import umxhq
        except Exception as e:
            logger.warning(
                f"openunmix not available, falling back to spectral "
                f"flux: {e}"
            )
            return False
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA not available; Open-Unmix on CPU is sub-realtime. "
                "Falling back to spectral flux."
            )
            return False
        self._unmix_device = "cuda"
        try:
            # residual=True needed because EM filter requires >=2 targets;
            # we still only USE the drums target (out[:, 0]).
            self._unmix_model = umxhq(
                targets=["drums"], residual=True,
                device=self._unmix_device,
            )
            self._unmix_model.eval()
            logger.info(
                f"Open-Unmix drums model loaded on "
                f"{self._unmix_device}"
            )
            return True
        except Exception as e:
            logger.warning(
                f"Open-Unmix model load failed: {e}; "
                f"falling back to spectral flux"
            )
            self._unmix_model = None
            return False

    def _process_unmix_window(self, audio_window: np.ndarray, sr: int,
                              flux_threshold_mult: float) -> None:
        """Run drum isolation on a `audio_window` (channels, samples),
        compute flux on the isolated drums, emit OSC onsets for any
        peaks above threshold. Called once per UNMIX_HOP_SEC interval.
        """
        import torch
        # Tensor shape (batch=1, channels=2, samples)
        if audio_window.ndim == 1:
            audio_window = np.stack([audio_window, audio_window])
        elif audio_window.shape[0] > 2:
            audio_window = audio_window[:2]
        elif audio_window.shape[0] == 1:
            audio_window = np.repeat(audio_window, 2, axis=0)
        tensor = torch.from_numpy(audio_window).unsqueeze(0).float().to(
            self._unmix_device)
        try:
            with torch.inference_mode():
                out = self._unmix_model(tensor)
        except Exception as e:
            logger.warning(f"unmix inference failed: {e}")
            return
        # out shape: (batch, target=2, channels, samples)
        # target 0 = drums, target 1 = residual.
        drums = out[0, 0].mean(0).cpu().numpy()  # mono drums (samples,)
        # Run per-hop flux on the drums waveform — same window/threshold
        # algorithm as the raw-audio path, just operating on isolated
        # drums. Onset positions within the window get timestamps near
        # window end (we don't bother offset-correcting since picker
        # biasing tolerates 0-2s lag).
        hop = self.HOP_SAMPLES
        win = np.hanning(hop).astype(np.float32)
        kick_lo_bin = max(0, int(self.KICK_LO_HZ * hop / sr))
        kick_hi_bin = max(kick_lo_bin + 1,
                          int(self.KICK_HI_HZ * hop / sr))
        n_hops = len(drums) // hop
        prev_mag = 0.0
        recent_flux: deque[float] = deque(maxlen=self.BUF_LEN)
        # Pre-fill recent flux from current window to seed threshold
        # (otherwise first window has no median).
        peaks = []  # list of (hop_idx, flux_value)
        for h in range(n_hops):
            seg = drums[h * hop:(h + 1) * hop]
            if len(seg) < hop:
                break
            fft = np.abs(np.fft.rfft(seg * win))
            kick_mag = float(np.sum(fft[kick_lo_bin:kick_hi_bin + 1]))
            flux = max(0.0, kick_mag - prev_mag)
            prev_mag = kick_mag
            recent_flux.append(flux)
            if len(recent_flux) < 8:
                continue
            median = float(np.median(recent_flux)) or 1e-9
            if median < 1e-6:
                continue
            threshold = median * flux_threshold_mult
            if flux > threshold:
                strength = min(
                    1.0, (flux - threshold) / max(threshold, 1.0))
                peaks.append((h, strength))
        # Refractory + emit. Onset hop index → wall-clock by current time.
        now = time.time()
        last_at = self._last_onset_at
        for h_idx, strength in peaks:
            # Approximate hop timestamp within the window. Window ends
            # at `now`, so this hop fired at now - (n_hops - h_idx) * hop_s.
            hop_s = hop / sr
            t_est = now - (n_hops - h_idx) * hop_s
            if t_est - last_at < self.REFRACTORY_S:
                continue
            self._send_onset(strength)
            last_at = t_est
        self._last_onset_at = last_at

    def stop(self) -> None:
        self._running = False

    def run(self) -> int:
        pa = pyaudio.PyAudio()
        try:
            dev_idx = self._find_loopback_device(pa)
            info = pa.get_device_info_by_index(dev_idx)
            sr = int(info["defaultSampleRate"])
            ch = int(info["maxInputChannels"])
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=ch,
                rate=sr,
                input=True,
                input_device_index=dev_idx,
                frames_per_buffer=self.HOP_SAMPLES,
            )
            logger.info(
                f"loopback stream open: sr={sr} ch={ch} hop={self.HOP_SAMPLES}"
            )
            # Pre-compute FFT bin boundaries for the kick band.
            kick_lo_bin = int(self.KICK_LO_HZ * self.HOP_SAMPLES / sr)
            kick_hi_bin = int(self.KICK_HI_HZ * self.HOP_SAMPLES / sr)
            kick_lo_bin = max(0, kick_lo_bin)
            kick_hi_bin = max(kick_lo_bin + 1, kick_hi_bin)
            logger.info(
                f"kick band: bins {kick_lo_bin}-{kick_hi_bin} "
                f"({self.KICK_LO_HZ}-{self.KICK_HI_HZ} Hz)"
            )
            logger.info(
                f"emitting OSC -> {self.host}:{self.port}  "
                f"(/stem/drums/onset)"
            )

            # Optional Open-Unmix path: lazy-load + use 2s-window
            # processing. Falls back to spectral flux if model load fails.
            if self.use_unmix:
                if self._load_unmix():
                    logger.info(
                        f"OPEN-UNMIX MODE: 2s windows, {self.UNMIX_HOP_SEC}s "
                        f"slide. Latency ~{self.UNMIX_WINDOW_SEC:.1f}s but "
                        "drums isolated from bass synths/pads."
                    )
                else:
                    self.use_unmix = False  # fallback
                    logger.info("falling back to spectral flux mode")

            # Spectral-flux state (legacy default path)
            prev_kick_mag = 0.0
            flux_buf: deque[float] = deque(maxlen=self.BUF_LEN)
            window = np.hanning(self.HOP_SAMPLES).astype(np.float32)
            last_log = time.time()
            # Open-Unmix state — sliding 2s audio buffer
            unmix_window_samples = int(sr * self.UNMIX_WINDOW_SEC)
            unmix_hop_samples = int(sr * self.UNMIX_HOP_SEC)
            unmix_buffer: list[np.ndarray] = []
            unmix_buffer_filled = 0
            unmix_last_process_at = 0
            while self._running:
                try:
                    data = stream.read(
                        self.HOP_SAMPLES, exception_on_overflow=False
                    )
                except OSError as e:
                    logger.warning(f"stream read failed: {e}")
                    time.sleep(0.1)
                    continue
                audio_raw = np.frombuffer(data, dtype=np.float32)

                # ── Open-Unmix path (window-buffered) ──────────────
                if self.use_unmix and self._unmix_model is not None:
                    # Keep raw stereo (or duplicated mono) for the model.
                    if ch > 1 and len(audio_raw) >= ch:
                        stereo = audio_raw.reshape(-1, ch)[:, :2].T  # (2, N)
                    else:
                        stereo = np.stack([audio_raw, audio_raw])
                    unmix_buffer.append(stereo)
                    unmix_buffer_filled += stereo.shape[1]
                    if unmix_buffer_filled >= unmix_window_samples:
                        # Concatenate the buffered audio along the
                        # time axis and run inference on the latest
                        # UNMIX_WINDOW_SEC of audio.
                        full = np.concatenate(unmix_buffer, axis=1)
                        if full.shape[1] > unmix_window_samples:
                            window_audio = full[:, -unmix_window_samples:]
                        else:
                            window_audio = full
                        self._process_unmix_window(
                            window_audio, sr,
                            self.ONSET_THRESHOLD_MULT,
                        )
                        # Slide: keep only the last
                        # (UNMIX_WINDOW_SEC - UNMIX_HOP_SEC) seconds.
                        keep = unmix_window_samples - unmix_hop_samples
                        if full.shape[1] > keep:
                            kept = full[:, -keep:]
                            unmix_buffer = [kept]
                            unmix_buffer_filled = kept.shape[1]
                        else:
                            unmix_buffer = [full]
                            unmix_buffer_filled = full.shape[1]
                    # ALSO continue to spectral flux below as a
                    # fast-and-dirty latency-zero second emitter? No —
                    # let's not double-emit; if unmix is on, it's the
                    # sole source. Operator can toggle off if they need
                    # the low-latency signal.
                    # Heartbeat log every 5s
                    now = time.time()
                    if now - last_log > 5.0:
                        logger.info(
                            f"[heartbeat-unmix] {self._onset_count} "
                            f"onsets so far, buf={unmix_buffer_filled} samples"
                        )
                        last_log = now
                    continue

                # ── Spectral-flux path (legacy default) ────────────
                if ch > 1 and len(audio_raw) >= ch:
                    audio = audio_raw.reshape(-1, ch).mean(axis=1)
                else:
                    audio = audio_raw
                if len(audio) < self.HOP_SAMPLES:
                    continue
                # Hanning-windowed FFT
                fft = np.abs(np.fft.rfft(audio * window))
                # Kick band magnitude
                kick_mag = float(np.sum(fft[kick_lo_bin:kick_hi_bin + 1]))
                # Half-wave rectified flux
                flux = max(0.0, kick_mag - prev_kick_mag)
                prev_kick_mag = kick_mag
                flux_buf.append(flux)
                # Median-based threshold over recent ~2s of frames
                if len(flux_buf) < 20:
                    continue
                median = float(np.median(flux_buf))
                if median < 1e-6:
                    continue
                threshold = median * self.ONSET_THRESHOLD_MULT
                now = time.time()
                if flux > threshold and (now - self._last_onset_at) >= self.REFRACTORY_S:
                    # Strength = normalized excess over threshold,
                    # clipped to [0, 1].
                    strength = min(1.0, (flux - threshold) / max(threshold, 1.0))
                    self._send_onset(strength)
                    self._last_onset_at = now
                # 5s heartbeat log so we know the daemon is alive
                if now - last_log > 5.0:
                    logger.info(
                        f"[heartbeat] {self._onset_count} onsets so far, "
                        f"flux_med={median:.4f} thresh={threshold:.4f}"
                    )
                    last_log = now

            stream.stop_stream()
            stream.close()
        finally:
            pa.terminate()
            try:
                self._sock.close()
            except Exception:
                pass
        logger.info(f"stem_daemon exiting. {self._onset_count} onsets total.")
        return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] stem_daemon: %(message)s",
    )
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7400)
    ap.add_argument("--device-index", type=int, default=-1,
                    help="WASAPI loopback device index (default: auto)")
    ap.add_argument("--list-devices", action="store_true",
                    help="List all WASAPI input devices and exit")
    ap.add_argument("--verbose", action="store_true",
                    help="Log every detected onset (loud)")
    ap.add_argument("--use-unmix", action="store_true",
                    help="Use Open-Unmix drums-only GPU isolation "
                         "(needs torch+cuda+openunmix). Latency ~2s "
                         "but accurate — won't false-fire on bass "
                         "synths. Falls back to spectral flux if "
                         "model load fails.")
    args = ap.parse_args()

    if args.list_devices:
        StemDaemon(args.host, args.port).list_devices()
        return 0

    # Auto-pickup device-index from ~/.setpiece/settings.json if
    # the user hasn't overridden via --device-index. Saved by the main
    # app when the user confirms a working device (or set manually).
    # Avoids the "auto-pick chose silent VB-Audio Cable" footgun on
    # every relaunch.
    device_index = args.device_index
    if device_index < 0:
        try:
            import json
            from pathlib import Path
            cfg_path = Path.home() / ".setpiece" / "settings.json"
            if cfg_path.is_file():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                saved = cfg.get("stem_daemon_device_index", -1)
                if isinstance(saved, int) and saved >= 0:
                    device_index = saved
                    name = cfg.get("stem_daemon_device_name", "(no name)")
                    logger.info(
                        f"using saved device-index {device_index} "
                        f"from settings.json: {name}"
                    )
        except Exception as e:
            logger.debug(f"settings.json read failed: {e}")

    daemon = StemDaemon(args.host, args.port,
                        device_index=device_index,
                        verbose=args.verbose,
                        use_unmix=args.use_unmix)
    # Clean SIGINT handler so Ctrl-C exits gracefully
    signal.signal(signal.SIGINT, lambda *_: daemon.stop())
    return daemon.run()


if __name__ == "__main__":
    sys.exit(main())
