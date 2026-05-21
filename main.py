"""
setpiece: VJ practice tool with Traktor Kontrol S2 MK1 control.

Main display: fullscreen video playback (QMediaPlayer)
iPad display: control UI via HTTP polling (port 8765)
S2 hardware: clip marking, audio-reactive flips, clip triggering
"""

import sys
import os
import json
import logging
import threading
import queue
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QKeySequence, QShortcut, QPalette, QColor
from PyQt6.QtMultimediaWidgets import QVideoWidget

from player_mpv import VideoPlayer  # was: from player import VideoPlayer (QMediaPlayer didn't render)
from clips_db import ClipDatabase
from audio_reactive import AudioReactive
# Opt-in PLL beat detector. Import is lazy inside _init_audio_reactive so
# a missing BeatNet/madmom install doesn't break startup when the user
# is on the default spectral_flux backend.
from s2_controller import S2Controller
from app_state import AppState
from http_server import HTTPServerThread
from decks import DeckStore, make_deck_entry
from scenes import SceneStore
from scratch import ScratchStore, ScratchSetStore
from banks import BankStore, SLOT_LETTERS as BANK_LETTERS
from smart_playlists import SmartPlaylistStore
from session_log import SessionLog
from magic_mix import MagicMix
from analytics import Analytics
from osc_out import OSCBroadcaster
# AI tagger — Anthropic Vision over clip thumbnails. Optional.
try:
    from ai_tagger import AITagger
    _AI_TAGGER_AVAILABLE = True
except Exception as _e:
    _AI_TAGGER_AVAILABLE = False
    AITagger = None  # type: ignore
from proxy_cache import ProxyCache
from working_set import WorkingSetWatcher, DEFAULT_DIR as WORKING_DIR
from maschine_mk2 import MaschineMK2
from path_tags_v2 import PathTagIndex, canonical_path  # v2: smarter noise filtering + multi-word phrases
from channels import ChannelStore, CHANNEL_COUNT
from preview_streams import PreviewStreamManager
import thumbnails

# Stream Deck is OPTIONAL. The streamdeck library + Pillow are not in the
# core requirements; if either is missing, main.py still boots — we just
# skip the deck. Same idea as libmpv being optional.
try:
    from stream_deck import StreamDeckController
    _STREAM_DECK_IMPORT_ERR: Optional[str] = None
except Exception as _sd_err:  # pragma: no cover - environment-dependent
    StreamDeckController = None  # type: ignore[assignment]
    _STREAM_DECK_IMPORT_ERR = str(_sd_err)

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".setpiece"
CONFIG_DIR.mkdir(exist_ok=True)

# Video file extensions the library browser will list. Hidden / temp files
# (leading "." or "~") are filtered out by _scan_folder.
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

# Default library root. Empty until the user picks a folder (file
# picker on first launch, or POST /api/library/scan); the choice is
# persisted to settings.json.
DEFAULT_LIBRARY = ""


class VJPracticeApp(QMainWindow):
    """Main VJ practice window. Video on primary display, controls via S2 + iPad."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("setpiece")
        self.setGeometry(100, 100, 1280, 720)

        # Settings
        self.config_file = CONFIG_DIR / "settings.json"
        self.config = self._load_config()

        # Shared state (PC writes only, iPad reads via HTTP)
        self.state = AppState()
        self.state.set_volume(self.config.get("volume", 80))

        # Components
        self.clips_db = ClipDatabase(str(CONFIG_DIR / "clips.json"))
        # When background BPM analysis lands, refresh the iPad so the
        # new tempo badge / sort / filter results show up on the next
        # 500ms poll. Wired here (not inside ClipDatabase.__init__) so
        # the database stays decoupled from the rest of the app.
        try:
            self.clips_db.set_on_bpm_updated(
                lambda cid, bpm: self._refresh_clips_for_ipad()
            )
        except Exception as e:
            logger.debug(f"set_on_bpm_updated failed: {e}")
        self.decks_store = DeckStore(CONFIG_DIR / "decks.json")
        self.scene_store = SceneStore(CONFIG_DIR / "scenes.json")
        # Scratch list: a curated session basket of library file paths.
        # Lighter than working_set (no filesystem hot folder, no proxy
        # queue), separate from clips bank (full files vs cue points).
        self.scratch_store = ScratchStore(CONFIG_DIR / "scratch.json")
        self.scratch_set_store = ScratchSetStore(CONFIG_DIR / "scratch_sets.json")
        self.bank_store = BankStore(CONFIG_DIR / "banks.json")
        # Build categorical bank map from BANK_THEMES (canonical letter→
        # name source for the legacy auto-reroll) + _BANK_FOCUS_TAGS
        # (the rare/focus tags this categorical split needs to actually
        # discriminate). Single mnemonic source = no drift.
        try:
            built = {}
            for letter, name, base_tags in self.BANK_THEMES:
                focus = list(
                    self._BANK_FOCUS_TAGS.get(letter, []) or []
                )
                # Combine, dedupe, preserve order.
                seen = set()
                merged = []
                for t in list(base_tags) + focus:
                    if t not in seen:
                        seen.add(t)
                        merged.append(t)
                built[letter] = {
                    "name": name.lower(),
                    "tags": merged,
                }
            self._DEFAULT_BANK_CATEGORIES = built
            logger.info(
                f"[bank-categories] built from BANK_THEMES: "
                + ", ".join(
                    f"{l}={d['name']}({len(d['tags'])})"
                    for l, d in built.items()
                )
            )
        except Exception as e:
            logger.warning(f"bank category build failed: {e}")
        # Seed the iPad-facing bank-layer state so the layer chip
        # shows the restored layer name from page load (rather than
        # only after the user first cycles it).
        try:
            self.state.set_bank_layer(
                self._get_active_layer_name(),
                self._all_layer_names(),
            )
            logger.info(
                f"[bank-layer] active={self._get_active_layer_name()!r}  "
                f"order={self._all_layer_names()}"
            )
        except Exception as e:
            logger.debug(f"bank-layer state seed failed: {e}")
        self.smart_playlists = SmartPlaylistStore(CONFIG_DIR / "smart_playlists.json")
        # Session log — append-only JSONL per day under ~/.setpiece/sessions/.
        # Cross-cutting; hooked in from main.py wrappers, not from each module
        # individually. Safe to call from any thread.
        self.session_log = SessionLog(CONFIG_DIR / "sessions")
        # Magic-mix recommender — ranks "what plays next" by tag overlap,
        # BPM proximity, recency penalty, variety bonus, star bonus.
        try:
            self.magic = MagicMix(
                CONFIG_DIR / "clips.json",
                CONFIG_DIR / "scratch.json",
                CONFIG_DIR / "path_tags.db3",
            )
            logger.info("Magic-mix recommender ready")
        except Exception as e:
            logger.warning(f"Magic-mix init failed: {e}")
            self.magic = None
        # Set-arc state restored from settings.json. _set_arc_phase is
        # used by the picker's set-arc boost (set_arc.py); _set_arc_enabled
        # gates the whole feature. Class-level defaults handle a fresh
        # install; this override loads any persisted state.
        self._set_arc_enabled = bool(self.config.get("set_arc_enabled", False))
        self._set_arc_phase = str(
            self.config.get("set_arc_phase", "opening") or "opening"
        )
        if self._set_arc_enabled:
            logger.info(
                f"Set-arc mode restored: phase={self._set_arc_phase}"
            )
        # Stem-separation OSC listener. Opt-in via
        # `stem_osc_listen_port` in settings.json (default off). When
        # enabled + `stem_daemon.py` is running, this listener receives
        # /stem/drums/onset messages and records them so the picker /
        # auto-flip logic can boost based on per-stem onsets instead
        # of full-mix flux. See STEM_SEPARATION_RESEARCH.md.
        self._stem_listener = None
        self._stem_recent_onsets: list = []  # rolling (timestamp, strength)
        self._stem_drum_boost_until: float = 0.0
        stem_port = int(self.config.get("stem_osc_listen_port", 0) or 0)
        if stem_port > 0:
            try:
                from osc_in import OSCListener
                self._stem_listener = OSCListener(
                    host="127.0.0.1", port=stem_port,
                )
                self._stem_listener.on(
                    "/stem/drums/onset", self._on_stem_drum_onset,
                )
                if self._stem_listener.start():
                    logger.info(
                        f"Stem OSC listener: ON, port {stem_port}. "
                        f"Run `python stem_daemon.py --port {stem_port}` "
                        f"to feed it."
                    )
                else:
                    logger.warning(
                        f"Stem OSC listener: bind failed on port {stem_port}"
                    )
                    self._stem_listener = None
            except Exception as e:
                logger.warning(f"Stem listener init failed: {e}")
                self._stem_listener = None
        # Auto-set-arc detector. Reads BPM + flip rate, picks the most
        # appropriate phase, updates self._set_arc_phase if changed.
        # Off by default -- the operator opts in via HTTP / future MK2
        # binding. Even off, we still record flip timestamps so it
        # has data ready when toggled on.
        try:
            from auto_set_arc import AutoSetArc
            self._auto_set_arc = AutoSetArc()
        except Exception as e:
            logger.debug(f"AutoSetArc init failed: {e}")
            self._auto_set_arc = None
        self._set_arc_auto = bool(self.config.get("set_arc_auto", False))
        if self._set_arc_auto:
            logger.info("Set-arc AUTO detection ON (restored from config)")
        # Rolling fire history for magic-mix recency penalty.
        self._fire_history: list[str] = []
        # Today-counters for analytics. Increment from event paths; reset
        # at local midnight via the 2.5s timer's date-change check.
        self._analytics_today = {"beats": 0, "flips": 0, "fires": 0, "last_mday": None}
        # OSC broadcaster — sends BPM/beat/clip/fire/xfade to external
        # software (Resolume, TouchDesigner, QLC+, lighting). Disabled
        # by default; set osc_enabled:true + osc_host/osc_port in settings.
        try:
            self.osc = OSCBroadcaster(
                host=str(self.config.get("osc_host", "127.0.0.1")),
                port=int(self.config.get("osc_port", 9000)),
                enabled=bool(self.config.get("osc_enabled", False)),
            )
            if self.config.get("osc_enabled", False):
                logger.info(f"OSC broadcaster enabled → {self.config.get('osc_host','127.0.0.1')}:{self.config.get('osc_port',9000)}")
        except Exception as e:
            logger.warning(f"OSC init failed: {e}")
            self.osc = None
        self._osc_beat_counter = 0
        # Analytics — read-only stats over clips + proxy/thumb dirs.
        try:
            self.analytics = Analytics(
                CONFIG_DIR / "clips.json",
                CONFIG_DIR / "proxy",
                CONFIG_DIR / "thumbnails",
            )
        except Exception as e:
            logger.warning(f"Analytics init failed: {e}")
            self.analytics = None
        # AI clip tagger. Disabled by default — set ai_tagger_enabled:true
        # in settings.json AND configure ANTHROPIC_API_KEY (env or
        # ai_tagger_api_key in settings) to actually fire on clip save.
        self.ai_tagger = None
        if _AI_TAGGER_AVAILABLE and bool(self.config.get("ai_tagger_enabled", False)):
            try:
                backend = str(self.config.get("ai_tagger_backend", "cloud"))
                api_key = self.config.get("ai_tagger_api_key") or os.environ.get("ANTHROPIC_API_KEY")
                self.ai_tagger = AITagger(
                    backend=backend,
                    api_key=api_key,
                    max_tags=5,
                )
                logger.info(f"AI tagger ready (backend={backend})")
            except Exception as e:
                logger.warning(f"AI tagger init failed: {e}")
                self.ai_tagger = None
        # Path-derived auto-tag index. SQLite-backed, scans library tree
        # in the background. Surfaces folder/filename tokens as filter
        # chips on the iPad library.
        self.path_tags = PathTagIndex(CONFIG_DIR / "path_tags.db3")
        # Vote store: hardware up/down votes during live VJ. Picker
        # uses score as a multiplicative selection weight so up-voted
        # clips rise to top, down-voted clips get suppressed (but not
        # eliminated). See vote_store.py for schema + weight curve.
        try:
            from vote_store import VoteStore
            self.vote_store = VoteStore(CONFIG_DIR / "votes.db3")
            logger.info(
                f"[votes] loaded — {self.vote_store.total_voted()} "
                "files have votes"
            )
        except Exception as e:
            logger.warning(f"vote store init failed: {e}")
            self.vote_store = None
        # Proxy cache: 4K → 1080p H.264 NVENC in the background. The
        # original path is still the source of truth (clips, decks,
        # library) — the proxy is an invisible swap at load time.
        # Plex-style: finished proxies persist under ~/.setpiece/proxy
        # and are LRU-evicted once they exceed the disk budget. Budget is
        # proxy_cache_gb in settings.json (default 100 GB).
        _proxy_gb = float(self.config.get("proxy_cache_gb", 100) or 100)
        self.proxy_cache = ProxyCache(
            max_cache_bytes=int(_proxy_gb * 1024 ** 3),
        )
        logger.info(f"Proxy cache: {_proxy_gb:.0f} GB disk budget, LRU eviction")
        # Boot RUNNING by default now (was paused) — the user wants the
        # Plex-style cache building as they play. Transcodes run at
        # BELOW_NORMAL priority, 1 at a time, queue capped at 3, so it
        # won't fight mpv/Qt. Set proxy_auto_start_paused:true in
        # settings.json to restore the old pause-on-boot behaviour.
        if bool(self.config.get("proxy_auto_start_paused", False)):
            self.proxy_cache.pause()
            logger.info("Proxy cache started in PAUSED state (toggle via iPad)")
            # Publish so iPad shows the paused state immediately.
            try:
                self.state.set_proxy_status({
                    "paused": True, "queued": 0, "in_flight": "",
                })
            except Exception:
                pass
        # Working-set hot folder. Default ~/Setpiece-Working/, override via
        # `working_set_folder` in settings.json. Drag files in → auto-
        # queue proxy transcode + status nudge. We deliberately DON'T
        # queue existing files at startup — that would speculatively
        # transcode files you may never play, pegging the GPU for ages.
        # Proxies build on-demand: load_video, deck-load, and new-file-
        # in-watch-folder events all queue.
        ws_path = (self.config.get("working_set_folder") or "").strip()
        ws_folder = Path(ws_path) if ws_path else WORKING_DIR
        self.working_set = WorkingSetWatcher(
            folder=ws_folder,
            on_added=self._on_working_set_added,
            on_removed=self._on_working_set_removed,
        )
        # Optional second hot folder — anything copied / dropped here via
        # Explorer auto-appends to the Scratch basket. Different from the
        # working-set folder: no proxy queue, just curation. Default
        # ~/Setpiece-Scratch/, override via `scratch_folder` in settings.
        sc_path = (self.config.get("scratch_folder") or "").strip()
        sc_folder = Path(sc_path) if sc_path else (Path.home() / "Setpiece-Scratch")
        self.scratch_watcher = WorkingSetWatcher(
            folder=sc_folder,
            on_added=self._on_scratch_folder_added,
            on_removed=None,  # removing the file shouldn't dump the entry
        )
        # The watcher seeds existing files as "already known" so they
        # don't trigger on_added at startup. For the scratch folder we
        # WANT them in the basket though — bulk-add existing entries.
        # scratch_add is idempotent so duplicate paths are no-ops.
        try:
            existing = self.scratch_watcher.list_files()
            if existing:
                logger.info(f"Scratch folder startup sync: {len(existing)} existing file(s)")
                for p in existing:
                    self.scratch_store.add(str(p))
        except Exception as e:
            logger.debug(f"Scratch startup sync failed: {e}")
        # Maschine MK2 — opt-in via settings. Requires NIHardwareAgent
        # to be stopped (it holds an exclusive USB lock otherwise).
        # First pass = discovery mode: just logs HID report shapes so
        # we can pin down pad-byte offsets for this user's firmware.
        self.mk2: Optional[MaschineMK2] = None
        if self.config.get("maschine_mk2_enabled", False):
            try:
                self.mk2 = MaschineMK2(
                    on_pad_press=self._on_mk2_pad_press,
                    on_pad_release=self._on_mk2_pad_release,
                    on_button_press=self._on_mk2_button_press,
                    on_button_release=self._on_mk2_button_release,
                    on_encoder_delta=self._on_mk2_encoder_delta,
                    on_error=lambda m: logger.warning(f"MK2: {m}"),
                    discovery_mode=True,
                )
                ok, msg = self.mk2.start()
                logger.info(f"MK2 start: ok={ok} msg={msg}")
                if not ok:
                    self.mk2 = None
                else:
                    # Brief startup blip so you can see the controller is
                    # alive even before scratch has anything, then settle
                    # into the real-state pad map.
                    try:
                        self.mk2.set_pad_colors_by_label(
                            {n: (40, 60, 80) for n in range(1, 17)}
                        )
                    except Exception:
                        pass
                    # Paint Group A-H LEDs immediately so the device
                    # looks alive on boot. Active bank reads from
                    # bank_store; others dim to ~18% theme color.
                    try:
                        self._refresh_mk2_group_leds()
                    except Exception as e:
                        logger.debug(f"initial group LED paint failed: {e}")
                    # Light the lower transport row at a low baseline
                    # so PLAY/STOP/NEXT/PREV all visibly "exist" instead
                    # of looking dead. PLAY brightens further when
                    # video is actually playing — see _refresh_mk2_transport.
                    try:
                        self._refresh_mk2_transport_leds()
                    except Exception as e:
                        logger.debug(f"initial transport LED paint failed: {e}")
            except Exception as e:
                logger.warning(f"MK2 init failed: {e}")
                self.mk2 = None
        # Channels are seeded with the user's current library_root so
        # first-run defaults all point somewhere real (instead of the
        # iPad showing 4 chips with empty folders).
        self.channels_store = ChannelStore(
            CONFIG_DIR / "channels.json",
            default_folder=self.config.get("library_root", DEFAULT_LIBRARY) or "",
        )
        self.player: Optional[VideoPlayer] = None
        self.audio_reactive: Optional[AudioReactive] = None
        self.s2: Optional[S2Controller] = None
        self.http_server: Optional[HTTPServerThread] = None
        self.stream_deck = None  # type: Optional[StreamDeckController]

        # Per-deck MJPEG preview streams → multipart/x-mixed-replace to
        # iPad. One ffmpeg per loaded deck. Disabled silently if ffmpeg
        # isn't on PATH.
        #
        # Decode path: SOFTWARE by default (preview_hwaccel:false). Four
        # 4K NVDEC decode sessions are fat in VRAM — on a 4GB card that
        # alone can push the GPU into memory pressure (the app "chugs"
        # even though GPU compute % looks low). Software decode at 3fps
        # is cheap on the CPU and leaves VRAM for the live player. Set
        # preview_hwaccel:true in settings.json on an 8GB+ card.
        self.preview_streams: PreviewStreamManager = PreviewStreamManager(
            prefer_hwaccel=bool(self.config.get("preview_hwaccel", False)),
        )

        # Beat-driven flip debounce. Beat detection itself fires often
        # (every snare, hat, kick, etc) — at musical tempos that's 4-8
        # beats/sec. Flipping that fast is unwatchable, so enforce a
        # minimum gap between flip-on-beat events, independent of the
        # onset detector's own min_interval.
        self._min_flip_interval = 4.0  # seconds (raised from 1.5 — was making flips a strobe rather than a beat)
        self._last_flip_beat_time = 0.0
        # Auto-flip phrase length in beats. Adjusted via A RIGHT encoder.
        # Power-of-2 musical phrases — 8 = one bar of 4/4 with kick on
        # every other beat; 16 = a "phrase"; 32 = two phrases.
        self._flip_beats = 8
        self._jog_mode_idx = 1  # default = medium

        # Auto-flip mode state. _random_next + _auto_flip_use_folder are
        # toggled from the S2 action-worker thread (toggle_random_next /
        # toggle_flip_on_beat) and read from the Qt thread (flip / auto_flip).
        # A dedicated lock keeps the tri-state OFF→BANK→FOLDER machine and
        # the random toggle consistent. Instance attrs (not class attrs) so
        # a second instance never shares them. (Audit fix H1 / H2.)
        # (Was OFF→DECKS→FOLDER before 2026-05-16; DECKS mode dropped per
        # user req — auto-flip stays tied to the active bank, deck firing
        # belongs to S2 pads.)
        self._auto_flip_lock = threading.Lock()
        self._random_next = False
        self._auto_flip_use_folder = False  # False = cycle bank/scratch pool, True = cycle folder

        # Phrase-based cut density (2026-05-19): auto-detect the
        # musical downbeat by tracking per-beat kick energy bucketed
        # by beat-position-mod-4. _downbeat_phase is whichever of the
        # 4 positions carries the most kick energy = the "1". Used by
        # _on_beat to redistribute auto-flips onto the downbeat
        # instead of a flat metronome interval.
        from collections import deque as _dq
        self._downbeat_phase = 0
        self._downbeat_energy = [_dq(maxlen=32) for _ in range(4)]

        # Motif callbacks (2026-05-19): clips that were hero-held at a
        # PEAK are the set's recurring "motifs". They get registered
        # here and re-deployed on a later peak as theatrical callbacks
        # — self-bootstrapping, no vote/embedding dependency. A
        # per-clip cooldown stops echo-chambering.
        self._motif_registry: list[str] = []
        self._motif_last_played: dict[str, float] = {}

        # Play-history stack for the `<` button (2026-05-16, user request:
        # "if a fire happens but the video that was on was good would like
        # to go back to it in nearly same spot it was"). Tuple of
        # (path, position_seconds). load_video() pushes the OUTGOING clip
        # on every clip-change; flip_back() pops + uses the mpv `start`
        # preload trick to land on the remembered position.
        #
        # _flip_back_in_progress is the "don't push during pop" guard —
        # otherwise pressing < would push the about-to-be-replaced clip
        # back onto the stack and the user would oscillate.
        self._flip_history: list[tuple[str, float]] = []
        self._flip_back_in_progress: bool = False

        # Thumb backfill dedup: rapid peek-spam shouldn't spawn 5
        # parallel ffmpeg processes all writing the same JPEG. Set
        # holds paths currently being backfilled; freed in finally.
        self._thumb_backfill_lock = threading.Lock()
        self._thumb_backfill_inflight: set[str] = set()
        # Per-file last auto-seek target, keyed by filepath. Used by
        # _auto_seek_into_body / _compute_body_seek_target to spread visits
        # across a clip. Instance attr + capped (see _remember_auto_seek)
        # so it can't leak across a long session. (Audit fix H10.)
        self._last_auto_seek_per_file: dict = {}

        # UI
        self._build_ui()

        # Init non-Qt-window-dependent subsystems immediately
        self._init_audio_reactive()
        self._init_s2()
        self._init_stream_deck()
        self._init_http_server()

        # Player init MUST happen after window.show() so winId() returns
        # a real, mapped HWND. Otherwise libmpv attaches to a phantom
        # window and renders to a void (black video, audio works).
        # Deferred via QTimer.singleShot(0) which fires after the show event.
        QTimer.singleShot(0, self._init_player_deferred)
        # Fullscreen hang-watchdog — also post-show so winId() is real.
        QTimer.singleShot(0, self._init_hang_watchdog)

        # 10Hz state sync timer (player → AppState)
        self._tick_timer = QTimer()
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.start(100)

        # UI-thread heartbeat for the fullscreen hang-watchdog. A plain
        # monotonic stamp bumped on the Qt thread; the watchdog thread reads
        # it to tell a live event loop from a wedged one.
        self._ui_heartbeat = time.monotonic()
        self._hb_timer = QTimer()
        self._hb_timer.timeout.connect(self._beat)
        self._hb_timer.start(500)

        # Refresh clips for iPad
        self._refresh_clips_for_ipad()
        self._refresh_scenes_for_ipad()
        self._refresh_scratch_for_ipad()
        self._refresh_scratch_sets_for_ipad()
        self._refresh_banks_for_ipad()
        self._refresh_smart_playlists_for_ipad()
        # Kick off the path-tag index scan in the background. First run
        # on a TB-scale library takes a minute or two; subsequent runs
        # skip unchanged files via (mtime, size) check.
        self._kick_path_tag_scan()
        # Schedule periodic top-tags refresh so the iPad picks up new
        # tags as the scan progresses without polling.
        from PyQt6.QtCore import QTimer as _QT
        self._path_tags_timer = _QT(self)
        self._path_tags_timer.timeout.connect(self._refresh_path_tags_top)
        self._path_tags_timer.timeout.connect(self._refresh_proxy_status)
        # Periodic OLED refresh so the displayed BPM tracks audio-reactive
        # without waiting for a pad-LED state change.
        self._path_tags_timer.timeout.connect(self._refresh_mk2_pad_leds)
        self._path_tags_timer.timeout.connect(self._refresh_analytics)
        self._path_tags_timer.start(2500)  # every 2.5s

        # Restore deck slots from disk + backfill missing filmstrips
        self._init_decks()

        # Channels: load saved presets + push to AppState so the iPad
        # chip row renders as soon as it polls. LED feedback is deferred
        # until S2 is up — _init_s2 above already ran, so that's fine.
        self._init_channels()

        # Library: scan default folder so iPad has something to show on
        # first load. Failures are non-fatal — UI just shows an empty list.
        self._init_library()
        # Note: last video is loaded by _init_player_deferred (player must exist)

        # Plex-style whole-library proxy pre-build. Deferred 8s so boot +
        # the first live load settle first, then it walks the working-set
        # / scratch / library folders and queues every un-cached file onto
        # the proxy cache's BACKGROUND batch queue. Gated by proxy_prebuild
        # in settings.json (default on). No-op if the proxy cache is paused.
        if bool(self.config.get("proxy_prebuild", True)):
            QTimer.singleShot(8000, self._kick_proxy_prebuild)

    def _kick_proxy_prebuild(self):
        """Queue the CURATED hot folders for background proxy transcode.
        Runs the folder walk in a daemon thread so it can't stall the Qt
        loop. Idempotent — already-cached files are skipped.

        Deliberately only walks the working-set + scratch folders, NOT
        library_root. library_root is a *browse* root and can be a
        TB-scale archive (e.g. D:\\Recycle Bin with thousands of files) —
        blindly transcoding all of it on every boot is wrong. Library
        files get proxies on-demand at load time (load_video queues one),
        and batch_transcode_recent.py is the tiered bulk tool for
        pre-filling a big folder. Working-set + scratch are the small,
        curated 'I'm working with these' sets — those make sense to
        prebuild eagerly."""
        folders = []
        # Working-set + scratch hot folders — use the watchers' resolved
        # paths (these honour the home-dir defaults).
        for watcher_attr in ("working_set", "scratch_watcher"):
            w = getattr(self, watcher_attr, None)
            if w is not None:
                try:
                    folders.append(str(w.folder))
                except Exception:
                    pass
        if not folders:
            return

        def _walk():
            try:
                n = self.proxy_cache.prebuild(folders)
                if n:
                    self.state.set_message(
                        f"Proxy pre-build: {n} file(s) queued in background"
                    )
            except Exception as e:
                logger.debug(f"proxy prebuild kick failed: {e}")

        threading.Thread(
            target=_walk, name="proxy-prebuild-kick", daemon=True,
        ).start()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top control strip
        top = QHBoxLayout()
        top.setContentsMargins(8, 4, 8, 4)
        btn_open = QPushButton("Open Video")
        btn_open.clicked.connect(self.open_video_dialog)
        self.label_file = QLabel("No video loaded")
        self.label_file.setFont(QFont("Mono", 10))
        self.label_ip = QLabel("")
        self.label_ip.setStyleSheet("color: #888;")
        top.addWidget(btn_open)
        top.addWidget(self.label_file, 1)
        top.addWidget(self.label_ip)
        layout.addLayout(top)

        # QVideoWidget directly in layout. Wrapping in a styled container
        # makes Qt paint the container background over the video output.
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_widget, 1)

        # Bottom control strip
        bot = QHBoxLayout()
        bot.setContentsMargins(8, 4, 8, 4)
        btn_play = QPushButton("Play (Space)")
        btn_pause = QPushButton("Pause")
        btn_flip = QPushButton("Flip (B)")
        btn_in = QPushButton("Mark IN (I)")
        btn_out = QPushButton("Mark OUT (O)")
        btn_play.clicked.connect(self.play)
        btn_pause.clicked.connect(self.pause)
        btn_flip.clicked.connect(self.flip)
        btn_in.clicked.connect(self.mark_in)
        btn_out.clicked.connect(self.mark_out)
        bot.addWidget(btn_play)
        bot.addWidget(btn_pause)
        bot.addStretch()
        bot.addWidget(btn_in)
        bot.addWidget(btn_out)
        bot.addStretch()
        bot.addWidget(btn_flip)
        layout.addLayout(bot)

        # Keyboard shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(self.toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_I), self).activated.connect(self.mark_in)
        QShortcut(QKeySequence(Qt.Key.Key_O), self).activated.connect(self.mark_out)
        QShortcut(QKeySequence(Qt.Key.Key_B), self).activated.connect(self.flip)
        QShortcut(QKeySequence(Qt.Key.Key_F), self).activated.connect(self.toggle_fullscreen)
        # ESC always *exits* fullscreen (never enters) — the universal
        # "get me out" key. F still toggles.
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self).activated.connect(self._exit_fullscreen)
        # Shader overlays — toggleable visual texture on TOP of whatever
        # clip is playing. Each shader is a mpv user-shader at shaders/*.glsl.
        # Add more by dropping a .glsl into that folder + a hotkey here.
        QShortcut(QKeySequence("Shift+S"), self).activated.connect(
            lambda: self.toggle_shader("scanlines"))
        QShortcut(QKeySequence("Shift+V"), self).activated.connect(
            lambda: self.toggle_shader("vignette"))

    def _load_config(self) -> dict:
        if self.config_file.exists():
            try:
                cfg = json.loads(
                    self.config_file.read_text(encoding="utf-8"))
                return self._normalize_config(cfg)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        return {
            "last_video": "",
            "volume": 80,
            "audio_reactive_sensitivity": 1.8,
            "audio_reactive_device_substring": "",
            # Hero-hold-on-drop (set-craft rule from "Architectural
            # Paradigms in Live Visual Performance"): when a drop is
            # detected, suppress auto-flip for N beats so the current
            # hero clip rides through the peak. Opt-in by default —
            # auto-flipping on every beat is fine for many sets and
            # this changes the feel.
            "drop_detection_enabled": False,
            "drop_hero_hold_beats": 16,
            "drop_energy_delta_threshold": 2.5,
            "library_root": DEFAULT_LIBRARY,
            "library_folder": DEFAULT_LIBRARY,
        }

    @staticmethod
    def _normalize_config(cfg: dict) -> dict:
        """Auto-promote any legacy/inconsistent config shapes to the
        canonical form so downstream code can assume a single shape.

        Shapes normalized:
        1. ``mk2_vertical_pages`` entries that are
           ``{"name": str, "verticals": list}`` dicts → flattened to
           plain lists, names captured into ``mk2_vertical_page_names``.
           Bug seen 2026-05-18: pages 6/7/8 written as dicts caused
           ``page[slot]`` KeyError → vertical buttons silently failed.
        """
        pages = cfg.get("mk2_vertical_pages") or []
        names = cfg.get("mk2_vertical_page_names")
        # names is canonically a parallel list; default empty list
        if not isinstance(names, list):
            names = list(names) if isinstance(names, (tuple,)) else []
        # Ensure names list is long enough for any pages with custom labels
        while len(names) < len(pages):
            names.append(f"page {len(names)+1}")
        promoted = 0
        for i, page in enumerate(pages):
            if isinstance(page, dict):
                page_name = page.get("name", f"Page{i+1}")
                pages[i] = list(page.get("verticals", []) or [])
                if i < len(names):
                    names[i] = page_name
                promoted += 1
        if promoted:
            logger.info(
                f"[config-normalize] promoted {promoted} dict-shape "
                f"page(s) to list shape (names preserved into "
                f"mk2_vertical_page_names)"
            )
        cfg["mk2_vertical_pages"] = pages
        cfg["mk2_vertical_page_names"] = names
        return cfg

    _config_lock = threading.Lock()

    def _save_config(self):
        """Atomic config write — serialize to a temp file then os.replace
        so a crash mid-write can't corrupt settings.json. (Audit fix C2.)
        Guarded by a lock since multiple paths can trigger a save."""
        try:
            with self._config_lock:
                tmp = self.config_file.with_suffix(".json.tmp")
                tmp.write_text(
                    json.dumps(self.config, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                os.replace(str(tmp), str(self.config_file))
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def _init_player_deferred(self):
        """Called after window.show() so the QVideoWidget has a real HWND."""
        try:
            self.player = VideoPlayer(self.video_widget)
            logger.info("Player initialized (deferred, post-show)")
            # Pin video audio to a specific output if configured. Use case:
            # route to the S2's Monitor output so video audio doesn't bleed
            # into the SMSL loopback the beat detector listens to. The S2's
            # MIX/CUE knob then controls how much video audio reaches the
            # headphones — purely passive, no software involvement.
            sub = (self.config.get("mpv_audio_device_substring") or "").strip()
            if sub:
                try:
                    self.player.set_audio_device_by_substring(sub)
                except Exception as e:
                    logger.debug(f"set_audio_device_by_substring failed: {e}")
            # Mute at startup. Previously mpv defaulted to volume=80 in
            # player_mpv.py, which meant audio leaked the moment the
            # app launched (regardless of where MIX knob sat). Now we
            # set 0 by default so launch is SILENT, and the S2 MIX knob
            # (or audio_mute_toggle action) brings it in.
            startup_vol = int(self.config.get("mpv_startup_volume", 0))
            try:
                self.player.set_volume(startup_vol)
                logger.info(f"mpv startup volume set to {startup_vol}")
                # Remember a sane unmute target — used by audio_mute_toggle
                # when the user toggles from 0 -> non-zero with no prior
                # context (e.g. knob hasn't been touched yet).
                self._audio_last_unmute_vol = (
                    startup_vol if startup_vol > 0
                    else int(self.config.get("audio_unmute_default", 65))
                )
            except Exception as e:
                logger.debug(f"startup volume set failed: {e}")
            # Restore last video now that player exists. force=True so
            # the authoritative startup restore isn't debounced away by
            # an early deck-preview or pad-press load. (Audit fix M1.)
            last = self.config.get("last_video", "")
            if last and Path(last).exists():
                self.load_video(last, force=True)
        except Exception as e:
            logger.error(f"Player init failed: {e}")
            self.state.set_message(f"Player error: {e}", error=True)

    def _init_audio_reactive(self):
        try:
            sensitivity = float(self.config.get("audio_reactive_sensitivity", 1.8))
            device_sub = self.config.get("audio_reactive_device_substring", "") or ""
            # Drop detection is opt-in. When disabled (default) we still
            # pass the callback so the user can flip it on at runtime
            # without re-init — DropDetector.set_enabled handles the
            # gate internally.
            drop_enabled = bool(self.config.get("drop_detection_enabled", False))
            drop_thresh = float(self.config.get("drop_energy_delta_threshold", 2.5))
            # Backend selection — settings.json key "beat_detection_backend".
            # Default "spectral_flux" preserves the existing ~40-LOC NumPy
            # onset pipeline (audio_reactive.py). Setting it to "beatnet"
            # swaps in the BeatNet PLL detector (predicts the next beat
            # via a particle filter instead of reacting to onsets). The
            # on_beat callback contract is identical, so picker /
            # lyric_drive / S2 / OLED don't know or care which is firing.
            backend = str(
                self.config.get("beat_detection_backend", "spectral_flux")
            ).strip().lower()
            if backend == "beatnet":
                try:
                    from beatnet_detector import BeatNetDetector
                    BackendCls = BeatNetDetector
                    logger.info("Audio-reactive backend: BeatNet (PLL)")
                except Exception as _e:
                    logger.warning(
                        "BeatNet backend requested but import failed (%s) — "
                        "falling back to spectral_flux", _e)
                    BackendCls = AudioReactive
            else:
                BackendCls = AudioReactive
            self.audio_reactive = BackendCls(
                on_beat=self._on_beat,
                on_bpm=self._on_bpm,
                on_error=self._on_audio_reactive_error,
                on_drop_detected=self._on_drop_detected,
                sensitivity=sensitivity,
                device_substring=device_sub,
                drop_energy_delta_threshold=drop_thresh,
            )
            # Mirror initial sensitivity into AppState so the iPad shows
            # the right value before the crossfader is ever touched.
            self.state.set_audio_reactive(enabled=False, sensitivity=sensitivity)
            # Sync detector enable state with the config flag.
            dd = getattr(self.audio_reactive, "_drop_detector", None)
            if dd is not None:
                dd.set_enabled(drop_enabled)
        except Exception as e:
            logger.warning(f"Audio reactive unavailable: {e}")

    def _init_s2(self):
        try:
            self.s2 = S2Controller()

            # Buttons
            self.s2.on_action("a_loop_in", self.mark_in)
            self.s2.on_action("a_loop_out", self.mark_out)
            self.s2.on_action("a_play", self.toggle_play)
            self.s2.on_action("a_cue", self.pause)
            # A PFL = AUTO-FLIP ON BEAT toggle (was: restart_video — that
            # action is still on A SYNC). The lit bar above the A volume
            # fader is the most prominent S2 button for a frequent toggle.
            self.s2.on_action("a_pfl", self.toggle_flip_on_beat)
            self.s2.on_action("b_play", self.flip)
            self.s2.on_action("b_cue", self.flip_back)
            # Load buttons → push current library selection into preview decks
            self.s2.on_action("a_load", lambda: self.load_selected_to_deck(0))
            self.s2.on_action("b_load", lambda: self.load_selected_to_deck(1))
            self.s2.on_action("a_samples", lambda: self.load_selected_to_deck(2))
            self.s2.on_action("b_samples", lambda: self.load_selected_to_deck(3))
            # CONTROLLER_LAYOUT.md Part 3 — A=LIVE, B=NEXT. The four
            # buttons that ALL did reset_speed now each have a real job:
            #   A SYNC  → restart LIVE from the top   (re-trigger)
            #   B SYNC  → random next file            (shuffle now)
            #   A RESET → reset playback speed → 1.0x (the ONE speed-reset)
            #   B RESET → BLACKOUT toggle             (pinned global panic)
            self.s2.on_action("a_sync", self.restart_video)
            self.s2.on_action("b_sync", self.flip_random)
            self.s2.on_action("a_reset", self.reset_speed)
            self.s2.on_action("b_reset", self.blackout_toggle)
            self.s2.on_action("fx1_focus", self.cycle_jog_sensitivity)
            self.s2.on_action("browse_enc_press", self.fire_selected_library_file)
            # Audio mute toggle on encoder PRESS. User wants a clean
            # kill-switch for video audio. The S2's GAIN encoder presses
            # are unused otherwise and easy to reach. Wired to BOTH A
            # and B sides so user can press whichever hand is free. If
            # the user prefers a different press, rebind in settings or
            # edit here. Other available presses (currently unbound):
            # a_left_enc_press, a_right_enc_press, b_left_enc_press,
            # b_right_enc_press.
            self.s2.on_action("a_gain_enc_press", self.audio_mute_toggle)
            self.s2.on_action("b_gain_enc_press", self.audio_mute_toggle)
            # SHIFT + browse encoder PRESS = load to currently-selected
            # crossfade preview deck (set by FX2 cluster). Prevents the
            # foot-gun where SHIFT-held + browse-press otherwise falls
            # through to fire-LIVE during a load gesture.
            self.s2.on_action("browse_enc_press_shift", self._load_selected_to_preview_deck)

            self.s2.on_action("b_loop_in", self.mark_in)
            self.s2.on_action("b_loop_out", self.mark_out)

            # ── FX1 cluster = automations + audio-reactive ─────────────────
            # FX1 Param 1: tap = silence/un-silence auto-flip (BPM lock keeps
            # tracking either way). SHIFT+tap = toggle random vs sequential
            # next-video order. Lets you keep beat detection running while
            # playing manually.
            self.s2.on_action("fx1_param1_btn", self.toggle_flip_on_beat)
            self.s2.on_action("fx1_param1_btn_shift", self.toggle_random_next)
            self.s2.on_action("fx1_param2_btn", self.audio_reactive_start)
            self.s2.on_action("fx1_param3_btn", self.audio_reactive_stop)

            # Halve/double moved entirely to the transport area
            # (SHIFT+SYNC = halve, SHIFT+CUE_transport = double).
            # SHIFT + Loop Out = save the currently-active loop bounds as
            # a clip (no remarking, no losing the tight bounds you just
            # halved into place). Auto-applies the sticky active tag.
            self.s2.on_action("a_loop_out_shift", self.save_loop_as_clip)
            self.s2.on_action("b_loop_out_shift", self.save_loop_as_clip)
            # SHIFT + Loop In = toggle auto-flip on/off without reaching
            # for FX1 Param 1. BPM lock keeps running either way. Same
            # action on both decks so muscle memory carries.
            self.s2.on_action("a_loop_in_shift", self.toggle_flip_on_beat)
            self.s2.on_action("b_loop_in_shift", self.toggle_flip_on_beat)

            # SHIFT + CUE (transport) = DOUBLE loop, SHIFT + CUE (monitor) = frame FORWARD
            # SHIFT + RESET = clear all clips on this video (less common,
            # promoted off non-shift since RESET is now reset-tempo).
            self.s2.on_action("a_cue_shift", lambda: self.loop_length_nudge(1))
            self.s2.on_action("a_pfl_shift", lambda: self.frame_step(forward=True))
            self.s2.on_action("b_cue_shift", lambda: self.loop_length_nudge(1))
            self.s2.on_action("b_pfl_shift", lambda: self.frame_step(forward=True))
            self.s2.on_action("a_reset_shift", self.clear_current_video_clips)
            self.s2.on_action("b_reset_shift", self.clear_current_video_clips)

            # SHIFT + sync = HALVE loop length (was tap_tempo — recover BPM
            # via the background analyzer instead).
            self.s2.on_action("a_sync_shift", lambda: self.loop_length_nudge(-1))
            self.s2.on_action("b_sync_shift", lambda: self.loop_length_nudge(-1))

            # ── EQ knob cluster (CONTROLLER_LAYOUT.md Part 3) ──────────────
            # One coherent colour/visual cluster. Speed is OFF the EQ knobs
            # now (it was mapped four ways — fader + LEFT encoder is plenty).
            #   A EQ HI  → saturation     B EQ HI  → brightness
            #   A EQ MID → contrast       B EQ MID → hue
            #   A EQ LOW → zoom / scale   B EQ LOW → audio-reactive sens
            self.s2.on_continuous("a_eq_hi", self._knob_saturation)
            self.s2.on_continuous("b_eq_hi", self.set_master_brightness)

            # ── Encoder rotations ───────────────────────────────────────────
            # A LEFT = tempo. B LEFT repurposed → auto-flip phrase length
            # (1/2/4/8/16/32 beats) since the A-side speed control is
            # enough and the B-side LEFT was a redundant duplicate. Both
            # RIGHT encoders are volume nudge (faders cover most volume
            # work but the encoders are nice for fine touch).
            self.s2.on_encoder("a_left_enc_4bit", self.fine_speed_nudge)
            # a_right was live-volume. Freed 2026-05-18 since volume
            # lives on the ATEM/mixer. Rebound to live saturation nudge.
            self.s2.on_encoder("a_right_enc_4bit", self.saturation_nudge)
            self.s2.on_encoder("b_left_enc_4bit", self.adjust_flip_beats)
            # b_right was a duplicate of a_right (both fine_volume_nudge).
            # Unblocked 2026-05-18 overnight: rebound to live
            # audio-reactive sensitivity nudge so the operator can
            # dial-in beat sensitivity tactilely as the mix changes.
            # (a_right stays as live-volume; only the B-side dup moved.)
            self.s2.on_encoder("b_right_enc_4bit", self.audio_sensitivity_nudge)
            # b_gain was master brightness. Freed 2026-05-18 since
            # brightness lives on the projector/ATEM. Rebound to
            # relative crossfade-blend nudge (B-side = preview/the
            # one you're bringing in, fits muscle memory).
            self.s2.on_encoder("b_gain_enc_4bit", self.crossfade_blend_nudge)
            # A Gain encoder rotate intentionally unbound — press still exits
            # loop. Halve/double moved to transport-area buttons (SHIFT+SYNC
            # / SHIFT+CUE) because the encoder compounds too aggressively
            # (one fast spin = 6+ ticks = loop floored to 0.05s instantly).

            # ── Encoder presses ─────────────────────────────────────────────
            # Encoder presses mirror what their rotations control:
            # A LEFT → reset speed; A RIGHT → reset volume; B LEFT → reset
            # auto-flip beats (default 8); B RIGHT → reset volume.
            self.s2.on_action("a_left_enc_press", self.reset_speed)
            self.s2.on_action("a_right_enc_press", self.reset_volume)
            self.s2.on_action("b_left_enc_press", self.reset_flip_beats)
            # B RIGHT encoder press: lyric toggle moved to B PFL (the lit
            # bar above the B volume fader — more prominent, easier to
            # find by feel during a set). Reverting B RIGHT enc press to
            # its original reset_volume so the encoder cluster is
            # symmetric again. Available to repurpose later if wanted.
            self.s2.on_action("b_right_enc_press", self.reset_volume)
            # B Gain encoder press = reset brightness to normal (0).
            self.s2.on_action("b_gain_enc_press", self.reset_brightness)
            # A Gain encoder press = exit loop (return to normal playback).
            self.s2.on_action("a_gain_enc_press", self.exit_loop)
            # Browse encoder rotation = move library cursor
            self.s2.on_encoder("browse_enc_4bit", self.library_cursor_move)

            # FX2 row → SET CROSSFADE PREVIEW SOURCE (was: fire decks).
            # Tap to make that deck the crossfader's blend target — now
            # you can FADE to any of the 4 decks, not just slot 1.
            self.s2.on_action("fx2_focus", lambda: self.set_preview_deck(0))
            self.s2.on_action("fx2_param1_btn", lambda: self.set_preview_deck(1))
            self.s2.on_action("fx2_param2_btn", lambda: self.set_preview_deck(2))
            self.s2.on_action("fx2_param3_btn", lambda: self.set_preview_deck(3))

            # FX1 channel-enable buttons → channel preset switching.
            self.s2.on_action("fx1_ch1", lambda: self.switch_channel(0))
            self.s2.on_action("fx2_ch1", lambda: self.switch_channel(1))
            self.s2.on_action("fx1_ch2", lambda: self.switch_channel(2))
            self.s2.on_action("fx2_ch2", lambda: self.switch_channel(3))
            # SHIFT + FX2 CH = shader overlay toggles (visual texture on
            # top of the live video). Same effects as Shift+S / Shift+V
            # keyboard, just on hardware. Channel presets stay on the
            # unshifted button — these are an additive layer.
            self.s2.on_action("fx2_ch1_shift", lambda: self.toggle_shader("scanlines"))
            self.s2.on_action("fx2_ch2_shift", lambda: self.toggle_shader("vignette"))

            # A pads = saved cue points (clips for current video).
            # B pads = INSTANT CUT to deck slots 1-4 (replaces FX2 cluster).
            self.s2.on_action("a_pad1", lambda: self.play_clip(0))
            self.s2.on_action("a_pad2", lambda: self.play_clip(1))
            self.s2.on_action("a_pad3", lambda: self.play_clip(2))
            self.s2.on_action("a_pad4", lambda: self.play_clip(3))
            self.s2.on_action("b_pad1", lambda: self.fire_deck(0))
            self.s2.on_action("b_pad2", lambda: self.fire_deck(1))
            self.s2.on_action("b_pad3", lambda: self.fire_deck(2))
            self.s2.on_action("b_pad4", lambda: self.fire_deck(3))
            # SHIFT + B Pad N = load currently browse-selected library file
            # into Deck N (1-4). Quick hardware-only library → deck routing.
            # SHIFT + B Pad N = fire that deck AND seek into the body
            # (skips intros + last 20s, reuses the auto-flip seek logic).
            # Loading-to-deck still works via hold-B-Pad + browse-press,
            # and via the LOAD/SAMPLES buttons.
            self.s2.on_action("b_pad1_shift", lambda: self.fire_deck_random_seek(0))
            self.s2.on_action("b_pad2_shift", lambda: self.fire_deck_random_seek(1))
            self.s2.on_action("b_pad3_shift", lambda: self.fire_deck_random_seek(2))
            self.s2.on_action("b_pad4_shift", lambda: self.fire_deck_random_seek(3))

            # Faders — both deck volume faders control the live player.
            # Whichever fader is higher wins (acts as a "max volume" master),
            # so you can ride either deck's fader without one zeroing it out.
            self.s2.on_continuous("a_volume", lambda v: self._volume_from_fader("a", v))
            self.s2.on_continuous("b_volume", lambda v: self._volume_from_fader("b", v))
            # Crossfader: real alpha-blend between live and preview deck.
            # Falls back to flip-on-midpoint if libmpv lavfi-complex refuses
            # to attach a 2nd track (logged as a warning, see crossfade_blend).
            self.s2.on_continuous("crossfader", self.crossfade_blend)
            # Tempo faders → playback speed (exponential, center = 1.0x)
            self.s2.on_continuous("a_tempo", self.set_speed)
            self.s2.on_continuous("b_tempo", self.set_speed)
            # EQ MID/LOW — rest of the colour/visual cluster (see EQ HI
            # block above). All take a 0..127 raw value; each method maps
            # it to its mpv property's range.
            self.s2.on_continuous("a_eq_mid", self._knob_contrast)
            self.s2.on_continuous("b_eq_mid", self._knob_hue)
            self.s2.on_continuous("a_eq_low", self._knob_zoom)
            self.s2.on_continuous("b_eq_low", self.set_audio_sensitivity)
            # S2 HEADPHONE MIX knob (front panel) = audio leak control.
            # Twist clockwise to bring video audio in, ccw to silence.
            # Continuous + hands-free — DJ-native alternative to the
            # audio_leak_hold MK2 button. Both still work; pick whichever
            # you prefer or use the knob as baseline and button as peak.
            # Routing isolation via mpv_audio_device_substring means this
            # cannot pollute the BPM detector's loopback input.
            self.s2.on_continuous("headphone_mix", self.audio_leak_knob)

            # LEFT jog wheel = scrub video. RIGHT jog wheel = speed
            # control (turntable-style pitch wheel). Wide range without
            # losing the tempo fader's fine control — fader stays at
            # ±13% for precision, jog gives you instant 0.5x..2.0x reach.
            self.s2.on_continuous("a_jog_velocity", self.jog_scrub)
            self.s2.on_continuous("b_jog_velocity", self.jog_speed_nudge)
            # Jog touch: top platter capacitive sensor. Touched = coarse
            # sensitivity (active scrub), released = back to base mode
            # (DJ convention — hold the record for scratch, release for nudge).
            self.s2.on_continuous("a_jog_touch", lambda v: self._jog_touch_changed(v))
            self.s2.on_continuous("b_jog_touch", lambda v: self._jog_touch_changed(v))

            # Connection events update AppState (and trigger LED state refresh)
            self.s2.on_connect(self._on_s2_connect)
            self.s2.on_disconnect(self._on_s2_disconnect)

            ok, msg = self.s2.start()
            self.state.set_s2_connected(ok and "connected" in msg.lower())
            logger.info(f"S2: {msg}")

            # Push initial LED state (armed pads, audio-reactive indicator)
            self._refresh_s2_leds()
        except Exception as e:
            logger.error(f"S2 init failed: {e}")

    def _init_stream_deck(self):
        """Bring up the Stream Deck thumbnail browser. Optional — failure
        here never blocks the rest of the app.

        Disabled by default for this user because the deck has a Windows
        driver bug: output reports rejected (only feature reports work).
        See memory: stream_deck_output_report_bug.md. To re-enable after
        firmware update or different hardware, set in settings.json:
            "stream_deck_enabled": true
        """
        if not self.config.get("stream_deck_enabled", False):
            logger.info("Stream Deck: disabled (set stream_deck_enabled=true in settings.json to enable)")
            return
        if StreamDeckController is None:
            logger.info(
                f"Stream Deck unavailable: {_STREAM_DECK_IMPORT_ERR}. "
                "Install with: pip install streamdeck Pillow"
            )
            return
        try:
            self.stream_deck = StreamDeckController(self)
            ok, msg = self.stream_deck.start()
            logger.info(f"Stream Deck: {msg}")
            if not ok:
                # start() failed cleanly (e.g. missing deps) — drop the
                # reference so closeEvent doesn't try to stop nothing useful
                self.stream_deck = None
        except Exception as e:
            logger.error(f"Stream Deck init failed: {e}", exc_info=True)
            self.stream_deck = None

    def _init_http_server(self):
        try:
            # LAN exposure is opt-in: localhost-only unless the user sets
            # "lan_access": true in settings.json or passes --lan. The
            # control server has no auth, so this stays off by default.
            self.http_server = HTTPServerThread(
                self.state, port=8765,
                preview_manager=self.preview_streams,
                lan=(bool(self.config.get("lan_access", False))
                     or "--lan" in sys.argv),
            )
            self.http_server.register("play", lambda d: self.play())
            self.http_server.register("pause", lambda d: self.pause())
            # MK2 mono-LED discovery helpers (report 0x82). POST these to
            # find which physical buttons have backlight LEDs.
            self.http_server.register(
                "mk2/mono-all-on",
                lambda d: self.mk2.mono_leds_all_on(int(d.get("brightness", 200))) if self.mk2 else False,
            )
            self.http_server.register(
                "mk2/mono-all-off",
                lambda d: self.mk2.mono_leds_all_off() if self.mk2 else False,
            )
            self.http_server.register(
                "mk2/mono-set",
                lambda d: self.mk2.set_mono_leds(
                    {int(k): int(v) for k, v in (d.get("leds") or {}).items()},
                    merge=bool(d.get("merge", True))
                ) if self.mk2 else False,
            )
            # Set-arc auto-detect toggle (iPad / future MK2 binding).
            # Returns the new bool state in `result`.
            self.http_server.register(
                "set-arc/auto",
                lambda d: bool(self.set_arc_auto_toggle()),
            )
            # Also expose the manual phase cycle + on/off via HTTP so
            # the iPad badge can be tapped directly.
            self.http_server.register(
                "set-arc/cycle",
                lambda d: (self.set_arc_cycle(),
                           getattr(self, "_set_arc_phase", "opening"))[1],
            )
            self.http_server.register(
                "set-arc/toggle",
                lambda d: (self.set_arc_toggle(),
                           bool(getattr(self, "_set_arc_enabled", False)))[1],
            )
            self.http_server.register("flip", lambda d: self.flip())
            self.http_server.register("mark_in", lambda d: self.mark_in())
            self.http_server.register("mark_out", lambda d: self.mark_out())
            self.http_server.register("play_clip", lambda d: self.play_clip(d.get("idx", 0)))
            self.http_server.register("clip/delete", lambda d: self.delete_clip(d.get("clip_id", "")))
            self.http_server.register("clip/clear_current", lambda d: self.clear_current_video_clips())
            # Bank: fire any saved clip by id, possibly switching videos.
            self.http_server.register("bank/fire", lambda d: self.fire_clip_by_id(str(d.get("clip_id", ""))))
            # Force re-read of clips.json from disk. Used after batch-
            # external-edits (e.g. live curation scripts that mark
            # clips while the app is running). No restart needed.
            self.http_server.register(
                "clips/reload",
                lambda d: self.clips_db.reload(),
            )
            # Recent picker decisions for iPad debug panel. Returns
            # last N picks with what drove the choice (sequential,
            # random, bpm-match, stem-restrict, kick-jump).
            self.http_server.register(
                "picker/recent",
                lambda d: self.picker_recent(int(d.get("limit", 10))),
            )
            # Vote panel: top-voted clips, bottom-voted clips, current
            # clip's vote stats + category corrections. Visibility into
            # what the system has learned from hardware votes.
            self.http_server.register(
                "votes/summary",
                lambda d: self.votes_summary(
                    int(d.get("limit", 10))
                ),
            )
            # CLIP semantic search — takes a natural-language query +
            # top N, returns ranked clips by visual meaning. Caches
            # the CLIP text encoder + embeddings matrix after first
            # call so subsequent queries are sub-second.
            self.http_server.register(
                "clip/search",
                lambda d: self.clip_search(
                    str(d.get("query", "")).strip(),
                    int(d.get("top", 15) or 15),
                ),
            )
            # Cohesion anchor: see the current "music-video shoot"
            # subset + seconds remaining. Optional POST "refresh":True
            # forces an immediate re-roll.
            self.http_server.register(
                "cohesion/status",
                lambda d: (
                    self._force_cohesion_refresh(reason="ipad")
                    or self.cohesion_status()
                    if d.get("refresh")
                    else self.cohesion_status()
                ),
            )
            # Create a clip segment with explicit in/out. Used by the
            # iPad "Mark clip" form. Accepts a fname substring (we
            # resolve to full path via path_tags) OR a full filepath.
            self.http_server.register(
                "clips/create_segment",
                lambda d: self._create_clip_segment_action(d),
            )
            # Clips filtered by tag -- powers the iPad "Openers"
            # launcher panel. Returns trimmed clip dicts (id, name,
            # filepath, in/out, duration, tags).
            self.http_server.register(
                "clips/by_tag",
                lambda d: {
                    "ok": True,
                    "tag": str(d.get("tag", "") or ""),
                    "clips": [
                        {
                            "id": c.get("id"),
                            "name": c.get("name"),
                            "filepath": c.get("filepath"),
                            "in_seconds": c.get("in_seconds"),
                            "out_seconds": c.get("out_seconds"),
                            "duration": c.get("duration"),
                            "starred": bool(c.get("starred")),
                            "tags": c.get("tags") or [],
                        }
                        for c in self.clips_db.clips_by_tag(
                            str(d.get("tag", "") or "")
                        )
                    ],
                },
            )
            # Scenes: snapshot/restore the 4 deck slots under a name.
            self.http_server.register("scene/save", lambda d: self.scene_save(str(d.get("name", ""))))
            self.http_server.register("scene/load", lambda d: self.scene_load(str(d.get("scene_id", ""))))
            self.http_server.register("scene/delete", lambda d: self.scene_delete(str(d.get("scene_id", ""))))
            self.http_server.register("scene/rename", lambda d: self.scene_rename(str(d.get("scene_id", "")), str(d.get("name", ""))))
            # Scratch list: session basket of library file paths.
            self.http_server.register("scratch/add", lambda d: self.scratch_add(str(d.get("path", ""))))
            self.http_server.register("scratch/add_file", lambda d: self.scratch_add_library_file(str(d.get("file", ""))))
            self.http_server.register("scratch/add_current", lambda d: self.scratch_add_current())
            self.http_server.register("scratch/remove", lambda d: self.scratch_remove(str(d.get("path", ""))))
            self.http_server.register("scratch/clear", lambda d: self.scratch_clear())
            self.http_server.register("scratch/fire", lambda d: self.scratch_fire(str(d.get("path", ""))))
            self.http_server.register("scratch/shuffle", lambda d: self.scratch_shuffle())
            self.http_server.register("scratch/save_set", lambda d: self.scratch_save_set(str(d.get("name", ""))))
            self.http_server.register("scratch/load_set", lambda d: self.scratch_load_set(str(d.get("set_id", ""))))
            self.http_server.register("scratch/delete_set", lambda d: self.scratch_delete_set(str(d.get("set_id", ""))))
            self.http_server.register("deck/clear_all", lambda d: self.decks_clear_all())
            # Proxy transcode controls
            self.http_server.register("proxy/pause", lambda d: self._proxy_pause())
            self.http_server.register("proxy/resume", lambda d: self._proxy_resume())
            # Audio-reactive (live BPM detector) controls — useful for
            # iPad / external tools that want to enable BPM without
            # turning on auto-flip. 2026-05-17 user req: "is there a
            # place I can hit to get just whatever is playing to live bpm"
            self.http_server.register("audio_reactive/start", lambda d: self._http_ar_start())
            self.http_server.register("audio_reactive/stop", lambda d: self._http_ar_stop())
            self.http_server.register("audio_reactive/toggle", lambda d: self._http_ar_toggle())
            # Convenience: just return the current BPM in a tiny payload.
            self.http_server.register("bpm", lambda d: self._http_bpm())
            self.http_server.register("proxy/cancel_current", lambda d: self._proxy_cancel_current())
            self.http_server.register("proxy/cancel_pending", lambda d: self._proxy_cancel_pending())
            # Banks: 8-slot quick-switch (A-H) for scratch lists.
            self.http_server.register("bank/save", lambda d: self.bank_save(str(d.get("letter", "")), str(d.get("name", ""))))
            self.http_server.register("bank/load", lambda d: self.bank_load(str(d.get("letter", ""))))
            self.http_server.register("bank/clear", lambda d: self.bank_clear(str(d.get("letter", ""))))
            self.http_server.register(
                "bank/reroll",
                lambda d: self.bank_reroll_one(str(d.get("letter", "")))
            )
            self.http_server.register(
                "bank/auto_split",
                lambda d: self.bank_auto_split(
                    seed=(int(d["seed"]) if d.get("seed") is not None else None),
                    prefix=str(d.get("prefix", "") or ""),
                ),
            )
            # Recursive folder->banks. Pass folder name (relative to
            # current library_folder OR to library_root). Recurses
            # subfolders by default so banking "Lil Humpers" parent
            # gathers all its year-subfolder clips. The iPad per-folder
            # BANK button and MK2 `>` arrow both use this.
            self.http_server.register(
                "bank/from_folder",
                lambda d: self.bank_from_folder(
                    folder_rel=str(d.get("folder", "") or ""),
                    recursive=bool(d.get("recursive", True)),
                ),
            )
            # Tag search: substring-match path_tags so the iPad can
            # find buried tags (e.g. specific performer names) that
            # never make the top-20 chip strip. Returns
            # {ok, query, results: [{tag, count}]}.
            self.http_server.register(
                "path_tags/search",
                lambda d: self._tag_search_action(
                    str(d.get("q", "") or ""),
                    int(d.get("limit", 30) or 30),
                ),
            )
            # One-shot "go from search to banks": find best-matching
            # tag for the query, set it as filter, then auto-split.
            # The iPad search box ENTER key fires this. Returns the
            # auto_split result so the button can show the dist.
            self.http_server.register(
                "bank/tag_to_banks",
                lambda d: self._tag_to_banks_action(
                    str(d.get("q", "") or ""),
                ),
            )
            # Same flow but driven by CLIP semantic search (visual
            # similarity, natural-language query). "rooftop sunset
            # glow" → top-64 visual matches → categorically split into
            # banks A-H. Sister to bank/tag_to_banks.
            self.http_server.register(
                "bank/clip_to_banks",
                lambda d: self._clip_to_banks_action(
                    str(d.get("q", "") or ""),
                    int(d.get("top", 64) or 64),
                ),
            )
            # Saved CLIP-search "vibes": star a query, recall it later.
            # Persisted to ~/.setpiece/saved_searches.json. The
            # iPad strip renders the most recent 6 as one-tap chips.
            self.http_server.register(
                "saved_vibes/list",
                lambda d: self._saved_vibes_list(),
            )
            self.http_server.register(
                "saved_vibes/save",
                lambda d: self._saved_vibes_save(
                    str(d.get("q", "") or "").strip(),
                ),
            )
            self.http_server.register(
                "saved_vibes/delete",
                lambda d: self._saved_vibes_delete(
                    str(d.get("q", "") or "").strip(),
                ),
            )
            # OPENERS: instant 8-bank set from intro:N-tagged files
            # (sourced by intro_tagger.py). Split by intro LENGTH so
            # A=short snappy openers, H=longest setup pieces.
            self.http_server.register(
                "bank/openers",
                lambda d: self._openers_to_banks_action(),
            )
            # THEMES: curated `theme:` tag layer (theme_apply.py).
            # theme/list -> the chip vocabulary; bank/theme -> load a
            # theme's clips, categorically split across banks A-H.
            self.http_server.register(
                "theme/list",
                lambda d: self._theme_list(),
            )
            self.http_server.register(
                "bank/theme",
                lambda d: self._theme_to_banks_action(
                    str(d.get("theme", "") or "").strip(),
                ),
            )
            # Peek: return a bank's full file list WITHOUT loading it.
            # Drives the iPad "see what's in here before I commit" panel
            # (2026-05-16). Returns {ok, letter, name, folder, files: [...]}.
            self.http_server.register("bank/peek", lambda d: self.bank_peek(str(d.get("letter", ""))))
            # Layer overlays (2026-05-18): cycle the active A-H
            # categorization layer (default/positions/pov/mood/...).
            # Returns {ok, active_layer, order} so the iPad chip can
            # update its label immediately without polling state.
            self.http_server.register(
                "bank/cycle_layer", lambda d: self.cycle_bank_layer()
            )
            # MK2 mirror: return the 8 vertical-button bindings (bits
            # 40-47) so the iPad bank drawer can label them next to
            # the bank/pad grids — same spatial layout as the hardware.
            # 2026-05-16 user req: mirror MK2 layout on iPad.
            self.http_server.register("mk2/verticals", lambda d: self.mk2_get_verticals())
            self.http_server.register("mk2/fire_vertical", lambda d: self.mk2_fire_vertical(int(d.get("idx", -1))))
            # Smart playlists: saved (library_root + tag filter) combos.
            self.http_server.register("smart_playlist/save", lambda d: self.smart_playlist_save(str(d.get("name", ""))))
            self.http_server.register("smart_playlist/apply", lambda d: self.smart_playlist_apply(str(d.get("id", ""))))
            self.http_server.register("smart_playlist/delete", lambda d: self.smart_playlist_delete(str(d.get("id", ""))))
            # Magic-mix recommender.
            self.http_server.register("magic/suggest", lambda d: self.magic_suggest(int(d.get("top_n", 5))))
            # Path-tag filter — iPad sends list of selected tag chips.
            self.http_server.register("path_tags/set_filter", lambda d: self.set_path_tag_filter(d.get("tags", [])))
            # Working-set quick-pivot: swap the library browser to the
            # configured hot folder (or back to the original library root).
            self.http_server.register("working_set/open", lambda d: self._library_pivot_to_working_set())
            self.http_server.register("working_set/exit", lambda d: self._library_pivot_to_default_root())
            # Sticky active tag — auto-applied to every Mark OUT and SHIFT+Loop Out.
            self.http_server.register("active_tag/set", lambda d: self.set_active_tag(str(d.get("tag", ""))))

            # Library browser endpoints (iPad)
            self.http_server.register("library/scan", lambda d: self.library_scan(d.get("folder", "")))
            self.http_server.register("library/cd", lambda d: self.library_cd(d.get("path", "")))
            self.http_server.register("library/load_file", lambda d: self.library_load_file(d.get("file", "")))

            # Deck (launchpad) endpoints
            self.http_server.register(
                "deck/load_clip",
                lambda d: self.load_clip_to_deck(int(d.get("deck", 0)), str(d.get("clip_id", ""))),
            )
            self.http_server.register(
                "deck/load_file",
                lambda d: self.load_file_to_deck(int(d.get("deck", 0)), str(d.get("file", ""))),
            )
            self.http_server.register(
                "deck/clear",
                lambda d: self.deck_clear(int(d.get("deck", 0))),
            )
            self.http_server.register(
                "deck/fire",
                lambda d: {"ok": bool(self.fire_deck(int(d.get("deck", 0))))},
            )

            # Clip metadata: star / tags. All four return the updated
            # clip's tag list (or starred flag) so the iPad can do an
            # optimistic update + reconcile on next poll.
            self.http_server.register(
                "clip/star",
                lambda d: self.clip_set_starred(
                    str(d.get("clip_id", "")),
                    bool(d.get("value", False)),
                ),
            )
            self.http_server.register(
                "clip/tags",
                lambda d: self.clip_set_tags(
                    str(d.get("clip_id", "")),
                    d.get("tags") or [],
                ),
            )
            self.http_server.register(
                "clip/add_tag",
                lambda d: self.clip_add_tag(
                    str(d.get("clip_id", "")),
                    str(d.get("tag", "")),
                ),
            )
            self.http_server.register(
                "clip/remove_tag",
                lambda d: self.clip_remove_tag(
                    str(d.get("clip_id", "")),
                    str(d.get("tag", "")),
                ),
            )

            # BPM endpoints. The iPad already gets per-clip bpm in
            # state.clips[*].bpm, but a dedicated lookup is handy when
            # the UI wants to show an "analyzing..." spinner for a
            # specific clip without waiting for the next poll.
            self.http_server.register(
                "clip/bpm",
                lambda d: self.clip_get_bpm(str(d.get("clip_id", ""))),
            )
            self.http_server.register(
                "clip/reanalyze",
                lambda d: self.clip_reanalyze_bpm(str(d.get("clip_id", ""))),
            )

            # Channels (saved library-folder presets) endpoints
            self.http_server.register(
                "channel/switch",
                lambda d: self.switch_channel(int(d.get("idx", -1))),
            )
            self.http_server.register(
                "channel/save",
                lambda d: self.save_channel(
                    int(d.get("idx", -1)),
                    name=d.get("name"),
                    folder=d.get("folder"),
                    color=d.get("color"),
                    tag_filter=d.get("tag_filter"),
                ),
            )
            self.http_server.register(
                "channel/set_current",
                lambda d: self.set_current_as_channel(int(d.get("idx", -1))),
            )

            ip = self.http_server.start()
            self.label_ip.setText(f"iPad: http://{ip}:8765")
            # Give the request handler a ref to the app so endpoints
            # like /api/set-arc can read live runtime state.
            try:
                self.http_server.set_app_ref(self)
            except Exception as e:
                logger.debug(f"app ref HTTP wire failed: {e}")
        except Exception as e:
            logger.error(f"HTTP server failed: {e}")

    def _tick(self):
        """10Hz: sync player state into AppState for iPad."""
        if not self.player:
            return
        self.state.update_playback(
            position=self.player.get_position(),
            duration=self.player.get_duration(),
            is_playing=self.player.is_playing(),
            file=self.player.current_file,
        )
        # Auto-set-arc check (every ~5s). Cheap: deque ops + a few
        # comparisons. Only does work if both set-arc AND auto mode
        # are on AND the detector exists.
        self._auto_set_arc_tick()
        # Mono LED blink animation. The _refresh_mk2_mono_leds path
        # uses time.time() to compute a 0.7Hz blink for the ALL button
        # when set-arc AUTO is on. We need to actually CALL the
        # refresh on each tick (every 100ms) so the blink animates.
        # Only when AUTO is on -- skip the USB write otherwise.
        if (getattr(self, "_set_arc_auto", False)
                and getattr(self, "_set_arc_enabled", False)
                and getattr(self, "mk2", None) is not None):
            try:
                self._refresh_mk2_mono_leds()
            except Exception:
                pass

    def _refresh_clips_for_ipad(self):
        """Push only clips for the CURRENT video. Pads remap per-video so
        each video has its own deck of cue points (deck-per-video model).

        Also republishes the global tag union so the iPad's tag picker
        always sees every tag the user has invented across the whole
        library (not just on the current video).

        AUDIT FIX (2026-05-16): was doing 3 full-table scans
        (`get_clips_for_file` + `get_all_tags` + `get_all_clips`) PLUS
        `_refresh_s2_leds` did a 4th. Now we fetch all_clips ONCE and
        compute the per-file slice + tag union + s2 LED count from it
        in-memory. One scan instead of four per load_video."""
        cur_file = self.player.current_file if self.player else ""
        # Single scan: pull everything, partition in-memory.
        try:
            all_clips = self.clips_db.get_all_clips() or []
        except Exception as e:
            logger.debug(f"get_all_clips failed in _refresh_clips_for_ipad: {e}")
            all_clips = []
        # Per-file slice for current video pads (use same _key
        # normalization clips_db.get_clips_for_file uses).
        if cur_file:
            try:
                cur_key = self.clips_db._key(cur_file)
                clips = [c for c in all_clips
                         if self.clips_db._key(c.get("filepath", "")) == cur_key]
            except Exception:
                clips = []
        else:
            clips = []
        self.state.set_clips(clips)
        # Tag union, built from the same all_clips snapshot.
        try:
            tags: set[str] = set()
            for c in all_clips:
                for t in (c.get("tags") or []):
                    if isinstance(t, str) and t:
                        tags.add(t)
            self.state.set_all_tags(sorted(tags))
        except Exception as e:
            logger.debug(f"set_all_tags failed: {e}")
        # Bank: global cross-video shortlist. Starred first (most recent),
        # then unstarred (also by last_played_ts desc). Cap at 24.
        try:
            starred = [c for c in all_clips if c.get("starred")]
            others = [c for c in all_clips if not c.get("starred")]
            starred.sort(key=lambda c: float(c.get("last_played_ts") or 0), reverse=True)
            others.sort(key=lambda c: float(c.get("last_played_ts") or 0), reverse=True)
            self.state.set_bank_clips((starred + others)[:24])
        except Exception as e:
            logger.debug(f"set_bank_clips failed: {e}")
        n = len(clips)
        self.state.set_armed_pads(list(range(min(n, 8))))
        # Pass the snapshot to avoid s2 doing its own get_all_clips call.
        self._refresh_s2_leds(all_clips_snapshot=all_clips)

    def _refresh_s2_leds(self, all_clips_snapshot=None):
        """Push current app state into S2 LEDs (armed pads + audio-reactive).

        AUDIT FIX (2026-05-16): accept an optional pre-fetched snapshot
        from _refresh_clips_for_ipad so we don't do a 2nd get_all_clips
        scan on the same hot path. Standalone callers (e.g. audio-reactive
        toggle) still get the safe default of fetching themselves."""
        if not self.s2:
            return
        try:
            if all_clips_snapshot is None:
                all_clips_snapshot = self.clips_db.get_all_clips() or []
            n_clips = len(all_clips_snapshot)
            a_armed = [i for i in range(min(n_clips, 4))]
            b_armed = [i - 4 for i in range(4, min(n_clips, 8))]
            self.s2.set_pads_armed("a", a_armed)
            self.s2.set_pads_armed("b", b_armed)
            self.s2.set_audio_reactive_indicator(self.state.audio_reactive_enabled)
        except Exception as e:
            logger.debug(f"S2 LED refresh failed: {e}")

    def _on_s2_connect(self):
        """S2 watchdog: device became reachable."""
        self.state.set_s2_connected(True)
        self._refresh_s2_leds()
        # Channel button LEDs are persistent state — repaint on every
        # connect so the active-channel indicator survives a reconnect.
        # Wrapped in getattr so partial-init failures don't crash here.
        if hasattr(self, "_refresh_channel_leds"):
            self._refresh_channel_leds()
        logger.info("S2 connect event")

    def _on_s2_disconnect(self):
        """S2 watchdog: device went away."""
        self.state.set_s2_connected(False)
        logger.info("S2 disconnect event")

    # Jog sensitivity modes: divisor → seconds per velocity tick.
    # Lower = bigger jumps. Touch defaults to coarse, release goes to base.
    _JOG_MODES = [
        ("gentle", 200.0),  # release default — fine nudging
        ("medium", 100.0),
        ("coarse", 70.0),   # touch override — moderate scrub
    ]
    _JOG_COARSE_IDX = 2  # index of "coarse" — used by touch override
    _jog_base_mode_idx: int = 0  # user's chosen baseline (set via FX1 Focus)
    _jog_touched: bool = False

    # Jog scrub coalescing — many fast events flooded mpv and starved
    # the audio buffer. We accumulate deltas and flush at ~16 Hz max.
    _jog_accum_seconds: float = 0.0
    # Created at class-definition time (before any thread exists) so the
    # jog reader thread + Qt flush thread can't race two separate locks
    # into existence. (Audit fix C6.)
    _jog_accum_lock = threading.Lock()
    _jog_flush_pending: bool = False

    # Frame-step accumulator for gentle mode (separate from seconds accumulator)
    _jog_accum_frames: int = 0

    def jog_scrub(self, velocity: int):
        """Scrub player by jog-wheel velocity.
        - GENTLE mode = frame-precise stepping (1 frame per ~30 velocity units).
        - MEDIUM/COARSE = keyframe-relative seek by seconds.
        Coalesces rapid ticks (jog emits ~50Hz) into a flush every ~60ms."""
        if not self.player or velocity == 0:
            return
        name, divisor = self._JOG_MODES[self._jog_mode_idx]
        with self._jog_accum_lock:
            if name == "gentle":
                # Gentle = frame-step. ~30 vel units per frame so a slow
                # spin steps a frame at a time, fast spin steps several.
                # Sign preserved.
                step = int(velocity / 30)
                if step == 0:
                    step = 1 if velocity > 0 else -1
                self._jog_accum_frames += step
            else:
                self._jog_accum_seconds += velocity / divisor
            if self._jog_flush_pending:
                return
            self._jog_flush_pending = True
        QTimer.singleShot(60, self._jog_flush)

    def _jog_flush(self):
        with self._jog_accum_lock:
            delta_sec = self._jog_accum_seconds
            delta_frames = self._jog_accum_frames
            self._jog_accum_seconds = 0.0
            self._jog_accum_frames = 0
            self._jog_flush_pending = False
        if not self.player:
            return
        if delta_frames != 0:
            forward = delta_frames > 0
            # Cap to avoid spamming mpv; ~12 frames per flush is plenty
            n = min(12, abs(delta_frames))
            for _ in range(n):
                self.player.frame_step(forward=forward)
        elif abs(delta_sec) >= 0.01:
            self.player.seek_relative(delta_sec, exact=False)

    def cycle_jog_sensitivity(self):
        """FX1 Focus: cycle the BASE jog sensitivity (gentle/medium/coarse).
        Touching the jog wheel temporarily forces coarse; releasing returns
        to whatever the user picked here."""
        self._jog_base_mode_idx = (self._jog_base_mode_idx + 1) % len(self._JOG_MODES)
        # Only update the active mode if the wheel isn't currently being
        # touched (touch override wins while held).
        if not self._jog_touched:
            self._jog_mode_idx = self._jog_base_mode_idx
        name, _ = self._JOG_MODES[self._jog_base_mode_idx]
        self.state.set_message(f"Jog base: {name.upper()}")
        self.state.set_jog_mode(name)
        if self.s2:
            self.s2.flash_led("fx1_focus")

    def _jog_touch_changed(self, value: int):
        """Top platter capacitive sensor. Touch=coarse, release=base mode.

        Optional "grab-the-frame" mode (`s2_jog_scrub_grab` config flag,
        default False): touching the platter ALSO pauses LIVE, releasing
        resumes. Combined with the existing coarse-on-touch scrub mode,
        this gives a tactile "freeze and scrub" feel without needing
        any new button — same gesture the operator would naturally
        make to scrub. Opt-in because some users prefer the platter to
        scrub without affecting playback."""
        touched = bool(value)
        if touched == self._jog_touched:
            return
        self._jog_touched = touched
        # Track when this touch event arrived. If the capacitive sensor
        # sticks at "touched" (moisture, firmware glitch, missed release
        # packet at startup), the auto-flip gate would silently block
        # all flips forever. Watchdog in _on_beat auto-clears stuck
        # touches after _JOG_TOUCH_STUCK_TIMEOUT_S seconds.
        self._jog_touched_at = time.time() if touched else 0.0
        if touched:
            self._jog_mode_idx = self._JOG_COARSE_IDX
        else:
            self._jog_mode_idx = self._jog_base_mode_idx
        name, _ = self._JOG_MODES[self._jog_mode_idx]
        self.state.set_jog_mode(name)
        # Optional grab-the-frame: pause LIVE on touch, resume on
        # release. Gated behind config so it's opt-in (reversible).
        if self.config.get("s2_jog_scrub_grab", False) and self.player:
            try:
                if touched:
                    # Remember whether we were playing -- only resume
                    # if the user wasn't already paused.
                    self._jog_grab_was_playing = bool(
                        self.player.is_playing())
                    if self._jog_grab_was_playing:
                        self.player.pause()
                else:
                    if getattr(self, "_jog_grab_was_playing", False):
                        self.player.play()
                    self._jog_grab_was_playing = False
            except Exception as e:
                logger.debug(f"jog-grab pause/play failed: {e}")

    # If the touch sensor reports "touched" continuously for longer than
    # this without a release event, _on_beat will auto-clear the state.
    # 3 seconds = longer than any plausible deliberate jog touch.
    _JOG_TOUCH_STUCK_TIMEOUT_S: float = 3.0
    _jog_touched_at: float = 0.0

    def restart_video(self):
        """A CUE / B CUE (headphone-cue button, a_pfl in HID): jump to start
        of current file. Also clears any A-B loop so the video plays through
        normally instead of ping-ponging."""
        if self.player:
            for prop in ("ab-loop-a", "ab-loop-b"):
                try:
                    self.player.player.command("set", prop, "no")
                except Exception:
                    pass
            self.player.seek(0)

    def frame_step(self, forward: bool = True):
        """SHIFT + Loop In/Out: nudge one frame for precise cue-point placement."""
        if self.player:
            self.player.frame_step(forward=forward)

    def flip_back(self):
        """B CUE / MK2 < : go BACK to the most-recently-replaced clip,
        landing where we left off (Stage 2, 2026-05-16).

        Two paths:
          1. History stack non-empty → pop (path, position), preload
             mpv.start so the new clip lands at that position with no
             title-frame flash, load it. This is the user's "fire
             happened but the previous video was good, jump back to it
             in nearly the same spot" flow.
          2. History empty (just-launched / first clip) → fall back to
             pool sequential prev (active bank / scratch back-step).

        Sequential only (random mode applies forward only — < should
        feel like a real step-back, not another roll)."""
        if not self.player or not self.player.current_file:
            return
        # Path 1 — play-history pop with position memory.
        if self._flip_history:
            path, pos = self._flip_history.pop()
            logger.info(f"[flip_back] history pop → {Path(path).name} @ "
                        f"{pos:.1f}s (stack={len(self._flip_history)})")
            # Pre-load seek: set mpv.start to the remembered position so
            # the first decoded frame IS where we left off. 500ms auto-
            # reset so unrelated loads later aren't poisoned. Same trick
            # as _apply_preload_body_seek, but with our remembered pos
            # instead of the body-seek default.
            try:
                if pos and pos > 0.5:
                    self.player.player.start = float(pos)
                    QTimer.singleShot(500, self._reset_mpv_start_option)
            except Exception as e:
                logger.debug(f"flip_back start-preset failed: {e}")
            # The in-progress flag stops load_video from re-pushing the
            # about-to-be-replaced clip back onto the stack (oscillation).
            self._flip_back_in_progress = True
            try:
                # CRITICAL FIX (code-review agent, 2026-05-16 night):
                # force=True bypasses the 300ms same-path debounce. Without
                # force, mash-fire scenarios can pop a path identical to
                # what's currently playing → debounce returns False → no
                # play() → next < pop pushes the current path back onto
                # history via load_video's push-hook → stack corruption.
                if self.load_video(path, force=True):
                    self.player.play()
            finally:
                self._flip_back_in_progress = False
            self.state.record_s2_action("flip_back (history)")
            return
        # Path 2 — empty history, fall back to pool sequential prev.
        prv = self._pick_in_pool(direction=-1)
        if not prv:
            logger.info("[flip_back] empty history AND empty pool")
            return
        cur = Path(self._current_source_path or self.player.current_file)
        same = (str(prv) == str(cur))
        logger.info(f"[flip_back] history empty → pool prev "
                    f"({Path(prv).name}) "
                    f"{'(SAME FILE — pool has only 1 video!)' if same else ''}")
        # Body-seek (not position memory — we don't have one for this path).
        self._apply_preload_body_seek(str(prv))
        if self.load_video(str(prv)):
            self.player.play()
        self.state.record_s2_action("flip_back (pool)")

    def _update_downbeat_phase(self) -> None:
        """Auto-detect which beat-of-4 is the musical downbeat.

        Buckets the most recent beat's kick flux by
        (beat_count mod 4); the phase with the highest average kick
        energy is the "1". Self-correcting — if the music shifts the
        running average tracks it. Called once per beat from
        _on_beat. Cheap (4 deque appends + 4 means)."""
        try:
            flux = float(getattr(
                self.audio_reactive, "last_beat_flux", 0.0) or 0.0)
        except Exception:
            flux = 0.0
        phase = self._beat_diag_count % 4
        self._downbeat_energy[phase].append(flux)
        # Need a little history before the detection is trustworthy.
        total = sum(len(b) for b in self._downbeat_energy)
        if total < 16:
            return
        means = [
            (sum(b) / len(b)) if b else 0.0
            for b in self._downbeat_energy
        ]
        self._downbeat_phase = max(range(4), key=lambda i: means[i])

    def _on_beat(self):
        """Called from audio-reactive thread on detected beat.

        Always records the beat (so the iPad pulses) but only flips when
        audio-reactive is enabled AND the per-flip debounce has elapsed.
        Without that gap, flips fire on every snare/hat — unwatchable.
        """
        self.state.record_beat()
        # Throttled inside session_log (1 in 32 to disk) — safe to always call.
        try:
            self.session_log.record_beat(bpm=float(self.state.detected_bpm or 0.0))
        except Exception:
            pass
        try:
            self._analytics_today["beats"] = int(self._analytics_today.get("beats", 0)) + 1
        except Exception:
            pass
        # OSC beat broadcast (no-op when disabled).
        try:
            if self.osc:
                self._osc_beat_counter += 1
                self.osc.send_beat(self._osc_beat_counter, float(self.state.detected_bpm or 0.0))
        except Exception:
            pass
        # Beat-pulse the video brightness — subtle "punch" felt on each
        # kick. Always-on while audio-reactive runs; rides on top of
        # the user's brightness knob (B EQ HI). Marshaled to Qt thread.
        QTimer.singleShot(0, self._beat_pulse_brightness)
        # Per-beat diagnostic counter — log every 8 beats so we can SEE
        # in the console whether beats are being detected at all. Without
        # this, "auto-flip not working" could be any of: no audio input,
        # beat detector silent, flip gated, flip dispatched but failing.
        # The counter narrows it down.
        if not hasattr(self, "_beat_diag_count"):
            self._beat_diag_count = 0
            self._beat_diag_flips = 0
        self._beat_diag_count += 1
        if self._beat_diag_count % 8 == 0:
            logger.info(f"[autoflip] beats={self._beat_diag_count} "
                        f"flips={self._beat_diag_flips} "
                        f"bpm={self.state.detected_bpm or 0:.0f} "
                        f"audio_reactive={self.state.audio_reactive_enabled} "
                        f"flip_on_beat={self.state.flip_on_beat} "
                        f"jog_touched={self._jog_touched}")
        if not self.state.audio_reactive_enabled:
            return
        # BPM lock keeps running but flips can be silenced independently.
        if not self.state.flip_on_beat:
            return
        # HERO-HOLD on drop: when DropDetector spiked recently, we sit
        # on the current clip through the peak instead of cutting more.
        # Set-craft from the PDF — pros LOCK on one hero visual at the
        # drop; rookies machine-gun more flips. Self-clears when the
        # timer expires (no separate teardown path needed).
        if time.time() < self._auto_flip_suppressed_until:
            remaining = self._auto_flip_suppressed_until - time.time()
            # Quietly log every few beats so the console shows the hold
            # is doing something — but not on every beat (would spam).
            if self._beat_diag_count % 4 == 0:
                logger.info(
                    f"[hero-hold] flip suppressed, "
                    f"{remaining:.1f}s remaining"
                )
            return
        # HOLD: temporarily pin the current clip. Auto-flip silently
        # waits. Auto-release after timeout so a forgotten hold can't
        # silently trap the visuals (user can also tap GRID to release).
        if self._hold_clip:
            if time.time() >= self._hold_clip_until:
                self._hold_clip = False
                logger.info("[hold-clip] auto-released after timeout")
            else:
                return  # held — skip flip
        # Don't auto-flip while the user is hand-scrubbing the jog wheel.
        # Touching the jog is an explicit "I'm driving manually" signal —
        # an auto-flip mid-scrub reloads the file and _auto_seek_into_body
        # yanks their position, which reads as the video "flashing" and
        # fighting the jog. Auto-flip resumes the moment they let go.
        # SAFETY: capacitive touch sensors can stick (moisture, missed
        # release packet, startup state). If touched > N seconds, treat
        # as released so auto-flip recovers instead of going silently dead.
        if self._jog_touched:
            stuck_for = time.time() - (self._jog_touched_at or 0.0)
            if stuck_for > self._JOG_TOUCH_STUCK_TIMEOUT_S:
                logger.warning(f"[autoflip] auto-clearing stuck jog_touched "
                               f"(stuck for {stuck_for:.1f}s) — capacitive "
                               f"sensor must have missed a release packet")
                self._jog_touched = False
                self._jog_touched_at = 0.0
            else:
                return
        now = time.time()
        # Beat-aware debounce: if BPM is known, wait `_flip_beats` between
        # flips (configurable via the encoder). HARD-FLOORED at
        # _min_flip_interval (4s) — beat-awareness can only make flips
        # SLOWER than the cap, never faster, so a high BPM or a low
        # beat-count can never strobe the output. (Audit fix N1.)
        bpm = float(self.state.detected_bpm or 0)
        if 60.0 <= bpm <= 200.0:
            interval = max(self._min_flip_interval,
                           (60.0 / bpm) * float(self._flip_beats))
        else:
            interval = self._min_flip_interval
        # ── PHRASE-BASED CUT DENSITY (2026-05-19) ─────────────────
        # The flat `interval` debounce is a metronome — cuts at rigid
        # uniform spacing, which reads as an unthinking machine.
        # Phrase-cut keeps the SAME rate (a cut becomes *eligible* at
        # the same `interval`) but then QUANTIZES the flip onto the
        # auto-detected musical downbeat: once eligible it waits for
        # the "1" — strongly preferring it, occasionally taking an
        # off-beat for variety, and force-flipping if it drifts too
        # long. Net: same average rate, but cuts land musically.
        phrase_cut = bool(
            self.config.get("picker_phrase_cut_enabled", True)
        )
        flip_reason = f"interval={interval:.1f}s"
        if phrase_cut and bpm > 0:
            elapsed = now - self._last_flip_beat_time
            # Pull eligibility in by ~half a bar so the downbeat-wait
            # below nets back out to roughly the legacy average rate
            # ("same rate, redistributed"). Never below the 4s floor.
            eligible_at = max(self._min_flip_interval,
                              interval - 2.0 * (60.0 / bpm))
            if elapsed < eligible_at:
                return                       # not eligible yet
            self._update_downbeat_phase()
            bar_pos = (self._beat_diag_count - self._downbeat_phase) % 4
            # Per-bar-position take-probability once eligible: the
            # downbeat is near-certain; other positions occasional so
            # the edit isn't rigidly "every cut on the 1" either.
            bar_weight = (0.92, 0.12, 0.45, 0.22)
            # Anti-drift: past 2x the interval, take whatever beat.
            if elapsed < interval * 2.0:
                import random as _rnd
                if _rnd.random() >= bar_weight[bar_pos]:
                    return                   # wait for a better beat
            flip_reason = f"phrase bar_pos={bar_pos}"
        else:
            # Legacy flat-interval behaviour (phrase-cut disabled).
            if now - self._last_flip_beat_time < interval:
                return
        self._last_flip_beat_time = now
        self._beat_diag_flips += 1
        logger.info(f"[autoflip] FIRING flip #{self._beat_diag_flips} "
                    f"(mode={'FOLDER' if self._auto_flip_use_folder else 'BANK'}, "
                    f"{flip_reason})")
        # Bounce the flip onto the Qt thread; load_video / mpv calls
        # are not safe to invoke directly from the audio capture thread.
        # auto_flip seeks past the intro after the load lands.
        QTimer.singleShot(0, self.auto_flip)

    def _on_bpm(self, bpm: float):
        """Called from audio-reactive thread when the BPM estimate updates."""
        self.state.set_detected_bpm(bpm)

    def _library_pivot_to_working_set(self) -> dict:
        """iPad button: swap the library browser to the configured
        working_set_folder (so its files become loadable). Persists the
        prior root to settings so "← Library" works across restarts."""
        ws = (self.config.get("working_set_folder") or "").strip()
        if not ws:
            self.state.set_message("No working set folder configured", error=True)
            return {"ok": False, "error": "working_set_folder not set"}
        if not Path(ws).is_dir():
            self.state.set_message(f"Working set folder missing: {ws}", error=True)
            return {"ok": False, "error": "folder not found"}
        # Save the current root persistently so "← Library" can restore
        # it after a restart. Don't overwrite if we're already pivoted
        # (cur_root == ws) — that would lose the original.
        cur_root = (self.state.library_root or "").strip()
        if cur_root and cur_root != ws:
            self.config["previous_library_root"] = cur_root
            self._save_config()
        return self.library_scan(ws)

    def _library_pivot_to_default_root(self) -> dict:
        """iPad button: undo the working-set pivot — restore the library
        root that was active before. Persistent across restarts (reads
        from config). Falls back to the project's DEFAULT_LIBRARY if
        the saved prior root is missing or doesn't exist."""
        target = (self.config.get("previous_library_root") or "").strip()
        if not target or not Path(target).is_dir():
            target = DEFAULT_LIBRARY if Path(DEFAULT_LIBRARY).is_dir() else ""
        if not target:
            return {"ok": False, "error": "no prior root to restore"}
        result = self.library_scan(target)
        # Clear the saved prior root after restoring — so a subsequent
        # working-set pivot will capture the (now-current) root fresh.
        self.config.pop("previous_library_root", None)
        self._save_config()
        return result

    # Queue (not single-slot) so back-to-back pad strikes from the MK2
    # reader thread can't overwrite each other before the Qt loop drains
    # them. Each QTimer.singleShot(0, _fire_pending_mk2) drains exactly
    # one item. (Audit fix C1.)
    _mk2_fire_queue: "queue.Queue" = None  # set in __init__-time guard below

    def _on_mk2_pad_press(self, pad_no: int, velocity: int):
        """Maschine MK2 pad strike → fire the Nth scratch entry to LIVE.
        Pad VELOCITY (0..4095, 12-bit pressure) shapes the body-seek
        landing point. Enqueued (not single-slot stashed) so rapid
        strikes never clobber each other; drained one-per-singleShot."""
        files = self.scratch_store.all()
        idx = pad_no - 1
        logger.info(
            f"MK2 pad {pad_no} v={velocity} → scratch idx {idx} "
            f"(basket has {len(files)} entries)"
        )
        if 0 <= idx < len(files):
            if self._mk2_fire_queue is None:
                self._mk2_fire_queue = queue.Queue()
            self._mk2_fire_queue.put((files[idx], velocity))
            logger.info(f"MK2 pad {pad_no} firing: {Path(files[idx]).name}")
            QTimer.singleShot(0, self._fire_pending_mk2)
            # Press flash: blast the struck pad to full brightness.
            self._flash_mk2_pad(pad_no)
        else:
            logger.info(f"MK2 pad {pad_no}: nothing at scratch idx {idx}")

    def _flash_mk2_pad(self, pad_no: int):
        """Briefly light the struck pad bright orange so the user sees
        the hardware acknowledge the hit. Called from MK2 thread —
        the LED write is safely thread-handled inside MaschineMK2."""
        if not self.mk2:
            return
        try:
            # We can't read the current colors back; rebuild the full
            # map with this pad overridden.
            files = self.scratch_store.all()
            live_path = (self._current_source_path
                         or (self.player.current_file if self.player else None) or "")
            colors = {}
            for i, path in enumerate(files[:16]):
                label = i + 1
                if label == pad_no:
                    colors[label] = (255, 140, 0)  # warm flash
                elif path == live_path:
                    colors[label] = (255, 255, 255)
                else:
                    colors[label] = (0, 70, 90)
            self.mk2.set_pad_colors_by_label(colors)
        except Exception as e:
            logger.debug(f"_flash_mk2_pad failed: {e}")

    def _fire_pending_mk2(self):
        if self._mk2_fire_queue is None:
            return
        try:
            target, velocity = self._mk2_fire_queue.get_nowait()
        except queue.Empty:
            return
        if not target:
            return
        try:
            self.session_log.record_fire(source="mk2_pad", filepath=target, velocity=velocity)
        except Exception:
            pass
        self._record_fire_history(target)
        # Compute body-seek position BEFORE the load. Velocity from the
        # pad strike shapes WHERE in the playable region we land — soft
        # → early body, hard → deep body. mpv start option means the
        # first decoded frame IS the target — no title flash.
        seek_to = self._compute_body_seek_target(target, velocity=velocity)
        try:
            if self.player and self.player.player and seek_to > 0.5:
                self.player.player.start = float(seek_to)
        except Exception as e:
            logger.debug(f"mpv start property set failed: {e}")
        self._fire_scratch_path(target)
        # Reset start option so unrelated subsequent loads don't inherit it.
        QTimer.singleShot(500, self._reset_mpv_start_option)

    def _reset_mpv_start_option(self):
        try:
            if self.player and self.player.player:
                self.player.player.start = "none"
        except Exception:
            pass

    def _compute_body_seek_target(self, path: str, velocity: int = 2000) -> float:
        """Pick a body-seek time for a file before it loads.

        `velocity` is the pad-strike pressure (0..4095, 12-bit) that
        shaped this fire. It biases where in the playable region we
        land: soft strike → earlier in body; hard strike → deeper.
        Within the velocity-biased band there's still random jitter
        plus an "avoid recent visit" check.

        Default velocity 2000 ≈ medium hit ≈ middle of the body."""
        import random as _random
        intro_skip = 15.0
        outro_skip = 20.0
        try:
            duration = float(self._probe_duration(path))
        except Exception:
            duration = 0.0
        if duration <= 0:
            return intro_skip
        if duration < intro_skip + outro_skip + 5.0:
            if duration > intro_skip + 2.0:
                return intro_skip
            return max(0.0, duration * 0.2)
        playable_start = intro_skip
        playable_end = duration - outro_skip
        playable_dur = playable_end - playable_start

        # Velocity → center position in the playable region.
        # v=0    → 0.25 of playable (early body)
        # v=2000 → 0.50 of playable (middle, the previous default behavior)
        # v=4095 → 0.80 of playable (deep body, near outro_skip boundary)
        v = max(0, min(4095, int(velocity or 2000)))
        v_norm = v / 4095.0
        center_frac = 0.25 + v_norm * 0.55
        center = playable_start + center_frac * playable_dur
        # Jitter window: ±20% of playable around the velocity-biased
        # center. Clamped to the playable bounds.
        jitter = 0.20 * playable_dur
        lo = max(playable_start, center - jitter)
        hi = min(playable_end, center + jitter)
        if hi <= lo:
            hi = lo + 0.1

        # "Don't land where we just landed" filter (relaxed from 1/4 →
        # 1/5 of playable since velocity already adds spread).
        min_distance = playable_dur / 5.0
        prev = self._last_auto_seek_per_file.get(path, -999.0)
        target = _random.uniform(lo, hi)
        for _ in range(5):
            if abs(target - prev) >= min_distance:
                break
            target = _random.uniform(lo, hi)
        self._remember_auto_seek(path, target)
        return target

    def _on_mk2_pad_release(self, pad_no: int):
        """Restore the pad's normal color after a press flash."""
        # Refresh the whole map — cheap, and ensures LIVE highlighting
        # stays correct if the press fired a load.
        QTimer.singleShot(150, self._refresh_mk2_pad_leds)

    def _on_mk2_encoder_delta(self, encoder_idx: int, delta: int):
        """MK2 encoder rotation callback. encoder_idx 0 = master encoder
        (discovered 2026-05-17 night: byte[8] low nibble, 4-bit counter).

        Master encoder rotation steps stutter division: CW (positive
        delta) toward smaller divisions (faster stutter / build /
        riser); CCW (negative) toward bigger (slower / calmer)."""
        if encoder_idx == 0:
            # Master encoder → stutter division step. Each detent =
            # one step in the preset ladder.
            try:
                self.stutter_division_step(delta)
            except Exception as e:
                logger.debug(f"stutter_division_step from encoder failed: {e}")
        # Other encoder indices not wired yet.

    # ─── MK2 vertical-page state (2026-05-17) ────────────────────────
    # Top row 0-7 selects which of 8 "vertical pages" is active. Each
    # page is a list of up to 7 folder paths (the 8th vertical stays
    # REROLL always). When a vertical bit 40-46 fires, we resolve its
    # folder from the active page rather than from the static
    # `mk2_button_map`. `<` (bit 12) / `>` (bit 13) cycle pages.
    # See VJ_ROADMAP.md / MK2_BUTTON_MAP.md.
    _mk2_active_page: int = 0

    def _on_mk2_button_press(self, bit: int):
        """MK2 non-pad button strike (Group A-H, transport, etc.). The
        bit index identifies the physical button. Config map lets the
        user (or a discovery session) bind bits to actions. Default
        map empty → strikes are logged in MK2 log lines but do nothing."""
        # ─── Page-selector handling (new bits 0-7, 12, 13) ──────────
        # Only active if user has configured `mk2_vertical_pages` in
        # settings.json; otherwise falls through to the static action
        # map (no behavior change for users who haven't opted in).
        pages = self.config.get("mk2_vertical_pages") or []
        if pages:
            if 0 <= bit <= 7 and bit < len(pages):
                # Top-row direct page selector (bit N -> page N)
                self._set_mk2_page(bit)
                return
            if bit == 12:
                self._set_mk2_page((self._mk2_active_page - 1) % len(pages))
                return
            if bit == 13:
                self._set_mk2_page((self._mk2_active_page + 1) % len(pages))
                return
            if 40 <= bit <= 46:
                # Vertical bits 1-7 resolve their folder from active page
                slot = bit - 40
                page = pages[self._mk2_active_page] if (
                    self._mk2_active_page < len(pages)) else []
                folder = page[slot] if slot < len(page) else None
                if folder:
                    action = f"folder:{folder}"
                    self._mk2_last_vertical = (slot, folder)
                    logger.info(
                        f"MK2 vertical slot {slot+1} (bit {bit}) on "
                        f"page {self._mk2_active_page+1} -> {folder}"
                    )
                    if self._mk2_button_queue is None:
                        self._mk2_button_queue = queue.Queue()
                    self._mk2_button_queue.put(action)
                    QTimer.singleShot(0, self._fire_pending_mk2_button)
                    return
                # No folder configured for this slot on this page -> noop
                logger.info(f"MK2 vertical slot {slot+1} empty on page "
                            f"{self._mk2_active_page+1}")
                return

        # ─── Direct-handler bits (newly bound, 2026-05-17) ──────────
        # These bypass the action-map entirely. Hardware-bound, not
        # user-configurable -- they are essential function buttons
        # whose binding matches NI's silkscreen intent.
        if bit == 9:   # STEP -> cycle bank layer (default/positions/pov/mood)
            # Rebound 2026-05-18 from pin_file_toggle. STEP semantically
            # matches "step through layers". pin_file_toggle is still
            # callable via HTTP if anyone needs it.
            self.cycle_bank_layer()
            return
        if bit == 10:  # BROWSE -> toggle MK2 OLED browse mode
            self.mk2_browse_toggle()
            return
        if bit == 13:  # `>` page-next arrow -> bank current folder to A-H
            self.mk2_bank_current_folder()
            return
        if bit == 14:  # ALL -> hold-detect: short tap = cycle, hold = auto toggle
            # Record press timestamp; defer action to release so we
            # can distinguish tap from hold. Hold threshold is 800ms.
            if not hasattr(self, "_mk2_hold_press_at"):
                self._mk2_hold_press_at = {}
            self._mk2_hold_press_at[14] = time.time()
            logger.info("[mk2/all-press] timestamp recorded, "
                        "awaiting release for tap-vs-hold dispatch")
            return
        if bit == 15:  # AUTO WR
            self.toggle_flip_on_beat()
            return
        # GROUP buttons (bits 24-31, letters A-H) — tap = bank load,
        # HOLD (>=800ms) while a clip plays = reclassify that clip to
        # this category. Fuzzy / cumulative so a single mis-press
        # doesn't lock the wrong bucket. Press just records the
        # timestamp; release does the tap-vs-hold dispatch.
        if 24 <= bit <= 31:
            if not hasattr(self, "_mk2_hold_press_at"):
                self._mk2_hold_press_at = {}
            self._mk2_hold_press_at[bit] = time.time()
            return

        # ─── Existing action-map dispatch ───────────────────────────
        action_map = (self.config.get("mk2_button_map") or {})
        action = action_map.get(str(bit))
        logger.info(f"MK2 button bit {bit} → action={action!r}")
        if not action:
            return
        # HOLD-style actions bypass the queue: they need precise press/
        # release pairing, and queueing would let the press get held up
        # behind other actions while the release fires immediately.
        if action == "audio_leak_hold":
            self.audio_leak_press()
            return
        # Enqueue (not single-slot) so rapid button presses don't clobber.
        if self._mk2_button_queue is None:
            self._mk2_button_queue = queue.Queue()
        self._mk2_button_queue.put(action)
        QTimer.singleShot(0, self._fire_pending_mk2_button)

    _mk2_last_vertical = None  # (slot_idx, folder_path) — for OLED later

    def _set_mk2_page(self, page_idx: int) -> None:
        """Switch the active MK2 vertical-page. Loads that page's
        folder labels into the on-app status (visible via banner +
        future OLED render). Bit 47 stays as REROLL regardless of page.
        """
        pages = self.config.get("mk2_vertical_pages") or []
        if not pages or page_idx < 0 or page_idx >= len(pages):
            return
        self._mk2_active_page = page_idx
        page = pages[page_idx]
        name = self._mk2_page_name(page_idx)
        # Status banner — visible in iPad strip + main window
        try:
            self.state.set_message(f"MK2 PAGE {page_idx+1}: {name}")
        except Exception:
            pass
        logger.info(
            f"[mk2-page] switched to page {page_idx+1}: {name}  "
            f"(verticals: {page})"
        )
        # Repaint the right OLED + top-row LEDs NOW so the new page is
        # visible immediately. Without this the page change is invisible
        # until some unrelated event (a clip flip, a bank load) happens
        # to refresh the OLED — which is exactly why page-select felt
        # intermittent: "did it switch? press a vertical and find out."
        # Marshalled to the Qt thread (button press is on the HID
        # thread; OLED writes must not race a chunked frame). (2026-05-20)
        try:
            QTimer.singleShot(0, self._refresh_mk2_pad_leds)
        except Exception:
            pass

    def _mk2_page_name(self, page_idx: int) -> str:
        """Pretty name for a page. If the page is a list of folders,
        derive a name from the first folder's tail or use config-named
        pages if user supplied `mk2_vertical_page_names`."""
        names = self.config.get("mk2_vertical_page_names") or []
        if page_idx < len(names) and names[page_idx]:
            return str(names[page_idx])
        pages = self.config.get("mk2_vertical_pages") or []
        if page_idx < len(pages) and pages[page_idx]:
            tails = []
            for f in pages[page_idx][:3]:
                if f:
                    tails.append(Path(f).name)
            return " · ".join(tails) if tails else f"page {page_idx+1}"
        return f"page {page_idx+1}"

    # ─── Set-arc mode state + cycle (2026-05-17) ───────────────────
    # OPENING -> BUILD -> PEAK -> BREAKDOWN -> OPENING cycle. State
    # persisted to settings so restart restores last phase. Enabled
    # flag is independent so you can be in BREAKDOWN but with arc
    # mode OFF and the picker doesn't apply any phase boost. See
    # set_arc.py for the per-phase tag profiles.
    _set_arc_phase: str = "opening"
    _set_arc_enabled: bool = False

    # ─── MK2 browse mode (BROWSE button, bit 10) ───────────────────
    # Toggles the right OLED into a scrolling file-list view tied to
    # the existing S2 browse cursor (state.library_files +
    # library_selected_idx). The S2 turntable encoder already drives
    # the cursor; this just visualizes it on the MK2 OLED. When OFF,
    # right OLED reverts to the page/pad layout. Not persisted -- a
    # restart leaves you in non-browse mode.
    _mk2_browse_mode: bool = False

    # Cap on how deep / how wide a recursive bank scan can go.
    _BANK_RECURSIVE_MAX_FILES = 500
    _BANK_RECURSIVE_MAX_DEPTH = 6

    @staticmethod
    def _gather_videos_recursive(
        folder, max_files: int, max_depth: int
    ) -> list[str]:
        """BFS scan for video files under `folder`. Caps total files
        and recursion depth so a bank operation on the library root
        doesn't try to enumerate 50k clips. Returns absolute path
        strings."""
        from collections import deque
        from pathlib import Path as _P
        out: list[str] = []
        # (path, depth)
        queue = deque([(_P(folder), 0)])
        while queue and len(out) < max_files:
            cur, depth = queue.popleft()
            try:
                for entry in cur.iterdir():
                    try:
                        if entry.is_file():
                            if entry.suffix.lower() in VIDEO_EXTS:
                                out.append(str(entry))
                                if len(out) >= max_files:
                                    return out
                        elif entry.is_dir() and depth < max_depth:
                            queue.append((entry, depth + 1))
                    except OSError:
                        continue
            except OSError:
                continue
        return out

    def bank_from_folder(self, folder_rel: str = "",
                         recursive: bool = True) -> dict:
        """Auto-split a folder's videos across banks A-H. Recursive
        by default so banking a parent folder like 'Lil Humpers' (which
        only has subfolders, no direct videos) walks ALL the year
        subfolders and gathers their clips.

        Args:
          folder_rel: folder path RELATIVE TO library_root. Empty
                      string = use current library_folder.
          recursive: if True (default), walks subfolders up to
                     _BANK_RECURSIVE_MAX_DEPTH. If False, only files
                     directly in the folder.

        This is the unified backend for the iPad per-folder BANK
        button (the ⊞ icon on each library row) and the MK2 `>`
        arrow shortcut. Both call this path now."""
        from pathlib import Path as _P
        snap = self.state.get_library_snapshot()
        root_str = snap.get("root") or ""
        if not root_str:
            return {"ok": False, "error": "no library root"}
        root = _P(root_str)
        if folder_rel:
            # iPad sends a name relative to current folder. Allow
            # bare name (= subfolder of current) OR a relative path
            # under root.
            cur_folder = _P(snap.get("folder") or root_str)
            candidate = cur_folder / folder_rel
            if not candidate.is_dir():
                # Try as relative-to-root.
                candidate = root / folder_rel
            if not candidate.is_dir():
                return {"ok": False,
                        "error": f"folder not found: {folder_rel}"}
            # folder_rel arrives from an HTTP client — a "../.." or an
            # absolute path could otherwise escape the library root.
            # Mirror the boundary check library_cd / library_load_file
            # already use; never scan outside root.
            resolved = self._resolve_inside_root(root, candidate)
            if resolved is None:
                return {"ok": False,
                        "error": "folder is outside the library root"}
            folder = resolved
        else:
            folder = _P(snap.get("folder") or root_str)
        if not folder.is_dir():
            return {"ok": False, "error": "not a dir"}
        # Recursive (or shallow) walk for video files.
        try:
            if recursive:
                files = self._gather_videos_recursive(
                    folder,
                    self._BANK_RECURSIVE_MAX_FILES,
                    self._BANK_RECURSIVE_MAX_DEPTH,
                )
            else:
                files = [
                    str(p) for p in folder.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in VIDEO_EXTS
                ]
        except OSError as e:
            return {"ok": False, "error": f"scan failed: {e}"}
        if not files:
            return {"ok": False,
                    "error": f"no videos under {folder.name}"}
        # Hand off to the same split+pad+save logic as tag_to_banks.
        # Pass the resolved folder path so banks get folder-locked to
        # this source — reroll then re-pulls FRESH files from same
        # folder ("stay in the vibe I just picked").
        return self._split_files_into_banks(
            files, short_name=folder.name[:24] or "auto",
            source_folder=str(folder).replace("\\", "/"),
        )

    # Bank category map: A-H are visual-attribute buckets keyed to the
    # axes the offline taggers produce (colour, motion/energy, geometry,
    # symmetry). They are defaults only -- the operator relabels and
    # retags them to taste; see _BUILTIN_BANK_LAYERS for alternates.
    # Each file is scored against every category (count of tag matches,
    # IDF-weighted so rare tags win); highest-scoring bucket wins.
    # Leftovers (no matches) are distributed to under-populated banks.
    # Override via config["bank_categories"] (a one-off inline layer) or
    # config["bank_layers"] (named layers, cycled with the layer button).
    #
    # Per-letter FOCUS-tag overlay on top of BANK_THEMES. BANK_THEMES
    # carries the base names + a small generic tag list; this overlay
    # adds finer-grained tags so the categorical scorer can actually
    # discriminate between buckets.
    _BANK_FOCUS_TAGS: dict = {
        "A": ["color:hue:red", "color:hue:orange", "color:hue:yellow"],
        "B": ["color:hue:blue", "color:hue:cyan", "color:hue:teal"],
        "C": ["complexity:8", "complexity:9", "intense"],
        "D": ["complexity:0", "complexity:1", "still", "calm"],
        "E": ["geometry:linear", "geometry:polygons", "edges", "angular"],
        "F": ["geometry:particles", "geometry:masks", "organic", "grain"],
        "G": ["symmetry:cohesive", "centered", "mirror"],
        "H": ["symmetry:offset", "asymmetric", "off-center"],
    }
    # _DEFAULT_BANK_CATEGORIES is BUILT IN __init__ from BANK_THEMES
    # (canonical letter->name source, lives further down the class) +
    # _BANK_FOCUS_TAGS above, so the theme system and the categorical
    # scorer share one source of truth.
    # Each entry: {"name": "Bank A", "tags": [...BANK_THEMES.tags..., ...focus...]}
    _DEFAULT_BANK_CATEGORIES: dict = {}

    # -- LAYER OVERLAYS -----------------------------------------------
    # Multiple named A-H category maps the operator can cycle between
    # on the fly. The "default" layer is _DEFAULT_BANK_CATEGORIES (built
    # from BANK_THEMES + _BANK_FOCUS_TAGS in __init__). The alternates
    # below are FRESH category dicts keyed to different visual axes.
    #
    # Layer config shape (settings.json):
    #   "bank_layers": {
    #     "palette": { "A": {"name": "warm", "tags": [...]}, ... },
    #     ...
    #   },
    #   "bank_active_layer": "default"   # "default", "palette",
    #                                    # "energy", "form", or "_custom"
    #
    # Legacy compat: config["bank_categories"] still works as a one-off
    # override (treated as inline layer named "_custom").
    _BUILTIN_BANK_LAYERS: dict = {
        # PALETTE -- bucket by dominant colour temperature / hue.
        "palette": {
            "A": {"name": "warm",   "tags": ["color:warm", "color:hue:red", "color:hue:orange"]},
            "B": {"name": "cool",   "tags": ["color:cool", "color:hue:blue", "color:hue:cyan"]},
            "C": {"name": "yellow", "tags": ["color:hue:yellow", "color:hue:gold"]},
            "D": {"name": "green",  "tags": ["color:hue:green"]},
            "E": {"name": "purple", "tags": ["color:hue:purple", "color:hue:magenta"]},
            "F": {"name": "mono",   "tags": ["color:mono", "grayscale", "black-white"]},
            "G": {"name": "dark",   "tags": ["color:dark", "low-key", "shadow"]},
            "H": {"name": "bright", "tags": ["color:bright", "high-key", "saturated"]},
        },
        # ENERGY -- motion + complexity gradient, chill -> peak.
        "energy": {
            "A": {"name": "ambient", "tags": ["motion:static", "complexity:0", "complexity:1", "still", "calm"]},
            "B": {"name": "soft",    "tags": ["motion:smooth", "complexity:2", "complexity:3", "slow"]},
            "C": {"name": "groove",  "tags": ["complexity:4", "rhythmic", "groove"]},
            "D": {"name": "build",   "tags": ["motion:dynamic", "complexity:5", "complexity:6"]},
            "E": {"name": "drive",   "tags": ["complexity:7", "energetic"]},
            "F": {"name": "peak",    "tags": ["complexity:8", "motion:jumpy", "intense"]},
            "G": {"name": "frantic", "tags": ["complexity:9", "fast", "strobe"]},
            "H": {"name": "chaos",   "tags": ["motion:jumpy", "glitch", "frenetic"]},
        },
        # FORM -- geometry + symmetry of the frame.
        "form": {
            "A": {"name": "particles", "tags": ["geometry:particles", "grain", "dots"]},
            "B": {"name": "linear",    "tags": ["geometry:linear", "lines", "edges"]},
            "C": {"name": "polygons",  "tags": ["geometry:polygons", "shapes", "angular"]},
            "D": {"name": "masks",     "tags": ["geometry:masks", "organic", "blobs"]},
            "E": {"name": "centered",  "tags": ["symmetry:cohesive", "centered", "mirror"]},
            "F": {"name": "offset",    "tags": ["symmetry:offset", "off-center"]},
            "G": {"name": "dense",     "tags": ["complexity:8", "complexity:9", "busy"]},
            "H": {"name": "sparse",    "tags": ["complexity:0", "complexity:1", "negative-space"]},
        },
    }


    def _get_active_layer_name(self) -> str:
        """Return the currently active layer name. 'default' if not set.
        If config has a legacy `bank_categories` override and no
        `bank_active_layer`, returns '_custom' to surface that fact."""
        name = (self.config.get("bank_active_layer") or "").strip().lower()
        if name:
            return name
        # Legacy: an inline override with no named layer = "_custom".
        if self.config.get("bank_categories"):
            return "_custom"
        return "default"

    def _all_layer_names(self) -> list:
        """Cycle order: default first, then builtins (sorted for
        determinism), then any user-defined layers under
        config["bank_layers"], then "_custom" if a legacy
        `bank_categories` override is present."""
        order = ["default"]
        for n in sorted(self._BUILTIN_BANK_LAYERS.keys()):
            if n not in order:
                order.append(n)
        user_layers = self.config.get("bank_layers") or {}
        for n in user_layers.keys():
            n2 = (n or "").strip().lower()
            if n2 and n2 not in order:
                order.append(n2)
        if self.config.get("bank_categories") and "_custom" not in order:
            order.append("_custom")
        return order

    def _get_active_layer_categories(self) -> dict:
        """Return the A-H category dict for the currently active layer.
        Resolution order (first match wins):
          1. "_custom"  -> config["bank_categories"]   (legacy override)
          2. user layer -> config["bank_layers"][name]
          3. builtin    -> self._BUILTIN_BANK_LAYERS[name]
          4. default    -> self._DEFAULT_BANK_CATEGORIES
        """
        name = self._get_active_layer_name()
        if name == "_custom":
            return dict(self.config.get("bank_categories") or {})
        user_layers = self.config.get("bank_layers") or {}
        if name in user_layers and isinstance(user_layers[name], dict):
            return dict(user_layers[name])
        if name in self._BUILTIN_BANK_LAYERS:
            return dict(self._BUILTIN_BANK_LAYERS[name])
        return dict(self._DEFAULT_BANK_CATEGORIES)

    def _layer_short(self, name: str | None = None) -> str:
        """Short suffix shown on bank labels. 'default' is silent
        (returns ''); other layers return their first 3 chars."""
        n = (name if name is not None else self._get_active_layer_name())
        n = (n or "").strip().lower()
        if not n or n == "default":
            return ""
        # Strip leading underscore for "_custom" -> "cus"
        return n.lstrip("_")[:3]

    def cycle_bank_layer(self) -> dict:
        """Rotate to the next named layer in `_all_layer_names()`.
        Persists the new active layer, refreshes iPad bank previews,
        and sets a status message. Intended for the MK2 STEP button
        (bit 9) + iPad layer chip. The newly-active layer takes
        effect on the NEXT bank rebuild — currently-loaded bank
        contents are not re-bucketed (matches the hold-correction
        contract; user can re-press a vertical to apply)."""
        order = self._all_layer_names()
        if not order:
            return {"ok": False, "error": "no layers"}
        current = self._get_active_layer_name()
        try:
            idx = order.index(current)
        except ValueError:
            idx = -1
        next_name = order[(idx + 1) % len(order)]
        self.config["bank_active_layer"] = next_name
        try:
            self.state.set_bank_layer(next_name, order)
        except Exception:
            pass
        try:
            self.state.set_message(
                f"BANK LAYER: {next_name}  "
                f"(re-press vertical / reroll to apply)"
            )
        except Exception:
            pass
        logger.info(
            f"[bank-layer] cycled {current!r} -> {next_name!r}  "
            f"(order={order})"
        )
        try:
            self._refresh_banks_for_ipad()
        except Exception:
            pass
        # Flash a quick pad-toggle visual so the user sees the change.
        try:
            self._pad_flash_toggle("on", label=f"LAYER {next_name}")
        except Exception:
            pass
        # OLED flash so the change is visible on the controller too
        # without looking at the iPad. Persists for 2s.
        try:
            if self.mk2:
                self.mk2.push_oled_layer_flash(
                    layer_name=next_name, hold_seconds=2.0,
                )
        except Exception as e:
            logger.debug(f"layer flash OLED push failed: {e}")
        return {
            "ok": True,
            "active_layer": next_name,
            "order": order,
        }

    # Tag rarity cache (IDF). Built lazily on first categorical split.
    # Generic tags that appear on most files get weight ~0; rare,
    # specific tags get weight ~1. Without this, a bucket keyed to a
    # common tag overflows because nearly every clip matches it.
    _tag_idf_cache: dict | None = None

    def _build_tag_idf(self) -> dict:
        """Build per-tag IDF weights from path_tags.db3. Weight =
        1 - (df / total_files), so a tag on every file gets 0,
        a tag on 1 file gets ~1. Cheap one-shot SQL aggregate."""
        import math
        try:
            with self.path_tags._lock:
                total_row = self.path_tags._conn.execute(
                    "SELECT COUNT(DISTINCT filepath) FROM file_tags"
                ).fetchone()
                total = int(total_row[0] or 1)
                rows = self.path_tags._conn.execute(
                    "SELECT tag, COUNT(DISTINCT filepath) "
                    "FROM file_tags GROUP BY tag"
                ).fetchall()
        except Exception as e:
            logger.warning(f"IDF build failed: {e}")
            return {}
        idf = {}
        for tag, df in rows:
            # Smoothed: weight in [0, 1]. log-scaled so the curve
            # is steep for ultra-common, flat for moderately-common.
            ratio = max(0, min(1, (df or 0) / max(1, total)))
            idf[tag] = max(0.05, 1.0 - math.sqrt(ratio))
        logger.info(
            f"[bank-idf] built weights for {len(idf)} tags "
            f"across {total} files"
        )
        return idf

    def _categorize_file_to_bank(
        self, path: str, categories: dict
    ) -> tuple[str | None, float]:
        """Score this file's tags against each bank category with
        IDF weighting (rare tags count more). Returns (letter, score).
        Letter is None if no category matched at all. The H
        bucket gets a small motion-tag bonus."""
        try:
            tags = self.path_tags.tags_for_file(path) or set()
        except Exception:
            tags = set()
        if not tags:
            return (None, 0.0)
        if self._tag_idf_cache is None:
            self._tag_idf_cache = self._build_tag_idf()
        idf = self._tag_idf_cache or {}
        # User category corrections (hold GROUP X while clip plays).
        # Fuzzy: 1 correction = strong nudge (might flip a close call),
        # 2 = almost always wins, 3+ = locks the bucket. Weighted at
        # 0.8 per count so a single mis-press won't always override
        # clear tag-driven scoring, but consistent corrections do.
        corrections = {}
        if self.vote_store is not None:
            try:
                corrections = self.vote_store.category_corrections(path) or {}
            except Exception:
                corrections = {}
        best_letter = None
        best_score = 0.0
        for ltr in ("A", "B", "C", "D", "E", "F", "G", "H"):
            cat = categories.get(ltr)
            if not cat:
                continue
            cat_tags = set(cat.get("tags") or [])
            # IDF-weighted sum: rare tag matches dominate.
            score = sum(idf.get(t, 0.5) for t in tags if t in cat_tags)
            # High-energy (H) bonus: motion:jumpy + high complexity is
            # often an H clip even without an explicit tag. Small weight so
            # we don't drown real H-tag matches.
            if ltr == "H":
                if "motion:jumpy" in tags:
                    score += 0.3
                for t in tags:
                    if t.startswith("complexity:"):
                        try:
                            if int(t[11:]) >= 7:
                                score += 0.3
                        except ValueError:
                            pass
                        break
            # User correction boost.
            corr_count = int(corrections.get(ltr, 0))
            if corr_count > 0:
                score += corr_count * 0.8
            if score > best_score:
                best_score = score
                best_letter = ltr
        return (best_letter, best_score)

    # NOTE: _energy_score_for_file removed 2026-05-18 afternoon —
    # was a vestigial scorer from the original 1D energy-gradient bank
    # split, replaced by the categorical split system. The mood layer
    # in _BUILTIN_BANK_LAYERS now provides the equivalent functionality
    # via tag-based categorization instead of a hand-tuned score.


    def _split_files_into_banks(
        self, files: list[str], short_name: str = "auto",
        source_folder: str | None = None,
    ) -> dict:
        """Unified split + similarity-pad + save for the
        bank_from_folder + (future) other paths.

        Default: CATEGORICAL banks keyed to visual attributes. Each file
        is scored against every category by tag matches; highest-scoring
        bucket wins. Files that match no category get distributed to
        the under-populated banks so nothing is dropped.

        Override the category map via ``config["bank_categories"]``
        (same shape as ``_DEFAULT_BANK_CATEGORIES``). Set
        ``config["bank_categorical"] = False`` to fall back to plain
        random shuffle."""
        from pathlib import Path as _P
        import random as _random
        if not files:
            return {"ok": False, "error": "empty file list"}
        existing = list(files)
        use_categorical = bool(
            self.config.get("bank_categorical", True)
        )
        bank_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        primary_banks: dict[str, list[str]] = {
            ltr: [] for ltr in bank_letters
        }
        if use_categorical:
            # LAYER OVERLAYS (2026-05-18): resolve active layer's
            # category dict instead of the old single _DEFAULT path.
            # _get_active_layer_categories handles all fallback —
            # legacy override, user-named layer, builtin, default.
            categories = self._get_active_layer_categories()
            if not categories:
                categories = self._DEFAULT_BANK_CATEGORIES
            layer_name = self._get_active_layer_name()
            layer_short = self._layer_short(layer_name)
            unmatched: list[str] = []
            score_log: dict[str, int] = {ltr: 0 for ltr in bank_letters}
            for fp in existing:
                letter, score = self._categorize_file_to_bank(
                    fp, categories
                )
                if letter and score > 0:
                    primary_banks[letter].append(fp)
                    score_log[letter] += 1
                else:
                    unmatched.append(fp)
            # Spread unmatched files into the LEAST-populated banks
            # so no clip is dropped + sparse categories get filler
            # at least vaguely from the same source pool.
            _random.shuffle(unmatched)
            for fp in unmatched:
                target = min(bank_letters, key=lambda l: len(primary_banks[l]))
                primary_banks[target].append(fp)
            # Within each bank, shuffle for varied playback order,
            # but then FLOAT top-voted clips to the front (vote v2):
            # the user's favorites in each category get position 0+
            # so they fire first when the bank loads.
            for ltr in bank_letters:
                _random.shuffle(primary_banks[ltr])
                if self.vote_store is not None and primary_banks[ltr]:
                    try:
                        scored = [
                            (self.vote_store.score(p), -i, p)
                            for i, p in enumerate(primary_banks[ltr])
                        ]
                        # Sort: high vote-score first, then preserve
                        # original shuffle order for ties (via -i).
                        scored.sort(key=lambda t: (-t[0], t[1]))
                        primary_banks[ltr] = [t[2] for t in scored]
                    except Exception:
                        pass
            logger.info(
                f"[bank-categorical] {len(existing)} files: "
                f"matched={score_log} unmatched={len(unmatched)}"
            )
        else:
            _random.shuffle(existing)
            n = len(existing)
            n_banks = min(n, 8)
            per = max(1, n // n_banks) if n_banks else 0
            for i, ltr in enumerate(bank_letters):
                if i >= n_banks:
                    primary_banks[ltr] = []
                    continue
                start = i * per
                end = start + per if i < (n_banks - 1) else n
                primary_banks[ltr] = list(existing[start:end])
        # n + per are referenced by the summary / state-message / log
        # lines further down. The non-categorical `else` branch above
        # defines them, but the categorical branch never did — so a
        # categorical split (the DEFAULT) crashed with "local variable
        # 'n' referenced before assignment". Define them unconditionally
        # here. (Fixed 2026-05-19 — caught by the CLIP→BANKS smoke test.)
        n = len(existing)
        per = max(1, n // 8) if n else 0
        # GROUNDEDNESS over fullness: keep banks ANCHORED in the picked
        # folder. Only pad sparse buckets up to a small MIN_BANK_SIZE
        # floor (not the old TARGET_BANK_SIZE=14 which over-padded).
        # User noted clearly: "should first anchor in main folder of
        # that pick, then fill out IF it lacks enough tags." So a bank
        # with 18 folder clips stays at 18 (no padding). A bank with 2
        # folder clips gets padded up to MIN_BANK_SIZE so it's not
        # one-tap-and-out, but not flooded.
        # Cross-folder similarity-padding is the LAST resort — only
        # fires if even the folder remainder can't get us to the floor.
        MIN_BANK_SIZE = int(self.config.get("bank_min_size", 4) or 4)
        TARGET_BANK_SIZE = MIN_BANK_SIZE  # legacy name some logs use
        # Cross-folder similar files for FALLBACK padding only.
        padding_pool: list[str] = []
        try:
            similar = self.path_tags.similar_files(
                existing, limit=8 * MIN_BANK_SIZE
            )
            padding_pool = [
                p for p, _s in similar if _P(p).is_file()
            ]
        except Exception as e:
            logger.debug(f"similarity padding skipped: {e}")
        out_counts: dict[str, int] = {}
        pad_counts: dict[str, int] = {}  # how many came from cross-folder
        pad_idx = 0
        # Bank label format depends on whether we used categorical
        # splitting — if so, show the category name on the bank label
        # so user sees "Disco A=warm" not just "Disco A". Pulls names
        # from the layer that drove this split (layer overlay,
        # 2026-05-18) so positions/pov/mood layers show their own
        # bucket names rather than the default.
        cats_for_labels = (
            categories if use_categorical else {}
        )
        # Layer short-suffix appears after each cat name, e.g.
        # "Disco D=build(energy)". Default layer = silent (no suffix).
        layer_suffix = ""
        if use_categorical:
            ls = self._layer_short()
            if ls:
                layer_suffix = f"({ls})"
        for ltr in bank_letters:
            primary = primary_banks.get(ltr, [])
            if not primary:
                out_counts[ltr] = -1
                pad_counts[ltr] = 0
                continue
            # PAD ONLY IF below the floor. A well-populated bucket
            # stays at its real size.
            need = max(0, MIN_BANK_SIZE - len(primary))
            if need > 0 and pad_idx < len(padding_pool):
                take = padding_pool[pad_idx:pad_idx + need]
                pad_idx += len(take)
                bank_files = primary + take
                pad_counts[ltr] = len(take)
            else:
                bank_files = primary
                pad_counts[ltr] = 0
            # Composite bank name: includes category name so iPad +
            # OLED show "Disco A=warm" rather than "Disco A". Keeps
            # within reasonable name length (BankStore truncates).
            cat = cats_for_labels.get(ltr, {})
            cat_name = cat.get("name", "")
            if cat_name:
                # Layer-suffix appended in non-default layers so the
                # user sees which categorization drove the split:
                # "Disco A=warm(palette)" vs the default "Disco A=warm".
                bank_label = f"{short_name} {ltr}={cat_name}{layer_suffix}"
            else:
                bank_label = f"{short_name} {ltr}"
            # SET folder lock to the source folder of this split.
            # Reroll will re-pull fresh files from this same folder,
            # so the user stays in the folder they picked rather than
            # drifting to other folders. If no source_folder
            # was passed (legacy callers), pass empty string to CLEAR
            # the lock — safer than preserving an ancient stale one.
            self.bank_store.save_into(
                ltr, bank_label, bank_files,
                set_folder_lock=(source_folder or ""),
            )
            out_counts[ltr] = len(bank_files)
        try:
            self._refresh_banks_for_ipad()
        except Exception:
            pass
        try:
            total = sum(c for c in out_counts.values() if c > 0)
            # Compact per-bank summary in state message: "A=warm(8)
            # B=cool(21) F=offset(12) ..." — only banks with content,
            # so empty buckets don't clutter.
            if use_categorical:
                pieces = []
                for ltr in bank_letters:
                    n_b = out_counts.get(ltr, -1)
                    if n_b > 0:
                        cn = cats_for_labels.get(ltr, {}).get(
                            "name", ltr
                        )
                        n_pad = pad_counts.get(ltr, 0)
                        grounded = n_b - n_pad
                        # Format: A=warm(8) when fully grounded;
                        # D=build(2+3) when 2 from folder + 3 pad.
                        if n_pad > 0:
                            pieces.append(
                                f"{ltr}={cn}({grounded}+{n_pad})"
                            )
                        else:
                            pieces.append(f"{ltr}={cn}({n_b})")
                summary = " ".join(pieces)
                total_padded = sum(pad_counts.values())
                self.state.set_message(
                    f"BANK[{short_name}]: {n}f → {summary}"
                    + (f" (+{total_padded} padded)"
                       if total_padded else " (all grounded)")
                )
            else:
                self.state.set_message(
                    f"BANK[{short_name}]: {n} videos found, "
                    f"{total} fired across A-H"
                )
        except Exception:
            pass
        logger.info(
            f"[bank_from_folder] '{short_name}' total={n} per={per} "
            f"min_size={MIN_BANK_SIZE} dist={out_counts} "
            f"padding={pad_counts}"
        )
        try:
            self._pad_flash_toggle("on", label=f"BANK {short_name}")
        except Exception:
            pass
        return {
            "ok": True,
            "name": short_name,
            "total": n,
            "per_bank": per,
            "distribution": out_counts,
            "categories": {
                ltr: cats_for_labels.get(ltr, {}).get("name", "")
                for ltr in bank_letters
            } if use_categorical else {},
        }

    def mk2_bank_current_folder(self) -> dict:
        """MK2 `>` arrow (bit 13): auto-split the contents of the
        current library folder across banks A-H. Recursive by default
        so pointing at 'Lil Humpers' parent grabs all the year
        subfolders' contents, not just the empty parent. Delegates
        to bank_from_folder (the unified path)."""
        return self.bank_from_folder(folder_rel="", recursive=True)

    def _mk2_bank_current_folder_OLD(self) -> dict:
        """Old shallow-only variant kept inert for reference."""
        from pathlib import Path as _P
        snap = self.state.get_library_snapshot()
        folder_str = snap.get("folder") or snap.get("root") or ""
        if not folder_str:
            try:
                self.state.set_message("MK2 BANK: no folder open")
            except Exception:
                pass
            return {"ok": False, "error": "no folder"}
        folder = _P(folder_str)
        if not folder.is_dir():
            try:
                self.state.set_message(
                    f"MK2 BANK: not a folder: {folder.name}"
                )
            except Exception:
                pass
            return {"ok": False, "error": "not a dir"}
        # Walk the folder for video files. Same extensions as
        # _scan_folder uses (VIDEO_EXTS) but we scan directly here
        # so we get the actual filesystem state, not a stale
        # snapshot's filtered subset.
        try:
            files = [str(p) for p in folder.iterdir()
                     if p.is_file()
                     and p.suffix.lower() in VIDEO_EXTS]
        except OSError as e:
            try:
                self.state.set_message(f"MK2 BANK: scan failed: {e}")
            except Exception:
                pass
            return {"ok": False, "error": f"scan failed: {e}"}
        if not files:
            try:
                self.state.set_message(
                    f"MK2 BANK: no videos in {folder.name}"
                )
            except Exception:
                pass
            return {"ok": False, "error": "no videos"}
        # Adaptive bucket count + preserve untouched letters. Same
        # rationale as bank_auto_split.
        import random as _random
        _random.shuffle(files)
        bank_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        n = len(files)
        n_banks = min(n, 8)
        per = max(1, n // n_banks) if n_banks else 0
        short_name = folder.name[:24] or "auto"
        out_counts: dict[str, int] = {}
        for i, ltr in enumerate(bank_letters):
            if i >= n_banks:
                out_counts[ltr] = -1
                continue
            start = i * per
            end = start + per if i < (n_banks - 1) else n
            bank_files = files[start:end]
            if bank_files:
                self.bank_store.save_into(
                    ltr, f"{short_name} {ltr}", bank_files,
                    set_folder_lock=str(folder).replace("\\", "/"),
                )
            out_counts[ltr] = len(bank_files)
        try:
            self._refresh_banks_for_ipad()
        except Exception:
            pass
        try:
            self.state.set_message(
                f"MK2 BANK[{short_name}]: {n} files -> A-H"
            )
        except Exception:
            pass
        # Pad flash so the operator sees acknowledgement on hardware.
        try:
            self._pad_flash_toggle("on", label=f"MK2 BANK {short_name}")
        except Exception:
            pass
        logger.info(
            f"[mk2/bank-current-folder] '{short_name}' total={n} "
            f"per={per} dist={out_counts}"
        )
        return {
            "ok": True,
            "name": short_name,
            "total": n,
            "per_bank": per,
            "distribution": out_counts,
        }

    def mk2_browse_toggle(self) -> bool:
        """BROWSE button (bit 10): toggle MK2 OLED browse-mode view."""
        self._mk2_browse_mode = not self._mk2_browse_mode
        msg = ("MK2 BROWSE ON (pads = populate)" if self._mk2_browse_mode
               else "MK2 BROWSE OFF (pads = fire)")
        try:
            self.state.set_message(msg)
        except Exception:
            pass
        logger.info(f"[mk2] browse mode = {self._mk2_browse_mode}")
        # Immediate refresh so the OLED reflects state without waiting
        # for the next pad-LED tick.
        try:
            self._refresh_mk2_pad_leds()
        except Exception:
            pass
        return self._mk2_browse_mode

    def _mk2_browse_populate(self, pad_no: int) -> None:
        """Pad press while in BROWSE mode: append the cursored library
        file to the scratch basket and auto-advance the cursor so the
        operator can rapid-tap to build a batch.

        Any of the 16 pads fires this -- the pad number doesn't pick
        a slot (the scratch list is append-only and pad N fires the
        Nth entry in order). The struck pad flashes green to confirm.

        Visual: after the add, _refresh_mk2_pad_leds re-paints all 16
        pads with the new fill state so the operator sees how much
        of the basket is populated."""
        try:
            snap = self.state.get_library_snapshot()
        except Exception as e:
            logger.debug(f"browse populate: snapshot failed: {e}")
            return
        files = snap.get("files") or []
        sel = int(snap.get("selected_idx") or 0)
        if not files or sel < 0 or sel >= len(files):
            try:
                self.state.set_message("BROWSE: no file under cursor")
            except Exception:
                pass
            return
        entry = files[sel]
        # files can be a list of dicts OR plain strings depending on
        # the publisher. Normalise to an absolute path string.
        if isinstance(entry, dict):
            name = entry.get("name") or ""
        else:
            name = str(entry)
        folder = snap.get("folder") or snap.get("root") or ""
        from pathlib import Path as _P
        abs_path = str(_P(folder) / name) if folder and name else name
        if not abs_path or not _P(abs_path).is_file():
            try:
                self.state.set_message(
                    f"BROWSE: can't resolve {name[:40]}"
                )
            except Exception:
                pass
            logger.info(f"[mk2/browse] populate skipped (no file at "
                        f"{abs_path!r})")
            return
        added = self.scratch_store.add(abs_path)
        if added:
            try:
                self._refresh_scratch_for_ipad()
            except Exception:
                pass
            try:
                self.state.set_message(
                    f"+ pad{pad_no}: {_P(abs_path).name[:32]}"
                )
            except Exception:
                pass
            logger.info(
                f"[mk2/browse] populated pad{pad_no} with "
                f"{_P(abs_path).name}"
            )
        else:
            try:
                self.state.set_message(
                    f"already in scratch: {_P(abs_path).name[:32]}"
                )
            except Exception:
                pass
        # Auto-advance cursor +1 so rapid-tapping marches through
        # the library without needing to spin between every assign.
        try:
            self.state.move_library_cursor(1)
        except Exception:
            pass
        # Flash the struck pad green to confirm, then re-paint.
        try:
            self._flash_mk2_pad(pad_no)
        except Exception:
            pass
        try:
            self._refresh_mk2_pad_leds()
        except Exception:
            pass

    # ─── Toggle-state pad flash (2026-05-17) ───────────────────────
    # User feedback: most MK2 buttons are unlit (only transport + groups
    # + 16 pads have LEDs). So when a toggle action fires on an unlit
    # button (SAMPLING / ALL / AUTO WR / top row), the operator gets no
    # button-level feedback. Compensate by briefly painting ALL 16 pads
    # uniform color = the new toggle state. Color is the universal
    # signal: green = on/active, amber = secondary mode, red = off/danger,
    # plus phase-specific colors for set-arc.
    _PAD_FLASH_COLORS = {
        "on":         (40, 200, 80),    # green = active / enabled
        "off":        (180, 60, 60),    # red = disabled
        "neutral":    (90, 90, 90),     # dim gray = neutral / off
        "amber":      (240, 180, 30),   # amber = secondary mode
        "cyan":       (40, 180, 220),   # cyan = tempo lock
        # Set-arc phase colors (match the PDF table)
        "opening":    (30, 80, 200),    # cool blue
        "build":      (240, 160, 30),   # warm amber
        "peak":       (240, 50, 50),    # red
        "breakdown":  (20, 30, 80),     # dark blue
    }

    def _pad_flash_toggle(
        self, color_key_or_rgb, hold_ms: int = 800, label: str = ""
    ) -> None:
        """Flash all 16 MK2 pads uniform color for hold_ms then restore.
        Used as a universal toggle-state confirmation since unlit MK2
        buttons (top row, edit cluster) can't show state. Accepts either
        a key from _PAD_FLASH_COLORS or a raw (r,g,b) tuple."""
        if not self.mk2:
            return
        if isinstance(color_key_or_rgb, str):
            rgb = self._PAD_FLASH_COLORS.get(
                color_key_or_rgb, self._PAD_FLASH_COLORS["neutral"])
        else:
            rgb = tuple(color_key_or_rgb)
        try:
            self.mk2.set_pad_colors_by_label(
                {n: rgb for n in range(1, 17)}
            )
        except Exception as e:
            logger.debug(f"pad flash failed: {e}")
            return
        QTimer.singleShot(int(hold_ms), self._refresh_mk2_pad_leds)
        if label:
            logger.debug(f"[pad-flash] {label} rgb={rgb}")

    def set_arc_cycle(self) -> None:
        """ALL button (bit 14): cycle phase + flip enabled ON if it
        was OFF. So first press = arc on + OPENING; subsequent presses
        = next phase. Reverse cycle would need a separate binding.

        Honors config knobs:
          set_arc_change_delay_ms (default 0): wait N ms before the
              phase actually changes. Gives operator a "press it just
              before the drop" lead-time -- pressing during a buildup
              lets the visible flip land closer to the actual drop.
          set_arc_force_flip_on_change (default True): trigger an
              immediate flip() right after the phase change so the
              new look appears NOW, not on the next auto-beat. Without
              this, operator presses PEAK but visible content doesn't
              shift until the next 5-15s auto-flip cadence."""
        from set_arc import next_phase, label_for
        if not self._set_arc_enabled:
            new_phase = self._set_arc_phase or "opening"
        else:
            new_phase = next_phase(self._set_arc_phase or "opening")
        delay_ms = int(self.config.get("set_arc_change_delay_ms", 0) or 0)
        if delay_ms > 0:
            # Schedule the phase change after `delay_ms`. Pad-flash
            # immediately so the operator sees "PEAK in 1s" feedback.
            try:
                self.state.set_message(
                    f"SET-ARC: {label_for(new_phase)} in {delay_ms}ms..."
                )
            except Exception:
                pass
            self._pad_flash_toggle(new_phase,
                                   label=f"set-arc {new_phase} (incoming)")
            QTimer.singleShot(delay_ms,
                              lambda: self._apply_set_arc_phase(new_phase))
        else:
            self._apply_set_arc_phase(new_phase)

    def _apply_set_arc_phase(self, new_phase: str) -> None:
        """Actually commit the phase change. Separated from
        set_arc_cycle so the delayed-fire path can share it."""
        from set_arc import label_for
        self._set_arc_enabled = True
        self._set_arc_phase = new_phase
        self.config["set_arc_enabled"] = True
        self.config["set_arc_phase"] = new_phase
        try:
            self._save_config()
        except Exception:
            pass
        label = label_for(new_phase)
        try:
            self.state.set_message(f"SET-ARC: {label}")
        except Exception:
            pass
        logger.info(f"[set-arc] phase -> {new_phase} ({label})")
        # Phase change = energy shift; re-roll cohesion anchor so the
        # picker can find a subset that matches the new vibe.
        self._force_cohesion_refresh(reason=f"phase->{new_phase}")
        # Pad flash with phase color
        self._pad_flash_toggle(new_phase, label=f"set-arc {new_phase}")
        # MOTIF CALLBACK on PEAK entry: re-deploy a clip that was the
        # hero of an earlier drop, instead of a fresh random pick —
        # the audience's subconscious recognises the recurring visual
        # and the set reads as designed. Falls back silently to the
        # normal force-flip when no motif is eligible (cooldown / too
        # few peaks so far).
        motif_played = False
        if (new_phase == "peak"
                and self.config.get("picker_motif_enabled", True)):
            try:
                motif_played = self._play_motif_callback()
            except Exception as e:
                logger.debug(f"motif callback failed: {e}")
        # Force-flip on phase change: triggers an immediate pick
        # using the new phase's profile. Means the visible clip
        # changes RIGHT NOW instead of waiting for the next
        # auto-flip beat cadence. Skipped when a motif already took
        # the slot — the motif callback IS the flip.
        if (not motif_played
                and self.config.get("set_arc_force_flip_on_change", True)):
            try:
                logger.info(f"[set-arc] force-flip on phase change")
                self.flip()
            except Exception as e:
                logger.debug(f"set-arc force-flip failed: {e}")
        # HERO-HOLD ON PEAK ENTRY: when the operator (or auto-arc)
        # commits to PEAK, suppress auto-flip for N beats and lock to
        # whatever just flipped in. Per VJ research PDF: the pros sit
        # on a single hero visual at peak; cutting more is rookie.
        # Drop-detector already triggers this on audio spikes — this
        # path covers manual+auto phase changes so the hold happens
        # whether the audio gave us a spike or the operator pre-empted.
        if (new_phase == "peak"
                and self.config.get("set_arc_hero_hold_on_peak", True)):
            try:
                now = time.time()
                beats = int(self.config.get("drop_hero_hold_beats", 16))
                bpm = float(self.state.detected_bpm or 0.0)
                if 60.0 <= bpm <= 200.0:
                    hold_seconds = (60.0 / bpm) * beats
                else:
                    hold_seconds = 8.0
                hold_seconds = max(2.0, min(30.0, hold_seconds))
                self._auto_flip_suppressed_until = now + hold_seconds
                cur = getattr(self, "_current_source_path", None) or (
                    self.player.current_file if self.player else None
                )
                self._hero_locked_clip = str(cur) if cur else None
                # Register this peak's hero clip as a future motif —
                # whatever carries the drop earns a callback later.
                if self._hero_locked_clip:
                    self._register_motif(self._hero_locked_clip)
                logger.info(
                    f"[peak-hero-hold] {beats} beats "
                    f"({hold_seconds:.1f}s) — clip "
                    f"{Path(self._hero_locked_clip).name if self._hero_locked_clip else '<none>'}"
                )
                try:
                    self.state.set_message(
                        f"PEAK — hero hold ({beats} beats)"
                    )
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"peak hero-hold failed: {e}")

    def _register_motif(self, path: str) -> None:
        """Add a clip to the motif registry — the running list of
        clips that have carried a drop (been hero-held at a PEAK).
        Deduped (re-registering moves it to most-recent) and capped
        so stale motifs age out of the rotation."""
        p = str(path or "")
        if not p:
            return
        if p in self._motif_registry:
            self._motif_registry.remove(p)
        self._motif_registry.append(p)
        if len(self._motif_registry) > 12:
            self._motif_registry = self._motif_registry[-12:]

    def _play_motif_callback(self) -> bool:
        """On PEAK entry, force-play a registered motif (a past
        drop's hero clip) that isn't on cooldown — a deliberate
        theatrical callback. Returns True if a motif was played
        (caller then skips the normal force-flip), False to fall
        through to a fresh pick.

        Needs ≥2 registered motifs first, so the opening peaks just
        build the registry before callbacks start."""
        import time as _t
        import random as _r
        if len(self._motif_registry) < 2:
            return False
        cooldown = float(
            self.config.get("motif_cooldown_seconds", 600) or 600
        )
        now = _t.time()
        cur = str(self._current_source_path or "")
        eligible = [
            m for m in self._motif_registry
            if m != cur
            and now - self._motif_last_played.get(m, 0.0) > cooldown
            and Path(m).is_file()
        ]
        if not eligible:
            return False
        motif = _r.choice(eligible)
        self._motif_last_played[motif] = now
        try:
            self._apply_preload_body_seek(motif)
        except Exception:
            pass
        if self.load_video(motif):
            try:
                self.player.play()
            except Exception:
                pass
            logger.info(f"[motif] PEAK callback → {Path(motif).name}")
            try:
                self.state.set_message(
                    f"MOTIF callback: {Path(motif).name[:34]}")
            except Exception:
                pass
            return True
        return False

    def set_arc_toggle(self) -> None:
        """Toggle the set-arc mode on/off without changing phase.
        Exposed via HTTP for iPad / future SHIFT+ALL binding."""
        from set_arc import label_for
        self._set_arc_enabled = not self._set_arc_enabled
        self.config["set_arc_enabled"] = self._set_arc_enabled
        try:
            self._save_config()
        except Exception:
            pass
        msg = (f"SET-ARC ON ({label_for(self._set_arc_phase)})"
               if self._set_arc_enabled else "SET-ARC OFF")
        try:
            self.state.set_message(msg)
        except Exception:
            pass
        logger.info(f"[set-arc] enabled={self._set_arc_enabled}")

    # ── auto-detect ───────────────────────────────────────────────────
    _last_auto_set_arc_check: float = 0.0

    def set_arc_auto_toggle(self) -> bool:
        """Toggle the auto-detect mode on/off. Returns the new state.
        When ON, BPM + flip-rate drive the phase automatically. Manual
        MK2 ALL cycle still works -- but auto will override on the
        next tick if it disagrees."""
        self._set_arc_auto = not getattr(self, "_set_arc_auto", False)
        self.config["set_arc_auto"] = self._set_arc_auto
        try:
            self._save_config()
        except Exception:
            pass
        msg = ("SET-ARC AUTO ON" if self._set_arc_auto
               else "SET-ARC AUTO OFF (manual)")
        try:
            self.state.set_message(msg)
        except Exception:
            pass
        logger.info(f"[set-arc] auto={self._set_arc_auto}")
        # Refresh MK2 mono-LED so the ALL button reflects new state
        # (medium = manual, max = AUTO).
        try:
            self._refresh_mk2_pad_leds()
        except Exception:
            pass
        # Pad flash so operator sees the toggle even without looking
        # at the ALL LED.
        try:
            self._pad_flash_toggle(
                "on" if self._set_arc_auto else "off",
                label=f"set-arc auto {'ON' if self._set_arc_auto else 'OFF'}",
            )
        except Exception:
            pass
        return self._set_arc_auto

    def _auto_set_arc_tick(self) -> None:
        """Called from the 10Hz _tick(). Throttles to ~5s and only
        runs work when both set-arc AND auto-detect are enabled."""
        asa = getattr(self, "_auto_set_arc", None)
        if asa is None:
            return
        if not getattr(self, "_set_arc_enabled", False):
            return
        if not getattr(self, "_set_arc_auto", False):
            return
        now = time.time()
        if now - self._last_auto_set_arc_check < 5.0:
            return
        self._last_auto_set_arc_check = now
        # Read BPM from audio_reactive if alive.
        bpm = 0.0
        ar = getattr(self, "audio_reactive", None)
        if ar is not None:
            try:
                bpm = float(ar.current_bpm() or 0.0)
            except Exception:
                bpm = 0.0
        cur = self._set_arc_phase or "opening"
        try:
            proposed = asa.detect_phase(
                bpm=bpm, current_phase=cur, now=now,
            )
        except Exception as e:
            logger.debug(f"auto-set-arc detect failed: {e}")
            return
        if proposed and proposed != cur:
            from set_arc import label_for
            logger.info(
                f"[set-arc] AUTO phase change: {cur} -> {proposed} "
                f"(bpm={bpm:.1f}, "
                f"rate={asa.flip_rate_per_min(now):.1f}/min, "
                f"trend={asa.bpm_trend():+d})"
            )
            self.state.set_message(
                f"SET-ARC AUTO -> {label_for(proposed)}"
            )
            # Reuse the unified _apply_set_arc_phase so AUTO mode
            # benefits from the same force-flip-on-change + config
            # knobs as manual cycle (no need to duplicate the logic).
            self._apply_set_arc_phase(proposed)

    def _on_mk2_button_release(self, bit: int):
        """MK2 button release. Only HOLD-style actions (e.g.
        audio_leak_hold) react to release; tap-style actions only fire
        on press and ignore release."""
        # ALL button (bit 14): tap = cycle phase, hold (>=800ms) =
        # toggle auto-set-arc. Press handler stored the timestamp in
        # _mk2_hold_press_at; we decide the action here.
        if bit == 14:
            press_at = (getattr(self, "_mk2_hold_press_at", {}) or {}).pop(14, None)
            if press_at is None:
                logger.warning(
                    "[mk2/all-release] no press timestamp -- press "
                    "handler didn't fire or another handler cleared it"
                )
                # Fallback so the button still does something: cycle.
                self.set_arc_cycle()
                return
            held = time.time() - press_at
            logger.info(f"[mk2/all-release] held {held*1000:.0f}ms "
                        f"(threshold 800ms)")
            if held >= 0.8:
                logger.info("[mk2/all-release] -> auto toggle")
                self.set_arc_auto_toggle()
            else:
                logger.info("[mk2/all-release] -> cycle phase")
                self.set_arc_cycle()
            return
        # GROUP buttons (bits 24-31) — letters A-H.
        # Tap = normal bank-load. Hold (>=800ms) while a clip plays =
        # category correction (mk2_reclassify_to(letter)).
        if 24 <= bit <= 31:
            press_at = (
                getattr(self, "_mk2_hold_press_at", {}) or {}
            ).pop(bit, None)
            letter = "ABCDEFGH"[bit - 24]
            if press_at is None:
                # Lost press timestamp — fall back to plain bank load
                # so the button still does something.
                logger.warning(
                    f"[mk2/group-{letter}-release] no press timestamp, "
                    f"falling back to bank_load"
                )
                self.bank_load(letter)
                return
            held = time.time() - press_at
            if held >= 0.8:
                logger.info(
                    f"[mk2/group-{letter}-release] held {held*1000:.0f}ms "
                    f"→ category correction"
                )
                self.mk2_reclassify_to(letter)
            else:
                logger.info(
                    f"[mk2/group-{letter}-release] tap {held*1000:.0f}ms "
                    f"→ bank_load({letter})"
                )
                self.bank_load(letter)
            return
        action_map = (self.config.get("mk2_button_map") or {})
        action = action_map.get(str(bit))
        if action == "audio_leak_hold":
            self.audio_leak_release()

    _mk2_button_queue: "queue.Queue" = None

    def _fire_pending_mk2_button(self):
        if self._mk2_button_queue is None:
            return
        try:
            action = self._mk2_button_queue.get_nowait()
        except queue.Empty:
            return
        if not action:
            return
        try:
            # Action grammar:
            #   "bank:A" .. "bank:H"  → load that bank
            #   "folder:<path>"       → jump library browser to that folder
            #   "reroll_banks"        → re-roll all 8 banks with fresh
            #                           clips (no-repeat memory)
            #   "play"  → toggle play
            #   "stop"  → pause
            #   "next"  → flip()  (folder sibling)
            #   "prev"  → flip_back()
            if action.startswith("bank:"):
                ltr = action.split(":", 1)[1]
                self.bank_load(ltr)
            elif action.startswith("folder:"):
                folder_path = action.split(":", 1)[1]
                # NEW BEHAVIOR 2026-05-18 overnight (revised): vertical
                # buttons now AUTO-SPLIT the folder across all 8 banks
                # A-H instead of loading just the active bank.
                # Workflow: pick page → pick vertical (e.g. "Siri")
                # → all 8 banks fill with chunks of Siri content +
                # similarity-padding. Then Group A/B/C/.../H gives
                # different curated chunks of that performer. Matches
                # the iPad per-folder ⊞ button behavior.
                #
                # Recursive=True so picking a parent folder (e.g.
                # "Lil Humpers") grabs all its year-subfolder clips.
                # `bank_from_folder` also auto-refreshes banks_for_ipad
                # so the iPad bank-preview + left OLED bank label
                # update without a manual poll.
                try:
                    result = self.bank_from_folder(
                        folder_rel=folder_path, recursive=True,
                    )
                    if result.get("ok"):
                        name = result.get("name", "")
                        total = result.get("total", 0)
                        self.state.set_message(
                            f"VERT: {name} → A-H split ({total} files)"
                        )
                    else:
                        self.state.set_message(
                            f"VERT failed: {result.get('error', '?')}"
                        )
                except Exception as e:
                    logger.warning(f"folder vertical-split failed: {e}")
                    self.state.set_message(f"VERT exception: {e}")
            elif action == "reroll_banks":
                # Manual reroll — also reload the active bank's pads so
                # the next pad press fires from the fresh set (the
                # auto-on-track-change reroll deliberately doesn't, to
                # avoid yanking pads mid-tap; manual trigger means the
                # user EXPLICITLY wants the change).
                result = self.reroll_banks(reload_active=True)
                if result.get("ok"):
                    self.state.set_message("Banks rerolled (fresh clips)")
                else:
                    self.state.set_message(
                        f"Reroll failed: {result.get('error','?')}",
                        error=True)
            elif action == "play":
                self.toggle_play()
            elif action == "stop":
                self.pause()
            elif action == "next":
                self.flip()
            elif action == "prev":
                self.flip_back()
            # ── MK2 transport-cluster bindings (discovered 2026-05-16) ──
            elif action == "toggle_flip_on_beat":
                # RESTART button on MK2 — dedicated auto-flip toggle
                # so you don't have to reach for the S2 PFL.
                self.toggle_flip_on_beat()
            elif action == "flip":
                # > arrow on MK2 transport
                self.flip()
            elif action == "flip_back":
                # < arrow on MK2 transport
                self.flip_back()
            elif action == "hold_clip_toggle":
                # GRID button → temporarily pin current clip (auto-flip
                # silently won't fire). Tap again to release; 90s auto-
                # release safety net.
                self.hold_clip_toggle()
            elif action == "tempo_lock_toggle":
                # TEMPO button → toggle whether playback speed is
                # preserved across clip loads or resets to 1.0
                self.tempo_lock_toggle()
            elif action == "tempo_nudge_up":
                # MK2 master-section `>` (bit 20, near TEMPO) → small
                # speed bump. Honors soft-takeover so subsequent
                # tempo-match-on-load won't slam your nudge back.
                # NOTE: bits 19/20 rebound to mk2_vote_down/up
                # 2026-05-18; keep this handler reachable for any
                # config that still calls it explicitly.
                self.tempo_nudge_up()
            elif action == "tempo_nudge_down":
                # MK2 master-section `<` (bit 19, near TEMPO)
                self.tempo_nudge_down()
            elif action == "mk2_vote_up":
                # MK2 master `>` (bit 20, rebound 2026-05-18): upvote
                # the current clip → picker weights it higher next time.
                self.mk2_vote_up()
            elif action == "mk2_vote_down":
                # MK2 master `<` (bit 19, rebound 2026-05-18): downvote
                # current clip → picker suppresses (but doesn't kill it).
                self.mk2_vote_down()
            elif action == "reseek_current_clip":
                # MK2 ENTER (bit 21): re-roll body position on current clip
                self.reseek_current_clip()
            elif action == "stutter_toggle":
                # MK2 NOTE REPEAT (bit 22): beat-locked stutter hold
                self.stutter_toggle()
            elif action == "bpm_tolerance_cycle":
                # MK2 MASTER click (bit 23): cycle BPM-pref tolerance
                self.bpm_tolerance_cycle()
            elif action == "stutter_division_cycle":
                # MK2 top-row bit 0 (temporary home until master encoder
                # rotation gets wired): cycle stutter loop division
                # 4 → 2 → 1 → 1/2 → 1/4 → 1/8 → 1/16 beats. Live-updates
                # active stutter loop if engaged.
                self.stutter_division_cycle()
            # ── MK2 transport-row rebinds (2026-05-18) ──
            elif action == "mk2_blackout":
                # STOP (bit 37) → toggle hard blackout
                self.mk2_blackout()
            elif action == "mk2_force_stem_jump":
                # NEXT (bit 38) → force manual kick-jump (dynamic-motion
                # restricted pick + immediate flip)
                self.mk2_force_stem_jump()
            elif action == "mk2_unhold_and_flip":
                # PREV (bit 39) → release HOLD (if engaged) then flip
                self.mk2_unhold_and_flip()
            else:
                logger.info(f"MK2 button action {action!r} unhandled")
        except Exception as e:
            logger.error(f"MK2 button action {action!r} failed: {e}")

    def _fire_scratch_path(self, path: str) -> None:
        if not path:
            return
        try:
            # play() only if the load wasn't debounced (Audit fix H4).
            if self.load_video(path) and self.player:
                self.player.play()
        except Exception as e:
            logger.error(f"_fire_scratch_path failed: {e}")

    def _on_scratch_folder_added(self, path: str):
        """Drop-in handler for the scratch hot folder. Auto-appends to
        the Scratch basket so the user can curate via Explorer copy-paste
        instead of tapping ✚ on every iPad library row."""
        try:
            self.scratch_add(path)
        except Exception as e:
            logger.debug(f"_on_scratch_folder_added failed: {e}")

    def _on_working_set_added(self, path: str):
        """Hot-folder drop handler: queue the file for proxy transcode and
        nudge the status bar so the user knows the system noticed."""
        try:
            self.proxy_cache.queue(path)
        except Exception:
            pass
        try:
            self.state.set_message(f"Working-set + queued proxy: {Path(path).name}")
        except Exception:
            pass

    def _on_working_set_removed(self, path: str):
        try:
            self.state.set_message(f"Working-set removed: {Path(path).name}")
        except Exception:
            pass

    def _on_audio_reactive_error(self, msg: str):
        """Surface audio-reactive errors to the iPad."""
        self.state.set_audio_reactive(
            enabled=False,
            sensitivity=self.state.audio_sensitivity,
        )
        self.state.set_message(f"Audio reactive: {msg}", error=True)

    def _on_drop_detected(self, intensity: float):
        """Called from the audio-reactive thread when DropDetector spikes.

        Set-craft rule: lock the current hero visual through the peak
        instead of machine-gunning more flips. We suppress auto-flip for
        N beats (one phrase; configurable) and remember the playing clip
        so the lyric_drive picker can prefer hero-tagged clips while the
        hold is active.

        Runs on the audio capture thread → keep it cheap, no Qt mutation.
        Setting floats and a string is fine; the iPad banner & log push
        are safe (set_message is lock-protected, logger is thread-safe).
        """
        now = time.time()
        # Convert "16 beats" to seconds using the current BPM. Fall back
        # to 8 seconds (a reasonable phrase at 120 BPM) if we don't have
        # a tempo lock yet. Capped at 30s so a wildly wrong BPM estimate
        # can't lock visuals for half a minute.
        beats = int(self.config.get("drop_hero_hold_beats", 16))
        bpm = float(self.state.detected_bpm or 0.0)
        if 60.0 <= bpm <= 200.0:
            hold_seconds = (60.0 / bpm) * beats
        else:
            hold_seconds = 8.0
        hold_seconds = max(2.0, min(30.0, hold_seconds))
        self._auto_flip_suppressed_until = now + hold_seconds
        # Remember whatever is playing right now as the "hero" clip so
        # lyric_drive can boost similar tagged clips while held. Safe
        # if nothing is playing — picker just gets None.
        try:
            cur = getattr(self, "_current_source_path", None) or (
                self.player.current_file if self.player else None
            )
            self._hero_locked_clip = str(cur) if cur else None
        except Exception:
            self._hero_locked_clip = None
        logger.info(
            f"[drop] DETECTED intensity={intensity:.2f}x "
            f"-> hero-hold {beats} beats ({hold_seconds:.1f}s) "
            f"clip={Path(self._hero_locked_clip).name if self._hero_locked_clip else '<none>'}"
        )
        try:
            self.state.set_message(
                f"DROP DETECTED — hero hold ({beats} beats)"
            )
        except Exception:
            pass

    # ── User actions ─────────────────────────────────────────────────────

    def open_video_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm);;All Files (*)"
        )
        if path:
            self.load_video(path)

    _last_load_video_ts: float = 0.0
    _LOAD_VIDEO_DEBOUNCE_S: float = 0.30  # mpv needs ~250-300ms to settle a load
    # Track the ORIGINAL source path separately from player.current_file
    # because the latter may point at a transparent proxy. Flip / library
    # operations should walk the original folder, not the proxy folder.
    _current_source_path: Optional[str] = None
    # Play-history stack tuning. _HISTORY_MAX caps memory (covers ~80s
    # at 4s/fire — what "go back" means in practice). Older entries
    # drop off the bottom. _HISTORY_PUSH_MIN_POS_S filters out true
    # mash-fire noise (< 0.5s). AUDIT FIX (2026-05-16): was 1.5s; at
    # 4s auto-flip cadence with HOLD-then-release patterns, 1.5 was
    # too aggressive — < would skip past clips the user actually saw.
    _HISTORY_MAX: int = 20
    _HISTORY_PUSH_MIN_POS_S: float = 0.5
    # Thumb backfill in-flight tracking — prevents rapid double-peeks
    # from spawning racing ffmpeg processes writing the same .jpg.
    # AUDIT FIX (2026-05-16).
    _thumb_backfill_lock: threading.Lock = None  # initialized in __init__
    _thumb_backfill_inflight: set = None         # initialized in __init__

    def load_video(self, path: str, force: bool = False) -> bool:
        """Load `path` into the live player. Returns True if the load
        actually happened, False if it was debounced or failed — callers
        that follow up with play()/seek() MUST check the return so they
        don't operate on the previously-loaded file. (Audit fix H4.)

        `force=True` bypasses the debounce — use for authoritative loads
        like the startup last-video restore."""
        if not self.player:
            return False
        # Debounce: rapid back-to-back loads (auto-flip flurry, bank tap
        # storm, manual flip mash) leave mpv in an undefined state and
        # can black out the display. Drop loads inside a 300ms window
        # of the last successful one — the LAST load wins, but we don't
        # tear-down + restart 5x in 1.5s. Same path always wins through
        # though (idempotent).
        now = time.time()
        if not force:
            if (path == getattr(self.player, "current_file", None)
                    and now - self._last_load_video_ts < self._LOAD_VIDEO_DEBOUNCE_S):
                logger.debug(f"load_video debounced (same path within {self._LOAD_VIDEO_DEBOUNCE_S}s): {path}")
                return False
            if now - self._last_load_video_ts < self._LOAD_VIDEO_DEBOUNCE_S:
                logger.warning(
                    f"load_video debounced ({(now - self._last_load_video_ts)*1000:.0f}ms since last): {Path(path).name}"
                )
                return False
        self._last_load_video_ts = now
        # Play-history push: snapshot the OUTGOING clip + position so
        # `<` (flip_back) can jump back to it where we left off. Skip
        # when:
        #   - flip_back itself is the caller (would re-push the popped
        #     entry → oscillation)
        #   - no real previous clip (first load)
        #   - same path (just a re-fire of what's playing — pointless)
        #   - position < 1.5s (mash-fire / barely-seen — clutter)
        # Cap stack at 20 entries — covers ~80s at 4s/fire which is
        # what "go back" means in practice. Older drops off the bottom.
        try:
            cur_src = self._current_source_path
            if (not self._flip_back_in_progress
                    and cur_src and cur_src != path
                    and self.player and self.player.player):
                cur_pos = float(getattr(self.player.player, "time_pos", 0.0) or 0.0)
                if cur_pos >= self._HISTORY_PUSH_MIN_POS_S:
                    self._flip_history.append((cur_src, cur_pos))
                    if len(self._flip_history) > self._HISTORY_MAX:
                        self._flip_history.pop(0)
                    logger.debug(f"[flip-history] pushed {Path(cur_src).name} @ "
                                 f"{cur_pos:.1f}s (stack={len(self._flip_history)})")
        except Exception as e:
            logger.debug(f"flip-history push failed: {e}")
        # Remember the ORIGINAL path for flip / folder navigation. mpv's
        # current_file may end up pointing at a proxy (transparent swap),
        # but flip() must walk the original file's folder, not the proxy
        # cache directory.
        self._current_source_path = path
        # Track this fire in the no-repeat memory so the next bank
        # reroll won't re-suggest the same clip.
        self._remember_recent_clip(path)
        # Proxy cache: if we already have a 1080p transcode of this file
        # cached, hand mpv the proxy instead — saves NVDEC + PCIe pressure
        # for 4K sources. Original path stays the metadata source of truth.
        # If no proxy yet, queue one for later — second time you load this
        # file it'll play back from the proxy.
        playback_path = path
        try:
            proxy = self.proxy_cache.get_proxy(path)
            if proxy:
                playback_path = proxy
                logger.debug(f"Using proxy for {Path(path).name}")
            else:
                self.proxy_cache.queue(path)
        except Exception as e:
            logger.debug(f"proxy_cache lookup failed: {e}")
        try:
            self.player.load(playback_path)
            # Save the ORIGINAL path as last_video so a restart hits the
            # cache check, not a stale proxy file. DEBOUNCED — only
            # persist at most once per 10s so an auto-flip storm doesn't
            # hammer the disk (audit fix C3). The last write before
            # shutdown still lands via closeEvent's flush.
            self.config["last_video"] = path
            now = time.time()
            if now - getattr(self, "_last_config_save_ts", 0.0) > 10.0:
                self._last_config_save_ts = now
                self._save_config()
            self.label_file.setText(f"Loaded: {Path(path).name}")
            self.state.set_message(f"Loaded {Path(path).name}")
            logger.info(f"Loaded video: {path}")
            # Tempo-match (2026-05-16): a clip cut to one song can be
            # warped to the live BPM. If both the clip's tagged BPM and
            # the live detected BPM are known, warp playback speed =
            # live / clip so the clip's music aligns with what's playing.
            # No data on either side → fall through to sticky speed.
            try:
                self._apply_tempo_match(path)
            except Exception as e:
                logger.debug(f"tempo match failed: {e}")
            # Per-video pad/clip remap — new video = new deck of cue points.
            self._refresh_clips_for_ipad()
            # MK2 LED feedback: the live pad changes to white, others
            # revert to cyan/off. No-op if MK2 isn't connected.
            self._refresh_mk2_pad_leds()
            # OSC clip-change broadcast (no-op when disabled).
            try:
                if self.osc:
                    self.osc.send_clip_change(path, Path(path).name)
            except Exception as e:
                logger.debug(f"osc send_clip_change failed: {e}")
            return True
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            self.state.set_message(f"Load failed: {e}", error=True)
            return False

    def play(self):
        if self.player:
            self.player.play()

    def pause(self):
        if self.player:
            self.player.pause()
        # Reflect new pause state on MK2 PLAY LED
        try:
            self._refresh_mk2_transport_leds()
        except Exception as e:
            logger.debug(f"pause: mk2 transport LED refresh failed: {e}")

    def toggle_play(self):
        if not self.player:
            return
        if self.player.is_playing():
            self.player.pause()
        else:
            self.player.play()
        # Reflect new play state on MK2 PLAY LED
        try:
            self._refresh_mk2_transport_leds()
        except Exception as e:
            logger.debug(f"toggle_play: mk2 transport LED refresh failed: {e}")

    # Fullscreen safety. showFullScreen() is a *borderless window*, not
    # exclusive-mode — so even if the Qt event loop wedges while fullscreen,
    # Alt-Tab / Win / Task Manager still work. The hang-watchdog below is
    # belt-and-suspenders on top: it force-minimizes a frozen fullscreen
    # window off the screen so the desktop is never trapped behind it.
    _is_fullscreen: bool = False
    _HANG_SECONDS = 5.0

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self._is_fullscreen = False
        else:
            self.showFullScreen()
            self._is_fullscreen = True

    def _exit_fullscreen(self):
        """ESC handler — only ever leaves fullscreen, never enters it."""
        if self.isFullScreen():
            self.showNormal()
        self._is_fullscreen = False

    def _beat(self):
        """Fullscreen hang-watchdog heartbeat — bumped on the Qt thread."""
        self._ui_heartbeat = time.monotonic()

    def _init_hang_watchdog(self):
        """Arm the fullscreen hang-watchdog. Post-show so winId() is real."""
        try:
            self._hwnd = int(self.winId())
        except Exception as e:
            logger.debug(f"hang watchdog: winId() failed, disabled: {e}")
            self._hwnd = 0
            return
        self._watchdog_stop = threading.Event()
        threading.Thread(
            target=self._hang_watchdog_loop, name="fullscreen-hang-watchdog",
            daemon=True,
        ).start()
        logger.info("fullscreen hang-watchdog armed")

    def _hang_watchdog_loop(self):
        """Daemon thread. If the Qt event loop stops heartbeating WHILE the
        window is fullscreen, force-minimize it via the OS so a frozen app
        can't cover the screen. On Windows that's SW_FORCEMINIMIZE — built
        for exactly this (minimizing a hung window from another thread);
        elsewhere it just logs (Alt-Tab still works regardless)."""
        import ctypes
        SW_FORCEMINIMIZE = 11
        panicked = False
        while not self._watchdog_stop.wait(1.0):
            stale = time.monotonic() - self._ui_heartbeat
            hung = stale > self._HANG_SECONDS
            if hung and self._is_fullscreen and not panicked:
                panicked = True
                logger.error(
                    f"UI thread unresponsive {stale:.0f}s while fullscreen — "
                    "force-minimizing the window so it can't trap the screen"
                )
                try:
                    if sys.platform == "win32" and self._hwnd:
                        ctypes.windll.user32.ShowWindowAsync(
                            self._hwnd, SW_FORCEMINIMIZE)
                    else:
                        logger.error("no OS panic-minimize on this platform — "
                                     "use Alt-Tab to escape the frozen window")
                except Exception as e:
                    logger.error(f"hang watchdog panic-minimize failed: {e}")
            elif not hung:
                panicked = False  # event loop recovered — re-arm

    # ─── MK2 transport-row rebinds (2026-05-18) ───────────────────
    # STOP / NEXT / PREV were duplicates of existing flip / pause
    # paths. Rebound to higher-value VJ ops without disturbing the
    # plain `stop` / `next` / `prev` actions (iPad / web UI still
    # call those directly).

    def mk2_blackout(self) -> None:
        """MK2 STOP (bit 37): toggle a hard blackout. Wraps the
        existing brightness-based `blackout_toggle` so the visual
        and on-screen message match the keyboard B RESET path."""
        try:
            was_active = bool(getattr(self, "_blackout_active", False))
            self.blackout_toggle()
            if not was_active:
                self.state.set_message(
                    "BLACKOUT (STOP again to release)")
            else:
                self.state.set_message("BLACKOUT released")
        except Exception as e:
            logger.debug(f"mk2_blackout failed: {e}")

    def mk2_force_stem_jump(self) -> None:
        """MK2 NEXT (bit 38): force a manual stem kick-jump
        regardless of audio. Opens the stem boost window for 0.5s
        so the next _pick_in_pool path applies the dynamic-motion
        restriction, then fires flip() immediately so the operator
        feels the jump now."""
        try:
            self._stem_drum_boost_until = time.time() + 0.5
            logger.info("MK2 FORCE STEM-JUMP")
            self.state.set_message("STEM-JUMP forced")
            self.flip()
        except Exception as e:
            logger.debug(f"mk2_force_stem_jump failed: {e}")

    def mk2_unhold_and_flip(self) -> None:
        """MK2 PREV (bit 39): if HOLD is engaged, release it via
        the existing hold_clip_toggle path (preserves its log +
        auto-release bookkeeping), then flip() to advance. When
        no hold is active this just flips."""
        try:
            had_hold = bool(getattr(self, "_hold_clip", False))
            if had_hold:
                self.hold_clip_toggle()
                self.state.set_message("HOLD released -> FLIP")
            else:
                self.state.set_message("FLIP")
            self.flip()
        except Exception as e:
            logger.debug(f"mk2_unhold_and_flip failed: {e}")

    # ── Shader overlays ──────────────────────────────────────────────────
    # mpv user-shaders applied as a stack on top of the live video. Each
    # is in shaders/<name>.glsl. Toggle on/off independently; they stack
    # (e.g. scanlines + vignette together).
    _active_shaders: set = set()

    def toggle_shader(self, name: str) -> None:
        """Toggle one of the shaders/<name>.glsl files on/off in mpv's
        glsl-shaders stack. Multiple can stack (scanlines + vignette).
        Settings are recomputed and pushed each toggle."""
        if not self.player or not self.player.player:
            return
        shader_path = Path(__file__).resolve().parent / "shaders" / f"{name}.glsl"
        if not shader_path.is_file():
            self.state.set_message(f"shader not found: {name}.glsl", error=True)
            return
        if name in self._active_shaders:
            self._active_shaders.discard(name)
            on = False
        else:
            self._active_shaders.add(name)
            on = True
        try:
            paths = [
                str(Path(__file__).resolve().parent / "shaders" / f"{s}.glsl")
                for s in self._active_shaders
            ]
            # mpv glsl-shaders takes a list of paths. Empty list -> no shaders.
            self.player.player.glsl_shaders = paths if paths else ""
            active_label = ", ".join(sorted(self._active_shaders)) if self._active_shaders else "none"
            self.state.set_message(
                f"Shader {name} {'ON' if on else 'OFF'} (active: {active_label})")
        except Exception as e:
            logger.debug(f"toggle_shader({name}) failed: {e}")
            self.state.set_message(f"shader toggle failed: {e}", error=True)

    def _flip_pool(self) -> "list[Path]":
        """Return the ordered pool of paths to cycle through on every
        flip (manual > / < and auto-flip alike).

        Priority (2026-05-16 rework per user req):
          1. Active bank / scratch contents — the curated pool. This is
             what `bank_load` poured into `scratch_store`, so the live
             scratch list IS the active bank's files.
          2. Folder siblings — safety fallback when scratch is empty or
             trivially small (≤1 file). Prevents getting stuck.

        FOLDER auto-flip mode overrides #1 and forces folder siblings —
        the explicit "I want library-folder random walk" escape hatch.
        """
        with self._auto_flip_lock:
            use_folder = self._auto_flip_use_folder
        if not use_folder:
            try:
                # AUDIT FIX (2026-05-16): dropped per-file Path.exists()
                # check — that's N syscalls on the Qt thread on every
                # flip, can stall on cold cache / network drives. load_video
                # already fails loudly if a file's gone, and removing
                # missing files is a scratch-store cleanup concern.
                scratch = [Path(p) for p in (self.scratch_store.all() or [])
                           if p]
            except Exception:
                scratch = []
            if len(scratch) > 1:
                return scratch
        # FOLDER mode (or empty bank) → folder siblings of current file.
        if not self.player or not self.player.current_file:
            return []
        cur = Path(self._current_source_path or self.player.current_file)
        try:
            return sorted(
                [p for p in cur.parent.iterdir()
                 if p.suffix.lower() in (".mp4", ".mkv", ".mov", ".avi", ".webm")]
            )
        except Exception:
            return []

    # BPM-aware picker (2026-05-16 night). With 3644 clips BPM-tagged
    # and live audio_reactive detecting current music BPM, the picker
    # can prefer clips whose source music is within tolerance of what's
    # playing live. Compounds with tempo-match-on-load: picked clips
    # need less warping, so the music aligns more naturally.
    _bpm_preference_enabled: bool = True
    _BPM_PREF_TOLERANCE: float = 10.0  # ±10 BPM = musically tight match

    def _bulk_get_clip_bpms(self, paths) -> dict:
        """One SQL query for many paths' bpm:N tags. Returns
        {filepath_str: bpm_int} for any path with a tag. Much faster
        than N round-trips through tags_for_file()."""
        if not paths:
            return {}
        out: dict[str, int] = {}
        try:
            placeholders = ",".join("?" * len(paths))
            path_strs = tuple(str(p) for p in paths)
            with self.path_tags._lock:
                rows = self.path_tags._conn.execute(
                    f"SELECT filepath, tag FROM file_tags "
                    f"WHERE filepath IN ({placeholders}) AND tag LIKE 'bpm:%'",
                    path_strs,
                ).fetchall()
            for fp, tag in rows:
                try:
                    out[fp] = int(tag[4:])
                except (ValueError, IndexError):
                    pass
        except Exception as e:
            logger.debug(f"_bulk_get_clip_bpms failed: {e}")
        return out

    def _bpm_matches(self, clip_bpm: int, live_bpm: float,
                     tolerance: float = None) -> bool:
        """True if clip_bpm aligns to live_bpm within tolerance,
        accounting for autocorrelation octave-doubling (clip tagged
        180 might really be 90; matches live=90 via the /2 octave)."""
        if clip_bpm <= 0 or live_bpm <= 0:
            return False
        tol = tolerance if tolerance is not None else self._BPM_PREF_TOLERANCE
        for src in (float(clip_bpm), clip_bpm * 2.0, clip_bpm / 2.0):
            if abs(src - live_bpm) <= tol:
                return True
        return False

    def _filter_pool_by_bpm(self, pool, live_bpm: float):
        """Subset of pool whose tagged BPM matches live within
        tolerance. Untagged clips are EXCLUDED (unknown != mismatch,
        but we want to prefer KNOWN-good choices when available).
        Returns [] if no matches — caller should fall back to full pool."""
        if not pool or live_bpm <= 0:
            return []
        bpm_map = self._bulk_get_clip_bpms(pool)
        matched = []
        for p in pool:
            cb = bpm_map.get(str(p))
            if cb and self._bpm_matches(cb, live_bpm):
                matched.append(p)
        return matched

    def _filter_pool_by_dynamic_motion(self, pool):
        """Subset of pool tagged with motion:dynamic or motion:jumpy.
        Used by the stem-onset boost in _pick_in_pool: when a kick
        just hit, restrict random picks to high-energy clips.

        Untagged clips are EXCLUDED -- we'd rather pick from a known
        smaller dynamic-motion set than guess. Returns [] when nothing
        matches; caller falls back to the unfiltered candidates."""
        if not pool:
            return []
        try:
            import sqlite3
            from pathlib import Path as _P
            db_path = _P.home() / ".setpiece" / "path_tags.db3"
            str_pool = [str(p) for p in pool]
            placeholders = ",".join("?" * len(str_pool))
            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True, timeout=1.0,
            )
            rows = conn.execute(
                f"SELECT DISTINCT filepath FROM file_tags "
                f"WHERE filepath IN ({placeholders}) "
                f"AND tag IN ('motion:dynamic', 'motion:jumpy')",
                str_pool,
            ).fetchall()
            conn.close()
            matches = {r[0] for r in rows}
            # Preserve original pool order
            return [p for p in pool if str(p) in matches]
        except Exception as e:
            logger.debug(f"_filter_pool_by_dynamic_motion failed: {e}")
            return []

    def bpm_preference_toggle(self) -> None:
        """Toggle BPM-aware picking. When ON (default), auto-flip and
        random > prefer clips within ±10 BPM of live music. When OFF,
        all clips in the pool are equally weighted (old behavior)."""
        self._bpm_preference_enabled = not self._bpm_preference_enabled
        state_str = "ON" if self._bpm_preference_enabled else "OFF"
        self.state.set_message(f"BPM-pick: {state_str}")
        logger.info(f"[bpm-pick] toggled {state_str}")

    def _picker_record(self, picked: object, reason: str,
                        extras: dict = None) -> None:
        """Append a pick decision to the rolling history for the iPad
        debug panel. Bounded to last 30 entries. Best-effort; never
        raises into the picker hot path."""
        try:
            from pathlib import Path as _P
            if not hasattr(self, "_picker_history"):
                from collections import deque as _deque
                self._picker_history = _deque(maxlen=30)
            entry = {
                "ts": time.time(),
                "name": _P(str(picked)).name if picked else "",
                "reason": reason,
            }
            if extras:
                entry.update(extras)
            self._picker_history.append(entry)
        except Exception:
            pass

    def picker_recent(self, limit: int = 10) -> dict:
        """Return last N pick decisions for iPad debug panel."""
        history = list(getattr(self, "_picker_history", []) or [])
        # Most recent first, age in seconds
        now = time.time()
        recent = []
        for e in reversed(history[-int(limit):]):
            d = dict(e)
            d["age_s"] = round(now - d.get("ts", now), 2)
            recent.append(d)
        return {"ok": True, "picks": recent}

    # CLIP semantic search state — cached after first call so the
    # text encoder + embeddings matrix don't reload per query.
    _clip_search_state: dict = {}

    def clip_search(self, query: str, top: int = 15) -> dict:
        """Run a natural-language semantic search over the library.

        First call loads the CLIP text encoder + entire embeddings
        matrix (~5 sec). Subsequent calls reuse them for sub-second
        searches. Falls back to a clean error if no embeddings exist
        yet (clip_tagger.py hasn't run / no rows in clip_embeddings).

        Returns:
          {ok, query, count, results: [{score, path, name}], elapsed_ms}
        """
        import time as _time
        from pathlib import Path as _P
        if not query:
            return {"ok": False, "error": "empty query"}
        t0 = _time.time()
        try:
            # Lazy-load on first call: CLIP text model + tokenizer +
            # entire embeddings matrix. All cached so subsequent
            # queries are sub-second (just text encode + matmul).
            state = self._clip_search_state
            if "text_model" not in state:
                try:
                    import clip_search as _cs
                    import sqlite3 as _sql
                    state["_cs"] = _cs
                    # 1) text encoder + tokenizer
                    tm, tok, dev = _cs._load_text_model()
                    state["text_model"] = tm
                    state["tokenizer"] = tok
                    state["device"] = dev
                    # 2) embeddings matrix
                    conn = _sql.connect(str(_cs.DB_PATH), timeout=30.0)
                    conn.execute("PRAGMA busy_timeout=30000")
                    paths, mat = _cs._load_embeddings_matrix(conn, None)
                    conn.close()
                    state["paths"] = paths
                    state["matrix"] = mat
                    logger.info(
                        f"[clip-search] warmed: text model on {dev}, "
                        f"{len(paths)} embeddings loaded into memory"
                    )
                except Exception as e:
                    return {
                        "ok": False,
                        "error": f"clip_search warmup failed: {e}",
                    }
            cs = state["_cs"]
            # Embed query (fast — single tokenize + encode pass)
            q = cs.embed_text(
                query,
                model=state["text_model"],
                tokenizer=state["tokenizer"],
                device=state["device"],
            )
            # Cosine sim via dot product (both L2-normalized)
            sims = state["matrix"] @ q
            n = len(state["paths"])
            top_n = int(top)
            if top_n >= n:
                import numpy as _np
                order = _np.argsort(-sims)
            else:
                import numpy as _np
                idx = _np.argpartition(-sims, top_n)[:top_n]
                order = idx[_np.argsort(-sims[idx])]
            elapsed_ms = int((_time.time() - t0) * 1000)
            results = []
            for i in order[:top_n]:
                results.append({
                    "score": round(float(sims[i]), 4),
                    "path": state["paths"][i],
                    "name": _P(state["paths"][i]).name,
                })
            return {
                "ok": True,
                "query": query,
                "count": len(results),
                "results": results,
                "elapsed_ms": elapsed_ms,
            }
        except FileNotFoundError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            logger.warning(f"clip_search failed: {e}")
            return {"ok": False, "error": str(e)}

    def votes_summary(self, limit: int = 10) -> dict:
        """Summary of vote / category-correction state for iPad panel.
        Returns:
          - total_voted: count of files with any vote
          - top: list of {name, path, score} for highest-scored files
          - bottom: list of {name, path, score} for lowest-scored
          - current: the currently-playing clip's stats + corrections
        """
        if self.vote_store is None:
            return {"ok": False, "error": "vote store unavailable"}
        from pathlib import Path as _P
        try:
            top = [
                {"name": _P(p).name, "path": p, "score": s}
                for p, s in self.vote_store.top_voted(limit)
            ]
            bottom = [
                {"name": _P(p).name, "path": p, "score": s}
                for p, s in self.vote_store.bottom_voted(limit)
            ]
            cur_path = self._current_clip_path()
            current = None
            if cur_path:
                score, ups, downs = self.vote_store.stats(cur_path)
                corrections = self.vote_store.category_corrections(
                    cur_path
                )
                current = {
                    "name": _P(cur_path).name,
                    "path": cur_path,
                    "score": score,
                    "ups": ups,
                    "downs": downs,
                    "corrections": corrections,
                    "picker_weight": round(
                        self.vote_store.picker_weight(cur_path), 2
                    ),
                }
            return {
                "ok": True,
                "total_voted": self.vote_store.total_voted(),
                "top": top,
                "bottom": bottom,
                "current": current,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── COHESION ANCHOR (2026-05-19) ─────────────────────────────────
    # User feedback: "as it's picking maybe a bit more reduction... a
    # lot of hopping... keeping it more grounded in a couple of files
    # or real similar for say a couple mins would help... almost
    # treating like this is a music video so should look like from
    # the same video shoot ish."
    #
    # Strategy: lock the picker to a small subset (3 files) of the
    # active pool for ~90s. Subset chosen by CLIP visual similarity
    # to a seed (the currently-playing file, or top-voted in pool,
    # or random fallback). BPM-match + stem-jump + vote-weight still
    # apply WITHIN the subset — cohesion is just the outer filter.
    # Re-rolls on: timer expiry, pool change (bank load), or operator
    # force (hold-correction, ARC phase change).
    _cohesion_subset: list = []
    _cohesion_until: float = 0.0
    _cohesion_seed: str = ""
    _cohesion_pool_hash: int = 0

    def _clip_neighbors(self, seed_path: str, pool: list,
                        top_k: int = 2) -> list:
        """Find top_k most CLIP-visually-similar files to `seed_path`
        from `pool`. Returns list of paths (excluding seed). Uses the
        cached CLIP state from clip_search if available; otherwise
        returns []. Caller falls back to random when this returns []."""
        import numpy as _np
        state = self._clip_search_state
        # Need the matrix + paths loaded. They warm on first
        # /api/clip/search call. If user hasn't searched yet, lazy-
        # load them now (~3-5s one-time cost on first cohesion roll).
        if "matrix" not in state:
            try:
                import clip_search as _cs
                import sqlite3 as _sql
                state["_cs"] = _cs
                tm, tok, dev = _cs._load_text_model()
                state["text_model"] = tm
                state["tokenizer"] = tok
                state["device"] = dev
                conn = _sql.connect(str(_cs.DB_PATH), timeout=30.0)
                conn.execute("PRAGMA busy_timeout=30000")
                paths, mat = _cs._load_embeddings_matrix(conn, None)
                conn.close()
                state["paths"] = paths
                state["matrix"] = mat
                # path → matrix-row index lookup for O(1) seed query
                state["path_to_idx"] = {p: i for i, p in enumerate(paths)}
                logger.info(
                    f"[cohesion] lazy-loaded CLIP cache: "
                    f"{len(paths)} embeddings"
                )
            except Exception as e:
                logger.debug(f"cohesion CLIP load failed: {e}")
                return []
        idx_map = state.get("path_to_idx")
        if not idx_map:
            idx_map = {p: i for i, p in enumerate(state["paths"])}
            state["path_to_idx"] = idx_map
        seed_idx = idx_map.get(seed_path)
        if seed_idx is None:
            return []
        matrix = state["matrix"]
        paths_all = state["paths"]
        seed_vec = matrix[seed_idx]
        # Compute sim only against pool members that HAVE embeddings.
        pool_idx = [idx_map[p] for p in pool
                    if p in idx_map and p != seed_path]
        if not pool_idx:
            return []
        # Cosine sim via dot product (matrix L2-normalized at write time)
        sims = matrix[pool_idx] @ seed_vec
        # Top-k indices into pool_idx (then map back to paths)
        order = _np.argsort(-sims)[:top_k]
        return [paths_all[pool_idx[i]] for i in order]

    def _force_cohesion_refresh(self, reason: str = "") -> None:
        """Invalidate the cohesion subset so the next pick re-rolls.
        Called on bank_load, hold-correction, ARC phase change."""
        self._cohesion_until = 0.0
        self._cohesion_pool_hash = 0
        if reason:
            logger.info(f"[cohesion] refresh forced ({reason})")

    def _get_cohesion_subset(self, pool: list) -> list:
        """Return the currently-locked subset, re-rolling if expired.
        Falls back to the full pool when cohesion is disabled or the
        pool is too small to benefit (<= subset_size + 1)."""
        import time as _time
        if not self.config.get("picker_cohesion_enabled", True):
            return pool
        subset_size = int(
            self.config.get("picker_cohesion_size", 3) or 3
        )
        lock_seconds = float(
            self.config.get("picker_cohesion_lock_seconds", 90)
            or 90
        )
        if len(pool) <= subset_size + 1:
            # Pool too small to subset; return as-is.
            return pool
        now = _time.time()
        pool_hash = hash(tuple(pool))
        if (now >= self._cohesion_until
                or pool_hash != self._cohesion_pool_hash
                or not self._cohesion_subset
                or not all(p in pool for p in self._cohesion_subset)):
            self._roll_cohesion_subset(pool, subset_size)
            self._cohesion_until = now + lock_seconds
            self._cohesion_pool_hash = pool_hash
        # Filter to pool members in case pool changed slightly mid-window
        active = [p for p in self._cohesion_subset if p in pool]
        return active or pool

    def _phase_score_for_file(self, path: str, phase: str) -> float:
        """Return the set_arc.score_clip multiplier for `path` under
        `phase`. Extracts the dimensions (thermal/hue/motion/...) from
        the file's tag set. Returns 1.0 if anything fails."""
        try:
            from set_arc import score_clip
            tags = self.path_tags.tags_for_file(path) or set()
        except Exception:
            return 1.0
        if not tags:
            return 1.0
        # Pull dimensions out of the tag set
        thermal = None
        hue = None
        complexity = None
        geometry = None
        symmetry = None
        motion_tags: set[str] = set()
        transition_rich = False
        bg_friendly = "tag:bg_friendly" in tags or "bg_friendly" in tags
        dialog = "tag:dialog" in tags or "dialog" in tags
        for t in tags:
            if t == "color:warm" or t == "color:cool":
                thermal = t.split(":", 1)[1]
            elif t.startswith("color:hue:"):
                hue = t[10:]
            elif t.startswith("motion:"):
                motion_tags.add(t[7:])
            elif t.startswith("complexity:"):
                try:
                    complexity = int(t[11:])
                except ValueError:
                    pass
            elif t.startswith("geometry:"):
                geometry = t[9:]
            elif t.startswith("symmetry:"):
                symmetry = t[9:]
            elif t == "transition_rich":
                transition_rich = True
        try:
            return float(score_clip(
                phase,
                thermal=thermal,
                hue=hue,
                motion_tags=motion_tags or None,
                complexity=complexity,
                geometry=geometry,
                symmetry=symmetry,
                transition_rich=transition_rich,
                bg_friendly=bg_friendly,
                dialog=dialog,
            ))
        except Exception:
            return 1.0

    def _phase_weighted_seed(self, pool: list,
                             phase: str) -> "str | None":
        """Weighted-random seed pick. Each candidate's weight is its
        phase_score; result is sampled proportional to weight^1.5 so
        strong matches dominate without locking out variety. Returns
        None if all scores are equal (caller should fall back to
        plain random)."""
        import random as _random
        if not pool:
            return None
        scores = [self._phase_score_for_file(p, phase) for p in pool]
        # Power-up the weights so strong matches dominate
        weights = [max(0.05, s) ** 1.5 for s in scores]
        if not any(w > 1.0 for w in weights):
            # Nothing scored above 1.0 — phase has no opinion on this pool
            return None
        try:
            return _random.choices(pool, weights=weights, k=1)[0]
        except Exception:
            return None

    def _roll_cohesion_subset(self, pool: list,
                              subset_size: int) -> None:
        """Pick a new cohesion subset: seed (current/voted/phase/random) +
        top (subset_size - 1) CLIP nearest-neighbors in the pool.
        Sets self._cohesion_subset + self._cohesion_seed."""
        import random as _random
        # Seed selection priority:
        # 1. currently-playing file IF in pool
        # 2. top-voted in pool (if any votes)
        # 3. phase-weighted-random IF set-arc is enabled
        # 4. uniform-random pool member
        cur = str(
            self._current_source_path
            or (self.player.current_file if self.player else "")
            or ""
        )
        seed = cur if cur and cur in pool else None
        mode_seed = "current"
        if seed is None and self.vote_store is not None:
            try:
                scored = [
                    (self.vote_store.score(p), p) for p in pool
                ]
                scored.sort(reverse=True, key=lambda t: t[0])
                if scored and scored[0][0] > 0:
                    seed = scored[0][1]
                    mode_seed = "voted"
            except Exception:
                pass
        if (seed is None
                and getattr(self, "_set_arc_enabled", False)
                and getattr(self, "_set_arc_phase", None)):
            phase_seed = self._phase_weighted_seed(
                pool, self._set_arc_phase
            )
            if phase_seed is not None:
                seed = phase_seed
                mode_seed = f"phase:{self._set_arc_phase}"
        if seed is None:
            seed = _random.choice(pool)
            mode_seed = "random"
        # CLIP neighbors first; random fallback if no embeddings.
        neighbors = self._clip_neighbors(seed, pool, top_k=subset_size - 1)
        if not neighbors:
            others = [p for p in pool if p != seed]
            neighbors = _random.sample(
                others, min(subset_size - 1, len(others))
            )
            mode = "random-fallback"
        else:
            mode = "clip"
        self._cohesion_subset = [seed] + neighbors
        self._cohesion_seed = seed
        from pathlib import Path as _P
        names = " · ".join(_P(p).name[:20] for p in self._cohesion_subset)
        logger.info(
            f"[cohesion] new anchor (seed={mode_seed}, "
            f"neigh={mode}, {subset_size} files): {names}"
        )

    def cohesion_status(self) -> dict:
        """Snapshot for iPad: current anchor subset + seconds left."""
        import time as _time
        from pathlib import Path as _P
        now = _time.time()
        remaining = max(0.0, self._cohesion_until - now)
        return {
            "enabled": bool(
                self.config.get("picker_cohesion_enabled", True)
            ),
            "subset": [
                {"path": p, "name": _P(p).name}
                for p in (self._cohesion_subset or [])
            ],
            "seed": self._cohesion_seed,
            "seconds_remaining": round(remaining, 1),
            "lock_seconds": int(
                self.config.get("picker_cohesion_lock_seconds", 90)
                or 90
            ),
        }

    def _match_cut_score(self, cur_path: str, cand_path: str,
                         cur_tags: set | None = None) -> float:
        """Pairwise A→B cut-continuity score in [0, 1].

        How cleanly does `cand` cut in after `cur`? A high score means
        motion character, visual complexity and geometry/symmetry
        carry across the cut — so the edit reads as *designed* rather
        than a random jump (the "kinetic whiplash" rookie mistake).

        Thematic / shoot-level similarity is already handled upstream
        by the cohesion anchor; this scores the per-cut KINETIC
        continuity on top. Tag-only (no CLIP embed) so it's cheap
        enough to run on every candidate at pick time.

        `cur_tags`: pre-fetched tag set for `cur_path`. The caller
        (`_pick_in_pool`) hoists this fetch out of its per-candidate
        loop — `cur` is constant across the loop, so re-querying it
        per candidate was 60-odd redundant DB hits per flip.

        Weights (motion .34 / complexity .36 / geometry .15 /
        symmetry .15) follow the VJ-craft research brief, dropping
        its CLIP-cosine term since the cohesion anchor covers it."""
        try:
            a = (cur_tags if cur_tags is not None
                 else self.path_tags.tags_for_file(cur_path)) or set()
            b = self.path_tags.tags_for_file(cand_path) or set()
        except Exception:
            return 0.5
        if not a or not b:
            return 0.5

        # ── motion-vector continuity ──
        a_act = {t for t in a
                 if t.startswith("action:") and not t.startswith("action:_")}
        b_act = {t for t in b
                 if t.startswith("action:") and not t.startswith("action:_")}
        if a_act and b_act and (a_act & b_act):
            motion = 1.0                       # same rhythmic action
        elif "motion:static" in a and "motion:static" in b:
            motion = 1.0                       # both still
        elif (("motion:jumpy" in a) and ("motion:static" in b)) or \
             (("motion:static" in a) and ("motion:jumpy" in b)):
            motion = 0.15                      # jumpy↔static = whiplash
        else:
            a_m = {t for t in a if t.startswith("motion:")}
            b_m = {t for t in b if t.startswith("motion:")}
            motion = 0.7 if (a_m & b_m) else 0.45

        # ── complexity-delta continuity ──
        def _complexity(tags: set) -> "int | None":
            for t in tags:
                if t.startswith("complexity:"):
                    try:
                        return int(t.split(":", 1)[1])
                    except (ValueError, IndexError):
                        return None
            return None
        ca, cb = _complexity(a), _complexity(b)
        if ca is not None and cb is not None:
            comp = 1.0 - abs(ca - cb) / 9.0
        else:
            comp = 0.6                         # neutral when untagged

        # ── geometry / symmetry continuity ──
        a_geo = {t for t in a if t.startswith("geometry:")}
        b_geo = {t for t in b if t.startswith("geometry:")}
        a_sym = {t for t in a if t.startswith("symmetry:")}
        b_sym = {t for t in b if t.startswith("symmetry:")}
        geo = 1.0 if (a_geo & b_geo) else 0.0
        sym = 1.0 if (a_sym & b_sym) else 0.0

        return (0.34 * motion + 0.36 * comp
                + 0.15 * geo + 0.15 * sym)

    def _pick_in_pool(self, direction: int = 1) -> "Path | None":
        """Pick the next (direction=+1) or previous (-1) pool entry
        relative to the currently-playing file.

        Random mode (self._random_next, toggled on FX1 Param 1) applies
        only to forward flips — backward is always sequential so `<`
        feels like a real step-back, not another roll of the dice.

        Random forward ALSO honors BPM-preference: if there are clips
        in the pool that match live BPM (with octave-awareness), the
        random pick is restricted to those. No matches / no live BPM /
        toggle off → falls back to whole-pool random.
        """
        pool = self._flip_pool()
        if not pool:
            return None
        cur_str = str(
            self._current_source_path
            or (self.player.current_file if self.player else "")
            or ""
        )
        cur = Path(cur_str) if cur_str else None
        with self._auto_flip_lock:
            random_next = self._random_next
        # ── COHESION ANCHOR (2026-05-19): treat the pool as the FULL
        # universe but pick within a smaller "music-video-shoot"
        # subset for 60-90s at a time. Falls back to full pool when
        # disabled or pool is small. Applies to forward+random only;
        # sequential `<`/`>` always uses the full pool so the user's
        # explicit nav isn't restricted.
        if direction > 0 and random_next:
            cohesion_pool = self._get_cohesion_subset(pool)
        else:
            cohesion_pool = pool
        # Forward + random → BPM-prefer when possible, exclude current
        if direction > 0 and random_next and len(pool) > 1:
            import random as _random
            matched = []
            live_bpm = 0.0
            if self._bpm_preference_enabled:
                try:
                    live_bpm = float(self.state.detected_bpm or 0.0)
                except Exception:
                    live_bpm = 0.0
                if live_bpm > 0:
                    matched = self._filter_pool_by_bpm(
                        cohesion_pool, live_bpm)
            # Restrict to BPM-matches if any; else cohesion subset.
            candidates = matched if matched else cohesion_pool
            # ── Stem-onset boost (2026-05-18 overnight) ──
            # If stem_daemon's last drum onset fired within the boost
            # window AND we have any dynamic-motion clips in the
            # candidate pool, prefer those for this pick. Falls back
            # cleanly when nothing matches or stem daemon is off.
            stem_restricted = False
            try:
                if time.time() < getattr(
                        self, "_stem_drum_boost_until", 0.0):
                    dyn = self._filter_pool_by_dynamic_motion(candidates)
                    if len(dyn) >= 3:
                        logger.info(
                            f"[stem-pick] kick-boost: restricting "
                            f"{len(candidates)} → {len(dyn)} "
                            f"dynamic-motion candidates"
                        )
                        candidates = dyn
                        stem_restricted = True
            except Exception as e:
                logger.debug(f"stem-pick filter failed: {e}")
            choices = [p for p in candidates if not cur or p != cur]
            if not choices:
                return pool[0]
            # Vote-weighted random pick: up-voted clips get more odds,
            # down-voted clips get fewer (but never zero). vote_store
            # returns 1.0 when no vote / when store is None.
            # Base weights = vote-store picker weight (1.0 if no store
            # or no vote on that clip).
            base_w = [1.0] * len(choices)
            if self.vote_store is not None:
                try:
                    base_w = [
                        self.vote_store.picker_weight(str(p))
                        for p in choices
                    ]
                except Exception as e:
                    logger.debug(f"vote-weight failed: {e}")
            vote_used = any(w != 1.0 for w in base_w)
            # MATCH-CUT CONTINUITY (2026-05-19): multiply each
            # candidate's weight by how cleanly it cuts from the
            # CURRENT clip — motion-vector + complexity-delta +
            # geometry/symmetry carry-over. A soft bias (score 0 still
            # gets ~0.35× weight, never excluded) so variety survives
            # but the sequence reads as designed cuts, not a shuffle.
            # Thematic similarity is already handled by the cohesion
            # anchor; this is the per-cut KINETIC continuity on top.
            match_cut_used = False
            weights = list(base_w)
            if cur and self.config.get("picker_match_cut_enabled", True):
                try:
                    # Fetch the current clip's tags ONCE — it's constant
                    # across the candidate loop (was re-queried per pick).
                    cur_tags = self.path_tags.tags_for_file(cur_str) or set()
                    weights = [
                        w * (0.35 + 0.65
                             * self._match_cut_score(
                                 cur_str, str(p), cur_tags=cur_tags))
                        for w, p in zip(base_w, choices)
                    ]
                    match_cut_used = True
                except Exception as e:
                    logger.debug(f"match-cut weighting failed: {e}")
                    weights = list(base_w)
            try:
                if weights and any(w != weights[0] for w in weights):
                    picked = _random.choices(
                        choices, weights=weights, k=1)[0]
                else:
                    picked = _random.choice(choices)
            except Exception as e:
                logger.debug(f"weighted pick failed: {e}")
                picked = _random.choice(choices)
            if matched:
                logger.info(f"[bpm-pick] live={live_bpm:.1f} matched "
                            f"{len(matched)}/{len(pool)} → {Path(picked).name}")
            if vote_used:
                logger.info(
                    f"[vote-pick] vote-weighted from "
                    f"{len(choices)} candidates → {Path(picked).name} "
                    f"(score={self.vote_store.score(str(picked))})"
                )
            # Record this pick for the iPad debug panel.
            reason_parts = ["random"]
            if matched:
                reason_parts.append(f"bpm-match {len(matched)}/{len(pool)}")
            if stem_restricted:
                reason_parts.append("stem-restrict")
            if match_cut_used:
                reason_parts.append("match-cut")
            self._picker_record(
                picked,
                " · ".join(reason_parts),
                {"kind": "random", "pool_size": len(pool),
                 "bpm_matched": len(matched) if matched else 0,
                 "stem_boost": stem_restricted,
                 "match_cut": match_cut_used},
            )
            return picked
        # Sequential (forward or back). User-driven navigation; don't
        # impose BPM filtering — they explicitly asked for next/prev.
        # If current file isn't in pool (e.g. just fired a deck slot
        # from S2), start at the head of the pool.
        try:
            idx = pool.index(cur) if cur else -1
        except ValueError:
            idx = -1
        if idx < 0:
            return pool[0] if direction > 0 else pool[-1]
        nxt_idx = (idx + direction) % len(pool)
        nxt = pool[nxt_idx]
        # ── Vote-aware sequential skip (vote v2, 2026-05-18) ──
        # Heavily-downvoted clips (score <= -3, picker_weight ~0.4 or
        # less) get skipped IN SEQUENTIAL MODE TOO. The user wanted
        # downvotes to suppress globally, not just in random. Keeps
        # backward `<` from landing on a clip they explicitly down-
        # voted out. Skips up to 3 in a row to avoid infinite loops
        # on a heavily-trashed bank.
        if self.vote_store is not None:
            try:
                for skip_attempt in range(3):
                    w = self.vote_store.picker_weight(str(nxt))
                    if w >= 0.5:
                        break  # acceptable, take it
                    # Skip — advance one more in same direction
                    nxt_idx = (nxt_idx + direction) % len(pool)
                    candidate = pool[nxt_idx]
                    logger.info(
                        f"[vote-pick] sequential skip "
                        f"{Path(nxt).name[:35]} (w={w:.2f}) -> "
                        f"{Path(candidate).name[:35]}"
                    )
                    nxt = candidate
            except Exception as e:
                logger.debug(f"vote-sequential skip failed: {e}")
        # ── Stem-onset boost in SEQUENTIAL mode (2026-05-18 overnight) ──
        # If a kick just hit AND we're heading to a non-dynamic-motion
        # clip AND the pool has a dynamic-motion alternative, jump
        # forward to the nearest dynamic clip instead. Keeps sequential
        # mostly-predictable while still letting kicks pull energy.
        # Only fires for FORWARD direction so backward < is exactly
        # what the operator asked for.
        try:
            if (direction > 0
                    and time.time() < getattr(
                        self, "_stem_drum_boost_until", 0.0)):
                dyn = self._filter_pool_by_dynamic_motion(pool)
                if dyn and nxt not in dyn:
                    # Find the closest dyn file *after* nxt_idx
                    # (or wrap around if needed).
                    dyn_set = set(dyn)
                    n = len(pool)
                    for off in range(1, n):
                        j = (nxt_idx + off) % n
                        if pool[j] in dyn_set:
                            jumped = pool[j]
                            logger.info(
                                f"[stem-pick] kick-jump: skipped "
                                f"{Path(nxt).name[:35]} -> "
                                f"{Path(jumped).name[:35]}"
                            )
                            self._picker_record(
                                jumped,
                                f"kick-jump (skipped {Path(nxt).name[:30]})",
                                {"kind": "kick-jump",
                                 "skipped": str(nxt),
                                 "pool_size": len(pool)},
                            )
                            return jumped
        except Exception as e:
            logger.debug(f"stem-pick sequential jump failed: {e}")
        # Plain sequential pick -- no stem influence
        self._picker_record(
            nxt,
            f"sequential +{direction}",
            {"kind": "sequential", "pool_size": len(pool)},
        )
        return nxt

    def flip(self):
        """Jump to next clip in the flip pool (active bank / scratch,
        falling back to folder siblings when bank is empty). Random or
        sequential per self._random_next (toggle on FX1 Param 1).

        Pool semantics changed 2026-05-16: was strictly "next file in
        same folder," now prefers the bank pool. > should fire whatever
        auto-flip would fire on the next beat. Folder cycling is still
        reachable via the FOLDER auto-flip mode."""
        if not self.player or not self.player.current_file:
            return
        nxt = self._pick_in_pool(direction=+1)
        if not nxt:
            return
        # Pre-load body seek so the first decoded frame IS the body,
        # not the title bumper. Same fix shipped for auto_flip path
        # + flip_back; needed here too for manual > / B PLAY / MK2 >.
        self._apply_preload_body_seek(str(nxt))
        # play() only if the load wasn't debounced (Audit fix H4).
        if self.load_video(str(nxt)):
            self.player.play()
        with self._auto_flip_lock:
            random_next = self._random_next
        self.state.record_s2_action("flip" + (" (random)" if random_next else ""))
        # Record flip timestamp for auto-set-arc detector (regardless
        # of whether auto mode is on -- having the history primed
        # means it works the instant the operator enables it).
        if getattr(self, "_auto_set_arc", None) is not None:
            try:
                self._auto_set_arc.record_flip()
            except Exception:
                pass

    def flip_random(self):
        """B SYNC: jump to a RANDOM sibling in the current folder, ignoring
        the _random_next mode flag — the 'shuffle now' button. Mirrors
        flip()'s random branch but is unconditional."""
        if not self.player or not self.player.current_file:
            return
        cur = Path(self._current_source_path or self.player.current_file)
        siblings = sorted(
            [p for p in cur.parent.iterdir()
             if p.suffix.lower() in (".mp4", ".mkv", ".mov", ".avi", ".webm")]
        )
        choices = [s for s in siblings if s != cur]
        if not choices:
            return
        import random as _random
        nxt = _random.choice(choices)
        # Pre-load body seek (same pattern as flip / flip_back)
        self._apply_preload_body_seek(str(nxt))
        if self.load_video(str(nxt)):
            self.player.play()
        self.state.record_s2_action("flip (random)")

    # Pinned BLACKOUT panic (B RESET). CLAUDE.md's "Pinned Peripheral" rule
    # wants a global kill that never moves — this is it.
    _blackout_active: bool = False
    _blackout_saved_brightness: int = 0

    def blackout_toggle(self):
        """B RESET: toggle the video to full black (mpv brightness -100)
        and back. Saves/restores whatever brightness was set before, so it
        composes with the B EQ HI brightness knob. While blacked out,
        set_master_brightness is a no-op — blackout is a hard override
        until you press B RESET again."""
        if not self.player:
            return
        try:
            if self._blackout_active:
                self.player.player.brightness = self._blackout_saved_brightness
                self._blackout_active = False
                self.state.set_message("Blackout OFF")
            else:
                try:
                    self._blackout_saved_brightness = int(
                        getattr(self.player.player, "brightness", 0) or 0)
                except Exception:
                    self._blackout_saved_brightness = 0
                self.player.player.brightness = -100
                self._blackout_active = True
                self.state.set_message("BLACKOUT")
        except Exception as e:
            logger.debug(f"blackout_toggle failed: {e}")

    def _predict_flip_target(self) -> str:
        """Predict what flip() would load WITHOUT loading it. Used by
        auto_flip to set the mpv ``start`` property BEFORE the load —
        kills the ~150ms title-frame flash that post-load seeking
        exposes. Now thin: delegates to _pick_in_pool so prediction and
        actual fire share the exact same logic (no drift)."""
        if not self.player or not self.player.current_file:
            return ""
        nxt = self._pick_in_pool(direction=+1)
        return str(nxt) if nxt else ""

    # (_auto_flip_deck_cursor + _predict_deck_target removed 2026-05-16
    # along with the DECKS auto-flip mode. Auto-flip now always cycles
    # the bank pool — per-deck fire stays on S2 pads exclusively.)

    def _apply_preload_body_seek(self, path: str) -> None:
        """Set mpv's `start` property to the body-seek target for `path`
        BEFORE the loadfile call so the first decoded frame IS the body,
        not the title bumper. Schedules an auto-reset 500ms out so
        unrelated loads aren't poisoned by the start property."""
        if not path or not self.player or not self.player.player:
            return
        try:
            seek_to = float(self._compute_body_seek_target(path))
        except Exception as e:
            logger.debug(f"pre-load body-seek compute failed: {e}")
            return
        if seek_to <= 0.5:
            return
        try:
            self.player.player.start = float(seek_to)
            self._remember_auto_seek(path, seek_to)
            QTimer.singleShot(500, self._reset_mpv_start_option)
        except Exception as e:
            logger.debug(f"mpv start preset failed: {e}")

    def auto_flip(self):
        """Audio-reactive flip — fires on detected beat (gated by HOLD,
        debounce, and audio_reactive_enabled in _on_beat).

        Pool model (reworked 2026-05-16):
          - BANK mode (default): cycles the active bank / scratch pool.
            Falls back to folder siblings only when the bank has ≤1 file.
          - FOLDER mode: forces folder siblings of the current file.
          - Deck cycling is GONE — deck firing belongs to the S2 pads
            exclusively. The user wants flips tied to banks; decks stay
            S2-driven.

        Manual > (MK2 bit 34 / B PLAY) calls self.flip() directly which
        uses the same _pick_in_pool helper — so > fires whatever auto-flip
        would have fired on the next beat (predictable manual step).
        """
        if not self.player or not self.player.current_file:
            return
        # AUDIT FIX (code-review agent, 2026-05-16 night): the old code
        # called _predict_flip_target() to LOG the target, then called
        # flip() which re-rolls a DIFFERENT random pick. Log lied.
        # Now: just announce mode + random flag; flip()'s own
        # _pick_in_pool logs the actual pick via [bpm-pick] / similar.
        with self._auto_flip_lock:
            use_folder = self._auto_flip_use_folder
            random_next = self._random_next
        mode = "FOLDER" if use_folder else "BANK"
        logger.info(f"AUTO_FLIP {mode} (random={random_next}) — picking…")
        # flip() applies the preload body-seek itself; no need to repeat.
        self.flip()

    # NOTE: _last_auto_seek_per_file is an INSTANCE attr set in __init__
    # (Audit fix H10). _remember_auto_seek caps it so it can't leak.
    _AUTO_SEEK_MEMORY_CAP = 256

    def _remember_auto_seek(self, path: str, target: float) -> None:
        """Record the last auto-seek target for a file, capping the dict
        so a long session over a big library can't leak memory. (Audit
        fix H10.) When full, drop the oldest insertion (dict preserves
        insertion order in CPython 3.7+)."""
        if not path:
            return
        d = self._last_auto_seek_per_file
        # Re-insert so a re-seek of an existing file moves it to newest.
        d.pop(path, None)
        d[path] = target
        if len(d) > self._AUTO_SEEK_MEMORY_CAP:
            try:
                oldest = next(iter(d))
                del d[oldest]
            except (StopIteration, KeyError):
                pass

    def _auto_seek_into_body(self):
        """Seek into the body of the just-flipped clip.

        Hard rules:
        - Skip the first 15s (intros are almost always slow build-up)
        - Skip the last 20s (outros / endings)
        - Force the seek to land ≥ 1/4 of the playable duration away
          from where we landed last time on this file → so each visit
          covers a different chunk instead of always the same middle.
        - Short clips: if the 15+20s floors don't fit, just skip the
          intro if possible; bail entirely on micro-clips."""
        if not self.player:
            return
        # If the user grabbed the jog wheel in the ~150ms between the flip
        # and this deferred seek, don't override their hand-scrub position.
        if self._jog_touched:
            return
        try:
            duration = float(self.player.get_duration() or 0.0)
            intro_skip = 15.0
            outro_skip = 20.0
            cur_file = self.player.current_file or ""
            # Short clip: can't honor both floors. Best effort.
            if duration < intro_skip + outro_skip + 5.0:
                if duration > intro_skip + 2.0:
                    self.player.seek(intro_skip)
                return
            playable_start = intro_skip
            playable_end = duration - outro_skip
            playable_dur = playable_end - playable_start
            min_distance = playable_dur / 4.0
            import random as _random
            prev = self._last_auto_seek_per_file.get(cur_file, -999.0)
            target = _random.uniform(playable_start, playable_end)
            # Try up to 5 picks to find one far enough from last visit.
            for _ in range(5):
                if abs(target - prev) >= min_distance:
                    break
                target = _random.uniform(playable_start, playable_end)
            self._remember_auto_seek(cur_file, target)
            self.player.seek(target)
        except Exception as e:
            logger.debug(f"_auto_seek_into_body failed: {e}")

    def _loop_is_active(self) -> bool:
        """True when a saved-clip A-B loop is currently engaged in mpv.
        mpv returns either a number, the string "no", or sometimes a
        stringified number depending on the python-mpv version. Be lenient."""
        if not self.player:
            return False
        try:
            a = self.player.player.ab_loop_a
            b = self.player.player.ab_loop_b
        except Exception:
            return False
        return self._is_loop_value(a) and self._is_loop_value(b)

    @staticmethod
    def _is_loop_value(v) -> bool:
        """Decide whether an ab-loop-{a,b} property reading represents an
        active loop bound (a number) vs disabled (None / 'no' / 0)."""
        if v is None or v is False:
            return False
        if isinstance(v, str):
            if v.strip().lower() == "no":
                return False
            try:
                return float(v) > 0
            except ValueError:
                return False
        if isinstance(v, (int, float)):
            return v > 0
        return False

    def _loop_get_bounds(self):
        """Return (a, b) as floats if loop is active, else None."""
        if not self.player:
            return None
        try:
            a = self.player.player.ab_loop_a
            b = self.player.player.ab_loop_b
        except Exception:
            return None
        if not (self._is_loop_value(a) and self._is_loop_value(b)):
            return None
        try:
            return (float(a), float(b))
        except (TypeError, ValueError):
            return None

    def _quantize_loop_end(self, in_sec: float, raw_out_sec: float) -> tuple[float, str]:
        """Snap a loop OUT so the duration is a whole-beat multiple of IN
        when BPM is known. Returns (snapped_out, label). Label is empty
        when no quantize was applied."""
        bpm = float(self.state.detected_bpm or 0)
        raw_dur = raw_out_sec - in_sec
        if bpm < 30 or bpm > 250 or raw_dur <= 0:
            return raw_out_sec, ""
        beat_sec = 60.0 / bpm
        beats = max(1, round(raw_dur / beat_sec))
        snapped = in_sec + beats * beat_sec
        return snapped, f" [Q→{beats}b @ {bpm:.0f}]"

    def _quantize_loop_start(self, raw_in_sec: float, out_sec: float) -> tuple[float, str]:
        """Snap a loop IN so the duration is a whole-beat multiple, anchored
        on OUT. Returns (snapped_in, label)."""
        bpm = float(self.state.detected_bpm or 0)
        raw_dur = out_sec - raw_in_sec
        if bpm < 30 or bpm > 250 or raw_dur <= 0:
            return raw_in_sec, ""
        beat_sec = 60.0 / bpm
        beats = max(1, round(raw_dur / beat_sec))
        snapped = max(0.0, out_sec - beats * beat_sec)
        return snapped, f" [Q→{beats}b @ {bpm:.0f}]"

    def mark_in(self):
        if not self.player:
            return
        pos = self.player.get_position()
        # When a loop is active: REPOSITION the loop start to current
        # playback (live loop adjustment). When no loop: mark a new IN
        # for clip-saving via mark_out.
        if self._loop_is_active():
            bounds = self._loop_get_bounds()
            cur_b = bounds[1] if bounds else pos
            new_a, qlabel = self._quantize_loop_start(pos, cur_b)
            try:
                self.player.player.ab_loop_a = float(new_a)
                self.state.set_message(f"Loop IN → {new_a:.2f}s{qlabel}")
            except Exception as e:
                logger.debug(f"adjust loop_in failed: {e}")
            self.state.record_s2_action("loop_in_adjust")
            if self.s2:
                self.s2.flash_led("a_loop_in")
            return
        result = self.clips_db.mark_in(self.player.current_file or "", pos)
        self.state.set_pending_in(pos if result["ok"] else None)
        self.state.record_s2_action("mark_in")
        self.state.set_message(result["message"])
        if self.s2:
            self.s2.flash_led("a_loop_in")

    def mark_out(self):
        if not self.player:
            return
        pos = self.player.get_position()
        # When a loop is active: REPOSITION the loop end. Same pattern as
        # mark_in above — adjusts the running loop instead of saving a clip.
        if self._loop_is_active():
            bounds = self._loop_get_bounds()
            cur_a = bounds[0] if bounds else 0.0
            new_b, qlabel = self._quantize_loop_end(cur_a, pos)
            try:
                self.player.player.ab_loop_b = float(new_b)
                self.state.set_message(f"Loop OUT → {new_b:.2f}s{qlabel}")
            except Exception as e:
                logger.debug(f"adjust loop_out failed: {e}")
            self.state.record_s2_action("loop_out_adjust")
            if self.s2:
                self.s2.flash_led("a_loop_out")
            return
        # Quantize the new clip's OUT to whole beats from the pending IN
        # when BPM is known. DJ convention — keeps loops on the grid.
        in_sec = self.state.pending_in
        if in_sec is not None:
            snapped, _ = self._quantize_loop_end(float(in_sec), pos)
            pos = snapped
        clip_name = f"clip_{len(self.clips_db.get_all_clips())}"
        cap_speed, cap_bpm = self._capture_speed_bpm()
        result = self.clips_db.mark_out(
            self.player.current_file or "", pos, clip_name,
            capture_speed=cap_speed, capture_bpm=cap_bpm,
        )
        if result["ok"]:
            self.state.set_pending_in(None)
            self._apply_active_tag(result.get("clip") or {})
            self._apply_ai_tags(result.get("clip") or {})
            self._refresh_clips_for_ipad()
            # If any deck slot already references this clip id, regenerate
            # its filmstrip in the background. (No-op if no deck matches.)
            saved = result.get("clip") or {}
            sid = saved.get("id")
            if sid:
                for i, d in enumerate(self.decks_store.all()):
                    if d and d.get("source_type") == "clip" and d.get("source_id") == sid:
                        self.decks_store.regenerate_filmstrip_async(i)
        self.state.record_s2_action("mark_out")
        self.state.set_message(result["message"], error=not result["ok"])
        if self.s2:
            self.s2.flash_led("a_loop_out")

    def save_loop_as_clip(self):
        """SHIFT + Loop Out: snapshot the currently-active A-B loop as a
        saved clip. No remarking — uses the live ab_loop_a / ab_loop_b
        bounds you've already dialed in via halve/double + adjust. Auto-
        applies the sticky active tag if one is set."""
        if not self.player:
            logger.info("SAVE_LOOP: no player — abort")
            return
        bounds = self._loop_get_bounds()
        logger.info(f"SAVE_LOOP bounds={bounds} active_tag={self.state.active_tag!r}")
        if bounds is None:
            self.state.set_message("No active loop to save", error=True)
            return
        in_s, out_s = bounds
        cur_file = self.player.current_file or ""
        if not cur_file:
            self.state.set_message("No video loaded", error=True)
            return
        # Stash IN, then mark OUT — same DB path as the regular flow so
        # thumbnail + BPM analysis hooks fire identically.
        self.clips_db.mark_in(cur_file, float(in_s))
        clip_name = f"loop_{len(self.clips_db.get_all_clips())}"
        cap_speed, cap_bpm = self._capture_speed_bpm()
        result = self.clips_db.mark_out(
            cur_file, float(out_s), clip_name,
            capture_speed=cap_speed, capture_bpm=cap_bpm,
        )
        if result.get("ok"):
            self._apply_active_tag(result.get("clip") or {})
            self._apply_ai_tags(result.get("clip") or {})
            self._refresh_clips_for_ipad()
            tag_note = f" #{self.state.active_tag}" if self.state.active_tag else ""
            self.state.set_message(
                f"Saved loop {in_s:.2f}-{out_s:.2f}s ({out_s - in_s:.2f}s){tag_note}"
            )
        else:
            self.state.set_message(result.get("message", "save failed"), error=True)
        self.state.record_s2_action("save_loop")
        if self.s2:
            self.s2.flash_led("a_loop_out")

    def _capture_speed_bpm(self) -> tuple[float, float]:
        """Snapshot the live playback speed and detected BPM right now.
        Used at clip-save time so the firing path can restore them."""
        speed = 1.0
        try:
            if self.player and self.player.player:
                speed = float(self.player.player.speed or 1.0)
        except Exception:
            pass
        bpm = 0.0
        try:
            bpm = float(self.state.detected_bpm or 0.0)
        except Exception:
            pass
        return speed, bpm

    def _restore_clip_context(self, clip: dict) -> None:
        """Apply a saved clip's capture_speed (and log capture_bpm) when
        firing. Safe no-op for legacy clips (capture_speed defaults to 1.0
        via _migrate_clip)."""
        if not self.player or not clip:
            return
        try:
            cs = float(clip.get("capture_speed") or 1.0)
            cs = max(0.25, min(3.0, cs))
            self.player.player.speed = cs
            cb = float(clip.get("capture_bpm") or 0.0)
            note = f" (was {cb:.0f} BPM)" if cb > 0 else ""
            if abs(cs - 1.0) > 0.01:
                self.state.set_message(f"Fire: speed → {cs:.2f}x{note}")
            # Engage soft-takeover so the live tempo fader doesn't yank
            # the speed away on the next jitter.
            self._tempo_soft_takeover_lockout = True
        except Exception as e:
            logger.debug(f"_restore_clip_context failed: {e}")

    def _apply_ai_tags(self, clip: dict) -> None:
        """Fire-and-forget AI vision tagging on a newly-saved clip's
        thumbnail. Returned tags are merged into the clip's tag list
        when the background callback fires. No-op if ai_tagger is
        disabled, no thumbnail yet, or anything goes sideways."""
        if not self.ai_tagger or not clip:
            return
        cid = clip.get("id")
        if not cid:
            return
        try:
            import thumbnails as _th
            thumb_path = _th.thumbnail_path(cid)
            if not Path(thumb_path).exists():
                return
        except Exception as e:
            logger.debug(f"_apply_ai_tags: thumbnail lookup failed: {e}")
            return

        def _on_tags_ready(tags):
            # Worker thread → Qt thread marshalling. Enqueue (not single-
            # slot) so concurrent tagger callbacks don't clobber each other.
            if self._ai_tag_queue is None:
                self._ai_tag_queue = queue.Queue()
            self._ai_tag_queue.put((cid, list(tags or [])))
            QTimer.singleShot(0, self._apply_pending_ai_tags)

        try:
            self.ai_tagger.tag_image_async(str(thumb_path), _on_tags_ready)
        except Exception as e:
            logger.debug(f"_apply_ai_tags: tag_image_async failed: {e}")

    _ai_tag_queue: "queue.Queue" = None

    def _apply_pending_ai_tags(self):
        if self._ai_tag_queue is None:
            return
        try:
            cid, new_tags = self._ai_tag_queue.get_nowait()
        except queue.Empty:
            return
        if not new_tags:
            return
        try:
            # Merge with existing tags; clips_db handles normalization + dedup.
            clip = None
            for c in self.clips_db.get_all_clips() or []:
                if c.get("id") == cid:
                    clip = c
                    break
            if not clip:
                return
            existing = list(clip.get("tags") or [])
            merged = list(dict.fromkeys(existing + list(new_tags)))  # preserve order, dedup
            self.clips_db.set_tags(cid, merged)
            self._refresh_clips_for_ipad()
            logger.info(f"AI tags applied to {cid[:8]}: {new_tags}")
        except Exception as e:
            logger.debug(f"_apply_pending_ai_tags failed: {e}")

    def _apply_active_tag(self, clip: dict) -> None:
        """If a sticky active tag is set, attach it to a freshly-saved clip."""
        tag = (self.state.active_tag or "").strip()
        if not tag or not clip:
            return
        cid = clip.get("id")
        if not cid:
            return
        try:
            self.clips_db.set_tags(cid, [tag])
        except Exception as e:
            logger.debug(f"_apply_active_tag failed: {e}")

    def set_active_tag(self, tag: str):
        """iPad: set the sticky tag that auto-applies to every save."""
        logger.info(f"SET_ACTIVE_TAG: {tag!r}")
        self.state.set_active_tag(tag or "")
        if (tag or "").strip():
            self.state.set_message(f"Active tag: #{tag.strip()}")
        else:
            self.state.set_message("Active tag cleared")
        return {"ok": True, "active_tag": self.state.active_tag}

    def play_clip(self, idx: int):
        """Fire the Nth MARKED SEGMENT on the current video and LOOP
        between its IN/OUT marks. Marked segments are user-defined via
        the iPad Mark Clip form. NOT the same as a video file —
        "clip" here = a sub-region of the current playing video.

        Fallback: if the current video has no marked segments, treat
        the call as "fire item N from the active bank" (the user's
        intuitive "tap a clip slot" expectation when they hit S2
        A-pad N / Stream Deck slot N)."""
        cur_file = self.player.current_file if self.player else ""
        per_video = self.clips_db.get_clips_for_file(cur_file)
        if not per_video or idx >= len(per_video):
            # FALLBACK: no marked segment N on this video → fire item N
            # of active bank's scratch list. This makes the gesture
            # always do SOMETHING useful instead of just a red banner.
            try:
                scratch = self.scratch_store.all() or []
                if scratch and idx < len(scratch):
                    target = scratch[idx]
                    self.state.set_message(
                        f"No marked segment {idx} — firing bank slot {idx}"
                    )
                    self.load_video(target)
                    return
            except Exception as e:
                logger.debug(f"play_clip fallback failed: {e}")
            # Truly nothing to do — quiet info, not red ERROR.
            self.state.set_message(
                f"No marked segment {idx} on this video "
                "(set via Mark Clip form on iPad)"
            )
            return
        clip = per_video[idx]
        try:
            if self.player:
                # Set A-B loop on the clip's marks, then seek + play
                in_s = float(clip["in_seconds"])
                out_s = float(clip["out_seconds"])
                try:
                    # Direct property assignment with native floats — the
                    # command("set", ..., str) form gives MPV_ERROR_INVALID_PARAMETER
                    # because mpv wants a number for numeric properties.
                    self.player.player.ab_loop_a = in_s
                    self.player.player.ab_loop_b = out_s
                    a = self.player.player.ab_loop_a
                    b = self.player.player.ab_loop_b
                    logger.info(f"PLAY_CLIP set loop in={in_s} out={out_s} → mpv reports a={a!r} b={b!r}")
                except Exception as e:
                    logger.info(f"set ab-loop failed: {e}")
                self.player.seek(in_s)
                self.player.play()
                # Restore the playback speed the clip was captured at
                # (no-op for legacy clips — defaults to 1.0).
                self._restore_clip_context(clip)
            self.state.record_s2_action(f"clip:{idx}")
            # Bump play stats — used for the iPad's "Most Played" /
            # "Recently Played" sort modes. Best-effort: never let a
            # stat update break the actual fire.
            cid = clip.get("id")
            if cid:
                try:
                    self.clips_db.bump_play_count(cid)
                    # Re-publish so the iPad sees the new count immediately
                    # (without waiting for the next mark_in/mark_out cycle).
                    self._refresh_clips_for_ipad()
                except Exception as e:
                    logger.debug(f"bump_play_count failed for {cid}: {e}")
        except Exception as e:
            logger.error(f"Play clip {idx} failed: {e}")

    def delete_clip(self, clip_id: str):
        """Remove a saved clip by its uuid."""
        result = self.clips_db.delete_clip_by_id(clip_id)
        self.state.set_message(result["message"], error=not result["ok"])
        self._refresh_clips_for_ipad()
        return result

    def fire_clip_by_id(self, clip_id: str) -> dict:
        """Bank: fire any saved clip by id. If the clip is on the current
        video, plays it directly (sets A-B loop). If it's on a different
        video, loads that video first, then sets up the loop and seeks
        to the IN. Updates play_count + recency stats."""
        if not clip_id:
            return {"ok": False, "message": "no clip_id"}
        clip = None
        for c in self.clips_db.get_all_clips() or []:
            if c.get("id") == clip_id:
                clip = c
                break
        if clip is None:
            return {"ok": False, "message": "clip not found"}
        target_file = clip.get("filepath") or ""
        cur_file = self.player.current_file if self.player else ""
        # If we're on the same video, just use the existing path so play_count
        # and message handling are uniform.
        if target_file == cur_file:
            per_video = self.clips_db.get_clips_for_file(cur_file) or []
            for i, c in enumerate(per_video):
                if c.get("id") == clip_id:
                    self.play_clip(i)
                    return {"ok": True, "message": "fired"}
        # Different video: load it, then set the A-B loop on the new file.
        # force=True — a clip fire is an explicit user action and must
        # never be silently debounced (Audit fix H4).
        try:
            in_s = float(clip.get("in_seconds") or 0.0)
            out_s = float(clip.get("out_seconds") or in_s + 1.0)
            if not self.load_video(target_file, force=True):
                return {"ok": False, "message": "load failed"}
            if self.player:
                try:
                    self.player.player.ab_loop_a = in_s
                    self.player.player.ab_loop_b = out_s
                except Exception as e:
                    logger.debug(f"bank set ab-loop failed: {e}")
                self.player.seek(in_s)
                self.player.play()
                # Restore captured speed (no-op for legacy clips).
                self._restore_clip_context(clip)
            try:
                self.clips_db.bump_play_count(clip_id)
                self._refresh_clips_for_ipad()
            except Exception as e:
                logger.debug(f"bank bump_play_count failed: {e}")
            self.state.set_message(f"Bank: {clip.get('name') or clip_id[:8]}")
            return {"ok": True, "message": "fired (cross-video)"}
        except Exception as e:
            logger.error(f"fire_clip_by_id failed: {e}")
            return {"ok": False, "message": str(e)}

    # ── Clip metadata helpers (star + tags) ──────────────────────────────
    # Each wraps a clips_db call and re-publishes the per-video clip list
    # + global tag union, so the iPad reconciles on its next 500ms poll.
    # We don't push status messages on every star toggle (would spam the
    # status panel during a tagging session) — only on tag operations.

    def clip_set_starred(self, clip_id: str, value: bool) -> dict:
        result = self.clips_db.set_starred(clip_id, value)
        if result.get("ok"):
            self._refresh_clips_for_ipad()
        else:
            self.state.set_message(result.get("message", "star failed"), error=True)
        return result

    def clip_set_tags(self, clip_id: str, tags) -> dict:
        result = self.clips_db.set_tags(clip_id, tags)
        if result.get("ok"):
            self._refresh_clips_for_ipad()
            self.state.set_message(f"Tags: {' '.join(result.get('tags') or []) or '(none)'}")
        else:
            self.state.set_message(result.get("message", "tags failed"), error=True)
        return result

    def clip_add_tag(self, clip_id: str, tag: str) -> dict:
        result = self.clips_db.add_tag(clip_id, tag)
        if result.get("ok"):
            self._refresh_clips_for_ipad()
            self.state.set_message(f"+#{tag.lower()}")
        else:
            self.state.set_message(result.get("message", "add tag failed"), error=True)
        return result

    def clip_remove_tag(self, clip_id: str, tag: str) -> dict:
        result = self.clips_db.remove_tag(clip_id, tag)
        if result.get("ok"):
            self._refresh_clips_for_ipad()
        else:
            self.state.set_message(result.get("message", "remove tag failed"), error=True)
        return result

    def clip_get_bpm(self, clip_id: str) -> dict:
        """Return the current BPM (or 0 if not yet analysed) for a clip,
        plus an `analyzing` flag the iPad can use to drive a spinner.
        bpm == 0 implies analysis is either pending or has failed; the
        iPad treats it the same way (no badge until non-zero arrives)."""
        clip = self.clips_db._find_by_id(clip_id)
        if not clip:
            return {"ok": False, "error": f"Clip {clip_id} not found"}
        bpm = float(clip.get("bpm") or 0.0)
        return {
            "ok": True,
            "clip_id": clip_id,
            "bpm": bpm,
            "analyzing": bpm == 0.0,
        }

    def clip_reanalyze_bpm(self, clip_id: str) -> dict:
        """Kick off a fresh BPM analysis for a clip — drops the cached
        result first so even an identical (file, in, out) re-runs."""
        result = self.clips_db.reanalyze_bpm(clip_id)
        if result.get("ok"):
            # Clear the badge optimistically so the iPad sees "analyzing"
            # on its very next poll instead of the stale value lingering.
            self._refresh_clips_for_ipad()
            self.state.set_message("Re-analysing BPM...")
        else:
            self.state.set_message(result.get("message", "reanalyze failed"), error=True)
        return result

    @staticmethod
    def _file_entry_name(entry) -> str:
        """library_files is now list[dict] {name, hash, has_thumb}.
        This helper tolerates the legacy list[str] shape too so the S2
        / iPad helpers don't need to know."""
        if isinstance(entry, dict):
            return str(entry.get("name") or "")
        return str(entry or "")

    # Browse-encoder acceleration: a relative encoder at one-detent-one-file
    # is glacial over a multi-thousand-file library. Track the gap between
    # ticks — a fast spin scales the step up to ~8 files/tick to cover
    # ground, a slow deliberate turn stays 1:1 for precise landing.
    _browse_last_ns: int = 0
    _BROWSE_FAST_NS = 70_000_000   # ticks <70ms apart = "spinning fast"

    def library_cursor_move(self, delta: int):
        """Browse encoder turn: move the library selection cursor.

        Runs on the S2 action-worker thread. Velocity-accelerated (see
        above). AppState.move_library_cursor does the len()/index/clamp/
        store under one lock so the cursor can't land out of range because
        _publish_library swapped the file list mid-move. (Audit fix H8.)"""
        now = time.perf_counter_ns()
        gap = now - self._browse_last_ns
        self._browse_last_ns = now
        step = delta
        if 0 < gap < self._BROWSE_FAST_NS:
            # Scale up to ~8x as the inter-tick gap shrinks toward zero.
            mult = 1 + int(7 * (self._BROWSE_FAST_NS - gap) / self._BROWSE_FAST_NS)
            step = delta * mult
        snap = self.state.move_library_cursor(step)
        files = snap["files"]
        new_idx = snap["selected_idx"]
        if new_idx < 0 or not files:
            return
        logger.info(f"BROWSE delta={delta:+d} step={step:+d} → idx={new_idx}/{len(files)}")
        self.state.set_message(
            f"Browse [{new_idx + 1}/{len(files)}]: "
            f"{self._file_entry_name(files[new_idx])}"
        )

    def _selected_library_path(self) -> Optional[str]:
        """Resolve current cursor → absolute filepath inside library_root.

        Returns None when the cursor is on a navigation entry (folder
        or '..') — those aren't loadable files. The dispatch in
        fire_selected_library_file handles the navigation case
        separately; LOAD/SAMPLES buttons just no-op on a folder.

        Snapshots files + idx + folder together under the lock so the
        three reads can't come from different library generations. (H8.)"""
        snap = self.state.get_library_snapshot()
        files = snap["files"]
        idx = snap["selected_idx"]
        if idx < 0 or idx >= len(files):
            return None
        entry = files[idx]
        if isinstance(entry, dict) and entry.get("_kind") in ("up", "folder"):
            return None
        folder = snap["folder"]
        if not folder:
            return None
        name = self._file_entry_name(entry)
        if not name:
            return None
        full = Path(folder) / name
        if not full.exists():
            return None
        return str(full)

    def load_selected_to_deck(self, slot: int):
        """A LOAD / B LOAD / A SAMPLES / B SAMPLES → push the currently
        browse-selected library file into preview deck `slot` (0..3)."""
        path = self._selected_library_path()
        if not path:
            self.state.set_message("Browse a file first", error=True)
            return
        # Use existing load_file_to_deck (created by deck agent).
        # Snapshot files+idx together (Audit fix H8) so the filename we
        # pass matches the path _selected_library_path just resolved.
        snap = self.state.get_library_snapshot()
        files = snap["files"]
        idx = snap["selected_idx"]
        filename = self._file_entry_name(files[idx]) if 0 <= idx < len(files) else ""
        try:
            self.load_file_to_deck(slot, filename)
        except Exception as e:
            logger.error(f"load_selected_to_deck({slot}) failed: {e}")
            self.state.set_message(f"Load failed: {e}", error=True)

    def fire_selected_library_file(self):
        """Browse encoder PRESS — context-aware:
          - on a file:   load it to LIVE
          - on a folder: navigate INTO it
          - on '..':     navigate UP to the parent folder
        The cursor walks a unified [up + subfolders + files] list so
        the encoder can reach every part of the library without the
        iPad."""
        snap = self.state.get_library_snapshot()
        files = snap["files"]
        idx = snap["selected_idx"]
        if idx < 0 or idx >= len(files):
            self.state.set_message("Browse first", error=True)
            return
        entry = files[idx]
        kind = (entry.get("_kind") if isinstance(entry, dict) else None) or "file"
        if kind == "up":
            self._navigate_lib_up()
            return
        if kind == "folder":
            target = entry.get("_target") or ""
            if target:
                self._navigate_lib_into(target)
            return
        # file
        path = self._selected_library_path()
        if not path:
            self.state.set_message("Browse a file first", error=True)
            return
        self.load_video(path)
        if self.player:
            self.player.play()

    def _navigate_lib_up(self):
        """Browse up one folder. Stays clamped to library_root."""
        snap = self.state.get_library_snapshot()
        folder_str = snap.get("folder") or ""
        root_str = snap.get("root") or ""
        if not folder_str or not root_str:
            return
        folder = Path(folder_str)
        root = Path(root_str)
        try:
            if folder == root:
                return  # already at root
            parent = folder.parent
            # Don't escape above root.
            if root not in parent.parents and parent != root:
                parent = root
            self._publish_library(root, parent)
            self.state.set_message(f"↑ {parent.name or root.name}")
        except Exception as e:
            logger.debug(f"_navigate_lib_up failed: {e}")

    def _navigate_lib_to(self, folder_path: str):
        """Jump library browser directly to an absolute folder. Used by
        the MK2 'folder:' button actions for favorite-folder shortcuts.
        Computes the right `root` for _publish_library — if the target
        is inside the current library_root we keep that root; otherwise
        we use the target itself as the new root."""
        if not folder_path:
            return
        target = Path(folder_path)
        if not target.is_dir():
            self.state.set_message(
                f"folder not found: {target.name}", error=True)
            return
        # Pick a sensible root — current root if target is below it, else
        # the target's own parent so the browser has somewhere to navigate.
        snap = self.state.get_library_snapshot()
        cur_root_str = snap.get("root") or ""
        if cur_root_str:
            cur_root = Path(cur_root_str)
            try:
                target.relative_to(cur_root)
                root = cur_root
            except ValueError:
                root = target
        else:
            root = target
        try:
            self._publish_library(root, target)
            self.state.set_message(f"→ {target.name}")
        except Exception as e:
            logger.debug(f"_navigate_lib_to failed: {e}")

    def _navigate_lib_into(self, subname: str):
        """Browse into the named subfolder of the current folder."""
        snap = self.state.get_library_snapshot()
        folder_str = snap.get("folder") or ""
        root_str = snap.get("root") or ""
        if not folder_str or not root_str or not subname:
            return
        try:
            target = Path(folder_str) / subname
            if not target.is_dir():
                self.state.set_message(f"not a folder: {subname}", error=True)
                return
            self._publish_library(Path(root_str), target)
            self.state.set_message(f"↓ {subname}")
        except Exception as e:
            logger.debug(f"_navigate_lib_into failed: {e}")

    def _load_selected_to_preview_deck(self):
        """SHIFT + browse encoder PRESS = load browse-selected library file
        into whichever deck is currently the crossfade preview source
        (FX2 cluster picks it). Safe alternative to plain browse-press
        when SHIFT happens to be held."""
        slot = int(self.state.preview_deck_idx or 0)
        self.load_selected_to_deck(slot)

    def clear_current_video_clips(self):
        """Drop all clips marked on the current video."""
        cur_file = self.player.current_file if self.player else ""
        if not cur_file:
            return {"ok": False, "message": "no video loaded"}
        result = self.clips_db.clear_clips_for_file(cur_file)
        self.state.set_message(result["message"], error=not result["ok"])
        self._refresh_clips_for_ipad()
        return result

    def audio_reactive_start(self):
        """A Load: toggle audio-reactive on/off (one button = one state change)."""
        if not self.audio_reactive:
            self.state.set_message("Audio reactive not available", error=True)
            return
        # Toggle: if already on, treat A Load as a disable.
        if self.state.audio_reactive_enabled:
            self.audio_reactive_stop()
            return
        # Reset flip debounce so first beat after enabling can fire.
        self._last_flip_beat_time = 0.0
        ok, msg = self.audio_reactive.start()
        logger.info(f"[autoflip] audio_reactive.start() ok={ok} msg={msg!r}")
        self.state.set_audio_reactive(
            enabled=ok,
            sensitivity=self.state.audio_sensitivity,
        )
        self.state.set_message(msg, error=not ok)
        if not ok:
            self.state.set_detected_bpm(0.0)
        self._refresh_s2_leds()

    def audio_reactive_stop(self):
        if self.audio_reactive:
            self.audio_reactive.stop()
        self.state.set_audio_reactive(
            enabled=False,
            sensitivity=self.state.audio_sensitivity,
        )
        self.state.set_detected_bpm(0.0)
        self.state.set_message("Audio-reactive stopped")
        self._refresh_s2_leds()

    # HTTP wrappers for audio-reactive — return a tiny status payload
    # so an external client (iPad / curl / shell script) can verify the
    # state change took. (2026-05-17 user req: "is there a place I can
    # hit to get just whatever is playing to live bpm".)
    def _http_ar_start(self) -> dict:
        self.audio_reactive_start()
        return {"ok": True, "enabled": bool(self.state.audio_reactive_enabled)}

    def _http_ar_stop(self) -> dict:
        self.audio_reactive_stop()
        return {"ok": True, "enabled": False}

    def _http_ar_toggle(self) -> dict:
        if self.state.audio_reactive_enabled:
            self.audio_reactive_stop()
        else:
            self.audio_reactive_start()
        return {"ok": True, "enabled": bool(self.state.audio_reactive_enabled)}

    def _http_bpm(self) -> dict:
        """Slim single-purpose endpoint: just return the current
        detected BPM (or 0.0 if detector off). Avoids parsing the
        whole /api/state JSON when all you want is the number."""
        try:
            bpm = float(self.state.detected_bpm or 0.0)
        except Exception:
            bpm = 0.0
        return {
            "ok": True,
            "bpm": bpm,
            "enabled": bool(self.state.audio_reactive_enabled),
        }

    def _volume_from_fader(self, deck: str, value: int):
        """Whichever volume fader the user just moved sets the volume.
        (Max-of-A-and-B was wrong — if B was at top from rest, A did
        nothing because B always won.)"""
        self.set_volume(value)

    def set_volume(self, value: int):
        # Linear from calibrated 0-127 fader to 0-100 mpv volume.
        # Floor at 1 — libmpv volume=0 can put WASAPI to sleep on Windows.
        v = int((value / 127.0) * 100)
        v = max(1, min(100, v))
        if self.player:
            self.player.set_volume(v)
            try:
                self.player.player.mute = False
            except Exception:
                pass
        self.state.set_volume(v)
        self.config["volume"] = v

    # Soft-takeover: after SYNC reset_speed() snaps to 1.0x, ignore the
    # tempo fader until it physically passes through the learned center.
    # Otherwise the fader's current non-center position immediately yanks
    # speed away from 1.0x — the "reset isn't sticking" symptom.
    _tempo_soft_takeover_lockout: bool = False
    _TEMPO_TAKEOVER_WINDOW = 8   # ticks around the learned center = "neutral"
    _TEMPO_CENTER_DEADZONE = 6   # ticks around center that snap to exactly 1.0x
    # The S2 tempo fader's physical detent rarely lands on raw ADC midpoint
    # (7-bit value 64). Learn the true resting value every time the fader
    # returns through center after a reset, so the deadzone — and the whole
    # speed map — tracks the real detent instead of a guessed midpoint.
    _tempo_center: int = 64

    def set_speed(self, value: int):
        """Tempo fader → mpv playback speed. ~±0.2% per tick measured from
        the *learned* center detent, with a deadzone that snaps to exactly
        1.0x so a fader at rest can't wobble the speed on ADC jitter. Honors
        soft-takeover after a sync reset, and re-learns the center every time
        the fader returns through it."""
        if not self.player:
            return
        if self._tempo_soft_takeover_lockout:
            if abs(value - self._tempo_center) <= self._TEMPO_TAKEOVER_WINDOW:
                # Fader is back at the detent → release lockout AND re-learn
                # where the detent actually sits (it drifts per device/fader).
                self._tempo_center = value
                self._tempo_soft_takeover_lockout = False
            else:
                return  # fader not yet back at center → ignore
        if abs(value - self._tempo_center) <= self._TEMPO_CENTER_DEADZONE:
            speed = 1.0
        else:
            speed = 1.0 + (value - self._tempo_center) * 0.002
            speed = max(0.5, min(1.5, speed))
        try:
            self.player.player.speed = speed
            self.player.player.audio_pitch_correction = True
        except Exception as e:
            logger.debug(f"set speed failed: {e}")

    def _left_enc_dispatch(self, delta: int):
        """LEFT loop encoder: when a loop is active → adjust loop start
        (each tick = ±0.1s nudge of the IN point). Otherwise → fine speed."""
        if self._loop_is_active():
            self._loop_in_nudge(delta * 0.1)
        else:
            self.fine_speed_nudge(delta)

    def _right_enc_dispatch(self, delta: int):
        """RIGHT loop encoder: when a loop is active → adjust loop end
        (each tick = ±0.1s nudge of the OUT point). Otherwise → fine volume."""
        if self._loop_is_active():
            self._loop_out_nudge(delta * 0.1)
        else:
            self.fine_volume_nudge(delta)

    def _loop_in_nudge(self, sec_delta: float):
        bounds = self._loop_get_bounds()
        if bounds is None or not self.player:
            return
        cur_a, cur_b = bounds
        new_a = max(0.0, min(cur_b - 0.05, cur_a + sec_delta))
        try:
            self.player.player.ab_loop_a = new_a
            self.state.set_message(f"Loop IN: {new_a:.2f}s ({cur_b - new_a:.2f}s long)")
        except Exception as e:
            logger.debug(f"loop_in_nudge failed: {e}")

    def _loop_out_nudge(self, sec_delta: float):
        bounds = self._loop_get_bounds()
        if bounds is None or not self.player:
            return
        cur_a, cur_b = bounds
        new_b = max(cur_a + 0.05, cur_b + sec_delta)
        try:
            self.player.player.ab_loop_b = new_b
            self.state.set_message(f"Loop OUT: {new_b:.2f}s ({new_b - cur_a:.2f}s long)")
        except Exception as e:
            logger.debug(f"loop_out_nudge failed: {e}")

    def fine_volume_nudge(self, delta: int):
        """Encoder rotation → ±1% volume per tick — fine adjustments above
        the fader's coarse curve."""
        if not self.player:
            return
        cur = self.player.get_volume()
        new_vol = max(1, min(100, cur + delta))
        self.player.set_volume(new_vol)
        self.state.set_volume(new_vol)
        try:
            self.player.player.mute = False
        except Exception:
            pass

    def adjust_flip_beats(self, delta: int):
        """A RIGHT encoder rotation → cycle auto-flip phrase length through
        the musical-phrase ladder [1, 2, 4, 8, 16, 32]. Each tick moves
        one rung. The chosen value is used by _on_beat to time auto-flips."""
        ladder = [1, 2, 4, 8, 16, 32]
        try:
            cur_idx = ladder.index(int(self._flip_beats))
        except ValueError:
            cur_idx = 3  # default = 8 beats
        new_idx = max(0, min(len(ladder) - 1, cur_idx + (1 if delta > 0 else -1 if delta < 0 else 0)))
        self._flip_beats = ladder[new_idx]
        # Show seconds-per-flip at the current BPM if we have one
        bpm = float(self.state.detected_bpm or 0)
        if 60.0 <= bpm <= 200.0:
            sec = (60.0 / bpm) * self._flip_beats
            self.state.set_message(f"Auto-flip: {self._flip_beats} beats (~{sec:.1f}s @ {bpm:.0f} BPM)")
        else:
            self.state.set_message(f"Auto-flip: {self._flip_beats} beats")

    def reset_flip_beats(self):
        """A RIGHT encoder press → snap auto-flip phrase length to default 8."""
        self._flip_beats = 8
        self.state.set_message("Auto-flip: 8 beats (default)")

    # HOLD: temporarily pause auto-flip on the CURRENT clip so a good
    # scene can breathe. Auto-flip silently no-ops while held. User can
    # still scrub the jog wheel, tweak tempo, manually flip, etc. — the
    # only thing held is "automatic next-clip on beat." Auto-releases
    # after a long timeout so a forgotten hold doesn't trap the visuals
    # forever (default 90s; tap GRID again to release sooner).
    _hold_clip: bool = False
    _hold_clip_until: float = 0.0
    _HOLD_TIMEOUT_S: float = 90.0

    # Hero-hold-on-drop suppression: when a drop is detected, auto-flip
    # is silently suppressed until this monotonic timestamp. Set by
    # _on_drop_detected (audio thread → just stamps the float, no Qt
    # marshaling needed since reads are racy-but-monotonic). Read in
    # _on_beat to gate the auto-flip path.
    _auto_flip_suppressed_until: float = 0.0
    _hero_locked_clip: Optional[str] = None

    def hold_clip_toggle(self):
        """MK2 GRID button → pin the current clip. Auto-flip won't
        fire while held. Press again to release; auto-releases after
        90s as a safety net."""
        import time as _time
        if self._hold_clip:
            self._hold_clip = False
            self._hold_clip_until = 0.0
            self.state.set_message("HOLD: off (auto-flip resumes)")
            logger.info("[hold-clip] released — auto-flip resumes")
        else:
            self._hold_clip = True
            self._hold_clip_until = _time.time() + self._HOLD_TIMEOUT_S
            self.state.set_message(
                f"HOLD: ON — staying on this clip (auto-release in "
                f"{int(self._HOLD_TIMEOUT_S)}s)")
            logger.info(f"[hold-clip] engaged — auto-flip suppressed for "
                        f"up to {self._HOLD_TIMEOUT_S}s or next GRID tap")

    # Tempo-match feature flag. ON by default since 2026-05-16 — once
    # the library is BPM-tagged, every load that has the data does the
    # warp. Toggle off when you want plain native speed
    # regardless of tempo data.
    _tempo_match_enabled: bool = True
    # Speed bounds match the jog-nudge clamps elsewhere — keep wide
    # enough for big tempo gaps, narrow enough to stay musical.
    _TEMPO_MATCH_MIN_SPEED: float = 0.5
    _TEMPO_MATCH_MAX_SPEED: float = 2.0

    def _get_clip_bpm(self, path: str) -> int:
        """Look up the bpm:N tag for a file from path_tags.db3. Returns
        0 when no tag exists (un-analysed). Cheap — single SQLite hit."""
        if not path:
            return 0
        try:
            for t in (self.path_tags.tags_for_file(path) or set()):
                if t.startswith("bpm:"):
                    try:
                        return int(t[4:])
                    except ValueError:
                        return 0
        except Exception:
            pass
        return 0

    def _apply_tempo_match(self, path: str) -> None:
        """Set playback speed so the clip's source music aligns with
        whatever's currently playing live.

        speed = live_bpm / clip_bpm. With octave-awareness — if the
        naive ratio is way off (e.g. clip tagged 180 BPM but it's
        really 90 BPM doubled by autocorrelation), try halving /
        doubling the source and pick whichever lands closest to 1.0.

        No-op when:
          - tempo-match toggle is off
          - mpv isn't ready
          - clip has no bpm tag
          - live BPM hasn't locked yet
        In any no-op case, sticky speed (whatever was set before) wins,
        which matches the user's preferred default behaviour."""
        if not self._tempo_match_enabled:
            return
        if not self.player or not self.player.player:
            return
        # CRITICAL FIX (code-review agent, 2026-05-16 night): if the user
        # has nudged speed via jog / encoder / fader since the last fader
        # detent, _tempo_soft_takeover_lockout is True. Don't slam their
        # manual feel — they're driving. Lockout clears when the fader
        # returns to detent (see _check_tempo_takeover near line 3252).
        if self._tempo_soft_takeover_lockout:
            logger.debug(f"[tempo-match] skipped {Path(path).name} — "
                         "user has manual speed lockout active")
            return
        clip_bpm = self._get_clip_bpm(path)
        if clip_bpm <= 0:
            return
        try:
            live_bpm = float(self.state.detected_bpm or 0.0)
        except Exception:
            live_bpm = 0.0
        if live_bpm <= 0:
            return
        # Octave-aware ratio pick — autocorrelation sometimes tags at
        # 2x or 0.5x the true tempo. Try all three and pick whichever
        # gives a speed closest to 1.0 (least extreme warp).
        candidates = []
        for src in (clip_bpm, clip_bpm * 2, clip_bpm / 2.0):
            if src > 0:
                candidates.append((live_bpm / src, src))
        # Filter to in-range, then pick min |1 - speed|
        in_range = [
            (sp, src) for sp, src in candidates
            if self._TEMPO_MATCH_MIN_SPEED <= sp <= self._TEMPO_MATCH_MAX_SPEED
        ]
        if not in_range:
            # All octaves clamp out — too far apart musically. Skip.
            logger.debug(f"[tempo-match] {Path(path).name}: clip={clip_bpm} "
                         f"live={live_bpm:.1f} → no in-range ratio, skip")
            return
        speed, source_bpm = min(in_range, key=lambda t: abs(1.0 - t[0]))
        try:
            self.player.player.speed = float(speed)
            self.player.player.audio_pitch_correction = True
            note = f" (octave-folded source {clip_bpm}→{source_bpm:.0f})" if abs(source_bpm - clip_bpm) > 0.5 else ""
            logger.info(f"[tempo-match] {Path(path).name}: clip={clip_bpm}bpm "
                        f"live={live_bpm:.1f}bpm → speed={speed:.3f}x{note}")
        except Exception as e:
            logger.debug(f"tempo-match speed set failed: {e}")

    def tempo_match_toggle(self) -> None:
        """Flip the global tempo-match enable. When ON (default), each
        load adjusts playback speed = live_bpm / clip_bpm. When OFF,
        playback speed is sticky/global only (no auto-warp)."""
        self._tempo_match_enabled = not self._tempo_match_enabled
        state_str = "ON" if self._tempo_match_enabled else "OFF"
        self.state.set_message(f"Tempo-match: {state_str}")
        logger.info(f"[tempo-match] toggled {state_str}")

    # ── Hardware vote up/down (master section < / > arrows) ──────────
    # During live VJ the clips pass too fast to manually annotate, so
    # the user wanted snap-vote buttons right under the screen. Up-votes
    # boost a file's selection weight in the random picker; down-votes
    # suppress it (but don't eliminate it — clamp at 0.1x so the user
    # can recover a wrong call). Tempo nudge moved entirely to the
    # TEMPO encoder spin (which always did the job better anyway).

    def _current_clip_path(self) -> Optional[str]:
        """Resolved path of the clip currently playing — used by vote
        + hold + pin operations. Falls back through several sources."""
        try:
            p = str(
                self._current_source_path
                or (self.player.current_file if self.player else "")
                or ""
            )
            return p or None
        except Exception:
            return None

    def mk2_vote_up(self) -> None:
        """MK2 master `>`: upvote the currently-playing clip. Picker
        will weight this clip more in future random picks."""
        if self.vote_store is None:
            self.state.set_message("Vote store unavailable", error=True)
            return
        path = self._current_clip_path()
        if not path:
            self.state.set_message("No current clip to vote on", error=True)
            return
        score, ups, downs = self.vote_store.upvote(path)
        from pathlib import Path as _P
        name = _P(path).name
        self.state.set_message(
            f"VOTE UP [{name[:24]}]: score={score:+d} (↑{ups}/↓{downs})"
        )
        logger.info(f"[vote-up] {name} → score={score:+d}")
        try:
            self._pad_flash_toggle("on", label=f"+1 {name[:14]}")
        except Exception:
            pass
        # OLED confirmation flash so the operator sees the vote land
        # without looking at the iPad. Persists 1.2s.
        try:
            if self.mk2:
                self.mk2.push_oled_vote_flash(
                    direction="+", score=score, ups=ups, downs=downs,
                    clip_name=name, hold_seconds=1.2,
                )
        except Exception as e:
            logger.debug(f"vote-up OLED push failed: {e}")

    def mk2_reclassify_to(self, letter: str) -> None:
        """Hold-GROUP-X gesture: tell the categorizer that the currently-
        playing clip really belongs in category X. Cumulative — one
        press = 0.8 score boost (nudge), 2 presses = ~1.6 (will flip
        most close calls), 3+ = locks the bucket.

        VOTE V2 (2026-05-18 afternoon): auto-reshuffles the active
        vertical's banks immediately after the correction is recorded
        — no need to manually re-press the vertical. The currently-
        playing clip's scratch pool is NOT touched (we only update
        bank_store); the next tap-load of a GROUP letter picks up
        the corrected split."""
        if self.vote_store is None:
            self.state.set_message("Vote store unavailable", error=True)
            return
        path = self._current_clip_path()
        if not path:
            self.state.set_message(
                "Reclassify: no current clip", error=True
            )
            return
        letter = letter.upper()
        count, all_corrections = self.vote_store.correct_category(
            path, letter
        )
        from pathlib import Path as _P
        name = _P(path).name[:24]
        # Compact summary of all corrections on this file, sorted by
        # count desc. "H=3 B=1" reads as "user has marked this H
        # three times, B once."
        summary = " ".join(
            f"{l}={c}"
            for l, c in sorted(
                all_corrections.items(), key=lambda x: -x[1]
            )
            if c > 0
        )
        # ── Auto-reshuffle: re-split the active vertical so the
        # correction takes effect immediately. We detect the active
        # source folder by looking at the folder lock on bank slots —
        # vertical-press auto-split sets the same lock on all 8 banks,
        # so any one of them tells us the source.
        reshuffle_note = "— pending next vertical re-press"
        try:
            source_folder = ""
            for ltr in "ABCDEFGH":
                s = self.bank_store.get(ltr) or {}
                f = (s.get("folder") or "").strip()
                if f:
                    source_folder = f
                    break
            if source_folder:
                from pathlib import Path as _PP
                # Compute the relative path from library_root so the
                # bank_from_folder helper resolves cleanly.
                root_str = (
                    self.state.get_library_snapshot().get("root") or ""
                )
                folder_arg = source_folder
                if root_str:
                    try:
                        rel = _PP(source_folder).relative_to(
                            _PP(root_str)
                        )
                        folder_arg = str(rel).replace("\\", "/")
                    except ValueError:
                        pass  # not under root, pass full path
                result = self.bank_from_folder(
                    folder_rel=folder_arg, recursive=True,
                )
                if result.get("ok"):
                    reshuffle_note = (
                        f"— reshuffled {_PP(source_folder).name} "
                        "(tap a GROUP letter to load corrected bank)"
                    )
                    logger.info(
                        f"[reclassify] auto-reshuffled "
                        f"{source_folder} after correction"
                    )
                else:
                    reshuffle_note = (
                        f"— reshuffle failed: "
                        f"{result.get('error', '?')}"
                    )
        except Exception as e:
            logger.debug(f"reclassify auto-reshuffle failed: {e}")
        self.state.set_message(
            f"RECLASSIFY [{name}] → {letter} ({summary}) "
            f"{reshuffle_note}"
        )
        logger.info(
            f"[reclassify] {name} → {letter} (count={count}, "
            f"all={all_corrections})"
        )
        try:
            self._pad_flash_toggle(
                "on", label=f"→ {letter} x{count}"
            )
        except Exception:
            pass
        # OLED confirmation flash (user said pads flashed but OLED was
        # silent — added 2026-05-19). Shows RECLASSIFY -> X header +
        # clip name + correction count for 1.5s.
        try:
            if self.mk2:
                self.mk2.push_oled_reclassify_flash(
                    letter=letter, name=name, count=count,
                    hold_seconds=1.5,
                )
        except Exception as e:
            logger.debug(f"reclassify OLED push failed: {e}")
        # Reclassify changed user intent for this clip — re-roll
        # cohesion so the next picks align with the corrected mood.
        self._force_cohesion_refresh(reason="reclassify")

    def mk2_vote_down(self) -> None:
        """MK2 master `<`: downvote the currently-playing clip. Picker
        will weight this clip less. Clamped at 0.1x so a single misvote
        doesn't kill a clip — just demotes it."""
        if self.vote_store is None:
            self.state.set_message("Vote store unavailable", error=True)
            return
        path = self._current_clip_path()
        if not path:
            self.state.set_message("No current clip to vote on", error=True)
            return
        score, ups, downs = self.vote_store.downvote(path)
        from pathlib import Path as _P
        name = _P(path).name
        self.state.set_message(
            f"VOTE DN [{name[:24]}]: score={score:+d} (↑{ups}/↓{downs})"
        )
        logger.info(f"[vote-down] {name} → score={score:+d}")
        try:
            self._pad_flash_toggle("off", label=f"-1 {name[:14]}")
        except Exception:
            pass
        try:
            if self.mk2:
                self.mk2.push_oled_vote_flash(
                    direction="-", score=score, ups=ups, downs=downs,
                    clip_name=name, hold_seconds=1.2,
                )
        except Exception as e:
            logger.debug(f"vote-down OLED push failed: {e}")

    # Tempo nudge step — how much each < / > master-section press
    # moves playback speed. 1.5% = musically meaningful but small
    # enough you can hammer the button to dial in. Tunable.
    _TEMPO_NUDGE_STEP: float = 0.015

    def tempo_nudge_up(self) -> None:
        """MK2 master-section `>` (bit 20): bump playback speed by
        _TEMPO_NUDGE_STEP. Same soft-takeover lockout as the jog so
        tempo-match doesn't slam it back on the next flip."""
        if not (self.player and self.player.player):
            return
        try:
            cur = float(self.player.player.speed or 1.0)
            new = max(0.25, min(3.0, cur + self._TEMPO_NUDGE_STEP))
            self.player.player.speed = new
            self.player.player.audio_pitch_correction = True
            self._tempo_soft_takeover_lockout = True
            self.state.set_message(f"Speed: {new:.3f}x ▲")
            logger.info(f"[tempo-nudge] up: {cur:.3f} → {new:.3f}")
        except Exception as e:
            logger.debug(f"tempo_nudge_up failed: {e}")

    def tempo_nudge_down(self) -> None:
        """MK2 master-section `<` (bit 19): drop playback speed by
        _TEMPO_NUDGE_STEP. Same soft-takeover lockout."""
        if not (self.player and self.player.player):
            return
        try:
            cur = float(self.player.player.speed or 1.0)
            new = max(0.25, min(3.0, cur - self._TEMPO_NUDGE_STEP))
            self.player.player.speed = new
            self.player.player.audio_pitch_correction = True
            self._tempo_soft_takeover_lockout = True
            self.state.set_message(f"Speed: {new:.3f}x ▼")
            logger.info(f"[tempo-nudge] down: {cur:.3f} → {new:.3f}")
        except Exception as e:
            logger.debug(f"tempo_nudge_down failed: {e}")

    def reseek_current_clip(self) -> None:
        """MK2 ENTER (bit 21): jump to a new random body position in
        the CURRENT clip. Lets you stay on a clip but reshuffle which
        part is playing — same gesture as tapping a pad twice to get
        a new random offset, but for the flip path where there's no
        pad to re-tap. (2026-05-17 user req.)

        Reuses _auto_seek_into_body's logic: skip intro/outro,
        avoid recent visit, land in the body."""
        if not (self.player and self.player.player):
            return
        try:
            self._auto_seek_into_body()
            logger.info("[reseek] re-rolled body position on current clip")
            self.state.set_message("Re-seek: new position")
        except Exception as e:
            logger.debug(f"reseek_current_clip failed: {e}")

    # BPM-tolerance presets, cycled by MASTER encoder click (bit 23).
    # ±5 = tight match (most musical, smallest candidate pool)
    # ±10 = default (the current value), good balance
    # ±20 = loose (forgiving — fills the pool even when live BPM is weird)
    # 999 = OFF (BPM-pref disabled, whole pool eligible)
    _BPM_TOL_PRESETS: tuple = (5.0, 10.0, 20.0, 999.0)

    def bpm_tolerance_cycle(self) -> None:
        """MK2 MASTER click (bit 23): step through ±5 / ±10 / ±20 /
        OFF on the BPM-preference filter. 999 = disabled (whole-pool
        random). (2026-05-17 user req.)"""
        try:
            idx = self._BPM_TOL_PRESETS.index(self._BPM_PREF_TOLERANCE)
        except ValueError:
            idx = 1  # default to ±10
        idx = (idx + 1) % len(self._BPM_TOL_PRESETS)
        new_tol = self._BPM_TOL_PRESETS[idx]
        self._BPM_PREF_TOLERANCE = new_tol
        if new_tol >= 999:
            self._bpm_preference_enabled = False
            label = "BPM-pref: OFF (whole pool)"
        else:
            self._bpm_preference_enabled = True
            label = f"BPM-pref: ±{int(new_tol)} BPM"
        self.state.set_message(label)
        logger.info(f"[bpm-tol] {label}")

    # Beat-locked stutter hold (NOTE REPEAT, bit 22). User pitch
    # 2026-05-17: "when a clip is playing let me hit it to do like a
    # build or a drop where it hold and repeats... so when its hitting
    # it doesnt flip or move past that specific action."
    #
    # Implementation: A-B loop on the current position (1 beat at
    # current live BPM, or 0.5s fallback). Engages HOLD so auto-flip
    # won't fire over it. Press again to release.
    _stutter_engaged: bool = False
    _stutter_saved_hold: bool = False  # to restore HOLD state on release

    def stutter_toggle(self) -> None:
        """MK2 NOTE REPEAT (bit 22): toggle beat-locked stutter hold
        on the current clip. ON: 1-beat A-B loop at current position +
        HOLD engaged (no auto-flip). OFF: clear loop + restore HOLD.

        TODO: press-and-hold variant + accelerating loop division
        (1 → 1/2 → 1/4 → 1/8 beat) for the actual riser-into-drop
        effect. v1 is just the constant-rate loop."""
        if not (self.player and self.player.player):
            return
        try:
            if self._stutter_engaged:
                # Release: clear A-B loop + restore HOLD
                self.player.player.command("set", "ab-loop-a", "no")
                self.player.player.command("set", "ab-loop-b", "no")
                if not self._stutter_saved_hold:
                    self._hold_clip = False
                self._stutter_engaged = False
                self.state.set_message("Stutter: OFF")
                logger.info("[stutter] released")
                return
            # Engage: snapshot current pos + bpm, build the loop using
            # the configured division. Reset to 2 beats on every fresh
            # engage (2026-05-17 user req: "let start at like two
            # starting too sharp" — 1-beat default was too aggressive
            # as the entry point). From there, encoder rotation builds.
            self._stutter_division_multiplier = 2.0
            cur_pos = float(getattr(self.player.player, "time_pos", 0.0) or 0.0)
            live_bpm = float(self.state.detected_bpm or 0.0)
            # 1-beat duration. Default to 0.5s if no live BPM (= 120 BPM).
            beat_sec = (60.0 / live_bpm) if live_bpm > 0 else 0.5
            loop_len = beat_sec * self._stutter_division_multiplier
            loop_a = max(0.0, cur_pos)
            loop_b = loop_a + loop_len
            self.player.player.command("set", "ab-loop-a", str(loop_a))
            self.player.player.command("set", "ab-loop-b", str(loop_b))
            # HOLD on (remember prior state so we don't release it on
            # stutter-off if user had HOLD engaged before).
            self._stutter_saved_hold = bool(getattr(self, "_hold_clip", False))
            self._hold_clip = True
            self._stutter_engaged = True
            self.state.set_message(
                f"Stutter: ON (1 beat @ {live_bpm:.0f} BPM = {beat_sec*1000:.0f}ms)")
            logger.info(f"[stutter] engaged: loop {loop_a:.2f}→{loop_b:.2f}s "
                        f"({beat_sec*1000:.0f}ms @ {live_bpm:.0f} BPM)")
        except Exception as e:
            logger.debug(f"stutter_toggle failed: {e}")

    def tempo_lock_toggle(self):
        """MK2 TEMPO button → tempo-match-only mode toggle. Different
        from RESTART (bit 32, the full auto-flip toggle).

        2026-05-17 user req: "I would like it to just toggle between
        audio reactive bpm without flipping just adjusting the speed of
        the video to the live bpm."

        ON: audio-reactive ON (provides BPM for tempo-match-on-load),
            flip_on_beat FORCED OFF (no auto-flipping — user manually
            controls clip selection while videos warp to live tempo).
        OFF: audio-reactive OFF, speed snapped to 1.0x.

        If you want auto-flip + tempo-match together, use RESTART
        (which starts audio-react as a dependency and leaves
        flip_on_beat ON). TEMPO is the "just warp my videos to the
        music, I'll fire them manually" gesture.

        LED indication of state is a future ship (master-section LEDs
        live on HID report 0x82 which the driver doesn't write yet)."""
        currently_on = self.state.audio_reactive_enabled
        if currently_on:
            # OFF: disable detector + snap speed to 1.0x
            self.audio_reactive_stop()
            if self.player and self.player.player:
                try:
                    cur = float(self.player.player.speed or 1.0)
                    self.player.player.speed = 1.0
                    self.state.set_message(f"TEMPO: OFF (speed → 1.0x, was {cur:.2f}x)")
                    logger.info(f"[tempo] OFF: audio-react stopped, speed reset 1.0x (was {cur:.2f}x)")
                except Exception as e:
                    logger.debug(f"tempo OFF speed snap failed: {e}")
            self._pad_flash_toggle("off", label="tempo OFF")
        else:
            # ON: enable detector. SUPPRESS auto-flip so it's tempo-match
            # only (per user req — no flipping just from pressing TEMPO).
            # Don't touch speed; tempo-match-on-load handles future loads.
            was_flipping = self.state.get_flip_on_beat()
            if was_flipping:
                with self._auto_flip_lock:
                    self.state.set_flip_on_beat(False)
                    self._auto_flip_use_folder = False
                logger.info("[tempo] suppressing flip_on_beat (TEMPO is "
                            "tempo-match-only — use RESTART for auto-flip)")
            self.audio_reactive_start()
            label = "TEMPO: ON (BPM lock, no flip"
            if was_flipping:
                label += " — flip suppressed)"
            else:
                label += ")"
            self.state.set_message(label)
            logger.info(f"[tempo] ON: audio-react started, flip_on_beat=False")
            self._pad_flash_toggle("cyan", label="tempo ON")

    # Stutter loop division presets (for stutter_division_cycle).
    # Multiplier × 1-beat duration. Cycles slowest → fastest, then loops.
    # 4 → 1.85s @ 130 BPM (slow half-bar chop)
    # 1/16 → 0.029s @ 130 BPM (granular, near-pitch)
    _STUTTER_DIVISIONS: tuple = (4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625)
    _stutter_division_multiplier: float = 1.0  # default = 1 beat

    def stutter_division_step(self, delta: int) -> None:
        """Step stutter division by `delta` positions in the preset
        ladder. Positive delta = move toward SMALLER divisions
        (faster stutter / build / riser). Negative = bigger (slower /
        calmer). Clamps at the ladder ends — does NOT wrap, so you
        can feel the limit without surprise jumps.

        Wired to MK2 master encoder rotation (CW = positive delta).
        Live-updates engaged stutter loop. (2026-05-17 ship.)"""
        if not delta:
            return
        try:
            cur_idx = self._STUTTER_DIVISIONS.index(self._stutter_division_multiplier)
        except ValueError:
            cur_idx = 2  # default = 1 beat
        new_idx = max(0, min(len(self._STUTTER_DIVISIONS) - 1, cur_idx + delta))
        if new_idx == cur_idx:
            # At an end — say so so user knows.
            edge = "max slow (4 beats)" if new_idx == 0 else "max fast (1/16 beat)"
            self.state.set_message(f"Stutter: at {edge}")
            return
        new_mult = self._STUTTER_DIVISIONS[new_idx]
        self._apply_stutter_division(new_mult)

    def stutter_division_cycle(self) -> None:
        """Cycle the stutter loop length one step toward smaller (faster).
        Same as stutter_division_step(+1) but wraps at the end. Used
        for button bindings; rotation uses _step directly.

        (Parked for now — top row is reserved for folder nav; master
        encoder rotation is the actual home as of 2026-05-17 night.)"""
        try:
            idx = self._STUTTER_DIVISIONS.index(self._stutter_division_multiplier)
        except ValueError:
            idx = 2
        idx = (idx + 1) % len(self._STUTTER_DIVISIONS)
        new_mult = self._STUTTER_DIVISIONS[idx]
        self._apply_stutter_division(new_mult)

    def _apply_stutter_division(self, new_mult: float) -> None:
        """Update _stutter_division_multiplier + the live A-B loop if
        stutter is engaged. Shared by step (encoder) and cycle (button)."""
        self._stutter_division_multiplier = new_mult
        # Pretty label: 4 / 2 / 1 / 1/2 / 1/4 / 1/8 / 1/16
        label = (str(int(new_mult)) if new_mult >= 1
                 else f"1/{int(1.0 / new_mult)}")
        self.state.set_message(f"Stutter division: {label} beats")
        logger.info(f"[stutter-div] cycled to {label} beats ({new_mult}x)")
        # Live-update the loop if stutter is currently engaged.
        if self._stutter_engaged and self.player and self.player.player:
            try:
                live_bpm = float(self.state.detected_bpm or 0.0)
                beat_sec = (60.0 / live_bpm) if live_bpm > 0 else 0.5
                loop_len = beat_sec * new_mult
                cur_pos = float(getattr(self.player.player, "time_pos", 0.0) or 0.0)
                # Anchor loop start at current pos (avoids glitch on shrinking).
                loop_a = max(0.0, cur_pos)
                loop_b = loop_a + loop_len
                self.player.player.command("set", "ab-loop-a", str(loop_a))
                self.player.player.command("set", "ab-loop-b", str(loop_b))
                logger.info(f"[stutter] live-updated loop to {loop_len*1000:.0f}ms")
            except Exception as e:
                logger.debug(f"stutter live-update failed: {e}")

    # ────────────────────────── AUDIO LEAK ─────────────────────────────────
    # Hold-to-engage transient audio leak. Useful for "moment" effects: a
    # moan, a quotable line, a sound effect — pushed out the device mpv is
    # pinned to (mpv_audio_device_substring; typically the S2 Monitor circuit
    # so it doesn't leak into the BPM detector's input). User holds the
    # bound MK2 button → video audio fades in to a configurable max; release
    # → fades back to baseline. Implementation is in-app mpv-volume animation
    # via 60Hz QTimer; respects the existing audio-device pinning so the BPM
    # counter (which lives on a different loopback bus) is not perturbed.
    _audio_leak_active: bool = False
    _audio_leak_baseline: int = 0
    _audio_leak_timer = None

    def audio_leak_press(self) -> None:
        """Engage transient audio leak. Captures current mpv volume as
        baseline, ramps to leak max over fade_ms, holds until release."""
        if not self.player:
            return
        if self._audio_leak_active:
            return
        try:
            self._audio_leak_baseline = int(self.player.get_volume())
        except Exception:
            self._audio_leak_baseline = 0
        target = int(self.config.get("audio_leak_max_vol", 65))
        fade_ms = int(self.config.get("audio_leak_fade_ms", 80))
        self._audio_leak_active = True
        self._audio_leak_fade(self._audio_leak_baseline, target, fade_ms)
        self.state.set_message(
            f"AUDIO LEAK ON  ({self._audio_leak_baseline}→{target}%)")
        logger.info(
            "[audio-leak] engage: vol %d → %d over %dms",
            self._audio_leak_baseline, target, fade_ms,
        )

    def audio_leak_release(self) -> None:
        """Disengage leak. Fades volume back to the captured baseline."""
        if not self._audio_leak_active or not self.player:
            return
        fade_ms = int(self.config.get("audio_leak_fade_ms", 80))
        try:
            cur = int(self.player.get_volume())
        except Exception:
            cur = self._audio_leak_baseline
        self._audio_leak_fade(cur, self._audio_leak_baseline, fade_ms)
        self._audio_leak_active = False
        self.state.set_message(
            f"audio leak off (→{self._audio_leak_baseline}%)")
        logger.info(
            "[audio-leak] release: vol %d → %d over %dms",
            cur, self._audio_leak_baseline, fade_ms,
        )

    def _audio_leak_fade(self, start: int, end: int, ms: int) -> None:
        """Animate mpv volume start→end over ms with a 60Hz QTimer.
        Stops any in-flight fade first (so press-release-press doesn't
        race)."""
        if self._audio_leak_timer is not None:
            try:
                self._audio_leak_timer.stop()
            except Exception:
                pass
            self._audio_leak_timer = None
        if not self.player:
            return
        if ms <= 16 or start == end:
            try:
                self.player.set_volume(end)
            except Exception:
                pass
            return
        steps = max(2, ms // 16)
        delta = (end - start) / steps
        # Use a list so the inner closure can mutate via list-index assign.
        state = [0]
        t = QTimer(self)
        t.setInterval(16)

        def _tick():
            state[0] += 1
            try:
                if state[0] >= steps:
                    self.player.set_volume(end)
                    t.stop()
                    if self._audio_leak_timer is t:
                        self._audio_leak_timer = None
                else:
                    self.player.set_volume(int(start + delta * state[0]))
            except Exception as e:
                logger.debug(f"audio_leak fade tick failed: {e}")
                t.stop()
                if self._audio_leak_timer is t:
                    self._audio_leak_timer = None

        t.timeout.connect(_tick)
        self._audio_leak_timer = t
        t.start()

    # ── audio mute toggle (S2 encoder press) ───────────────────────────
    _audio_last_unmute_vol: int = 65

    def audio_mute_toggle(self) -> None:
        """Toggle mpv volume between 0 (muted) and the last non-zero
        value. Bound to an S2 encoder PRESS (push-the-knob-in gesture).
        Use case: clean kill-switch for video audio without losing your
        MIX-knob position. Press again to restore."""
        if not self.player:
            return
        try:
            cur = int(self.player.get_volume())
        except Exception:
            cur = 0
        if cur > 0:
            # Remember current as the unmute target, then mute
            self._audio_last_unmute_vol = cur
            try:
                self.player.set_volume(0)
                self.state.set_message(f"AUDIO MUTED (was {cur}%)")
                logger.info(f"[audio-mute] muted (saved {cur}% for unmute)")
            except Exception as e:
                logger.debug(f"mute failed: {e}")
        else:
            # Unmute back to last value (or default)
            target = int(getattr(self, "_audio_last_unmute_vol", 65)) or 65
            try:
                self.player.set_volume(target)
                self.state.set_message(f"AUDIO UN-MUTED ({target}%)")
                logger.info(f"[audio-mute] unmuted to {target}%")
            except Exception as e:
                logger.debug(f"unmute failed: {e}")

    def audio_leak_knob(self, value: int) -> None:
        """Direct mpv-volume control from the S2 HEADPHONE MIX front-panel
        knob. value is 0..127 (7-bit debounced by the S2 driver). Scales
        linearly to mpv volume 0..100.

        DIAGNOSTIC: this method also logs every reading so we can see
        the raw range the knob actually sends — needed to verify the
        wire is live and the knob isn't actually a push-button.

        Why MIX and not VOL: VOL is the headphone-amp pot wired directly
        to the headphone amp, not exposed over USB. MIX is the cue-blend
        knob, exposed via report_02 byte 0x31, and currently unbound in
        setpiece (it had no DJ role here). Repurposing as the leak
        knob is the most DJ-natural mapping we can get.

        Composes with audio_leak_press/release: if you bind a hold button
        too, the button captures the knob's current value as baseline,
        ramps up to leak_max_vol, then snaps back to the knob's reading
        on release. So you can dial in an ambient bed with the knob and
        punch peaks with the button."""
        # Diagnostic-only log; demoted to DEBUG so a knob twist doesn't
        # flood the session log at ~60Hz under normal use. Promote to
        # INFO again temporarily if you ever need to re-verify the wire.
        logger.debug(f"[audio-leak-knob] raw={value} → vol={value * 100 // 127}%")
        if not self.player:
            return
        vol = max(0, min(100, int(value * 100 / 127)))
        try:
            self.player.set_volume(vol)
            # When the knob is being driven, leak isn't "active" in the
            # hold-button sense — make sure that state is consistent so
            # a subsequent button-hold captures the knob's value as the
            # baseline (not a stale 0).
            self._audio_leak_active = False
        except Exception as e:
            logger.debug(f"audio_leak_knob set_volume failed: {e}")

    # Right-jog speed-nudge gain. 0.00025 untouched = ~0.025% per
    # velocity unit ≈ 0.25-1 %/sec on a slow deliberate spin (true
    # DJ "fine pitch bend" feel — the previous 0.0005 felt twitchy
    # for last-percent dial-in). Touching the platter scales 8x for
    # big jumps — grab the platter to sweep across 0.5x..2x range
    # fast, let go to dial in the last few percent. Tunable here.
    _JOG_SPEED_GAIN_FINE: float = 0.00025
    _JOG_SPEED_GAIN_COARSE_MULT: float = 8.0

    def jog_speed_nudge(self, velocity: int):
        """RIGHT jog rotation → playback speed accumulator. Untouched
        spin = fine pitch bend (~0.05% per velocity unit). Touching the
        platter scales the gain 8x for fast traversal — release to
        switch back to fine. Persists across clip loads. Engages tempo
        soft-takeover so the fader won't yank speed back until
        physically returned to its detent.

        Range clamped 0.25x..3.0x. The tempo fader stays at ±13% for
        precision; the jog gives wide reach + fine bend on top."""
        if not self.player or not velocity:
            return
        gain = self._JOG_SPEED_GAIN_FINE
        if self._jog_touched:
            gain *= self._JOG_SPEED_GAIN_COARSE_MULT
        delta = velocity * gain
        try:
            cur = float(self.player.player.speed or 1.0)
            new_speed = max(0.25, min(3.0, cur + delta))
            self.player.player.speed = new_speed
            self.player.player.audio_pitch_correction = True
            self._tempo_soft_takeover_lockout = True
        except Exception as e:
            logger.debug(f"jog_speed_nudge failed: {e}")

    def fine_speed_nudge(self, delta: int):
        """LEFT encoder rotation → ±5% speed per tick (coarse — primary
        speed control while the tempo fader is unreliable). Engages tempo
        soft-takeover lockout so the fader can't immediately yank the
        speed back — fader has to physically return to center before it
        grabs control again."""
        if not self.player:
            return
        try:
            cur = float(self.player.player.speed or 1.0)
            new_speed = max(0.25, min(3.0, cur + delta * 0.08))
            self.player.player.speed = new_speed
            self._tempo_soft_takeover_lockout = True
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "·")
            self.state.set_message(f"Speed: {new_speed:.2f}x {arrow}")
        except Exception as e:
            logger.debug(f"fine_speed_nudge failed: {e}")

    def fine_brightness_nudge(self, delta: int):
        """Encoder rotation → ±2 brightness per tick. Pairs with B EQ HI knob."""
        if not self.player:
            return
        try:
            cur = int(getattr(self.player.player, 'brightness', 0) or 0)
            new_b = max(-100, min(50, cur + delta * 2))
            self.player.player.brightness = new_b
        except Exception as e:
            logger.debug(f"fine_brightness_nudge failed: {e}")

    def reset_speed(self):
        """Snap playback speed to 1.0x. Engages soft-takeover so the
        tempo fader's current (probably off-center) position doesn't
        immediately yank the speed away — fader must physically return
        to center first."""
        if not self.player:
            return
        try:
            self.player.player.speed = 1.0
            self._tempo_soft_takeover_lockout = True
            self.state.set_message("Speed: 1.00x — slide tempo fader to its detent to re-engage (re-learns center)")
        except Exception as e:
            logger.debug(f"reset_speed failed: {e}")

    def reset_volume(self):
        """Snap volume back to a sensible default (80%)."""
        if not self.player:
            return
        self.player.set_volume(80)
        self.state.set_volume(80)
        self.state.set_message("Volume: 80")
        try:
            self.player.player.mute = False
        except Exception:
            pass

    def loop_length_nudge(self, delta: int):
        """A Gain encoder rotation when a loop is active: each tick CW
        DOUBLES the loop, each tick CCW HALVES it. Anchors on the loop
        IN point — only the OUT moves. Beat-snaps if BPM is known."""
        bounds = self._loop_get_bounds()
        logger.info(f"LOOP_NUDGE delta={delta} bounds={bounds}")
        if bounds is None or not self.player:
            return
        cur_a, cur_b = bounds
        duration = cur_b - cur_a
        if duration <= 0:
            return
        # delta > 0 → double, delta < 0 → halve. Multiple ticks compound.
        factor = 2.0 ** delta
        new_dur = max(0.05, duration * factor)
        # Beat-snap if we know BPM
        bpm = float(self.state.detected_bpm or 0)
        if bpm > 30 and bpm < 250:
            beat_sec = 60.0 / bpm
            # Round to nearest power-of-2 beats: 1/4, 1/2, 1, 2, 4, 8, 16
            target_beats = new_dur / beat_sec
            allowed = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
            best = min(allowed, key=lambda b: abs(b - target_beats))
            new_dur = best * beat_sec
        new_b = float(cur_a) + new_dur
        try:
            self.player.player.ab_loop_b = new_b
            # If the playhead is now past the new OUT (common when halving),
            # mpv's ab-loop won't fire on its own — seek back to IN so the
            # new loop is immediately audible.
            cur_pos = self.player.get_position()
            if cur_pos is not None and cur_pos > new_b:
                self.player.seek(float(cur_a))
        except Exception as e:
            logger.debug(f"loop resize failed: {e}")
        if bpm > 30:
            beats = new_dur / (60.0 / bpm)
            self.state.set_message(f"Loop: {new_dur:.2f}s ({beats:.2g} beats @ {bpm:.0f} BPM)")
        else:
            self.state.set_message(f"Loop: {new_dur:.2f}s")

    def exit_loop(self):
        """A Gain encoder press: clear the A-B loop, video plays through normally."""
        if not self.player:
            return
        for prop in ("ab-loop-a", "ab-loop-b"):
            try:
                self.player.player.command("set", prop, "no")
            except Exception:
                pass
        self.state.set_message("Loop OFF")

    def reset_brightness(self):
        """Snap master brightness back to 0 (normal)."""
        if not self.player:
            return
        try:
            self.player.player.brightness = 0
            self.state.set_message("Brightness: normal")
        except Exception as e:
            logger.debug(f"reset_brightness failed: {e}")

    def set_speed_from_knob(self, value: int):
        """A EQ HI knob → playback speed. Full 0-127 range now.
        Linear: low (0) = 0.5x, center (64) = 1.0x, high (127) = 2.0x."""
        if not self.player:
            return
        if value <= 64:
            speed = 0.5 + (value / 64.0) * 0.5  # 0→0.5, 64→1.0
        else:
            speed = 1.0 + ((value - 64) / 63.0)  # 64→1.0, 127→2.0
        speed = max(0.25, min(3.0, speed))
        try:
            self.player.player.speed = speed
            self.player.player.audio_pitch_correction = True
        except Exception as e:
            logger.debug(f"set_speed_from_knob failed: {e}")

    # Tracked separately from mpv's brightness because the beat-pulse
    # rides on top — pulse decay restores to THIS value, not whatever
    # the current mpv brightness happens to be (which may be the
    # pulse-amplified version mid-decay).
    _user_brightness: int = 0

    def set_master_brightness(self, value: int):
        """B EQ HI knob → master video brightness. Full 0-127 range now.
        Center (64) = normal. Down (0) = fade-to-black. Up brightens slightly.
        mpv's brightness property: -100=black, 0=normal, +100=white."""
        if not self.player:
            return
        if getattr(self, "_blackout_active", False):
            return  # blackout is a hard override — press B RESET to exit
        if value <= 64:
            brightness = int((value / 64.0) * 100) - 100  # 0→-100, 64→0
        else:
            brightness = int(((value - 64) / 63.0) * 30)  # 64→0, 127→+30
        brightness = max(-100, min(50, brightness))
        self._user_brightness = brightness
        try:
            self.player.player.brightness = brightness
        except Exception as e:
            logger.debug(f"set_master_brightness failed: {e}")

    # ── Audio-reactive beat pulse on mpv brightness ───────────────────
    # Subtle visual punch on each detected kick. Rides ON TOP of the
    # user's knob-set brightness (_user_brightness) so concurrent knob
    # movement during a pulse decay doesn't get stomped — the decay
    # restores to whatever value the user knob is at NOW.
    _BEAT_PULSE_AMOUNT: int = 8   # mpv brightness units added briefly
    _BEAT_PULSE_DECAY_MS: int = 150
    _beat_pulse_enabled: bool = True  # cosmetic toggle, future hotkey

    def _beat_pulse_brightness(self):
        """Called from _on_beat via QTimer (Qt thread only). Briefly
        bumps mpv brightness, schedules a decay back to _user_brightness."""
        if not self._beat_pulse_enabled:
            return
        if not self.player or not self.player.player:
            return
        if getattr(self, "_blackout_active", False):
            return
        try:
            self.player.player.brightness = (self._user_brightness
                                              + self._BEAT_PULSE_AMOUNT)
        except Exception:
            return
        QTimer.singleShot(self._BEAT_PULSE_DECAY_MS,
                          self._beat_pulse_decay)

    def _beat_pulse_decay(self):
        """Restore brightness to the user's current knob value
        (not whatever's currently in mpv — supports knob movement
        during the pulse window)."""
        if not self.player or not self.player.player:
            return
        if getattr(self, "_blackout_active", False):
            return
        try:
            self.player.player.brightness = self._user_brightness
        except Exception:
            pass

    # ── Filter knobs (S2 EQ MID + B LOW → mpv visual filters) ──────────
    # All take a raw 0..127 knob value. Center (64) = neutral pass-through
    # so the picture isn't disturbed when the knob is at rest. Twist away
    # from center to apply the effect; press fader-style fully one way
    # or the other for maximum effect.

    def _knob_saturation(self, value: int):
        """A EQ MID → mpv saturation. 64 = normal (1.0). 0 = grayscale.
        127 = oversaturated (~2.0). mpv property range: -100..+100."""
        if not self.player:
            return
        # Map 0..127 → -100..+100, centered on 64.
        sat = int(((value - 64) / 63.0) * 100)
        sat = max(-100, min(100, sat))
        try:
            self.player.player.saturation = sat
        except Exception as e:
            logger.debug(f"_knob_saturation failed: {e}")

    def _knob_contrast(self, value: int):
        """B EQ MID → mpv contrast. 64 = normal."""
        if not self.player:
            return
        c = int(((value - 64) / 63.0) * 100)
        c = max(-100, min(100, c))
        try:
            self.player.player.contrast = c
        except Exception as e:
            logger.debug(f"_knob_contrast failed: {e}")

    def _knob_hue(self, value: int):
        """B EQ MID → mpv hue rotation. 64 = no rotation. Twist for
        psychedelic color shifts."""
        if not self.player:
            return
        h = int(((value - 64) / 63.0) * 100)
        h = max(-100, min(100, h))
        try:
            self.player.player.hue = h
        except Exception as e:
            logger.debug(f"_knob_hue failed: {e}")

    def _knob_zoom(self, value: int):
        """A EQ LOW → mpv video-zoom (frame scale). The knob's bottom half
        is a neutral dead-zone (always exactly 1.0x) so a resting knob
        never disturbs the picture; twist UP from center to punch in. mpv
        video-zoom is log2 (1.0 → 2x), so 0..0.585 spans 1.0x .. ~1.5x —
        enough to pulse the frame to the music without losing the edges."""
        if not self.player:
            return
        if value <= 64:
            z = 0.0
        else:
            z = ((value - 64) / 63.0) * 0.585  # 64→1.0x ... 127→~1.5x
        try:
            self.player.player.video_zoom = z
        except Exception as e:
            logger.debug(f"_knob_zoom failed: {e}")

    # ── Tap tempo ────────────────────────────────────────────────────────
    # tap_tempo + _tap_times/_TAP_* were removed (Audit fix H9): the method
    # was wired to nothing — SHIFT+SYNC is bound to loop_length_nudge
    # (HALVE loop), see _init_s2. BPM comes from audio_reactive now. If a
    # manual tap-tempo is ever wanted again, bind a button to a fresh
    # implementation with an INSTANCE-level _tap_times list.

    # ── Random-next mode ────────────────────────────────────────────────
    # NOTE: _random_next and _auto_flip_use_folder are INSTANCE attributes
    # set in __init__ and guarded by self._auto_flip_lock (Audit fix H1/H2).

    def toggle_flip_on_beat(self):
        """FX1 Param 1 (or SHIFT+Loop In): cycle auto-flip mode through
        OFF → BANK → FOLDER → OFF. BPM lock keeps running regardless.

        - OFF: detector still tracks BPM; no flips fire.
        - BANK (default after OFF): flip cycles the active bank /
          scratch pool. The curated set the user just loaded. Falls
          back to folder siblings only when the bank is ≤1 file.
        - FOLDER: explicit override — cycle every video file in the
          current library folder, ignoring the bank.

        DECKS mode was removed 2026-05-16 per user req — deck firing
        belongs to S2 pads exclusively; auto-flip stays tied to banks.

        Runs on the S2 action-worker thread. Goes through AppState's
        set_flip_on_beat accessor (no reaching into state._lock) and
        guards the tri-state read+write with _auto_flip_lock so a
        concurrent toggle_random_next can't tear the mode. (Audit fix H1.)"""
        with self._auto_flip_lock:
            currently_on = self.state.get_flip_on_beat()
            currently_folder = self._auto_flip_use_folder
            if not currently_on:
                # OFF → BANK
                self.state.set_flip_on_beat(True)
                self._auto_flip_use_folder = False
                msg = "Auto-flip: BANK (cycles active bank / scratch)"
                led_brightness = 0x1F  # LED_MAX — binary: ON
                _flash_key = "on"        # green = BANK active
            elif not currently_folder:
                # BANK → FOLDER
                self.state.set_flip_on_beat(True)
                self._auto_flip_use_folder = True
                msg = "Auto-flip: FOLDER (cycles current library folder)"
                led_brightness = 0x1F  # LED_MAX — binary: ON (mode is the OSD)
                _flash_key = "amber"     # amber = FOLDER (secondary mode)
            else:
                # FOLDER → OFF
                self.state.set_flip_on_beat(False)
                self._auto_flip_use_folder = False
                msg = "Auto-flip: OFF (BPM lock still running)"
                led_brightness = 0x00  # LED_OFF — binary: OFF
                _flash_key = "off"       # red = OFF
        # Pad-flash to confirm new mode at a glance (unlit AUTO WR
        # button gives no feedback otherwise).
        self._pad_flash_toggle(_flash_key, label=f"auto-flip {msg.split(':')[1].strip().split()[0]}")
        # AUTO-FLIP REQUIRES AUDIO-REACTIVE: _on_beat() only fires flip
        # logic when audio_reactive_enabled is True. Without this, the
        # toggle just set state but no beats were detected → no flips.
        # When the user enables auto-flip, auto-start audio-reactive if
        # it isn't already running. We do NOT auto-stop audio-reactive
        # when auto-flip turns OFF — user may want BPM detection
        # running for other features (beat-pulse shader, OSC, etc.).
        if (self.state.get_flip_on_beat()
                and not self.state.audio_reactive_enabled
                and self.audio_reactive):
            logger.info("[auto-flip] starting audio-reactive (dependency)")
            try:
                self.audio_reactive_start()
            except Exception as e:
                logger.warning(f"audio-reactive auto-start failed: {e}")
        self.state.set_message(msg)
        # Light the A PFL cue button to mirror tri-state: off/dim/max.
        # Tri-state on a single LED gives the user an at-a-glance read
        # of the mode without needing the on-screen toaster.
        # INFO log so the user can verify in console whether the toggle
        # actually fired + what brightness it requested. Helps debug
        # "LED not lighting" complaints.
        logger.info(f"[s2-led] toggle_flip_on_beat: msg={msg!r} "
                    f"requesting a_pfl brightness=0x{led_brightness:02X} "
                    f"({led_brightness}) s2_connected={bool(self.s2)}")
        try:
            if self.s2:
                self.s2.set_led_state("a_pfl", led_brightness)
        except Exception as e:
            logger.warning(f"a_pfl LED set failed: {e}")

    def toggle_random_next(self):
        """SHIFT + FX1 Param 1: toggle whether flip() picks random vs sequential."""
        with self._auto_flip_lock:
            self._random_next = not self._random_next
            now_on = self._random_next
        self.state.set_message(f"Random flip: {'ON' if now_on else 'OFF'}")
        if self.s2:
            self.s2.flash_led("fx1_param1_btn")

    def clear_pending_in(self):
        """SHIFT+CUE: abandon the current IN marker without saving a clip."""
        cur_file = self.player.current_file if self.player else ""
        if not cur_file:
            return
        try:
            key = self.clips_db._key(cur_file)
            had = self.clips_db._pending_in.pop(key, None)
            self.state.set_pending_in(None)
            self.state.set_message(
                f"Cleared pending IN @ {had:.1f}s" if had is not None
                else "No pending IN to clear"
            )
        except Exception as e:
            logger.debug(f"clear_pending_in failed: {e}")

    # Crossfader state. Both flags initialize TRUE so the first event
    # at startup (often a value reflecting the rest position) cannot
    # fire a flip. They reset to False once the fader is seen near
    # center (proving the user actually moved it through the middle).
    _XFADER_FLIPPED_RIGHT = True
    _XFADER_FLIPPED_LEFT = True
    _XFADER_SEEN_CENTER = False

    # Alpha-blend crossfade state.
    # _xfade_fallback: flips to True if load_preview() ever fails (e.g. the
    # libmpv build has lavfi-complex disabled). After that we route the
    # crossfader callback to the legacy flip-on-midpoint behaviour so the
    # fader still does *something*.
    # _xfade_full_since: when did the fader cross the auto-promote
    # threshold? None = not currently dwelled. ~1s of dwell triggers
    # promote_preview() to swap preview into live.
    # _last_preview_filepath: cached so we don't re-attach the same file
    # every tick when deck 0 hasn't changed.
    _xfade_fallback: bool = False
    _xfade_full_since: Optional[float] = None
    _last_preview_filepath: Optional[str] = None
    _XFADE_AUTOPROMOTE_DWELL = 1.0  # seconds at full-preview before promote
    _XFADE_AUTOPROMOTE_THRESHOLD = 0.97  # opacity above this counts as "full"

    def _refresh_preview_from_deck(self):
        """Pull deck `preview_deck_idx` (default 0) into the player's
        preview slot. Call when deck contents change OR when blend is
        about to engage. No-op if the deck is empty / file missing /
        same file already loaded."""
        if not self.player:
            return
        slot = self.state.preview_deck_idx
        deck = self.decks_store.get(slot) if self.decks_store else None
        path = (deck or {}).get("filepath") if deck else None
        if not path:
            # Deck empty — drop any stale preview
            if self._last_preview_filepath is not None:
                self.player.load_preview(None)
                self._last_preview_filepath = None
                self.state.set_crossfade(preview_file=None, blend_active=False)
            return
        # Don't re-attach the same file (saves an expensive video-add)
        if path == self._last_preview_filepath and self.player.blend_active:
            return
        # If preview points at the live file, skip — blending a file with
        # itself is just a darken effect and is confusing.
        if path == self.player.current_file:
            return
        try:
            ok = self.player.load_preview(path)
            self._last_preview_filepath = path if ok else None
            self.state.set_crossfade(
                preview_file=path if ok else None,
                blend_active=ok,
            )
            if not ok:
                self._xfade_fallback = True
                logger.warning(
                    "load_preview returned False for %s — falling back to "
                    "flip-on-midpoint crossfader behaviour", path,
                )
        except Exception as e:
            logger.error("Preview attach failed: %s", e)
            self._xfade_fallback = True

    # _xfade_pending_value: latest crossfader value stashed by the S2
    # action-worker thread for the Qt thread to pick up (see crossfade_blend).
    _xfade_pending_value: Optional[int] = None

    def crossfade_blend(self, value: int):
        """S2 crossfader callback. RUNS ON THE S2 ACTION-WORKER THREAD.

        The real work (set_blend / load_preview / promote — each of which
        rebuilds mpv's lavfi-complex graph) MUST run on the Qt thread.
        Doing it from the worker thread stalls the worker — so jog and
        every other S2 input backs up behind it — and used to deadlock
        outright against a concurrent load() on the Qt thread. So we just
        stash the latest fader value and marshal the work over via a
        BOUND METHOD (lambdas scheduled by QTimer from a non-Qt thread get
        silently dropped — the project learned that the hard way)."""
        self._xfade_pending_value = int(value)
        QTimer.singleShot(0, self._crossfade_blend_apply)

    def _crossfade_blend_apply(self):
        """Qt thread — apply the latest stashed crossfader value. This is
        the old crossfade_blend body; it now only ever runs on the Qt
        thread, so every mpv mutation it triggers is naturally serialised
        with load_video / _do_promote / etc.

        S2 crossfader → real alpha-blend between live and preview deck.
        Value is 0..127 (calibrated). 0 = pure live, 127 = pure preview.
        Uses a sqrt curve so the perceived fade isn't lopsided.
        Auto-attaches deck `preview_deck_idx` (default 0) on demand and
        auto-promotes preview → live after a 1s dwell at the right edge.
        """
        value = self._xfade_pending_value
        if value is None:
            return
        self._xfade_pending_value = None

        position = max(0.0, min(1.0, value / 127.0))
        self.state.set_crossfade(position=position)

        if self._xfade_fallback or not self.player:
            self.crossfader_transition(value)
            return

        # Lazy attach: only attempt to load the preview track when the
        # user actually moves off pure-live. Attaching mid-startup (when
        # decks may still be loading filmstrips) just causes ugly logs.
        if position > 0.005 and not self.player.blend_active:
            self._refresh_preview_from_deck()
            if not self.player.blend_active:
                # Either deck 0 is empty or attach failed. If empty, hint
                # the user; if failed, the warning above already logged.
                if not self._xfade_fallback:
                    # Deck empty — silent no-op; ipad shows blend_active=False.
                    return
                self.crossfader_transition(value)
                return

        # sqrt curve for perceived equal-power: at 50% fader, both are
        # ~70% visible (looks like a real crossfade, not a fade-to-black-
        # then-fade-up). The lavfi blend filter does straight alpha-mix
        # so we have to shape the curve here.
        opacity = position ** 0.5 if position < 0.5 else 1.0 - (1.0 - position) ** 0.5
        # Above is equivalent to sqrt-symmetric: equal-power crossfade.
        # Hard-clamp the rails so the user gets a true "off" / "on" feel.
        if position < 0.01:
            opacity = 0.0
        elif position > 0.99:
            opacity = 1.0

        try:
            self.player.set_blend(opacity)
        except Exception as e:
            logger.debug("set_blend failed: %s", e)

        # Auto-promote: dwell at full preview for ~1s, then swap.
        if position >= self._XFADE_AUTOPROMOTE_THRESHOLD:
            now = time.time()
            if self._xfade_full_since is None:
                self._xfade_full_since = now
            elif now - self._xfade_full_since >= self._XFADE_AUTOPROMOTE_DWELL:
                self._xfade_full_since = None
                self._auto_promote_preview()
        else:
            self._xfade_full_since = None

    def _auto_promote_preview(self):
        """Promote preview → live (called after dwell at full-preview).
        Bounce onto the Qt thread because crossfade_blend runs from the
        S2 callback queue / Qt timer mix; player.load() is Qt-safe but
        we're already on Qt here. Still, defer a tick to let the current
        callback unwind cleanly."""
        if not self.player or not self.player.preview_file:
            return
        prev = self.player.preview_file
        logger.info("Auto-promote preview → live: %s", prev)
        QTimer.singleShot(0, self._do_promote)

    def _do_promote(self):
        if not self.player:
            return
        new_live = self.player.promote_preview()
        if not new_live:
            return
        self._last_preview_filepath = None
        self.state.set_crossfade(
            preview_file=None,
            blend_active=False,
            position=0.0,
        )
        # Mirror into config so re-launch picks the promoted file as last_video
        self.config["last_video"] = new_live
        self._save_config()
        self.label_file.setText(f"Loaded: {Path(new_live).name}")
        self.state.set_message(f"PROMOTED: {Path(new_live).name}")
        self._refresh_clips_for_ipad()

    def crossfader_transition(self, value: int):
        """Stop-gap 'transition' until proper deck crossfade lands:
        flip to next file when the crossfader is dragged through center
        and into the right edge; flip back on the left edge. Requires
        seeing center first — startup rest position never fires."""
        # Re-arm near center
        if 45 < value < 82:
            if not self._XFADER_SEEN_CENTER:
                self._XFADER_SEEN_CENTER = True
            self._XFADER_FLIPPED_RIGHT = False
            self._XFADER_FLIPPED_LEFT = False
            return
        if not self._XFADER_SEEN_CENTER:
            return  # never trip until user moves through center first
        if value > 95 and not self._XFADER_FLIPPED_RIGHT:
            self._XFADER_FLIPPED_RIGHT = True
            self.flip()
        elif value < 32 and not self._XFADER_FLIPPED_LEFT:
            self._XFADER_FLIPPED_LEFT = True
            self.flip_back()

    def set_audio_sensitivity(self, value: int):
        """Crossfader: map 0-127 -> 1.0-4.0 sensitivity, in real time.

        Lower number = more sensitive (smaller flux triggers a beat),
        higher number = stricter. We update both AudioReactive (live)
        and AppState (so the iPad shows the new value next poll).
        """
        sensitivity = round(1.0 + (value / 127) * 3.0, 2)
        if self.audio_reactive:
            self.audio_reactive.sensitivity = sensitivity
        self.state.set_audio_reactive(
            enabled=self.state.audio_reactive_enabled,
            sensitivity=sensitivity,
        )
        self.state.set_message(f"Sensitivity: {sensitivity:.1f}")
        self.config["audio_reactive_sensitivity"] = sensitivity

    def saturation_nudge(self, delta: int):
        """Encoder-driven relative saturation. Walks the mpv
        saturation property (-100..+100) by 5 per detent so a full
        edge-to-edge sweep is ~40 ticks. Bound to S2 A RIGHT encoder
        (was live-volume; freed 2026-05-18 since volume lives on the
        ATEM/mixer). Right side = LIVE deck, so saturation feels
        natural here -- you punch up the LIVE color in real time."""
        if not self.player or not self.player.player:
            return
        STEP = 5
        try:
            cur = int(self.player.player.saturation or 0)
        except Exception:
            cur = 0
        new = max(-100, min(100, cur + int(delta) * STEP))
        if new == cur:
            return
        try:
            self.player.player.saturation = new
            self.state.set_message(f"Sat: {new:+d}")
        except Exception as e:
            logger.debug(f"saturation_nudge failed: {e}")

    def crossfade_blend_nudge(self, delta: int):
        """Encoder-driven relative crossfade blend. Adds delta*4 to
        the stored crossfader value (0-127), clamps, dispatches via
        the standard crossfade_blend path (which marshals to Qt
        thread for the lavfi-graph rebuild). Bound to S2 B GAIN
        encoder (was brightness; freed since brightness lives on
        the projector / ATEM). B-side = preview, so crossfade
        blend feels natural here."""
        STEP = 4
        cur = int(getattr(self, "_xfade_nudge_value",
                          self._xfade_pending_value or 0) or 0)
        new = max(0, min(127, cur + int(delta) * STEP))
        if new == cur:
            return
        self._xfade_nudge_value = new
        self.state.set_message(f"Xfade: {new}")
        # Reuse the same Qt-thread-safe dispatch as the physical
        # crossfader path. _xfade_pending_value is the value the
        # apply step will read.
        self._xfade_pending_value = new
        QTimer.singleShot(0, self._crossfade_blend_apply)

    def audio_sensitivity_nudge(self, delta: int):
        """Encoder-driven relative sensitivity adjustment. Step =
        +/- 0.05 per detent so a full edge-to-edge sweep of the
        1.0-4.0 range is ~60 ticks -- enough resolution to dial in
        without being too slow. Bound to S2 B RIGHT encoder (was a
        duplicate of A RIGHT volume; unblocked 2026-05-18 overnight)."""
        STEP = 0.05
        MIN_S, MAX_S = 1.0, 4.0
        cur = float(getattr(self.audio_reactive, "sensitivity",
                            self.config.get("audio_reactive_sensitivity",
                                            2.5))
                    or 2.5)
        new = max(MIN_S, min(MAX_S, cur + (int(delta) * STEP)))
        new = round(new, 2)
        if abs(new - cur) < 1e-3:
            return
        if self.audio_reactive:
            self.audio_reactive.sensitivity = new
        self.state.set_audio_reactive(
            enabled=self.state.audio_reactive_enabled,
            sensitivity=new,
        )
        self.state.set_message(f"Sens: {new:.2f}")
        self.config["audio_reactive_sensitivity"] = new

    # ── Library browser ───────────────────────────────────────────────────

    def _init_library(self):
        """Scan the configured library folder on startup. Pushes folder
        contents into AppState so the iPad library panel renders immediately."""
        root = (self.config.get("library_root") or DEFAULT_LIBRARY).strip()
        folder = (self.config.get("library_folder") or root).strip()
        try:
            root_p = Path(root).resolve()
        except Exception:
            root_p = None
        if not root_p or not root_p.exists() or not root_p.is_dir():
            logger.warning(f"Library root missing: {root!r}")
            self.state.set_library(folder="", files=[], subfolders=[],
                                   rel_path="", root="")
            return
        # Reset folder to root if the saved folder is invalid or escaped root.
        try:
            folder_p = Path(folder).resolve()
            folder_p.relative_to(root_p)
            if not folder_p.exists() or not folder_p.is_dir():
                folder_p = root_p
        except Exception:
            folder_p = root_p
        self.config["library_root"] = str(root_p)
        self.config["library_folder"] = str(folder_p)
        self._save_config()
        self._publish_library(root_p, folder_p)

    def _scan_folder(self, folder: Path) -> tuple[list[str], list[str]]:
        """Return (subfolders, video_files) for `folder`, sorted case-insensitive,
        hiding entries starting with '.' or '~'. Filenames only — no full paths."""
        subs: list[str] = []
        files: list[str] = []
        try:
            for entry in folder.iterdir():
                name = entry.name
                if name.startswith(".") or name.startswith("~"):
                    continue
                try:
                    if entry.is_dir():
                        subs.append(name)
                    elif entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
                        files.append(name)
                except OSError:
                    # Permission denied / broken symlink — skip silently
                    continue
        except OSError as e:
            logger.warning(f"Cannot scan {folder}: {e}")
        subs.sort(key=str.lower)
        files.sort(key=str.lower)
        return subs, files

    def _publish_library_current(self):
        """Republish whatever folder is currently active (used after a
        path-tag filter change so the listing updates without a manual
        cd)."""
        try:
            root_str = self.state.library_root
            folder_str = self.state.library_folder or root_str
            if root_str and folder_str:
                self._publish_library(Path(root_str), Path(folder_str))
        except Exception as e:
            logger.debug(f"_publish_library_current failed: {e}")

    def _publish_library(self, root: Path, folder: Path):
        """Scan `folder` and push contents into AppState. Each file entry
        is a dict {name, hash, has_thumb} so the iPad can render thumbs
        without a second round trip. Also kicks off a low-priority
        background backfill that generates any missing thumbnails.

        Honors the active path-tag filter: if `state.path_tag_filter`
        is non-empty, files are restricted to those whose path-tag
        index entry matches ALL the listed tags."""
        subs, files = self._scan_folder(folder)
        # Apply path-tag filter if active. Empty filter = no restriction.
        try:
            active_filter = list(self.state.path_tag_filter or [])
        except Exception:
            active_filter = []
        if active_filter:
            try:
                matching_paths = self.path_tags.files_with_all_tags(active_filter)
                # Canonicalize BOTH sides so a case/separator mismatch
                # between how the scan walked the tree and how we build
                # query paths can't make the filter match nothing
                # (Audit fix H7).
                canon_matches = {canonical_path(p) for p in matching_paths}
                files = [
                    n for n in files
                    if canonical_path(folder / n) in canon_matches
                ]
            except Exception as e:
                logger.debug(f"path-tag filter apply failed: {e}")
        # Build the dict shape the iPad UI expects. Now ALSO injects
        # navigation entries — a ".." entry (unless we're at root) and
        # one entry per subfolder — at the top of the file list so the
        # browse encoder can scroll through folders + files in one
        # unified cursor. Each non-file entry carries `_kind` so the
        # press handler dispatches: file → fire, folder → navigate,
        # ".." → go up. iPad UI sees them as bracketed-name entries
        # mixed in with the files (visually distinct).
        try:
            rel = folder.relative_to(root).as_posix()
        except ValueError:
            rel = ""
        if rel == ".":
            rel = ""

        nav_entries = []
        if rel:
            # Only show ".." when we're below the library root.
            nav_entries.append({
                "name": "[..]",
                "hash": "",
                "has_thumb": False,
                "_kind": "up",
            })
        for sub in subs:
            nav_entries.append({
                "name": f"[{sub}]",
                "hash": "",
                "has_thumb": False,
                "_kind": "folder",
                "_target": sub,
            })

        file_entries = []
        for name in files:
            full = folder / name
            try:
                h = thumbnails.lib_thumbnail_hash(str(full))
                has_thumb = thumbnails.lib_thumbnail_path(h).exists()
            except Exception:
                h = ""
                has_thumb = False
            file_entries.append({"name": name, "hash": h, "has_thumb": has_thumb})

        unified = nav_entries + file_entries

        self.state.set_library(
            folder=str(folder),
            files=unified,
            subfolders=subs,
            rel_path=rel,
            root=str(root),
        )
        # Cursor lands on the first NAV entry (".." or first folder) so a
        # press goes somewhere useful. If nothing's there, jump to the
        # first file.
        self.state.set_library_selected_idx(0 if unified else -1)
        logger.info(f"Library: {folder} ({len(subs)} folders, {len(file_entries)} files, "
                    f"{len(nav_entries)} nav entries)")
        # Daemon thumbnail backfill — fire-and-forget. Idempotent: only
        # generates files whose hash file doesn't exist yet. Skip the
        # nav entries (they have no underlying file).
        self._start_lib_thumbnail_backfill(folder, [e["name"] for e in file_entries])

    def _lib_thumbnail_hash(self, filepath: str) -> str:
        """16-char sha256 hash of an absolute filepath. Cache key for
        library file thumbs. Thin wrapper around thumbnails.lib_thumbnail_hash
        so callers within main.py don't need to import the module."""
        return thumbnails.lib_thumbnail_hash(filepath)

    # Tracks the most recently kicked-off backfill folder so a fast
    # series of folder-changes doesn't pile up overlapping workers.
    _lib_backfill_lock = threading.Lock()
    _lib_backfill_folder: str = ""

    def _start_lib_thumbnail_backfill(self, folder: Path, names: list[str]):
        """Kick off (or restart) the lazy library-thumbnail backfill.

        Walks the file list slowly (100ms gap between files so it never
        competes with user actions). If the user navigates to another
        folder, the next call supersedes this one — the running worker
        bails as soon as it notices the folder changed. Daemon thread
        so it never blocks shutdown."""
        if not names:
            return
        folder_key = str(folder)
        with self._lib_backfill_lock:
            self._lib_backfill_folder = folder_key
        snapshot = list(names)

        def worker(target_folder=folder, target_key=folder_key, files=snapshot):
            try:
                thumbnails.ensure_thumbs_dir()
                for name in files:
                    # Bail out if the user navigated elsewhere.
                    with self._lib_backfill_lock:
                        if self._lib_backfill_folder != target_key:
                            return
                    full = target_folder / name
                    try:
                        h = thumbnails.lib_thumbnail_hash(str(full))
                        out = thumbnails.lib_thumbnail_path(h)
                        if out.exists():
                            time.sleep(0.005)
                            continue
                        thumbnails.generate_library_thumbnail(str(full))
                    except Exception as e:
                        logger.debug(f"Lib backfill error on {name}: {e}")
                    # Throttle so backfill doesn't compete with user actions.
                    time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Lib thumbnail backfill crashed: {e}")

        t = threading.Thread(
            target=worker, name="lib-thumb-backfill", daemon=True
        )
        t.start()

    def _resolve_inside_root(self, root: Path, candidate: Path) -> Optional[Path]:
        """Resolve `candidate` and ensure it sits inside `root`. Returns
        the resolved path, or None if it escapes root or doesn't exist."""
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
            return resolved
        except (ValueError, OSError):
            return None

    def library_scan(self, folder: str) -> dict:
        """POST /api/library/scan — point the library at a new folder.
        Resets BOTH library_root and library_folder (this is the only
        endpoint that can move the root boundary)."""
        folder = (folder or "").strip()
        if not folder:
            return {"ok": False, "error": "no folder given"}
        try:
            new_root = Path(folder).resolve()
        except Exception as e:
            return {"ok": False, "error": f"bad path: {e}"}
        if not new_root.exists() or not new_root.is_dir():
            return {"ok": False, "error": f"not a directory: {new_root}"}
        self.config["library_root"] = str(new_root)
        self.config["library_folder"] = str(new_root)
        self._save_config()
        self._publish_library(new_root, new_root)
        # Kick off path-tag re-scan of the new root in the background.
        # No-op if it's the same root as before (re-scan is cheap — mtime
        # check skips unchanged files).
        try:
            self.path_tags.scan_async(str(new_root))
        except Exception:
            pass
        self.state.set_message(f"Library: {new_root.name}")
        return {"ok": True, "folder": str(new_root)}

    def library_cd(self, path: str) -> dict:
        """POST /api/library/cd — navigate inside the current root.
        `path` can be ".." (up one), "" (root), a subfolder name, or a
        relative posix path. Absolute paths and any traversal that escapes
        root are rejected."""
        root_str = self.state.library_root
        cur_str = self.state.library_folder or root_str
        if not root_str:
            return {"ok": False, "error": "no library root configured"}
        root = Path(root_str)
        cur = Path(cur_str)

        path = (path or "").strip()
        # Reject absolute paths and Windows drive prefixes outright — only
        # relative navigation within root is allowed.
        if not path or path == "." or path == "/":
            target = root
        elif path == "..":
            target = cur.parent
        else:
            p = Path(path)
            if p.is_absolute() or (len(path) >= 2 and path[1] == ":"):
                return {"ok": False, "error": "absolute paths not allowed"}
            target = cur / path

        resolved = self._resolve_inside_root(root, target)
        if resolved is None:
            return {"ok": False, "error": "path outside library root"}
        if not resolved.exists() or not resolved.is_dir():
            return {"ok": False, "error": "not a directory"}
        self.config["library_folder"] = str(resolved)
        self._save_config()
        self._publish_library(root, resolved)
        return {"ok": True, "folder": str(resolved)}

    def library_load_file(self, filename: str) -> dict:
        """POST /api/library/load_file — load a video from the CURRENT
        library folder. Filename is a basename only; '..', path separators,
        and absolute paths are rejected."""
        filename = (filename or "").strip()
        if not filename:
            return {"ok": False, "error": "no filename"}
        # Filename must be a single path segment — no slashes, no ".."
        if filename in (".", "..") or "/" in filename or "\\" in filename:
            return {"ok": False, "error": "invalid filename"}
        if Path(filename).is_absolute() or (len(filename) >= 2 and filename[1] == ":"):
            return {"ok": False, "error": "absolute paths not allowed"}

        root_str = self.state.library_root
        cur_str = self.state.library_folder or root_str
        if not root_str:
            return {"ok": False, "error": "no library root configured"}
        root = Path(root_str)
        candidate = Path(cur_str) / filename
        resolved = self._resolve_inside_root(root, candidate)
        if resolved is None:
            return {"ok": False, "error": "path outside library root"}
        if not resolved.exists() or not resolved.is_file():
            return {"ok": False, "error": "file not found"}
        if resolved.suffix.lower() not in VIDEO_EXTS:
            return {"ok": False, "error": "not a video file"}
        # force=True — an explicit library tap should never be debounced.
        self.load_video(str(resolved), force=True)
        return {"ok": True, "file": str(resolved)}

    # ── Deck slots (launchpad) ───────────────────────────────────────────

    def _init_decks(self):
        """Mirror persisted decks into AppState; backfill missing filmstrips
        in a daemon thread so startup stays snappy. Also re-spawns the
        live MJPEG preview pipelines so the iPad picks up where it left
        off after a restart."""
        try:
            decks = self.decks_store.all()
            self.state.set_decks(decks)
            self.decks_store.backfill_filmstrips_async()
            # Re-attach previews for any deck restored from disk.
            for entry in decks:
                if entry and entry.get("filepath"):
                    self._start_deck_preview(entry)
        except Exception as e:
            logger.warning(f"Deck init failed: {e}")

    def _refresh_decks_for_ipad(self):
        """Push current deck slots into AppState for iPad rendering."""
        try:
            self.state.set_decks(self.decks_store.all())
        except Exception as e:
            logger.debug(f"Deck refresh failed: {e}")
        # If the preview-source deck just changed, invalidate the cached
        # preview filepath so the next crossfader tick re-attaches it.
        # We don't proactively load_preview here — that pulls a video-add
        # which costs ~50-200ms; doing it lazily on first fader move keeps
        # deck assignment instant.
        try:
            slot = self.state.preview_deck_idx
            cur = (self.decks_store.get(slot) or {}).get("filepath")
            if cur != self._last_preview_filepath:
                self._last_preview_filepath = None
                # If we're currently mid-blend, the existing graph is now
                # pointing at the OLD preview. Tear it down so the user
                # sees the change on next fader move.
                if self.player and self.player.blend_active:
                    try:
                        self.player.load_preview(None)
                    except Exception:
                        pass
                    self.state.set_crossfade(preview_file=None, blend_active=False)
        except Exception as e:
            logger.debug("Preview invalidation skipped: %s", e)

    # Per-filepath cached durations. ffprobe is 100-400 ms per call on
    # Windows (subprocess + binary launch + container parse) and
    # _compute_body_seek_target hits this on EVERY pad fire — that was
    # the source of the perceived 500 ms latency on bank pad fires when
    # the working-folder used to feel snappy (working files were
    # already-probed proxies; D:\Recycle Bin sources are cold). One
    # cache, file-keyed; no eviction needed at the rig's scale (a few
    # hundred unique files in active rotation, each <16 bytes per entry).
    _duration_cache: dict = {}

    def _probe_duration(self, filepath: str) -> float:
        """Best-effort duration probe for a video file. Cached after the
        first probe — subsequent calls for the same path are O(1) dict
        lookups. Uses ffprobe if available; falls back to 0 (caller
        must tolerate)."""
        cached = self._duration_cache.get(filepath)
        if cached is not None:
            return cached
        import shutil as _sh
        import subprocess as _sp
        ffprobe = _sh.which("ffprobe")
        if not ffprobe:
            return 0.0
        try:
            r = _sp.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nokey=1:noprint_wrappers=1", filepath],
                capture_output=True, timeout=4,
                creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0),
            )
            duration = float(
                (r.stdout or b"").decode("utf-8", "replace").strip() or 0.0)
            # Only cache successful probes (>0). A 0 might be a transient
            # ffprobe error — let next call retry.
            if duration > 0:
                self._duration_cache[filepath] = duration
            return duration
        except Exception:
            return 0.0

    def load_clip_to_deck(self, deck_idx: int, clip_id: str) -> dict:
        """Assign a saved clip to a deck slot, kick off filmstrip generation."""
        if not (0 <= deck_idx < 4):
            return {"ok": False, "error": "deck must be 0..3"}
        if not clip_id:
            return {"ok": False, "error": "no clip_id"}
        # Find clip by id
        clip = next((c for c in self.clips_db.get_all_clips() if c.get("id") == clip_id), None)
        if not clip:
            return {"ok": False, "error": "clip not found"}
        entry = make_deck_entry(
            slot=deck_idx,
            source_type="clip",
            source_id=clip_id,
            filepath=clip.get("filepath", ""),
            in_sec=float(clip.get("in_seconds", 0.0)),
            out_sec=float(clip.get("out_seconds", 0.0)),
            name=clip.get("name", "clip"),
        )
        self.decks_store.set(deck_idx, entry)
        # Pre-queue proxy for the source file behind this clip.
        try:
            self.proxy_cache.queue(clip.get("filepath", ""))
        except Exception:
            pass
        self._refresh_decks_for_ipad()
        self.decks_store.regenerate_filmstrip_async(deck_idx)
        # Spin up the live MJPEG preview for the iPad.
        self._start_deck_preview(entry)
        self.state.set_message(f"Deck {deck_idx + 1}: {entry['name']}")
        return {"ok": True, "deck": entry}

    def load_file_to_deck(self, deck_idx: int, library_filename: str) -> dict:
        """Assign a library file (relative basename) to a deck slot."""
        if not (0 <= deck_idx < 4):
            return {"ok": False, "error": "deck must be 0..3"}
        filename = (library_filename or "").strip()
        if not filename:
            return {"ok": False, "error": "no filename"}
        # Same hardening as library_load_file: basename only, no traversal.
        if filename in (".", "..") or "/" in filename or "\\" in filename:
            return {"ok": False, "error": "invalid filename"}
        if Path(filename).is_absolute() or (len(filename) >= 2 and filename[1] == ":"):
            return {"ok": False, "error": "absolute paths not allowed"}

        root_str = self.state.library_root
        cur_str = self.state.library_folder or root_str
        if not root_str:
            return {"ok": False, "error": "no library root configured"}
        root = Path(root_str)
        candidate = Path(cur_str) / filename
        resolved = self._resolve_inside_root(root, candidate)
        if resolved is None:
            return {"ok": False, "error": "path outside library root"}
        if not resolved.exists() or not resolved.is_file():
            return {"ok": False, "error": "file not found"}
        if resolved.suffix.lower() not in VIDEO_EXTS:
            return {"ok": False, "error": "not a video file"}

        duration = self._probe_duration(str(resolved))
        # If we can't probe duration, fall back to a 60s window so the
        # filmstrip still gets *some* sampling; the file plays full anyway.
        if duration <= 0:
            duration = 60.0
        entry = make_deck_entry(
            slot=deck_idx,
            source_type="file",
            source_id=str(resolved),
            filepath=str(resolved),
            in_sec=0.0,
            out_sec=duration,
            name=resolved.name,
        )
        self.decks_store.set(deck_idx, entry)
        # Pre-queue proxy for this file so it's 1080p-ready when fired.
        try:
            self.proxy_cache.queue(str(resolved))
        except Exception:
            pass
        self._refresh_decks_for_ipad()
        self.decks_store.regenerate_filmstrip_async(deck_idx)
        self._start_deck_preview(entry)
        self.state.set_message(f"Deck {deck_idx + 1}: {entry['name']}")
        return {"ok": True, "deck": entry}

    def _start_deck_preview(self, entry: dict) -> None:
        """Kick off the live MJPEG preview ffmpeg for a deck slot.
        Best-effort — never raises into the caller. If ffmpeg is
        missing the manager just no-ops and the iPad <img> 503s.

        Feeds the preview ffmpeg the 1080p PROXY when one exists, not the
        4K original — a 1080p decode is ~4x cheaper (CPU + RAM) than 4K,
        and for a 240x135 thumbnail the proxy is more than enough quality.
        Falls back to the original if no proxy is cached yet."""
        try:
            src = str(entry.get("filepath") or "")
            stream_path = src
            try:
                proxy = self.proxy_cache.get_proxy(src)
                if proxy:
                    stream_path = proxy
            except Exception:
                pass
            self.preview_streams.start_stream(
                deck_idx=int(entry.get("slot", 0)),
                filepath=stream_path,
                in_sec=float(entry.get("in_sec") or 0.0),
                out_sec=float(entry.get("out_sec") or 0.0) or None,
            )
        except Exception as e:
            logger.debug(f"start preview failed: {e}")

    # ── Path-tag index (auto-tags derived from folder/file names) ──────

    def _kick_path_tag_scan(self):
        root = (self.config.get("library_root") or "").strip()
        if not root:
            return
        started = self.path_tags.scan_async(root)
        if started:
            logger.info(f"Path-tag scan started: {root}")

    def _refresh_path_tags_top(self):
        try:
            top = self.path_tags.top_tags(limit=40)
            self.state.set_path_tags_top(
                [{"tag": t, "count": c} for t, c in top]
            )
        except Exception as e:
            logger.debug(f"_refresh_path_tags_top failed: {e}")

    def set_path_tag_filter(self, tags) -> dict:
        """iPad: set the active path-tag filter list. Empty list = clear.
        Library publish picks this up on next refresh."""
        if isinstance(tags, str):
            tags = [tags] if tags else []
        if not isinstance(tags, list):
            tags = []
        self.state.set_path_tag_filter([str(t) for t in tags])
        # Force a library republish so the filter takes immediate effect.
        try:
            self._publish_library_current()
        except Exception:
            pass
        return {"ok": True, "filter": list(self.state.path_tag_filter)}

    # ── Scratch list (session basket of library files) ─────────────────

    def _refresh_scratch_for_ipad(self):
        """Re-publish the scratch list to AppState. Also refresh the MK2
        pad LEDs so pads with content are lit."""
        try:
            self.state.set_scratch(self.scratch_store.all())
        except Exception as e:
            logger.debug(f"_refresh_scratch_for_ipad failed: {e}")
        self._refresh_mk2_pad_leds()

    # ── MK2 mono-LED byte offsets (per MK2_BUTTON_MAP.md discovery) ──
    # Confident mappings:
    _MONO_LED_CONTROL = 1   # pin-release indicator
    _MONO_LED_STEP = 2      # pin-file indicator
    _MONO_LED_BROWSE = 3    # bright when MK2 BROWSE mode active
    _MONO_LED_SAMPLING = 4
    _MONO_LED_PAGE_NEXT = 6 # `>` arrow — flashes briefly on bank fire
    _MONO_LED_ALL = 7
    _MONO_LED_AUTO_WR = 8
    # TBD: master section (25-30). TEMPO best-guess is 27.
    _MONO_LED_TEMPO = 27  # verify by pulsing alone; adjust if wrong

    # LED brightness levels
    _MONO_LED_OFF = 0
    _MONO_LED_DIM = 30
    _MONO_LED_MED = 100
    _MONO_LED_MAX = 220

    def _refresh_mk2_mono_leds(self):
        """Push current toggle state to the silkscreen-labeled button
        LEDs via report 0x82. Called from _refresh_mk2_pad_leds so it
        updates whenever pad LEDs do (every state change).

        Strategy: start with EVERY utility LED at DIM brightness so
        the surface looks alive (operator can see what buttons exist).
        Then override individual slots with their state-driven
        brightness (toggles, blinks, off). As we wire more buttons to
        real state, more slots become driven; un-wired ones stay
        dim-default."""
        if not self.mk2:
            return
        # Dim-default for the full 30-slot range. Specific overrides
        # below take precedence.
        leds = {off: self._MONO_LED_DIM for off in range(1, 31)}
        # ALL -- 3-state set-arc indicator, communicated via blink
        # because the hardware can't actually distinguish brightness
        # levels (medium vs max look identical to the human eye):
        #   set-arc OFF           = dark (steady)
        #   set-arc MANUAL phase  = max steady (always on)
        #   set-arc AUTO          = blinking at ~0.7Hz so the operator
        #                            instantly knows "the system is
        #                            driving phases"
        # The blink toggles the LED on/off based on a wall-clock period
        # so it's predictable across refreshes (no skipping/stutter).
        if getattr(self, "_set_arc_enabled", False):
            if getattr(self, "_set_arc_auto", False):
                # ~0.7Hz blink between MAX and DIM (not OFF) so the
                # button stays visibly lit even during the off-half.
                phase = int(time.time() * 1000) % 1400
                leds[self._MONO_LED_ALL] = (
                    self._MONO_LED_MAX if phase < 700
                    else self._MONO_LED_DIM
                )
            else:
                leds[self._MONO_LED_ALL] = self._MONO_LED_MAX
        else:
            leds[self._MONO_LED_ALL] = self._MONO_LED_DIM
        # AUTO WR — bright when auto-flip on, DIM at rest. The
        # bank-vs-folder mode distinction (previously encoded as
        # MAX vs MED brightness) doesn't read on the hardware -- the
        # iPad badge / state-message is the canonical mode display.
        leds[self._MONO_LED_AUTO_WR] = (
            self._MONO_LED_MAX if self.state.get_flip_on_beat()
            else self._MONO_LED_DIM
        )
        # TEMPO — bright when audio-react / tempo lock is on
        leds[self._MONO_LED_TEMPO] = (
            self._MONO_LED_MAX if self.state.audio_reactive_enabled
            else self._MONO_LED_DIM
        )
        # BROWSE — bright when MK2 browse-mode is active (right OLED
        # showing folder/file list). Mirrors the on/off state visually
        # so the operator sees at a glance whether pads / verticals
        # are in normal mode or in browse-readonly mode.
        leds[self._MONO_LED_BROWSE] = (
            self._MONO_LED_MAX
            if getattr(self, "_mk2_browse_mode", False)
            else self._MONO_LED_DIM
        )
        # ── Mode column (bytes 17-24: SCENE / PATTERN / PAD MODE /
        #    NAVIGATE / DUPLICATE / SELECT / SOLO / MUTE).
        #    CORRECTION 2026-05-18 overnight: an earlier version lit
        #    these bytes by "vertical-page-filled" assuming they
        #    corresponded to the verticals (bits 40-47). They do NOT.
        #    Mode column = its own button group, input bits 17-24,
        #    LED bytes 17-24, bound by the user to music-control
        #    actions in settings.json. Verticals = bits 40-47, LED
        #    byte mapping still unknown (probably 25-30 or no LEDs).
        #
        #    Light by ACTUAL state of the bound action:
        action_map = (self.config.get("mk2_button_map") or {})
        # Active bank letter (used by the bit-24 MUTE/bank:X state)
        active_bank_letter = ""
        try:
            active_bank_letter = (self.bank_store.active() or "").upper()
        except Exception:
            pass
        # Default whole column to DIM; override the toggle-state slots.
        # Press-only actions (tempo_nudge / reseek / bpm_tolerance)
        # stay DIM -- the existing pad-flash on press already shows
        # acknowledgement; trying to "light" a momentary press would
        # require timers + complicate the LED loop.
        for byte_off in range(17, 25):
            leds[byte_off] = self._MONO_LED_DIM
        # bit 18 (PATTERN) / tempo_lock_toggle -- light by audio-react
        # or tempo_lock state. tempo_lock is the iPad TEMPO badge target.
        if action_map.get("18") == "tempo_lock_toggle":
            tempo_on = (getattr(self.state, "audio_reactive_enabled", False)
                        or bool(getattr(self.state, "tempo_lock", False)))
            leds[18] = self._MONO_LED_MAX if tempo_on else self._MONO_LED_DIM
        # bit 22 (SELECT) / stutter_toggle -- light by stutter_active
        if action_map.get("22") == "stutter_toggle":
            stutter_on = bool(getattr(self.state, "stutter_active", False))
            leds[22] = self._MONO_LED_MAX if stutter_on else self._MONO_LED_DIM
        # bit 24 (MUTE) / bank:X -- light when that bank is currently
        # active. User has MUTE bound to bank:A.
        mute_action = action_map.get("24", "")
        if mute_action.startswith("bank:"):
            target = mute_action.split(":", 1)[1].upper()
            leds[24] = (self._MONO_LED_MAX if target == active_bank_letter
                        else self._MONO_LED_DIM)
        # pages / active_page_idx kept for the top-row loop below.
        pages = self.config.get("mk2_vertical_pages") or []
        active_page_idx = getattr(self, "_mk2_active_page", 0)
        # ── Top row (bytes 9-16 = page selectors 1-8). Light BRIGHT
        #    on the active page, DIM on the rest. Lets the operator
        #    see which page is active without looking at the right
        #    OLED's page-name display.
        for page_idx in range(8):
            byte_off = 9 + page_idx
            is_active = (page_idx == active_page_idx
                         and page_idx < len(pages))
            leds[byte_off] = (self._MONO_LED_MAX if is_active
                              else self._MONO_LED_DIM)
        try:
            self.mk2.set_mono_leds(leds, merge=True)
        except Exception as e:
            logger.debug(f"set_mono_leds failed: {e}")

    def _refresh_mk2_pad_leds(self):
        """Light MK2 pads to reflect scratch state and push a matching
        OLED status frame:
        - empty slot: off
        - has a scratch entry: dim cyan
        - matches current LIVE source: bright white
        Skips silently if MK2 isn't connected. Also drives the OLED
        screens (left = LIVE info, right = pad map)."""
        if not self.mk2:
            return
        try:
            files = self.scratch_store.all()
            live_path = (self._current_source_path
                         or (self.player.current_file if self.player else None) or "")
            colors = {}
            for i, path in enumerate(files[:16]):
                label = i + 1  # scratch idx 0..15 → pad label 1..16
                if path == live_path:
                    colors[label] = (255, 255, 255)  # bright white = LIVE
                else:
                    colors[label] = (0, 70, 90)      # dim cyan = has content

            # ─── Persistent toggle-state indicators (2026-05-17) ──────
            # User feedback: unlit hardware buttons (TEMPO, RESTART/AUTO
            # WR, SAMPLING, ALL) give no state feedback. Reserve the
            # RIGHTMOST PAD COLUMN (4, 8, 12, 16) as always-on status
            # indicators that override any clip color.
            #   pad 4  = SET-ARC  — off if disabled, phase color if on
            #   pad 8  = SPOTTER  — off if not running, green if running
            #   pad 12 = AUTO-FLIP — off / green (BANK) / amber (FOLDER)
            #   pad 16 = TEMPO LOCK — off if disabled, cyan if on
            # Status pads still FIRE their scratch clip if pressed — just
            # their color is locked to mode state instead of clip state.
            # User can press them for clips, look at them for state.
            status_overrides = {}
            # pad 4 — set-arc
            if getattr(self, "_set_arc_enabled", False):
                phase = getattr(self, "_set_arc_phase", "opening") or "opening"
                status_overrides[4] = self._PAD_FLASH_COLORS.get(
                    phase, (60, 60, 60))
            else:
                status_overrides[4] = (10, 10, 10)
            # pad 8 — (reserved)
            status_overrides[8] = (10, 10, 10)
            # pad 12 — auto-flip
            if self.state.get_flip_on_beat():
                if getattr(self, "_auto_flip_use_folder", False):
                    status_overrides[12] = (240, 160, 30)  # amber FOLDER
                else:
                    status_overrides[12] = (0, 180, 60)    # green BANK
            else:
                status_overrides[12] = (10, 10, 10)
            # pad 16 — tempo lock (== audio_reactive_enabled in current
            # design; TEMPO button toggles audio-react)
            if self.state.audio_reactive_enabled:
                status_overrides[16] = (40, 180, 220)  # cyan = locked
            else:
                status_overrides[16] = (10, 10, 10)
            # Apply overrides on top of clip colors
            colors.update(status_overrides)

            self.mk2.set_pad_colors_by_label(colors)

            # ─── Mono utility-button LEDs (2026-05-17) ────────────────
            # Light the actual silkscreen-labeled buttons via report 0x82
            # to reflect toggle state at-a-glance. Byte map per
            # MK2_BUTTON_MAP.md discovery session.
            try:
                self._refresh_mk2_mono_leds()
            except Exception as e:
                logger.debug(f"_refresh_mk2_mono_leds failed: {e}")
            # Push matching OLED frame. Render pad RGB in LABEL order
            # (1..16) — StatusRenderer expects that mapping.
            pad_rgb = []
            for label in range(1, 17):
                pad_rgb.append(colors.get(label, (0, 0, 0)))
            live_name = (Path(live_path).name if live_path else "")
            bpm = float(self.state.detected_bpm or 0.0)
            scratch_count = len(files)
            active_bank = self.bank_store.active() if hasattr(self, "bank_store") else "A"
            # Look up the bank's display name so the OLED can show it.
            bank_name = ""
            try:
                s = self.bank_store.get(active_bank) or {}
                bank_name = s.get("name") or ""
            except Exception:
                pass
            # MK2 vertical-page context for the LEFT OLED bottom strip.
            page_name = ""
            try:
                page_name = self._mk2_page_name(self._mk2_active_page) or ""
            except Exception:
                pass
            last_vertical = ""
            try:
                lv = getattr(self, "_mk2_last_vertical", None)
                if lv:
                    slot_idx, folder = lv
                    folder_name = Path(folder).name if folder else ""
                    last_vertical = f"v{int(slot_idx) + 1} {folder_name}".strip()
            except Exception:
                pass
            # MK2 BROWSE mode: pull library cursor state + slice a
            # window of names around the cursor so the OLED can
            # render a scrolling list. The window keeps the cursor
            # in row 1 (second visible row) so the user always sees
            # "what's next" -- typical file-browser convention.
            browse_mode = bool(getattr(self, "_mk2_browse_mode", False))
            browse_items: list = []
            browse_cursor = 0
            browse_header = ""
            # Build the 8-slot vertical labels for the right OLED's
            # 2-col x 4-row folder list. Each slot = {"label": "..."}.
            # Slot 7 is always "Reroll" (bound to reroll_banks in
            # _on_mk2_button_press regardless of page).
            verticals_for_oled: list = []
            active_page_idx_for_oled = 0
            try:
                pages_cfg = self.config.get("mk2_vertical_pages") or []
                active_page_idx_for_oled = getattr(
                    self, "_mk2_active_page", 0) or 0
                if not (0 <= active_page_idx_for_oled < len(pages_cfg)):
                    active_page_idx_for_oled = 0
                cur_page = (pages_cfg[active_page_idx_for_oled]
                            if active_page_idx_for_oled < len(pages_cfg)
                            else [])
                for slot in range(8):
                    if slot == 7:
                        verticals_for_oled.append({"label": "Reroll"})
                        continue
                    folder = (cur_page[slot] if slot < len(cur_page)
                              else None)
                    if folder:
                        verticals_for_oled.append({
                            "label": Path(folder).name or folder
                        })
                    else:
                        verticals_for_oled.append({"label": ""})
            except Exception as e:
                logger.debug(f"verticals_for_oled build failed: {e}")
                verticals_for_oled = []
            if browse_mode:
                try:
                    snap = self.state.get_library_snapshot()
                    files = snap.get("files") or []
                    sel = int(snap.get("selected_idx") or 0)
                    n = len(files)
                    if n > 0:
                        # 5-row window with cursor pinned at index 1
                        # (one above-context, four below). Clamp.
                        WIN = 5
                        start = max(0, min(sel - 1, n - WIN))
                        end = min(n, start + WIN)
                        names = []
                        for i in range(start, end):
                            e = files[i]
                            name = (e.get("name") if isinstance(e, dict)
                                    else str(e))
                            names.append(f"{i+1:3d} {name}")
                        browse_items = names
                        browse_cursor = sel - start
                        folder = snap.get("folder") or snap.get("root") or ""
                        folder_label = Path(folder).name if folder else ""
                        browse_header = (
                            f"{folder_label[:24]}  ({sel+1}/{n})"
                            if folder_label else f"({sel+1}/{n})"
                        )
                    else:
                        browse_items = ["(library empty)"]
                        browse_header = "BROWSE"
                except Exception as e:
                    logger.debug(f"browse snapshot failed: {e}")
                    browse_items = ["(snapshot error)"]
                    browse_header = "BROWSE"
            self.mk2.push_oled_status(
                live_filename=live_name,
                bpm=bpm,
                scratch_count=scratch_count,
                pad_rgb=pad_rgb,
                active_bank=active_bank,
                bank_name=bank_name,
                page_name=page_name,
                last_vertical=last_vertical,
                browse_mode=browse_mode,
                browse_items=browse_items,
                browse_cursor=browse_cursor,
                browse_header=browse_header,
                verticals=verticals_for_oled,
                active_page_idx=active_page_idx_for_oled,
            )
        except Exception as e:
            logger.debug(f"_refresh_mk2_pad_leds failed: {e}")

    def scratch_add(self, path: str) -> dict:
        """Append a file path to the scratch basket. Idempotent."""
        if not path or not Path(path).exists():
            return {"ok": False, "error": "path missing"}
        added = self.scratch_store.add(path)
        if added:
            self._refresh_scratch_for_ipad()
            self.state.set_message(f"Scratch + {Path(path).name}")
        return {"ok": True, "added": added}

    def scratch_add_library_file(self, filename: str) -> dict:
        """Resolve a library filename → full path → add to scratch.
        Same path-safety check as library_load_file. Used by the iPad's
        per-row + button."""
        filename = (filename or "").strip()
        if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
            return {"ok": False, "error": "invalid filename"}
        root_str = self.state.library_root
        cur_str = self.state.library_folder or root_str
        if not root_str:
            return {"ok": False, "error": "no library root configured"}
        candidate = Path(cur_str) / filename
        resolved = self._resolve_inside_root(Path(root_str), candidate)
        if resolved is None or not resolved.exists():
            return {"ok": False, "error": "file not found"}
        return self.scratch_add(str(resolved))

    def scratch_add_current(self) -> dict:
        """Add whatever's currently LIVE to the scratch basket."""
        src = (self._current_source_path
               or (self.player.current_file if self.player else None))
        if not src:
            return {"ok": False, "error": "no live video"}
        return self.scratch_add(src)

    def scratch_remove(self, path: str) -> dict:
        ok = self.scratch_store.remove(path)
        if ok:
            self._refresh_scratch_for_ipad()
        return {"ok": ok}

    def scratch_clear(self) -> dict:
        n = self.scratch_store.clear()
        self._refresh_scratch_for_ipad()
        self.state.set_message(f"Scratch cleared ({n} entries)")
        return {"ok": True, "removed": n}

    def scratch_shuffle(self) -> dict:
        n = self.scratch_store.shuffle()
        self._refresh_scratch_for_ipad()
        self.state.set_message(f"Scratch shuffled ({n} entries)")
        return {"ok": True, "count": n}

    def scratch_save_set(self, name: str) -> dict:
        files = self.scratch_store.all()
        if not files:
            return {"ok": False, "error": "scratch is empty"}
        s = self.scratch_set_store.save_set(name, files)
        self._refresh_scratch_sets_for_ipad()
        self.state.set_message(f"Scratch set saved: {s.get('name')}")
        return {"ok": True, "set": {"id": s.get("id"), "name": s.get("name")}}

    def scratch_load_set(self, set_id: str) -> dict:
        s = self.scratch_set_store.get(set_id)
        if not s:
            return {"ok": False, "error": "set not found"}
        n = self.scratch_store.replace_all(s.get("files") or [])
        self._refresh_scratch_for_ipad()
        self.state.set_message(f"Scratch set loaded: {s.get('name')} ({n})")
        return {"ok": True, "count": n}

    def scratch_delete_set(self, set_id: str) -> dict:
        ok = self.scratch_set_store.delete(set_id)
        if ok:
            self._refresh_scratch_sets_for_ipad()
        return {"ok": ok}

    # ── Analytics ──────────────────────────────────────────────────────

    def _refresh_analytics(self):
        """Aggregate analytics blob → push to AppState. Cheap (cached
        10s internally). Resets today-counters at local midnight."""
        if not self.analytics:
            return
        try:
            # Midnight reset check
            cur_mday = time.localtime().tm_mday
            if self._analytics_today.get("last_mday") != cur_mday:
                self._analytics_today = {"beats": 0, "flips": 0, "fires": 0, "last_mday": cur_mday}
            top = self.analytics.top_played(10) or []
            hist = self.analytics.bpm_histogram() or []
            disk = self.analytics.disk_usage() or {}
            today_base = self.analytics.today_stats() or {}
            today = dict(today_base)
            today.update({
                "beats_today": int(self._analytics_today.get("beats", 0)),
                "flips_today": int(self._analytics_today.get("flips", 0)),
                "fires_today": int(self._analytics_today.get("fires", 0)),
            })
            self.state.set_analytics({
                "top_clips": top,
                "bpm_histogram": hist,
                "disk_usage": disk,
                "today": today,
            })
        except Exception as e:
            logger.debug(f"_refresh_analytics failed: {e}")

    # ── Magic-mix recommender ──────────────────────────────────────────

    def _record_fire_history(self, path: str):
        """Tracks the last N fired paths for the recency penalty in magic-mix
        and increments the today-counter for analytics."""
        if not path:
            return
        self._fire_history.append(path)
        if len(self._fire_history) > 20:
            self._fire_history = self._fire_history[-20:]
        try:
            self._analytics_today["fires"] = int(self._analytics_today.get("fires", 0)) + 1
        except Exception:
            pass

    def magic_suggest(self, top_n: int = 5) -> dict:
        """iPad/HTTP: ask the recommender for what to play next."""
        if not self.magic:
            return {"ok": False, "error": "magic mix unavailable"}
        try:
            live = (self._current_source_path
                    or (self.player.current_file if self.player else None) or "")
            bpm = float(self.state.detected_bpm or 0.0)
            scratch = self.scratch_store.all()
            # TTL-gated: a full reload re-reads clips.json + scratch.json +
            # the entire path_tags table — too heavy to do on every ~15s
            # iPad poll. reload_if_stale() throttles it to ~once a minute.
            # (Audit fix M13.) The candidates list is passed in live below,
            # so scratch changes still show up immediately regardless.
            self.magic.reload_if_stale()
            suggestions = self.magic.suggest_next(
                live_path=live,
                live_bpm=bpm,
                recent_fires=list(self._fire_history)[-10:],
                candidates=scratch,
                top_n=int(top_n or 5),
            )
            return {"ok": True, "suggestions": suggestions}
        except Exception as e:
            logger.warning(f"magic_suggest failed: {e}")
            return {"ok": False, "error": str(e)}

    # ── Smart playlists (saved root + tag-filter combos) ───────────────

    def _refresh_smart_playlists_for_ipad(self):
        try:
            self.state.set_smart_playlists(self.smart_playlists.all())
        except Exception as e:
            logger.debug(f"_refresh_smart_playlists_for_ipad failed: {e}")

    def smart_playlist_save(self, name: str) -> dict:
        """Snapshot current (library_root + path-tag filter) as a named search."""
        root = (self.state.library_root or "").strip()
        tags = list(self.state.path_tag_filter or [])
        if not root:
            return {"ok": False, "error": "no library root active"}
        s = self.smart_playlists.save(name, root, tags)
        self._refresh_smart_playlists_for_ipad()
        self.state.set_message(f"Smart playlist saved: {s.get('name')}")
        return {"ok": True, "id": s.get("id")}

    def smart_playlist_apply(self, pid: str) -> dict:
        s = self.smart_playlists.get(pid)
        if not s:
            return {"ok": False, "error": "playlist not found"}
        root = (s.get("library_root") or "").strip()
        tags = list(s.get("tags") or [])
        if root and Path(root).is_dir():
            self.library_scan(root)
        self.set_path_tag_filter(tags)
        self.state.set_message(f"Smart playlist: {s.get('name')}")
        return {"ok": True}

    def smart_playlist_delete(self, pid: str) -> dict:
        ok = self.smart_playlists.delete(pid)
        if ok:
            self._refresh_smart_playlists_for_ipad()
        return {"ok": ok}

    # ── Banks (8-slot scratch-list quick-switch) ───────────────────────

    def _refresh_banks_for_ipad(self):
        try:
            self.state.set_banks(self.bank_store.all_slots())
        except Exception as e:
            logger.debug(f"_refresh_banks_for_ipad failed: {e}")

    def bank_save(self, letter: str, name: str) -> dict:
        """Snapshot current scratch list into bank slot `letter` (A-H)."""
        files = self.scratch_store.all()
        ok = self.bank_store.save_into(letter, name, files)
        if ok:
            self._refresh_banks_for_ipad()
            self.state.set_message(f"Bank {letter.upper()} saved ({len(files)} files)")
        return {"ok": ok}

    # ── Bank auto-reroll on track change ─────────────────────────────────
    # Each bank holds 16 clips queried by tag-overlap. With static banks,
    # an hour set cycles the same 128 files (16 × 8) — staleness within
    # a couple of songs. Re-rolling on every track change pulls a
    # fresh 16 per theme per song; the no-repeat memory excludes recent
    # fires so the same clip won't pop up again for ~200 plays.
    BANK_THEMES = [
        ('A', 'Bank A', ['color:warm']),
        ('B', 'Bank B', ['color:cool']),
        ('C', 'Bank C', ['motion:dynamic', 'complexity:6', 'complexity:7']),
        ('D', 'Bank D', ['motion:static', 'motion:smooth', 'complexity:1']),
        ('E', 'Bank E', ['geometry:linear', 'geometry:polygons']),
        ('F', 'Bank F', ['geometry:particles', 'geometry:masks']),
        ('G', 'Bank G', ['symmetry:cohesive']),
        ('H', 'Bank H', ['symmetry:offset', 'complexity:9', 'motion:jumpy']),
    ]
    # Per-bank signature color for the MK2 pad-load sweep animation.
    # Picked to be visually distinct so that after a session or two the
    # operator starts associating "the cyan sweep" with bank D etc. --
    # peripheral-vision pattern recognition.

    BANK_THEME_COLORS = {
        'A': (255,  60, 130),  # pink
        'B': (255,   0, 200),  # magenta
        'C': (255, 200,   0),  # yellow
        'D': (  0, 200, 255),  # cyan
        'E': ( 60, 255,  60),  # green
        'F': (255,  60,   0),  # red
        'G': (200,  60, 255),  # purple
        'H': (255, 130,   0),  # orange
    }
    # Inactive-bank dim factor — applied to all NON-active group LEDs
    # so you can still see what color each letter is at a glance, but
    # the active bank obviously stands out. 0.18 ≈ 18% of full bright;
    # bright enough to read color in stage light, dim enough to make
    # the active bank "pop." Tunable if it feels off.
    _GROUP_LED_DIM_FACTOR: float = 0.18

    # MK2 transport LED brightness levels
    _TRANSPORT_LED_DIM: int = 40       # baseline "available" — visible but not loud
    _TRANSPORT_LED_ACTIVE: int = 200   # play/rec when actively engaged

    def _refresh_mk2_transport_leds(self):
        """Paint the MK2 lower transport row. PLAY brightens when video
        is actually playing; others stay at a dim baseline so they're
        visible peripherally without dominating. Safe to call from any
        thread (MK2 driver serializes its own writes)."""
        if not self.mk2:
            return
        # Default: every transport button dim-on (visible).
        bright = {
            "restart": self._TRANSPORT_LED_DIM,
            "left":    self._TRANSPORT_LED_DIM,
            "right":   self._TRANSPORT_LED_DIM,
            "grid":    self._TRANSPORT_LED_DIM,
            "play":    self._TRANSPORT_LED_DIM,
            "rec":     self._TRANSPORT_LED_DIM,
            "erase":   self._TRANSPORT_LED_DIM,
            "shift":   self._TRANSPORT_LED_DIM,
        }
        # PLAY brightens when the mpv player is actually playing.
        try:
            if self.player and self.player.player:
                paused = bool(self.player.player.pause)
                bright["play"] = (self._TRANSPORT_LED_DIM if paused
                                  else self._TRANSPORT_LED_ACTIVE)
        except Exception:
            pass
        try:
            self.mk2.set_transport_leds(bright)
        except Exception as e:
            logger.debug(f"_refresh_mk2_transport_leds failed: {e}")

    def _refresh_mk2_group_leds(self):
        """Paint Group A-H LEDs on the MK2: active bank in its full
        theme color, others dimmed. Call on startup + every bank
        change. No-op if MK2 isn't connected."""
        if not self.mk2:
            return
        try:
            active = (self.bank_store.active()
                      if hasattr(self, "bank_store") else "A")
        except Exception:
            active = "A"
        active = (active or "A").upper()
        dim = self._GROUP_LED_DIM_FACTOR
        colors = {}
        for letter, theme_rgb in self.BANK_THEME_COLORS.items():
            if letter == active:
                colors[letter] = theme_rgb
            else:
                colors[letter] = tuple(int(c * dim) for c in theme_rgb)
        try:
            self.mk2.set_group_leds(colors)
        except Exception as e:
            logger.debug(f"_refresh_mk2_group_leds failed: {e}")
    _RECENT_CLIPS_MAX = 200

    def _remember_recent_clip(self, path: str) -> None:
        """Record a fired clip in the rolling no-repeat memory. Bounded
        deque — older entries fall out automatically."""
        if not path:
            return
        if not hasattr(self, "_recent_clips") or self._recent_clips is None:
            from collections import deque as _deque
            self._recent_clips = _deque(maxlen=self._RECENT_CLIPS_MAX)
        self._recent_clips.append(path)

    # Video extensions that count as bank-eligible. Keep aligned with
    # the rest of the app (library / scratch use the same set).
    _BANK_FOLDER_EXTS = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v")
    # Cache of (folder_lock -> [paths]) so repeated rerolls don't
    # re-walk the disk every time. Invalidated by _bump_folder_cache.
    _bank_folder_cache: dict = {}

    def _collect_folder_clips(self, folder: str, shuffle: bool = True) -> list:
        """Recursive video scan of `folder`. Returns a shuffled list of
        absolute paths (string). Results cached per-folder until
        explicitly bumped — folders rarely change mid-set, and the walk
        is hot path on every reroll. Pass shuffle=False for stable order
        (mostly for tests)."""
        import os as _os
        import random as _random
        key = folder.strip().replace("\\", "/").rstrip("/")
        if not key:
            return []
        cached = self._bank_folder_cache.get(key)
        if cached is None:
            base = Path(key)
            if not base.exists() or not base.is_dir():
                logger.warning(f"Bank folder missing: {key}")
                self._bank_folder_cache[key] = []
                return []
            found: list = []
            exts = self._BANK_FOLDER_EXTS
            try:
                for root, _dirs, files in _os.walk(base):
                    for fname in files:
                        if fname.lower().endswith(exts):
                            found.append(str(Path(root) / fname))
            except Exception as e:
                logger.warning(f"folder walk failed {key}: {e}")
            self._bank_folder_cache[key] = found
            logger.info(f"Bank folder scan {key}: {len(found)} clip(s)")
            cached = found
        if shuffle:
            out = list(cached)
            _random.shuffle(out)
            return out
        return list(cached)

    def bump_bank_folder_cache(self, folder: str = "") -> None:
        """Invalidate the folder-scan cache. Pass a specific folder to
        drop just that entry; empty drops everything. Call after adding
        new files to a folder you want reflected on next reroll."""
        if not folder:
            self._bank_folder_cache.clear()
            return
        key = folder.strip().replace("\\", "/").rstrip("/")
        self._bank_folder_cache.pop(key, None)

    def bind_bank_folder(self, letter: str, folder: str) -> dict:
        """Public API: lock bank `letter` to a source folder (empty
        string clears the lock). Immediately rerolls JUST THAT BANK
        (not all 8 — which was the old behavior; wasteful + slow).
        Callable from settings, iPad, MK2 vertical buttons."""
        ok = self.bank_store.save_folder(letter, folder)
        if not ok:
            return {"ok": False, "error": "invalid bank letter"}
        self.bump_bank_folder_cache(folder)
        # Reroll JUST this one bank — pull 16 fresh clips from the
        # newly-locked folder.
        self._reroll_single_bank(letter)
        self._refresh_banks_for_ipad()
        return {"ok": True, "letter": letter.upper(), "folder": folder}

    # ── Bank candidate pulling (shared by _reroll_single_bank +
    # reroll_banks). Extracted 2026-05-17 morning (post-audit) to drop
    # ~75 lines of near-identical SQL + filtering logic that lived in
    # both methods. Change the picking algorithm in ONE place now. ──
    def _pull_bank_candidates(
        self, letter: str, recent: "set[str]",
    ) -> "tuple[list[str], str]":
        """Pick 16 candidate clips for ONE bank slot.

        Mode selection:
          - folder-locked (existing bank has non-empty ``folder``):
            random walk that folder, exclude recent fires, backfill
            from non-fresh if <16.
          - theme-tag (default): query path_tags.db3 for clips
            matching BANK_THEMES tags, same fresh/backfill logic,
            sparse-tag floor (≥10 total tags) + quality:bad exclusion.

        Returns ``(fresh_paths, mode)`` where mode is ``"folder"`` /
        ``"tag"`` / ``"skip"`` (skip = early failure, caller should
        leave the existing bank untouched).
        """
        ltr = (letter or "A").upper()
        theme_entry = next(
            (t for t in self.BANK_THEMES if t[0] == ltr), None)
        if not theme_entry:
            return [], "skip"
        _, _, tags = theme_entry
        existing_bank = self.bank_store.get(ltr) or {}
        folder_lock = (existing_bank.get("folder") or "").strip()
        # Folder-locked branch — filesystem walk, no DB.
        if folder_lock:
            candidates = self._collect_folder_clips(folder_lock)
            fresh = [p for p in candidates if p not in recent][:16]
            if len(fresh) < 16:
                seen = set(fresh)
                for p in candidates:
                    if p not in seen:
                        fresh.append(p)
                        seen.add(p)
                        if len(fresh) >= 16:
                            break
            return fresh, "folder"
        # Theme-tag branch — path_tags.db3 query.
        import sqlite3
        db_path = CONFIG_DIR / "path_tags.db3"
        if not db_path.is_file():
            return [], "skip"
        try:
            conn = sqlite3.connect(str(db_path))
            ph = ",".join("?" for _ in tags)
            # Sparse-tag filter (≥10 total tags per file) matches the
            # lyric picker — same junk excluded. Quality:bad files
            # (score <30) never served from banks.
            rows = conn.execute(
                f"SELECT filepath, COUNT(*) AS m FROM file_tags "
                f"WHERE tag IN ({ph}) "
                f"  AND filepath IN ("
                f"    SELECT filepath FROM file_tags "
                f"    GROUP BY filepath HAVING COUNT(*) >= ?"
                f"  ) "
                f"  AND filepath NOT IN ("
                f"    SELECT filepath FROM file_tags "
                f"    WHERE tag = 'quality:bad'"
                f"  ) "
                f"GROUP BY filepath "
                f"ORDER BY m DESC, RANDOM() "
                f"LIMIT 60",
                tuple(tags) + (10,),
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.warning(f"bank candidates DB query failed for {ltr}: {e}")
            return [], "skip"
        # Prefer fresh; backfill with recent if we'd fall short.
        fresh = [r[0] for r in rows if r[0] not in recent][:16]
        if len(fresh) < 16:
            seen = set(fresh)
            for r in rows:
                if r[0] not in seen:
                    fresh.append(r[0])
                    seen.add(r[0])
                    if len(fresh) >= 16:
                        break
        return fresh, "tag"

    def _ensure_recent_clips(self) -> "set[str]":
        """Lazy-init the recent-clips deque + return a snapshot set
        of its current contents (shared by reroll paths)."""
        from collections import deque as _deque
        if not hasattr(self, "_recent_clips") or self._recent_clips is None:
            self._recent_clips = _deque(maxlen=self._RECENT_CLIPS_MAX)
        return set(self._recent_clips)

    def _reroll_single_bank(self, letter: str) -> None:
        """Re-pull 16 clips for ONE bank slot. Honors folder lock if
        set; falls back to BANK_THEMES tag query otherwise. Same
        recent-fires filter as the all-banks reroll uses.

        Avoids the 'rerolling all 8 banks every time' wasted work that
        happens when a vertical button just wants to refresh one slot.

        Implementation is now a thin wrapper over _pull_bank_candidates
        (refactored 2026-05-17 — was ~75 lines duplicating reroll_banks)."""
        recent = self._ensure_recent_clips()
        ltr = (letter or "A").upper()
        fresh, mode = self._pull_bank_candidates(ltr, recent)
        if mode == "skip":
            return
        theme_entry = next(
            (t for t in self.BANK_THEMES if t[0] == ltr), None)
        default_name = theme_entry[1] if theme_entry else ""
        existing_bank = self.bank_store.get(ltr) or {}
        name = existing_bank.get("name") or default_name
        self.bank_store.save_into(ltr, name, fresh)
        folder_lock = (existing_bank.get("folder") or "").strip()
        logger.info(f"Bank {ltr} rerolled ({len(fresh)} clips, "
                    f"{'folder=' + Path(folder_lock).name if folder_lock else 'theme'})")

    def reroll_banks(self, reload_active: bool = False) -> dict:
        """Re-pull a fresh 16 clips per bank, excluding clips in
        the recent-fire memory. By default updates the bank STORE only —
        the active bank's pad surface DOESN'T change mid-tap (that'd be
        jarring). Pass reload_active=True to also refresh the live pads
        immediately. The next time the user taps a Group letter, the
        new contents land naturally.

        Per-bank picking lives in _pull_bank_candidates (shared with
        _reroll_single_bank). Refactored 2026-05-17 — was 75 lines of
        duplicate SQL + filtering."""
        recent = self._ensure_recent_clips()
        rolled = []
        try:
            for letter, default_name, _tags in self.BANK_THEMES:
                fresh, mode = self._pull_bank_candidates(letter, recent)
                if mode == "skip":
                    continue
                existing_bank = self.bank_store.get(letter) or {}
                name = existing_bank.get("name") or default_name
                self.bank_store.save_into(letter, name, fresh)
                rolled.append((letter, name, len(fresh), mode))
        except Exception as e:
            logger.warning(f"reroll_banks failed: {e}")
            return {"ok": False, "error": str(e)}
        self._refresh_banks_for_ipad()
        active = self.bank_store.active()
        if reload_active and active:
            self.bank_load(active)
        log_summary = ", ".join(
            f"{l}({c}{'F' if m == 'folder' else ''})"
            for l, _, c, m in rolled)
        logger.info(f"Banks rerolled (recent-exclude): {log_summary}  "
                    f"recent_memory={len(recent)}")
        return {"ok": True, "rolled": [
            {"letter": l, "name": n, "count": c, "mode": m}
            for l, n, c, m in rolled]}

    def bank_load(self, letter: str) -> dict:
        """Load bank `letter` into the scratch list. Replaces current
        basket. Also fires a 3s OLED flash on the MK2's left screen
        showing the bank name + clip count + first two file names —
        hardware confirmation that the switch registered."""
        s = self.bank_store.get(letter)
        if not s:
            return {"ok": False, "error": "invalid bank letter"}
        self.bank_store.set_active(letter)
        # New pool = new vibe. Re-roll the cohesion anchor so we don't
        # carry over the previous bank's "music-video shoot."
        self._force_cohesion_refresh(reason=f"bank_load({letter})")
        try:
            self.session_log.record_bank_switch(letter=letter.upper())
        except Exception:
            pass
        files = s.get("files") or []
        n = self.scratch_store.replace_all(files)
        self._refresh_scratch_for_ipad()
        self._refresh_banks_for_ipad()
        # Bank label: prefer the LIVE layer-aware category name over the
        # saved bank name. So if banks were last loaded under default
        # layer ("D=dancing") but user has since cycled to positions
        # layer, OLED shows "D=doggy" instead of stale "D=dancing".
        nm_saved = s.get("name") or f"Bank {letter.upper()}"
        try:
            cats = self._get_active_layer_categories()
            cat = cats.get(letter.upper(), {}) if cats else {}
            cat_name = cat.get("name", "")
            layer_name = self._get_active_layer_name()
        except Exception:
            cat_name = ""
            layer_name = ""
        if cat_name:
            # Build display name like "Strapon D=doggy" (folder prefix
            # if available from saved name, then live category).
            # Saved name often looks like "Strapon D=dancing" so strip
            # the "=X" suffix and replace with the live category.
            base = nm_saved
            if "=" in base:
                base = base.split("=", 1)[0].rstrip(" " + letter.upper())
            base = base.strip() or nm_saved
            nm = f"{base} {letter.upper()}={cat_name}"
            if layer_name and layer_name != "default":
                nm += f" [{layer_name}]"
        else:
            nm = nm_saved
        self.state.set_message(f"Bank {letter.upper()} loaded: {nm} ({n})")
        # OLED flash — gives the user immediate visual feedback on the
        # MK2 itself that the bank switched. Persists for 3s, then the
        # normal now-playing display takes over.
        try:
            if self.mk2:
                self.mk2.push_oled_bank_flash(
                    letter=letter.upper(),
                    name=nm,
                    count=n,
                    first_files=files[:2],
                    hold_seconds=3.0,
                )
        except Exception as e:
            logger.debug(f"bank flash OLED push failed: {e}")
        # Pad LED sweep — column-fill animation in the bank's signature
        # color, ~300ms. After the sweep completes, refresh pad LEDs
        # back to the normal scratch-state colors. The sweep is on its
        # own daemon thread; the on_done callback marshals back to Qt
        # so the refresh happens on the right thread.
        try:
            if self.mk2:
                theme = self.BANK_THEME_COLORS.get(
                    letter.upper(), (180, 180, 220))
                def _restore_pads():
                    QTimer.singleShot(0, self._refresh_mk2_pad_leds)
                self.mk2.play_bank_load_sweep(theme, on_done=_restore_pads)
        except Exception as e:
            logger.debug(f"bank sweep failed: {e}")
        # Repaint Group A-H LEDs so the now-active bank lights bright
        # and the previously-active one fades to its dim theme color.
        self._refresh_mk2_group_leds()
        return {"ok": True, "count": n}

    # ── Stem-separation OSC handler ──────────────────────────────────
    # Called on the OSC listener thread when stem_daemon.py emits
    # /stem/drums/onset. Adds the onset to a rolling history + sets a
    # short boost-active window. Cheap; non-blocking. Picker / flip
    # logic can read self._stem_drum_boost_until to decide whether
    # to apply the 1.5x boost.
    _STEM_BOOST_WINDOW_S = 0.20    # how long an onset stays "active"
    _STEM_RECENT_MAX = 60          # rolling history cap

    def _on_stem_drum_onset(self, strength: float) -> None:
        now = time.time()
        try:
            s = float(strength)
        except (TypeError, ValueError):
            s = 0.0
        self._stem_recent_onsets.append((now, s))
        # Trim to the most recent N entries
        if len(self._stem_recent_onsets) > self._STEM_RECENT_MAX:
            del self._stem_recent_onsets[:-self._STEM_RECENT_MAX]
        # Open the boost window
        self._stem_drum_boost_until = now + self._STEM_BOOST_WINDOW_S
        # Quiet log -- one per onset is OK at ~kick-rate
        logger.debug(f"[stem] drums onset s={s:.3f}")

    def mode_summary(self) -> dict:
        """Snapshot of every operator-facing 'mode' for the iPad Mode
        Dashboard. Each entry is {label, value, state} where `state`
        drives the pill colour: off / on / alt / hot / neutral.

        Every read is best-effort getattr — a missing or half-wired
        subsystem shows a safe default, never crashes the state poll.
        Order here is the display order on the iPad."""
        import time as _t
        m: dict = {}

        # Auto-flip — tri-state OFF / BANK / FOLDER.
        try:
            on = bool(self.state.get_flip_on_beat())
            folder = bool(getattr(self, "_auto_flip_use_folder", False))
            m["autoflip"] = {
                "label": "Auto-Flip",
                "value": "OFF" if not on else ("FOLDER" if folder else "BANK"),
                "state": "off" if not on else ("alt" if folder else "on"),
            }
        except Exception:
            pass

        # Audio-react + live BPM.
        try:
            ar = getattr(self, "audio_reactive", None)
            ar_on = bool(getattr(self.state, "audio_reactive_enabled", False))
            bpm = 0.0
            if ar is not None:
                _cb = getattr(ar, "current_bpm", None)
                bpm = float((_cb() if callable(_cb) else _cb) or 0.0)
            m["audio"] = {
                "label": "Audio React",
                "value": ((f"{bpm:.0f} BPM" if bpm else "ON")
                          if ar_on else "OFF"),
                "state": "on" if ar_on else "off",
            }
        except Exception:
            pass

        # Set-arc — OFF, or PHASE + AUTO/MANUAL.
        try:
            sa_on = bool(getattr(self, "_set_arc_enabled", False))
            phase = str(getattr(self, "_set_arc_phase", "opening")).upper()
            auto = bool(getattr(self, "_set_arc_auto", False))
            m["setarc"] = {
                "label": "Set-Arc",
                "value": ("OFF" if not sa_on else
                          f"{phase} {'AUTO' if auto else 'MAN'}"),
                "state": ("off" if not sa_on else
                          ("hot" if phase == "PEAK" else "on")),
            }
        except Exception:
            pass

        # Bank categorisation layer (STEP cycles it — easy to lose track).
        try:
            layer = str(self._get_active_layer_name())
            m["layer"] = {"label": "Bank Layer", "value": layer.upper(),
                          "state": "neutral" if layer == "default" else "alt"}
        except Exception:
            pass

        # MK2 vertical page (only when pages are configured).
        try:
            pages = self.config.get("mk2_vertical_pages") or []
            if pages:
                idx = getattr(self, "_mk2_active_page", 0) or 0
                m["mk2page"] = {
                    "label": "MK2 Page",
                    "value": f"{idx + 1}·{self._mk2_page_name(idx)}",
                    "state": "neutral",
                }
        except Exception:
            pass

        # Cohesion anchor lock.
        try:
            until = float(getattr(self, "_cohesion_until", 0) or 0)
            coh = until > _t.time()
            m["cohesion"] = {"label": "Cohesion",
                             "value": "LOCKED" if coh else "—",
                             "state": "on" if coh else "off"}
        except Exception:
            pass

        # Hold (current clip pinned).
        try:
            held = bool(getattr(self, "_hold_clip", None))
            m["hold"] = {"label": "Hold",
                         "value": "HELD" if held else "—",
                         "state": "hot" if held else "off"}
        except Exception:
            pass

        # Blackout.
        try:
            bo = bool(getattr(self, "_blackout_active", False))
            m["blackout"] = {"label": "Blackout",
                             "value": "ON" if bo else "—",
                             "state": "hot" if bo else "off"}
        except Exception:
            pass

        return m

    def stem_status(self) -> dict:
        """Snapshot of stem-listener state for /api/state + iPad
        visibility."""
        now = time.time()
        # onsets within the last 60 sec
        recent = [o for o in self._stem_recent_onsets if (now - o[0]) <= 60]
        last_age = (now - recent[-1][0]) if recent else None
        return {
            "enabled": self._stem_listener is not None
                       and self._stem_listener.is_running(),
            "port": int(self.config.get("stem_osc_listen_port", 0) or 0),
            "onsets_60s": len(recent),
            "last_onset_age_s": (round(last_age, 2)
                                 if last_age is not None else None),
            "boost_active": now < self._stem_drum_boost_until,
        }

    def _create_clip_segment_action(self, data: dict) -> dict:
        """iPad Mark-clip form handler. Accepts:
          file:  full path OR a fragment that resolves to a unique
                 path via path_tags. Required.
          in:    seconds (number). Required.
          out:   seconds (number). Required.
          name:  optional clip name.
          tags:  optional list[str].
          starred: optional bool.

        Path resolution: if `file` doesn't exist as a literal path,
        we LIKE-search path_tags for it. Multiple matches → error
        (caller should be more specific)."""
        from pathlib import Path as _P
        raw_file = str(data.get("file", "") or "").strip()
        if not raw_file:
            return {"ok": False, "error": "missing 'file'"}
        # Try literal path first
        fp = None
        if _P(raw_file).is_file():
            fp = raw_file
        else:
            # LIKE-search path_tags
            try:
                import sqlite3
                db = sqlite3.connect(
                    f"file:{self.path_tags._db_path}?mode=ro",
                    uri=True, timeout=1.0,
                )
                rows = db.execute(
                    "SELECT filepath FROM files WHERE filepath "
                    "LIKE ? COLLATE NOCASE LIMIT 5",
                    (f"%{raw_file}%",),
                ).fetchall()
                db.close()
            except Exception as e:
                return {"ok": False,
                        "error": f"path lookup failed: {e}"}
            if not rows:
                return {"ok": False,
                        "error": f"no file matches '{raw_file}'"}
            if len(rows) > 1:
                names = [_P(r[0]).name for r in rows[:5]]
                return {
                    "ok": False,
                    "error": f"ambiguous '{raw_file}' matches "
                              f"{len(rows)} files",
                    "matches": names,
                }
            fp = rows[0][0]
        # Hand off to ClipDatabase.create_segment
        return self.clips_db.create_segment(
            filepath=fp,
            in_seconds=data.get("in", 0),
            out_seconds=data.get("out", 0),
            name=str(data.get("name", "") or ""),
            tags=list(data.get("tags") or []),
            starred=bool(data.get("starred", False)),
        )

    def bank_clear(self, letter: str) -> dict:
        ok = self.bank_store.clear_slot(letter)
        if ok:
            self._refresh_banks_for_ipad()
        return {"ok": ok}

    def bank_reroll_one(self, letter: str) -> dict:
        """Reroll JUST one bank's contents (16 new candidates) without
        touching the others. Honors folder lock if set; otherwise
        pulls from BANK_THEMES tag query. Wraps the internal
        `_reroll_single_bank` + publishes refresh."""
        ltr = (letter or "").upper().strip()
        if not ltr or ltr not in "ABCDEFGH":
            return {"ok": False, "error": "invalid bank letter"}
        try:
            self._reroll_single_bank(ltr)
            self._refresh_banks_for_ipad()
            slot = self.bank_store.get(ltr) or {}
            return {
                "ok": True,
                "letter": ltr,
                "count": len(slot.get("files") or []),
                "name": slot.get("name", ""),
            }
        except Exception as e:
            logger.error(f"bank_reroll_one({ltr}) failed: {e}",
                         exc_info=True)
            return {"ok": False, "error": str(e)}

    def bank_auto_split(self, *,
                        seed: int | None = None,
                        prefix: str = "") -> dict:
        """Snapshot the currently-filtered library file list into all
        8 banks A-H, shuffled then evenly distributed (last bank
        absorbs any remainder).

        The "currently-filtered" list is whatever's visible in the
        library browser RIGHT NOW -- so the iPad workflow is:
          1. Tap tag chips to narrow ("vampires", "scary", ...)
          2. Tap AUTO-BANK button -> all 8 banks built from that filter
          3. Press MK2 Group A..H to swap to each curated set live

        Bank names are built from the active path-tag filter so the
        operator can see what's in each bank (e.g. "vampires A",
        "vampires B"). When no filter is active, the current
        folder's name is used.

        Args:
          seed: optional RNG seed for reproducible splits (mostly
                for testing).
          prefix: override the auto-derived bank name prefix. Empty
                  string = derive from filter / folder."""
        snap = self.state.get_library_snapshot()
        files = snap.get("files") or []
        folder = snap.get("folder") or snap.get("root") or ""
        # Extract real file entries (skip nav entries: up, folders).
        file_names: list[str] = []
        for e in files:
            if isinstance(e, dict):
                if e.get("_kind") in ("up", "folder"):
                    continue
                n = e.get("name") or ""
            else:
                n = str(e)
            if n:
                file_names.append(n)
        paths = [str(Path(folder) / n) for n in file_names
                 if folder and n]
        # Resolve to existing-file paths only.
        existing = [p for p in paths if Path(p).is_file()]
        if not existing:
            return {"ok": False, "error": "no files in current filter"}

        import random as _random
        if seed is not None:
            _rng = _random.Random(seed)
            _rng.shuffle(existing)
        else:
            _random.shuffle(existing)

        # Name prefix: active filter > folder name > "auto"
        try:
            filter_tags = list(self.state.path_tag_filter or [])
        except Exception:
            filter_tags = []
        if prefix:
            base_name = prefix
        elif filter_tags:
            # Strip "namespace:" prefixes for readability
            short = [t.split(":")[-1] for t in filter_tags]
            base_name = " ".join(short)[:24]
        else:
            base_name = Path(folder).name[:24] if folder else "auto"

        # Adaptive bucket count: when fewer than 8 files, use only as
        # many banks as needed (e.g. 5 files -> A..E with 1 file each)
        # and LEAVE the unused letters untouched -- preserves whatever
        # bank you had loaded on F-H beforehand. Avoids the "split 5
        # files into 8 banks gives 7 dud letters" failure mode.
        bank_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        n = len(existing)
        n_banks = min(n, 8)
        per = max(1, n // n_banks) if n_banks else 0
        out_counts: dict[str, int] = {}
        for i, ltr in enumerate(bank_letters):
            if i >= n_banks:
                # Leave this slot untouched -- whatever was loaded
                # before survives, and the operator sees a clear
                # "this bank wasn't part of the new split" via the
                # untouched bank-preview UI.
                out_counts[ltr] = -1   # sentinel: "skipped"
                continue
            start = i * per
            end = start + per if i < (n_banks - 1) else n
            bank_files = existing[start:end]
            if bank_files:
                # Folder lock = the source `folder` if provided, else
                # clear (tag-only split has no anchor folder to set).
                lock_val = (
                    str(folder).replace("\\", "/") if folder else ""
                )
                self.bank_store.save_into(
                    ltr,
                    f"{base_name} {ltr}",
                    bank_files,
                    set_folder_lock=lock_val,
                )
            out_counts[ltr] = len(bank_files)

        try:
            self._refresh_banks_for_ipad()
        except Exception:
            pass
        try:
            total_assigned = sum(out_counts.values())
            self.state.set_message(
                f"AUTO-BANK: {total_assigned} files -> "
                f"{base_name} A-H ({per} per bank)"
            )
        except Exception:
            pass
        logger.info(
            f"[bank/auto_split] name='{base_name}' total={n} "
            f"per_bank={per} dist={out_counts}"
        )
        return {
            "ok": True,
            "name": base_name,
            "total": n,
            "per_bank": per,
            "distribution": out_counts,
        }

    def _tag_search_action(self, q: str, limit: int = 30) -> dict:
        """Substring-match path_tags. Returns the top matches by
        file count so the iPad search input can show a live list."""
        q = (q or "").strip()
        if not q:
            return {"ok": True, "query": q, "results": []}
        try:
            matches = self.path_tags.search_tags(q, limit=limit)
        except Exception as e:
            logger.debug(f"tag search failed: {e}")
            return {"ok": False, "error": str(e), "results": []}
        return {
            "ok": True,
            "query": q,
            "results": [{"tag": t, "count": c} for t, c in matches],
        }

    def _tag_to_banks_action(self, q: str) -> dict:
        """One-shot: take a query string ("abigail"), find the
        best-matching path tag, pull EVERY file tagged with it
        across the whole library (NOT just the current folder), and
        auto-split into banks A-H. Hot path for the iPad
        search-and-bank flow.

        Strategy for picking the "best" tag when multiple match:
          1. Exact match wins (case-insensitive)
          2. Otherwise prefer the tag with the most files (top of
             search result), so 'abigail' -> 'abigaiil-morris' (19)
             not 'performer:abigaiil-morris-curvy' (1)."""
        q = (q or "").strip()
        if not q:
            return {"ok": False, "error": "empty query"}
        try:
            matches = self.path_tags.search_tags(q, limit=10)
        except Exception as e:
            return {"ok": False, "error": f"search failed: {e}"}
        if not matches:
            return {"ok": False, "error": f"no tag matches '{q}'"}
        # Pick the tag with the most files. Exact-match was the old
        # strategy but data quirks (typo'd "abigaiil-morris" has 19
        # files; exact "abigail" only 1) mean what the user typed
        # rarely IS the canonical tag. They typed it to FIND files,
        # not to literally match. Highest-count gives them files.
        # `matches` is already sorted by count desc; matches[0] wins.
        best_tag = matches[0][0]
        # Pull EVERY file tagged with best_tag from the global path-tag
        # index. The previous version set the iPad filter + called
        # bank_auto_split, but that snapshot only includes files in the
        # current folder -- so if you typed "abigail" while sitting in
        # D:\Recycle Bin root, the bank built from 2 loose files
        # instead of the 19 in D:\Recycle Bin\Abigaiil.Morris\.
        try:
            tagged_paths = self.path_tags.files_with_all_tags([best_tag])
        except Exception as e:
            return {"ok": False, "error": f"tag lookup failed: {e}"}
        if not tagged_paths:
            return {"ok": False,
                    "error": f"tag '{best_tag}' has no files indexed"}
        existing = [p for p in tagged_paths if Path(p).is_file()]
        if not existing:
            return {"ok": False,
                    "error": f"no existing files for '{best_tag}' "
                              f"(stale path-tag index?)"}
        # Apply as filter so the iPad library view reflects the bank
        # we just built. Best-effort; failure here is non-fatal.
        try:
            self.state.set_path_tag_filter([best_tag])
            self._publish_library_current()
        except Exception:
            pass
        # Bank name prefix from the matched tag for clarity.
        short_name = best_tag.split(":")[-1][:24]
        # Build the primary split first: shuffle exact-match files
        # across as many banks as there are files (adaptive count).
        import random as _random
        _random.shuffle(existing)
        bank_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        n = len(existing)
        n_banks = min(n, 8)
        per = max(1, n // n_banks) if n_banks else 0

        primary_banks: dict[str, list[str]] = {}
        for i, ltr in enumerate(bank_letters):
            if i >= n_banks:
                primary_banks[ltr] = []
                continue
            start = i * per
            end = start + per if i < (n_banks - 1) else n
            primary_banks[ltr] = list(existing[start:end])

        # Pad each populated bank up to TARGET_BANK_SIZE using
        # tag-similarity-ranked candidates -- not just any co-tag.
        # The similar_files() helper scores each non-primary file by
        # how much of the primary set's tag profile it shares (with
        # generic tags like color: motion: excluded), so we get
        # "adjacent" content: same studio's other releases, same
        # performer's frequent costars, same action / scene type,
        # etc. NOT "any clip with breasts."
        TARGET_BANK_SIZE = 14
        padding_pool: list[str] = []
        try:
            similar = self.path_tags.similar_files(
                existing, limit=8 * TARGET_BANK_SIZE
            )
            logger.info(
                f"[tag_to_banks] top similar to '{best_tag}': "
                + ", ".join(
                    f"{Path(p).name[:30]}(s={s})"
                    for p, s in similar[:3]
                )
            )
            padding_pool = [
                p for p, _s in similar if Path(p).is_file()
            ]
            logger.info(
                f"[tag_to_banks] similarity pool: {len(padding_pool)} "
                f"ranked candidates available for top-up"
            )
        except Exception as e:
            logger.debug(f"similarity padding skipped: {e}", exc_info=True)

        out_counts: dict[str, int] = {}
        pad_idx = 0
        for ltr in bank_letters:
            primary = primary_banks.get(ltr, [])
            if not primary:
                # Empty bank -- leave untouched (preserve previous
                # contents). Sentinel -1 tells the response shape.
                out_counts[ltr] = -1
                continue
            # Top up with padding files until we hit TARGET_BANK_SIZE
            # or we run out of padding.
            need = max(0, TARGET_BANK_SIZE - len(primary))
            if need > 0 and pad_idx < len(padding_pool):
                take = padding_pool[pad_idx:pad_idx + need]
                pad_idx += len(take)
                bank_files = primary + take
            else:
                bank_files = primary
            # Tag-driven split has no source folder to lock to —
            # clear any stale lock so reroll falls back to theme tags.
            self.bank_store.save_into(
                ltr, f"{short_name} {ltr}", bank_files,
                set_folder_lock="",
            )
            out_counts[ltr] = len(bank_files)
        try:
            self._refresh_banks_for_ipad()
        except Exception:
            pass
        try:
            self.state.set_message(
                f"AUTO-BANK[{short_name}]: {sum(out_counts.values())} "
                f"files -> A-H ({per} per bank)"
            )
        except Exception:
            pass
        logger.info(
            f"[tag_to_banks] q='{q}' tag='{best_tag}' total={n} "
            f"per_bank={per} dist={out_counts}"
        )
        return {
            "ok": True,
            "query": q,
            "matched_tag": best_tag,
            "candidates": [t for t, _ in matches[:5]],
            "name": short_name,
            "total": n,
            "per_bank": per,
            "distribution": out_counts,
        }

    def _clip_to_banks_action(self, q: str, top: int = 64) -> dict:
        """One-shot: take a NATURAL-LANGUAGE query ("rooftop sunset
        glow"), run CLIP semantic search, then categorically split the
        top-N matches into banks A-H using the active layer's category
        map. Hot path for the iPad "type a vibe → instantly playable
        banks" workflow.

        Differs from _tag_to_banks_action in two ways:
          1. Uses CLIP embeddings (visual semantics) not tag-substring
             match. So "neon city night" finds clips with that LOOK,
             even if no tag literally says it.
          2. Splits by current bank-layer categories (A, B, ..., per
             the active layer's definitions) rather than
             by tag presence. The top-N matches get sorted into
             whichever bank their content fits best.
        """
        q = (q or "").strip()
        if not q:
            return {"ok": False, "error": "empty query"}

        # 1) Run semantic search. Wide net — top 64 by default so each
        #    of the 8 banks gets ~8 candidates to chew on.
        try:
            search = self.clip_search(q, top=int(top))
        except Exception as e:
            return {"ok": False, "error": f"clip_search failed: {e}"}
        if not search.get("ok"):
            return {"ok": False, "error": search.get("error",
                                                    "clip_search failed")}
        hits = search.get("results") or []
        if not hits:
            return {"ok": False, "error": f"no matches for '{q}'"}

        # 2) Filter to existing files (CLIP embeddings can outlive the
        #    file system if a clip was moved/deleted without re-indexing).
        existing = [
            h["path"] for h in hits
            if h.get("path") and Path(h["path"]).is_file()
        ]
        if not existing:
            return {"ok": False,
                    "error": f"no on-disk files for '{q}' "
                              "(stale embeddings?)"}

        # 3) Hand off to the categorical split (same path as
        #    folder→banks and tag→banks). _split_files_into_banks
        #    does: per-file category scoring, similarity padding,
        #    vote-promotion, save_into each letter.
        short_name = q[:24]
        try:
            result = self._split_files_into_banks(
                existing, short_name=short_name, source_folder=None,
            )
        except Exception as e:
            logger.warning(f"clip_to_banks split failed: {e}",
                           exc_info=True)
            return {"ok": False, "error": f"split failed: {e}"}

        try:
            self._refresh_banks_for_ipad()
        except Exception:
            pass
        try:
            n_assigned = sum(
                v for v in (result.get("distribution") or {}).values()
                if v > 0
            )
            self.state.set_message(
                f"AUTO-BANK[{short_name}]: {n_assigned} clips "
                f"({search.get('elapsed_ms', 0)}ms search)"
            )
        except Exception:
            pass
        # MK2 OLED flash — show the query + dist on the left screen
        # so the operator sees the routing took effect from across the
        # room. ~2.5s hold, then normal now-playing display takes over.
        try:
            if self.mk2:
                self.mk2.push_oled_clip_to_banks_flash(
                    query=q, matches=len(existing),
                    distribution=result.get("distribution") or {},
                    elapsed_ms=search.get("elapsed_ms", 0),
                )
        except Exception as e:
            logger.debug(f"clip_to_banks OLED flash failed: {e}")
        logger.info(
            f"[clip_to_banks] q='{q}' top={top} matches={len(existing)} "
            f"elapsed_ms={search.get('elapsed_ms', 0)} "
            f"dist={result.get('distribution')}"
        )
        return {
            "ok": True,
            "query": q,
            "name": short_name,
            "elapsed_ms": search.get("elapsed_ms", 0),
            "matches": len(existing),
            "distribution": result.get("distribution") or {},
            "top_results": hits[:8],   # for iPad preview
        }

    # ── Curated theme layer ───────────────────────────────────────
    # theme: tags are derived by rule (theme_apply.py) from the tags
    # already in the DB. Unlike CLIP search (visual guess) or raw tag
    # match (too granular), themes give a small, stable, curated set
    # of vibes the operator can grab in one move. Canonical order is
    # also the A-H / MK2-top-row order.
    _THEME_ORDER = ["dance", "rave", "peak", "horror",
                    "bright", "dark", "opener", "chill"]

    def _theme_list(self) -> dict:
        """POST /api/theme/list — distinct theme: tags + file counts,
        in canonical order. Powers the iPad theme-chip strip."""
        try:
            with self.path_tags._lock:
                rows = self.path_tags._conn.execute(
                    "SELECT tag, COUNT(*) FROM file_tags "
                    "WHERE tag LIKE 'theme:%' GROUP BY tag"
                ).fetchall()
        except Exception as e:
            return {"ok": False, "error": f"db query failed: {e}"}
        counts = {t.split(":", 1)[1]: int(n) for t, n in rows}
        themes = [
            {"name": th, "count": counts[th]}
            for th in self._THEME_ORDER if counts.get(th, 0) > 0
        ]
        # Append any non-canonical themes that exist (forward-compat).
        for th, n in sorted(counts.items()):
            if th not in self._THEME_ORDER:
                themes.append({"name": th, "count": n})
        return {"ok": True, "themes": themes}

    def _theme_to_banks_action(self, theme: str) -> dict:
        """POST /api/bank/theme — take a curated theme name ('horror',
        'dance', ...), pull every file carrying theme:<theme>, and
        categorically split into banks A-H (same split path as
        clip_to_banks / tag_to_banks). The theme tag is also applied
        as the library filter so the iPad list reflects it."""
        theme = (theme or "").strip().lower()
        if not theme:
            return {"ok": False, "error": "no theme given"}
        tag = "theme:" + theme
        try:
            tagged = self.path_tags.files_with_all_tags([tag])
        except Exception as e:
            return {"ok": False, "error": f"theme lookup failed: {e}"}
        existing = [p for p in tagged if Path(p).is_file()]
        if not existing:
            return {"ok": False,
                    "error": f"no files for theme '{theme}'"}
        try:
            self.state.set_path_tag_filter([tag])
            self._publish_library_current()
        except Exception:
            pass
        try:
            result = self._split_files_into_banks(
                existing, short_name="theme " + theme,
                source_folder=None,
            )
        except Exception as e:
            logger.warning(f"theme_to_banks split failed: {e}",
                           exc_info=True)
            return {"ok": False, "error": f"split failed: {e}"}
        try:
            self._refresh_banks_for_ipad()
        except Exception:
            pass
        dist = result.get("distribution") or {}
        try:
            n_assigned = sum(v for v in dist.values() if v > 0)
            self.state.set_message(
                f"THEME[{theme}]: {n_assigned} clips across banks"
            )
        except Exception:
            pass
        try:
            if self.mk2:
                self.mk2.push_oled_clip_to_banks_flash(
                    query=f"THEME {theme}", matches=len(existing),
                    distribution=dist, elapsed_ms=0,
                )
        except Exception as e:
            logger.debug(f"theme_to_banks OLED flash failed: {e}")
        logger.info(
            f"[theme_to_banks] theme='{theme}' "
            f"matches={len(existing)} dist={dist}"
        )
        return {
            "ok": True, "theme": theme,
            "matches": len(existing), "distribution": dist,
        }

    # ── Saved CLIP-search "vibes" ─────────────────────────────────
    # Persisted set of natural-language queries the user has starred.
    # Renders as one-tap chips on the iPad so productive vibes stick
    # around between sessions.
    _SAVED_VIBES_MAX = 12   # cap; oldest fall off
    _SAVED_VIBES_PATH_KEY = "_saved_vibes_path"

    def _saved_vibes_path(self) -> "Path":
        return CONFIG_DIR / "saved_searches.json"

    def _saved_vibes_load(self) -> list[dict]:
        import json as _json
        p = self._saved_vibes_path()
        if not p.is_file():
            return []
        try:
            data = _json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                # Each entry: {"q": str, "added_ts": float}
                return [e for e in data
                        if isinstance(e, dict) and e.get("q")]
        except Exception as e:
            logger.debug(f"saved_vibes load failed: {e}")
        return []

    def _saved_vibes_write(self, vibes: list[dict]) -> None:
        import json as _json
        p = self._saved_vibes_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                _json.dumps(vibes, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"saved_vibes write failed: {e}")

    def _saved_vibes_list(self) -> dict:
        vibes = self._saved_vibes_load()
        # Most-recent first
        vibes_sorted = sorted(
            vibes, key=lambda e: -float(e.get("added_ts", 0))
        )
        return {
            "ok": True,
            "vibes": [{"q": e["q"],
                       "added_ts": float(e.get("added_ts", 0))}
                      for e in vibes_sorted[:6]],
        }

    def _saved_vibes_save(self, q: str) -> dict:
        import time as _time
        q = (q or "").strip()
        if not q:
            return {"ok": False, "error": "empty query"}
        if len(q) > 80:
            return {"ok": False, "error": "query too long (>80 chars)"}
        vibes = self._saved_vibes_load()
        # Dedupe — if already present, bump its timestamp (move to top)
        existing = [e for e in vibes if e.get("q") == q]
        for e in existing:
            e["added_ts"] = _time.time()
        if not existing:
            vibes.insert(0, {"q": q, "added_ts": _time.time()})
        # Cap at MAX, sorted by recency (newest first)
        vibes.sort(key=lambda e: -float(e.get("added_ts", 0)))
        vibes = vibes[:self._SAVED_VIBES_MAX]
        self._saved_vibes_write(vibes)
        logger.info(f"[saved_vibes] saved '{q}' "
                    f"(total {len(vibes)})")
        return {"ok": True, "q": q, "total": len(vibes)}

    def _openers_to_banks_action(self) -> dict:
        """Build banks A-H from intro:N-tagged files (sourced from
        intro_tagger.py). These are typically dance/walking/setup
        openers from studio-produced files — projection-friendly
        content perfect for set openings. Splits by intro LENGTH
        buckets so A=short fast openers, H=longest setup pieces.

        Returns the bank distribution + total count."""
        try:
            with self.path_tags._lock:
                rows = self.path_tags._conn.execute(
                    "SELECT filepath, tag FROM file_tags "
                    "WHERE tag LIKE 'intro:%' "
                    "AND tag NOT LIKE 'intro:_%'"
                ).fetchall()
        except Exception as e:
            return {"ok": False, "error": f"db query failed: {e}"}
        # Parse intro length from tag
        items: list[tuple[str, int]] = []   # (path, intro_seconds)
        for fp, tag in rows:
            try:
                sec = int(tag.split(":", 1)[1])
                if Path(fp).is_file():
                    items.append((fp, sec))
            except (ValueError, IndexError):
                continue
        if not items:
            return {"ok": False, "error":
                    "no intro-tagged files yet — "
                    "run `python intro_tagger.py` first"}
        # Sort by intro length, then split into 8 evenly-sized buckets
        # so A = shortest punchy openers, H = longest setup pieces.
        items.sort(key=lambda t: t[1])
        n = len(items)
        bank_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        n_banks = min(n, 8)
        per = max(1, n // n_banks) if n_banks else 0
        out_counts: dict[str, int] = {}
        for i, ltr in enumerate(bank_letters):
            if i >= n_banks:
                out_counts[ltr] = -1
                continue
            start = i * per
            end = start + per if i < (n_banks - 1) else n
            bank_files = [fp for fp, _sec in items[start:end]]
            if bank_files:
                # Label with the intro-length range for clarity
                lo_sec = items[start][1]
                hi_sec = items[end - 1][1]
                label = f"openers {lo_sec}-{hi_sec}s {ltr}"
                try:
                    self.bank_store.save_into(
                        ltr, label, bank_files,
                        set_folder_lock="",
                    )
                except Exception as e:
                    logger.debug(f"openers save_into {ltr} failed: {e}")
            out_counts[ltr] = len(bank_files)
        try:
            self._refresh_banks_for_ipad()
        except Exception:
            pass
        try:
            self.state.set_message(
                f"OPENERS: {n} files split across "
                f"{n_banks} banks (by intro length)"
            )
        except Exception:
            pass
        logger.info(
            f"[openers_to_banks] n={n} per_bank={per} "
            f"dist={out_counts}"
        )
        # MK2 OLED flash for visual feedback
        try:
            if self.mk2:
                self.mk2.push_oled_clip_to_banks_flash(
                    query="OPENERS",
                    matches=n,
                    distribution=out_counts,
                    elapsed_ms=0,
                )
        except Exception as e:
            logger.debug(f"openers OLED flash failed: {e}")
        return {
            "ok": True,
            "total": n,
            "distribution": out_counts,
        }

    def _saved_vibes_delete(self, q: str) -> dict:
        q = (q or "").strip()
        if not q:
            return {"ok": False, "error": "empty query"}
        vibes = self._saved_vibes_load()
        before = len(vibes)
        vibes = [e for e in vibes if e.get("q") != q]
        if len(vibes) == before:
            return {"ok": False, "error": "not found"}
        self._saved_vibes_write(vibes)
        logger.info(f"[saved_vibes] deleted '{q}' "
                    f"(remaining {len(vibes)})")
        return {"ok": True, "q": q, "total": len(vibes)}

    def mk2_get_verticals(self) -> dict:
        """Return labels for the 8 vertical buttons (bits 40-47) so
        the iPad can render them in the MK2-mirror bank drawer.

        Resolves the EFFECTIVE binding for each vertical, in priority
        order (matches what `_on_mk2_button_press` actually fires):
          1. `mk2_vertical_pages[active_page][slot]` if a page is set
             -- this is the LIVE per-page folder mapping.
          2. `mk2_button_map["<bit>"]` static fallback (per-bit, no
             page concept).
          3. Bit 47 (slot 7 / MUTE) is hardcoded to reroll_banks.

        Also returns the active page index + name so the iPad can
        show context (e.g. "PAGE 1: CURRENT" header).

        Returns: {ok, active_page, page_name, verticals: [
                    {idx, label, action, source}]} length=8."""
        pages = self.config.get("mk2_vertical_pages") or []
        action_map = (self.config.get("mk2_button_map") or {})
        active_idx = getattr(self, "_mk2_active_page", 0) or 0
        if not (0 <= active_idx < len(pages)):
            active_idx = 0
        active_page = pages[active_idx] if active_idx < len(pages) else []
        try:
            page_name = self._mk2_page_name(active_idx) or ""
        except Exception:
            page_name = ""

        out = []
        for slot in range(8):
            bit = 40 + slot
            folder = None
            source = "empty"
            # 1. Per-page binding (slot 0-6; slot 7 is always reroll)
            if slot < 7 and slot < len(active_page) and active_page[slot]:
                folder = active_page[slot]
                source = "page"
            # 2. Static fallback from mk2_button_map (if no page binding)
            action_str = action_map.get(str(bit), "") or ""
            if not folder and action_str.startswith("folder:"):
                folder = action_str.split(":", 1)[1]
                source = "static"
            # 3. Slot 7 = reroll (hardcoded in _on_mk2_button_press)
            if slot == 7:
                source = "reroll"
                folder = None
            if folder:
                try:
                    label = Path(folder).name or folder
                except Exception:
                    label = folder
                action = f"folder:{folder}"
            elif slot == 7:
                label = "Reroll"
                action = "reroll_banks"
            elif action_str == "reroll_banks":
                label = "Reroll"
                action = action_str
            elif action_str:
                label = action_str
                action = action_str
            else:
                label = ""
                action = ""
            out.append({
                "idx": slot,
                "label": label,
                "action": action,
                "source": source,
            })
        return {
            "ok": True,
            "active_page": active_idx,
            "page_name": page_name,
            "verticals": out,
        }

    def mk2_fire_vertical(self, idx: int) -> dict:
        """Trigger the same action a physical MK2 vertical-button press
        would fire (bits 40-47, idx 0-7). Routes through the existing
        button-action dispatcher so behavior stays in lockstep."""
        # Coerce JSON-num / JSON-str into int defensively (audit fix).
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return {"ok": False, "error": "vertical idx must be an integer"}
        if not (0 <= idx <= 7):
            return {"ok": False, "error": "vertical idx out of range"}
        # Enqueue the action just like _on_mk2_button_press would.
        bit = 40 + idx
        action_map = (self.config.get("mk2_button_map") or {})
        action = action_map.get(str(bit), "")
        if not action:
            return {"ok": False, "error": f"no binding for vertical {idx}"}
        if self._mk2_button_queue is None:
            self._mk2_button_queue = queue.Queue()
        self._mk2_button_queue.put(action)
        # CRITICAL FIX (code-review agent, 2026-05-16 night): without this
        # pump, iPad vertical taps stack up on the queue but never drain
        # until the next physical MK2 button press happens to fire the
        # dispatcher. Match the pattern at _on_mk2_button_press (line 1621).
        QTimer.singleShot(0, self._fire_pending_mk2_button)
        logger.info(f"[iPad] fired vertical {idx} → action={action!r}")
        return {"ok": True, "action": action}

    def bank_peek(self, letter: str) -> dict:
        """Return a bank's full contents WITHOUT loading it. Powers the
        iPad's "Bank Preview" drawer (2026-05-16) — user wanted to see
        what's in a bank before committing to switch into it.

        Returns: {ok, letter, name, folder, count, files: [{name, bpm,
        cues, starred}]}. Joins per-file BPM (from path_tags `bpm:N`
        tag) + cue-point count + starred-cue count (from clips_db) so
        the preview is decision-useful, not just a filename list.

        Per-file lookup is ~2 DB hits (path_tags + clips_db); both
        in-memory after warmup so even a maxed-out 64-file bank stays
        under ~30ms total. Full paths stay server-side — only basenames
        + metadata go over the wire."""
        s = self.bank_store.get(letter)
        if not s:
            return {"ok": False, "error": "invalid bank letter"}
        full_paths = [fp for fp in (s.get("files") or []) if fp]
        # AUDIT FIX (2026-05-16): bulk DB lookups instead of N per-file
        # round-trips. For a 16-file bank: was 32 lookups (each O(N
        # total_clips) for clips_db.get_clips_for_file); now 2 grouped
        # queries that scan everything once.
        # --- Bulk BPM lookup (one SQL query for all paths' bpm tags)
        bpm_map = self._bulk_get_clip_bpms(full_paths)
        # --- Bulk cue lookup: single walk over all clips, grouped by
        # filepath key. {filepath_norm: (cue_count, starred_count)}
        cue_map: dict[str, tuple[int, int]] = {}
        try:
            wanted_keys = {self.clips_db._key(fp) for fp in full_paths}
            counts: dict[str, list[int]] = {k: [0, 0] for k in wanted_keys}
            with self.clips_db._lock:
                for c in self.clips_db._clips:
                    k = self.clips_db._key(c.get("filepath", ""))
                    if k in counts:
                        counts[k][0] += 1
                        if c.get("starred"):
                            counts[k][1] += 1
            cue_map = {k: (v[0], v[1]) for k, v in counts.items()}
        except Exception as e:
            logger.debug(f"bank_peek cue bulk-scan failed: {e}")
        # Collect missing thumb paths as we go — kick a background
        # backfill at the end so the second peek of a never-browsed
        # bank has thumbs ready. First peek will show placeholders for
        # cold paths; iPad <img loading=lazy> handles graceful 404.
        import thumbnails as _th
        missing_thumb_paths: list[str] = []
        enriched = []
        for fp in full_paths:
            entry = {
                "name": Path(fp).name,
                "bpm": bpm_map.get(fp, 0),
                "cues": 0,
                "starred": 0,
                "hash": "",
            }
            # Thumb hash — content-addressed (rename = new hash), so
            # this is stable across moves. The iPad builds the URL
            # /lib_thumbnails/<hash>.jpg from it.
            try:
                h = _th.lib_thumbnail_hash(fp)
                entry["hash"] = h
                if not _th.lib_thumbnail_path(h).exists():
                    missing_thumb_paths.append(fp)
            except Exception as e:
                logger.debug(f"thumb hash failed for {fp}: {e}")
            # Cue counts from the bulk-built map.
            try:
                k = self.clips_db._key(fp)
                cues, starred = cue_map.get(k, (0, 0))
                entry["cues"] = cues
                entry["starred"] = starred
            except Exception as e:
                logger.debug(f"cue lookup failed for {fp}: {e}")
            enriched.append(entry)
        # Background thumb backfill — daemon thread, doesn't block
        # the peek response. ffmpeg is ~1-2s/thumb so a 16-file bank
        # takes ~30s to fully populate; second peek will show them.
        # AUDIT FIX (2026-05-16): dedup against an in-progress set so
        # rapid double-peeks don't spawn duplicate ffmpeg processes
        # racing to write the same .jpg. Filter the work list down to
        # paths not currently being backfilled by another thread.
        if missing_thumb_paths:
            with self._thumb_backfill_lock:
                to_do = [p for p in missing_thumb_paths
                         if p not in self._thumb_backfill_inflight]
                for p in to_do:
                    self._thumb_backfill_inflight.add(p)
            if to_do:
                def _backfill(paths):
                    try:
                        _th.ensure_thumbs_dir()
                        for p in paths:
                            try:
                                _th.generate_library_thumbnail(p)
                            except Exception as e:
                                logger.debug(f"thumb gen failed {p}: {e}")
                            finally:
                                # Always release the in-flight slot so a
                                # failed gen doesn't permanently block retries.
                                with self._thumb_backfill_lock:
                                    self._thumb_backfill_inflight.discard(p)
                    except Exception as e:
                        logger.debug(f"thumb backfill thread failed: {e}")
                t = threading.Thread(
                    target=_backfill,
                    args=(to_do,),
                    daemon=True,
                    name=f"bank_peek_thumbs_{letter}",
                )
                t.start()
                logger.info(f"[bank-peek] kicked thumb backfill for "
                            f"{len(to_do)}/{len(missing_thumb_paths)} files "
                            f"in bank {letter.upper()} ({len(missing_thumb_paths)-len(to_do)} already in flight)")
        return {
            "ok": True,
            "letter": (letter or "").upper(),
            "name": s.get("name", ""),
            "folder": s.get("folder", ""),
            "count": len(enriched),
            "files": enriched,
        }

    def _refresh_scratch_sets_for_ipad(self):
        try:
            self.state.set_scratch_sets(self.scratch_set_store.all())
        except Exception as e:
            logger.debug(f"_refresh_scratch_sets_for_ipad failed: {e}")

    def _proxy_pause(self) -> dict:
        self.proxy_cache.pause()
        self._refresh_proxy_status()
        self.state.set_message("Proxies: paused (in-flight finishes)")
        return {"ok": True}

    def _proxy_resume(self) -> dict:
        self.proxy_cache.resume()
        self._refresh_proxy_status()
        self.state.set_message("Proxies: resumed")
        return {"ok": True}

    def _proxy_cancel_current(self) -> dict:
        killed = self.proxy_cache.cancel_current()
        self._refresh_proxy_status()
        self.state.set_message("Proxies: in-flight cancelled" if killed else "No transcode in flight")
        return {"ok": True, "killed": killed}

    def _proxy_cancel_pending(self) -> dict:
        n = self.proxy_cache.cancel_pending()
        self._refresh_proxy_status()
        self.state.set_message(f"Proxies: dropped {n} queued")
        return {"ok": True, "dropped": n}

    def _refresh_proxy_status(self):
        try:
            self.state.set_proxy_status({
                "paused": self.proxy_cache.is_paused(),
                "queued": self.proxy_cache.qsize(),
                "in_flight": self.proxy_cache.current_file_name(),
            })
        except Exception as e:
            logger.debug(f"_refresh_proxy_status failed: {e}")

    def decks_clear_all(self) -> dict:
        """Empty all 4 deck slots in one shot. Useful between sets."""
        cleared = 0
        for i in range(4):
            try:
                self.decks_store.clear(i)
                try:
                    self.preview_streams.stop_stream(i)
                except Exception:
                    pass
                cleared += 1
            except Exception as e:
                logger.debug(f"decks_clear_all: slot {i} failed: {e}")
        self._refresh_decks_for_ipad()
        self.state.set_message(f"All {cleared} decks cleared")
        return {"ok": True, "cleared": cleared}

    def scratch_fire(self, path: str) -> dict:
        """Tap a scratch chip → load that file to LIVE."""
        if not path or not Path(path).exists():
            return {"ok": False, "error": "path missing"}
        try:
            self.session_log.record_fire(source="scratch_chip", filepath=path)
        except Exception:
            pass
        self._record_fire_history(path)
        # force=True — an explicit scratch-chip tap must never be debounced.
        if self.load_video(path, force=True) and self.player:
            self.player.play()
        return {"ok": True}

    # ── Scenes (saved deck-layout snapshots) ──────────────────────────

    def _refresh_scenes_for_ipad(self):
        """Re-publish scene list to AppState. Cheap — slimmed to id/name/count."""
        try:
            self.state.set_scenes(self.scene_store.all())
        except Exception as e:
            logger.debug(f"_refresh_scenes_for_ipad failed: {e}")

    def scene_save(self, name: str) -> dict:
        """Snapshot the current 4 deck slots under `name`."""
        snapshot = self.decks_store.all() or []
        scene = self.scene_store.save(name, snapshot)
        self._refresh_scenes_for_ipad()
        self.state.set_message(f"Scene saved: {scene.get('name')}")
        return {"ok": True, "scene": {"id": scene.get("id"), "name": scene.get("name")}}

    def scene_load(self, scene_id: str) -> dict:
        """Restore a saved scene to the deck slots. Each slot's preview
        ffmpeg is restarted to match the new file."""
        scene = self.scene_store.get(scene_id)
        if not scene:
            self.state.set_message("Scene not found", error=True)
            return {"ok": False, "error": "scene not found"}
        decks = scene.get("decks") or []
        for i in range(4):
            entry = decks[i] if i < len(decks) else None
            if entry and isinstance(entry, dict) and entry.get("filepath"):
                # Verify the file still exists; skip silently if it moved.
                if not Path(entry["filepath"]).exists():
                    logger.warning(f"Scene load: skipping deck {i + 1}, missing file: {entry['filepath']}")
                    continue
                self.decks_store.set(i, dict(entry))
                try:
                    self._start_deck_preview(entry)
                except Exception as e:
                    logger.debug(f"scene_load preview restart failed for slot {i}: {e}")
                self.decks_store.regenerate_filmstrip_async(i)
            else:
                self.decks_store.clear(i)
                try:
                    self.preview_streams.stop_stream(i)
                except Exception:
                    pass
        self._refresh_decks_for_ipad()
        self.state.set_message(f"Scene loaded: {scene.get('name')}")
        return {"ok": True, "scene": {"id": scene.get("id"), "name": scene.get("name")}}

    def scene_delete(self, scene_id: str) -> dict:
        ok = self.scene_store.delete(scene_id)
        if ok:
            self._refresh_scenes_for_ipad()
            self.state.set_message("Scene deleted")
        return {"ok": ok}

    def scene_rename(self, scene_id: str, name: str) -> dict:
        ok = self.scene_store.rename(scene_id, name)
        if ok:
            self._refresh_scenes_for_ipad()
        return {"ok": ok}

    def deck_clear(self, deck_idx: int) -> dict:
        """Empty a deck slot."""
        if not (0 <= deck_idx < 4):
            return {"ok": False, "error": "deck must be 0..3"}
        self.decks_store.clear(deck_idx)
        # Tear down the matching ffmpeg subprocess (subscribers wake up
        # and the iPad <img> falls back to its onerror placeholder).
        try:
            self.preview_streams.stop_stream(deck_idx)
        except Exception as e:
            logger.debug(f"stop preview failed: {e}")
        self._refresh_decks_for_ipad()
        self.state.set_message(f"Deck {deck_idx + 1} cleared")
        return {"ok": True}

    def set_preview_deck(self, deck_idx: int) -> dict:
        """Choose which deck slot is the crossfade source. Tap FX2 Focus
        for slot 1, Param 1 for slot 2, etc. — then the crossfader fades
        between LIVE and that slot. Solves 'can only fade to deck 1'."""
        if not (0 <= deck_idx < 4):
            return {"ok": False, "error": "deck_idx out of range"}
        # Drop any cached preview attachment so the next crossfade move
        # re-reads from the new slot.
        self._last_preview_filepath = None
        self.state.set_crossfade(preview_deck_idx=deck_idx, blend_active=False)
        deck = self.decks_store.get(deck_idx) if self.decks_store else None
        if deck:
            name = deck.get("name") or Path(deck.get("filepath", "")).name
            self.state.set_message(f"Crossfade source → Deck {deck_idx + 1}: {name}")
        else:
            self.state.set_message(f"Crossfade source → Deck {deck_idx + 1} (empty)")
        return {"ok": True, "deck_idx": deck_idx}

    def fire_deck_random_seek(self, deck_idx: int) -> bool:
        """SHIFT + B Pad N: GO LIVE on deck N, then seek into the body
        (skip intro + outro, land somewhere not recently shown). Same
        seek logic as audio-reactive auto-flip — keeps you out of the
        first 15s and last 20s, picks a position ≥1/4 of the playable
        duration away from the last visit on this file."""
        if not self.fire_deck(deck_idx):
            return False
        # Defer seek so mpv has duration available. Same delay as auto_flip.
        QTimer.singleShot(150, self._auto_seek_into_body)
        return True

    def fire_deck(self, deck_idx: int) -> bool:
        """GO LIVE: load this deck's clip into the live player and play it."""
        if not (0 <= deck_idx < 4):
            return False
        deck = self.decks_store.get(deck_idx)
        if not deck:
            self.state.set_message(f"Deck {deck_idx + 1} is empty", error=True)
            return False
        path = deck.get("filepath") or ""
        if not path or not Path(path).exists():
            self.state.set_message(
                f"Deck {deck_idx + 1}: file missing", error=True,
            )
            return False
        # Prefer firing at where the user can SEE the preview right now
        # (minus a fixed latency offset to compensate for ffmpeg pipeline
        # buffering + iPad render lag). Falls back to clip in_sec if the
        # preview hasn't started yet or position is unknown.
        LATENCY_COMPENSATION = 3.5  # seconds to subtract — tunable
        seek_to = None
        in_sec = float(deck.get("in_sec") or 0.0)
        # Clamp ceiling: out_sec if set, else the probed duration. Stops
        # an unbounded preview-position (looping clip with no out_sec)
        # from seeking us to EOF / a black frame. (Audit fix H6.)
        out_sec = float(deck.get("out_sec") or 0.0)
        try:
            cur_pos = self.preview_streams.get_position(deck_idx)
            if cur_pos is not None:
                seek_to = max(in_sec, cur_pos - LATENCY_COMPENSATION)
        except Exception as e:
            logger.debug(f"fire_deck({deck_idx}): preview pos unavailable: {e}")
        if seek_to is None:
            seek_to = in_sec
        # Clamp to the playable window so we never land past EOF.
        if out_sec > in_sec:
            seek_to = min(seek_to, out_sec - 1.0)
        seek_to = max(0.0, seek_to)
        try:
            # force=True — GO LIVE is an explicit user action; never debounce.
            if not self.load_video(path, force=True):
                return False
            if self.player:
                self.player.seek(seek_to)
                self.player.play()
            self.state.record_s2_action(f"deck:{deck_idx}")
            self.state.set_message(
                f"GO LIVE: {deck.get('name', '')} @ {seek_to:.1f}s"
            )
            return True
        except Exception as e:
            logger.error(f"fire_deck({deck_idx}) failed: {e}")
            self.state.set_message(f"Deck fire failed: {e}", error=True)
            return False

    # ── Channels (saved library-folder presets) ──────────────────────────

    def _init_channels(self):
        """Push the persisted channel list into AppState so the iPad
        chip row renders on first poll. Defaults are written to
        ~/.setpiece/channels.json on first run by ChannelStore."""
        try:
            self.state.set_channels(self.channels_store.all())
            self.state.set_active_channel(-1)
            # Initial LED dim-paint so the channel buttons all glow
            # faintly even before the user picks one — telegraphs that
            # the buttons are "live and waiting".
            self._refresh_channel_leds()
        except Exception as e:
            logger.warning(f"Channel init failed: {e}")

    # S2 has only single-color amber LEDs on the FX channel buttons, so
    # the per-channel "color" lives only on the iPad chip. On hardware
    # we convey state via brightness: active = LED_LIVE, others = LED_DIM.
    _CH_LED_NAMES = ("fx1_ch1", "fx2_ch1", "fx1_ch2", "fx2_ch2")
    _CH_LED_ACTIVE = 0x14   # mirrors LED_LIVE in s2_controller
    _CH_LED_IDLE = 0x06     # mirrors LED_DIM

    def _refresh_channel_leds(self):
        """Push channel-active state into the FX channel button LEDs.
        Idempotent — safe to call from anywhere (S2 connect, channel
        switch, channel save). No-op when S2 isn't up yet."""
        if not self.s2:
            return
        try:
            active = self.state.active_channel_idx
            for i, led_name in enumerate(self._CH_LED_NAMES):
                bright = (
                    self._CH_LED_ACTIVE if i == active else self._CH_LED_IDLE
                )
                self.s2.set_led_state(led_name, bright)
        except Exception as e:
            logger.debug(f"Channel LED refresh failed: {e}")

    def switch_channel(self, idx: int) -> dict:
        """Jump the library browser to channel[idx]'s folder, mark it
        active, light the LED. Called by S2 fx channel buttons AND the
        iPad. Returns the standard {ok, error} dict so HTTP callers see
        a useful response."""
        if not (0 <= idx < CHANNEL_COUNT):
            return {"ok": False, "error": f"channel idx must be 0..{CHANNEL_COUNT - 1}"}
        ch = self.channels_store.get(idx)
        if not ch:
            return {"ok": False, "error": "channel not found"}
        folder = (ch.get("folder") or "").strip()
        if not folder:
            # Channel exists but has no folder configured yet — still
            # mark it active and surface a message so the user knows to
            # configure it from the iPad.
            self.state.set_active_channel(idx)
            self._refresh_channel_leds()
            self.state.set_message(
                f"Channel {idx + 1}: no folder set — open editor to configure",
                error=True,
            )
            return {"ok": False, "error": "channel has no folder"}
        # Reuse library_scan to move BOTH library_root AND library_folder.
        # That's the right semantics: a channel switch is the user
        # declaring a new "home folder" for browsing, not a sub-cd.
        result = self.library_scan(folder)
        if not result.get("ok"):
            self.state.set_message(
                f"Channel {idx + 1}: {result.get('error', 'switch failed')}",
                error=True,
            )
            return result
        self.state.set_active_channel(idx)
        self._refresh_channel_leds()
        # Friendly status — the tape-deck preset feel.
        name = ch.get("name") or f"Channel {idx + 1}"
        self.state.set_message(f"CH {idx + 1}: {name}")
        self.state.record_s2_action(f"channel:{idx}")
        # Flash the button LED so the press feels acknowledged on
        # hardware (matches how the deck/clip buttons flash).
        try:
            if self.s2:
                self.s2.flash_led(self._CH_LED_NAMES[idx])
        except Exception:
            pass
        return {"ok": True, "channel": ch}

    def save_channel(
        self,
        idx: int,
        name: Optional[str] = None,
        folder: Optional[str] = None,
        color: Optional[str] = None,
        tag_filter: Optional[str] = None,
    ) -> dict:
        """Partial update of channel[idx]. Any field passed as None is
        left untouched; empty strings reset that field to the default."""
        if not (0 <= idx < CHANNEL_COUNT):
            return {"ok": False, "error": f"channel idx must be 0..{CHANNEL_COUNT - 1}"}
        # Validate folder (if provided non-empty) — must exist and be a
        # directory. Non-existent folder is a config error worth
        # surfacing immediately rather than silently failing later.
        if folder is not None and folder.strip():
            try:
                p = Path(folder).expanduser().resolve()
            except Exception as e:
                return {"ok": False, "error": f"bad folder path: {e}"}
            if not p.exists() or not p.is_dir():
                return {"ok": False, "error": f"not a directory: {p}"}
            folder = str(p)
        updated = self.channels_store.update(
            idx,
            name=name,
            folder=folder,
            color=color,
            tag_filter=tag_filter,
        )
        if not updated:
            return {"ok": False, "error": "update failed"}
        # Mirror the new list into AppState so the iPad sees it on the
        # next 500ms poll.
        self.state.set_channels(self.channels_store.all())
        self.state.set_message(f"Saved Channel {idx + 1}: {updated.get('name', '')}")
        return {"ok": True, "channel": updated}

    def set_current_as_channel(self, idx: int) -> dict:
        """Snapshot the CURRENT library folder into channel[idx]. One-tap
        "remember where I am" from the iPad."""
        if not (0 <= idx < CHANNEL_COUNT):
            return {"ok": False, "error": f"channel idx must be 0..{CHANNEL_COUNT - 1}"}
        cur = (self.state.library_folder or "").strip()
        if not cur:
            return {"ok": False, "error": "no library folder loaded"}
        return self.save_channel(idx, folder=cur)

    def closeEvent(self, event):
        if self.s2:
            self.s2.stop()
        if self.stream_deck:
            try:
                self.stream_deck.stop()
            except Exception as e:
                logger.debug(f"stream_deck shutdown error: {e}")
        # Stop the MK2 — blanks pad LEDs + OLED screens so the controller
        # doesn't sit glowing after the app closes. (Audit fix M9.)
        if getattr(self, "mk2", None):
            try:
                self.mk2.stop()
            except Exception as e:
                logger.debug(f"mk2 shutdown error: {e}")
        if self.player:
            self.player.close()
        if self.audio_reactive:
            self.audio_reactive.stop()
        # Stop the proxy transcoder + any in-flight ffmpeg so no orphans.
        if getattr(self, "proxy_cache", None):
            try:
                self.proxy_cache.cancel_current()
                self.proxy_cache.stop()
            except Exception as e:
                logger.debug(f"proxy_cache shutdown error: {e}")
        # Stop the two folder watchers + path-tag scan thread.
        for watcher_attr in ("working_set", "scratch_watcher"):
            w = getattr(self, watcher_attr, None)
            if w:
                try:
                    w.stop()
                except Exception:
                    pass
        # Flush + close the session log so the final events land.
        if getattr(self, "session_log", None):
            try:
                self.session_log.close()
            except Exception:
                pass
        # Close the OSC socket.
        if getattr(self, "osc", None):
            try:
                self.osc.close()
            except Exception:
                pass
        # Kill all preview ffmpegs BEFORE the HTTP server so any open
        # /preview/N.mjpg request thread wakes up + drains, instead of
        # holding the server's shutdown() back.
        try:
            self.preview_streams.stop_all()
        except Exception as e:
            logger.debug(f"preview_streams shutdown error: {e}")
        if self.http_server:
            self.http_server.stop()
        wd = getattr(self, "_watchdog_stop", None)
        if wd is not None:
            wd.set()
        self._save_config()
        event.accept()


def _apply_dark_theme(app: QApplication) -> None:
    """Dark UI chrome for the desktop window. A VJ tool lives in dark
    rooms — the controls should never spill light onto the projection.
    Fusion style + a hand-tuned QPalette covers every Qt widget including
    the file-open dialog; the embedded mpv video output is unaffected
    (it renders straight into the HWND, not through Qt's painter)."""
    app.setStyle("Fusion")
    bg     = QColor(24, 24, 27)
    base   = QColor(18, 18, 20)
    panel  = QColor(38, 38, 44)
    text   = QColor(222, 222, 226)
    dim    = QColor(138, 138, 146)
    accent = QColor(80, 160, 255)
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, bg)
    pal.setColor(QPalette.ColorRole.WindowText, text)
    pal.setColor(QPalette.ColorRole.Base, base)
    pal.setColor(QPalette.ColorRole.AlternateBase, panel)
    pal.setColor(QPalette.ColorRole.Text, text)
    pal.setColor(QPalette.ColorRole.Button, panel)
    pal.setColor(QPalette.ColorRole.ButtonText, text)
    pal.setColor(QPalette.ColorRole.ToolTipBase, panel)
    pal.setColor(QPalette.ColorRole.ToolTipText, text)
    pal.setColor(QPalette.ColorRole.Highlight, accent)
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(10, 10, 12))
    pal.setColor(QPalette.ColorRole.PlaceholderText, dim)
    pal.setColor(QPalette.ColorRole.Link, accent)
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.WindowText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, dim)
    app.setPalette(pal)
    app.setStyleSheet(
        "QPushButton{background:#26262c;border:1px solid #3a3a44;"
        "border-radius:4px;padding:6px 12px;}"
        "QPushButton:hover{background:#33333c;}"
        "QPushButton:pressed{background:#1c1c20;}"
        "QToolTip{background:#26262c;color:#dedee2;border:1px solid #3a3a44;}"
    )


def main():
    # Windows default stderr/stdout is cp1252 — a unicode char in any
    # logged string (song titles with ⭐, filenames with non-ascii)
    # would crash the logger mid-set. Reconfigure to utf-8 with
    # replace-on-fail so the app never dies trying to print.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = QApplication(sys.argv)
    _apply_dark_theme(app)
    window = VJPracticeApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
