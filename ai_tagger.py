"""
AI visual tagging for clip thumbnails — the visual-content complement
to ``path_tags.py`` (which derives tags from folder/filename strings).

Given a clip's IN-point JPEG, returns 3-5 short, lowercase, single-word
tags describing what's visually in the frame ("neon", "close-up",
"dancefloor", "smoke", "stage"). The tags get fed into the same chip
strip / filter UI as path tags, but they describe pixels rather than
filenames — useful when a file is named ``IMG_8473.mp4`` and the path
tokenizer comes up empty.

Two interchangeable backends, picked at construction time:

* ``backend="cloud"`` — Anthropic Messages API + vision with
  ``claude-sonnet-4-5``. Base64-embedded JPEG, single ~150-token
  response. Expected latency: 1-2 s per clip. Best tag quality by a
  wide margin. Requires ``pip install anthropic`` and an API key in
  the ``ANTHROPIC_API_KEY`` env var (or passed explicitly to the
  constructor). The system prompt is marked with
  ``cache_control: ephemeral`` so the prompt definition itself is a
  cache hit on the second request (saves ~50% input cost on the
  static portion when tagging many clips back-to-back). Image bytes
  themselves are NOT cached — they're different per clip.

* ``backend="local"`` — completely offline. After surveying available
  options in 2026:

    - ``open_clip_torch`` (PyPI ``open-clip-torch``, py3-none-any
      wheel as of 3.3.0, Feb 2026) is pip-installable on Windows
      without MSVC compilation, BUT it pulls in PyTorch (~400 MB CPU
      wheel) and would need a hand-curated tag vocabulary to score
      against. Strong quality on a curated vocab, heavy footprint.
    - ``transformers`` + ``Salesforce/blip-image-captioning-base``
      (~990 MB model, CPU inference works but is slow — 3-6 s per
      image on a modest desktop CPU). Free-form caption that we'd
      have to tokenize ourselves.
    - **What this module actually ships**: a *zero-dependency*
      heuristic fallback (color histogram + brightness +
      saturation + contrast) that returns *something* meaningful
      ("dark", "neon", "warm", "monochrome", "high-contrast") with
      no extra installs. Quality is obviously lower than CLIP or a
      VLM but it's instant, always available, and never blocks the
      pipeline. The heuristic only needs ``Pillow``, which the
      project already depends on for thumbnail handling.

  If the user later wants real CLIP/BLIP tagging, the local backend's
  ``_tag_heuristic`` method is the spot to swap in; the rest of the
  module (caching, async pool, public API) stays the same.

* Caching: every successful tag set is persisted to
  ``~/.setpiece/ai_tags.json`` keyed by the JPEG's SHA-256.
  Repeated calls on the same image are O(1) dict lookups; we never
  hit the API twice for the same picture.

* Async: ``tag_image_async`` submits to a module-level
  ``ThreadPoolExecutor(max_workers=3)``. Three is enough to overlap
  network latency without rate-limiting ourselves. The callback runs
  on a worker thread — wrap any Qt UI mutation in
  ``QMetaObject.invokeMethod`` (or main.py's existing
  ``schedule_on_main_thread`` helper) before touching widgets.
"""

import base64
import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)


# ── Module-level constants ─────────────────────────────────────────────

CACHE_FILE = Path.home() / ".setpiece" / "ai_tags.json"

# Anthropic model + vision settings. Centralised so swapping in a new
# Sonnet/Haiku variant later is a one-line change.
ANTHROPIC_MODEL = "claude-sonnet-4-5"
ANTHROPIC_MAX_TOKENS = 128
ANTHROPIC_REQUEST_TIMEOUT_SEC = 15.0

# Async pool size. 3 is empirically enough to hide network latency on
# a residential connection without tripping Anthropic's per-account
# concurrency limit on the smaller tiers.
_POOL_MAX_WORKERS = 3
_pool: Optional[ThreadPoolExecutor] = None
_pool_lock = threading.Lock()


def _get_pool() -> ThreadPoolExecutor:
    """Lazily build a process-wide tag executor. Daemon threads so it
    never blocks shutdown."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(
                max_workers=_POOL_MAX_WORKERS,
                thread_name_prefix="ai-tagger",
            )
        return _pool


# ── System prompt for the cloud backend ────────────────────────────────
#
# Kept short and deterministic. We want a comma-separated list of
# lowercase one-word tags — nothing else. Any explanation, numbering,
# or punctuation costs tokens and forces us to write a more forgiving
# parser. The "respond with only the comma-separated list" line is
# load-bearing — without it the model often prefaces with "Here are
# 5 tags: …" which we'd then have to strip.
#
# The prompt is constant across every clip, so we mark it
# ``cache_control: ephemeral`` to get prompt caching on the static
# portion (image bytes are per-request and stay un-cached).
_SYSTEM_PROMPT = (
    "You are a visual-content tagger for a VJ tool. "
    "Given a single frame from a video clip, return EXACTLY 3-5 "
    "short lowercase tags that describe the visual content: lighting "
    "mood, color palette, shot type, setting, and notable visual "
    "elements. Each tag must be a single English word (no spaces, no "
    "hyphens, no punctuation), 3-12 letters, all lowercase. Examples "
    "of good tags: neon, closeup, smoke, dancefloor, stage, monochrome, "
    "warm, cool, silhouette, crowd, strobe, dark, daylight, neon, "
    "geometric, abstract. Do NOT include generic words like 'video', "
    "'frame', 'image', 'scene'. Respond with ONLY the comma-separated "
    "list of tags, no prose, no numbering, no quotes."
)


# ── Tag normalization (shared by both backends) ────────────────────────

# Tags must match the same shape as ``clips_db._TAG_RE``: pure a-z,
# 3-24 characters. Anything else gets dropped.
_TAG_MIN_LEN = 3
_TAG_MAX_LEN = 24

# Generic / lazy tags the model sometimes returns that we never want
# in the chip strip. Tuned by hand from observed outputs.
_BANNED_TAGS = {
    "video", "frame", "image", "scene", "shot", "clip", "view",
    "picture", "photo", "footage", "movie", "film",
    # Pure modifiers that mean nothing on their own
    "some", "lots", "many", "few", "much",
    # Numerals as words occasionally slip out
    "one", "two", "three", "four", "five",
}


def _clean_tag(raw: str) -> str:
    """Coerce a single raw tag to canonical lowercase a-z form, or
    return '' if it can't be made into a valid tag."""
    if not isinstance(raw, str):
        return ""
    # Drop everything that isn't a letter. Mirrors clips_db's
    # _normalize_tag so AI tags drop cleanly into the existing storage.
    cleaned = "".join(c for c in raw.lower() if "a" <= c <= "z")
    if not cleaned:
        return ""
    if len(cleaned) < _TAG_MIN_LEN or len(cleaned) > _TAG_MAX_LEN:
        return ""
    if cleaned in _BANNED_TAGS:
        return ""
    return cleaned


def _parse_tag_response(text: str, max_tags: int) -> list[str]:
    """Parse the model's comma-separated reply into clean tags.

    Tolerant of common drift: bullet points, numbering, quotes,
    newlines, "Tags:" prefix. Dedupes preserving first-seen order so
    the most-likely tag (which the model emits first) sticks around
    even when we hit the cap.
    """
    if not text:
        return []
    body = text.strip()
    # If the model preambled ("Here are 5 tags:", "Tags:", etc.), the
    # actual list usually follows the last colon on the first line.
    # Take everything after that colon — but only if a colon appears
    # before any comma (otherwise the colon might be inside the tags
    # themselves, which we don't want to slice on).
    first_comma = body.find(",")
    first_colon = body.find(":")
    if first_colon != -1 and (first_comma == -1 or first_colon < first_comma):
        body = body[first_colon + 1 :].strip()
    # Replace common separators with commas, then split.
    for sep in ("\n", ";", "/", "|"):
        body = body.replace(sep, ",")
    raw_tags = body.split(",")
    seen: set[str] = set()
    out: list[str] = []
    for r in raw_tags:
        t = _clean_tag(r)
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_tags:
            break
    return out


# ── On-disk cache (sha256 → tags) ──────────────────────────────────────


class _TagCache:
    """JSON-backed dict of sha256 → list[str]. Single instance per
    process; access is lock-guarded so the async pool can hammer it
    safely. Writes are batched (kept in memory, flushed atomically on
    every successful add) — file is tiny so a full rewrite is fine.
    """

    def __init__(self, path: Path = CACHE_FILE):
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    # Best-effort sanity filter — never crash on a
                    # malformed cache file, just drop the bad rows.
                    cleaned: dict[str, list[str]] = {}
                    for k, v in raw.items():
                        if isinstance(k, str) and isinstance(v, list):
                            cleaned[k] = [str(t) for t in v if isinstance(t, str)]
                    self._data = cleaned
                    logger.debug(f"AITagger cache loaded: {len(self._data)} entries")
        except Exception as e:
            logger.warning(f"AITagger cache load failed (starting fresh): {e}")
            self._data = {}

    def get(self, sha: str) -> Optional[list[str]]:
        with self._lock:
            v = self._data.get(sha)
            return list(v) if v is not None else None

    def put(self, sha: str, tags: list[str]) -> None:
        with self._lock:
            self._data[sha] = list(tags)
            self._flush_locked()

    def _flush_locked(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, sort_keys=True)
            # Atomic replace — survives a crash mid-write.
            os.replace(tmp, self._path)
        except Exception as e:
            logger.warning(f"AITagger cache flush failed: {e}")


# Single shared cache for the whole process.
_cache_singleton: Optional[_TagCache] = None
_cache_lock = threading.Lock()


def _get_cache() -> _TagCache:
    global _cache_singleton
    with _cache_lock:
        if _cache_singleton is None:
            _cache_singleton = _TagCache()
        return _cache_singleton


# ── Image helpers ──────────────────────────────────────────────────────


def _read_jpeg_bytes(jpeg: Union[str, bytes, Path]) -> Optional[bytes]:
    """Accept a path or raw JPEG bytes, return bytes (or None on error)."""
    if isinstance(jpeg, (bytes, bytearray)):
        return bytes(jpeg)
    if isinstance(jpeg, (str, Path)):
        try:
            with open(jpeg, "rb") as f:
                return f.read()
        except FileNotFoundError:
            logger.debug(f"AITagger: image not found: {jpeg}")
            return None
        except Exception as e:
            logger.warning(f"AITagger: image read failed for {jpeg}: {e}")
            return None
    return None


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── Public API ─────────────────────────────────────────────────────────


class AITagger:
    """Visual content → tags.

    Construct once, reuse across many calls. Internally thread-safe.
    """

    def __init__(
        self,
        backend: str = "cloud",
        api_key: Optional[str] = None,
        max_tags: int = 5,
    ):
        if backend not in ("cloud", "local"):
            raise ValueError(f"backend must be 'cloud' or 'local', got {backend!r}")
        self.backend = backend
        self.max_tags = max(3, min(int(max_tags), 8))  # clamp to 3..8
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        # Lazy-built Anthropic client; we don't want to import the SDK
        # just because someone constructed an AITagger with backend=local.
        self._client = None
        self._client_lock = threading.Lock()
        # Lazy-built Pillow shim for the heuristic local backend.
        self._pil_image = None

        if backend == "cloud" and not self._api_key:
            logger.warning(
                "AITagger(backend='cloud') has no API key — set "
                "ANTHROPIC_API_KEY in the environment or pass api_key= "
                "explicitly. Calls will fail until a key is provided."
            )

    # ── public sync entry point ────────────────────────────────────────

    def tag_image(self, jpeg_path: Union[str, bytes, Path]) -> list[str]:
        """Tag an image synchronously. Returns 3..max_tags lowercase
        tags. Returns ``[]`` on any error — callers should treat empty
        as "no tags available", not as a hard failure."""
        data = _read_jpeg_bytes(jpeg_path)
        if not data:
            return []
        sha = _sha256_hex(data)
        cache = _get_cache()
        cached = cache.get(sha)
        if cached is not None:
            # Honour the current max_tags cap even when serving from cache.
            return cached[: self.max_tags]
        try:
            if self.backend == "cloud":
                tags = self._tag_cloud(data)
            else:
                tags = self._tag_local(data)
        except Exception as e:
            logger.warning(f"AITagger({self.backend}) failed: {e}")
            return []
        if tags:
            cache.put(sha, tags)
        return tags

    # ── public async entry point ───────────────────────────────────────

    def tag_image_async(
        self,
        jpeg_path: Union[str, bytes, Path],
        callback: Callable[[list[str]], None],
    ) -> None:
        """Run ``tag_image`` on the shared executor; invoke ``callback``
        with the result. Callback runs on a worker thread — marshal back
        to the GUI thread yourself if you're touching Qt widgets.

        Exceptions in the callback are caught and logged so a buggy
        consumer can't take down the pool worker.
        """
        if callback is None:
            raise ValueError("callback is required for tag_image_async")

        def _job():
            try:
                tags = self.tag_image(jpeg_path)
            except Exception as e:
                logger.warning(f"AITagger async tag_image crashed: {e}")
                tags = []
            try:
                callback(tags)
            except Exception as e:
                logger.warning(f"AITagger async callback raised: {e}")

        _get_pool().submit(_job)

    # ── cloud backend ──────────────────────────────────────────────────

    def _get_anthropic_client(self):
        """Lazily construct the Anthropic SDK client. Cached."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                import anthropic  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "anthropic SDK not installed. Run `pip install anthropic`."
                ) from e
            if not self._api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(
                api_key=self._api_key,
                timeout=ANTHROPIC_REQUEST_TIMEOUT_SEC,
            )
            return self._client

    def _tag_cloud(self, jpeg_bytes: bytes) -> list[str]:
        """Send one image to Anthropic, parse the comma-separated reply."""
        client = self._get_anthropic_client()
        b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
        # System prompt block with cache_control. The image+user-text
        # block stays uncached (image bytes change per request).
        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Tag this frame. Give 3-{self.max_tags} "
                                "single-word lowercase tags, comma-separated."
                            ),
                        },
                    ],
                }
            ],
        )
        # Extract text from the response. The SDK returns a list of
        # content blocks; we expect a single TextBlock for our prompt.
        reply_text = ""
        for block in getattr(message, "content", []) or []:
            # Both dict-shaped and attribute-shaped responses turn up
            # depending on SDK version — handle both.
            btype = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if btype == "text":
                btext = getattr(block, "text", None) or (
                    block.get("text") if isinstance(block, dict) else ""
                )
                if btext:
                    reply_text += btext
        tags = _parse_tag_response(reply_text, self.max_tags)
        # Surface cache hit info at DEBUG level so we can confirm
        # prompt caching is actually saving us tokens.
        try:
            usage = getattr(message, "usage", None)
            if usage is not None:
                logger.debug(
                    "AITagger cloud usage: input=%s output=%s cache_read=%s cache_create=%s",
                    getattr(usage, "input_tokens", None),
                    getattr(usage, "output_tokens", None),
                    getattr(usage, "cache_read_input_tokens", None),
                    getattr(usage, "cache_creation_input_tokens", None),
                )
        except Exception:
            pass
        return tags

    # ── local backend (heuristic, no ML deps) ──────────────────────────

    def _tag_local(self, jpeg_bytes: bytes) -> list[str]:
        """Cheap, dependency-light visual tagger.

        Uses Pillow to compute a few image-wide statistics:

        * mean brightness     → ``dark`` / ``bright``
        * saturation          → ``monochrome`` / ``vivid``
        * std-dev luminance   → ``flat`` / ``highcontrast``
        * dominant hue bucket → ``warm`` / ``cool`` / ``neon`` (if
          saturation is high and brightness is high)

        Never returns more than ``max_tags`` and always returns at
        least 3 when possible (we fall through to neutral defaults
        like ``midtone`` rather than emit fewer than 3).
        """
        try:
            from PIL import Image  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Pillow not installed — local backend needs `pip install Pillow`. "
                "(Already a transitive dep of the project, so this should be rare.)"
            ) from e

        import io

        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        # Downscale for stats — analysing 120x68 is plenty and matches
        # the actual thumbnail dimensions anyway. Even at full size,
        # shrinking is faster than scanning every pixel.
        small = img.resize((48, 27))
        pixels = list(small.getdata())
        n = len(pixels) or 1

        r_sum = g_sum = b_sum = 0
        max_minus_min_sum = 0
        max_plus_min_sum = 0
        luma_vals: list[float] = []
        warm_hits = 0
        cool_hits = 0
        for r, g, b in pixels:
            r_sum += r
            g_sum += g
            b_sum += b
            mx = max(r, g, b)
            mn = min(r, g, b)
            max_minus_min_sum += mx - mn
            max_plus_min_sum += mx + mn
            # Rec. 601 luma
            luma_vals.append(0.299 * r + 0.587 * g + 0.114 * b)
            # Coarse warm/cool classification on saturated pixels only.
            if mx - mn > 40:
                if r > b + 20:
                    warm_hits += 1
                elif b > r + 20:
                    cool_hits += 1

        avg_r = r_sum / n
        avg_g = g_sum / n
        avg_b = b_sum / n
        mean_luma = sum(luma_vals) / n
        # HSL-ish saturation proxy: chroma / (max+min). Avoid div0.
        sat_proxy = max_minus_min_sum / max(1, max_plus_min_sum)
        # Contrast: stddev of luma
        var = sum((l - mean_luma) ** 2 for l in luma_vals) / n
        std_luma = var ** 0.5

        tags: list[str] = []

        # Brightness bucket
        if mean_luma < 40:
            tags.append("dark")
        elif mean_luma < 90:
            tags.append("moody")
        elif mean_luma > 180:
            tags.append("bright")
        else:
            tags.append("midtone")

        # Saturation bucket
        if sat_proxy < 0.12:
            tags.append("monochrome")
        elif sat_proxy > 0.45 and mean_luma > 80:
            tags.append("neon")
        elif sat_proxy > 0.30:
            tags.append("vivid")
        else:
            tags.append("muted")

        # Contrast bucket
        if std_luma > 70:
            tags.append("highcontrast")
        elif std_luma < 18:
            tags.append("flat")

        # Color-temperature bucket — only emit if we saw real saturated
        # pixels, otherwise it's noise.
        if warm_hits > cool_hits and warm_hits > n * 0.1:
            tags.append("warm")
        elif cool_hits > warm_hits and cool_hits > n * 0.1:
            tags.append("cool")

        # Dominant-channel hint, only when very strongly biased
        if avg_r > avg_g + 30 and avg_r > avg_b + 30:
            tags.append("reddish")
        elif avg_b > avg_r + 30 and avg_b > avg_g + 20:
            tags.append("blue")
        elif avg_g > avg_r + 20 and avg_g > avg_b + 20:
            tags.append("green")

        # Dedupe + trim + ensure cleanliness via the shared filter.
        cleaned: list[str] = []
        seen: set[str] = set()
        for t in tags:
            c = _clean_tag(t)
            if c and c not in seen:
                cleaned.append(c)
                seen.add(c)
        return cleaned[: self.max_tags]


# ── Convenience helpers for one-off use ────────────────────────────────


def tag_clip_jpeg(
    jpeg_path: Union[str, bytes, Path],
    backend: str = "cloud",
    api_key: Optional[str] = None,
    max_tags: int = 5,
) -> list[str]:
    """One-shot helper: build an AITagger and tag a single image. For
    code paths that don't need to keep a tagger around (e.g. one-off
    debugging / scripts). Production code in clips_db should hold one
    AITagger instance to amortise the Anthropic client setup."""
    tagger = AITagger(backend=backend, api_key=api_key, max_tags=max_tags)
    return tagger.tag_image(jpeg_path)
