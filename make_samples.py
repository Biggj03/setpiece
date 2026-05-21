"""Generate a small set of synthetic demo clips.

Setpiece needs a folder of video clips to perform. For a quick demo
(or CI / first-run smoke test) you may not have a library handy — this
script synthesises one with ffmpeg's built-in `lavfi` sources.

Every clip is generated from a procedural source: test patterns,
fractals, cellular automata, colour fields, gradients. Nothing is
downloaded; the output is 100% self-produced and content-neutral, and
it spans enough visual variety (warm/cool colour, static/dynamic
motion, geometric/organic form) to give the taggers something real to
chew on.

USAGE
-----
    python make_samples.py                 # -> ./samples/
    python make_samples.py --out demo_lib  # custom output folder
    python make_samples.py --duration 12   # longer clips

Requires ffmpeg on PATH. Then:

    python ingest.py --root samples --skip pose --skip vision
    python main.py        # point it at the samples/ folder
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# (filename, lavfi source spec) — each is a procedural ffmpeg source.
# Names are descriptive so the demo library reads sensibly.
CLIPS: list[tuple[str, str]] = [
    ("mandelbrot_zoom",  "mandelbrot=s={w}x{h}:rate={fps}"),
    ("game_of_life",     "life=s={w}x{h}:rate={fps}:mold=10:"
                         "life_color=0x33ff66:death_color=0x001022"),
    ("cellular_110",     "cellauto=s={w}x{h}:rate={fps}:rule=110:scroll=1"),
    ("test_pattern",     "testsrc2=s={w}x{h}:rate={fps}"),
    ("rgb_bars",         "rgbtestsrc=s={w}x{h}:rate={fps}"),
    ("smpte_bars",       "smptebars=s={w}x{h}:rate={fps}"),
    ("yuv_pattern",      "yuvtestsrc=s={w}x{h}:rate={fps}"),
    ("warm_field",       "color=c=0xCC4422:s={w}x{h}:rate={fps}"),
    ("cool_field",       "color=c=0x2244CC:s={w}x{h}:rate={fps}"),
    ("gradient_warm",    "gradients=s={w}x{h}:rate={fps}:"
                         "c0=0xff6600:c1=0x220000:speed=0.012"),
    ("gradient_cool",    "gradients=s={w}x{h}:rate={fps}:"
                         "c0=0x0088ff:c1=0x000022:speed=0.012"),
    ("mono_noise",       "nullsrc=s={w}x{h}:rate={fps},"
                         "geq=lum='random(1)*255':cb=128:cr=128"),
]


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _gen_one(source: str, dest: Path, duration: int) -> bool:
    """Render one lavfi source to an h264 mp4. Returns True on success."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", source,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        str(dest),
    ]
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"  ! {dest.name}: {e}")
        return False
    if rc != 0:
        print(f"  ! {dest.name}: ffmpeg exited {rc}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", default="samples",
                    help="output folder (default: samples)")
    ap.add_argument("--duration", type=int, default=8,
                    help="clip length in seconds (default: 8)")
    ap.add_argument("--width", type=int, default=854)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()

    if not _have_ffmpeg():
        print("ERROR: ffmpeg not found on PATH. Install it first "
              "(see SETUP.md), then re-run.")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(CLIPS)} demo clips into {out}/ "
          f"({args.width}x{args.height}, {args.duration}s each)")

    ok = 0
    for name, spec in CLIPS:
        dest = out / f"{name}.mp4"
        if dest.exists():
            print(f"  = {dest.name} (exists, skipped)")
            ok += 1
            continue
        source = spec.format(w=args.width, h=args.height, fps=args.fps)
        if _gen_one(source, dest, args.duration):
            print(f"  + {dest.name}")
            ok += 1

    print(f"\nDone: {ok}/{len(CLIPS)} clips in {out}/")
    if ok:
        print("\nNext:")
        print(f"  python ingest.py --root {args.out} --skip pose --skip vision")
        print(f"  python main.py        # then point it at {args.out}/")
    return 0 if ok == len(CLIPS) else 1


if __name__ == "__main__":
    sys.exit(main())
