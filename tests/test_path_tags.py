"""Tests for the path-derived tagger primitives.

These are pure functions — no Qt, no database, no ffmpeg — so they run
fast and anywhere.
"""
import path_tags_v2 as ptv2


def test_canonical_path_is_idempotent():
    once = ptv2.canonical_path("C:/Foo/Bar/../Bar/clip.mp4")
    twice = ptv2.canonical_path(once)
    assert once == twice


def test_fold_stem_collapses_known_plurals():
    assert ptv2._fold_stem("girls") == "girl"
    assert ptv2._fold_stem("parties") == "party"
    assert ptv2._fold_stem("loops") == "loop"


def test_fold_stem_leaves_short_words_alone():
    # <= 4 chars are never folded ("bus" must not become "bu").
    assert ptv2._fold_stem("bus") == "bus"


def test_drop_token_discards_noise():
    # codec / container / resolution cruft
    assert ptv2._drop_token("x265")
    assert ptv2._drop_token("1080p")
    assert ptv2._drop_token("mp4")
    # too-short and pure-digit tokens
    assert ptv2._drop_token("ab")
    assert ptv2._drop_token("12345")


def test_drop_token_keeps_real_words():
    assert not ptv2._drop_token("sunset")
    assert not ptv2._drop_token("aurora")


def test_tokenize_path_extracts_words_and_drops_noise():
    tags = ptv2.tokenize_path(
        "/library/Neon Sunset/aurora-loop-1080p.mp4",
        library_root="/library",
    )
    # descriptive filename tokens survive
    assert "aurora" in tags
    assert "loop" in tags
    # the folder yields one whole-folder tag
    assert "neon-sunset" in tags
    # codec / resolution noise is filtered out
    assert "1080p" not in tags
    assert "mp4" not in tags
