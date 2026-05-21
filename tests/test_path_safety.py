"""Tests for the library-root boundary check.

`VJPracticeApp._resolve_inside_root` is the primitive that keeps
client-supplied folder paths from escaping the configured library root.
`bank_from_folder`, `library_cd`, `library_load_file` and
`load_file_to_deck` all delegate to it — these tests are the regression
guard for that whole class of path-traversal bug.

Importing `main` pulls in PyQt6; the test is skipped if it's absent.
"""
import pytest

pytest.importorskip("PyQt6")
import main  # noqa: E402

# Unbound method — its body never touches `self`, so any dummy works.
resolve_inside_root = main.VJPracticeApp._resolve_inside_root


def test_path_inside_root_is_accepted(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    inside = root / "sub"
    inside.mkdir()
    assert resolve_inside_root(None, root, inside) is not None


def test_dotdot_escape_is_rejected(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    (tmp_path / "secret").mkdir()
    escape = root / ".." / "secret"
    assert resolve_inside_root(None, root, escape) is None


def test_absolute_path_outside_root_is_rejected(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    assert resolve_inside_root(None, root, other) is None
