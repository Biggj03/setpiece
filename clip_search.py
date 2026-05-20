"""
CLIP semantic search over the library.

Encodes a natural-language query with the CLIP text encoder, then ranks
every clip in `clip_embeddings` by cosine similarity to that query.
Since both query and stored vectors are L2-normalised float32, cosine
similarity is just the dot product — we run it as a single matrix
multiply.

USAGE
-----
    python clip_search.py "girl twerking outdoors" --top 20
    python clip_search.py "intense closeup" --top 10
    python clip_search.py "dance club vibe with neon lights" --top 30

OPTIONS
-------
    --root PATH    only consider files under this folder
    --top N        how many results to print (default 20)
    --min SCORE    drop results below this cosine similarity
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np

# Reuse the model + DB constants from the tagger so they can't drift.
from clip_tagger import (
    DB_PATH,
    EMBED_DIM,
    MODEL_NAME,
    MODEL_PRETRAINED,
    MODEL_TAG,
)

logger = logging.getLogger(__name__)


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _load_text_model(device: str | None = None):
    """Load the CLIP text encoder + tokenizer on `device`. Returns
    (model, tokenizer, device_used)."""
    import torch
    import open_clip

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _train_preprocess, _preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=MODEL_PRETRAINED
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, tokenizer, device


def embed_text(text: str, model=None, tokenizer=None, device: str | None = None) -> np.ndarray:
    """Return L2-normalised float32 (512,) embedding of `text`."""
    import torch

    if model is None or tokenizer is None or device is None:
        model, tokenizer, device = _load_text_model(device)
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)  # (1, 512)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-9)
    return feats[0].detach().cpu().to(torch.float32).numpy().astype(np.float32, copy=False)


def _load_embeddings_matrix(
    conn: sqlite3.Connection, root: str | None
) -> tuple[list[str], np.ndarray]:
    """Pull every clip_embeddings row into a (N, EMBED_DIM) float32
    matrix in memory. Returns (filepaths, matrix). At 512 dims * 4
    bytes = 2 KB per clip, the full 5678-file library is ~11 MB —
    trivial to hold in RAM."""
    cur = conn.cursor()
    if root:
        like_a = root.replace("/", "\\") + "%"
        like_b = root.replace("\\", "/") + "%"
        rows = cur.execute(
            "SELECT filepath, vec FROM clip_embeddings "
            "WHERE model = ? AND (filepath LIKE ? OR filepath LIKE ?)",
            (MODEL_TAG, like_a, like_b),
        ).fetchall()
    else:
        rows = cur.execute(
            "SELECT filepath, vec FROM clip_embeddings WHERE model = ?",
            (MODEL_TAG,),
        ).fetchall()
    if not rows:
        return [], np.zeros((0, EMBED_DIM), dtype=np.float32)
    paths = [r[0] for r in rows]
    mat = np.empty((len(rows), EMBED_DIM), dtype=np.float32)
    for i, (_, blob) in enumerate(rows):
        v = np.frombuffer(blob, dtype=np.float32)
        if v.shape[0] != EMBED_DIM:
            raise ValueError(
                f"bad vec dim {v.shape[0]} for {paths[i]}"
            )
        mat[i] = v
    return paths, mat


def search(query: str, top: int = 20, root: str | None = None,
           min_score: float = 0.0) -> list[tuple[float, str]]:
    """Return top-N (score, filepath) pairs sorted descending by cosine
    similarity to `query`."""
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"no DB at {DB_PATH}")
    conn = _open_db()
    try:
        paths, mat = _load_embeddings_matrix(conn, root)
    finally:
        conn.close()
    if not paths:
        return []
    q = embed_text(query)
    # Both sides L2-normalised -> dot product == cosine similarity.
    sims = mat @ q  # shape (N,)
    if top >= len(paths):
        order = np.argsort(-sims)
    else:
        # argpartition: cheap top-k for big N.
        idx = np.argpartition(-sims, top)[:top]
        order = idx[np.argsort(-sims[idx])]
    out: list[tuple[float, str]] = []
    for i in order[:top]:
        s = float(sims[i])
        if s < min_score:
            continue
        out.append((s, paths[i]))
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("query", help="natural-language search query")
    ap.add_argument("--top", type=int, default=20,
                    help="how many results to print (default 20)")
    ap.add_argument("--root", default=None,
                    help="only consider files under this folder")
    ap.add_argument("--min", dest="min_score", type=float, default=0.0,
                    help="drop results below this cosine similarity")
    args = ap.parse_args()

    results = search(
        args.query, top=args.top, root=args.root, min_score=args.min_score
    )
    if not results:
        print("(no results — is the library embedded yet? "
              "run clip_tagger.py first)")
        return 1
    print(f"Top {len(results)} for: {args.query!r}")
    for score, fp in results:
        print(f"  {score:+.4f}  {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
