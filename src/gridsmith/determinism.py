"""Determinism plumbing.

The brief (section 3.2) wants two runs to be byte-identical with timestamps isolated to
one ``extraction_runs`` row.  Three things break that by default, each handled here:

* **Iteration order.**  Anything derived from a ``set``, or from a dict keyed by floats,
  is re-orderable.  Every collection that reaches an output is sorted explicitly.
* **Ids.**  ``uuid4`` and ``AUTOINCREMENT`` differ between runs.  All ids here are
  content-addressed: ``sid("cell", table_id, row, col)`` gives a short blake2b digest.
  Same input, same id, forever -- which also makes the SQLite bytes reproducible.
* **Hidden clocks.**  ``openpyxl`` stamps ``docProps/core.xml`` and every zip entry with
  ``datetime.now()``.  ``gridsmith.s10_excel`` rewrites the archive with FIXED_ZIP_TIME.

Floating point: no raw float from a geometric computation is persisted without passing
through ``q`` (4 decimals).  1e-4 pt is 35 nanometres -- far below anything meaningful,
and it removes last-bit wobble from platform math libraries.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any, Iterable

FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
FIXED_ISO = "1970-01-01T00:00:00Z"
CHUNK = 1048576


def seed_everything(seed: int) -> None:
    """Seed every RNG we could plausibly touch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:  # pragma: no cover
        pass


def q(x, nd: int = 4):
    """Quantise a float for persistence.  ``-0.0`` normalises to ``0.0``."""
    if x is None:
        return None
    r = round(float(x) + 0.0, nd)
    return 0.0 if r == 0 else r


def sid(kind: str, *parts: Any) -> str:
    """Content-addressed, stable, short id.

    Collision budget: 96 bits over at most ~1e6 rows per document.  Birthday
    probability ~1e-17, orders of magnitude below the chance of an undetected disk
    error -- so a digest is safer here than a counter that would reorder between runs.
    """
    h = hashlib.blake2b(digest_size=12)
    h.update(kind.encode())
    for p in parts:
        h.update(b"\x1f")
        h.update(repr(p).encode("utf-8", "replace"))
    return kind[:3] + "_" + h.hexdigest()


def doc_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any, indent=2) -> str:
    """JSON that is byte-identical for equal objects: sorted keys, no NaN, LF only."""
    return json.dumps(
        obj, sort_keys=True, indent=indent, ensure_ascii=False, allow_nan=False,
        separators=(",", ": ") if indent else (",", ":"),
    )


def write_text(path: str, text: str) -> None:
    """UTF-8 with LF endings regardless of platform (Windows would emit CRLF)."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _norm(v: Any):
    if v is None:
        return (0, 0.0, "")
    if isinstance(v, bool):
        return (1, float(v), "")
    if isinstance(v, (int, float)):
        return (1, float(v), "")
    return (2, 0.0, str(v))


def stable_sort_key(*keys: Any):
    def _k(obj: Any):
        return tuple(_norm(getattr(obj, k, None) if isinstance(k, str) else k)
                     for k in keys)

    return _k


def sorted_stable(it: Iterable, key) -> list:
    """``sorted`` is already stable; this wrapper exists to make the intent greppable."""
    return sorted(it, key=key)
