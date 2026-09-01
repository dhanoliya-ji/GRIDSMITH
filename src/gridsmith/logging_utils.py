"""Structured, deterministic run log.

``logs/run.jsonl`` has to be byte-identical between runs (§3.2), which rules out the
two things a log normally contains: a wall-clock timestamp and a duration.  Events are
therefore numbered with a monotonic sequence instead, and anything genuinely temporal
goes to ``logs/timing.json``, which is explicitly excluded from the determinism claim.

Everything is buffered and written once at the end so a partially-flushed file can
never look like a complete one.
"""
from __future__ import annotations

from .determinism import canonical_json, write_text


class RunLog:
    def __init__(self):
        self._events = []

    def event(self, kind: str, **fields) -> None:
        self._events.append({"seq": len(self._events), "event": kind,
                             **{k: _plain(v) for k, v in fields.items()}})

    def write(self, path: str) -> None:
        lines = [canonical_json(e, None) for e in self._events]
        write_text(path, "\n".join(lines) + "\n")

    def __len__(self):
        return len(self._events)


def _plain(v):
    """Coerce to JSON-safe, sorted, finite values."""
    if isinstance(v, dict):
        return {str(k): _plain(v[k]) for k in sorted(v, key=str)}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, set):
        return sorted(_plain(x) for x in v)
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return str(v)
