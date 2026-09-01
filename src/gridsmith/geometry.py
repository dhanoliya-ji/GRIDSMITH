"""Rectangle and interval helpers.

Coordinate convention for the whole pipeline: **PDF points, origin at the top-left of
the page as displayed**, x growing right, y growing down.  That is PyMuPDF's convention
and it already accounts for a page /Rotate entry, so a 90-degree-rotated page arrives
here upright and no stage below S1 needs to think about it.

Content rotated *without* a /Rotate entry (a landscape table dropped into a portrait
page) is a different problem, handled in gridsmith.s1_intake by transposing the
coordinates of that region alone.
"""
from __future__ import annotations

from typing import Iterable, Sequence

Box = tuple  # (x0, y0, x1, y1) in points


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def width(b: Box) -> float:
    return max(0.0, b[2] - b[0])


def height(b: Box) -> float:
    return max(0.0, b[3] - b[1])


def union(bs: Iterable[Box]):
    bs = list(bs)
    if not bs:
        return None
    return (min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs))


def intersect(a: Box, b: Box):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def iou(a: Box, b: Box) -> float:
    inter = intersect(a, b)
    if inter is None:
        return 0.0
    ia = area(inter)
    return ia / (area(a) + area(b) - ia + 1e-9)


def contain_frac(inner: Box, outer: Box) -> float:
    """How much of ``inner`` sits inside ``outer`` (0..1)."""
    inter = intersect(inner, outer)
    return 0.0 if inter is None else area(inter) / (area(inner) + 1e-9)


def overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def iou_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    ov = overlap_1d(a0, a1, b0, b1)
    un = (a1 - a0) + (b1 - b0) - ov
    return ov / (un + 1e-9)


def inflate(b: Box, dx: float, dy=None) -> Box:
    dy = dx if dy is None else dy
    return (b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy)


def cluster_1d(values: Sequence, tol: float) -> list:
    """Single-link clustering on a line.

    Returns clusters of *indices into the original sequence*, ordered by position.
    Used for snapping collinear rules, column edges and row baselines.  Deterministic:
    equal values are broken by original index.
    """
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    clusters = [[order[0]]]
    last = values[order[0]]
    for i in order[1:]:
        if values[i] - last <= tol:
            clusters[-1].append(i)
        else:
            clusters.append([i])
        last = values[i]
    return clusters


def cluster_centers(values: Sequence, tol: float) -> list:
    return [sum(values[i] for i in c) / len(c) for c in cluster_1d(values, tol)]


def merge_intervals(iv: Sequence, gap: float = 0.0) -> list:
    if not iv:
        return []
    out = []
    for a, b in sorted(iv):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def gaps_in(iv: Sequence, lo: float, hi: float) -> list:
    """Complement of merged intervals inside ``[lo, hi]`` -- i.e. the whitespace."""
    merged = merge_intervals(iv)
    out, cur = [], lo
    for a, b in merged:
        if a > cur:
            out.append((cur, min(a, hi)))
        cur = max(cur, b)
        if cur >= hi:
            break
    if cur < hi:
        out.append((cur, hi))
    return [(a, b) for a, b in out if b > a]


def median(xs: Sequence) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return float(s[n // 2]) if n % 2 else (float(s[n // 2 - 1]) + float(s[n // 2])) / 2.0
