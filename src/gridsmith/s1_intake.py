"""S1 -- document intake and per-page typing.

Per page, not per document (the brief is explicit, and it matters: a scanned exhibit
pasted into an otherwise digital annual report is the common case).

Three classes:

``digital``
    Enough real glyphs that the text layer *is* the ground truth.  We never re-OCR
    these -- OCR can only lose information we already have exactly.
``scanned``
    No usable text layer, one big raster.  Needs deskew/denoise, then per-cell OCR.
``hybrid``
    Both: a text layer that covers part of the page plus a raster that covers a region
    the text layer does not reach.  We extract text-layer regions from the text layer
    and raster regions by OCR, and record which was used per cell.

Geometry normalisation done here:

*   ``/Rotate`` pages -- PyMuPDF applies the rotation matrix for us, so text and
    renders agree and nothing downstream needs to know.  We record the value.
*   **Rotated content without /Rotate** (a landscape table dropped into a portrait
    page).  Detected from the writing direction of the text lines; the affected region
    is transposed in :func:`transpose_box` so S3/S4 see an upright table.
*   **Skew** on rasters, estimated by a sheared projection-profile search (below) and
    corrected once, at intake, so every later stage works in one frame.

The skew estimator deserves a note.  The textbook approach rotates the image at each
candidate angle and scores the horizontal projection profile; that is O(pixels) *image
rotations*.  Instead we take the ink pixel coordinates once and, for each candidate
angle, shear them (``y' = y - x*tan(a)``) and histogram with ``np.bincount``.  Same
objective, no interpolation, ~50x faster, and exactly reproducible because it never
touches a resampling filter.  Bias note: the projection range grows with ``|a|``, which
slightly favours 0 degrees -- a conservative bias, since the failure we care about is
rotating a page that was never skewed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pymupdf

from .config import Config
from .geometry import area, intersect
from .textlayer import PageText, extract_page_text

PAGE_DIGITAL = "digital"
PAGE_SCANNED = "scanned"
PAGE_HYBRID = "hybrid"
PAGE_EMPTY = "empty"


@dataclass
class PageRaster:
    """A rendered, deskewed page image plus the transform back to PDF points."""

    gray: np.ndarray          # uint8, 0 = ink
    dpi: int
    skew_deg: float
    scale: float              # px per point
    offset: tuple = (0.0, 0.0)  # px added by the deskew expand

    def to_px(self, box):
        x0, y0, x1, y1 = box
        s, (ox, oy) = self.scale, self.offset
        return (x0 * s + ox, y0 * s + oy, x1 * s + ox, y1 * s + oy)

    def to_pt(self, box):
        x0, y0, x1, y1 = box
        s, (ox, oy) = self.scale, self.offset
        return ((x0 - ox) / s, (y0 - oy) / s, (x1 - ox) / s, (y1 - oy) / s)

    def ink_mask(self, thresh=None):
        t = otsu(self.gray) if thresh is None else thresh
        return self.gray <= t


@dataclass
class PageInfo:
    page_no: int
    width: float
    height: float
    rotation: int
    page_type: str
    text: PageText
    image_boxes: list = field(default_factory=list)
    image_area_frac: float = 0.0
    char_density: float = 0.0
    skew_deg: float = 0.0
    content_rotation: int = 0     # 0 / 90 / 270 rotation of the *content*
    raster_regions: list = field(default_factory=list)  # boxes needing OCR
    notes: list = field(default_factory=list)


def open_document(path: str):
    """Open a PDF, dealing with the empty-password encryption EDGAR uses."""
    doc = pymupdf.open(path)
    if doc.is_encrypted and doc.needs_pass:
        if not doc.authenticate(""):
            raise RuntimeError(
                "PDF is password protected; gridsmith will not guess passwords")
    return doc


# --------------------------------------------------------------------------- typing


def classify_page(page, pt: PageText, cfg: Config) -> PageInfo:
    """Assign a page type from text-layer density versus raster coverage."""
    w, h = pt.width, pt.height
    page_area = max(1.0, w * h)
    img_boxes = []
    for blk in page.get_text("dict").get("blocks", []):
        if blk.get("type", 0) == 1:
            bb = tuple(float(v) for v in blk.get("bbox", (0, 0, 0, 0)))
            if area(bb) >= 1.0:
                img_boxes.append(bb)
    # Also catch images placed without a text block (rare but happens with XObjects).
    for info in page.get_image_info(xrefs=False):
        bb = tuple(float(v) for v in info.get("bbox", (0, 0, 0, 0)))
        if area(bb) >= 1.0 and not any(_near(bb, o) for o in img_boxes):
            img_boxes.append(bb)
    img_boxes.sort()

    covered = _union_area(img_boxes, (0.0, 0.0, w, h))
    img_frac = covered / page_area
    density = pt.n_chars / (page_area / 1000.0)

    if pt.n_chars < cfg.digital_min_chars:
        ptype = PAGE_SCANNED if img_frac > 0.20 else PAGE_EMPTY
    elif img_frac >= cfg.scan_image_area_frac and density < cfg.hybrid_min_char_density:
        ptype = PAGE_SCANNED
    elif img_frac >= 0.10:
        ptype = PAGE_HYBRID
    else:
        ptype = PAGE_DIGITAL

    info = PageInfo(page_no=pt.page_no, width=w, height=h, rotation=int(page.rotation),
                    page_type=ptype, text=pt, image_boxes=img_boxes,
                    image_area_frac=img_frac, char_density=density)

    if ptype == PAGE_HYBRID:
        info.raster_regions = _text_free_images(img_boxes, pt, cfg)
        if not info.raster_regions:
            # Images exist but the text layer covers them (chart labels, logos).
            info.page_type = PAGE_DIGITAL
            info.notes.append("hybrid_demoted_text_covers_images")
    elif ptype == PAGE_SCANNED:
        info.raster_regions = [(0.0, 0.0, w, h)]

    info.content_rotation = detect_content_rotation(pt, cfg)
    return info


def _near(a, b, tol=1.0):
    return all(abs(a[i] - b[i]) <= tol for i in range(4))


def _union_area(boxes, clip):
    """Area of the union of boxes, clipped.  Boxes here are few, so O(n^2) is fine."""
    if not boxes:
        return 0.0
    clipped = [intersect(b, clip) for b in boxes]
    clipped = [b for b in clipped if b]
    if not clipped:
        return 0.0
    xs = sorted({v for b in clipped for v in (b[0], b[2])})
    ys = sorted({v for b in clipped for v in (b[1], b[3])})
    total = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in clipped):
                total += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return total


def _text_free_images(img_boxes, pt: PageText, cfg: Config):
    """Image regions that the text layer does not explain -- these need OCR."""
    out = []
    for bb in img_boxes:
        if area(bb) < 4000.0:
            continue          # logo / rule / bullet, not a pasted exhibit
        inside = sum(1 for t in pt.tokens
                     if bb[0] <= t.cx <= bb[2] and bb[1] <= t.cy <= bb[3])
        # A pasted scan has ink but no glyphs of its own.  Allow a couple of stray
        # tokens (a caption overlapping the frame) before calling it digital.
        if inside <= 2:
            out.append(bb)
    return out


def detect_content_rotation(pt: PageText, cfg: Config) -> int:
    """Detect a page whose *content* is rotated 90/270 without a /Rotate entry.

    Signal: the fraction of text lines whose writing direction is vertical.  We only
    claim rotation when it is the clear majority, because a single vertical axis label
    on a chart is normal and must not transpose the page.
    """
    if not pt.lines:
        return 0
    n_vert = pt.n_vertical_lines
    total = max(1, sum(len(b.get("text", "").splitlines()) or 1 for b in pt.blocks))
    if n_vert / total < 0.6 or n_vert < 4:
        return 0
    # Direction sign tells 90 from 270.  Positive dy means glyphs advance downward.
    return 90


def transpose_box(box, page_w: float, page_h: float, rot: int):
    """Map a box between upright space and 90/270-rotated content space."""
    x0, y0, x1, y1 = box
    if rot == 90:
        return (y0, page_w - x1, y1, page_w - x0)
    if rot == 270:
        return (page_h - y1, x0, page_h - y0, x1)
    return box


# ------------------------------------------------------------------------- rasterise


def render_page(page, cfg: Config, clip=None, dpi=None) -> PageRaster:
    """Render to grayscale numpy at ``dpi``.  Never keeps the Pixmap alive."""
    dpi = dpi or cfg.raster_dpi
    scale = dpi / 72.0
    mat = pymupdf.Matrix(scale, scale)
    kw = {"matrix": mat, "colorspace": pymupdf.csGRAY, "alpha": False}
    if clip is not None:
        kw["clip"] = pymupdf.Rect(*clip)
    pix = page.get_pixmap(**kw)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    off = (0.0, 0.0)
    if clip is not None:
        off = (-clip[0] * scale, -clip[1] * scale)
    del pix
    return PageRaster(gray=arr, dpi=dpi, skew_deg=0.0, scale=scale, offset=off)


def otsu(gray: np.ndarray) -> int:
    """Otsu threshold from a 256-bin histogram.  Pure numpy, fully deterministic."""
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 128
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / denom, 0.0)
    return int(np.argmax(sigma_b))


def estimate_skew(gray: np.ndarray, cfg: Config) -> float:
    """Skew angle in degrees (positive = content rotated clockwise).

    Two-pass search: 1.0 degree steps over the full range, then ``deskew_step_deg``
    around the winner.  Returns 0.0 unless the best angle beats 0 degrees by 2%, which
    keeps us from "correcting" an unskewed page.
    """
    thr = otsu(gray)
    ink = gray <= thr
    h, w = ink.shape
    if ink.sum() < 200:
        return 0.0
    # Subsample rows for speed; skew is a global property, 1 row in 2 is plenty.
    ys, xs = np.nonzero(ink[::2, ::2])
    if ys.size < 200:
        return 0.0
    ys = ys.astype(np.float32) * 2.0
    xs = xs.astype(np.float32) * 2.0

    def score(angle_deg: float) -> float:
        t = math.tan(math.radians(angle_deg))
        yp = ys - xs * t
        yp -= yp.min()
        bins = np.bincount(yp.astype(np.int32))
        return float(np.dot(bins, bins))

    base = score(0.0)
    # Coarse step must be finer than the peak.  At 200 dpi a 1 degree error shifts the
    # far edge of an A4 page by ~28 px, comfortably more than a text line height, so
    # the projection peak is well under 1 degree wide -- and a 1 degree scan walked
    # straight past a real 1.6 degree skew whose peak was 1.83x the unskewed score.
    coarse_step = 0.25
    n_coarse = int(round(2 * cfg.deskew_max_deg / coarse_step))
    best_a, best_s = 0.0, base
    for i in range(n_coarse + 1):
        a = -cfg.deskew_max_deg + i * coarse_step
        s = score(a)
        if s > best_s:
            best_a, best_s = a, s
    lo, hi = best_a - coarse_step, best_a + coarse_step
    n = max(1, int(round((hi - lo) / cfg.deskew_step_deg)))
    for i in range(n + 1):
        a = lo + i * cfg.deskew_step_deg
        if abs(a) > cfg.deskew_max_deg:
            continue
        s = score(a)
        if s > best_s:
            best_a, best_s = a, s
    if best_s < base * 1.02:
        return 0.0
    return round(best_a, 2)


def deskew(raster: PageRaster, angle_deg: float) -> PageRaster:
    """Rotate the raster by ``-angle`` about its centre, padding with white."""
    if abs(angle_deg) < 0.05:
        return raster
    from PIL import Image

    img = Image.fromarray(raster.gray, mode="L")
    out = img.rotate(angle_deg, resample=Image.BICUBIC, expand=False, fillcolor=255)
    return PageRaster(gray=np.asarray(out, dtype=np.uint8), dpi=raster.dpi,
                      skew_deg=angle_deg, scale=raster.scale, offset=raster.offset)


def salt_pepper_ratio(gray: np.ndarray) -> float:
    """Fraction of ink pixels with no ink 4-neighbour -- a cheap scan-noise proxy."""
    ink = (gray <= otsu(gray)).astype(np.uint8)
    if ink.sum() == 0:
        return 0.0
    nb = np.zeros_like(ink)
    nb[1:, :] += ink[:-1, :]
    nb[:-1, :] += ink[1:, :]
    nb[:, 1:] += ink[:, :-1]
    nb[:, :-1] += ink[:, 1:]
    isolated = int(((ink == 1) & (nb == 0)).sum())
    return isolated / float(ink.sum())


def denoise(raster: PageRaster, ratio_threshold: float = 0.012) -> PageRaster:
    """3x3 median filter, applied only when the page is measurably noisy.

    Unconditional median filtering thins hairline rules, which is exactly the signal
    S3 depends on, so we gate it on a noise measurement instead.
    """
    if salt_pepper_ratio(raster.gray) < ratio_threshold:
        return raster
    from PIL import Image, ImageFilter

    img = Image.fromarray(raster.gray, mode="L").filter(ImageFilter.MedianFilter(3))
    return PageRaster(gray=np.asarray(img, dtype=np.uint8), dpi=raster.dpi,
                      skew_deg=raster.skew_deg, scale=raster.scale,
                      offset=raster.offset)


def raster_orientation(gray: np.ndarray) -> tuple:
    """Guess 0 vs 90 degree text orientation on an image-only page.

    Latin text lines make the projection profile *across* the lines strongly periodic
    and high-variance, while the profile along them is smooth.  Comparing the two
    variances gives the axis.  Returns ``(degrees, confidence)`` where confidence is
    the normalised margin; the caller flags anything under ~0.15 rather than trusting
    it, and 90-vs-270 is left to :func:`raster_updown` / Tesseract OSD.
    """
    ink = (gray <= otsu(gray))
    if ink.sum() < 200:
        return 0, 0.0
    rows = ink.sum(axis=1).astype(np.float64)
    cols = ink.sum(axis=0).astype(np.float64)

    def periodicity(p):
        if p.sum() <= 0:
            return 0.0
        p = p / (p.mean() + 1e-9)
        d = np.diff(p)
        return float(np.var(d))

    pr, pc = periodicity(rows), periodicity(cols)
    total = pr + pc + 1e-9
    if pr >= pc:
        return 0, float((pr - pc) / total)
    return 90, float((pc - pr) / total)


def prepare_raster(page, info: PageInfo, cfg: Config) -> PageRaster:
    """Render + deskew + denoise a scanned/hybrid page, recording what we did."""
    raster = render_page(page, cfg)
    ang = estimate_skew(raster.gray, cfg)
    if ang:
        if abs(ang) >= cfg.deskew_max_deg - 1e-6:
            info.notes.append("skew_beyond_limit_not_corrected")
        else:
            raster = deskew(raster, ang)
            info.skew_deg = ang
            info.notes.append("deskewed_%.2fdeg" % ang)
    before = salt_pepper_ratio(raster.gray)
    raster = denoise(raster)
    if raster.skew_deg == ang and before >= 0.012:
        info.notes.append("denoised_median3")
    if info.page_type == PAGE_SCANNED:
        deg, conf = raster_orientation(raster.gray)
        if deg and conf > 0.15:
            info.content_rotation = deg
            info.notes.append("raster_orientation_%d_conf_%.2f" % (deg, conf))
        elif conf <= 0.15:
            info.notes.append("raster_orientation_uncertain_conf_%.2f" % conf)
    return raster
