"""S7 value normalisation and S8 table context (titles, units, footnotes).

Sections, in order: s7_normalize, s8_context.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .core import Config, area, cluster_1d, contain_frac, union
from .reading import CAPTION_RE, SCALE_RE


# --------------------------------------------------------------------------------------------------
# S7 -- normalisation and typing.  Additive, never destructive.
# --------------------------------------------------------------------------------------------------
# ``raw_text`` is always kept exactly as the page had it.  Everything here is a *second*
# opinion recorded alongside it, and every opinion carries flags saying how sure it is.
#
# The governing rule is section 4 of the brief: no wrong value at high confidence.  That
# turns several ordinary parsing decisions into refusals:
#
# ``12,34,567`` vs ``12,345,67``
#     Indian and Western grouping are both accepted, but the grouping *pattern* is
#     validated rather than the commas simply deleted.  A string whose grouping matches
#     neither convention parses to a number and is flagged ``AMBIGUOUS_GROUPING``, so a
#     typo in the source cannot arrive as a clean value.
#
# ``1.234,56``
#     Continental European notation collides head-on with Western: is ``1.234`` one
#     thousand two hundred or one point two three four?  There is no way to know from one
#     cell.  We flag ``AMBIGUOUS_DECIMAL_SEPARATOR`` and leave ``normalized_value`` null.
#     Guessing here would be exactly the silent error the brief is built around.
#
# ``03/04/2024``
#     Same problem for dates: 3 April or 4 March.  If both fields are 12 or under and the
#     document has not established an order, ``normalized_value`` stays null with
#     ``AMBIGUOUS_DATE_ORDER``.  A date with an unambiguous field (``31/03/2024``) parses
#     normally and also *teaches* the document its order, which later resolves the
#     ambiguous ones -- see :func:`DateOrderModel`.
#
# ``#REF!``
#     Excel error values survive printing to PDF.  ``Shell.pdf`` ships six of them inside
#     the cash-flow statement.  They are typed ``error``, never coerced to 0 or null-and-
#     forgotten, and they raise an issue -- a number that the source document itself could
#     not compute must not look like a number downstream.
#
# ``-`` / ``Nil`` / ``NA`` / ``0`` / empty
#     Five distinct value types, because they mean five different things to an accountant
#     and collapsing them loses information no downstream user can recover.

# ------------------------------------------------------------------- value types
T_EMPTY = "empty"
T_DASH = "dash"
T_NIL = "nil"
T_NA = "na"
T_NUMBER = "number"
T_CURRENCY = "currency"
T_PERCENT = "percent"
T_DATE = "date"
T_PERIOD = "period"
T_ERROR = "error"
T_REFERENCE = "reference"
T_TEXT = "text"

NUMERIC_TYPES = (T_NUMBER, T_CURRENCY, T_PERCENT)
NULLISH_TYPES = (T_EMPTY, T_DASH, T_NIL, T_NA)

CURRENCY_SYMBOLS = {"$", "US$", "₹", "€", "£", "¥", "₨", "`", "R$", "A$", "C$", "CHF"}
CURRENCY_WORDS = {"rs", "rs.", "inr", "usd", "eur", "gbp", "jpy", "aed", "sgd",
                  "rupees", "dollars", "euro", "euros"}
# The ICRA and Shell documents both render the rupee sign as U+0060 GRAVE ACCENT via a
# legacy font encoding: "(` Million)".  Treating it as INR is a *flagged* inference.
LEGACY_RUPEE = "`"

MAGNITUDES = {
    "k": 1e3, "thousand": 1e3, "thousands": 1e3, "000s": 1e3,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
    "mn": 1e6, "m": 1e6, "million": 1e6, "millions": 1e6,
    "cr": 1e7, "crore": 1e7, "crores": 1e7,
    "bn": 1e9, "b": 1e9, "billion": 1e9, "billions": 1e9,
    "tn": 1e12, "trillion": 1e12,
}

ERROR_VALUES = {"#ref!", "#div/0!", "#value!", "#name?", "#num!", "#null!",
                "#n/a", "#na", "#error", "#calc!", "#spill!", "#getting_data"}
NIL_WORDS = {"nil", "none"}
NA_WORDS = {"na", "n.a.", "n.a", "n/a", "not applicable", "not available", "n.m.",
            "nm", "not meaningful", "-na-"}
DASH_CHARS = {"-", "–", "—", "‐", "‑", "‒", "―", "−", "﹘", "－"}

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

WESTERN_GROUP_RE = re.compile(r"^\d{1,3}(,\d{3})+$")
INDIAN_GROUP_RE = re.compile(r"^\d{1,2}(,\d{2})+,\d{3}$")
PLAIN_INT_RE = re.compile(r"^\d+$")
# Digits and numeric punctuation only -- the gate before any number parse.
NUMERIC_SHAPE_RE = re.compile(r"^[\d.,'\s]*\d[\d.,'\s]*$")
PERIOD_RE = re.compile(
    r"^(?:(?:fy|ay|cy|q[1-4]|h[12])\s*[-']?\s*)+"
    r"(?:\d{2,4}(?:\s*[-/–]\s*\d{2,4})?)?$|"
    r"^\d{4}\s*[-–]\s*\d{2,4}$|"
    r"^q[1-4]\s*(?:fy)?\s*\d{2,4}$", re.I)
REFERENCE_HEADER_RE = re.compile(r"^\s*(note|notes|note\s*no\.?|ref\.?|refer|"
                                 r"schedule|annexure)\b", re.I)
FOOTNOTE_SUFFIX_RE = re.compile(r"^(?P<body>.*?)(?P<mark>[*†‡§¶]{1,3}|\(\d{1,2}\))$")


@dataclass
class Normalized:
    raw_text: str
    value_type: str = T_TEXT
    number: float = None          # REAL for number/currency/percent
    date_iso: str = None          # TEXT for date
    currency: str = ""
    magnitude: float = 1.0
    percent: bool = False
    negative: bool = False
    footnote_marker: str = ""
    flags: list = field(default_factory=list)

    @property
    def value(self):
        """The single canonical value written to ``cells.normalized_value``."""
        if self.value_type == T_DATE:
            return self.date_iso
        if self.value_type in NUMERIC_TYPES:
            return self.number
        return None


def is_currency_symbol(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    return t in CURRENCY_SYMBOLS or t.lower() in CURRENCY_WORDS


def _nfkc_keep_minus(s: str) -> str:
    """NFKC-normalise but keep the distinction we care about: minus-like dashes."""
    out = unicodedata.normalize("NFKC", s)
    return out.replace(" ", " ").strip()


def classify_token(s: str) -> str:
    """Coarse *format* signature, used by S4 header detection and S9 type conformity.

    Deliberately finer than ``value_type``: the Apple header cell "2024" and the body
    cell "167,045" are both numbers, and only their formats tell them apart.
    """
    t = _nfkc_keep_minus(s)
    if not t:
        return "empty"
    low = t.lower()
    if low in ERROR_VALUES:
        return "error"
    if t in DASH_CHARS:
        return "dash"
    if low in NIL_WORDS:
        return "nil"
    if low in NA_WORDS:
        return "na"
    if is_currency_symbol(t):
        return "currency_symbol"
    if t.endswith("%") or t.endswith("‰"):
        return "percent"
    core = t.strip("()")
    core = core.lstrip("+-−–").rstrip("-−")
    if PLAIN_INT_RE.match(core):
        return "int%d" % min(len(core), 5)
    if WESTERN_GROUP_RE.match(core) or INDIAN_GROUP_RE.match(core):
        return "grouped"
    if re.match(r"^\d+\.\d+$", core):
        return "decimal"
    if re.match(r"^[\d,]+\.\d+$", core):
        return "grouped_decimal"
    if PERIOD_RE.match(t):
        return "period"
    if parse_date(t)[0]:
        return "date"
    return "text"


class DateOrderModel:
    """Document-level day/month order, learned from unambiguous dates only.

    ``31/03/2024`` can only be day-first, so a document containing one is day-first for
    all of its ``03/04/2024``-shaped dates too.  Until such evidence exists, ambiguous
    dates stay unparsed and flagged rather than guessed.  Deterministic: the model is
    built in one pass over the document before cells are normalised, so the answer does
    not depend on which cell happens to be processed first.
    """

    def __init__(self):
        self.day_first = 0
        self.month_first = 0

    def observe(self, text: str) -> None:
        m = re.match(r"^\s*(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{2,4})\s*$",
                     text or "")
        if not m:
            return
        a, b = int(m.group(1)), int(m.group(2))
        if a > 12 and b <= 12:
            self.day_first += 1
        elif b > 12 and a <= 12:
            self.month_first += 1

    @property
    def order(self):
        if self.day_first > self.month_first:
            return "DMY"
        if self.month_first > self.day_first:
            return "MDY"
        return None

    def to_dict(self):
        return {"day_first_evidence": self.day_first,
                "month_first_evidence": self.month_first,
                "resolved_order": self.order}


def parse_date(text: str, order_model: DateOrderModel = None):
    """``(iso_or_None, flags)`` over at least six formats.

    Formats: ``September 28, 2024`` / ``28 September 2024`` / ``28-Sep-2024`` /
    ``2024-09-28`` / ``28/09/2024`` / ``28.09.2024`` / ``Sep 2024`` / ``09-2024``.
    """
    t = _nfkc_keep_minus(text)
    if not t or len(t) > 40:
        return None, []
    flags = []

    # Month name forms.
    m = re.match(r"^([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})$", t)
    if m and m.group(1).lower() in MONTHS:
        return _iso(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2))), flags
    m = re.match(r"^(\d{1,2})(?:st|nd|rd|th)?[\s\-]+([A-Za-z]{3,9})\.?[\s\-,]+(\d{2,4})$", t)
    if m and m.group(2).lower() in MONTHS:
        return _iso(_year(m.group(3)), MONTHS[m.group(2).lower()],
                    int(m.group(1))), flags
    m = re.match(r"^([A-Za-z]{3,9})\.?[\s\-]+(\d{4})$", t)
    if m and m.group(1).lower() in MONTHS:
        return _iso(int(m.group(2)), MONTHS[m.group(1).lower()], 1), ["DAY_ASSUMED_1"]

    # ISO.
    m = re.match(r"^(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})$", t)
    if m:
        return _iso(int(m.group(1)), int(m.group(2)), int(m.group(3))), flags

    # All-numeric with a separator: the ambiguous family.
    m = re.match(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})$", t)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), _year(m.group(3))
        if a > 12 and b <= 12:
            return _iso(y, b, a), flags
        if b > 12 and a <= 12:
            return _iso(y, a, b), ["month_first"]
        order = order_model.order if order_model else None
        if order == "DMY":
            return _iso(y, b, a), ["order_from_document"]
        if order == "MDY":
            return _iso(y, a, b), ["order_from_document"]
        return None, ["AMBIGUOUS_DATE_ORDER"]
    return None, []


def _year(s: str) -> int:
    v = int(s)
    if v >= 100:
        return v
    # Two-digit years: 70-99 -> 1900s, 00-69 -> 2000s.  Documented, not guessed.
    return 1900 + v if v >= 70 else 2000 + v


def _iso(y: int, m: int, d: int):
    if not (1 <= m <= 12 and 1 <= d <= 31 and 1000 <= y <= 3000):
        return None
    if d > (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[m - 1]:
        return None
    return "%04d-%02d-%02d" % (y, m, d)


def _strip_footnote(t: str):
    """Split a trailing footnote marker off a value: ``1,872*`` -> ``1,872`` + ``*``."""
    m = FOOTNOTE_SUFFIX_RE.match(t)
    if not m:
        return t, ""
    body, mark = m.group("body"), m.group("mark")
    if not body or not any(ch.isdigit() for ch in body):
        return t, ""
    return body, mark


def parse_number(t: str):
    """``(value, flags)`` for the numeric core of a cell -- no sign, no currency."""
    flags = []
    s = t.replace(" ", "").replace("'", "").replace(" ", "")
    if not s:
        return None, flags
    has_comma, has_dot = "," in s, "." in s

    if has_comma and has_dot:
        last_comma, last_dot = s.rfind(","), s.rfind(".")
        if last_comma > last_dot:
            # 1.234,56 -- continental.  Ambiguous against 1.234 + noise; refuse.
            return None, ["AMBIGUOUS_DECIMAL_SEPARATOR"]
        int_part, frac = s[:last_dot], s[last_dot + 1:]
        if not re.match(r"^\d+$", frac):
            return None, ["MALFORMED_NUMBER"]
        val, f2 = _parse_grouped(int_part)
        if val is None:
            return None, f2
        return float("%s.%s" % (int(val), frac)), flags + f2
    if has_comma:
        # A single comma with exactly two trailing digits could be a decimal comma.
        if re.match(r"^\d{1,3},\d{2}$", s):
            return None, ["AMBIGUOUS_DECIMAL_SEPARATOR"]
        return _parse_grouped(s)
    if has_dot:
        if not re.match(r"^\d*\.\d+$|^\d+\.$", s):
            return None, ["MALFORMED_NUMBER"]
        try:
            return float(s.rstrip(".")), flags
        except ValueError:
            return None, ["MALFORMED_NUMBER"]
    if PLAIN_INT_RE.match(s):
        return float(s), flags
    return None, []


def _parse_grouped(s: str):
    """Validate the grouping convention before deleting the separators."""
    if PLAIN_INT_RE.match(s):
        return float(s), []
    if WESTERN_GROUP_RE.match(s):
        return float(s.replace(",", "")), []
    if INDIAN_GROUP_RE.match(s):
        return float(s.replace(",", "")), []
    if re.match(r"^[\d,]+$", s):
        # Digits and commas but neither convention: 12,345,67 or 1,2345.
        return float(s.replace(",", "")), ["AMBIGUOUS_GROUPING"]
    return None, ["MALFORMED_NUMBER"]


def normalize(raw: str, *, column_type: str = None, column_header: str = "",
              order_model: DateOrderModel = None, scale: float = 1.0,
              scale_currency: str = "") -> Normalized:
    """Turn one cell's raw text into a typed, flagged value."""
    n = Normalized(raw_text=raw if raw is not None else "")
    t = _nfkc_keep_minus(n.raw_text)
    if not t:
        n.value_type = T_EMPTY
        return n

    low = t.lower()
    if low in ERROR_VALUES:
        n.value_type = T_ERROR
        n.flags.append("SPREADSHEET_ERROR")
        return n
    if t in DASH_CHARS or (len(t) <= 3 and all(c in DASH_CHARS for c in t)):
        n.value_type = T_DASH
        return n
    if low in NIL_WORDS:
        n.value_type = T_NIL
        return n
    if low in NA_WORDS:
        n.value_type = T_NA
        return n
    if REFERENCE_HEADER_RE.match(column_header or "") or column_type == T_REFERENCE:
        n.value_type = T_REFERENCE
        return n

    body, mark = _strip_footnote(t)
    if mark:
        n.footnote_marker = mark
        t = body

    # Currency and percent decoration.
    work = t
    for sym in sorted(CURRENCY_SYMBOLS, key=len, reverse=True):
        if work.startswith(sym):
            n.currency = "INR" if sym == LEGACY_RUPEE else _code(sym)
            if sym == LEGACY_RUPEE:
                n.flags.append("LEGACY_RUPEE_GLYPH")
            work = work[len(sym):].strip()
            break
    m = re.match(r"^(?:(rs\.?|inr|usd|eur|gbp|jpy)\s*)", work, re.I)
    if m:
        n.currency = _code(m.group(1))
        work = work[m.end():].strip()
    if work.endswith("%"):
        n.percent = True
        work = work[:-1].strip()
    elif work.endswith("‰"):
        n.percent = True
        n.magnitude = 0.1
        work = work[:-1].strip()

    # Magnitude suffix.
    m = re.match(r"^(.*?)\s*(k|mn|m|bn|b|tn|cr|crores?|lakhs?|lacs?|million|millions|"
                 r"billion|billions|thousand|thousands|trillion)\.?$", work, re.I)
    if m and m.group(1).strip():
        suffix = m.group(2).lower().rstrip(".")
        if suffix in MAGNITUDES:
            n.magnitude = MAGNITUDES[suffix]
            work = m.group(1).strip()

    # Sign.
    neg = False
    if work.startswith("(") and work.endswith(")"):
        neg = True
        work = work[1:-1].strip()
    if work.startswith(("-", "−", "–", "‒")):
        neg = True
        work = work[1:].strip()
    elif work.endswith(("-", "−")):
        neg = True
        work = work[:-1].strip()
    if work.startswith("+"):
        work = work[1:].strip()

    # Dates and periods take precedence over "looks like digits with separators".
    if not neg and not n.percent:
        iso, dflags = parse_date(t, order_model)
        if iso:
            n.value_type = T_DATE
            n.date_iso = iso
            n.flags.extend(dflags)
            return n
        if "AMBIGUOUS_DATE_ORDER" in dflags:
            n.value_type = T_DATE
            n.flags.append("AMBIGUOUS_DATE_ORDER")
            return n
        if PERIOD_RE.match(t):
            n.value_type = T_PERIOD
            return n

    # Only attempt a numeric parse on something numeric-shaped.  Without this guard
    # every prose cell containing a comma came back MALFORMED_NUMBER -- "Cupertino,
    # California" and "0.875% Notes due 2025" produced 375 bogus mandatory-review
    # flags on the Apple 10-K, which is precisely the kind of false alarm that makes a
    # review queue useless.
    if not NUMERIC_SHAPE_RE.match(work):
        n.value_type = T_TEXT
        return n
    val, nflags = parse_number(work)
    n.flags.extend(nflags)
    if val is None:
        n.value_type = T_TEXT
        return n

    n.negative = neg
    val = -val if neg else val
    val *= n.magnitude
    if n.percent:
        n.value_type = T_PERCENT
        n.number = val
    else:
        if not n.currency and scale_currency:
            n.currency = scale_currency
        if scale and scale != 1.0:
            val *= scale
            n.flags.append("SCALE_APPLIED")
        n.number = val
        n.value_type = T_CURRENCY if n.currency else T_NUMBER
    return n


def _code(sym: str) -> str:
    s = sym.strip().rstrip(".").lower()
    table = {"$": "USD", "us$": "USD", "₹": "INR", "₨": "INR", "rs": "INR",
             "inr": "INR", "usd": "USD", "€": "EUR", "eur": "EUR", "£": "GBP",
             "gbp": "GBP", "¥": "JPY", "jpy": "JPY", "r$": "BRL", "a$": "AUD",
             "c$": "CAD", "chf": "CHF"}
    return table.get(s, sym.strip())


def infer_column_type(values, header_path=None, majority: float = 0.70):
    """Per-column type from its non-null cells, plus the conflicting minority.

    Returns ``(inferred_type, conflict_indices, stats)``.  A single string in a numeric
    column is precisely the red flag the brief asks for, so conflicts are returned
    rather than smoothed away.
    """
    header = " ".join(header_path or [])
    if REFERENCE_HEADER_RE.match(header.strip()):
        return T_REFERENCE, [], {"from": "header_keyword"}
    counts = {}
    considered = []
    for i, n in enumerate(values):
        if n.value_type in NULLISH_TYPES:
            continue
        considered.append(i)
        key = T_NUMBER if n.value_type == T_CURRENCY else n.value_type
        counts[key] = counts.get(key, 0) + 1
    if not considered:
        return T_EMPTY, [], {"from": "all_null"}
    total = len(considered)
    best, best_n = max(sorted(counts.items()), key=lambda kv: kv[1])
    frac = best_n / total
    conflicts = []
    if frac >= majority:
        for i in considered:
            vt = values[i].value_type
            key = T_NUMBER if vt == T_CURRENCY else vt
            if key != best:
                conflicts.append(i)
        inferred = best
    else:
        inferred = T_TEXT
    if best in (T_NUMBER,) and any(values[i].currency for i in considered):
        inferred = T_CURRENCY
    return inferred, conflicts, {"from": "majority", "counts": dict(sorted(
        counts.items())), "majority_frac": round(frac, 3), "n_considered": total}


# --------------------------------------------------------------------------------------------------
# S8 -- context capture: title, section, units and scale, footnotes, linked figures.
# --------------------------------------------------------------------------------------------------
# "4,812 means nothing until you know it is rupees, in lakhs, for Q4 FY25, for the
# Domestic segment, restated per note 14."  Each of those five facts lives somewhere
# different, and the awkward one is scale: it is commonly declared once under a section
# heading and governs every table below it, possibly on later pages.
#
# Scale resolution therefore walks outward from the table and *records where it stopped*:
#
# 1. the table's own caption or title,
# 2. text lines immediately above the region (the ``(All figures in Rs lakhs)`` line),
# 3. the enclosing section heading,
# 4. the nearest scale declaration earlier in the document, up to
#    ``scale_search_pages`` pages back.
#
# Each level is less certain than the last, so ``unit_scale_note`` is stored with the
# page and line it came from and a confidence that decays with distance.  A value whose
# unit came from two pages back is a materially weaker claim than one whose unit is in its
# own caption, and the schema keeps the difference instead of flattening it.
#
# **Footnote-to-cell linkage.**  Markers are collected from the cell text (``1,872*``,
# a superscript ``2``) and matched against footnote lines by marker, searching the
# table's own page first and then the *following* page -- item 10 of the adversarial set
# puts the footnote text on the next page, and a page-local search silently drops it.
# Links are stored in ``cell_footnotes``, cell to footnote, not table to page.
#
# **Table-to-figure association.**  The brief asks us to define and defend the rule, so
# it is an explicit weighted sum of four independent pieces of evidence:
#
# ===================================================  ======
# evidence                                             weight
# ===================================================  ======
# the table's caption or nearby text names the figure    0.45
# figure and table captions share a section number      0.25
# figure is adjacent to the table and inside the same    0.20
# section
# figure lies between the table and the next heading     0.10
# ===================================================  ======
#
# Textual reference dominates because it is the only signal that is *stated* rather than
# inferred; the rest are corroboration.  ``relation_confidence`` is the sum, and the
# component breakdown is stored so a low-confidence link can be inspected rather than
# guessed at.  Proximity alone caps out at 0.30, which is deliberately below any
# threshold a consumer should trust blindly.

FIG_REF_RE = re.compile(
    r"\b(figure|fig\.?|chart|graph|exhibit)\s*([0-9IVX]+(?:[.\-][0-9A-Za-z]+)*)", re.I)
NUMBER_IN_CAPTION_RE = re.compile(r"([0-9]+(?:[.\-][0-9A-Za-z]+)*)")
CURRENCY_IN_NOTE_RE = re.compile(r"(₹|`|\$|€|£|¥|\bRs\.?\b|\bINR\b|\bUSD\b|\bEUR\b|"
                                 r"\bGBP\b|\bJPY\b)", re.I)


@dataclass
class ScaleInfo:
    note: str = ""
    magnitude: float = 1.0
    currency: str = ""
    source: str = ""          # caption | above_region | section_heading | earlier_page
    page_no: int = 0
    bbox: tuple = None
    confidence: float = 0.0
    flags: list = field(default_factory=list)


@dataclass
class Footnote:
    marker: str
    text: str
    page_no: int
    bbox: tuple


@dataclass
class Asset:
    kind: str                 # image | chart
    bbox: tuple
    page_no: int
    caption: str = ""
    file_path: str = ""


@dataclass
class TableContext:
    title: str = ""
    caption: str = ""
    section_path: list = field(default_factory=list)
    scale: ScaleInfo = None
    footnotes: list = field(default_factory=list)          # Footnote
    cell_footnotes: list = field(default_factory=list)     # (row, col, footnote_index)
    assets: list = field(default_factory=list)             # (asset_index, relation,
    #                                                          confidence, components)
    evidence: dict = field(default_factory=dict)


# ------------------------------------------------------------------ title / caption


def find_title_caption(pt, layout, bbox, cfg: Config):
    """Nearest caption above the table, then nearest heading; both if present."""
    caption, cap_gap = "", 1e9
    for _i, txt, bb in getattr(layout, "captions", []):
        gap = bbox[1] - bb[3]
        if -2.0 <= gap < min(cap_gap, cfg.caption_max_gap):
            caption, cap_gap = txt, gap
    # A caption directly below is also common for figures-as-tables.
    for _i, txt, bb in getattr(layout, "captions", []):
        gap = bb[1] - bbox[3]
        if 0 <= gap < min(cap_gap, cfg.caption_max_gap):
            caption, cap_gap = txt, gap

    title, t_gap = "", 1e9
    for _i, _lvl, txt, bb in getattr(layout, "headings", []):
        gap = bbox[1] - bb[3]
        if -2.0 <= gap < min(t_gap, cfg.caption_max_gap * 2):
            title, t_gap = txt, gap
    if not title and caption:
        title = caption
    return title, caption, {"caption_gap": round(cap_gap, 2) if caption else None,
                            "title_gap": round(t_gap, 2) if title else None}


# ---------------------------------------------------------------------- unit / scale


def parse_scale_note(text: str) -> ScaleInfo:
    """Pull a magnitude and a currency out of a declaration line."""
    info = ScaleInfo(note=text.strip())
    m = SCALE_RE.search(text)
    if m:
        word = m.group(1).lower().rstrip(".")
        info.magnitude = MAGNITUDES.get(word, 1.0)
    c = CURRENCY_IN_NOTE_RE.search(text)
    if c:
        sym = c.group(1)
        info.currency = _code(sym if sym != "`" else "₹")
        if sym == "`":
            info.flags.append("LEGACY_RUPEE_GLYPH")
    return info


def resolve_scale(page_no: int, bbox, pts, layouts, section_path, caption: str,
                  cfg: Config, header_texts=None) -> ScaleInfo:
    """Walk outward from the table until a scale declaration is found."""
    # Step 0: the table's own header cells.  The ICRA summary table declares its
    # unit inside the header -- "Previous rated amount (Rs. crore)" -- which is
    # both the most specific place it can appear and the easiest to miss, since
    # it is a cell rather than a line of page text.
    for txt in (header_texts or []):
        if txt and SCALE_RE.search(txt):
            info = parse_scale_note(txt)
            info.source, info.page_no = "table_header", page_no
            info.confidence = 0.92
            return info
    if caption and SCALE_RE.search(caption):
        info = parse_scale_note(caption)
        info.source, info.page_no, info.confidence = "caption", page_no, 0.95
        return info

    lay = layouts[page_no - 1]
    pt = pts[page_no - 1]
    best = None
    for i, txt, bb in lay.scale_notes:
        gap = bbox[1] - bb[3]
        if -2.0 <= gap <= cfg.caption_max_gap * 2:
            if best is None or gap < best[0]:
                best = (gap, txt, bb)
    if best:
        info = parse_scale_note(best[1])
        info.source, info.page_no, info.bbox = "above_region", page_no, best[2]
        info.confidence = 0.88
        return info

    for txt in reversed(section_path or []):
        if SCALE_RE.search(txt):
            info = parse_scale_note(txt)
            info.source, info.page_no, info.confidence = "section_heading", page_no, 0.75
            return info

    # Earlier pages: a scale declared once at the top of a section governs the tables
    # below it, so search backwards and decay confidence with distance.
    for back in range(0, cfg.scale_search_pages + 1):
        p = page_no - back
        if p < 1:
            break
        l2 = layouts[p - 1]
        cands = [(i, txt, bb) for i, txt, bb in l2.scale_notes
                 if p < page_no or bb[3] <= bbox[1]]
        if not cands:
            continue
        i, txt, bb = max(cands, key=lambda c: c[2][3])
        info = parse_scale_note(txt)
        info.source = "earlier_page" if back else "same_page_above"
        info.page_no, info.bbox = p, bb
        info.confidence = max(0.35, 0.80 - 0.15 * back)
        if back:
            info.flags.append("SCALE_FROM_EARLIER_PAGE")
        return info
    return ScaleInfo(source="not_found", page_no=page_no, confidence=0.0)


# ------------------------------------------------------------------------ footnotes


def collect_footnotes(pts, layouts, page_no: int, look_ahead: int = 1):
    """Footnote lines on this page and the next ``look_ahead`` pages."""
    out = []
    for p in range(page_no, min(len(pts), page_no + look_ahead) + 1):
        lay = layouts[p - 1]
        for _i, marker, txt, bb in lay.footnotes:
            body = txt[len(marker):].strip() if txt.startswith(marker) else txt
            out.append(Footnote(marker=marker.strip(), text=body.strip(),
                                page_no=p, bbox=bb))
    return out


def link_footnotes(cells, footnotes):
    """Match cell markers to footnote markers.  Returns ``(links, unmatched)``."""
    by_marker = {}
    for i, fn in enumerate(footnotes):
        by_marker.setdefault(_norm_marker(fn.marker), []).append(i)
    links, unmatched = [], []
    for c in cells:
        marks = list(getattr(c, "footnote_markers", []) or [])
        raw = (c.raw_text or "").strip()
        m = re.search(r"([*†‡§¶]{1,3})$", raw)
        if m:
            marks.append(m.group(1))
        for mk in sorted(set(marks)):
            key = _norm_marker(mk)
            if key in by_marker:
                for fi in by_marker[key]:
                    links.append((c.row, c.col, fi))
            else:
                unmatched.append((c.row, c.col, mk))
    return links, unmatched


def _norm_marker(m: str) -> str:
    return (m or "").strip().strip("().").lower()


# --------------------------------------------------------------------------- assets


def find_assets(page, pt, layout, table_bboxes, cfg: Config):
    """Images and vector chart clusters on a page that are not tables."""
    out = []
    for info in page.get_text("dict").get("blocks", []):
        if info.get("type", 0) != 1:
            continue
        bb = tuple(float(v) for v in info.get("bbox", (0, 0, 0, 0)))
        if area(bb) < cfg.asset_min_area:
            continue
        if any(_overlaps(bb, tb, 0.6) for tb in table_bboxes):
            continue
        out.append(Asset(kind="image", bbox=bb, page_no=pt.page_no))

    clusters = _vector_clusters(page, pt, table_bboxes, cfg)
    out.extend(clusters)
    for a in out:
        a.caption = _caption_for(layout, a.bbox, cfg)
    out.sort(key=lambda a: (round(a.bbox[1], 2), round(a.bbox[0], 2), a.kind))
    return out


def _overlaps(a, b, frac: float) -> bool:

    return contain_frac(a, b) >= frac


def _vector_clusters(page, pt, table_bboxes, cfg: Config):
    """Dense vector-drawing clusters that are not table regions: bar/line charts."""

    boxes = []
    try:
        for path in page.get_drawings():
            for it in path.get("items", []):
                if it[0] == "re":
                    r = it[1]
                    boxes.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1)))
                elif it[0] == "l":
                    p0, p1 = it[1], it[2]
                    boxes.append((min(p0.x, p1.x), min(p0.y, p1.y),
                                  max(p0.x, p1.x), max(p0.y, p1.y)))
    except Exception:
        return []
    boxes = [b for b in boxes
             if not any(_overlaps(b, tb, 0.9) for tb in table_bboxes)]
    if len(boxes) < 12:
        return []
    ys = [(b[1] + b[3]) / 2.0 for b in boxes]
    out = []
    for cl in cluster_1d(ys, 24.0):
        if len(cl) < 12:
            continue
        bb = union(boxes[i] for i in cl)
        if bb is None or area(bb) < cfg.asset_min_area:
            continue
        if bb[2] - bb[0] < 60 or bb[3] - bb[1] < 40:
            continue
        if any(_overlaps(bb, tb, 0.5) for tb in table_bboxes):
            continue
        out.append(Asset(kind="chart", bbox=bb, page_no=pt.page_no))
    return out


def _caption_for(layout, bbox, cfg: Config) -> str:
    best, gap_best = "", 1e9
    for _i, txt, bb in getattr(layout, "captions", []):
        for gap in (bb[1] - bbox[3], bbox[1] - bb[3]):
            if 0 <= gap < min(gap_best, cfg.caption_max_gap):
                best, gap_best = txt, gap
    return best


def _section_number(text: str):
    m = CAPTION_RE.match(text or "")
    if m and m.group(2):
        return m.group(2).split(".")[0]
    m2 = NUMBER_IN_CAPTION_RE.search(text or "")
    return m2.group(1).split(".")[0] if m2 else None


def associate_assets(table_caption: str, table_title: str, table_bbox, page_no: int,
                     section_path, assets, next_heading_y, cfg: Config,
                     nearby_text: str = ""):
    """Score each asset against one table.  Returns a list of link dicts."""
    links = []
    haystack = " ".join([table_caption or "", table_title or "", nearby_text or ""])
    refs = {m.group(2).lower() for m in FIG_REF_RE.finditer(haystack) if m.group(2)}
    t_sec = _section_number(table_caption or table_title or "")

    for idx, a in enumerate(assets):
        comp = {"textual_reference": 0.0, "shared_numbering": 0.0,
                "proximity_same_section": 0.0, "before_next_heading": 0.0}
        a_num = _section_number(a.caption) if a.caption else None
        a_ref = None
        if a.caption:
            m = FIG_REF_RE.match(a.caption.strip())
            if m and m.group(2):
                a_ref = m.group(2).lower()

        if a_ref and a_ref in refs:
            comp["textual_reference"] = 0.45
        if t_sec and a_num and t_sec == a_num:
            comp["shared_numbering"] = 0.25

        if a.page_no == page_no:
            gap = min(abs(a.bbox[1] - table_bbox[3]), abs(table_bbox[1] - a.bbox[3]))
            if gap <= cfg.asset_link_max_gap:
                comp["proximity_same_section"] = 0.20 * max(
                    0.25, 1.0 - gap / cfg.asset_link_max_gap)
            if next_heading_y is not None and table_bbox[3] <= a.bbox[1] <= next_heading_y:
                comp["before_next_heading"] = 0.10

        conf = min(0.95, sum(comp.values()))
        if conf <= 0.0:
            continue
        relation = "visualises" if a.kind == "chart" else "illustrates"
        if comp["textual_reference"] == 0.0 and comp["shared_numbering"] == 0.0:
            relation = "adjacent_to"
        links.append({"asset_index": idx, "relation": relation,
                      "relation_confidence": round(conf, 3), "components": comp})
    links.sort(key=lambda l: (-l["relation_confidence"], l["asset_index"]))
    return links
