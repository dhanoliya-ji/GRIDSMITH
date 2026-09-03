"""S7 -- normalisation and typing.  Additive, never destructive.

``raw_text`` is always kept exactly as the page had it.  Everything here is a *second*
opinion recorded alongside it, and every opinion carries flags saying how sure it is.

The governing rule is section 4 of the brief: no wrong value at high confidence.  That
turns several ordinary parsing decisions into refusals:

``12,34,567`` vs ``12,345,67``
    Indian and Western grouping are both accepted, but the grouping *pattern* is
    validated rather than the commas simply deleted.  A string whose grouping matches
    neither convention parses to a number and is flagged ``AMBIGUOUS_GROUPING``, so a
    typo in the source cannot arrive as a clean value.

``1.234,56``
    Continental European notation collides head-on with Western: is ``1.234`` one
    thousand two hundred or one point two three four?  There is no way to know from one
    cell.  We flag ``AMBIGUOUS_DECIMAL_SEPARATOR`` and leave ``normalized_value`` null.
    Guessing here would be exactly the silent error the brief is built around.

``03/04/2024``
    Same problem for dates: 3 April or 4 March.  If both fields are 12 or under and the
    document has not established an order, ``normalized_value`` stays null with
    ``AMBIGUOUS_DATE_ORDER``.  A date with an unambiguous field (``31/03/2024``) parses
    normally and also *teaches* the document its order, which later resolves the
    ambiguous ones -- see :func:`DateOrderModel`.

``#REF!``
    Excel error values survive printing to PDF.  ``Shell.pdf`` ships six of them inside
    the cash-flow statement.  They are typed ``error``, never coerced to 0 or null-and-
    forgotten, and they raise an issue -- a number that the source document itself could
    not compute must not look like a number downstream.

``-`` / ``Nil`` / ``NA`` / ``0`` / empty
    Five distinct value types, because they mean five different things to an accountant
    and collapsing them loses information no downstream user can recover.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

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
