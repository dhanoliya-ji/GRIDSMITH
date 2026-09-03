"""Unit tests for the pieces where a regression would be silent.

Chosen on that basis rather than for coverage: normalisation, the reconciliation
tolerance and the confidence cap are the places where a bug produces a *plausible wrong
number* instead of a crash, which is the failure mode the whole project is built to
avoid.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gridsmith import geometry as G                                  # noqa: E402
from gridsmith.config import DEFAULT                                 # noqa: E402
from gridsmith.determinism import canonical_json, q, sid             # noqa: E402
from gridsmith.s3_negatives import _numeric                          # noqa: E402
from gridsmith.s4_structure import is_total_label, splits_word_gap   # noqa: E402
from gridsmith.s7_normalize import (DateOrderModel, classify_token,  # noqa: E402
                                   infer_column_type, normalize,
                                   parse_date, parse_number)
from gridsmith.s9_confidence import (MANDATORY_REVIEW_CODES,         # noqa: E402
                                    Calibration, is_reconcilable_total,
                                    rounding_tolerance, sigmoid)
from gridsmith.textlayer import Token, build_lines, join_tokens, norm_for_repeat  # noqa: E402


# --------------------------------------------------------------------- determinism


def test_sid_is_stable_and_content_addressed():
    assert sid("cell", "t1", 3, 4) == sid("cell", "t1", 3, 4)
    assert sid("cell", "t1", 3, 4) != sid("cell", "t1", 4, 3)
    assert sid("cell", "t1", 3, 4).startswith("cel_")


def test_q_normalises_negative_zero():
    # -0.0 and 0.0 must serialise identically or two runs differ in the bytes.
    assert q(-0.0) == 0.0
    assert repr(q(-0.0)) == repr(0.0)
    assert q(1.234567891) == 1.2346


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": 2}, None) == canonical_json({"a": 2, "b": 1},
                                                                    None)
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")}, None)


# ---------------------------------------------------------------------- geometry


def test_cluster_1d_is_deterministic_and_ordered():
    vals = [9.2, 1.0, 20.0, 1.5, 9.0]
    got = G.cluster_1d(vals, 1.0)
    assert got == [[1, 3], [4, 0], [2]]


def test_gaps_in_returns_the_complement():
    assert G.gaps_in([(0, 10), (14, 20)], 0, 30) == [(10, 14), (20, 30)]


# --------------------------------------------------------------- normalisation S7


@pytest.mark.parametrize("raw,vtype,value", [
    ("1,23,45,678", "number", 12345678.0),      # Indian grouping
    ("12,345,678", "number", 12345678.0),       # Western grouping
    ("(1,234)", "number", -1234.0),             # parenthesised negative
    ("-1,234", "number", -1234.0),
    ("1,234-", "number", -1234.0),              # trailing minus
    ("3 %", "percent", 3.0),
    ("(4)%", "percent", -4.0),
    ("$ 167,045", "currency", 167045.0),
    ("12.5 Cr", "number", 125000000.0),         # magnitude suffix
    ("150 lakh", "number", 15000000.0),
    ("September 28, 2024", "date", "2024-09-28"),
    ("31 March 2023", "date", "2023-03-31"),
    ("2023-03-31", "date", "2023-03-31"),
    ("28-Sep-2024", "date", "2024-09-28"),
    ("31.03.2023", "date", "2023-03-31"),
    ("31/03/2024", "date", "2024-03-31"),
    ("FY 2024-25", "period", None),
    ("#REF!", "error", None),
    ("-", "dash", None),
    ("Nil", "nil", None),
    ("NA", "na", None),
    ("0", "number", 0.0),
    ("", "empty", None),
])
def test_normalise_covers_every_required_format(raw, vtype, value):
    n = normalize(raw)
    assert n.value_type == vtype, raw
    assert n.value == value, raw
    assert n.raw_text == raw          # normalisation is additive, never destructive


@pytest.mark.parametrize("raw", ["1.234,56", "1,23"])
def test_ambiguous_decimal_separator_refuses_to_guess(raw):
    n = normalize(raw)
    assert n.value is None
    assert "AMBIGUOUS_DECIMAL_SEPARATOR" in n.flags
    assert n.flags[0] in MANDATORY_REVIEW_CODES


def test_ambiguous_date_order_refuses_until_the_document_decides():
    n = normalize("03/04/2024")
    assert n.value is None and "AMBIGUOUS_DATE_ORDER" in n.flags

    om = DateOrderModel()
    om.observe("31/03/2024")          # only day-first can explain a 31
    assert om.order == "DMY"
    iso, flags = parse_date("03/04/2024", om)
    assert iso == "2024-04-03" and "order_from_document" in flags


def test_malformed_grouping_yields_a_value_but_flags_it():
    n = normalize("12,345,67")
    assert n.value == 1234567.0
    assert "AMBIGUOUS_GROUPING" in n.flags


def test_prose_with_a_comma_is_not_a_malformed_number():
    for txt in ("Cupertino, California", "0.875% Notes due 2025",
                "June 30, 2024 to August 3, 2024:"):
        n = normalize(txt)
        assert n.value_type == "text" and not n.flags, txt


def test_footnote_marker_is_split_off_but_kept_in_raw_text():
    n = normalize("1,872*")
    assert n.value == 1872.0
    assert n.footnote_marker == "*"
    assert n.raw_text == "1,872*"


def test_legacy_rupee_glyph_is_flagged_not_assumed():
    n = normalize("` 165.0")
    assert n.currency == "INR"
    assert "LEGACY_RUPEE_GLYPH" in n.flags


def test_em_dash_percent_is_not_zero():
    # "-- %" means "no change"; coercing it to 0.0 invents a number.
    n = normalize("— %")
    assert n.value is None


def test_classify_token_separates_bare_year_from_grouped_thousands():
    assert classify_token("2024") == "int4"
    assert classify_token("167,045") == "grouped"


def test_column_type_reports_conflicts_rather_than_smoothing_them():
    vals = [normalize(x) for x in ("1.0", "2.0", "3.0", "oops", "5.0")]
    ty, conflicts, stats = infer_column_type(vals, [])
    assert ty == "number"
    assert conflicts == [3]
    assert stats["majority_frac"] == pytest.approx(0.8)


def test_note_column_is_typed_reference_not_number():
    n = normalize("5(a)", column_header="Note")
    assert n.value_type == "reference" and n.value is None


def test_parse_number_rejects_letters():
    assert parse_number("abc") == (None, [])


# ------------------------------------------------------------------ reconciliation


def test_rounding_tolerance_scales_with_printed_precision():
    ints = [normalize(x) for x in ("1184", "2421", "3505")]
    # 1,184 + 2,421 = 3,605 vs a stated 3,505: a 100 error, far outside rounding.
    assert abs(3605 - 3505) > rounding_tolerance(ints)
    # Three values printed as 33.3 summing to 99.9 against a stated 100.0 is innocent.
    rounded = [normalize("33.3")] * 3 + [normalize("100.0")]
    assert abs(99.9 - 100.0) < rounding_tolerance(rounded)


def test_ratio_labels_are_not_treated_as_totals():
    assert is_total_label("Total outside liabilities/Tangible net worth (times)")
    assert not is_reconcilable_total(
        "Total outside liabilities/Tangible net worth (times)")
    assert is_reconcilable_total("Total net sales")


# --------------------------------------------------------------------- confidence


def test_calibration_is_monotone_and_bounded():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src",
                        "gridsmith", "calibration.json")
    calib = Calibration.load(path)
    xs = [i / 50.0 for i in range(51)]
    ys = [calib.apply(x) for x in xs]
    assert all(b >= a - 1e-9 for a, b in zip(ys, ys[1:])), "calibration must be monotone"
    assert all(0.0 <= y <= 1.0 for y in ys)
    if calib.edges:
        assert max(ys) < 1.0, "never claim certainty"


def test_fallback_calibration_is_pessimistic():
    calib = Calibration()          # no file
    assert calib.apply(1.0) < 1.0
    assert calib.apply(0.5) == pytest.approx(0.5)


def test_sigmoid_is_stable_at_the_extremes():
    assert sigmoid(-1000) == pytest.approx(0.0)
    assert sigmoid(1000) == pytest.approx(1.0)


def test_mandatory_codes_cover_every_refusal_we_can_emit():
    for code in ("OCR_UNAVAILABLE", "SPREADSHEET_ERROR",
                 "AMBIGUOUS_DECIMAL_SEPARATOR", "AMBIGUOUS_DATE_ORDER",
                 "DOWNFOOT_FAIL", "CROSSFOOT_FAIL", "TOKEN_STRADDLES_COLUMN"):
        assert code in MANDATORY_REVIEW_CODES


# ----------------------------------------------------------------------- text layer


def _tok(text, x0, x1, y=100.0, size=9.0):
    return Token(text=text, x0=x0, y0=y - size, x1=x1, y1=y, size=size, font="helv",
                 flags=0, baseline=y, block=0, line_no=0)


def test_build_lines_clusters_across_blocks():
    toks = [_tok("A", 0, 10, y=100), _tok("B", 50, 60, y=100.3),
            _tok("C", 0, 10, y=120)]
    lines = build_lines(toks)
    assert [len(l.tokens) for l in lines] == [2, 1]


def test_join_tokens_inserts_spaces_only_at_real_gaps():
    # 0.5 pt is kerning; 20 pt is a space.
    toks = [_tok("Fund", 0, 20), _tok("ed", 20.4, 30), _tok("based", 50, 70)]
    assert join_tokens(toks) == "Funded based"


def test_norm_for_repeat_collapses_page_numbers():
    assert norm_for_repeat("Page | 7") == norm_for_repeat("Page | 12")


def test_splits_word_gap_detects_an_intra_cell_space():
    class L:
        tokens = [_tok("non-fund", 152.4, 213.0), _tok("based", 215.4, 237.4)]
    assert splits_word_gap(214.0, [L()])          # a 2.4 pt word space
    assert not splits_word_gap(300.0, [L()])      # outside the text entirely


def test_numeric_helper_tolerates_financial_decoration():
    assert _numeric("(1,234)") and _numeric("45.6%") and _numeric("-7")
    assert not _numeric("U-001") and not _numeric("Total")


def test_config_fingerprint_changes_with_any_threshold():
    a = DEFAULT.fingerprint()
    b = DEFAULT.replace(gutter_persist_frac=0.81).fingerprint()
    assert a != b and len(a) == 16
