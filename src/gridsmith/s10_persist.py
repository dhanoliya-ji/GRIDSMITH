"""S10 -- persistence: SQLite, one .xlsx per logical table, manifest.

Two things here are less obvious than they look.

**Byte-identical Excel.**  ``openpyxl`` stamps ``docProps/core.xml`` with
``datetime.now()`` and writes every zip entry with the current local time, so two runs
over the same input differ in dozens of bytes even when every cell matches.  We
therefore fix the workbook properties *and* rewrite the archive with a constant
timestamp, ``STORED``-order entries and no extra fields -- see
:func:`write_deterministic_xlsx`.  Without that step §3.2 simply cannot be satisfied,
and it is not the kind of thing a two-run diff lets you overlook.

**Byte-identical SQLite.**  A fresh database written by an identical sequence of
statements is already reproducible, provided nothing in the data varies.  The only
varying values are the three timestamps in ``extraction_runs`` -- which the brief
explicitly permits -- and ``--fixed-time`` pins even those, so a reviewer running
``gridsmith run`` twice and diffing ``out/`` byte-for-byte gets a clean result rather
than a one-row explanation.  Ids are content digests, not counters, for the same reason.

Native types in the Data sheet: a value is written as a native Excel number or date only
when its cell is *not* flagged for review.  Anything uncertain is written as text, so a
spreadsheet total can never quietly incorporate a value the pipeline does not trust.
"""
from __future__ import annotations

import io
import os
import shutil
import sqlite3
import zipfile

from .determinism import FIXED_ZIP_TIME, canonical_json, sid

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


# ------------------------------------------------------------------------ SQLite


def open_db(path: str) -> sqlite3.Connection:
    if os.path.exists(path):
        os.remove(path)
    for side in (path + "-wal", path + "-shm", path + "-journal"):
        if os.path.exists(side):
            os.remove(side)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(path)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
        conn.executescript(fh.read())
    return conn


def insert(conn, table: str, rows):
    """Insert dict rows in a fixed column order so the write sequence is stable."""
    rows = list(rows)
    if not rows:
        return
    cols = sorted(rows[0].keys())
    q = "INSERT OR REPLACE INTO %s (%s) VALUES (%s)" % (
        table, ",".join(cols), ",".join("?" * len(cols)))
    conn.executemany(q, [tuple(r[c] for c in cols) for r in rows])


# ------------------------------------------------------------------------- Excel


def write_deterministic_xlsx(workbook, path: str) -> None:
    """Save an openpyxl workbook so two runs produce identical bytes.

    ``openpyxl`` writes ``docProps/core.xml`` with the current time and each zip entry
    with the local clock.  We pin the document properties, save into memory, then
    rebuild the archive with :data:`gridsmith.determinism.FIXED_ZIP_TIME` on every entry
    and the entry order preserved.
    """
    from openpyxl.utils.datetime import from_ISO8601

    props = workbook.properties
    fixed = from_ISO8601("1980-01-01T00:00:00Z")
    props.created = fixed
    props.modified = fixed
    props.lastModifiedBy = "gridsmith"
    props.creator = "gridsmith"

    buf = io.BytesIO()
    workbook.save(buf)
    buf.seek(0)

    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(buf, "r") as src, \
            zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            data = src.read(name)
            if name == "docProps/core.xml":
                data = _pin_core_props(data)
            info = zipfile.ZipInfo(filename=name, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 0
            dst.writestr(info, data)


FIXED_XML_TIME = "1980-01-01T00:00:00Z"


def _pin_core_props(data: bytes) -> bytes:
    """Overwrite the timestamps inside ``docProps/core.xml``.

    Setting ``workbook.properties.modified`` before saving is not enough: openpyxl
    rewrites ``dcterms:modified`` with the current clock during ``save()``, so two runs
    a second apart differ by exactly those bytes -- which the two-run diff duly found.
    Rewriting the entry as we rebuild the archive is version-proof, because it does not
    depend on which openpyxl internal happens to own the value.
    """
    import re

    text = data.decode("utf-8", "replace")
    for tag in ("dcterms:created", "dcterms:modified"):
        text = re.sub(r"(<%s[^>]*>)[^<]*(</%s>)" % (tag, tag),
                      r"\g<1>%s\g<2>" % FIXED_XML_TIME, text)
    return text.encode("utf-8")


def table_slug(title: str, fallback: str = "table") -> str:
    import re

    s = (title or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    s = re.sub(r"_+", "_", s)
    return (s[:48] or fallback)


def excel_filename(logical_index: int, start: int, end: int, slug: str) -> str:
    return "T%03d__p%d-%d__%s.xlsx" % (logical_index, start, end, slug)


def write_table_xlsx(path: str, *, data_rows, merges, n_header_rows, context_rows,
                     provenance_rows, issue_rows, freeze_header=True):
    """Write one logical table as the four-sheet workbook the contract requires."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    head_font = Font(bold=True)
    head_fill = PatternFill("solid", fgColor="EEEEEE")
    wrap = Alignment(vertical="top", wrap_text=True)

    for r, row in enumerate(data_rows, start=1):
        for c, value in enumerate(row, start=1):
            cell = ws.cell(row=r, column=c, value=value)
            cell.alignment = wrap
            if r <= n_header_rows:
                cell.font = head_font
                cell.fill = head_fill
    for (r0, c0, r1, c1) in merges:
        if r1 > r0 or c1 > c0:
            ws.merge_cells(start_row=r0 + 1, start_column=c0 + 1,
                           end_row=r1 + 1, end_column=c1 + 1)
    if data_rows:
        widths = {}
        for row in data_rows:
            for c, v in enumerate(row, start=1):
                widths[c] = min(52, max(widths.get(c, 9),
                                        len(str(v)) + 2 if v is not None else 9))
        for c, w in sorted(widths.items()):
            ws.column_dimensions[get_column_letter(c)].width = w
    if freeze_header and n_header_rows > 0:
        # A coordinate string, not a cell object: when A1 falls inside a merged range
        # openpyxl hands back a MergedCell, which freeze_panes cannot consume.
        ws.freeze_panes = "A%d" % (n_header_rows + 1)

    ctx = wb.create_sheet("Context")
    ctx.append(["field", "value"])
    ctx.cell(row=1, column=1).font = head_font
    ctx.cell(row=1, column=2).font = head_font
    for k, v in context_rows:
        ctx.append([k, v])
    ctx.column_dimensions["A"].width = 26
    ctx.column_dimensions["B"].width = 96

    prov = wb.create_sheet("Provenance")
    prov_cols = ["row_idx", "col_idx", "raw_text", "normalized_value", "page_no",
                 "bbox", "source", "confidence", "reason_codes"]
    prov.append(prov_cols)
    for i in range(len(prov_cols)):
        prov.cell(row=1, column=i + 1).font = head_font
    for row in provenance_rows:
        prov.append([row.get(c) for c in prov_cols])
    prov.freeze_panes = "A2"
    for col, w in zip("ABCDEFGHI", (9, 9, 40, 18, 9, 30, 12, 12, 40)):
        prov.column_dimensions[col].width = w

    iss = wb.create_sheet("Issues")
    iss_cols = ["severity", "code", "row_idx", "col_idx", "message"]
    iss.append(iss_cols)
    for i in range(len(iss_cols)):
        iss.cell(row=1, column=i + 1).font = head_font
    for row in issue_rows:
        iss.append([row.get(c) for c in iss_cols])
    iss.freeze_panes = "A2"
    for col, w in zip("ABCDE", (10, 30, 9, 9, 110)):
        iss.column_dimensions[col].width = w

    write_deterministic_xlsx(wb, path)


# ---------------------------------------------------------------------- manifest


def write_manifest(path: str, payload: dict) -> None:
    from .determinism import write_text

    write_text(path, canonical_json(payload) + "\n")


def reset_dir(path: str, keep=()):
    """Recreate an output subdirectory so stale files never survive a rerun."""
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name in keep:
                continue
            p = os.path.join(path, name)
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
    else:
        os.makedirs(path, exist_ok=True)


def region_id(table_id: str, page_no: int, part: int) -> str:
    return sid("reg", table_id, page_no, part)
