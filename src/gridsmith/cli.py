"""Command-line interface.

The three commands the brief requires, plus two that earned their place while building:

    python -m gridsmith run    --input doc.pdf --out ./out --offline
    python -m gridsmith audit  --out ./out
    python -m gridsmith debug  --out ./out --page 42

    python -m gridsmith verify --a ./out1 --b ./out2      # two-run byte diff
    python -m gridsmith query  --out ./out --sql "..."    # or a named example

``--offline`` is accepted and asserted rather than merely tolerated: with it set, the
process installs a socket guard that raises on any attempt to open a network
connection.  The pipeline has no network code, so the guard should never fire -- which
is exactly why it is worth installing, because it converts "we believe it is offline"
into "it cannot have been otherwise".
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .config import DEFAULT, Config


def _force_utf8_stdout():
    """Windows consoles default to cp1252 and would crash on a rupee sign."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def install_offline_guard():
    """Make network access impossible for the rest of the process."""
    import socket

    class _Blocked(RuntimeError):
        pass

    def _deny(*a, **k):
        raise _Blocked("network access attempted while --offline was set")

    socket.socket = _deny            # type: ignore[assignment]
    socket.create_connection = _deny  # type: ignore[assignment]
    if hasattr(socket, "socketpair"):
        socket.socketpair = _deny     # type: ignore[assignment]
    for name in ("getaddrinfo", "gethostbyname"):
        if hasattr(socket, name):
            setattr(socket, name, _deny)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m gridsmith",
        description="Deterministic table and context extraction from large PDFs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="extract one PDF into an output directory")
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--offline", action="store_true",
                   help="assert no network access (installs a socket guard)")
    r.add_argument("--fixed-time", action="store_true",
                   help="pin extraction_runs timestamps so even the .db is "
                        "byte-identical between runs")
    r.add_argument("--max-pages", type=int, default=0)
    r.add_argument("--debug-pages", choices=("none", "tables", "all"),
                   default=DEFAULT.write_debug_pages)
    r.add_argument("--no-borderless", action="store_true",
                   help="ablation: disable the borderless detector")
    r.add_argument("--no-ruled", action="store_true",
                   help="ablation: disable the ruled detector")
    r.add_argument("--no-stitch", action="store_true",
                   help="ablation: disable cross-page stitching")
    r.add_argument("--review-budget", type=float, default=DEFAULT.review_budget_frac)
    r.add_argument("--seed", type=int, default=DEFAULT.seed)
    r.add_argument("--quiet", action="store_true")

    a = sub.add_parser("audit", help="print the metrics summary for an output dir")
    a.add_argument("--out", required=True)
    a.add_argument("--json", action="store_true", help="emit raw metrics.json")

    d = sub.add_parser("debug", help="write an overlay render for one page")
    d.add_argument("--out", required=True)
    d.add_argument("--page", type=int, required=True)
    d.add_argument("--input", default=None,
                   help="source PDF (defaults to the one recorded in the database)")
    d.add_argument("--dpi", type=int, default=130)

    v = sub.add_parser("verify", help="diff two output directories (determinism check)")
    v.add_argument("--a", required=True)
    v.add_argument("--b", required=True)

    qy = sub.add_parser("query", help="run SQL against extraction.db")
    qy.add_argument("--out", required=True)
    qy.add_argument("--sql", default=None)
    qy.add_argument("--example", default=None,
                    help="named example query; use --list-examples to see them")
    qy.add_argument("--list-examples", action="store_true")
    qy.add_argument("--limit", type=int, default=20)
    return p


def cfg_from_args(args) -> Config:
    return DEFAULT.replace(
        offline=bool(getattr(args, "offline", False)),
        fixed_time=bool(getattr(args, "fixed_time", False)),
        max_pages=int(getattr(args, "max_pages", 0) or 0),
        write_debug_pages=getattr(args, "debug_pages", DEFAULT.write_debug_pages),
        enable_borderless=not getattr(args, "no_borderless", False),
        enable_ruled=not getattr(args, "no_ruled", False),
        enable_stitch=not getattr(args, "no_stitch", False),
        review_budget_frac=float(getattr(args, "review_budget",
                                         DEFAULT.review_budget_frac)),
        seed=int(getattr(args, "seed", DEFAULT.seed)))


def main(argv=None) -> int:
    _force_utf8_stdout()
    args = build_parser().parse_args(argv)

    if args.cmd == "run":
        if args.offline:
            install_offline_guard()
        from .pipeline import run

        cfg = cfg_from_args(args)
        summary = run(args.input, args.out, cfg)
        if not args.quiet:
            _print_summary(summary, args.out)
        return 0

    if args.cmd == "audit":
        from .s11_metrics import audit

        return audit(args.out, as_json=args.json)

    if args.cmd == "debug":
        from .s11_metrics import debug_page

        return debug_page(args.out, args.page, args.input, args.dpi)

    if args.cmd == "verify":
        from .s11_metrics import verify_dirs

        return verify_dirs(args.a, args.b)

    if args.cmd == "query":
        from .s11_metrics import run_query

        return run_query(args.out, args.sql, args.example, args.limit,
                         args.list_examples)
    return 2


def _print_summary(s: dict, out_dir: str) -> None:
    c = s["counts"]
    r = s["resources"]
    q = s["review_queue"]
    conf = s.get("confidence", {})
    print("gridsmith %s  run %s" % (s["run"]["gridsmith_version"], s["run"]["run_id"]))
    print("  document      : %s  (%d pages, %s)"
          % (s["document"]["filename"], s["document"]["page_count"],
             ", ".join("%s=%d" % kv for kv in
                       sorted(s["document"]["page_types"].items()))))
    print("  tables        : %d logical (%d page-level, %d spanning pages)"
          % (c["logical_tables"], c["page_level_tables"], c["multi_page_tables"]))
    print("  cells         : %d   assets: %d   footnote links: %d"
          % (c["cells"], c["assets"], c["cell_footnote_links"]))
    print("  refused       : %d regions  %s"
          % (c["rejected_regions"],
             ", ".join("%s=%d" % kv for kv in
                       sorted(s["rejections_by_code"].items())) or "-"))
    print("  reconciliation: %s" % (", ".join(
        "%s=%s" % kv for kv in sorted(s["reconciliation"].items()))))
    print("  confidence    : mean %.3f  median %.3f  p05 %.3f  high-conf %.1f%%"
          % (conf.get("mean", 0), conf.get("median", 0), conf.get("p05", 0),
             100 * conf.get("frac_high_confidence", 0)))
    print("  review queue  : %d cells (%.2f%% of %d)  mandatory %d%s"
          % (q["n_queued"], 100 * q["queued_frac"], q["n_cells_total"],
             q["n_mandatory"],
             "  [mandatory exceeds budget]" if q["budget_exceeded_by_mandatory"]
             else ""))
    print("  resources     : %.1fs wall (%.2fs/page), peak RSS %.0f MB"
          % (r["wall_seconds"], r["seconds_per_page"], r["peak_rss_mb"]))
    print("  written to    : %s" % os.path.abspath(out_dir))
