"""Markdown reports for parquet datasets.

`markdown(t)` renders a one-page report — header, schema (arrow/polars/duckdb
types side-by-side), per-column summary stats, hive-partition breakdown, and
a contiguous-window sample — drop-in for code review, dataset onboarding, or
docs alongside the data. Built on duckdb for fast metadata reads and the
existing summarize() helper.

Operates on duckdb. Requires the `duckdb` extra.
"""

from __future__ import annotations

from datetime import datetime
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl

from arrow_dx.profile import _align_decimals, summarize

if TYPE_CHECKING:
    import duckdb

TableStyle = Literal["markdown", "box"]

_NUMERIC_TYPE_PREFIXES = (
    "BIGINT",
    "INTEGER",
    "SMALLINT",
    "TINYINT",
    "HUGEINT",
    "UBIGINT",
    "UINTEGER",
    "USMALLINT",
    "UTINYINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
    "REAL",
    "DECIMAL",
    "NUMERIC",
)
_FLOAT_TYPE_PREFIXES = ("FLOAT", "DOUBLE", "REAL")
_SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"
_HISTOGRAM_BAR = "█"
_BIN_COUNT = 10


def markdown(
    t: str | Path,
    *,
    output: str | Path | None = None,
    sample_n: int = 10,
    sample_seed: int = 1,
    style: TableStyle = "markdown",
    per_partition: bool = False,
    sparklines: bool = True,
    histograms: bool = False,
    con: duckdb.DuckDBPyConnection | None = None,
) -> str:
    """Render a markdown report for a parquet dataset (or any duckdb-readable thing).

    Sections: header (path, generated-at, total rows, size, partition count),
    schema (arrow / polars / duckdb types side-by-side, plus null %), summary
    (duckdb's SUMMARIZE for per-column min/max/avg/std/quartiles, optionally
    with a sparkline per numeric column), partitions (skipped if no hive
    layout detected), optional per-partition summary, optional histograms
    section with full bar charts, and a contiguous-window sample.

    Args:
        t: catalog table/view name, parquet file path, or parquet glob.
        output: optional path; if given, the markdown is also written there.
        sample_n: number of sample rows to include (default 10).
        sample_seed: seed for the deterministic sample offset (default 1, so
            re-running on the same data produces a stable report).
        style: 'markdown' (default) renders pipe-and-dash tables that markdown
            viewers turn into styled tables. 'box' renders Unicode box-drawing
            tables (┌─┬─┐ etc.) wrapped in code fences — useful when the
            report is read in monospace contexts (terminals, plain-text files)
            and column alignment matters. Sparklines and histograms always
            use Unicode block characters; both styles render in monospace
            once the table format is settled.
        per_partition: if True and hive partitions are detected, adds a
            Per-partition summary section faceted by the leftmost hive column.
            Runs one SUMMARIZE per distinct partition value, so this is fine
            for low-cardinality partitions (years, regions) and slow for
            high-cardinality ones. Silent no-op when there's no hive layout.
        sparklines: if True (default), the Summary table gets a `sparkline`
            column showing the per-column distribution for numeric columns
            (10 buckets, Unicode block characters ▁▂▃▄▅▆▇█). Empty cell for
            non-numeric columns. Set False to omit the column entirely.
        histograms: if True, adds a Histograms section with one full bar
            chart per numeric column (10 buckets, edges, counts). Default
            False — useful for deep dives, omitted to keep reports compact.
        con: optional duckdb connection; defaults to a fresh in-memory one.

    Returns:
        The markdown string. Caller can print it, write it, or further process.

    Operates on duckdb. Requires the `duckdb` extra.
    """
    import duckdb

    con = con or duckdb.connect()
    t_str = str(t)
    files = _files_for(t_str)

    summary_df = summarize(t_str, con=con)

    sections = [
        _header(t_str, con, files),
        _schema_section(t_str, con, summary_df, files, style),
        _summary_section(t_str, con, summary_df, style, sparklines),
        _partitions_section(con, files, style),
        _per_partition_section(t_str, con, files, style) if per_partition else "",
        _histograms_section(t_str, con, summary_df, style) if histograms else "",
        _sample_section(t_str, con, sample_n, sample_seed, style),
    ]

    text = "\n\n".join(s for s in sections if s)

    if output is not None:
        Path(output).write_text(text)

    return text


def _files_for(t: str) -> list[Path]:
    """Resolve `t` to a list of parquet files. Empty for catalog names."""
    p = Path(t)
    if p.is_dir():
        return sorted(p.rglob("*.parquet"))
    if any(c in t for c in "*?["):
        return sorted(Path(f) for f in glob(t, recursive=True))
    if p.is_file():
        return [p]
    return []


def _hive_key(f: Path) -> str:
    """Slash-joined hive key from a file path (e.g. 'year=2022/region=us'); '' if none."""
    return "/".join(p for p in f.parts if "=" in p)


def _first_hive_column(hive_key: str) -> str:
    """Extract the leftmost partition column name from a hive key."""
    return hive_key.split("/")[0].split("=", 1)[0]


def _cell_str(v: object) -> str:
    return "" if v is None else str(v)


def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    val: float = float(n)
    for unit in ("KB", "MB", "GB", "TB"):
        val /= 1024
        if val < 1024:
            return f"{val:.1f} {unit}"
    return f"{val:.1f} PB"


def _render_table(rows: list[dict[str, object]], style: TableStyle) -> str:
    """Render rows as a markdown table or a code-fenced Unicode-box table."""
    if not rows:
        return "(empty)"
    if style == "box":
        return "```\n" + _box_table(rows) + "\n```"
    return _markdown_table(rows)


def _markdown_table(rows: list[dict[str, object]]) -> str:
    cols = list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(_cell_str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join([header, sep] + body)


def _box_table(rows: list[dict[str, object]]) -> str:
    cols = list(rows[0].keys())
    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(_cell_str(r[c])))

    def divider(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (widths[c] + 2) for c in cols) + right

    top = divider("┌", "┬", "┐")
    sep = divider("├", "┼", "┤")
    bot = divider("└", "┴", "┘")
    header = "│ " + " │ ".join(c.ljust(widths[c]) for c in cols) + " │"
    body = [
        "│ " + " │ ".join(_cell_str(r[c]).ljust(widths[c]) for c in cols) + " │"
        for r in rows
    ]
    return "\n".join([top, header, sep] + body + [bot])


def _header(t: str, con: duckdb.DuckDBPyConnection, files: list[Path]) -> str:
    rows = con.sql(f"SELECT COUNT(*) FROM query_table('{t}')").fetchone()[0]
    size = _human_bytes(sum(f.stat().st_size for f in files)) if files else "n/a"
    partitions = len({_hive_key(f) for f in files if _hive_key(f)}) if files else 0
    partition_line = f"\n**Partitions:** {partitions}" if partitions else ""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"# Dataset: {t}\n\n"
        f"**Generated:** {timestamp}{partition_line}\n"
        f"**Rows:** {rows:,}\n"
        f"**Size on disk:** {size}\n"
        f"**Files:** {len(files) if files else 'n/a'}"
    )


def _schema_section(
    t: str,
    con: duckdb.DuckDBPyConnection,
    summary_df: pl.DataFrame,
    files: list[Path],
    style: TableStyle,
) -> str:
    pl_schema = pl.scan_parquet(t).collect_schema() if files else None
    duckdb_types = dict(
        con.sql(
            f"SELECT column_name, column_type FROM (DESCRIBE FROM query_table('{t}'))"
        ).fetchall()
    )
    null_pcts = dict(
        zip(
            summary_df["column_name"].to_list(),
            _align_decimals([str(p) for p in summary_df["null_perc"].to_list()]),
        )
    )

    arrow_types: dict[str, str] = {}
    if files:
        import pyarrow.parquet as pq

        arrow_schema = pq.read_schema(str(files[0]))
        arrow_types = {f.name: str(f.type) for f in arrow_schema}

    rows = []
    for col in duckdb_types:
        rows.append(
            {
                "column": col,
                "arrow": arrow_types.get(col, ""),
                "polars": str(pl_schema[col]) if pl_schema and col in pl_schema else "",
                "duckdb": duckdb_types[col],
                "null %": null_pcts.get(col, "0"),
            }
        )
    return "## Schema\n\n" + _render_table(rows, style)


def _summary_section(
    t: str,
    con: duckdb.DuckDBPyConnection,
    summary_df: pl.DataFrame,
    style: TableStyle,
    sparklines: bool,
) -> str:
    rows = [dict(zip(summary_df.columns, row)) for row in summary_df.iter_rows()]
    if sparklines:
        for r in rows:
            dtype = str(r["column_type"])
            if _is_numeric(dtype):
                bins = _bin_counts(t, str(r["column_name"]), dtype, con)
                r["sparkline"] = _render_sparkline(bins[2]) if bins else ""
            else:
                r["sparkline"] = ""
    return "## Summary\n\n" + _render_table(rows, style)


def _histograms_section(
    t: str,
    con: duckdb.DuckDBPyConnection,
    summary_df: pl.DataFrame,
    style: TableStyle,
) -> str:
    blocks: list[str] = []
    for col, dtype in summary_df.select("column_name", "column_type").iter_rows():
        if not _is_numeric(str(dtype)):
            continue
        bins = _bin_counts(t, str(col), str(dtype), con)
        if bins is None:
            continue
        blocks.append(f"### {col}\n\n```\n{_render_histogram(*bins)}\n```")
    if not blocks:
        return ""
    return "## Histograms\n\n" + "\n\n".join(blocks)


def _is_numeric(dtype: str) -> bool:
    return dtype.upper().startswith(_NUMERIC_TYPE_PREFIXES)


def _bin_counts(
    t: str,
    col: str,
    dtype: str,
    con: duckdb.DuckDBPyConnection,
) -> tuple[float, float, list[int]] | None:
    """Compute (min, max, counts) for a 10-bucket equal-width histogram.

    Returns None for degenerate inputs (all-null or single-value columns).
    Float columns route through `isnan(...) → NULL` since duckdb's MIN/MAX
    don't ignore NaN the way they ignore NULL.
    """
    col_expr = (
        f'CASE WHEN isnan("{col}") THEN NULL ELSE "{col}" END'
        if dtype.upper().startswith(_FLOAT_TYPE_PREFIXES)
        else f'"{col}"'
    )
    bounds = con.sql(
        f"SELECT MIN({col_expr}), MAX({col_expr}) FROM query_table('{t}')"
    ).fetchone()
    if bounds is None:
        return None
    lo, hi = bounds
    if lo is None or hi is None or lo == hi:
        return None

    rows = con.sql(
        f"""
        SELECT LEAST(GREATEST(
            FLOOR((CAST({col_expr} AS DOUBLE) - {lo}) / {hi - lo} * {_BIN_COUNT})::INT,
            0
        ), {_BIN_COUNT - 1}) AS bucket,
        COUNT(*) AS cnt
        FROM query_table('{t}')
        WHERE {col_expr} IS NOT NULL
        GROUP BY bucket
        ORDER BY bucket
        """
    ).fetchall()
    counts = [0] * _BIN_COUNT
    for bucket, cnt in rows:
        if bucket is not None and 0 <= bucket < _BIN_COUNT:
            counts[bucket] = cnt
    return (lo, hi, counts)


def _render_sparkline(counts: list[int]) -> str:
    max_count = max(counts) if counts else 0
    if max_count == 0:
        return ""
    levels = len(_SPARKLINE_CHARS) - 1
    return "".join(_SPARKLINE_CHARS[round(c / max_count * levels)] for c in counts)


def _render_histogram(lo: float, hi: float, counts: list[int]) -> str:
    bar_width = 30
    bin_width = (hi - lo) / len(counts)
    max_count = max(counts) if counts else 0
    if max_count == 0:
        return "(empty)"
    edge_w = max(len(f"{lo + i * bin_width:.4g}") for i in range(len(counts)))
    count_w = len(f"{max_count:,}")
    lines = []
    for i, c in enumerate(counts):
        bin_lo = lo + i * bin_width
        bar = _HISTOGRAM_BAR * round(c / max_count * bar_width)
        lines.append(f"{bin_lo:>{edge_w}.4g} | {bar:<{bar_width}} {c:>{count_w},}")
    header = f"n = {sum(counts):,}, min = {lo:.4g}, max = {hi:.4g}"
    return f"{header}\n\n" + "\n".join(lines)


def _partitions_section(
    con: duckdb.DuckDBPyConnection, files: list[Path], style: TableStyle
) -> str:
    if not files:
        return ""
    groups: dict[str, list[Path]] = {}
    for f in files:
        groups.setdefault(_hive_key(f), []).append(f)
    if not any(k for k in groups):
        return ""

    rows = []
    for key in sorted(groups):
        fs = groups[key]
        size = sum(f.stat().st_size for f in fs)
        paths = ", ".join(f"'{f}'" for f in fs)
        n = con.sql(f"SELECT COUNT(*) FROM read_parquet([{paths}])").fetchone()[0]
        rows.append(
            {
                "partition": key,
                "rows": f"{n:,}",
                "size": _human_bytes(size),
                "files": len(fs),
            }
        )
    return "## Partitions\n\n" + _render_table(rows, style)


def _per_partition_section(
    t: str,
    con: duckdb.DuckDBPyConnection,
    files: list[Path],
    style: TableStyle,
) -> str:
    if not files:
        return ""
    first_key = next((_hive_key(f) for f in files if _hive_key(f)), "")
    if not first_key:
        return ""
    first_col = _first_hive_column(first_key)

    g = summarize(t, group_by=first_col, con=con)
    rows = [dict(zip(g.columns, row)) for row in g.iter_rows()]
    return f"## Per-partition summary (by `{first_col}`)\n\n" + _render_table(
        rows, style
    )


def _sample_section(
    t: str, con: duckdb.DuckDBPyConnection, n: int, seed: int, style: TableStyle
) -> str:
    sql = f"""
        SELECT * FROM query_table('{t}')
        LIMIT {n}
        OFFSET (
            hash({seed}) % GREATEST(1,
                (SELECT COUNT(*) FROM query_table('{t}')) - {n} + 1
            )
        )::BIGINT
    """
    df = con.sql(sql).pl()
    rows = [dict(zip(df.columns, row)) for row in df.iter_rows()]
    return f"## Sample ({n} contiguous rows, seed={seed})\n\n" + _render_table(
        rows, style
    )
