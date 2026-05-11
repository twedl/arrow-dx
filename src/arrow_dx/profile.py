"""Dataset reconnaissance helpers.

For "what's going on" inspection of large parquet datasets where full
materialization is too expensive. Returns polars DataFrames for downstream
analysis. Operates on duckdb (under the hood) for streaming aggregates and
parquet-metadata reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import duckdb

_STAT_COLS = ("min", "max", "avg", "std", "q25", "q50", "q75")
_RENAMES = {"null_percentage": "null_perc", "approx_unique": "n_unique"}


def _fmt_sigfigs(s: str | None) -> str | None:
    if s is None:
        return None
    try:
        return f"{float(s):.4g}"
    except ValueError:
        return s


def _align_decimals(values: list[str | None]) -> list[str | None]:
    """Pad numeric strings for decimal alignment within a column. Non-numeric
    values (including scientific notation) and nulls pass through unchanged."""
    parsed: list[tuple[str, str] | None] = []
    for v in values:
        if v is None or "e" in v.lower():
            parsed.append(None)
            continue
        try:
            float(v)
        except ValueError:
            parsed.append(None)
            continue
        i, _, f = v.partition(".")
        parsed.append((i, f))

    numerics = [p for p in parsed if p is not None]
    if not numerics:
        return values

    max_int = max(len(i) for i, _ in numerics)
    max_frac = max(len(f) for _, f in numerics)

    out: list[str | None] = []
    for v, p in zip(values, parsed):
        if p is None:
            out.append(v)
            continue
        i, f = p
        if max_frac > 0:
            dot = "." if f else " "
            out.append(f"{i:>{max_int}}{dot}{f:<{max_frac}}")
        else:
            out.append(f"{i:>{max_int}}")
    return out


def summarize(
    t: str,
    *,
    group_by: str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pl.DataFrame:
    """Per-column summary statistics via duckdb's SUMMARIZE.

    Returns one row per column with: `column_name`, `column_type`, `min`,
    `max`, `n_unique` (approx-distinct count), `avg`, `std`, `q25`, `q50`,
    `q75`, `count`, `null_perc`. Numeric stats are formatted to 4
    significant figures for compact display. Quantiles are exact (duckdb
    spills to disk if needed). Works on 100M-row datasets without
    materialization.

    Args:
        t: catalog table/view name, parquet file path, or parquet glob
            (e.g. `'data/**/*.parquet'`).
        group_by: optional column to facet by. Rows are column-major: each
            non-group column appears as a contiguous block across partitions
            (schema column order preserved), with partitions sorted by
            `group_by` value within each block. `group_by` is the leftmost
            column. Implementation runs a separate `SUMMARIZE` per group,
            so this is fine for low-cardinality columns (regions, years)
            and slow for high-cardinality ones (user_id, sku).
        con: optional duckdb connection; defaults to a fresh in-memory
            connection. Pass an existing connection to query its catalog
            tables/views.

    Operates on duckdb. Requires the `duckdb` extra:
    `pip install "arrow-dx[duckdb]"`.
    """
    import duckdb

    con = con or duckdb.connect()

    # Coerce NaN→NULL for float columns: duckdb's STDDEV_SAMP errors with
    # 'Out of Range' on NaN inputs, and pandas-written parquet routinely
    # has NaN where the writer meant NULL.
    schema = con.sql(f"DESCRIBE FROM query_table('{t}')").fetchall()
    select_list = ", ".join(
        f'CASE WHEN isnan("{name}") THEN NULL ELSE "{name}" END AS "{name}"'
        if dtype in ("FLOAT", "DOUBLE", "REAL")
        else f'"{name}"'
        for name, dtype, *_ in schema
    )
    inner = f"SELECT {select_list} FROM query_table('{t}')"

    if group_by is None:
        result = con.sql(f"SUMMARIZE FROM ({inner})").pl()
    else:
        distinct = con.sql(f'SELECT DISTINCT "{group_by}" FROM ({inner})').fetchall()

        parts = []
        for (val,) in distinct:
            sub = con.execute(
                f'SUMMARIZE FROM (SELECT * FROM ({inner}) WHERE "{group_by}" = ?)',
                [val],
            ).pl()
            sub = sub.filter(pl.col("column_name") != group_by)
            sub = sub.with_columns(pl.lit(val).alias(group_by))
            sub = sub.with_row_index("_col_idx")
            parts.append(sub)

        result = (
            pl.concat(parts, how="vertical_relaxed")
            .sort("_col_idx", group_by)
            .drop("_col_idx")
            .select(group_by, pl.exclude(group_by))
        )

    result = result.rename(_RENAMES).with_columns(
        [
            pl.col(c).map_elements(_fmt_sigfigs, return_dtype=pl.String)
            for c in _STAT_COLS
        ]
    )
    return result.with_columns(
        [pl.Series(c, _align_decimals(result[c].to_list())) for c in _STAT_COLS]
    )
