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


def summarize(
    t: str,
    *,
    group_by: str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pl.DataFrame:
    """Per-column summary statistics via duckdb's SUMMARIZE.

    Returns one row per column with: `column_name`, `column_type`, `min`,
    `max`, `approx_unique`, `avg`, `std`, `q25`, `q50`, `q75`, `count`,
    `null_percentage`. Quantiles are exact (duckdb spills to disk if
    needed). Works on 100M-row datasets without materialization.

    Args:
        t: catalog table/view name, parquet file path, or parquet glob
            (e.g. `'data/**/*.parquet'`).
        group_by: optional column to facet by. Returns one summary block
            per distinct value, with `group_by` as the leftmost column.
            Implementation runs a separate `SUMMARIZE` per group, so this
            is fine for low-cardinality columns (regions, years) and slow
            for high-cardinality ones (user_id, sku).
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
        return con.sql(f"SUMMARIZE FROM ({inner})").pl()

    distinct = con.sql(f'SELECT DISTINCT "{group_by}" FROM ({inner})').fetchall()

    parts = []
    for (val,) in distinct:
        sub = con.execute(
            f'SUMMARIZE FROM (SELECT * FROM ({inner}) WHERE "{group_by}" = ?)',
            [val],
        ).pl()
        sub = sub.filter(pl.col("column_name") != group_by)
        sub = sub.with_columns(pl.lit(val).alias(group_by))
        parts.append(sub)

    return pl.concat(parts, how="vertical_relaxed").select(
        group_by, pl.exclude(group_by)
    )
