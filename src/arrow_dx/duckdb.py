"""DuckDB integration for arrow-dx.

Ships a small set of SQL macros that mirror the polars helpers' semantics for
use from a duckdb connection or the duckdb CLI. `install_macros(con)` registers
them on a Python connection; `macros_sql()` returns the raw SQL for piping into
`~/.duckdbrc`.

Operates on duckdb.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb


MACROS_SQL = """\
-- arrow-dx duckdb macros
-- https://github.com/twedl/arrow-dx

CREATE OR REPLACE MACRO sample_print(t, n := 10, seed := NULL) AS TABLE (
    WITH c AS (SELECT COUNT(*) AS h FROM query_table(t::VARCHAR))
    SELECT * FROM query_table(t::VARCHAR)
    LIMIT n
    OFFSET (
        CASE WHEN seed IS NULL
            THEN (random() * GREATEST(0, (SELECT h FROM c) - n))
            ELSE (hash(seed) % GREATEST(1, (SELECT h FROM c) - n + 1))
        END
    )::BIGINT
);
"""


def install_macros(con: duckdb.DuckDBPyConnection) -> None:
    """Register arrow-dx SQL macros on the given duckdb connection.

    Available macros:

    - `sample_print(t, n := 10, seed := NULL)` — table macro returning `n`
      contiguous rows from a random offset in `t`. `t` may be a catalog
      table/view name, a parquet file path, or a parquet glob (e.g.
      `'data/**/*.parquet'`). Caller is responsible for sort order. With
      `seed := <int>` the offset is deterministic; without it, a fresh
      random offset is chosen on each call.
    """
    con.execute(MACROS_SQL)


def macros_sql() -> str:
    """Return the raw SQL that registers arrow-dx's duckdb macros.

    Useful for piping into `~/.duckdbrc`:

        uvx arrow-dx duckdb-macros >> ~/.duckdbrc
    """
    return MACROS_SQL
