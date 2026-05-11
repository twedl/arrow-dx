# arrow-dx

Helpers for the Arrow-backed Python stack: polars, pyarrow, duckdb. Small, theme-bound — each helper smooths a rough edge in day-to-day dataframe work, nothing more.

## Install

```bash
pip install arrow-dx
# optional extras
pip install "arrow-dx[pandas]"
pip install "arrow-dx[pyarrow]"
pip install "arrow-dx[duckdb]"
pip install "arrow-dx[all]"
```

Requires Python 3.13+. Only polars is a hard runtime dependency.

## Helpers

### `unify_schema(root, *, output_dir=None, type_overrides=None, on_conflict="supertype", column_order=None)`

Rewrite every `*.parquet` under `root` to share one schema. Use case: a hive-partitioned dataset where files written at different times have drifted — a column added later, a type widened, a column missing in older files. After unification every reader (polars, pyarrow, duckdb, pandas) handles the dataset with its bare-minimum call.

```python
from arrow_dx import unify_schema

# rewrite files in place
unify_schema("data/")

# or copy into a new tree, preserving the partition layout
unify_schema("data/", output_dir="data-unified/")
```

- `on_conflict` ∈ `{"supertype" (default), "string", "error"}` — how to resolve a column whose type differs across files.
- `type_overrides={"col": pl.String}` — pin specific columns; wins over conflict resolution.
- `column_order=[...]` — pin a column-order prefix; remaining columns follow in first-seen order.
- Hive `key=value` partition columns are auto-detected and excluded from the file schema.

### `sample_print(df, n=10, seed=None)`

Print `n` contiguous rows from a random offset in a sorted polars, pandas, or pyarrow dataframe. Defeats each engine's default row-clipping so the full window is visible. Caller is responsible for sorting first — the slice is contiguous, so it's only meaningful in some intentional order. pyarrow `Dataset` (lazy) is out of scope; call `.to_table()` first.

```python
from arrow_dx import sample_print

sample_print(polars_df,    n=10, seed=42)
sample_print(pandas_df,    n=10, seed=42)
sample_print(pyarrow_table, n=10, seed=42)
```

### `summarize(t, *, group_by=None, con=None)`

Per-column summary statistics for a parquet dataset (or any duckdb-readable thing). Returns one row per column: `column_name`, `column_type`, `min`, `max`, `n_unique`, `avg`, `std`, `q25`, `q50`, `q75`, `count`, `null_perc`. Numeric stats are formatted to 4 significant figures and decimal-aligned for compact display. `min`/`max` are nulled for VARCHAR columns (lexical bounds rarely meaningful); DATE columns keep `YYYY-MM-DD` format in aggregates. Quantiles are exact (duckdb spills to disk if needed) and the underlying `SUMMARIZE` runs streaming, so 100M-row datasets don't materialize.

```python
from arrow_dx import summarize

summarize("data/**/*.parquet")
summarize("data/**/*.parquet", group_by="region")  # column-major: each column across regions
summarize("orders", con=my_con)                    # use a registered table
```

`t` is a catalog table/view name, parquet file path, or parquet glob. `group_by` runs a separate `SUMMARIZE` per distinct value — fast for low-cardinality columns (regions, years), slow for high-cardinality ones (`user_id`, `sku`). To summarize an in-memory polars DataFrame, register it on a connection first: `con.register("df", my_df); summarize("df", con=con)`.

Requires the `duckdb` extra: `pip install "arrow-dx[duckdb]"`.

## DuckDB

`sample_print` is also shipped as a duckdb SQL macro. To install it for the duckdb CLI:

```bash
uvx arrow-dx duckdb-macros >> ~/.duckdbrc
```

Then in any CLI session:

```sql
SELECT * FROM sample_print('orders', 10);                    -- catalog table
SELECT * FROM sample_print('s3://bucket/data.parquet', 10);  -- single file
SELECT * FROM sample_print('data/**/*.parquet', 10, seed := 42);  -- glob, deterministic
```

The argument is a string — a catalog table/view name, a parquet file path, or a parquet glob (`'dir/**/*.parquet'`). Caller still owns sort order. With `seed := <int>` the offset is deterministic; without it, a fresh random offset each call. The duckdb CLI's row-clip is `.maxrows N` in your `~/.duckdbrc`, separate from the macro.

For Python users with an open connection:

```python
import duckdb
from arrow_dx.duckdb import install_macros

con = duckdb.connect()
install_macros(con)
con.sql("SELECT * FROM sample_print('mytable', 10, seed := 42)")
```

## Reports

`arrow_dx.report.markdown(t)` renders a one-page markdown report — header (path, rows, size, partition count), schema (arrow / polars / duckdb types side-by-side + null %), summary (per-column min/max/avg/std/quartiles via duckdb's `SUMMARIZE`, plus an inline sparkline for numeric columns), hive-partition breakdown, and a contiguous-window sample.

```python
from arrow_dx import markdown

# print to stdout
print(markdown("data/**/*.parquet"))

# write to a file alongside the data
markdown("data/**/*.parquet", output="data/REPORT.md", sample_n=20, sample_seed=1)
```

Options:

- `style="markdown"` (default) renders pipe-and-dash tables that markdown viewers turn into styled tables; `style="box"` switches to Unicode box-drawing tables (┌─┬─┐) wrapped in code fences for monospace contexts (terminals, plain-text files). Sparklines and histograms always use Unicode block characters.
- `per_partition=True` adds a Per-partition summary section faceted by the leftmost hive column (low-cardinality only — runs one `SUMMARIZE` per group).
- `sparklines=True` (default) adds a `sparkline` column to the Summary table for numeric columns, showing the per-column distribution in 10 buckets via Unicode block chars (▁▂▃▄▅▆▇█).
- `histograms=True` (opt-in) adds a Histograms section with full bar-chart per numeric column — bin edges, bar, count.

Drop-in for code review, dataset onboarding, or living docs alongside the data. Deterministic sample seed by default so re-running on unchanged data produces a stable diff.

Sparkline rendering note: Unicode block chars (▁▂▃▄▅▆▇█) report a monospace width of 1 cell but can render slightly wider in VS Code / Cursor with the default font (Menlo, Monaco, SF Mono). Over 10 chars the drift accumulates and the sparkline overflows past the right `│` of its cell. This is an upstream Chromium/Electron limitation ([microsoft/vscode#1727](https://github.com/microsoft/vscode/issues/1727), closed won't-fix in 2017), not something the report can fix on its end. Switch your editor font to one with consistent monospace block-char widths — JetBrains Mono, Fira Code, Cascadia Code, IBM Plex Mono all work.

Requires the `duckdb` extra: `pip install "arrow-dx[duckdb]"`.

## Scope

In: ergonomics for the Arrow-backed dataframe stack — polars, pyarrow, duckdb (+ pandas as an extra). Out: anything that doesn't fit that theme. The discipline keeps this from rotting into a junk drawer.
