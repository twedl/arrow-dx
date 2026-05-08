# arrow-dx

Helpers for the Arrow-backed Python stack: polars, pyarrow, duckdb. Small, theme-bound — each helper smooths a rough edge in day-to-day dataframe work, nothing more.

## Install

```bash
pip install arrow-dx
# optional extras
pip install "arrow-dx[pandas]"
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

Print `n` contiguous rows from a random offset in a sorted polars df, with `pl.Config(tbl_rows=n)` so polars' default row-clipping doesn't truncate the window. Caller is responsible for sorting first — the slice is contiguous, so it's only meaningful in some intentional order.

```python
from arrow_dx import sample_print

sample_print(sorted_df, n=10, seed=42)
```

## Scope

In: ergonomics for the Arrow-backed dataframe stack — polars, pyarrow, duckdb (+ pandas as an extra). Out: anything that doesn't fit that theme. The discipline keeps this from rotting into a junk drawer.
