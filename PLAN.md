# arrow-dx — initial plan

## What it is
A small Python package of ergonomics helpers for the Arrow-backed dataframe stack: **polars, pyarrow, duckdb**. Not core data ops — just smoothing the rough edges you hit doing day-to-day work across these engines.

## Name
- **arrow-dx** (PyPI + GitHub: confirmed free as of bootstrap; double-check both before publishing).
- Fallbacks if taken later: `arrow-utils`, `arrow-helpers`, `df-dx`, `df-utils`.
- README's first line must say *"helpers for the Arrow-backed Python stack: polars, pyarrow, duckdb"* — heads off the assumption that it's a pyarrow-only package.

## Theme constraint (anti-junk-drawer rule)
Every helper must fit "ergonomics for Arrow-backed dataframes". If you can't justify a new helper in one sentence under that theme, it doesn't belong here — it goes in a different package or stays a snippet. This is the single rule that keeps this from rotting into a junk drawer.

## Project bootstrap
- uv-based Python project; mirror the layout/CI of `name-cluster`.
- `pyproject.toml` + hatchling, src/ layout (`src/arrow_dx/`).
- Reuse the PyPI trusted-publishing GitHub Action.

## Dependencies
- **Required:** polars (used internally by most helpers).
- **Optional extras:**
  - `arrow-dx[pandas]` → pandas
  - `arrow-dx[pyarrow]` → pyarrow
  - `arrow-dx[duckdb]` → duckdb
  - `arrow-dx[all]` → everything

## Initial helpers to port over
1. **`unify_schema(root, *, output_dir, type_overrides, on_conflict, column_order)`**
   Rewrites every `*.parquet` under `root` to share one schema (supertype promotion + null-fill for missing columns). Reference implementation already written and tested.
2. **`sample_print(df, n=10, seed=None)`**
   Prints `n` contiguous rows from a random offset in a sorted polars df, with `pl.Config(tbl_rows=n)` to defeat default clipping.

## `unify_schema` design decisions (already settled)
- **Type promotion:** polars supertype via `pl.concat(..., how="diagonal_relaxed")` with String fallback.
- **Hard conflicts:** `on_conflict` ∈ {`"supertype"` (default), `"string"`, `"error"`}.
- **Per-column overrides:** `type_overrides: dict[str, pl.DataType]` wins over conflict resolution.
- **Missing columns:** null-fill at the resolved target type.
- **Column order:** stable first-seen across files; `column_order=[...]` pins a prefix, rest appended in seen order; unknown names raise.
- **Partition columns:** auto-detected from `key=value` path segments; excluded from the file schema.
- **In-place vs copy:** `output_dir=None` rewrites in place (destructive); pass `output_dir` to mirror the partition tree under a new root.
- **Discovery:** single root + `rglob("*.parquet")`.

## Open decisions
- Should `unify_schema` drop polars and use pyarrow only? Probably no — `diagonal_relaxed` is what makes supertype computation tractable in ~5 lines.
- Streaming/chunked variant for files that don't fit in RAM? Defer until needed.
- Public API surface: single-file modules to start (`arrow_dx/parquet.py`, `arrow_dx/print.py`). Refactor only when a module passes ~500 lines or clearly splits into sub-themes.

## Candidate future helpers (log when they come up; don't pre-build)
- Hive-partitioned read with auto-schema reconciliation (so callers don't need to run `unify_schema` first).
- Cross-engine round-trip helpers (polars ↔ pandas ↔ pyarrow ↔ duckdb).
- Pretty-print variants for large/wide dataframes.
- Anything that comes from "I keep typing this same boilerplate".

## Discipline rules
1. **One theme.** Helpers that don't fit "Arrow-backed dataframe ergonomics" go elsewhere.
2. **Optional extras for non-polars deps.** No helper should drag pandas/duckdb into installs for users who don't need them.
3. **Engines named in docstrings.** Every public function says which of {polars, pyarrow, duckdb} it operates on.
4. **README declares scope explicitly.** Reduces "is this for me?" guesswork.
