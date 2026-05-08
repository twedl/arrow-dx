"""Rewrite every *.parquet under a root so they share one schema.

Use case: a hive-partitioned dataset where files written at different
times have drifted — a column added later, a type widened, a renamed
column. Once unified, every reader (polars, pandas, pyarrow, duckdb)
can read the dataset with its bare-minimum call.

The function loads each file fully, casts columns to the resolved
target type, fills missing columns with null, and writes back. Cost
is O(total_bytes_read + total_bytes_written) — there's no shortcut
that avoids materializing each file (parquet schemas aren't mutable
in place).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl


def _supertype(types: list[pl.DataType]) -> pl.DataType:
    """Return polars' inferred supertype for a list of dtypes, falling
    back to String if no clean promotion exists. Implementation piggybacks
    on the same logic pl.concat(..., how='diagonal_relaxed') uses."""
    distinct = list(dict.fromkeys(types))
    if len(distinct) == 1:
        return distinct[0]
    try:
        empties = [pl.DataFrame(schema={"x": t}) for t in distinct]
        return pl.concat(empties, how="diagonal_relaxed").schema["x"]
    except Exception:
        return pl.String()


def _resolve_target_type(
    col: str,
    types: list[pl.DataType],
    overrides: dict[str, pl.DataType] | None,
    on_conflict: Literal["supertype", "string", "error"],
) -> pl.DataType:
    if overrides and col in overrides:
        return overrides[col]
    distinct = list(dict.fromkeys(types))
    if len(distinct) == 1:
        return distinct[0]
    if on_conflict == "string":
        return pl.String()
    if on_conflict == "error":
        raise ValueError(
            f"column '{col}' has incompatible types across files: {distinct}; "
            f"pass type_overrides={{'{col}': ...}} or "
            f"on_conflict='string' / 'supertype'"
        )
    return _supertype(distinct)


def _partition_cols_for(rel_path: Path) -> set[str]:
    """Hive convention: each path segment shaped 'key=value' contributes 'key'."""
    return {p.split("=", 1)[0] for p in rel_path.parts if "=" in p}


def unify_schema(
    root: Path | str,
    *,
    output_dir: Path | str | None = None,
    type_overrides: dict[str, pl.DataType] | None = None,
    on_conflict: Literal["supertype", "string", "error"] = "supertype",
    column_order: list[str] | None = None,
) -> dict[str, pl.DataType]:
    """Rewrite every *.parquet under `root` to share a unified schema.

    Args:
        root: directory containing parquet files (recursively walked).
        output_dir: if given, write to this dir mirroring the partition
            tree; if None, rewrite files in place (DESTRUCTIVE).
        type_overrides: pin specific columns to a chosen dtype; takes
            precedence over conflict resolution.
        on_conflict: how to resolve a column whose type differs across
            files. 'supertype' (default) uses polars' diagonal_relaxed
            promotion with String fallback; 'string' always coerces
            mismatched columns to String; 'error' raises and forces an
            override.
        column_order: optional list of columns to place first in the
            output schema, in the given order. Columns not listed are
            appended in first-seen order (stable). Names that don't
            exist in any file raise an error.

    Returns:
        The unified schema actually written, as {col: dtype}, in the
        final column order.
    """
    root = Path(root)
    out_root = Path(output_dir) if output_dir is not None else None
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no *.parquet files under {root}")

    # Pass 1: discover partition columns + per-column observed types.
    partition_cols: set[str] = set()
    for f in files:
        partition_cols |= _partition_cols_for(f.relative_to(root))

    seen_order: list[str] = []
    types_by_col: dict[str, list[pl.DataType]] = {}
    for f in files:
        schema = pl.scan_parquet(f).collect_schema()
        for col, dtype in schema.items():
            if col in partition_cols:
                continue
            if col not in types_by_col:
                seen_order.append(col)
                types_by_col[col] = []
            types_by_col[col].append(dtype)

    target_types = {
        col: _resolve_target_type(col, ts, type_overrides, on_conflict)
        for col, ts in types_by_col.items()
    }

    # Validate column_order, then compute final order.
    if column_order is not None:
        unknown = [c for c in column_order if c not in target_types]
        if unknown:
            raise ValueError(
                f"column_order references columns not in any file: {unknown}; "
                f"available: {list(target_types.keys())}"
            )
        rest = [c for c in seen_order if c not in column_order]
        ordered_cols = list(column_order) + rest
    else:
        ordered_cols = seen_order

    # Pass 2: rewrite each file with the target schema and column order.
    for f in files:
        df = pl.read_parquet(f)
        df = df.select([c for c in df.columns if c not in partition_cols])
        for col in ordered_cols:
            target = target_types[col]
            if col not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=target).alias(col))
            elif df.schema[col] != target:
                df = df.with_columns(pl.col(col).cast(target))
        df = df.select(ordered_cols)

        out = out_root / f.relative_to(root) if out_root else f
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)

    return {c: target_types[c] for c in ordered_cols}
