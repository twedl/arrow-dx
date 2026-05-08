"""Pretty-print helpers for polars, pandas, and pyarrow dataframes."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa


def sample_print(
    df: pl.DataFrame | pd.DataFrame | pa.Table,
    n: int = 10,
    seed: int | None = None,
) -> None:
    """Print n contiguous rows from a random offset in `df`.

    Operates on polars, pandas, or pyarrow Table dataframes. Caller is
    responsible for sorting `df` first — the contiguous window is only
    meaningful when rows are already in some intentional order. Defeats
    each engine's default row-clipping so the full window is visible.
    pyarrow Tables render through polars (pyarrow's `__repr__` shows only
    the schema). pyarrow Datasets are out of scope — call `.to_table()`
    first.

    Args:
        df: a polars, pandas, or pyarrow dataframe.
        n: window size; clipped to len(df) if larger.
        seed: optional seed for the offset RNG.
    """
    n = min(n, len(df))
    offset = random.Random(seed).randint(0, len(df) - n)
    mod = type(df).__module__.split(".")[0]

    if mod == "polars":
        with pl.Config(tbl_rows=n):
            print(df.slice(offset, n))
    elif mod == "pandas":
        import pandas as pd

        with pd.option_context("display.max_rows", n):
            print(df.iloc[offset : offset + n])
    elif mod == "pyarrow":
        with pl.Config(tbl_rows=n):
            print(pl.from_arrow(df.slice(offset, n)))
    else:
        raise TypeError(
            f"unsupported dataframe type: {type(df)}; "
            "expected polars, pandas, or pyarrow"
        )
