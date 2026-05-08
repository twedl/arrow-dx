"""Pretty-print helpers for polars DataFrames."""

import random

import polars as pl


def sample_print(df: pl.DataFrame, n: int = 10, seed: int | None = None) -> None:
    """Print n contiguous rows from a random offset in `df`.

    Caller is responsible for sorting `df` first — the contiguous window is
    only meaningful when the rows are already in some intentional order.
    Operates on polars DataFrames. Uses pl.Config(tbl_rows=n) to defeat
    polars' default row-clipping so the full window is visible.

    Args:
        df: a polars DataFrame.
        n: window size; clipped to df.height if larger.
        seed: optional seed for the offset RNG.
    """
    n = min(n, df.height)
    offset = random.Random(seed).randint(0, df.height - n)
    with pl.Config(tbl_rows=n):
        print(df.slice(offset, n))
