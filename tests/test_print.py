import polars as pl

from arrow_dx.print import sample_print


def test_deterministic_for_fixed_seed(capsys):
    """Same seed → identical output across calls."""
    df = pl.DataFrame({"a": list(range(100))})
    sample_print(df, n=5, seed=42)
    first = capsys.readouterr().out
    sample_print(df, n=5, seed=42)
    second = capsys.readouterr().out
    assert first == second


def test_bounds_clipped_when_n_exceeds_height(capsys):
    """A df shorter than n prints whatever is there, no error."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    sample_print(df, n=10, seed=0)
    out = capsys.readouterr().out
    assert "1" in out
    assert "2" in out
    assert "3" in out


def test_defeats_default_clipping(capsys):
    """pl.Config(tbl_rows=n) is applied — no row-truncation marker appears."""
    df = pl.DataFrame({"a": list(range(100))})
    sample_print(df, n=15, seed=42)
    out = capsys.readouterr().out
    assert "…" not in out
