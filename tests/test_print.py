import polars as pl
import pytest

from arrow_dx.print import sample_print


@pytest.fixture(params=["polars", "pandas", "pyarrow"])
def df_factory(request):
    """Return a callable that builds a one-column dataframe in the named engine."""
    engine = request.param
    if engine == "pandas":
        pd = pytest.importorskip("pandas")
        return lambda values: pd.DataFrame({"a": values})
    if engine == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        return lambda values: pa.table({"a": values})
    return lambda values: pl.DataFrame({"a": values})


def test_deterministic_for_fixed_seed(df_factory, capsys):
    """Same seed → identical output across calls."""
    df = df_factory(list(range(100)))
    sample_print(df, n=5, seed=42)
    first = capsys.readouterr().out
    sample_print(df, n=5, seed=42)
    second = capsys.readouterr().out
    assert first == second


def test_bounds_clipped_when_n_exceeds_height(df_factory, capsys):
    """A df shorter than n prints whatever is there, no error."""
    df = df_factory([1, 2, 3])
    sample_print(df, n=10, seed=0)
    out = capsys.readouterr().out
    assert "1" in out
    assert "2" in out
    assert "3" in out


def test_defeats_default_clipping(df_factory, capsys):
    """No row-truncation marker (… or ...) appears in the output."""
    df = df_factory(list(range(100)))
    sample_print(df, n=15, seed=42)
    out = capsys.readouterr().out
    assert "…" not in out
    assert "..." not in out


def test_unsupported_type_raises():
    """Non-supported dataframe types raise TypeError."""
    with pytest.raises(TypeError, match="unsupported dataframe type"):
        sample_print([1, 2, 3])
