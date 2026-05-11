import pytest

duckdb = pytest.importorskip("duckdb")

import polars as pl  # noqa: E402

from arrow_dx.profile import summarize  # noqa: E402


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("""
        CREATE TABLE t AS
        SELECT
            range AS i,
            CAST(range * 1.5 AS DOUBLE) AS x,
            CASE WHEN range % 3 = 0 THEN NULL ELSE range::VARCHAR END AS s,
            CASE WHEN range < 50 THEN 'a' ELSE 'b' END AS g
        FROM range(100)
    """)
    return c


def test_returns_one_row_per_column(con):
    result = summarize("t", con=con)
    assert sorted(result["column_name"].to_list()) == ["g", "i", "s", "x"]


def test_includes_quantiles_and_null_perc(con):
    result = summarize("t", con=con)
    expected = {"min", "max", "avg", "std", "q25", "q50", "q75", "null_perc"}
    assert expected.issubset(set(result.columns))


def test_null_perc_reflects_nulls(con):
    """Column 's' has nulls every 3rd row → ~34/100 nulls."""
    result = summarize("t", con=con)
    s_row = result.filter(pl.col("column_name") == "s")
    i_row = result.filter(pl.col("column_name") == "i")
    assert float(s_row["null_perc"][0]) > 0
    assert float(i_row["null_perc"][0]) == 0


def test_group_by_facets_per_group(con):
    """group_by='g' yields one block per distinct g, group label leftmost,
    g itself dropped from the per-group summary."""
    result = summarize("t", group_by="g", con=con)
    assert result.columns[0] == "g"
    assert sorted(result["g"].unique().to_list()) == ["a", "b"]
    # 2 groups × 3 non-group columns (i, x, s)
    assert len(result) == 6
    assert "g" not in result["column_name"].to_list()


def test_group_by_orders_column_major(con):
    """Rows are column-major: each non-group column is a contiguous block,
    in original schema order; partitions sorted within each block."""
    result = summarize("t", group_by="g", con=con)
    rows = list(zip(result["column_name"].to_list(), result["g"].to_list()))
    # Schema order: i, x, s, g — g is the partition column. Distinct g
    # values are 'a' and 'b', sorted within each column block.
    assert rows == [
        ("i", "a"),
        ("i", "b"),
        ("x", "a"),
        ("x", "b"),
        ("s", "a"),
        ("s", "b"),
    ]


def test_group_by_stats_match_subset(con):
    """Per-group min/max/count match a manual filter."""
    result = summarize("t", group_by="g", con=con)
    a_i = result.filter((pl.col("g") == "a") & (pl.col("column_name") == "i"))
    assert int(a_i["min"][0]) == 0
    assert int(a_i["max"][0]) == 49
    assert a_i["count"][0] == 50


def test_parquet_path(con, tmp_path):
    p = tmp_path / "data.parquet"
    con.execute(f"COPY t TO '{p}'")
    result = summarize(str(p), con=con)
    assert sorted(result["column_name"].to_list()) == ["g", "i", "s", "x"]


def test_parquet_glob_with_hive_partitions(con, tmp_path):
    for grp in ("a", "b"):
        d = tmp_path / f"g={grp}"
        d.mkdir()
        con.execute(
            f"COPY (SELECT i, x, s FROM t WHERE g = '{grp}') TO '{d}/p.parquet'"
        )
    pattern = str(tmp_path / "**" / "*.parquet")
    result = summarize(pattern, con=con)
    cols = result["column_name"].to_list()
    assert "i" in cols
    assert "g" in cols  # hive partition column surfaces in the summary


def test_default_connection(tmp_path):
    """Without con=, summarize uses an ephemeral connection. Paths work
    because they don't need a registered catalog."""
    p = str(tmp_path / "data.parquet")
    duckdb.connect().execute(f"COPY (SELECT range AS i FROM range(10)) TO '{p}'")

    result = summarize(p)
    assert result["column_name"].to_list() == ["i"]
    assert result["count"][0] == 10


def test_handles_nan_in_float_column(tmp_path):
    """NaN in a float column is coerced to NULL — duckdb's STDDEV errors
    on NaN, and pandas-written parquet often uses NaN to mean NULL."""
    p = tmp_path / "data.parquet"
    pl.DataFrame({"score": [1.0, 2.0, float("nan"), 4.0, 5.0]}).write_parquet(p)

    result = summarize(str(p))

    score = result.filter(pl.col("column_name") == "score")
    assert float(score["null_perc"][0]) == 20.0
    assert int(score["min"][0]) == 1
    assert int(score["max"][0]) == 5
