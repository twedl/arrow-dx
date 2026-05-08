import pytest

duckdb = pytest.importorskip("duckdb")

from arrow_dx.duckdb import install_macros  # noqa: E402


@pytest.fixture
def con():
    c = duckdb.connect()
    install_macros(c)
    return c


def test_catalog_table(con):
    con.execute("CREATE TABLE t AS SELECT range AS i FROM range(100)")
    rows = con.sql("SELECT * FROM sample_print('t', 5, seed := 42)").fetchall()
    assert len(rows) == 5


def test_seed_deterministic(con):
    con.execute("CREATE TABLE t AS SELECT range AS i FROM range(100)")
    a = con.sql("SELECT * FROM sample_print('t', 5, seed := 42)").fetchall()
    b = con.sql("SELECT * FROM sample_print('t', 5, seed := 42)").fetchall()
    assert a == b


def test_different_seeds_give_different_windows(con):
    con.execute("CREATE TABLE t AS SELECT range AS i FROM range(1000)")
    a = con.sql("SELECT * FROM sample_print('t', 5, seed := 1)").fetchall()
    b = con.sql("SELECT * FROM sample_print('t', 5, seed := 2)").fetchall()
    assert a != b


def test_parquet_path(con, tmp_path):
    p = tmp_path / "data.parquet"
    con.execute(f"COPY (SELECT range AS i FROM range(100)) TO '{p}'")
    rows = con.sql(f"SELECT * FROM sample_print('{p}', 5, seed := 42)").fetchall()
    assert len(rows) == 5


def test_parquet_glob(con, tmp_path):
    for year in (2002, 2003):
        d = tmp_path / f"year={year}"
        d.mkdir()
        con.execute(f"COPY (SELECT range AS i FROM range(50)) TO '{d}/p.parquet'")
    pattern = str(tmp_path / "**" / "*.parquet")
    rows = con.sql(f"SELECT * FROM sample_print('{pattern}', 5, seed := 42)").fetchall()
    assert len(rows) == 5


def test_n_exceeds_height_returns_what_exists(con):
    con.execute("CREATE TABLE small AS SELECT range AS i FROM range(3)")
    rows = con.sql("SELECT * FROM sample_print('small', 10, seed := 42)").fetchall()
    assert len(rows) == 3


def test_install_macros_idempotent(con):
    install_macros(con)
    install_macros(con)
    con.execute("CREATE TABLE t AS SELECT range AS i FROM range(10)")
    rows = con.sql("SELECT * FROM sample_print('t', 3, seed := 42)").fetchall()
    assert len(rows) == 3
