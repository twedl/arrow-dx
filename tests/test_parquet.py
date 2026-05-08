from pathlib import Path

import polars as pl
import pytest

from arrow_dx.parquet import unify_schema


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_supertype_promotion(tmp_path):
    """Default on_conflict='supertype' promotes Int + Float → Float, in place."""
    _write(
        pl.DataFrame({"a": [1, 2]}, schema={"a": pl.Int64}),
        tmp_path / "year=2002" / "p.parquet",
    )
    _write(
        pl.DataFrame({"a": [1.5, 2.5]}, schema={"a": pl.Float64}),
        tmp_path / "year=2003" / "p.parquet",
    )

    schema = unify_schema(tmp_path)

    assert list(schema.keys()) == ["a"]
    assert schema["a"] == pl.Float64
    assert (
        pl.read_parquet(tmp_path / "year=2002" / "p.parquet").schema["a"] == pl.Float64
    )
    assert (
        pl.read_parquet(tmp_path / "year=2003" / "p.parquet").schema["a"] == pl.Float64
    )


def test_on_conflict_string(tmp_path):
    """on_conflict='string' coerces mismatched columns to String."""
    _write(
        pl.DataFrame({"a": [1, 2]}, schema={"a": pl.Int64}),
        tmp_path / "p1" / "p.parquet",
    )
    _write(
        pl.DataFrame({"a": [True, False]}, schema={"a": pl.Boolean}),
        tmp_path / "p2" / "p.parquet",
    )

    schema = unify_schema(tmp_path, on_conflict="string")

    assert schema["a"] == pl.String


def test_on_conflict_error(tmp_path):
    """on_conflict='error' raises on type mismatch."""
    _write(
        pl.DataFrame({"a": [1, 2]}, schema={"a": pl.Int64}),
        tmp_path / "p1" / "p.parquet",
    )
    _write(
        pl.DataFrame({"a": [True, False]}, schema={"a": pl.Boolean}),
        tmp_path / "p2" / "p.parquet",
    )

    with pytest.raises(ValueError, match="incompatible types"):
        unify_schema(tmp_path, on_conflict="error")


def test_type_overrides_beats_conflict_resolution(tmp_path):
    """type_overrides takes precedence over on_conflict resolution."""
    _write(
        pl.DataFrame({"a": [1, 2]}, schema={"a": pl.Int64}),
        tmp_path / "p1" / "p.parquet",
    )
    _write(
        pl.DataFrame({"a": [1.5]}, schema={"a": pl.Float64}),
        tmp_path / "p2" / "p.parquet",
    )

    schema = unify_schema(tmp_path, type_overrides={"a": pl.String})

    assert schema["a"] == pl.String


def test_column_order_pins_prefix(tmp_path):
    """column_order pins a prefix; the rest follow in seen order."""
    _write(
        pl.DataFrame({"a": [1], "b": [2], "c": [3]}),
        tmp_path / "p1" / "p.parquet",
    )

    schema = unify_schema(tmp_path, column_order=["c", "a"])

    assert list(schema.keys()) == ["c", "a", "b"]
    assert pl.read_parquet(tmp_path / "p1" / "p.parquet").columns == ["c", "a", "b"]


def test_column_order_unknown_name_raises(tmp_path):
    _write(pl.DataFrame({"a": [1]}), tmp_path / "p1" / "p.parquet")

    with pytest.raises(ValueError, match="not in any file"):
        unify_schema(tmp_path, column_order=["nope"])


def test_partition_columns_excluded(tmp_path):
    """Hive-style key=value path segments don't appear in the unified schema."""
    _write(pl.DataFrame({"a": [1]}), tmp_path / "year=2002" / "region=us" / "p.parquet")

    schema = unify_schema(tmp_path)

    assert "year" not in schema
    assert "region" not in schema
    assert "a" in schema


def test_missing_columns_null_filled(tmp_path):
    """A file missing a column gets it filled with nulls at the resolved type."""
    _write(
        pl.DataFrame({"a": [1]}, schema={"a": pl.Int64}), tmp_path / "p1" / "p.parquet"
    )
    _write(
        pl.DataFrame({"b": [2]}, schema={"b": pl.Int64}), tmp_path / "p2" / "p.parquet"
    )

    unify_schema(tmp_path)

    df1 = pl.read_parquet(tmp_path / "p1" / "p.parquet")
    df2 = pl.read_parquet(tmp_path / "p2" / "p.parquet")
    assert df1.columns == ["a", "b"]
    assert df2.columns == ["a", "b"]
    assert df1["b"].null_count() == 1
    assert df2["a"].null_count() == 1


def test_output_dir_does_not_modify_input(tmp_path):
    """output_dir mirrors the partition tree; originals are untouched."""
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write(
        pl.DataFrame({"a": [1, 2]}, schema={"a": pl.Int64}),
        src / "year=2002" / "p.parquet",
    )
    _write(
        pl.DataFrame({"a": [1.5]}, schema={"a": pl.Float64}),
        src / "year=2003" / "p.parquet",
    )

    unify_schema(src, output_dir=out)

    # Inputs still original
    assert pl.read_parquet(src / "year=2002" / "p.parquet").schema["a"] == pl.Int64
    assert pl.read_parquet(src / "year=2003" / "p.parquet").schema["a"] == pl.Float64
    # Outputs unified at the new root, mirroring the tree
    assert (out / "year=2002" / "p.parquet").exists()
    assert (out / "year=2003" / "p.parquet").exists()
    assert pl.read_parquet(out / "year=2002" / "p.parquet").schema["a"] == pl.Float64
    assert pl.read_parquet(out / "year=2003" / "p.parquet").schema["a"] == pl.Float64


def test_empty_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"no \*\.parquet"):
        unify_schema(tmp_path)
