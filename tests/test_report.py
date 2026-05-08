import pytest

duckdb = pytest.importorskip("duckdb")

import polars as pl  # noqa: E402

from arrow_dx.report import markdown  # noqa: E402


@pytest.fixture
def hive_dataset(tmp_path):
    """Two hive partitions with overlapping schemas + a NaN sprinkled in."""
    for year in (2022, 2023):
        d = tmp_path / f"year={year}"
        d.mkdir()
        df = pl.DataFrame(
            {
                "id": list(range(20)),
                "region": ["us"] * 10 + ["eu"] * 10,
                "amount": [float(i) for i in range(20)],
                "score": [float("nan") if i == 5 else float(i) for i in range(20)],
            }
        )
        df.write_parquet(d / "p.parquet")
    return tmp_path


def test_returns_string_with_all_sections(hive_dataset):
    md = markdown(str(hive_dataset / "**" / "*.parquet"))
    assert "# Dataset:" in md
    assert "## Schema" in md
    assert "## Summary" in md
    assert "## Partitions" in md
    assert "## Sample" in md
    assert "**Rows:** 40" in md


def test_output_writes_to_file(hive_dataset, tmp_path):
    out = tmp_path / "report.md"
    md = markdown(str(hive_dataset / "**" / "*.parquet"), output=out)
    assert out.read_text() == md
    assert "# Dataset:" in out.read_text()


def test_schema_has_three_engine_columns(hive_dataset):
    md = markdown(str(hive_dataset / "**" / "*.parquet"))
    schema = md.split("## Schema")[1].split("##")[0]
    assert "arrow" in schema
    assert "polars" in schema
    assert "duckdb" in schema
    assert "null %" in schema
    # Type names from each engine
    assert "int64" in schema  # arrow
    assert "Int64" in schema  # polars
    assert "BIGINT" in schema  # duckdb


def test_partitions_section_enumerates_hive(hive_dataset):
    md = markdown(str(hive_dataset / "**" / "*.parquet"))
    partitions = md.split("## Partitions")[1].split("##")[0]
    assert "year=2022" in partitions
    assert "year=2023" in partitions


def test_partitions_section_skipped_for_unpartitioned_file(tmp_path):
    """Single file with no hive layout → no Partitions section."""
    p = tmp_path / "data.parquet"
    pl.DataFrame({"x": [1, 2, 3]}).write_parquet(p)
    md = markdown(str(p))
    assert "## Partitions" not in md
    assert "## Schema" in md  # other sections still present


def test_sample_has_n_rows(hive_dataset):
    md = markdown(str(hive_dataset / "**" / "*.parquet"), sample_n=5, sample_seed=1)
    sample = md.split("## Sample")[1]
    # Header: caption + col header + separator + 5 data rows = 8 lines containing '|'
    pipe_lines = [line for line in sample.splitlines() if line.startswith("|")]
    # 1 column-header row + 1 separator + 5 data rows
    assert len(pipe_lines) == 7


def test_deterministic_for_fixed_seed(hive_dataset):
    a = markdown(str(hive_dataset / "**" / "*.parquet"), sample_seed=42)
    b = markdown(str(hive_dataset / "**" / "*.parquet"), sample_seed=42)
    # Headers contain timestamps that change — strip them before comparing
    a_body = a.split("## Schema", 1)[1]
    b_body = b.split("## Schema", 1)[1]
    assert a_body == b_body


def test_nan_in_float_column_doesnt_break(hive_dataset):
    """The NaN at index 5 of `score` (set up in the fixture) shouldn't break SUMMARIZE."""
    md = markdown(str(hive_dataset / "**" / "*.parquet"))
    # null_percentage for score should be > 0 due to the NaN coercion
    summary = md.split("## Summary")[1].split("##")[0]
    score_lines = [line for line in summary.splitlines() if line.startswith("| score")]
    assert score_lines, "score row missing from summary"


def test_box_style_uses_unicode_box_tables_in_code_fences(hive_dataset):
    """style='box' → tables use Unicode box-drawing chars inside ``` code fences."""
    md = markdown(str(hive_dataset / "**" / "*.parquet"), style="box")
    schema = md.split("## Schema", 1)[1].split("##", 1)[0]
    assert "```" in schema
    assert "┌" in schema  # top-left corner
    assert "┘" in schema  # bottom-right corner
    # The markdown pipe-and-dash separator should NOT appear
    assert "|---" not in schema


def test_per_partition_section_added_when_flag_and_hive(hive_dataset):
    md = markdown(str(hive_dataset / "**" / "*.parquet"), per_partition=True)
    assert "## Per-partition summary" in md
    pp = md.split("## Per-partition summary", 1)[1]
    # The grouping column 'year' should head each row as the leftmost column
    assert "year" in pp
    # 2 partitions × non-group columns gives multiple rows per group
    assert "2022" in pp
    assert "2023" in pp


def test_per_partition_skipped_when_no_hive(tmp_path):
    """Single unpartitioned file → flag is a silent no-op."""
    p = tmp_path / "data.parquet"
    pl.DataFrame({"x": [1, 2, 3]}).write_parquet(p)
    md = markdown(str(p), per_partition=True)
    assert "## Per-partition summary" not in md


def test_per_partition_skipped_when_flag_off(hive_dataset):
    md = markdown(str(hive_dataset / "**" / "*.parquet"))
    assert "## Per-partition summary" not in md


def test_sparklines_default_on_for_numeric_columns(hive_dataset):
    """Numeric columns get a sparkline cell; non-numeric get an empty cell."""
    md = markdown(str(hive_dataset / "**" / "*.parquet"))
    summary = md.split("## Summary", 1)[1].split("##", 1)[0]
    assert "sparkline" in summary  # column header present
    # At least one Unicode block char for the numeric columns
    assert any(c in summary for c in "▁▂▃▄▅▆▇█")


def test_sparklines_disabled_omits_column(hive_dataset):
    md = markdown(str(hive_dataset / "**" / "*.parquet"), sparklines=False)
    summary = md.split("## Summary", 1)[1].split("##", 1)[0]
    assert "sparkline" not in summary


def test_histograms_section_added_when_flag_on(hive_dataset):
    md = markdown(str(hive_dataset / "**" / "*.parquet"), histograms=True)
    assert "## Histograms" in md
    hist = md.split("## Histograms", 1)[1]
    # One subsection per numeric column (id, amount, score; 'region' is varchar)
    assert "### id" in hist
    assert "### amount" in hist
    assert "### score" in hist
    assert "### region" not in hist
    # Unicode bar chars regardless of style
    assert "█" in hist


def test_histograms_skipped_by_default(hive_dataset):
    md = markdown(str(hive_dataset / "**" / "*.parquet"))
    assert "## Histograms" not in md


def test_sparkline_skipped_for_degenerate_numeric_column(tmp_path):
    """All-null numeric column gets an empty sparkline cell, no crash."""
    p = tmp_path / "data.parquet"
    pl.DataFrame({"x": [None, None, None]}, schema={"x": pl.Int64}).write_parquet(p)
    md = markdown(str(p))
    # No exception; summary section still rendered
    assert "## Summary" in md
