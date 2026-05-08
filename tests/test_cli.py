import pytest

from arrow_dx.__main__ import main


def test_duckdb_macros_subcommand(capsys):
    rc = main(["duckdb-macros"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "CREATE OR REPLACE MACRO sample_print" in out


def test_no_subcommand_errors(capsys):
    with pytest.raises(SystemExit):
        main([])
