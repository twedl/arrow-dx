"""Verify README stays in sync with summarize()'s column contract.

Catches renames (e.g. null_percentage → null_perc) that would otherwise
silently make the README stale at release time.
"""

from __future__ import annotations

import re
from pathlib import Path

from arrow_dx.profile import _RENAMES, _STAT_COLS

README = Path(__file__).resolve().parent.parent / "README.md"


def test_readme_column_list_matches_summarize():
    expected = (
        {"column_name", "column_type", "count"}
        | set(_STAT_COLS)
        | set(_RENAMES.values())
    )
    text = README.read_text()
    m = re.search(r"Returns one row per column:([^.]+)\.", text)
    assert m, "couldn't find 'Returns one row per column:' line in README"
    found = set(re.findall(r"`([a-z0-9_]+)`", m.group(1)))
    assert found == expected, (
        f"README column list out of sync.\n"
        f"  missing from README: {sorted(expected - found)}\n"
        f"  extra in README:     {sorted(found - expected)}"
    )
