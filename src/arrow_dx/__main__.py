"""arrow-dx CLI entry point."""

import argparse
import sys

from arrow_dx.duckdb import macros_sql


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arrow-dx")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "duckdb-macros",
        help="print arrow-dx duckdb SQL macros to stdout (pipe into ~/.duckdbrc)",
    )
    args = parser.parse_args(argv)

    if args.cmd == "duckdb-macros":
        sys.stdout.write(macros_sql())
        return 0


if __name__ == "__main__":
    sys.exit(main())
