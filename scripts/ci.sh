#!/usr/bin/env bash
# Local CI — runs before `git push` via .githooks/pre-push.
# Bypass: `git push --no-verify`. Manual run: `scripts/ci.sh`.
set -euo pipefail

cd "$(dirname "$0")/.."

# Auto-activate .venv if not already inside one.
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

step "ruff check"
uvx ruff check .

step "ruff format --check"
uvx ruff format --check .

step "pytest"
pytest -q tests/

printf '\n\033[32mall checks passed\033[0m\n'
