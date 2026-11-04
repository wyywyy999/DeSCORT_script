#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 || "$1" == "--help" ]]; then
    echo "Usage: bash $0 SCRIPT.py"
    echo "Edit input/output paths inside the selected script before running."
    echo "Scripts:"
    find "$HERE" -maxdepth 1 -name '*.py' -printf '  %f\n' | sort
    exit 0
fi
SCRIPT="$1"
shift
if [[ "$SCRIPT" != "$(basename -- "$SCRIPT")" || ! -f "$HERE/$SCRIPT" || "$SCRIPT" != *.py ]]; then
    echo "Unknown Python entry: $SCRIPT" >&2
    exit 2
fi
if [[ $# -ne 0 ]]; then
    echo "These original download scripts do not accept CLI arguments; edit paths inside the script." >&2
    exit 2
fi
exec "${PYTHON_BIN:-python}" "$HERE/$SCRIPT"
