#!/usr/bin/env bash
set -euo pipefail
HERE="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/tutorials/01_preprocess_data"
if [[ $# -eq 0 || "$1" == "--help" ]]; then
    echo "Usage: bash $0 SCRIPT.py [arguments...]"
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
exec "${PYTHON_BIN:-python}" "$HERE/$SCRIPT" "$@"
