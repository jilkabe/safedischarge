#!/usr/bin/env bash
# Compile the report with Tectonic (XeLaTeX) and copy it to the submission file name.
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC="${TECTONIC:-$HOME/.local/bin/tectonic}"
"$TECTONIC" main.tex
cp main.pdf ../SafeDischarge-final-report.pdf
echo "wrote report/SafeDischarge-final-report.pdf"
