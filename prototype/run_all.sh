#!/usr/bin/env bash
# Download the data (if missing) and run all SafeDischarge experiments.
set -euo pipefail
cd "$(dirname "$0")"
RAW=data/raw
if [ ! -f "$RAW/diabetic_data.csv" ]; then
  mkdir -p "$RAW"
  curl -sSL -o "$RAW/diabetes130.zip" \
    "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
  unzip -o -q "$RAW/diabetes130.zip" -d "$RAW"
fi
uv sync -q
uv run python scripts/00_tune.py
uv run python scripts/01_central.py
uv run python scripts/02_fairness.py
uv run python scripts/03_federated.py
uv run python scripts/04_explain.py
echo "Done. Results in results/. Dashboard: uv run streamlit run app.py"
