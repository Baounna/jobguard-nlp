#!/usr/bin/env bash
# Reproducible download of the EMSCAD dataset from Kaggle.
#
# Prerequisites:
#   1. Kaggle API token at ~/.kaggle/kaggle.json (download from kaggle.com/account)
#   2. curl, unzip, python3
#
# Usage:
#   bash scripts/download_data.sh

set -euo pipefail

# Resolve project root (one directory above this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
RAW_DIR="$ROOT_DIR/data/raw"

mkdir -p "$RAW_DIR"

if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    echo "ERROR: ~/.kaggle/kaggle.json not found."
    echo "Generate it from https://www.kaggle.com/account → 'Create New API Token'."
    exit 1
fi

USER=$(python3 -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")
KEY=$(python3 -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['key'])")

echo "Downloading EMSCAD via Kaggle API as user '$USER'..."

cd "$RAW_DIR"
curl -sL -u "$USER:$KEY" \
     -o emscad.zip \
     -w "HTTP %{http_code}  size=%{size_download} bytes\n" \
     "https://www.kaggle.com/api/v1/datasets/download/shivamb/real-or-fake-fake-jobposting-prediction"

echo "Unzipping..."
unzip -o emscad.zip
rm emscad.zip

echo
echo "✅ Dataset ready at: $RAW_DIR/fake_job_postings.csv"
python3 -c "
import pandas as pd
df = pd.read_csv('$RAW_DIR/fake_job_postings.csv')
print(f'   {df.shape[0]:,} rows × {df.shape[1]} columns')
print(f'   {df[\"fraudulent\"].sum()} fraudulent ({df[\"fraudulent\"].mean()*100:.2f}%)')
"
