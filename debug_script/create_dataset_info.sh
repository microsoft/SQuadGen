#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

input_folder=results2/debug/gen_npz_data_all/npz
outdir=results2/debug/create_dataset_info

python -m data_tools.create_dataset_info \
    --input-folder="$input_folder" \
    --outdir="$outdir" \
    --test-num=8 \
    --train-ratio=0.95 \
    --seed=0
