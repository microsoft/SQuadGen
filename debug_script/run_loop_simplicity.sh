#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

input_mesh=${1:-examples/59a7e911d0ed408f89bfd3010564c03e.ply}
output_dir=results2/debug/loop_simplicity

mkdir -p "$output_dir"

sh QuadTools/run_loop_simplicity.sh "$input_mesh" | tee "$output_dir/score.txt"