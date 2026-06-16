#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [ ! -x QuadTools/build/CDFGen ]; then
    echo "QuadTools/build/CDFGen not found. Build QuadTools first:"
    echo "  cd QuadTools && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j\$(nproc)"
    exit 1
fi

results_dir=results2/debug/gen_npz_data_all
input_list="$results_dir/ply_filelist.txt"
outdir="$results_dir/npz"

mkdir -p "$results_dir"
cat > "$input_list" <<EOF
examples/59a7e911d0ed408f89bfd3010564c03e.ply
EOF

python -m data_tools.gen_npz_data \
    --input="$input_list" \
    --start=0 \
    --end=1 \
    --outdir="$outdir" \
    --num-points=50000
