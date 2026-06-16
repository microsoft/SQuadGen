#!/bin/sh
# Mirror of debug_script/gen_npz_data_all.sh but uses QuadTools/build/CDFGen
# so the two outputs can be diffed side by side.
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [ ! -x QuadTools/build/CDFGen ]; then
    echo "QuadTools/build/CDFGen not found. Build QuadTools first:"
    echo "  cd QuadTools && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j\$(nproc)"
    exit 1
fi

results_dir=results2/debug/gen_npz_data_all_quadtools
input_list="$results_dir/ply_filelist.txt"
outdir="$results_dir/npz"

mkdir -p "$outdir"
cat > "$input_list" <<EOF
examples/59a7e911d0ed408f89bfd3010564c03e.ply
EOF

while IFS= read -r ply_path; do
    [ -z "$ply_path" ] && continue
    echo ""
    echo "Processing: $ply_path"
    ./QuadTools/build/CDFGen \
        -i "$ply_path" \
        -d "$outdir" \
        -n 50000
done < "$input_list"

echo ""
echo "Done. Output: $outdir"
