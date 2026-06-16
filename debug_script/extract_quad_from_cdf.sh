#!/bin/sh
# Post-process SQDiffuse inference outputs into quad meshes via QuadTools/QuadExtraction,
# then evaluate loop simplicity on each extracted quad mesh via QuadTools/QuadQuality.
#
# For every <input_dir>/<mesh_id>/gen_*/extract_mesh.npz produced by
# infer_sqdiffuse.py, run:
#     QuadTools/build/QuadExtraction -i <mesh_id>/gt_mesh_subdiv.ply
#                                    -f gen_*/extract_mesh.npz
#                                    -o gen_*/extracted_quad.ply
#     QuadTools/build/QuadQuality    -i gen_*/extracted_quad.ply
#                                    -j gen_*/
# The loop simplicity JSON is written next to the extracted quad as
# gen_*/extracted_quad.json (FratioN / EratioN fields).
# Usage: sh debug_script/extract_quad_from_cdf.sh [inference_dir]
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

input_dir=results2/infer_sqdiffuse/test_data_4096_smooth/data
input_dir=results2/infer_sqdiffuse/test_data_8192_smooth/data

if [ ! -d "$input_dir" ]; then
    echo "Inference directory not found: $input_dir"
    exit 1
fi

if [ ! -x QuadTools/build/QuadExtraction ]; then
    echo "QuadTools/build/QuadExtraction not found. Build QuadTools first:"
    echo "  cd QuadTools && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j\$(nproc)"
    exit 1
fi

if [ ! -x QuadTools/build/QuadQuality ]; then
    echo "QuadTools/build/QuadQuality not found. Build QuadTools first:"
    echo "  cd QuadTools && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j\$(nproc)"
    exit 1
fi

# QuadExtraction hyper-parameters (mirror QuadTools/README.md sample command).
ringsize=8
sharpangle=150
div=0
binarize_threshold=0.1
subdiv=3
smooth=15

success=0
failed=0
quality_success=0
quality_failed=0
for mesh_dir in "$input_dir"/*/; do
    [ -d "$mesh_dir" ] || continue
    subdivided="${mesh_dir}gt_mesh_subdiv.ply"
    if [ ! -f "$subdivided" ]; then
        echo "Skip: missing $subdivided"
        continue
    fi

    for gen_dir in "$mesh_dir"gen_*/; do
        [ -d "$gen_dir" ] || continue
        features="${gen_dir}extract_mesh.npz"
        if [ ! -f "$features" ]; then
            echo "Skip: missing $features"
            continue
        fi

        output="${gen_dir}extracted_quad.ply"
        echo ""
        echo "Extracting: $output"
        if ./QuadTools/build/QuadExtraction \
            -i "$subdivided" \
            -f "$features" \
            -o "$output" \
            --ringsize=$ringsize \
            --verbose=false \
            -a $sharpangle \
            --div=$div \
            -t $binarize_threshold \
            -s $subdiv \
            -r $smooth; then
            success=$((success + 1))
        else
            echo "QuadExtraction failed for $features"
            failed=$((failed + 1))
            continue
        fi

        if [ ! -f "$output" ]; then
            echo "Skip loop simplicity: $output was not produced"
            continue
        fi

        echo "Loop simplicity: $output"
        if ./QuadTools/build/QuadQuality \
            -i "$output" \
            -j "$gen_dir" \
            -v; then
            quality_success=$((quality_success + 1))
        else
            echo "QuadQuality failed for $output"
            quality_failed=$((quality_failed + 1))
        fi
    done
done

echo ""
echo "Extraction done. Success: $success, failed: $failed."
echo "Loop simplicity done. Success: $quality_success, failed: $quality_failed."
