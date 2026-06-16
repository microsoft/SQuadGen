#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

debug=0
res=4096
num_ckpt=194

ae_config=squadgen/network/config/sqvae_geomae.yaml
ae_pth=/mnt/output/my_project/remesh/experiments/vecset/patch0928_0928_quad_1024x32_patch_cdf_v4_a100_bs64_ga1_lr0.0001_of0/nomask/checkpoint-375.pth

results_dir=results2/reconstruct_geometry_ae/test_data_${res}_smooth
start=0
end=6
is_skip=0

input_filelist=examples/test_data/filelist.json
name=data

python reconstruct_geometry_ae.py \
    --input_filelist="$input_filelist" \
    --results_dir="$results_dir" \
    --ae_config="$ae_config" \
    --ae_pth="$ae_pth" \
    --name="$name" \
    --debug="$debug" \
    --res="$res" \
    --start="$start" \
    --end="$end" \
    --is_skip="$is_skip"