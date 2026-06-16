#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

res=4096

ae_config=squadgen/network/config/sqvae_geomae.yaml
ae_pth=/mnt/output/my_project/remesh/experiments/vecset/patch0928_0928_quad_1024x32_patch_cdf_v4_a100_bs64_ga1_lr0.0001_of0/nomask/checkpoint-375.pth

results_dir=results2/reconstruct_sqvae/example_${res}_smooth
is_skip=0

input=examples/59a7e911d0ed408f89bfd3010564c03e_3.npz
name=data

python reconstruct_sqvae.py \
    --input="$input" \
    --results_dir="$results_dir" \
    --ae_config="$ae_config" \
    --ae_pth="$ae_pth" \
    --name="$name" \
    --res="$res" \
    --is_skip="$is_skip"
