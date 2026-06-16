#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

debug=0
res=1024
res=8192
num_ckpt=194

ae_config=squadgen/network/config/sqvae_geomae.yaml
ae_pth=/mnt/output/my_project/remesh/experiments/vecset/patch0928_0928_quad_1024x32_patch_cdf_v4_a100_bs64_ga1_lr0.0001_of0/nomask/checkpoint-375.pth
model_config=squadgen/network/config/sqdiffuse.yaml
model_pth=/mnt/output/my_project/remesh/experiments/vecset/patch0928f230kbins_1209_quad_ldm_1024x32_sitXL_patch_vae375_res4096_pcscale_v1_a100_40g_ib_bs64_ga8_lr0.0001_of0/checkpoint-${num_ckpt}.pth

results_dir=results2/infer_sqdiffuse/test_data_${res}_smooth
n_gen=1
start=0
# end=-1
# is_skip=1
end=-1
is_skip=1
use_latent_smoothing=1

input_filelist=examples/test_data/filelist.json
name=data_2

python infer_sqdiffuse.py \
    --input_filelist="$input_filelist" \
    --results_dir="$results_dir" \
    --ae_config="$ae_config" \
    --ae_pth="$ae_pth" \
    --model_config="$model_config" \
    --model_pth="$model_pth" \
    --name="$name" \
    --n_gen="$n_gen" \
    --debug="$debug" \
    --res="$res" \
    --start="$start" \
    --end="$end" \
    --is_skip="$is_skip" \
    --use_latent_smoothing="$use_latent_smoothing"