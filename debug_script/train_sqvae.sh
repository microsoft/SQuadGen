#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

# base_exp_name=patch0928_0928_quad_1024x32_patch_cdf_v4
gpu_type=${GPU_TYPE:-a100}
n_gpu=${N_GPU:-1}
batch_size=16
# res=-1
batch_size=2
res=4096
base_exp_name=patch0928_0928_quad_1024x32_patch_cdf_v4_res${res}
gradient_accumulate_steps=${GRADIENT_ACCUMULATE_STEPS:-1}
n_nodes=1
lr=${LR:-0.0001}
warmup_epochs=${WARMUP_EPOCHS:-5}
effective_batch_size=$((batch_size * gradient_accumulate_steps * n_gpu * n_nodes))

exp_name=${EXP_NAME:-${base_exp_name}_${gpu_type}_bs${effective_batch_size}_ga${gradient_accumulate_steps}_lr${lr}}
dataset_folder=${DATASET_FOLDER:-/mnt/ykongcus/my_project/remesh/dataset_color/_dataset/250928_230k_multibins_cc}
data_filter_name=${DATA_FILTER_NAME-}
output_dir=${OUTPUT_DIR:-results2/train_sqvae}

vae_config=${VAE_CONFIG:-squadgen/network/config/sqvae_geomae.yaml}
pretrained_geom_vae_pth=${PRETRAINED_GEOM_VAE_PTH:-/mnt/output/my_project/remesh/experiments/vecset/largebalance_0207_geom_ae_1024x64_old_normal_fps_resume_r2_a100_40g_basic_bs128_ga1_lr0.0001_of0_n4/checkpoint-300.pth}
resume_from_another=${RESUME_FROM_ANOTHER:-/mnt/output/my_project/remesh/experiments/vecset/patch0928_0928_quad_1024x32_patch_cdf_v4_a100_bs64_ga1_lr0.0001_of0/nomask/checkpoint-375.pth}
val_num=${VAL_NUM:-4}

mkdir -p "$output_dir"

torchrun --nproc_per_node="$n_gpu" -m squadgen.main_ae \
    --name="$exp_name" \
    --batch_size="$batch_size" \
    --continue_training=1 \
    --epochs=600 \
    --dataset_folder="$dataset_folder" \
    --output_dir="$output_dir" \
    --report_to=tensorboard \
    --model_config="$vae_config" \
    --accum_iter="$gradient_accumulate_steps" \
    --lr="$lr" \
    --val_num="$val_num" \
    --data_filter_name="$data_filter_name" \
    --val_model_per_epoch=1 \
    --num_workers=-1 \
    --warmup_epochs="$warmup_epochs" \
    --pretrained_geom_vae_pth="$pretrained_geom_vae_pth" \
    --use_geom_cond_mean=1 \
    --training_data_repeat_times=1 \
    --resume_from_another="$resume_from_another" \
    --mix_precision=1 \
    --mix_precision_dtype=bf16 \
    --clip_grad=1 \
    --labeling_fn=labeling_all.json \
    --training_epoch_len=450000 \
    --n_sample_surface_query=8192 \
    --dist_hook=0 \
    --is_render_image=1 \
    --val_dataset_type_list=test,train \
    --res="$res"
