#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

# base_exp_name=patch0928f230kbins_1209_quad_sqdiffuse_1024x32_sitXL_patch_vae375_res4096_pcscale_v1
gpu_type=${GPU_TYPE:-a100_40g_ib}
n_gpu=${N_GPU:-1}
batch_size=4
res=-1
batch_size=2
res=4096
base_exp_name=patch0928f230kbins_1209_quad_sqdiffuse_res${res}
gradient_accumulate_steps=${GRADIENT_ACCUMULATE_STEPS:-1}
n_nodes=1
lr=${LR:-0.0001}
warmup_epochs=${WARMUP_EPOCHS:-1}
effective_batch_size=$((batch_size * gradient_accumulate_steps * n_gpu * n_nodes))

exp_name=${EXP_NAME:-${base_exp_name}_${gpu_type}_bs${effective_batch_size}_ga${gradient_accumulate_steps}_lr${lr}}
dataset_folder=${DATASET_FOLDER:-/mnt/ykongcus/my_project/remesh/dataset_color/_dataset/250928_230k_multibins_cc}
data_filter_name=${DATA_FILTER_NAME-}
output_dir=${OUTPUT_DIR:-results2/train_sqdiffuse}

ae_config=${AE_CONFIG:-squadgen/network/config/sqvae_geomae.yaml}
ae_pth=${AE_PTH:-/mnt/output/my_project/remesh/experiments/vecset/patch0928_0928_quad_1024x32_patch_cdf_v4_a100_bs64_ga1_lr0.0001_of0/nomask/checkpoint-375.pth}
model_config=${MODEL_CONFIG:-squadgen/network/config/sqdiffuse.yaml}
resume_from_another=${RESUME_FROM_ANOTHER-}
val_num=${VAL_NUM:-2}

mkdir -p "$output_dir"

torchrun --nproc_per_node="$n_gpu" -m squadgen.main_sqdiffuse \
    --name="$exp_name" \
    --batch_size="$batch_size" \
    --continue_training=1 \
    --epochs=600 \
    --dataset_folder="$dataset_folder" \
    --output_dir="$output_dir" \
    --report_to=tensorboard \
    --model_config="$model_config" \
    --accum_iter="$gradient_accumulate_steps" \
    --lr="$lr" \
    --val_num="$val_num" \
    --data_filter_name="$data_filter_name" \
    --val_model_per_epoch=5 \
    --num_workers=-1 \
    --warmup_epochs="$warmup_epochs" \
    --ae_pth="$ae_pth" \
    --ae_config="$ae_config" \
    --is_mode=1 \
    --training_data_repeat_times=1 \
    --resume_from_another="$resume_from_another" \
    --mix_precision=1 \
    --mix_precision_dtype=bf16 \
    --drop_prob=0.1 \
    --sit_path_type=Linear \
    --sit_prediction=velocity \
    --sit_loss_weight=None \
    --use_const_lr=1 \
    --clip_grad=1 \
    --labeling_fn=labeling_all.json \
    --is_render_image=1 \
    --dist_hook=1 \
    --save_model_per_epoch=1 \
    --res="$res" \
    --training_epoch_len=200000 \
    --val_dataset_type_list=test,train
