import argparse
import json
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

os.environ["PYOPENGL_PLATFORM"] = "egl"

import squadgen.network.models_vae_joint as models_vae_joint
from data_tools.test_load_new_format import load_patch_data_reformat
from squadgen.util.misc import load_model_from_file
from squadgen.util.util import transform_to_original
from squadgen.network.utils import convert_gcolor_to_gcolormap, save_ply


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def load_ae(args, device):
    ae_config = OmegaConf.load(args.ae_config)
    ae_config.model.params.latent_vae_params.num_latents = args.res
    ae_config.model.params.cond_vae_params.num_latents = args.res

    ae = models_vae_joint.create_joint_vae_from_config(ae_config.model.params)
    print("Loading autoencoder %s" % args.ae_pth)
    ae.load_state_dict(load_model_from_file(args.ae_pth)["model"])
    ae.to(device)
    ae.eval()
    ae.requires_grad_(False)

    ae.latent_vae.out_keys_list = [
        output_key for output_key in ae.latent_vae.out_keys_list if not output_key.endswith("_edge")
    ]
    return ae


def load_input_list(args):
    if args.input_filelist:
        with open(args.input_filelist, "r") as input_file:
            input_list = json.load(input_file)
    elif args.input:
        input_list = [args.input]
    else:
        raise ValueError("Either --input or --input_filelist must be provided")

    end = len(input_list) if args.end == -1 else min(args.end, len(input_list))
    return input_list[args.start:end]


def create_batch_from_data(data):
    batch = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            batch[key] = torch.from_numpy(value).unsqueeze(0).to(torch.float32)
        else:
            batch[key] = value

    batch["batch_size"] = 1
    batch["invT"] = [data["invT"]]
    return batch


def prepare_color_batch(file_fn, n_fps, device, args):
    if not file_fn.endswith((".npz", ".h5")):
        raise ValueError(f"file {file_fn} is not .npz or .h5")

    load_data_reformat_args = {
        "num_surface": args.num_surface,
        "is_add_noise": 0,
        "fps_return_type": "first",
        "debug": 0,
        "fps_num_list": [n_fps, 4 * n_fps],
    }
    data = load_patch_data_reformat(file_fn, **load_data_reformat_args)
    batch = create_batch_from_data(data)

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def cleanup_generated_outputs(outdir, input_file):
    if not os.path.isdir(outdir):
        return

    generated_names = {os.path.basename(input_file)}
    for file_name in os.listdir(outdir):
        if file_name.endswith(".ply") or file_name.startswith("sqvae_recon_") or file_name in generated_names:
            file_path = os.path.join(outdir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)


def unwrap_ae_output(ae_out):
    if isinstance(ae_out, dict) and "out" in ae_out:
        output = ae_out["out"]
        if isinstance(output, dict) and "out" in output:
            return output["out"]
        return output
    return ae_out


def calc_recon_score(pred, gt):
    pred = pred.to(torch.float32)
    gt = gt.to(torch.float32)
    diff = pred - gt
    return {
        "L1": torch.mean(torch.abs(diff)).detach().cpu().item(),
        "L2": torch.mean(diff ** 2).detach().cpu().item(),
        "max_abs": torch.max(torch.abs(diff)).detach().cpu().item(),
    }


def save_scalar_field_original_space(batch, outdir, field_name, field_value, output_name):
    xyz_query = batch["xyz_query"][0]
    color_viz = convert_gcolor_to_gcolormap(field_value)
    xyz_original = transform_to_original(xyz_query, batch["invT"][0])
    save_ply(xyz_original, color_viz, path=outdir, fn=f"{field_name}_{output_name}_original_space.ply", data_type="color4")


def save_reconstruction_outputs(args, batch, ae_outputs, outdir):
    score = {}

    for output_key, pred_value in ae_outputs.items():
        if not isinstance(pred_value, torch.Tensor):
            continue
        gt_key = f"{output_key}_query"
        if gt_key not in batch:
            continue

        pred = pred_value[0].to(torch.float32)
        gt = batch[gt_key][0].to(torch.float32)

        if pred.shape == gt.shape:
            score[output_key] = calc_recon_score(pred, gt)

        if output_key in {"dcdf", "cdf"} and pred.ndim == 2 and pred.shape[1] == 1:
            save_scalar_field_original_space(batch, outdir, output_key, pred, "recon")
            save_scalar_field_original_space(batch, outdir, output_key, gt, "gt")

    return score


def reconstruct_one(args, ae, file_fn, outdir, device):
    n_fps = ae.latent_vae.num_latents
    os.makedirs(outdir, exist_ok=True)
    cleanup_generated_outputs(outdir, file_fn)

    batch = prepare_color_batch(file_fn, n_fps, device, args)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    encode_start = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.mix_precision, dtype=torch.bfloat16):
            latent = ae.encode(batch, only_trainable=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    encode_time = time.time() - encode_start

    decode_start = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.mix_precision, dtype=torch.bfloat16):
            ae_out = ae.decode(batch, latent, only_trainable=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    decode_time = time.time() - decode_start

    ae_outputs = unwrap_ae_output(ae_out)
    print(f"[sqvae] encoding time: {encode_time:.2f}s, decoding time: {decode_time:.2f}s ({file_fn})")
    scores = save_reconstruction_outputs(args, batch, ae_outputs, outdir)

    with open(os.path.join(outdir, "sqvae_recon_score.json"), "w") as output_file:
        json.dump(scores, output_file, indent=4)

    return encode_time, decode_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--input_filelist", type=str, default="")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--ae_config", required=True, type=str, metavar="MODEL", help="Name of autoencoder")
    parser.add_argument("--ae_pth", required=True, help="Autoencoder checkpoint")
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--mix_precision", default=False, type=str2bool)
    parser.add_argument("--res", default=4096, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--is_skip", default=1, type=int)
    parser.add_argument("--num_surface", default=50000, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.results_dir = os.path.join(args.results_dir, args.name)
    os.makedirs(args.results_dir, exist_ok=True)

    ae = load_ae(args, device)
    input_list = load_input_list(args)

    enc_times, dec_times = [], []
    for file_fn in input_list:
        ext = os.path.splitext(file_fn)[1]
        outdir = os.path.join(args.results_dir, os.path.basename(file_fn).replace(ext, ""))
        if args.is_skip == 1 and os.path.exists(os.path.join(outdir, "sqvae_recon_score.json")):
            print(f"Skip {file_fn}, reconstruction already exists")
            continue
        try:
            encode_time, decode_time = reconstruct_one(args, ae, file_fn, outdir, device)
            enc_times.append(encode_time)
            dec_times.append(decode_time)
        except Exception as err:
            print(f"Error: {err} on file {file_fn}")

    if enc_times:
        n = len(enc_times)
        print(f"[sqvae] processed {n} files, "
              f"avg encoding time: {sum(enc_times) / n:.2f}s, "
              f"avg decoding time: {sum(dec_times) / n:.2f}s")


if __name__ == "__main__":
    main()
