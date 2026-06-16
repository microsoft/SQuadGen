import argparse
import json
import os
import shutil
import time

import torch
from omegaconf import OmegaConf

os.environ["PYOPENGL_PLATFORM"] = "egl"

import squadgen.network.models_vae_joint as models_vae_joint
from squadgen.util.misc import load_model_from_file
from squadgen.util.util import create_batch_from_data, load_data, sample_points_on_mesh, transform_to_original
from squadgen.network.utils import extract_mesh_from_udf, get_errmap_viz, save_ply


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
    return ae


def load_input_list(args):
    with open(args.input_filelist, "r") as f:
        input_list = json.load(f)

    end = len(input_list) if args.end == -1 else min(args.end, len(input_list))
    return input_list[args.start:end]


def prepare_mesh_batch(file_fn, outdir, n_fps, device, debug):
    if not file_fn.endswith((".obj", ".ply")):
        raise ValueError(f"file {file_fn} is not .obj or .ply")

    os.makedirs(outdir, exist_ok=True)
    ext = os.path.splitext(file_fn)[1]
    input_mesh_copy = os.path.join(outdir, "gt_mesh" + ext)
    shutil.copy2(file_fn, input_mesh_copy)

    sampled_points_fn = os.path.join(outdir, "sampled_points.npz")
    sample_points_on_mesh(file_fn, sampled_points_fn, n_fps_list=[n_fps, 4 * n_fps])
    data = load_data(sampled_points_fn)
    assert "point" in data and "normal" in data, f"file {file_fn} does not contain point or normal"

    batch = create_batch_from_data(data, n_fps)
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    if debug:
        xyz_fps_save = batch[f"xyz_fps_{n_fps}"][0]
        normal_fps_save = batch[f"normal_fps_{n_fps}"][0]
        xyz_save = batch["xyz"][0]
        normal_save = batch["normal"][0]
        save_ply(xyz_fps_save, normal_fps_save, path=outdir, fn="fps.ply", data_type="normal")
        save_ply(xyz_save, normal_save, path=outdir, fn="kv.ply", data_type="normal")
        save_ply(transform_to_original(xyz_fps_save, batch["invT"][0]), normal_fps_save, path=outdir, fn="fps_original_space.ply", data_type="normal")
        save_ply(transform_to_original(xyz_save, batch["invT"][0]), normal_save, path=outdir, fn="kv_original_space.ply", data_type="normal")

    return batch


def reconstruct_one(args, ae, file_fn, outdir, device):
    n_fps = ae.cond_vae.num_latents
    batch = prepare_mesh_batch(file_fn, outdir, n_fps, device, args.debug)

    time_start = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.mix_precision, dtype=torch.bfloat16):
            latent_dict = ae.encode(batch, is_mode=args.is_mode, is_run_sqvae=False)
            x_last = ae.cond_vae.return_last_features(latent_dict["lat_cond"])
    x_last = x_last.to(torch.float32)
    print(f"encode geometry time: {time.time() - time_start:.2f}")

    try:
        from DualMeshUDF import write_obj

        with torch.enable_grad():
            mesh_v, mesh_f = extract_mesh_from_udf(ae, x_last)
        mesh_name = os.path.join(outdir, "mesh_recon_by_geom_ae.obj")
        write_obj(mesh_name, mesh_v, mesh_f)
        print(f"Saved {mesh_name}")
    except Exception as e:
        print(f"Error: {e}")

    with torch.no_grad():
        out_geom = ae.cond_vae.query_last_features(x_last, batch["xyz_query"])
    score_geom = {
        "udf_L1": torch.mean(out_geom["udf"]).to(torch.float32).cpu().numpy().item(),
        "udf_L2": torch.mean(out_geom["udf"] ** 2).to(torch.float32).cpu().numpy().item(),
    }
    with open(os.path.join(outdir, "geom_ae_recon_score.json"), "w") as f:
        json.dump(score_geom, f, indent=4)

    if args.debug:
        threshold = 1.0
        idx_select = out_geom["udf"][0, :, 0] < threshold
        xyz_select = batch["xyz_query"][0, idx_select, :]
        ans_recon_select = out_geom["udf"][0, idx_select, :]
        pred_color = get_errmap_viz(ans_recon_select, err_min=0.0, err_max=min(threshold, 1.0))
        save_ply(xyz_select, pred_color, outdir, "points_udf.ply", data_type="color4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filelist", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--ae_config", required=True, type=str, metavar="MODEL", help="Name of autoencoder")
    parser.add_argument("--ae_pth", required=True, help="Autoencoder checkpoint")
    parser.add_argument("--is_mode", default=1, type=int)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--mix_precision", default=False, type=str2bool)
    parser.add_argument("--debug", default=0, type=int)
    parser.add_argument("--res", default=4096, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--is_skip", default=1, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.results_dir = os.path.join(args.results_dir, args.name)
    os.makedirs(args.results_dir, exist_ok=True)

    ae = load_ae(args, device)
    input_list = load_input_list(args)

    for file_fn in input_list:
        ext = os.path.splitext(file_fn)[1]
        outdir = os.path.join(args.results_dir, os.path.basename(file_fn).replace(ext, ""))
        if args.is_skip == 1 and os.path.exists(os.path.join(outdir, "mesh_recon_by_geom_ae.obj")):
            print(f"Skip {file_fn}, reconstruction already exists")
            continue
        try:
            reconstruct_one(args, ae, file_fn, outdir, device)
        except Exception as e:
            print(f"Error: {e} on file {file_fn}")


if __name__ == "__main__":
    main()
