import argparse
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np
import torch
from omegaconf import OmegaConf

os.environ["PYOPENGL_PLATFORM"] = "egl"

import squadgen.network.models_sit as models_sit
import squadgen.network.models_vae_joint as models_vae_joint
from squadgen.util.misc import load_model_from_file
from squadgen.util.util import (
    QuadMapping,
    create_batch_from_data,
    de_norm_pca,
    get_extract_mesh_query_points,
    load_data,
    sample,
    sample_points_on_mesh,
    sample_with_spatial_smoothing,
    transform_to_original,
)
from squadgen.network.transport import Sampler, create_transport
from squadgen.network.utils import convert_gcolor_to_gcolormap, save_ply


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def load_input_list(args):
    with open(args.input_filelist, "r") as f:
        input_list = json.load(f)
    end = len(input_list) if args.end == -1 else min(args.end, len(input_list))
    return input_list[args.start:end]


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
    ae.latent_vae.out_keys_list = [x for x in ae.latent_vae.out_keys_list if not x.endswith("_edge")]
    return ae, ae_config


def load_sqdiffuse(args, ae_config, device):
    model_config = OmegaConf.load(args.model_config)

    model = models_sit.create_sqdiffuse_from_config(model_config.model.params, ae_config.model.params)
    model.to(device)
    model.load_state_dict(load_model_from_file(args.model_pth)["model"])
    model.eval()
    model.requires_grad_(False)

    transport = create_transport(args.sit_path_type, args.sit_prediction, args.sit_loss_weight)
    args.sit_transport = transport
    args.sit_transport_sampler = Sampler(transport)

    print(f"target_mu: {model.target_mu}, target_std: {model.target_std}")
    return model


def prepare_mesh_inputs(args, file_fn, outdir, n_fps, device):
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

    subdiv_mesh_fn = input_mesh_copy.replace(ext, "_subdiv" + ext)
    points_extract_mesh = get_extract_mesh_query_points(input_mesh_copy, subdiv_mesh_fn).numpy()
    invT = batch["invT"][0]
    points_extract_mesh = np.dot(np.linalg.inv(invT[:, :3]), (points_extract_mesh - invT[:, 3]).T).T
    points_extract_mesh = torch.from_numpy(points_extract_mesh).to(device).unsqueeze(0)
    print("points_extract_mesh", points_extract_mesh.shape)

    if args.debug:
        xyz_fps_save = batch[f"xyz_fps_{n_fps}"][0]
        normal_fps_save = batch[f"normal_fps_{n_fps}"][0]
        xyz_save = batch["xyz"][0]
        normal_save = batch["normal"][0]
        save_ply(xyz_fps_save, normal_fps_save, path=outdir, fn="fps.ply", data_type="normal")
        save_ply(xyz_save, normal_save, path=outdir, fn="kv.ply", data_type="normal")
        save_ply(transform_to_original(xyz_fps_save, invT), normal_fps_save, path=outdir, fn="fps_original_space.ply", data_type="normal")
        save_ply(transform_to_original(xyz_save, invT), normal_save, path=outdir, fn="kv_original_space.ply", data_type="normal")

    time_start = time.time()
    qm = QuadMapping(file_fn, args.texture_res)
    print(f"QuadMapping time: {time.time() - time_start:.2f}")

    projection_points = qm.get_projection_points()
    projection_points = np.dot(np.linalg.inv(invT[:, :3]), (projection_points - invT[:, 3]).T).T
    projection_points = torch.tensor(projection_points, dtype=torch.float32)
    assert torch.isnan(projection_points).sum() == 0, torch.isnan(projection_points).sum()
    projection_points_texture = projection_points.unsqueeze(0).to(device)
    print(f"projection_points_texture shape: {projection_points_texture.shape}")

    # _, raw_mesh_info = calc_info(input_mesh_copy)
    raw_mesh_info = None
    # if raw_mesh_info is not None:
    #     with open(os.path.join(outdir, "raw_mesh_info.json"), "w") as f:
    #         json.dump(raw_mesh_info, f, indent=4)
    #     if len(raw_mesh_info) > 1:
    #         print(f"{file_fn} has more than 1 component")
    #     raw_mesh_info = raw_mesh_info[0]
    #     if raw_mesh_info["is_nonmanifold"] == 1:
    #         print(f"{file_fn} is non-manifold")

    return batch, points_extract_mesh, projection_points_texture, qm


@torch.no_grad()
def infer_one(args, ae, model, file_fn, outdir, device):
    n_fps = ae.latent_vae.num_latents
    batch, points_extract_mesh, projection_points_texture, qm = prepare_mesh_inputs(args, file_fn, outdir, n_fps, device)
    batch_size = batch["batch_size"]
    assert batch_size == 1, f"batch size should be 1, but got {batch_size}"

    time_start = time.time()
    with torch.cuda.amp.autocast(enabled=args.mix_precision, dtype=torch.bfloat16):
        tmp = model.get_latent_and_condition(batch, ae, is_mode=args.is_mode, is_run_sqvae=False)
    print(f"get_latent_and_condition time: {time.time() - time_start:.2f}")

    condition_params = tmp["condition_params"]
    timings = {}
    for gen_idx in range(args.n_gen):
        seeds = gen_idx
        time_start = time.time()
        with torch.cuda.amp.autocast(enabled=args.mix_precision, dtype=torch.bfloat16):
            if args.use_latent_smoothing == 0:
                lat = sample(model, condition_params, device, batch_size=batch["xyz"].shape[0], seeds=seeds, args=args)
            else:
                lat = sample_with_spatial_smoothing(
                    model,
                    condition_params,
                    device,
                    batch_size=batch["xyz"].shape[0],
                    batch_fps_points=batch[f"xyz_fps_{n_fps}"],
                    batch_fps_normals=batch[f"normal_fps_{n_fps}"],
                    seeds=seeds,
                    args=args,
                )
        lat = lat.to(torch.float32)
        sample_time = time.time() - time_start
        print(f"sample time: {sample_time:.2f}")

        decode_geom_start = time.time()
        out = decode_in_chunks(ae, lat, points_extract_mesh, args.n_max_query, args.mix_precision)
        decode_geom_time = time.time() - decode_geom_start

        ans_extract = {key: value.to(torch.float32) for key, value in out.items()}
        ans_extract["xyz"] = points_extract_mesh
        ans_extract = {key: value[0].cpu().numpy() for key, value in ans_extract.items()}
        ans_extract["invT"] = batch["invT"][0]
        ans_extract = de_norm_pca(ans_extract, batch["invT"][0])
        ans_extract.pop("gdcdf", None)
        for key in ["xyz", "offset1", "offset2", "offset3", "offsetb", "offsetd"]:
            ans_extract.pop(key, None)

        decode_tex_start = time.time()
        out_tex = decode_in_chunks(ae, lat, projection_points_texture, args.n_max_query, args.mix_precision)
        decode_tex_time = time.time() - decode_tex_start
        color_all = out["dcdf"]
        color_all_texture = out_tex["cdf"]
        print(color_all.shape)

        for idx in range(batch_size):
            gen_outdir = os.path.join(outdir, f"gen_{gen_idx:03d}")
            os.makedirs(gen_outdir, exist_ok=True)

            color = color_all[idx]
            color_viz = convert_gcolor_to_gcolormap(color)
            projection_points = points_extract_mesh[idx]
            print(projection_points.shape, color.shape)

            if args.debug:
                save_ply(projection_points, color_viz, path=gen_outdir, fn=f"gen{gen_idx}_ori_viz.ply", data_type="color4")
                torch.save(lat[idx], os.path.join(gen_outdir, f"gen{gen_idx}_lat.pt"))
                save_ply(transform_to_original(projection_points, batch["invT"][0]), color_viz, path=gen_outdir, fn=f"gen{gen_idx}_ori_viz_original_space.ply", data_type="color4")

            np.savez(os.path.join(gen_outdir, "extract_mesh.npz"), **ans_extract)

            output_glb_fn = os.path.join(gen_outdir, f"gen{gen_idx}.glb")

            color_tex = convert_gcolor_to_gcolormap(color_all_texture[idx])
            color_tex = (color_tex.clamp(0, 1) * 255).to(torch.uint8)
            qm.write_glb(color_tex.cpu().numpy(), output_glb_fn)

            write_generation_info(gen_outdir, gen_idx, condition_params)

            timings[gen_idx] = {
                "sample_time": sample_time,
                "decode_geom_time": decode_geom_time,
                "decode_tex_time": decode_tex_time,
            }
            write_timing(gen_outdir, timings[gen_idx])

    run_quad_postprocess(file_fn, outdir, args.n_gen, timings)


def run_quad_postprocess(file_fn, outdir, n_gen, timings=None):
    """Run QuadExtraction + QuadQuality on each gen_*/extract_mesh.npz.

    Mirrors debug_script/extract_quad_from_cdf.sh; writes extracted_quad.ply and
    the loop simplicity JSON (FratioN / EratioN) next to it. If `timings` is
    provided, append the post-processing wall times into timing.json.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    quadtools_build = os.path.join(repo_root, "QuadTools", "build")
    quad_extraction = os.path.join(quadtools_build, "QuadExtraction")
    quad_quality = os.path.join(quadtools_build, "QuadQuality")
    for bin_path in (quad_extraction, quad_quality):
        if not (os.path.isfile(bin_path) and os.access(bin_path, os.X_OK)):
            print(f"Skip quad postprocess: {bin_path} not found or not executable. Build QuadTools first.")
            return

    ext = os.path.splitext(file_fn)[1]
    subdivided = os.path.join(outdir, "gt_mesh_subdiv" + ext)
    if not os.path.isfile(subdivided):
        print(f"Skip quad postprocess: missing {subdivided}")
        return

    # Hyper-parameters mirror debug_script/extract_quad_from_cdf.sh; verbose=true
    # so QuadExtraction prints its internal compute time.
    extract_flags = [
        "--ringsize=8", "--verbose=true",
        "-a", "150", "--div=0",
        "-t", "0.1", "-s", "3", "-r", "15",
    ]

    for gen_idx in range(n_gen):
        gen_dir = os.path.join(outdir, f"gen_{gen_idx:03d}")
        features = os.path.join(gen_dir, "extract_mesh.npz")
        if not os.path.isfile(features):
            continue
        output_quad = os.path.join(gen_dir, "extracted_quad.ply")

        print(f"\nExtracting: {output_quad}")
        extract_start = time.time()
        proc = subprocess.run(
            [quad_extraction, "-i", subdivided, "-f", features, "-o", output_quad, *extract_flags],
            capture_output=True, text=True,
        )
        extract_time = time.time() - extract_start
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="")
        extract_compute_time = parse_quad_extraction_time(proc.stdout)
        if proc.returncode != 0 or not os.path.isfile(output_quad):
            print(f"QuadExtraction failed for {features}")
            if timings is not None:
                gen_timing = timings.setdefault(gen_idx, {})
                gen_timing["quad_extraction_time"] = extract_time
                if extract_compute_time is not None:
                    gen_timing["quad_extraction_compute_time"] = extract_compute_time
                write_timing(gen_dir, gen_timing)
            continue

        print(f"Loop simplicity: {output_quad}")
        quality_start = time.time()
        subprocess.run([quad_quality, "-i", output_quad, "-j", gen_dir, "-v"])
        quality_time = time.time() - quality_start

        if timings is not None:
            gen_timing = timings.setdefault(gen_idx, {})
            gen_timing["quad_extraction_time"] = extract_time
            if extract_compute_time is not None:
                gen_timing["quad_extraction_compute_time"] = extract_compute_time
            gen_timing["quad_quality_time"] = quality_time
            write_timing(gen_dir, gen_timing)


_QUAD_EXTRACT_TIME_RE = re.compile(r"quad extraction time:\s*([\d.]+)\s*ms", re.IGNORECASE)


def parse_quad_extraction_time(stdout: str):
    """Return the compute-only time (seconds) printed by QuadExtraction --verbose, or None."""
    if not stdout:
        return None
    m = _QUAD_EXTRACT_TIME_RE.search(stdout)
    if not m:
        return None
    return float(m.group(1)) / 1000.0


def write_timing(gen_dir, gen_timing):
    timing_fn = os.path.join(gen_dir, "timing.json")
    payload = dict(gen_timing)
    # total_time excludes *_compute_time fields (already counted in their wall-time pair).
    payload["total_time"] = sum(
        v for k, v in payload.items()
        if isinstance(v, (int, float)) and not k.endswith("_compute_time")
    )
    with open(timing_fn, "w") as f:
        json.dump(payload, f, indent=4)


def decode_in_chunks(ae, lat, projection_points_all, n_max_query, mix_precision):
    with torch.cuda.amp.autocast(enabled=mix_precision, dtype=torch.bfloat16):
        projection_points_split = projection_points_all.split(n_max_query, dim=1)
        out_list = []
        for projection_points in projection_points_split:
            out = ae.latent_vae.decode_batch(lat, {"xyz_query": projection_points})
            out_list.append(out)
        return {key: torch.cat([item[key] for item in out_list], dim=1) for key in out_list[0]}


def write_generation_info(outdir, gen_idx, condition_params):
    info_fn = os.path.join(outdir, f"gen{gen_idx}_info.json")
    gt_info = {}
    if condition_params["condition"] is not None:
        gt_info["condition"] = {
            "shape": condition_params["condition"].shape,
            "max": condition_params["condition"].max().item(),
            "min": condition_params["condition"].min().item(),
            "mean": condition_params["condition"].mean().item(),
            "std": condition_params["condition"].std().item(),
            "abstract": condition_params["condition"].reshape(-1)[:5].to(torch.float32).cpu().numpy().tolist(),
        }
    with open(info_fn, "w") as f:
        json.dump(gt_info, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filelist", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model_config", required=True, type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--model_pth", required=True, type=str)
    parser.add_argument("--ae_config", required=True, type=str, metavar="MODEL", help="Name of autoencoder")
    parser.add_argument("--ae_pth", required=True, help="Autoencoder checkpoint")
    parser.add_argument("--sit_path_type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    parser.add_argument("--sit_prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    parser.add_argument("--sit_loss_weight", type=str, default="None", choices=["None", "velocity", "likelihood"])
    parser.add_argument("--is_mode", default=1, type=int)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--n_max_query", default=100000, type=int)
    parser.add_argument("--n_gen", default=1, type=int)
    parser.add_argument("--mix_precision", default=False, type=str2bool)
    parser.add_argument("--debug", default=0, type=int)
    parser.add_argument("--texture_res", type=int, default=512)
    parser.add_argument("--res", default=4096, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--is_skip", default=1, type=int)
    parser.add_argument("--use_latent_smoothing", default=1, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.results_dir = os.path.join(args.results_dir, args.name)
    os.makedirs(args.results_dir, exist_ok=True)

    ae, ae_config = load_ae(args, device)
    model = load_sqdiffuse(args, ae_config, device)
    input_list = load_input_list(args)

    for file_fn in input_list:
        ext = os.path.splitext(file_fn)[1]
        outdir = os.path.join(args.results_dir, os.path.basename(file_fn).replace(ext, ""))
        if args.is_skip == 1 and os.path.exists(os.path.join(outdir, f"gen_{args.n_gen - 1:03d}")):
            print(f"Skip {file_fn}, outdir {outdir} already exists")
            continue
        try:
            infer_one(args, ae, model, file_fn, outdir, device)
        except Exception as e:
            print(f"Error: {e} on file {file_fn}")


if __name__ == "__main__":
    main()
