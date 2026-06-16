# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch
import h5py

def convert_to_dict(x):
    # x is h5py.File or NpzFile
    ans = {}
    if isinstance(x, dict):
        ans = x
    elif isinstance(x, h5py.File):
        for key in x:
            ans[key] = x[key][()]
    elif isinstance(x, np.lib.npyio.NpzFile):
        for key in x.files:
            ans[key] = x[key]
    else:
        raise ValueError(f"Unknown type {type(x)}")
    return ans

def load_data(fn):
    if fn.endswith('.npz'):
        data = np.load(fn)
    elif fn.endswith('.h5'):
        data = h5py.File(fn, 'r')
    else:
        raise ValueError(f"Unknown extension {fn}")
    return convert_to_dict(data)

def is_norm_to_sphere(sample, is_norm=0):
    xyz_offset = -sample["norm_center"]
    xyz_scale = 1/sample["norm_scale"]

    xyz = sample['point']
    if is_norm == 0:
        xyz = (xyz + xyz_offset) * xyz_scale

    center = xyz.mean(axis=0)
    scale = np.linalg.norm(xyz - center, axis=-1).max()
    info = {"scale": scale.item(), "center": center.tolist()}
    if scale < 1.01 and scale > 0.99 and np.max(np.abs(center)) < 0.01:
        return True, info
    return False, info


class QuadMapping:

    def __init__(self, input_mesh_file_name: str, resolution: int = 256):
        import point_cloud_utils as pcu
        import trimesh
        import xatlas

        self.resolution = resolution

        mesh = trimesh.load(input_mesh_file_name)
        mesh_v, mesh_f = mesh.vertices, mesh.faces

        vmapping, indices, uvs = xatlas.parametrize(mesh_v, mesh_f)
        self.vmapping = vmapping
        self.indices = indices
        self.uvs = uvs
        self.mesh_v = mesh_v

        x = np.linspace(0, 1, resolution)
        xx, yy = np.meshgrid(x, x, indexing='xy')
        grid_points = np.stack([xx, yy, np.ones_like(xx, dtype=np.float32)], axis=-1).reshape(-1, 3)

        uvs_points = np.append(uvs, np.zeros((uvs.shape[0], 1)), axis=1)
        d, fi, bc = pcu.closest_points_on_mesh(grid_points, uvs_points, indices)
        self.mask = np.all(~np.isnan(bc), axis=1) & (d < 1+1.0e-5)
        self.projection_points = pcu.interpolate_barycentric_coords(indices, fi[self.mask], bc[self.mask], mesh_v[vmapping])

    def get_projection_points(self):
        return self.projection_points

    def write_glb(self, pointcolors: np.array, output_glb_file_name: str):
        import trimesh
        from PIL import Image

        if pointcolors.shape[-1] == 4:
            pointcolors = pointcolors[:, :3]
        texture = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8).reshape(-1, 3)
        texture[self.mask] = pointcolors
        texture = texture.reshape(self.resolution, self.resolution, 3)
        texture = np.flipud(texture)
        img = Image.fromarray(texture)

        mesh = trimesh.Trimesh(
            vertices=self.mesh_v[self.vmapping],
            faces=self.indices,
            visual=trimesh.visual.TextureVisuals(uv=self.uvs, image=img),
            process=False,
        )
        out_dir = os.path.dirname(output_glb_file_name)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        mesh.export(output_glb_file_name)


def calc_fps_online(xyz, n_fps, device=torch.device("cpu")):
    from torch_cluster import fps

    if isinstance(xyz, torch.Tensor):
        pos = xyz.to(device)
    else:
        pos = torch.from_numpy(xyz).to(device)
    batch = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
    ratio = n_fps / pos.shape[0]
    fps_idx = fps(pos, batch, ratio=ratio, random_start=False).cpu().numpy().astype(np.int32)[:n_fps]
    if fps_idx.shape[0] < n_fps:
        n_rest = n_fps - fps_idx.shape[0]
        rest_idx = np.setdiff1d(np.arange(pos.shape[0]), fps_idx)
        fps_idx = np.concatenate([fps_idx, np.random.choice(rest_idx, n_rest, replace=False)])
    assert not np.isnan(fps_idx).any() and not np.isinf(fps_idx).any()
    return fps_idx


@torch.no_grad()
def sample(model, condition_params, device, batch_size, seeds=0, args=None):
    from squadgen.network.models_sit import StackedRandomGenerator

    batch_seeds = torch.arange(batch_size*seeds, batch_size*(seeds+1), device=device)
    rnd = StackedRandomGenerator(device, batch_seeds)
    latents = rnd.randn([batch_size, model.n_latents, model.channels], device=device)

    print("seeds", batch_seeds)
    print("latents", latents.shape, latents.view(-1)[:10])

    transport_sampler = args.sit_transport_sampler
    sample_fn = transport_sampler.sample_ode()
    model_fn = model.forward
    sample_model_kwargs = {
        "condition_params": condition_params,
    }
    samples = sample_fn(latents, model_fn, **sample_model_kwargs)[-1]
    return model.denormalize_latents(samples)


@torch.no_grad()
def spatial_smoothing(pts, latents, k, ring_size, smooth_iter=10, use_taubin_smoothing=True):
    batch_size, n_points, latent_dim = latents.shape
    iter_num = smooth_iter if use_taubin_smoothing else (smooth_iter // 2) * 2
    sq_dist_matrix = torch.cdist(pts, pts, p=2) ** 2
    sq_distances, indices = torch.topk(sq_dist_matrix, k=k, dim=2, largest=False, sorted=True)
    min_sq_dis = sq_distances[:, :, 1]
    squared_length_scale = (ring_size**2) * min_sq_dis
    weights = torch.exp(-sq_distances / (2 * squared_length_scale.view(batch_size, n_points, 1)))
    normalized_weights = weights / torch.sum(weights, dim=2, keepdim=True)
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, latent_dim)
    weights_reshaped = normalized_weights.unsqueeze(-1)

    for i in range(iter_num):
        neighbor_noise_all = torch.gather(latents.unsqueeze(2).expand(batch_size, n_points, k, latent_dim), dim=1, index=indices_expanded)
        weighted_noise = neighbor_noise_all * weights_reshaped
        if use_taubin_smoothing:
            taubin_weight = 0.4507499669 if i % 2 == 0 else -0.4720265626
            latents = taubin_weight * torch.sum(weighted_noise, dim=2) + (1 - taubin_weight) * latents
        else:
            latents = torch.sum(weighted_noise, dim=2)

    return latents


@torch.no_grad()
def sample_with_spatial_smoothing(
    model,
    condition_params,
    device,
    batch_size,
    batch_fps_points,
    batch_fps_normals,
    seeds=0,
    args=None,
    k=32,
    ring_size=2,
    normal_weight=0.1,
):
    from squadgen.network.models_sit import StackedRandomGenerator

    batch_seeds = torch.arange(batch_size * seeds, batch_size * (seeds + 1), device=device)
    rnd = StackedRandomGenerator(device, batch_seeds)
    latents = rnd.randn([batch_size, model.n_latents, model.channels], device=device)

    k = max(2, k)
    pts = torch.cat([batch_fps_points, normal_weight * batch_fps_normals], dim=-1)
    latents = spatial_smoothing(pts, latents, k, ring_size, smooth_iter=10, use_taubin_smoothing=True)

    print("seeds", batch_seeds)
    print("latents", latents.shape, latents.view(-1)[:10])
    transport_sampler = args.sit_transport_sampler
    sample_fn = transport_sampler.sample_ode()
    model_fn = model.forward
    sample_model_kwargs = {
        "condition_params": condition_params,
    }
    samples = sample_fn(latents, model_fn, **sample_model_kwargs)[-1]
    return model.denormalize_latents(samples)


def _startswith_any(value, prefixes):
    return value.startswith(tuple(prefixes))


def create_batch_from_data(data, n_fps, n_sample_surface=16384, n_query=50000):
    from squadgen.network.utils import rotate_points_pca_mat

    data = {k: torch.from_numpy(v).to(torch.float32) for k, v in data.items()}

    xyz = data["point"]
    normal = data["normal"]

    center = xyz.mean(axis=0)
    scale = torch.max(torch.norm(xyz - center, dim=1))
    if scale < 1.01 and scale > 0.99 and torch.max(torch.abs(center)) < 0.01:
        norm_center = torch.zeros(3, dtype=torch.float32, device=xyz.device)
        norm_scale = torch.ones(1, dtype=torch.float32, device=xyz.device)
    elif "norm_center" in data:
        norm_center = data["norm_center"]
        norm_scale = data["norm_scale"]
    else:
        norm_center = torch.mean(xyz, dim=0)
        norm_scale = torch.max(torch.norm(xyz - norm_center, dim=1)) * 0.999

    xyz = (xyz - norm_center) / norm_scale

    def get_fps_idx(fps_num):
        if f"fps_idx_{fps_num}_0" in data:
            return data[f"fps_idx_{fps_num}_0"].to(torch.long)
        return calc_fps_online(xyz, fps_num)

    fps_idx = get_fps_idx(n_fps)
    fps_idx_context = get_fps_idx(4 * n_fps)

    generator = torch.Generator().manual_seed(0)

    def sample_surface_uniform(n, n_all=None):
        if n_all is None:
            n_all = xyz.shape[0]
        if n >= n_all:
            idx = torch.arange(0, n_all, dtype=torch.long)
            rest = n - n_all
            if rest > 0:
                idx = torch.cat([idx, torch.randint(0, n_all, (rest, ), generator=generator)], dim=0)
        else:
            idx = torch.randint(0, n_all, (n, ), generator=generator)
        return idx

    idx = sample_surface_uniform(n_sample_surface)
    idx_query = sample_surface_uniform(n_query)
    quadsize_mean = data["quadsize"].mean() if "quadsize" in data else torch.tensor(0.0, device=xyz.device)

    ans = {
        "xyz": xyz[idx],
        "normal": normal[idx],
        "order": xyz[idx],
        f"xyz_fps_{n_fps}": xyz[fps_idx],
        f"normal_fps_{n_fps}": normal[fps_idx],
        f"order_fps_{n_fps}": xyz[fps_idx],
        f"xyz_fps_{4 * n_fps}": xyz[fps_idx_context],
        f"normal_fps_{4 * n_fps}": normal[fps_idx_context],
        f"order_fps_{4 * n_fps}": xyz[fps_idx_context],
        "xyz_query": xyz[idx_query],
        "quadsize_mean": quadsize_mean.unsqueeze(0),
    }

    def rotate_pca(ans):
        ans_new = {}
        if "pca_mat" not in data:
            pca_mat = rotate_points_pca_mat(ans["xyz"].numpy())
            pca_mat = torch.from_numpy(pca_mat).to(torch.float32)
        else:
            pca_mat = data["pca_mat"]
        pca_mat = pca_mat.to(torch.float32)
        for k in ans:
            if _startswith_any(k, ["xyz", "normal", "offset", "offset1", "offset2", "offset3", "offsetb", "offsetc", "offsetd", "foffseta", "foffsetb", "foffsetc", "foffsetd", "doffseta", "doffsetb", "doffsetc"]):
                ans_new[k] = torch.matmul(ans[k], pca_mat)
            else:
                ans_new[k] = ans[k]
        return ans_new, pca_mat

    ans, pca_mat = rotate_pca(ans)
    batch = {k: v.unsqueeze(0) for k, v in ans.items()}
    batch["batch_size"] = 1

    invT = np.zeros((3, 4), dtype=np.float32)
    invT[:, :3] = pca_mat.cpu().numpy() * norm_scale.cpu().numpy()
    invT[:, 3] = norm_center.cpu().numpy()
    batch["invT"] = [invT]
    return batch


def transform_to_original(xyz, invT):
    if isinstance(xyz, torch.Tensor):
        device = xyz.device
        xyz_np = xyz.cpu().numpy()
        ans = np.dot(invT[:, :3], xyz_np.T).T + invT[:, 3]
        return torch.from_numpy(ans).to(device)
    return np.dot(invT[:, :3], xyz.T).T + invT[:, 3]


def sample_points_on_mesh(input, output, n_fps_list=(1024, 2048, 4096)):
    import trimesh

    npz_fn = input + ".npz"
    if os.path.exists(npz_fn):
        ans = np.load(npz_fn)
        np.savez(output, **ans)
        print(f"Load {npz_fn} and save to {output}")
        return

    mesh = trimesh.load(input)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    point, face_idx = trimesh.sample.sample_surface(mesh, 50000, seed=0)
    normal = mesh.face_normals[face_idx]

    ans = {}
    for n_fps in n_fps_list:
        ans[f"fps_idx_{n_fps}_0"] = calc_fps_online(point, n_fps, device=device)

    ans["point"] = point
    ans["normal"] = normal
    np.savez(output, **ans)


def remesh_model(input_file, output_file, percentage=0.4, feature_angle=10.0):
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_file)
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(percentage), featuredeg=feature_angle)
    ms.save_current_mesh(output_file)


def get_triangle_centers(file_path, output_file_path, percentage=0.2, feature_angle=10.0, is_return_mesh_vertices=False):
    import trimesh

    remesh_model(file_path, output_file_path, percentage=percentage, feature_angle=feature_angle)
    if os.path.exists(output_file_path):
        mesh = trimesh.load_mesh(output_file_path)
        pts = mesh.vertices[mesh.faces].mean(axis=1)
        if is_return_mesh_vertices:
            return {"centers": pts, "vertices": mesh.vertices}
        return pts
    assert False


def de_norm_pca(data, invT):
    if isinstance(invT, torch.Tensor):
        invT = invT.cpu().numpy()
    ans = {}
    for k in data:
        t = data[k]
        if _startswith_any(k, ["xyz"]):
            t = np.dot(invT[:, :3], t.T).T + invT[:, 3]
            ans[k] = t
            print(f"de_norm_pca xyz: {k}: {t.min()},{t.max()},{t.mean()}")
        elif _startswith_any(k, ["offset", "doffset", "foffset", "dcdfgrad", "cdfgrad"]):
            t = np.dot(invT[:, :3], t.T).T
            ans[k] = t
            print(f"de_norm_pca offset: {k}: {t.min()},{t.max()},{t.mean()}")
        else:
            ans[k] = t
            print(f"de_norm_pca None: {k}: {t.min()},{t.max()},{t.mean()}")
    return ans


def get_extract_mesh_query_points(raw_mesh_fn, subdiv_fn):
    try:
        tmp = get_triangle_centers(raw_mesh_fn, subdiv_fn)
        tmp = np.array(tmp)
        tmp = torch.from_numpy(tmp).to(torch.float32)
        print(subdiv_fn, tmp.shape)
    except Exception as e:
        import traceback

        print(f"Error: {e}: {raw_mesh_fn}")
        print(traceback.format_exc())
        tmp = np.zeros((1, 3))
        tmp = torch.from_numpy(tmp).to(torch.float32)
        save_path = os.path.dirname(subdiv_fn)
        with open(os.path.join(save_path, "get_extract_mesh_query_points_error.txt"), "w") as f:
            f.write(f"Error: {e}: {raw_mesh_fn}\n")
            f.write(traceback.format_exc())
    return tmp


# ----------------------------------------------------------------------------
# Vector operations
# ----------------------------------------------------------------------------


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)


def length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)
