import cv2
from contextlib import contextmanager
import sys

import torch.nn as nn
import torch.nn.functional as F
from ocnn.octree import Octree
import os
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt  

from squadgen.util.render_nvdiffrast import get_rendering_images
from squadgen.util.tet_render import NeuralRender, PerspectiveCamera

KEY_DIM_DICT = {
    "sdf": 1,
    "udf": 1,
    "udf_near": 1,
    "udf_global": 1,
    "normal": 3,
    "ggcolor": 3,
    "color": 3,
    "gcolor": 1,
    "sizing": 1,
    "quadsize": 1,
    "quadsize_mean": 1,
    "offset": 3, "offset1": 3, "offset2": 3, "offset3": 3,
    "offsetb": 3, "offsetc": 3, "offsetd": 3,
    "foffseta": 3, "foffsetb": 3, "foffsetc": 3, "foffsetd": 3,
    "doffseta": 3, "doffsetb": 3, "doffsetc": 3,
    "offset_div_sizing": 3, "offset1_div_sizing": 3, "offset2_div_sizing": 3,
    "xyz": 3,
    "singularpatch": 1,
    "dcdf": 1, "gdcdf": 3, "cdf": 1, "gcdf": 3,
}


def get_one_channel(color):
    color_softmax = torch.nn.functional.softmax(color, dim=1) # [N, 3]
    color_indices = torch.argmax(color_softmax, dim=1)
    color_indices = color_indices.unsqueeze(-1) - 1
    color = color_indices
    return color

def convert_to_rgba(features_cur):
    # features_cur: [N, 3],
    # std::array<unsigned char, 3> white = {255, 255, 255}; -1, (<-1/3)
    # std::array<unsigned char, 3> blue = {0, 162, 232}; 0, (-1/3, 1/3)
    # std::array<unsigned char, 3> black = {64, 64, 64}; 1, (1/3, 1)    
    if features_cur.shape[1] == 3:
        features_cur = get_one_channel(features_cur)
    N = features_cur.shape[0]
    features_cur_rgb = torch.zeros((N, 4), dtype=torch.uint8, device=features_cur.device)
    t = 1./3.
    features_cur_rgb = torch.where(features_cur < -t, torch.tensor([255, 255, 255, 255], dtype=torch.uint8, device=features_cur.device), features_cur_rgb)
    features_cur_rgb = torch.where((features_cur >= -t) & (features_cur < t), torch.tensor([0, 162, 232, 255], dtype=torch.uint8, device=features_cur.device), features_cur_rgb)
    features_cur_rgb = torch.where(features_cur >= t, torch.tensor([64, 64, 64, 255], dtype=torch.uint8, device=features_cur.device), features_cur_rgb)
    return features_cur_rgb

def get_img(xyz, features, res=256, border=1, data_type="feature", point_size=0.005):
    assert xyz.shape[0] == features.shape[0]
    if xyz.shape[0] == 0:
        return np.zeros((res*2, res*2, 4), dtype=np.uint8)
    img = render_point_cloud(xyz, features, res=res, data_type=data_type,
                                    rotation_range=[0, 360], rotation_step=4, 
                                    elevation_range=[30, 30], elevation_step=1, 
                                    H_N=2, W_N=2, point_size=point_size)
    d = border
    img[:d, :, :] = 255
    img[-d:, :, :] = 255
    img[:, :d, :] = 255
    img[:, -d:, :] = 255
    return img

def color_map(x, data_type="feature"):
    # ret: [N, 4], range in [0, 255], int
    if data_type == "feature":
        return convert_to_rgba(x)
    elif data_type in ["xyz", "xyz_10"]:
        # x: [N, 3], range in [-1, 1]
        x = x.clamp(-1, 1)
        x = (x + 1) / 2
        if data_type == "xyz_10":
            x = (x * 10).frac()
        ans = torch.cat([x, torch.ones_like(x[:, :1])], dim=1) * 255
        return ans.int()
    elif data_type == "normal":
        # x: [N, 3], range in [-1, 1]
        x = x.clamp(-1, 1)
        x = (x + 1) / 2
        ans = torch.cat([x, torch.ones_like(x[:, :1])], dim=1) * 255
        ans = ans.clamp(0, 255).int()
        return ans
    elif data_type == "color":
        # x: [N, 3], range in [0, 1]
        x = x.clamp(0, 1)
        ans = torch.cat([x, torch.ones_like(x[:, :1])], dim=1) * 255
        return ans.int()
    elif data_type == "color_1channel":
        # x: [N, 1], range in [0, 1]
        x = x.clamp(0, 1)
        ans = torch.cat([x, x, x, torch.ones_like(x)], dim=1) * 255
        return ans.int()
    elif data_type == "color4":
        # x: [N, 4], range in [0, 1]
        x = x.clamp(0, 1)
        ans = x * 255
        return ans.int()
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

def get_errmap_viz(err_color, cmap_name="coolwarm", err_min=0.0, err_max=0.2):
    if isinstance(err_color, np.ndarray):
        err_color = torch.from_numpy(err_color)
    device = err_color.device
    err_color = err_color.to(torch.float32)
    err_color = err_color.mean(dim=1).cpu().numpy()
    err_color = err_color.clip(err_min, err_max)
    err_color = (err_color-err_min)/(err_max-err_min)
    cmap = plt.get_cmap(cmap_name)
    err_color = torch.from_numpy(cmap(err_color)).to(device=device)
    return err_color

def get_offset_viz(xyz, xyz_add_offset, color, is_need_softmax=True):
    # xyz, xyz_add_offset: [N, 3], range in [-1, 1]
    # color: [N, 3], one hot
    xyz_ret = torch.cat([xyz, xyz_add_offset], dim=0)
    if is_need_softmax:
        color = F.softmax(color, dim=1)
    color_indices = torch.argmax(color, dim=1)
    color_indices = color_indices # [N]
    
    color_origin = color
    color_offset = torch.zeros_like(color_origin) # [N, 4]
    color_offset[color_indices == 0] = torch.tensor([1, 0, 0, 1.], device=color.device).to(dtype=color_offset.dtype)
    color_offset[color_indices == 1] = torch.tensor([0, 1, 0, 1.], device=color.device).to(dtype=color_offset.dtype)
    color_offset[color_indices == 2] = torch.tensor([0, 0, 1, 1.], device=color.device).to(dtype=color_offset.dtype)
    color_ret = torch.cat([color_origin, color_offset], dim=0)
    return xyz_ret, color_ret

def get_offset_viz_gcolor(xyz, xyz_add_offset, gcolor):
    # xyz, xyz_add_offset: [N, 3], range in [-1, 1]
    # gcolor: [N, 1]
    xyz_ret = torch.cat([xyz, xyz_add_offset], dim=0)
    
    gcolor_sq = gcolor.squeeze(-1) # [N]

    color_origin = convert_gcolor_to_gcolormap(gcolor)
    color_offset = torch.zeros((xyz_add_offset.shape[0], 4), device=xyz.device) # [N, 4]
    color_offset[gcolor_sq > 0] = torch.tensor([1, 0, 0, 1.], device=xyz.device)
    color_offset[gcolor_sq < 0] = torch.tensor([0, 0, 1, 1.], device=xyz.device)
    color_ret = torch.cat([color_origin, color_offset], dim=0)
    return xyz_ret, color_ret

def render_point_cloud(xyz, color, res, data_type="feature", rotation_range=[0, 0], rotation_step=1, elevation_range=[30, 30], elevation_step=1, H_N=1, W_N=1, point_size=0.005):
    """
    render point cloud to image
    Params:
        xyz: [N, 3], norm to [-1, 1]
        color: [N, 1], a scalar value
        res: int, resolution
        camera_rotation: a number, camera rotation angle
        camera_elevation: a number, camera elevation angle
    Return:
        image: [H, W, 3], rgb image
    """
    xyz = xyz
    color = color
    color = color_map(color, data_type)
    # xyz [N, 3], range in [0, 1]
    # color [N, 4], range in [0, 255], int

    def create_box(c=[0, 0, 0], r=1):  
        # c: [x, y, z], r  
        # ret: vertices and faces, the triangle mesh list  
        vertices = torch.tensor(
            [[-1., -1., 1.],
                [-1., 1., 1.],
                [-1., -1., -1.],
                [-1., 1., -1.],
                [1., -1., 1.],
                [1., 1., 1.],
                [1., -1., -1.],
                [1., 1., -1.]]
        )

        faces = torch.tensor(
            [[0, 1, 3],
                [2, 3, 7],
                [6, 7, 5],
                [4, 5, 1],
                [2, 6, 4],
                [7, 3, 1],
                [3, 2, 0],
                [7, 6, 2],
                [5, 4, 6],
                [1, 0, 4],
                [4, 0, 2],
                [1, 5, 7]]
        )

        vertices = vertices * torch.tensor(r) + torch.tensor(c)
    
        return vertices, faces

    def create_box_all(center_list, color_list, r=0.005):  
        # center_list: [N, 3]
        # color_list: [N, 4]
        # r: a float

        center_temp, faces_temp = create_box(r=r) # [8, 3], [12, 3]
        
        center_temp = center_temp.to(device=center_list.device)
        faces_temp = faces_temp.to(device=center_list.device)

        N = center_list.shape[0]
        vertices = torch.stack([center_list] * 8, axis=1)
        vertices = vertices[:, ...] + center_temp
        vertices = vertices.reshape(N*8, -1)

        faces = torch.stack([faces_temp] * N, axis=0) # [N, 12, 3]
        faces = faces + torch.arange(N, device=center_list.device)[:, None, None] * 8
        faces = faces.reshape(-1, 3)

        face_colors = torch.stack([color_list] * 12, axis=1)
        face_colors = face_colors.reshape(-1, 4)[..., :3]

        vertices = vertices.cpu().numpy()
        faces = faces.cpu().numpy()
        face_colors = face_colors.cpu().numpy()

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.visual.face_colors = face_colors

        return mesh

    mesh = create_box_all(xyz, color, r=point_size)
    render = NeuralRender(camera_model=PerspectiveCamera(fovy=49.0), device=xyz.device)

    _, _, _, ret_images, _ = get_rendering_images(mesh, 
                            rotation_range=rotation_range, 
                            elevation_range=elevation_range,
                            rotation_step=rotation_step,
                            elevation_step=elevation_step,
                            data_type_list="rgb",
                            img_resolution=[res, res],
                            checker_size=30,
                            render=render,
                            depth_format="origin",
                            is_normalize=False)

    ret_image = []
    for i in range(H_N*W_N):
        img = ret_images["rgb_list"][i][0]
        ret_image.append(img)
    ret_image = np.stack(ret_image, axis=0) # [H_N*W_N, H, W, 4]
    # convert to [H_N*H, W_N*W, 4]
    ret_image = ret_image.reshape(H_N, W_N, res, res, 4).transpose(0, 2, 1, 3, 4).reshape(H_N*res, W_N*res, 4)

    return ret_image

def calc_score_mse(pred, gt):
    ans = {}
    ans["mse"] = nn.MSELoss()(pred, gt).mean().item()
    ans["L1"] = nn.L1Loss()(pred, gt).mean().item()
    return ans

def save_ply(xyz_cur, features_cur, path, fn, is_out_raw=False, is_output_npz=False, offset_cur=None, data_type="feature"):
    # features_cur: [N, 1]
    if xyz_cur.shape[0] == 0:
        return 
    
    if data_type == "feature":
        if features_cur.shape[1] == 1:
            features_cur_rgb = convert_to_rgba(features_cur)
        else:
            features_cur_rgb = color_map(features_cur, "color")
    else:
        features_cur_rgb = color_map(features_cur, data_type)

    if offset_cur is not None:
        xyz_offset_cur = xyz_cur + offset_cur
        xyz_cur = torch.cat([xyz_cur, xyz_offset_cur], dim=0)
        offset_rgb = torch.ones_like(features_cur_rgb) * 255
        offset_rgb[:, 1] = offset_rgb[:, 2] = 0
        features_cur_rgb = torch.cat([features_cur_rgb, offset_rgb], dim=0)

    features_cur_rgb = features_cur_rgb.clamp(0, 255)

    os.makedirs(path, exist_ok=True)
    points_mesh_fn = os.path.join(path, fn)

    xyz_cur_np = xyz_cur.cpu().numpy()
    features_cur_rgb_np = features_cur_rgb.cpu().numpy() # [N, 4]
    features_cur_rgb_np = features_cur_rgb_np.astype(np.int64)
    trimesh.points.PointCloud(xyz_cur_np, colors=features_cur_rgb_np).export(points_mesh_fn)

    if is_out_raw:
        xyz_cur_raw = xyz_cur
        if features_cur.shape[1] == 1:
            features_cur_raw = torch.cat([features_cur]*4, dim=1)*127.5 + 127.5
        else:
            features_cur_raw = features_cur

            features_cur_raw = features_cur_raw*255
            features_cur_raw = torch.cat([features_cur_raw, torch.ones_like(features_cur_raw[:, :1])*255], dim=1)
            
        xyz_cur_raw = xyz_cur_raw.cpu().numpy()
        features_cur_raw = features_cur_raw.cpu().numpy().astype(np.int64)
        raw_points_mesh_fn = os.path.join(path, f"raw_{fn}")
        trimesh.points.PointCloud(xyz_cur_raw, colors=features_cur_raw).export(raw_points_mesh_fn)

    if is_output_npz:
        npz_fn = os.path.join(path, f"{fn}.npz")
        np.savez(npz_fn, point=xyz_cur_np, color=features_cur.cpu().numpy()+1)

def save_all(data_raw, path):
    device = torch.device("cpu")
    data = {}
    for k in data_raw.keys():
        if k.startswith("indicenearest"):
            dtype = torch.long
        else:
            dtype = torch.float32
        data[k] = torch.from_numpy(data_raw[k]).to(dtype).to(device)
    
    print(data["invT"])
    for suffix in [""]:
        if f"xyz{suffix}" not in data:
            continue
        # save sample points
        xyz = data[f"xyz{suffix}"]

        # color_grading
        for k in ["dcdf", "cdf"]:
            if f"{k}{suffix}" not in data:
                continue
            cur_color = data[f"{k}{suffix}"]
            save_ply(xyz, cur_color, path, f"{k}{suffix}.ply", data_type="color_1channel")
            cur_color = convert_gcolor_to_gcolormap(cur_color)
            save_ply(xyz, cur_color, path, f"{k}{suffix}.ply", data_type="color4")

        # ggcolor
        for k in ["gdcdf", "gcdf"]:
            if f"{k}{suffix}" not in data:
                continue
            cur_gcolor = data[f"{k}{suffix}"]
            # normalize along the last dimension
            gcdf_scale = torch.norm(cur_gcolor, dim=1, keepdim=True) # [N, 1]  
            gcdf_dir = cur_gcolor / torch.clamp(gcdf_scale, min=1e-6) # [N, 3]

            gcdf_scale_color = get_errmap_viz(gcdf_scale, cmap_name="coolwarm", err_min=0, err_max=gcdf_scale.max())
            save_ply(xyz, gcdf_scale_color, path, f"{k}{suffix}_scale.ply", data_type="color4")
            save_ply(xyz, gcdf_dir, path, f"{k}{suffix}_dir.ply", data_type="normal")


        # normal
        if f"normal{suffix}" in data:
            normal = data[f"normal{suffix}"]
            save_ply(xyz, normal, path, f"normal{suffix}.ply", data_type="normal")

        # quadsize
        for k_ in ["sizing", "quadsize"]:
            k = f"{k_}{suffix}"
            if k in data:
                sizing = data[k] # [N, 1]
                sizing = sizing / sizing.max()
                sizing = sizing.repeat(1, 3) # [N, 3]
                save_ply(xyz, sizing, path, f"{k}{suffix}.ply", data_type="xyz_10")

        for k in ["offset", "offset1", "offset2", "offset3"]:
            k = f"{k}{suffix}"
            if k in data:
                gcolor = data[f"cdf{suffix}"]
                xyz_gt_offset, color_gt_offset = get_offset_viz_gcolor(xyz, xyz+data[k], gcolor)
                save_ply(xyz_gt_offset, color_gt_offset, path, f"gt_viz_{k}{suffix}.ply", data_type="color4")
                
                offsetcolor = color_map(xyz+data[k], "xyz_10")
                trimesh.points.PointCloud(xyz.cpu().numpy(), colors=offsetcolor.cpu().numpy()).export(os.path.join(path, f"{k}_viz.ply"))
    

        if f"quadid{suffix}" in data:
            print(f"quadid{suffix}: ", data[f"quadid{suffix}"].shape, data[f"quadid{suffix}"].min(), data[f"quadid{suffix}"].max())

            num_color = int(data[f"quadid{suffix}"].max().item() + 1)
            color_table = torch.randint(0, 256, (num_color, 3), device=xyz.device) / 255

            color_quadid = color_table[data[f"quadid{suffix}"].long()]
            save_ply(xyz, color_quadid, path, f"quadid{suffix}.ply", data_type="color")

        if f"indicenearest{suffix}" in data:
            xyz_in = data[f"xyz"]
            xyz_cur = data[f"xyz{suffix}"]
            indice_nearest = data[f"indicenearest{suffix}"]
            print(f"xyz{suffix}, indice_nearest: {indice_nearest.shape} {indice_nearest.dtype}")
            offset = xyz_in[indice_nearest] - xyz_cur # [N, 3]
            offset_len = torch.norm(offset, dim=1)
            count_001 = torch.sum(offset_len > 0.01)
            count_005 = torch.sum(offset_len > 0.05)

            print(f"xyz{suffix}, >0.01: {count_001} {count_001 / xyz_cur.shape[0]:.2f}, min={offset_len.min()}, max={offset_len.max()}, mean={offset_len.mean()}, std={offset_len.std()}")
            print(f"xyz{suffix}, >0.05: {count_005} {count_005 / xyz_cur.shape[0]:.2f}, min={offset_len.min()}, max={offset_len.max()}, mean={offset_len.mean()}, std={offset_len.std()}")



@contextmanager
def stdout_redirected(to=os.devnull):
    """
    A context manager to temporarily redirect stdout to another file. This is useful when you want to suppress the
    output from blender, especially when you are rendering a large number of images.

    Example usage:
    >>> import os
    >>> filename = os.devnull
    >>> with stdout_redirected(to=filename):
    ...     print("from Python")
    ...     os.system("echo non-Python applications are also supported")

    :param to: The file to redirect stdout to. Default is os.devnull.
    :return: The context manager.
    """
    
    fd = sys.stdout.fileno()

    def _redirect_stdout(_to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(_to.fileno(), fd)  # fd writes to '_to' file
        sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(_to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(_to=old_stdout)  # restore stdout.

def copy_code(code_dir, result_folder):
    save_dir = os.path.join(result_folder, "_code")
    os.makedirs(save_dir, exist_ok=True)
    print(f"code save to {save_dir}")

    def is_code(fn):
        ext_list = [".py", ".sh", ".cpp", ".h", ".hpp"] 
        for ext in ext_list:
            if fn.endswith(ext):
                return True
        return False

    # for all .py files in code_dir, copy them to save_dir
    fn_list = []
    for root, dirs, files in os.walk(code_dir):
        for file in files:
            if is_code(file) and "_code" not in root:
                fn = os.path.join(root, file)
                fn_list.append(fn)

    fn_list_new = []
    for fn in fn_list:
        rel_p = os.path.relpath(fn, code_dir)
        folder = rel_p.replace("\\", "/").split("/")[0]
        if folder in ["outputs", "results"]:
            continue
        fn_list_new.append(fn)

    for fn in fn_list_new:
        save_fn = os.path.join(save_dir, os.path.relpath(fn, code_dir))
        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        # do not print to terminal
        with stdout_redirected():
            os.system(f"cp {fn} {save_fn}")

def to(x, **kwargs):
    if isinstance(x, dict):
        return {k: to(v, **kwargs) for k, v in x.items()}
    elif isinstance(x, list):
        return [to(v, **kwargs) for v in x]
    elif isinstance(x, torch.Tensor):
        return x.to(**kwargs)
    elif isinstance(x, Octree):
        return x.to(**kwargs)
    else:
        return x

def save_and_return_image_udf(xyz, pred, gt, save_path, name_suffix, res=256, is_save_img=True):
    pred_color = get_errmap_viz(pred, err_min=0.0, err_max=1.0)
    gt_color = get_errmap_viz(gt, err_min=0.0, err_max=1.0)
    save_ply(xyz, pred_color, save_path, f"points{name_suffix}.ply", data_type=f"color4")
    save_ply(xyz, gt_color, save_path, f"gt{name_suffix}.ply", data_type=f"color4")
    err_color = torch.abs(pred - gt)
    err_color = get_errmap_viz(err_color)

    img_net = get_img(xyz, pred_color, res=res, data_type=f"color4")
    img_gt = get_img(xyz, gt_color, res=res, data_type=f"color4")
    img_err = get_img(xyz, err_color, res=res, data_type="color4")
    img = np.concatenate([img_gt, img_net, img_err], axis=1)
    if is_save_img:
        cv2.imwrite(os.path.join(save_path, f"img{name_suffix}.png"), cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
    return img

def save_and_return_image_gcolor(xyz, pred, gt, save_path, name_suffix, res=256, is_save_img=True):
    pred_color = convert_gcolor_to_gcolormap(pred)
    gt_color = convert_gcolor_to_gcolormap(gt)
    save_ply(xyz, pred_color, save_path, f"points{name_suffix}.ply", data_type=f"color4")
    save_ply(xyz, gt_color, save_path, f"gt{name_suffix}.ply", data_type=f"color4")
    err_color = torch.abs(pred - gt)
    err_color = get_errmap_viz(err_color)

    img_net = get_img(xyz, pred_color, res=res, data_type=f"color4")
    img_gt = get_img(xyz, gt_color, res=res, data_type=f"color4")
    img_err = get_img(xyz, err_color, res=res, data_type="color4")
    img = np.concatenate([img_gt, img_net, img_err], axis=1)
    if is_save_img:
        cv2.imwrite(os.path.join(save_path, f"img{name_suffix}.png"), cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
    return img

def get_ggcolor_dir(ggcolor):
    ggcolor_scale = torch.norm(ggcolor, dim=1, keepdim=True) # [N, 1]  
    ggcolor_dir = ggcolor / torch.clamp(ggcolor_scale, min=1e-6) # [N, 3]

    return ggcolor_dir

def save_and_return_image_split(xyz, pred, gt, data_type, save_path, name_suffix, res=256, is_net=True, is_gt=True, is_err=True):
    save_ply(xyz, pred, save_path, f"points{name_suffix}.ply", data_type=data_type)
    if is_gt:
        save_ply(xyz, gt, save_path, f"gt{name_suffix}.ply", data_type=data_type)
    err_color = torch.abs(pred - gt)
    err_color = get_errmap_viz(err_color)

    ans = {}
    if is_net:
        ans["img"] = get_img(xyz, pred, res=res, data_type=data_type)
    if is_gt:
        ans["img_gt"] = get_img(xyz, gt, res=res, data_type=data_type)
    if is_err:
        ans["img_err"] = get_img(xyz, err_color, res=res, data_type="color4")
    return ans

def get_optimal(k, scores):
    if "accu" in k:
        return np.max(scores)
    else:
        return np.min(scores)

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def count_parameters(model):
    ans = {
        "all": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "non_trainable": sum(p.numel() for p in model.parameters() if not p.requires_grad)
    }
    for k in ans.keys():
        ans[k] = f"{ans[k]/1e6:.2f}M"
    return ans

def extract_mesh_from_udf(ae, x_last, batch_size=32768, max_depth=7):
    from DualMeshUDF import extract_mesh

    def normalize(v, dim=-1):
        norm = torch.linalg.norm(v, axis=dim, keepdims=True)
        norm[norm == 0] = 1
        return v / norm

    def udf_from_net(device):
        def udf(pts, device=device):
            # pts: [N, 3]
            pts = torch.from_numpy(pts).to(device)
            input = pts.reshape(1, -1, pts.shape[-1]).float()

            with torch.no_grad():
                udf_p = ae.cond_vae.query_last_features(x_last, input)["udf"] * 0.08 # rescale
            
            udf_p = udf_p.reshape(-1, 1).detach().cpu().numpy()
            return udf_p

        return udf

    def udf_grad_from_net(device):

        def grad(pts, device=device):
            # pts: [N, 3]
            pts = torch.from_numpy(pts).to(device)
            pts.requires_grad = True

            input = pts.reshape(1, -1, pts.shape[-1]).float()
            udf_p = ae.cond_vae.query_last_features(x_last, input)["udf"] * 0.08 # rescale

            udf_p.sum().backward()
            grad_p = pts.grad.detach()
            grad_p = normalize(grad_p)

            grad_p = grad_p.reshape(-1, 3).detach().cpu().numpy()
            udf_p = udf_p.reshape(-1, 1).detach().cpu().numpy()

            return udf_p, grad_p

        return grad


    # compose functions
    udf_func = udf_from_net(x_last.device)
    udf_grad_func = udf_grad_from_net(x_last.device)

    # get mesh
    mesh_v, mesh_f = extract_mesh(udf_func, udf_grad_func, batch_size, max_depth)

    return mesh_v, mesh_f    


def rotate_points_pca_mat(xyz_ori):
    from sklearn.decomposition import PCA
    if isinstance(xyz_ori, torch.Tensor):
        xyz = xyz_ori.cpu().numpy()
    else:
        xyz = xyz_ori

    pca = PCA(n_components=3)  
    pca.fit(xyz)  

    # the first axis is the longest axis, the second axis is the second longest axis, and the third axis is the shortest axis
    reference_directions = pca.components_[::-1].copy() # [3, 3]
    

    # rotate the point cloud, so that the longest axis aligns with the x-axis
    return reference_directions.T

def convert_gcolor_to_gcolormap(color, cmap_name="turbo"):
    assert len(color.shape) == 2
    if cmap_name == "" or cmap_name is None:
        color = color.repeat(1, 3)
    else:
        color = get_errmap_viz(color, cmap_name=cmap_name, err_min=0, err_max=1)
    return color
