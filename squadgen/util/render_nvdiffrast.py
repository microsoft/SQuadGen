import json
import cv2
import time
import copy
from typing import Union
import trimesh
import torch
import numpy as np
import os
import os.path as osp
import traceback
import zipfile
from typing import List
from abc import abstractmethod

depth_scale = 1/3.5

class ZipBase(object):
    def __init__(self):
        pass

    @abstractmethod
    def namelist(self): pass

    def listdir(self, dir_name):
        file_list = [f[f.find(dir_name) + len(dir_name):] for f in self.namelist() if f.find(dir_name) != -1]
        file_list = [f.lstrip('/').rstrip('/') for f in file_list]
        file_list = [f for f in file_list if len(f.split('/')) == 1 and f.split('/')[0]]
        return file_list


class ZipIO(ZipBase):
    def __init__(self, zip_path, mode='r', prefetch=False):
        super().__init__()
        if prefetch:
            raise NotImplementedError
        else:
            self._zip_meta = zipfile.ZipFile(zip_path, mode)

    def namelist(self) -> List[str]:
        return sorted(self._zip_meta.namelist())

    def has_file(self, file_name):
        return file_name in self._zip_meta.namelist()

    def read(self, file_name):
        if isinstance(file_name, list):
            file_name = '/'.join(tuple(file_name))
        return self._zip_meta.read(file_name)

    def writestr(self, file_name, data):
        return self._zip_meta.writestr(file_name, data)

    def read_image(self, file_name, rgb=True):
        sample_meta = self._zip_meta.read(file_name)
        print(sample_meta)
        img = cv2.imdecode(np.frombuffer(sample_meta, np.uint8), cv2.IMREAD_UNCHANGED)
        if rgb and img.shape[-1] == 3:
            img = img[..., ::-1]
        return np.array(img)
    

    def write_image(self, file_name, img, rgb=True):
        img_encode = cv2.imencode('.png', img)[1]
        str_encode = img_encode.tobytes()
        self.writestr(file_name, str_encode)


    def close(self):
        self._zip_meta.close()


class Logger:
    FLUSH_INTERVAL = 50

    def __init__(self, path):
        self.file = open(path, "w")
        self.cnt = 0

    def log(self, msg, end='\n'):
        self.file.write(msg+end)
        self.cnt += 1
        if self.cnt == Logger.FLUSH_INTERVAL:
            self.file.flush()
            self.cnt = 0

    def close(self):
        self.file.close()


def scale_to_unit_sphere(mesh, r=1, eps=1e-20):
    if isinstance(mesh, trimesh.Trimesh):
        offset = mesh.bounding_box.centroid
        vertices = mesh.vertices - mesh.bounding_box.centroid
        distances = np.linalg.norm(vertices, axis=1)
        vertices /= np.max(distances) / r + eps
        scale=1 / np.max(distances) / r
        ret_mesh = copy.deepcopy(mesh)
        ret_mesh.vertices = vertices
        return ret_mesh
    elif isinstance(mesh, trimesh.Scene):
        centroid = mesh.centroid
        max_distance = 0
        for k, v in mesh.geometry.items():
            vertices = v.vertices - centroid
            distances = np.linalg.norm(vertices, axis=1)
            max_distance = max(max_distance, np.max(distances))

        for k, v in mesh.geometry.items():
            vertices = v.vertices - centroid
            distances = np.linalg.norm(vertices, axis=1)
            vertices /= max_distance / r + eps
            mesh.geometry[k].vertices = vertices

        return mesh
def get_image_uv(mesh: Union[trimesh.Scene, trimesh.Trimesh]):
    if type(mesh) is trimesh.Scene:
        mesh = mesh.dump(concatenate=True)
    visual = mesh.visual
    if type(visual) is trimesh.visual.color.ColorVisuals:
        visual = visual.to_texture()
    material = visual.material
    if type(material) is trimesh.visual.material.SimpleMaterial:
        material = material.to_pbr()
    image = torch.Tensor(np.array(material.baseColorTexture))
    uv = torch.Tensor(visual.uv)
    return image, uv

def get_camera_positions_our(
        device, azimuth, elevation, n=1, r=[1.0, 1.0]):
    output_points = torch.zeros((n, 3), device=device)
    sample_r = torch.rand((n, 1), device=device)
    sample_r = sample_r * r[0] + (1 - sample_r) * r[1]

    output_points[:, 0:1] = sample_r * \
        torch.cos(elevation) * torch.cos(azimuth)
    output_points[:, 1:2] = sample_r * torch.sin(elevation)
    output_points[:, 2:3] = sample_r * \
        torch.cos(elevation) * torch.sin(azimuth)

    return output_points


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def create_my_world2cam_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a world2camera matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor(
        [0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(
        torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(
        forward_vector, left_vector, dim=-1))

    new_t = torch.eye(4, device=device).unsqueeze(
        0).repeat(forward_vector.shape[0], 1, 1)
    new_t[:, :3, 3] = -origin
    new_r = torch.eye(4, device=device).unsqueeze(
        0).repeat(forward_vector.shape[0], 1, 1)
    new_r[:, :3, :3] = torch.cat(
        (left_vector.unsqueeze(dim=1), up_vector.unsqueeze(dim=1), forward_vector.unsqueeze(dim=1)), dim=1)
    world2cam = new_r @ new_t
    return world2cam



def get_camera_our(azimuth, elevation, n, device='cuda', r=2.5):

    radius_range = [r, r]
    camera_origin = get_camera_positions_our(
        device, azimuth=azimuth, elevation=elevation,
        n=n, r=radius_range
    )
    forward_vector = normalize_vecs(camera_origin)
    # Camera is always looking at the Origin point
    world2cam_matrix = create_my_world2cam_matrix(
        forward_vector, camera_origin, device=device)
    return world2cam_matrix, forward_vector, camera_origin




def get_camera_matrix(batch_size, rotaion, elevation, device="cuda"):
    sample_r = None

    world2cam_matrix, forward_vector, camera_origin = get_camera_our(
        rotaion, elevation, batch_size, device)

    return camera_origin.reshape(batch_size, 1, 3), world2cam_matrix.reshape(batch_size, 1, 4, 4), sample_r

import trimesh  
import numpy as np  
  
# Create a checkerboard texture
def create_checkerboard(size=1024, num_squares=30, c0=0.2, c1=0.8):  
    checkerboard = np.zeros((size, size, 4), dtype=np.float32)  
    s = size // num_squares  
    for i in range(num_squares + 1):  
        for j in range(num_squares + 1):  
            
            if (i + j) % 2:  
                c = c0 * 255
            else:   
                c = c1 * 255
            
            checkerboard[i*s:min((i+1)*s, size), j*s:min(size, (j+1)*s)] = [c, c, c, 255] 
    checkerboard = torch.tensor(checkerboard, dtype=torch.float32)
    return checkerboard  
  
# Cache of checkerboard images keyed by size
checkerboard_texture_map = {}

def get_rendering_images(
    mesh,
    rotation_range,
    elevation_range,
    rotation_step,
    elevation_step,
    data_type_list,
    checker_size,
    render,
    img_resolution: List[int], # [512, 512]
    depth_format,
    r = 1,
    sparse_data = None,
    is_normalize = True,
):
    
    render_time_one_object = 0
    start_time = time.time()

    n_view = rotation_step * elevation_step

    rotation_save_list = []
    elevation_save_list = []
    rotation_sampled = np.linspace(
        rotation_range[0], rotation_range[1], rotation_step, endpoint=False)
    elevation_sampled = np.linspace(
        elevation_range[0], elevation_range[1], elevation_step, endpoint=True)
    
    tex_list = []

    batch_cam_mv = []
    for i in range(n_view):
        rotation = rotation_sampled[i % rotation_step]
        elevation = elevation_sampled[i // rotation_step]
        rotation_save_list.append(rotation)
        elevation_save_list.append(elevation)

        _, cam_mv,  _ = get_camera_matrix(
            1,
            rotaion=torch.tensor([rotation * np.pi / 180]).cuda(),
            elevation=torch.tensor([elevation * np.pi / 180]).cuda()
        )
        batch_cam_mv.append(cam_mv.squeeze(0))
    batch_cam_mv = torch.cat(batch_cam_mv, dim=0).cuda()
    try:
        tex, uv = get_image_uv(mesh)
        tex = tex.cuda()
        for data_type in data_type_list:
            if data_type == "checker":
                if checker_size not in checkerboard_texture_map:
                    checkerboard_texture = create_checkerboard(num_squares=checker_size)
                    checkerboard_texture_map[checker_size] = checkerboard_texture
                tex_list.append(checkerboard_texture_map[checker_size].cuda())
            elif data_type == "rgb" or "sparse":
                tex_list.append(tex)
            else:
                raise NotImplementedError(f"data_type: {data_type}")
        uv = uv.cuda()
    except Exception as e:
        print("using gray color!")
        tex, uv = None, None
        return False, 0, 0, {}, batch_cam_mv
    if is_normalize:
        mesh = scale_to_unit_sphere(mesh, r=r)
    mesh_v, mesh_f = mesh.vertices, mesh.faces

    rgb_list = []
    mask_list = []
    for data_type, tex in zip(data_type_list, tex_list):
        tex_pos, mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth, normal, rgb = render.render_mesh_with_rgb(
            torch.from_numpy(mesh_v).to(dtype=torch.float32,
                                        device="cuda").unsqueeze(dim=0).expand(n_view, -1, -1),
            torch.from_numpy(mesh_f).to(
                dtype=torch.float32, device="cuda").int(),
            batch_cam_mv,
            tex=tex,
            uv=uv,
            resolution=img_resolution[0],
            device="cuda",
            hierarchical_mask=False,
            tex_type=data_type,
            sparse_data=sparse_data,
        )        
        if rgb is None:
            rgb = torch.ones_like(normal) * 127
        rgb_list.append(rgb)
        mask_list.append(mask)
        
    normal = (normal + 1) / 2

    render_time_one_object += time.time() - start_time
    start_time = time.time()

    ret_images = {
        "mask_list": [],
        "rgb_list": [],
        "depth_list": [],
        "normal_list": [],
    }

    for i in range(n_view):
        depth_map = depth[i].cpu().numpy()
        rgb_img_list = []
        for rgb, mask in zip(rgb_list, mask_list):
            mask_map = mask[i].cpu().numpy()
            mask_map = mask_map > 0
            rgb_img = rgb[i].cpu().numpy()
            mask_map = (rgb_img[..., 3] > 1-1e-4).astype(np.float32)[..., np.newaxis] # [res, res, 1]

            if rgb_img.shape[-1] == 4:
                rgb_img[..., -1:] = mask_map * 255
            elif rgb_img.shape[-1] == 3:
                H, W, _ = tuple(rgb_img.shape)
                rgb_img = np.concatenate(
                    (rgb_img, np.ones((H, W, 1))), axis=-1)
                rgb_img[..., -1:] = mask_map * 255
            else:
                raise NotImplementedError(
                    f"rgb_img.shape[-1] must be 3 or 4, got {rgb_img.shape[-1]}")
            rgb_img_list.append(rgb_img)

        normal_map = normal[i].cpu().numpy()

        if depth_format == "origin":
            depth_map = -depth_map * depth_scale # [0, 1]
            depth_mask = depth_map > 0
        elif depth_format == "direct":
            depth_map = -depth_map # [0, max_d]
            depth_mask = depth_map > 0.0001
            depth_content = depth_map[depth_mask]
            depth_map = (depth_map - depth_content.min()) / (depth_content.max() - depth_content.min() + 1e-9) # normalize to [0, 1]
            mask_map = depth_mask
        else:
            raise NotImplementedError(f"depth_format: {depth_format}")
        
        one_bg = np.ones_like(depth_map)
        depth_map = depth_mask * depth_map + (~depth_mask) * one_bg

        for j in range(len(rgb_img_list)):
            rgb_img_list[j] = rgb_img_list[j].astype(np.uint8)
        depth_map = (depth_map * 65535.0).astype(np.uint16)

        normal_map = (normal_map * 65535.0).astype(np.uint16)
        mask_map = (mask_map * 255.0).astype(np.uint8)
        
        ret_images["mask_list"].append(mask_map)
        ret_images["rgb_list"].append(rgb_img_list)
        ret_images["depth_list"].append(depth_map)
        ret_images["normal_list"].append(normal_map)

    return True, render_time_one_object, time.time() - start_time, ret_images, batch_cam_mv

@torch.no_grad()
def render_an_object(
    args,
    mesh: trimesh.Trimesh,
    save_zip_path,
    render,
    is_save_zip=True,
    data_type_list=["rgb"],
    depth_format = "origin",
    n_view: int = 64,
    r: float = 1,
    fovy: float = 49.0,
):
    rotation_range = args.rotation_range
    elevation_range = args.elevation_range
    rotation_step = args.rotation_step
    elevation_step = args.elevation_step

    n_view = rotation_step * elevation_step

    write_time_one_object = 0

    transforms = {
        "engine": "nvdiffrast",
        "camera_angle_x": fovy / 2 * np.pi / 180,
        "frame": []
    }

    try:
        
        os.makedirs(osp.dirname(save_zip_path), exist_ok=True)
        if is_save_zip:
            save_zipio = ZipIO(save_zip_path, mode="w")

        ret, render_time_one_object, write_time_one_object, ret_images, batch_cam_mv = get_rendering_images(
            mesh,
            rotation_range,
            elevation_range,
            rotation_step,
            elevation_step,
            data_type_list,
            args.checker_size,
            render,
            args.img_resolution,
            depth_format,
            r=r,
            sparse_data=getattr(args, "sparse_data", None),
        )

        if not ret:
            raise Exception("rendering failed!")

        start_time = time.time()

        for i in range(n_view):
            mask_map = ret_images["mask_list"][i]
            rgb_img_list = ret_images["rgb_list"][i]
            depth_map = ret_images["depth_list"][i]
            normal_map = ret_images["normal_list"][i]

            mask_fn = "{:03d}_mask".format(i)
            rgb_fn = "{:03d}".format(i)
            depth_fn = "{:03d}_depth0001".format(i)
            normal_fn = "{:03d}_normal0001".format(i)
            if is_save_zip:
                save_zipio.write_image(depth_fn + ".png", depth_map)
                save_zipio.write_image(normal_fn + ".png", cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(osp.join(save_zip_path, mask_fn + ".png"), mask_map)
                for rgb_img, data_type in zip(rgb_img_list, data_type_list):
                    cv2.imwrite(osp.join(save_zip_path, rgb_fn + f"_{data_type}.png"), cv2.cvtColor(rgb_img, cv2.COLOR_RGBA2BGRA))
                cv2.imwrite(osp.join(save_zip_path, depth_fn + ".png"), depth_map)
                cv2.imwrite(osp.join(save_zip_path, normal_fn + ".png"), cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR))
            transforms["frame"].append({
                "file_path": "train/" + rgb_fn,
                "depth_file_path": "train/" + depth_fn,
                "normal_file_path": "train/" + normal_fn,
                "transform_matrix": batch_cam_mv[i].cpu().numpy().tolist(),
            })

        # save transforms
        if is_save_zip:
            save_zipio.writestr("transforms.json", json.dumps(transforms, indent=4))
            save_zipio.close()
        else:
            with open(osp.join(os.path.dirname(save_zip_path), "transforms_train.json"), "w") as f:
                json.dump(transforms, f, indent=4)

            if hasattr(args, "is_save_video") and args.is_save_video:
                import imageio
                fps = 8
                for data_type in data_type_list:
                    video_name = osp.join(os.path.dirname(save_zip_path), f"_{data_type}.mp4")
                    writer = imageio.get_writer(video_name, fps=fps, codec='libx264', pixelformat='yuv420p')  
                    for i in range(n_view): 
                        rgb_fn = osp.join(save_zip_path, "{:03d}_{}.png".format(i, data_type))  
                        img = imageio.imread(rgb_fn)  
                        writer.append_data(img)  
                    writer.close()  

    except Exception as e:
        print(traceback.format_exc())
        print(str(e))
        return False, 0, 0, transforms
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(str(e))
            pass

    write_time_one_object += time.time() - start_time


    return True, render_time_one_object, write_time_one_object, transforms
