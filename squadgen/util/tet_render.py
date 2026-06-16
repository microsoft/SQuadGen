import torch
import numpy as np

import torch.nn.functional as F
import nvdiffrast.torch as dr

import squadgen.util.util as util

def auto_normals(mesh_v_pos_bxnx3, mesh_t_pos_idx_fx3):
    i0 = mesh_t_pos_idx_fx3[:, 0].long()
    i1 = mesh_t_pos_idx_fx3[:, 1].long()
    i2 = mesh_t_pos_idx_fx3[:, 2].long()
 
    mesh_v_pos_nx3 = mesh_v_pos_bxnx3.squeeze(0)
    v0 = mesh_v_pos_nx3[i0, :]
    v1 = mesh_v_pos_nx3[i1, :]
    v2 = mesh_v_pos_nx3[i2, :]
 
    face_normals = torch.cross(v1 - v0, v2 - v0)       # reverse
 
    # Splat face normals to vertices
    v_nrm = torch.zeros_like(mesh_v_pos_nx3)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)
 
    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(util.dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.tensor(
        [0.0, 0.0, 1.0], dtype=torch.float32, device=mesh_v_pos_bxnx3.device))
    v_nrm = util.safe_normalize(v_nrm)
 
    return v_nrm, util.safe_normalize(face_normals)

def projection(x=0.1, n=1.0, f=50.0, near_plane=None):
    if near_plane is None:
        near_plane = n
    return np.array(
        [[n / x, 0, 0, 0],
         [0, n / -x, 0, 0],
         [0, 0, -(f + near_plane) / (f - near_plane), -
          (2 * f * near_plane) / (f - near_plane)],
         [0, 0, -1, 0]]).astype(np.float32)


class PerspectiveCamera():
    def __init__(self, fovy=49.0):
        super().__init__()

        focal = np.tan(fovy / 180.0 * np.pi * 0.5)
        self.proj_mtx = torch.from_numpy(projection(
            x=focal, f=1000.0, n=1.0, near_plane=0.1)).unsqueeze(dim=0)

    def project(self, points_bxnx4: torch.Tensor):
        out = torch.matmul(
            points_bxnx4,
            torch.transpose(self.proj_mtx.to(points_bxnx4.device), 1, 2))
        return out


def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr.contiguous(), rast, attr_idx, rast_db=rast_db,
        diff_attrs=None if rast_db is None else 'all')


def xfm_points(points, matrix):

    out = torch.matmul(F.pad(points, pad=(0, 1), mode='constant',
                       value=1.0), torch.transpose(matrix, 1, 2))
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(
            out)), "Output of xfm_points contains inf or NaN"
    return out


def render_normal(face_normals, rast):
    rast = rast.clone()
    face_idx = rast[..., -1].long().detach()
    B, H, W = face_idx.shape
    face_idx = face_idx.flatten()
    mask = face_idx == 0
    face_idx[mask] = 1
    normal = torch.index_select(face_normals, dim=0, index=face_idx - 1)
    normal[mask, :] = 1
    normal = normal.reshape(B, H, W, 3)
    return normal


def render_depth(vert_depth, mesh_t_pos_idx_bxfx3, rast):

    _, H, W, _ = rast.shape
    device = rast.device
    # get the full baryc_coord
    baryc_coord = rast[..., :2]         # [1, H, W, 2]
    third_baryc_coord = (torch.ones(
        baryc_coord.shape[:-1], device=device) - baryc_coord.sum(-1)).unsqueeze(-1)    # [1, H, W, 1]
    baryc_coord = torch.cat(
        [baryc_coord, third_baryc_coord], dim=-1)   # [1, H, W, 3]

    # get the face idx and invalid mask
    invalid_mask = rast[..., -1].long() == 0   # [1, H, W]
    face_idx = rast[..., -1].long() - 1        # [1, H, W]
    face_idx[invalid_mask] = 0

    tri_vert_idx = torch.index_select(
        mesh_t_pos_idx_bxfx3, dim=1, index=face_idx.flatten())  # [1, H * W, 3]

    depth_tri_vert = torch.index_select(
        vert_depth, dim=1, index=tri_vert_idx.flatten())  # [1, H * W * 3]
    depth_tri_vert = depth_tri_vert.reshape(1, H, W, 3)

    depth = torch.sum(depth_tri_vert * baryc_coord, dim=-1, keepdim=True)
    depth[invalid_mask] = 1
    return depth

def sparse_interp(gb_pos_bxhxwx3, points, colors, device, method='voxel_linear', volume_size=128):
    # gb_pos_bxhxwx3: [b,h,w,3], [-1, 1]
    # points: [N, 3], [-1, 1]
    # colors: [N, 4]
    # method: voxel_ceil, voxel_round, voxel_linear, ocnn_linear, ocnn_nearest

    points_query = ((gb_pos_bxhxwx3+1)*volume_size-1)/2 # [b,h,w,3], [0, size-1]

    if method[:5] == 'voxel':
        # splat points and colors to 3d tensor, [size, size, size, C]
        colors_3d = torch.ones([volume_size, volume_size, volume_size, colors.shape[-1]], device=device, dtype=colors.dtype) * 128
        points_indices = (((points+1) * volume_size-1)/2).long()
        colors_3d[points_indices[:, 0], points_indices[:, 1], points_indices[:, 2]] = colors # [n, n, n, 4]

        if method == 'voxel_ceil':
            points_ceil = torch.ceil(points_query).long().clamp(0, volume_size-1)
            colors = colors_3d[points_ceil[..., 0], points_ceil[..., 1], points_ceil[..., 2]]
        elif method == 'voxel_round':
            points_round = torch.round(points_query).long().clamp(0, volume_size-1)
            colors = colors_3d[points_round[..., 0], points_round[..., 1], points_round[..., 2]]
        elif method == 'voxel_linear':
            # trilinear interpolation
            direction = torch.tensor([[0, 0, 0],  
                                    [0, 0, 1],  
                                    [0, 1, 0],  
                                    [0, 1, 1],  
                                    [1, 0, 0],  
                                    [1, 0, 1],  
                                    [1, 1, 0],  
                                    [1, 1, 1]], device=colors_3d.device)  
            points_floor = torch.floor(points_query).long().clamp(0, volume_size-1) # [b,h,w,3]
            points_neigh = points_floor.unsqueeze(-2) + direction.unsqueeze(0) # [b,h,w,8,3]
            # for each point, we have 8 neighbors, and 8 weights
            points_weight_direction = (points_query - points_floor.float()) # [b,h,w,3], weight for each direction
            # get the 8 weights, [b,h,w,8]
            points_weight = torch.prod(direction.float()*points_weight_direction.unsqueeze(-2) + (1-direction.float())*(1-points_weight_direction.unsqueeze(-2)), dim=-1)

            # get the 8 colors, [b,h,w,8,4]
            colors_neigh = colors_3d[points_neigh[..., 0], points_neigh[..., 1], points_neigh[..., 2]]
            # get the final color, [b,h,w,4]
            colors = torch.sum(colors_neigh * points_weight.unsqueeze(-1), dim=-2)

        else:
            raise ValueError("Invalid voxel interp method")
    elif method[:4] == 'ocnn':
        from ocnn.octree import Points, Octree
        octree_points = Points(points, features=colors) # shape of points is [N, 3]
        depth=int(np.log2(volume_size.cpu().numpy()))
        octree = Octree(depth=depth, device=device)
        octree.build_octree(octree_points)
        features = octree.features[depth]

        ocnn_method = method[5:]
        if ocnn_method in ['nearest', 'linear']:
            from ocnn.nn.octree_interp import OctreeInterp
            interp = OctreeInterp(method=ocnn_method, nempty=True, rescale_pts=True)
            # data: [M, C], M points in the octree, C is the feature dimension
            # pts: [N, 4], query points, 4 dim is (x,y,z,b)
            b, h, w = gb_pos_bxhxwx3.shape[:3]
            pts = gb_pos_bxhxwx3.reshape(-1, 3)
            # add dim
            pts = torch.cat([pts, torch.zeros_like(pts[..., 0:1])], dim=-1)
            data = features
            colors = interp(data=data, octree=octree, depth=depth, pts=pts)
            colors = colors.reshape(b, h, w, -1)
        else:
            raise ValueError("Invalid ocnn interp method")
    else:
        raise ValueError("Invalid interp method")

    return colors



class NeuralRender():
    def __init__(self, device='cuda', camera_model=None):
        super().__init__()
        self.device = device
        self.ctx = None
        self.projection_mtx = None
        self.camera = camera_model

    def render_mesh(
            self,
            mesh_v_pos_bxnx3,
            mesh_t_pos_idx_fx3,
            camera_mv_bx4x4,
            mesh_v_feat_bxnxd,
            resolution=256,
            spp=1,
            device='cuda',
            hierarchical_mask=False
    ):
        assert not hierarchical_mask

        if self.ctx is None:
            self.ctx = dr.RasterizeCudaContext(device=self.device)

        mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(
            camera_mv_bx4x4) else camera_mv_bx4x4
        # Rotate it to camera coordinates

        v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)

        v_pos_clip = self.camera.project(v_pos).to(
            torch.float32)  # Projection in the camera

        num_layers = 1
        assert mesh_t_pos_idx_fx3.shape[0] > 0 

        vert_normals, face_normals = auto_normals(
            mesh_v_pos_bxnx3, mesh_t_pos_idx_fx3)
        mesh_v_feat_bxnxd = torch.cat([mesh_v_feat_bxnxd.expand(v_pos.shape[0], -1, -1),
                                       v_pos], dim=-1)  # [1, 74615, 3], [1, 74615, 4], [1, 74615, 3]

        # vert_depth = mesh_v_feat_bxnxd[..., 5:6].squeeze(-1)
        with dr.DepthPeeler(self.ctx, v_pos_clip, mesh_t_pos_idx_fx3, [resolution * spp, resolution * spp]) as peeler:
            for _ in range(num_layers):
                rast, db = peeler.rasterize_next_layer()
                gb_feat, _ = interpolate(
                    mesh_v_feat_bxnxd, rast, mesh_t_pos_idx_fx3)
                normal = render_normal(face_normals, rast)

        hard_mask = torch.clamp(rast[..., -1:], 0, 1)
        antialias_mask = dr.antialias(
            hard_mask.clone().contiguous(), rast, v_pos_clip,
            mesh_t_pos_idx_fx3)

        depth = gb_feat[..., 5:6]

        tex_pos = gb_feat[..., :3]

        return tex_pos, antialias_mask, hard_mask, rast, v_pos_clip, depth, normal

    def render_mesh_with_rgb(
            self,
            mesh_v_pos_bxnx3,
            mesh_t_pos_idx_fx3,
            camera_mv_bx4x4,
            tex=None,
            uv=None,
            resolution=256,
            spp=1,
            device='cuda',
            hierarchical_mask=False,
            tex_type=None,
            sparse_data=None,
    ):
        batch_size = mesh_v_pos_bxnx3.shape[0]
        assert not hierarchical_mask
        if self.ctx is None:
            self.ctx = dr.RasterizeCudaContext(device=self.device)

        mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(
            camera_mv_bx4x4) else camera_mv_bx4x4
        # Rotate it to camera coordinates
        v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)
        v_pos_clip = self.camera.project(v_pos)  # Projection in the camera

        # Render the image,
        # Here we only return the feature (3D location) at each pixel, which will be used as the input for neural render
        num_layers = 1
        mask_pyramid = None
        assert mesh_t_pos_idx_fx3.shape[0] > 0  # Make sure we have shapes
        # Concatenate the pos compute the supervision

        normals, face_normals = auto_normals(
            mesh_v_pos_bxnx3[0:1], mesh_t_pos_idx_fx3)
        mesh_v_feat_bxnxd = v_pos
        if uv is not None:
            mesh_v_feat_1xnxd = uv[None, ...]
        with dr.DepthPeeler(self.ctx, v_pos_clip, mesh_t_pos_idx_fx3, [resolution * spp, resolution * spp]) as peeler:
            for _ in range(num_layers):
                rast, db = peeler.rasterize_next_layer()
                gb_feat_bxhxwxd, _ = interpolate(
                    mesh_v_feat_bxnxd, rast, mesh_t_pos_idx_fx3)
                # mesh_v_feat_bxnxd: [bs, n, 4] each vertex has a 3d position, 4 dim representation
                if uv is not None:
                    if tex_type == 'sparse':
                        # Given the sparse_data, sparse_data["points"], sparse_data["sdfs"], sparse_data["colors"], the shape is [N, C], N is the number of sparse voxel
                        # The sparse_data["points"] is the regular grid 3d coordinate in resoluation sparse_data["size"]^3
                        # we need to interpolate the colors, and then render the rgb
                        # for each pixel, we can get the corresponding 3d coordinate, and then find the nearest point in the sparse_data["points"], and then get the color
                        
                        # get the original 3d coordinate for each pixel
                        gb_pos_bxhxwx3, _ = interpolate(
                            mesh_v_pos_bxnx3, rast, mesh_t_pos_idx_fx3)
                        # gb_pos_bxhxwx3: [bs, H, W, 3], the 3d coordinate for each pixel
                        # mesh_v_pos_bxnx3: [n, 3], the 3d coordinate for each vertex

                        # load data and convert to tensor
                        volume_size = sparse_data["size"] if "size" in sparse_data else 128
                        points = sparse_data["points"] # [N, 3], range in [-1, 1]
                        colors = sparse_data["colors"] # # [N, 4], range in [0, 255], uint8
                        volume_size = torch.tensor(volume_size, device=device)
                        points = torch.tensor(points, device=device)
                        colors = torch.tensor(colors, device=device) 

                        rgb = sparse_interp(gb_pos_bxhxwx3, points, colors, device=device, method='voxel_linear', volume_size=volume_size)
                    else:
                        gb_feat_1xhxwxd, _ = interpolate(
                            mesh_v_feat_1xnxd, rast, mesh_t_pos_idx_fx3)
                        # mesh_v_feat_1xnxd: [1, n, 2]: for this mesh, each vertex has a uv coordinate
                        # rast: [bs, H, W, 4]: for each pixel, we have the barycentric coordinate(2), depth, and face index
                        # mesh_t_pos_idx_fx3: [f, 3]: for each face, we have the index of the vertices
                        # gb_feat_1xhxwxd: [bs, H, W, 2]: the uv coordinate for each pixel

                        # get rgb
                        tex = torch.flip(tex, [0])
                        texc = gb_feat_1xhxwxd.expand(batch_size, -1, -1, -1)
                        rgb = dr.texture(tex[None, ...].contiguous(), texc.contiguous(), filter_mode='nearest')
                    rgb = rgb * torch.clamp(rast[..., -1:], 0, 1)
                else:
                    rgb = None
                # get normal
                normal = render_normal(face_normals, rast)

        hard_mask = torch.clamp(rast[..., -1:], 0, 1)
        antialias_mask = dr.antialias(
            hard_mask.clone().contiguous(), rast, v_pos_clip,
            mesh_t_pos_idx_fx3)

        depth = gb_feat_bxhxwxd[..., 2:3]
        return None, antialias_mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth, normal, rgb
