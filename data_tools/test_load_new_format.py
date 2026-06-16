import numpy as np
import copy
try:
    import point_cloud_utils as pcu  # used for debugging only
except ImportError:
    pcu = None
from squadgen.util.util import load_data
try:
    from noise import pnoise3
except ImportError:
    pnoise3 = None

def save_lines_as_obj(border_edge_ids, edge_id, subdiv_vertex, obj_file_name):
    count = 0
    border_edges = open(obj_file_name, "w")
    for i in range(border_edge_ids.shape[0]):
        eid = border_edge_ids[i]
        border_edges.write(
            "v %f %f %f\n"
            % (
                subdiv_vertex[edge_id[eid, 0], 0],
                subdiv_vertex[edge_id[eid, 0], 1],
                subdiv_vertex[edge_id[eid, 0], 2],
            )
        )
        border_edges.write(
            "v %f %f %f\n"
            % (
                subdiv_vertex[edge_id[eid, 1], 0],
                subdiv_vertex[edge_id[eid, 1], 1],
                subdiv_vertex[edge_id[eid, 1], 2],
            )
        )
        border_edges.write(f"l {count + 1} {count + 2}\n")
        count += 2

def save_lines_as_obj(border_edge_ids, edge_id, subdiv_vertex, obj_file_name):
    count = 0
    border_edges = open(obj_file_name, "w")
    for i in range(border_edge_ids.shape[0]):
        eid = border_edge_ids[i]
        border_edges.write("v %f %f %f\n" % (subdiv_vertex[edge_id[eid, 0], 0], subdiv_vertex[edge_id[eid, 0], 1], subdiv_vertex[edge_id[eid, 0], 2]))
        border_edges.write("v %f %f %f\n" % (subdiv_vertex[edge_id[eid, 1], 0], subdiv_vertex[edge_id[eid, 1], 1], subdiv_vertex[edge_id[eid, 1], 2]))
        border_edges.write(f"l {count+1} {count+2}\n")
        count += 2

def convert_offset_to_oldformat(key, value):
    ans = {}
    if key.startswith("offset_abcd"):
        suffix = key.replace("offset_abcd", "")
        ans.update({
            "offset": value[..., 0, :],
            "offsetb": value[..., 1, :],
            "offsetc": value[..., 2, :],
            "offsetd": value[..., 3, :],
        })
    elif key.startswith("offset_f_abcd"):
        suffix = key.replace("offset_f_abcd", "")
        ans.update({
            "foffseta": value[..., 0, :],
            "foffsetb": value[..., 1, :],
            "foffsetc": value[..., 2, :],
            "foffsetd": value[..., 3, :],
        }) 
    elif key.startswith("offset_d_abc"):
        suffix = key.replace("offset_d_abc", "")
        ans.update({
            "doffseta": value[..., 0, :],
            "doffsetb": value[..., 1, :],
            "doffsetc": value[..., 2, :],
        })     
    elif key.startswith("offset_123"):
        suffix = key.replace("offset_123", "")
        ans.update({
            "offset1": value[..., 0, :],
            "offset2": value[..., 1, :],
            "offset3": value[..., 2, :],
        })     
    else:
        assert False

    ans = {f"{k}{suffix}": v for k, v in ans.items()}
    return ans

def get_random_index(n, n_all, generator: np.random.Generator):
    n = max(n, 0)
    if n >= n_all:
        ans = np.concatenate(
            [np.arange(n_all), generator.integers(0, n_all, size=(n-n_all, ))], axis=0
        )
    else:
        ans = generator.integers(0, n_all, size=(n,))
    return ans


def add_perlin_noise(vertices, scale_range=[0.5, 1.5], intensity_range=[0.0, 0.2], generator=np.random.default_rng(0)):
    if pnoise3 is None:
        raise ImportError("The noise package is required when Perlin noise augmentation is enabled.")

    scale = generator.uniform(*scale_range)
    intensity = generator.uniform(*intensity_range)
    base = generator.integers(0, 10000)

    displaced_vertices = np.zeros_like(vertices)
    
    for i, (x, y, z) in enumerate(vertices):
        noise_x = pnoise3(x * scale, y * scale, z * scale, octaves=4, base=base)
        noise_y = pnoise3(y * scale, z * scale, x * scale, octaves=4, base=base)
        noise_z = pnoise3(z * scale, x * scale, y * scale, octaves=4, base=base)

        displacement = np.array([noise_x, noise_y, noise_z]) * intensity
        displaced_vertices[i] = vertices[i] + displacement

    return displaced_vertices

def add_noise_rotate(
    xyz, 
    rotate_range=[-5, 5],
    generator=np.random.default_rng(0),
):
    angle = generator.uniform(rotate_range[0], rotate_range[1], size=(3,))
    # degree to rad
    angle = angle * np.pi / 180

    for i in range(3):
        rot_cos, rot_sin = np.cos(angle[i]), np.sin(angle[i])
        if i == 0:
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        if i == 1:
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        if i == 2:
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])

        xyz = np.matmul(xyz, rot_t)

    return xyz


def add_noise_translation(
    xyz, 
    translation_range=[-0.05, 0.05],
    generator=np.random.default_rng(0),
):
    trans = generator.uniform(translation_range[0], translation_range[1], size=(3,))
    return xyz + trans[None, :]    

def add_noise_scale(
    xyz, 
    scale_range=[0.9, 1.1],
    generator=np.random.default_rng(0),
):
    scale = generator.uniform(scale_range[0], scale_range[1], size=(3,))
    return xyz * scale[None, :]

def add_noise_flip(xyz, generator=np.random.default_rng(0)):
    for i in range(3):
        if generator.random() > 0.5:
            xyz[:, i] = -xyz[:, i]
    return xyz

def add_noise(
    xyz, 
    generator=np.random.default_rng(0),
):
    if generator.random() < 0.95:
        # xyz = add_noise_flip(xyz, generator=generator)
        if generator.random() < 0.5:
            xyz = add_perlin_noise(xyz, generator=generator)

        # xyz = add_noise_rotate(xyz, generator=generator)
        xyz = add_noise_rotate(xyz, rotate_range=[-180, 180], generator=generator)
        xyz = add_noise_translation(xyz, generator=generator)
        xyz = add_noise_scale(xyz, generator=generator)
    return xyz

def get_patch_grading_colors(sample_points, sample_point_tri_id, cur_edge_dir, next_edge_dir, v0, v1, denom, indices, idx1, idx2, cur_edge_dir_color, next_edge_dir_color, tri_normals):
    def compute_patch_grading_t(diff, next_edge_dir, denom, indices, idx1, idx2, cur_edge_dir_color, tri_normals):
        selected_denom = denom[np.arange(diff.shape[0]), indices]
        t = np.clip(np.cross(diff, next_edge_dir)[np.arange(diff.shape[0]), indices] / selected_denom, 0, 1)
        color = cur_edge_dir_color[:, 0] + t * (cur_edge_dir_color[:, 1] - cur_edge_dir_color[:, 0])
        grad = np.zeros_like(diff)
        grad[np.arange(len(indices)), idx1] = next_edge_dir[np.arange(len(indices)), idx2]
        grad[np.arange(len(indices)), idx2] = -next_edge_dir[np.arange(len(indices)), idx1]
        grad = ((cur_edge_dir_color[:, 1] - cur_edge_dir_color[:, 0]) / selected_denom)[:, None] * grad
        grad = grad - np.sum(grad * tri_normals, axis=-1)[:, None] * tri_normals
        return color, grad

    diff_v0 = sample_points - v0[sample_point_tri_id]
    diff_v1 = sample_points - v1[sample_point_tri_id]
    cur_edge_dir_ = cur_edge_dir[sample_point_tri_id]
    next_edge_dir_ = next_edge_dir[sample_point_tri_id]
    denom_ = denom[sample_point_tri_id]
    indices_ = indices[sample_point_tri_id]
    idx1_ = idx1[sample_point_tri_id]
    idx2_ = idx2[sample_point_tri_id]
    cur_edge_dir_color_ = cur_edge_dir_color[sample_point_tri_id]
    next_edge_dir_color_ = next_edge_dir_color[sample_point_tri_id]
    tri_normals_ = tri_normals[sample_point_tri_id]
    color_v0, grad_v0 = compute_patch_grading_t(
        diff_v0,
        next_edge_dir_,
        denom_,
        indices_,
        idx1_,
        idx2_,
        cur_edge_dir_color_,
        tri_normals_,
    )

    color_v1, grad_v1 = compute_patch_grading_t(
        diff_v1,
        cur_edge_dir_,
        -denom_,
        indices_,
        idx1_,
        idx2_,
        next_edge_dir_color_,
        tri_normals_,
    )
    
    # normalize the gradients
    norm_v0 = np.linalg.norm(grad_v0, axis=1, keepdims=True) + 1e-10
    norm_v1 = np.linalg.norm(grad_v1, axis=1, keepdims=True) + 1e-10
    grad_v0 /= norm_v0
    grad_v1 /= norm_v1


    mask = color_v1 < color_v0
    color_dcdf = np.clip(np.where(mask, color_v1, color_v0), 0, 1)
    grad_dcdf = np.where(mask[:, None], grad_v1, grad_v0)

    mask = 1- color_v1 < 1 - color_v0
    color_cdf = np.clip(np.where(mask, 1 - color_v1, 1 - color_v0), 0, 1)
    grad_cdf = np.where(mask[:, None], -grad_v1, -grad_v0)
    return color_dcdf, grad_dcdf, color_cdf, grad_cdf, color_v0, grad_v0, color_v1, grad_v1 

def load_patch_data_reformat(
    npzfile: str,
    num_surface: int = 16384,
    num_surface_query: int = 50000,
    debug: bool = False,
    file=None,
    fps_num_list = None,
    generator = np.random.default_rng(0),
    fps_return_type = "all", # ["all", "random", "first"]
    is_add_noise=0,
    verbose = False,
):
    """
    checkerboard npz loader
    """
    # Load the npz file
    if file is None:
        # Load the npz file
        npzdata = load_data(npzfile)
        if npzdata is None:
            raise ValueError(f"npz file {npzfile} not found")
    else:
        npzdata = file

    # 0: non-checkerboard, 1: checkerboard, 2: patchboard
    checkerboard_tag = npzdata["is_checkerboard"].squeeze()
    if checkerboard_tag != 2:
        raise ValueError(f"npz file {npzfile} is not a patchboard")

    # transformation
    invT = npzdata["invT"]  # inverse transformation matrix (4 x 3)

    # Original mesh info
    vertex_num  = npzdata["quad_mesh_vertex_num"].squeeze()  # number of vertices
    # quad_vertex = npzdata["quad_vertex"]  # coordinates of quad mesh vertices (nv x 3)
    quad_facet = npzdata["quad_facet"]  # mesh facet indices (nf x 4)

    # SUBDIV MESH INFO
    subdiv_vertex = npzdata["subdiv_vertex"]  # mesh vertices (snv x 3)
    subdiv_facet = npzdata["subdiv_facet"]  # mesh facet indices (snf x 4)
    quad_split = npzdata["quad_split"].squeeze().astype(bool)  # how the quad is splitted into triangles facet indices (snf)

    # PATCH and QUAD INFO
    face2quad = npzdata["face2quad"].squeeze()  # quad face to quad patch id (snf)
    quad2patch = npzdata["quad2patch"].squeeze()  # quad patch id to checker id (nq)

    # OFFSET INFO
    offset_abcd_id = npzdata["offset_abcd_id"]  # offset vertex id of quad patch (nq x 4)
    offset_123_id = npzdata["offset_123_id"]  # offset vertex id of quad patch (pointing other three checkers) (nq x 3)

    # Edge eolor info
    edge_color_store = npzdata["edge_color_store"]  # edge color info (ne x 2, float)
    edge_vertex_color = npzdata["edge_vertex_color"].squeeze()  # edge color info (ne, int)

    ########
    # If you want to add more noise, add your function here
    # vertex_purturbation(subdiv_vertex)
    ########
    # norm to sphere
    center = np.mean(subdiv_vertex, axis=0)
    scale = np.max(np.linalg.norm(subdiv_vertex - center, axis=1))
    subdiv_vertex = (subdiv_vertex - center) / scale

    # apply scale and center to invT
    invT[:, 3] = invT[:, 3] + np.dot(invT[:3, :3], center[:, None]).squeeze()
    invT[:3, :3] = invT[:3, :3] * scale

    if is_add_noise:
        subdiv_vertex = add_noise(subdiv_vertex, generator)

    # TRIANGLE MESH INFO
    tri_faces = subdiv_facet[np.arange(subdiv_facet.shape[0])[:, None], [0, 1, 2, 2, 3, 0]]
    tri_faces_tmp = subdiv_facet[np.arange(subdiv_facet.shape[0])[:, None], [3, 0, 1, 1, 2, 3]]
    tri_faces[quad_split] = tri_faces_tmp[quad_split]
    tri_faces = tri_faces.reshape(-1, 3)  # triangle mesh faces (tnf x 3)

    # OFFSET INFO: index pointing to the subdivided mesh
    offset_abcd = subdiv_vertex[offset_abcd_id]
    offset_123 = subdiv_vertex[offset_123_id]

    # compute tri_normals, quad_sizing, checker_sizing
    tri_normal = np.cross(subdiv_vertex[tri_faces[:, 1]] - subdiv_vertex[tri_faces[:, 0]], subdiv_vertex[tri_faces[:, 2]] - subdiv_vertex[tri_faces[:, 0]])
    tri_area = np.linalg.norm(tri_normal, axis=1, keepdims=True)
    if verbose:
        print(f"tri_area: {tri_area.shape} max: {np.max(tri_area)}, min: {np.min(tri_area)}")
    tri_normal = tri_normal / tri_area
    tri_area = tri_area.squeeze()

    tri_to_quad_map = (np.arange(tri_faces.shape[0], dtype=np.int32) // 2).astype(np.int32)
    tri_to_QUAD = face2quad[tri_to_quad_map]
    if verbose:
        print(f"np.max(tri_to_QUAD): {np.max(tri_to_QUAD)}, QUAD_num: {np.max(tri_to_QUAD) + 1}")
    QUAD_num = np.max(tri_to_QUAD) + 1
    QUAD_area = np.zeros(QUAD_num)
    np.add.at(QUAD_area, tri_to_QUAD, tri_area / 2)
    quad_sizing = np.sqrt(QUAD_area)  # size of QUADs
    if verbose:
        print(f"np.max(quad2patch): {np.max(quad2patch)}, check_num: {np.max(quad2patch) + 1}")
    check_num = np.max(quad2patch) + 1
    checker_area = np.zeros(check_num)
    np.add.at(checker_area, quad2patch, QUAD_area)
    checker_sizing = np.sqrt(checker_area)  # size of checkers

    #compute quad face directions
    subdiv_quad_faces = subdiv_vertex[subdiv_facet]  # quad faces (snf, 4, 3)
    subdiv_quad_dirs = np.stack([subdiv_quad_faces[:, 1] - subdiv_quad_faces[:, 0], subdiv_quad_faces[:, 2] - subdiv_quad_faces[:, 1], subdiv_quad_faces[:, 3] - subdiv_quad_faces[:, 2], subdiv_quad_faces[:, 0] - subdiv_quad_faces[:, 3]], axis=1)  # quad face directions (snf, 4, 3)
    subdiv_quad_dirs = subdiv_quad_dirs / (np.linalg.norm(subdiv_quad_dirs, axis=-1, keepdims=True) + 1e-10)  # quad face directions (snf, 4, 3)
    quad_dir_len_0123 = np.sum(np.cross(subdiv_quad_dirs[:, 0], subdiv_quad_dirs[:, 2], axis=1)**2, axis=1, keepdims=True)  # quad face directions length (snf)
    quad_dir_len_1230 = np.sum(np.cross(subdiv_quad_dirs[:, 1], subdiv_quad_dirs[:, 3], axis=1)**2,axis=1, keepdims=True)# quad face directions length (snf)
    quad_dir = np.where(quad_dir_len_0123 < quad_dir_len_1230, subdiv_quad_dirs[:, 0]-subdiv_quad_dirs[:, 2], subdiv_quad_dirs[:, 1] - subdiv_quad_dirs[:, 3])  # quad face directions (snf, 3)
    quad_dir = np.repeat(quad_dir, 2, axis=0).reshape(-1, 3)  # quad face directions (2*snf, 3)
    dir_check = np.cross(tri_normal, quad_dir, axis=1)
    quad_dir = np.where(dir_check[:, 2, None] > 0, quad_dir, -quad_dir)  
    quad_dir = quad_dir - np.sum(quad_dir* tri_normal, axis=-1)[:, None] * tri_normal 
    quad_dir = quad_dir / (np.linalg.norm(quad_dir, axis=1, keepdims=True) + 1e-10) 

    tri_faces_edge_id = np.tile([0, 1, 2, 3], (subdiv_facet.shape[0], 1))
    tri_faces_edge_id[quad_split] = np.tile([3, 0, 1, 2], (quad_split.sum(), 1))
    tri_faces_edge_id = tri_faces_edge_id.reshape(-1, 2)
    tri_faces_edge_id = np.stack([4 * (np.arange(tri_faces_edge_id.shape[0]) // 2) + tri_faces_edge_id[:, 0], 4 * (np.arange(tri_faces_edge_id.shape[0]) // 2) + tri_faces_edge_id[:, 1]], axis=1)
    tri_faces_edge_color_id = edge_vertex_color[tri_faces_edge_id.reshape(-1)].reshape(tri_faces_edge_id.shape[0], 2)

    # Compute absolute values of edge color IDs for indexing
    tri_faces_edge_color_id_abs = np.abs(tri_faces_edge_color_id)
    # Current edge direction color
    cur_edge_dir_color = edge_color_store[tri_faces_edge_color_id_abs[:, 0]]
    negative_mask_cur = tri_faces_edge_color_id[:, 0] < 0
    cur_edge_dir_color[negative_mask_cur] = cur_edge_dir_color[negative_mask_cur][:, [1, 0]]
    # Next edge direction color
    next_edge_dir_color = edge_color_store[tri_faces_edge_color_id_abs[:, 1]]
    negative_mask_next = tri_faces_edge_color_id[:, 1] < 0
    next_edge_dir_color[negative_mask_next] = next_edge_dir_color[negative_mask_next][:, [1, 0]]

    interpolation_v0 = subdiv_vertex[tri_faces[:, 0]]
    interpolation_v1 = subdiv_vertex[tri_faces[:, 1]]
    cur_edge_dir = interpolation_v1 - interpolation_v0
    next_edge_dir = subdiv_vertex[tri_faces[:, 2]] - interpolation_v1

    denom = np.cross(cur_edge_dir, next_edge_dir)
    indices = np.argmax(np.abs(denom), axis=1)
    cross_indices = [(1, 2), (2, 0), (0, 1)]
    idx1, idx2 = np.array([cross_indices[i] for i in indices]).T

    # SAMPLE POINTS
    sample_bary_coords = npzdata["sample_bary_coords"]  ## barycentric coordinates (N x 3)
    sample_point_tri_id = npzdata["sample_point_tri_id"].squeeze()  ## triangle id (N)
    # compute sample_points using barycentric coordinates
    sample_points = np.einsum("ij,ijk->ik", sample_bary_coords, subdiv_vertex[tri_faces][sample_point_tri_id])
    sample_point_quad_id = tri_to_QUAD[sample_point_tri_id].astype(np.int32)
    sample_point_quad_size = quad_sizing[sample_point_quad_id]
    sample_point_normal = tri_normal[sample_point_tri_id]

    sample_point_dcdf_color_grading, sample_point_dcdf_color_gradient, sample_point_cdf_color_grading, sample_point_cdf_color_gradient, sample_point_xvalue, sample_point_xgradient, sample_point_yvalue, sample_point_ygradient = get_patch_grading_colors(sample_points, sample_point_tri_id, cur_edge_dir, next_edge_dir, interpolation_v0, interpolation_v1, denom, indices, idx1, idx2, cur_edge_dir_color, next_edge_dir_color, tri_normal)
    sample_point_offset_abcd = offset_abcd[sample_point_quad_id] - sample_points[:, None, :]
    sample_point_offset_123 = offset_123[sample_point_quad_id] - sample_points[:, None, :]
    sample_point_direction = quad_dir[sample_point_tri_id]

    # FPS POINTS
    if fps_num_list is None:
        fps_num_list = [int(k.split("_")[-1]) for k in npzdata.keys() if k.startswith("fps_bary_coords_")]
    fps_dict = {}
    for fps_num in fps_num_list:
        fps_num_real = fps_num
        if f"fps_bary_coords_{fps_num_real}" not in npzdata:
            # random sample points from sample_points
            maxn = sample_bary_coords.shape[0]
            fps_idx_random = generator.choice(maxn, size=fps_num, replace=maxn < fps_num)
            fps_bary_coords = sample_bary_coords[fps_idx_random][None, ...]
            fps_point_tri_id = sample_point_tri_id[fps_idx_random].squeeze()[None, ...]
        else:
            fps_bary_coords = npzdata[f"fps_bary_coords_{fps_num_real}"]  ## barycentric coordinates of fps points (M x nfps x 3)
            fps_point_tri_id = npzdata[f"fps_point_tri_id_{fps_num_real}"].squeeze()  ## triangle id of fps points (M x nfps)
        if fps_bary_coords.ndim == 2:
            fps_bary_coords = fps_bary_coords[None, :, :]
        if fps_bary_coords.ndim == 1:
            fps_bary_coords = fps_bary_coords[None, :]

        fps_points = np.einsum("ijkl,ijk->ijl", subdiv_vertex[tri_faces][fps_point_tri_id], fps_bary_coords)
        fps_point_quad_id = tri_to_QUAD[fps_point_tri_id].astype(np.int32)
        fps_point_normal = tri_normal[fps_point_tri_id]
        fps_point_grading_dcdf_color, fps_point_grading_dcdf_color_gradient, fps_point_grading_cdf_color, fps_point_grading_cdf_color_gradient, fps_point_grading_xvalue, fps_point_grading_xgradient, fps_point_grading_yvalue, fps_point_grading_ygradient = get_patch_grading_colors(fps_points.reshape(-1, 3), fps_point_tri_id.reshape(-1), cur_edge_dir, next_edge_dir, interpolation_v0, interpolation_v1, denom, indices, idx1, idx2, cur_edge_dir_color, next_edge_dir_color, tri_normal)

        fps_point_grading_dcdf_color = fps_point_grading_dcdf_color.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1])
        fps_point_grading_dcdf_color_gradient = fps_point_grading_dcdf_color_gradient.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1], 3)
        fps_point_grading_cdf_color = fps_point_grading_cdf_color.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1])
        fps_point_grading_cdf_color_gradient = fps_point_grading_cdf_color_gradient.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1], 3)
        fps_point_grading_xvalue = fps_point_grading_xvalue.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1])
        fps_point_grading_xgradient = fps_point_grading_xgradient.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1], 3)
        fps_point_grading_yvalue = fps_point_grading_yvalue.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1])
        fps_point_grading_ygradient = fps_point_grading_ygradient.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1], 3)

        fps_point_offset_abcd = offset_abcd[fps_point_quad_id] - fps_points[:, :, None, :]
        fps_point_offset_123 = offset_123[fps_point_quad_id] - fps_points[:, :, None, :]

        tmp_all = {
            f"xyz_fps_{fps_num}": fps_points,
            f"normal_fps_{fps_num}": fps_point_normal,
            f"dcdf_fps_{fps_num}": fps_point_grading_dcdf_color[..., None],
            f"gdcdf_fps_{fps_num}": fps_point_grading_dcdf_color_gradient,
            f"cdf_fps_{fps_num}": fps_point_grading_cdf_color[..., None],
            f"gcdf_fps_{fps_num}": fps_point_grading_cdf_color_gradient,
            **convert_offset_to_oldformat(f"offset_abcd_fps_{fps_num}", fps_point_offset_abcd),
            **convert_offset_to_oldformat(f"offset_123_fps_{fps_num}", fps_point_offset_123),
        }
        tmp = copy.deepcopy(tmp_all)
        if fps_return_type == "all":
            pass
        elif fps_return_type == "random":
            idx_fps_select = generator.integers(0, fps_points.shape[0])
            tmp = {k: v[idx_fps_select] for k, v in tmp.items()}
        elif fps_return_type == "first":
            idx_fps_select = 0
            tmp = {k: v[idx_fps_select] for k, v in tmp.items()}
        else:
            assert False

        fps_dict.update(tmp)

    ans_surface_all = {
        "xyz": sample_points,
        "normal": sample_point_normal,
        "dcdf": sample_point_dcdf_color_grading[..., None],
        "gdcdf": sample_point_dcdf_color_gradient,
        "cdf": sample_point_cdf_color_grading[..., None],
        "gcdf": sample_point_cdf_color_gradient,
        "fielddir": sample_point_direction,
        **convert_offset_to_oldformat("offset_abcd", sample_point_offset_abcd),
        **convert_offset_to_oldformat("offset_123", sample_point_offset_123),
        "quadsize": sample_point_quad_size[..., None],
    }
    idx_surface = get_random_index(num_surface, sample_points.shape[0], generator)
    ans_surface = {k: v[idx_surface] for k, v in ans_surface_all.items()}
    ans_surface["quadid"] = sample_point_quad_id[idx_surface]

    idx_query = get_random_index(num_surface_query, sample_points.shape[0], generator)
    ans_query = {f"{k}_query": v[idx_query] for k, v in ans_surface_all.items()}

    return {
        **ans_surface,
        **ans_query,

        "checker_sizing": checker_sizing,
        "quadsize_mean": np.array([np.mean(quad_sizing)]),
        "n_quad": np.array([quad_facet.shape[0]]),
        **fps_dict,
        "invT": invT.astype(np.float32),
    }
