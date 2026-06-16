"""Flip the embedded texture of a GLB along U and/or V and write copies next to the original.

For each input GLB this writes 3 variants in the same directory:
    <stem>_flipU.glb
    <stem>_flipV.glb
    <stem>_flipUV.glb

Usage:
    python debug_script/flip_glb_texture.py results2/infer_sqdiffuse/test_data_1024_smooth/data/3_holes/gen_000/gen0.glb
"""

import argparse
import copy
import io
from pathlib import Path

import trimesh
from PIL import Image, ImageOps


def get_visual(geom):
    vis = geom.visual
    if hasattr(vis, "to_texture"):
        try:
            vis = vis.to_texture()
        except Exception:
            pass
    return vis


def flip_geom(geom, axis: str):
    """axis in {'U', 'V', 'UV'}. Returns a new geometry with flipped texture."""
    geom = copy.deepcopy(geom)
    vis = get_visual(geom)
    img = vis.material.baseColorTexture if hasattr(vis, "material") and getattr(vis.material, "baseColorTexture", None) is not None else getattr(vis, "image", None)
    if img is None:
        raise RuntimeError("Geometry has no baseColorTexture / image to flip")
    if axis in ("U", "UV"):
        img = ImageOps.mirror(img)
    if axis in ("V", "UV"):
        img = ImageOps.flip(img)
    if hasattr(vis, "material") and getattr(vis.material, "baseColorTexture", None) is not None:
        vis.material.baseColorTexture = img
    else:
        vis.image = img
    geom.visual = vis
    return geom


def export_variant(src_path: Path, axis: str):
    scene = trimesh.load(src_path, process=False)
    if isinstance(scene, trimesh.Scene):
        new_geoms = {name: flip_geom(g, axis) for name, g in scene.geometry.items()}
        new_scene = trimesh.Scene(new_geoms)
    else:
        new_scene = flip_geom(scene, axis)
    out = src_path.with_name(f"{src_path.stem}_flip{axis}.glb")
    new_scene.export(out)
    print(f"  wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", nargs="+", help="GLB file(s) to process")
    args = parser.parse_args()
    for p in args.glb:
        p = Path(p).resolve()
        print(f"Processing {p}")
        for axis in ("U", "V", "UV"):
            export_variant(p, axis)


if __name__ == "__main__":
    main()
