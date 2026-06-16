"""Merge chunked checkpoints into single .pth files and upload to a HuggingFace model repo.

Layout uploaded:
    ckpts/
        sqvae_geomae.pth
        sqvae_geomae.yaml
        sqdiffuse.pth
        sqdiffuse.yaml

Background:
    Checkpoints in this repo may be saved as `<name>.pth.chunk0`, `<name>.pth.chunk1`, ...
    (see `save_model_chunk` / `load_model_chunk` in squadgen/util/misc.py). The chunks are
    raw bytes of a single torch.save() blob split every 512 MB, so converting to single
    file mode is just concatenating the chunks in numeric order.

Usage:
    python debug_script/upload_to_hf.py \
        --repo_id <user_or_org>/<repo_name> \
        --ae_pth    /path/to/ae/checkpoint-375.pth \
        --model_pth /path/to/sqdiffuse/checkpoint-194.pth \
        [--private] [--strip] \
        [--export_dir ./exported_ckpts] \
        [--keep_merged] [--skip_upload]
"""

import argparse
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo


CHUNK_READ_BUFFER = 16 * 1024 * 1024  # 16 MB

REMOTE_DIR = "ckpts"
AE_NAME = "sqvae_geomae"
MODEL_NAME = "sqdiffuse"

REPO_ROOT = Path(__file__).resolve().parent.parent
AE_CONFIG_SRC = REPO_ROOT / "squadgen" / "network" / "config" / f"{AE_NAME}.yaml"
MODEL_CONFIG_SRC = REPO_ROOT / "squadgen" / "network" / "config" / f"{MODEL_NAME}.yaml"


def list_chunks(pth_path: Path):
    chunks = []
    i = 0
    while True:
        c = pth_path.with_name(pth_path.name + f".chunk{i}")
        if not c.is_file():
            break
        chunks.append(c)
        i += 1
    return chunks


def merge_chunks(pth_path: Path, out_path: Path) -> Path:
    chunks = list_chunks(pth_path)
    if not chunks:
        raise FileNotFoundError(
            f"No single file and no chunks found for {pth_path}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(c.stat().st_size for c in chunks)
    print(f"  merging {len(chunks)} chunks ({total / 1024**3:.2f} GiB) -> {out_path}")
    with open(out_path, "wb") as dst:
        for c in chunks:
            with open(c, "rb") as src:
                shutil.copyfileobj(src, dst, length=CHUNK_READ_BUFFER)
    return out_path


def strip_checkpoint(pth_path: Path):
    """Drop optimizer/scaler. Keeps 'model', 'epoch', 'args' (enough for inference)."""
    print(f"  stripping optimizer/scaler from {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    keep = {k: ckpt[k] for k in ("model", "epoch", "args") if k in ckpt}
    torch.save(keep, pth_path)
    print(f"  after strip: {pth_path.stat().st_size / 1024**3:.2f} GiB")


def prepare_single_file(pth_path: Path, out_path: Path, strip: bool) -> Path:
    """Return a path to a single-file .pth at `out_path` (always materialized there)."""
    pth_path = pth_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if pth_path.is_file():
        print(f"  copying {pth_path} -> {out_path}")
        shutil.copyfile(pth_path, out_path)
    else:
        merge_chunks(pth_path, out_path)

    if strip:
        strip_checkpoint(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="e.g. DQS/SQuadGen")
    parser.add_argument("--ae_pth", required=True)
    parser.add_argument("--model_pth", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--strip", action="store_true",
                        help="Drop optimizer/scaler before upload (inference-only).")
    parser.add_argument("--export_dir", default="exported_ckpts",
                        help="Local staging dir for the renamed single-file ckpts.")
    parser.add_argument("--keep_merged", action="store_true",
                        help="Keep merged checkpoints locally after upload.")
    parser.add_argument("--skip_upload", action="store_true",
                        help="Only merge/strip locally; do not upload.")
    args = parser.parse_args()

    if not AE_CONFIG_SRC.is_file():
        raise FileNotFoundError(AE_CONFIG_SRC)
    if not MODEL_CONFIG_SRC.is_file():
        raise FileNotFoundError(MODEL_CONFIG_SRC)

    export_dir = Path(args.export_dir).expanduser().resolve() / REMOTE_DIR
    ae_local = export_dir / f"{AE_NAME}.pth"
    model_local = export_dir / f"{MODEL_NAME}.pth"
    ae_cfg_local = export_dir / f"{AE_NAME}.yaml"
    model_cfg_local = export_dir / f"{MODEL_NAME}.yaml"

    print("[1/4] Preparing AE checkpoint")
    prepare_single_file(Path(args.ae_pth), ae_local, args.strip)
    print(f"  ready: {ae_local}\n")

    print("[2/4] Preparing diffusion checkpoint")
    prepare_single_file(Path(args.model_pth), model_local, args.strip)
    print(f"  ready: {model_local}\n")

    print("[3/4] Copying configs")
    shutil.copyfile(AE_CONFIG_SRC, ae_cfg_local)
    shutil.copyfile(MODEL_CONFIG_SRC, model_cfg_local)
    print(f"  {ae_cfg_local}")
    print(f"  {model_cfg_local}\n")

    if args.skip_upload:
        print("[4/4] --skip_upload set, exiting.")
        return

    print("[4/4] Uploading to HuggingFace")
    api = HfApi()
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    uploads = [
        (ae_local, f"{REMOTE_DIR}/{AE_NAME}.pth"),
        (ae_cfg_local, f"{REMOTE_DIR}/{AE_NAME}.yaml"),
        (model_local, f"{REMOTE_DIR}/{MODEL_NAME}.pth"),
        (model_cfg_local, f"{REMOTE_DIR}/{MODEL_NAME}.yaml"),
    ]
    for local, remote in uploads:
        print(f"  {local}  ->  {args.repo_id}:{remote}")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=args.repo_id,
            repo_type="model",
        )

    if not args.keep_merged:
        for local, _ in uploads:
            if local.is_file():
                print(f"  cleaning up {local}")
                local.unlink()
        try:
            export_dir.rmdir()
            export_dir.parent.rmdir()
        except OSError:
            pass

    print(f"\nDone: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
