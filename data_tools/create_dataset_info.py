import os
import json
import argparse
import glob
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create all/train/test/rest file lists from npz files in a folder."
    )
    parser.add_argument(
        "--input-folder",
        "-i",
        required=True,
        help="Folder containing generated npz files.",
    )
    parser.add_argument(
        "--outdir",
        "-d",
        required=True,
        help="Folder to save _all_filelist.txt, _train_filelist.txt, _test_filelist.txt, and _rest_filelist.txt.",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=128,
        help="Number of test examples. Defaults to 128.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.95,
        help="Train ratio among non-test examples. Defaults to 0.95.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for splitting. Defaults to 0.",
    )
    return parser.parse_args()


def resolve_path(path, repo_root):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def write_filelist(outdir, name, filelist):
    save_fn = outdir / f"_{name}_filelist.txt"
    print(f"Save {save_fn}, len: {len(filelist)}")
    with save_fn.open("w") as file_obj:
        file_obj.write("\n".join(str(item) for item in filelist))


def main():
    args = parse_args()

    if args.test_num < 0:
        raise ValueError(f"--test-num must be >= 0, got {args.test_num}")
    if args.train_ratio < 0 or args.train_ratio > 1:
        raise ValueError(f"--train-ratio must be in [0, 1], got {args.train_ratio}")

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    input_folder = resolve_path(args.input_folder, repo_root)
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_folder}")

    outdir = resolve_path(args.outdir, repo_root)
    outdir.mkdir(parents=True, exist_ok=True)

    filelist_all = sorted(glob.glob(str(input_folder / "**" / "*.npz"), recursive=True))
    filelist_all = [Path(item).resolve() for item in filelist_all]
    print(f"Total {len(filelist_all)} npz files in {input_folder}")

    if len(filelist_all) < args.test_num:
        raise ValueError(
            f"Need at least {args.test_num} npz files for test split, got {len(filelist_all)}."
        )

    generator = np.random.default_rng(seed=args.seed)
    shuffled_filelist = list(generator.permutation(filelist_all))

    if args.test_num == 0:
        test_filelist = []
        remaining_filelist = shuffled_filelist
    else:
        test_filelist = shuffled_filelist[-args.test_num :]
        remaining_filelist = shuffled_filelist[: -args.test_num]

    train_num = int(len(remaining_filelist) * args.train_ratio)
    train_filelist = remaining_filelist[:train_num]
    rest_filelist = remaining_filelist[train_num:]

    print(
        f"Split: all={len(filelist_all)}, train={len(train_filelist)}, "
        f"test={len(test_filelist)}, rest={len(rest_filelist)}"
    )

    write_filelist(outdir, "all", filelist_all)
    write_filelist(outdir, "train", train_filelist)
    write_filelist(outdir, "test", test_filelist)
    write_filelist(outdir, "rest", rest_filelist)


    # export all labeling_fn
    data_fn_new = os.path.join(outdir, "labeling_all.json")
    print(f"Save labeling_all.json: {data_fn_new}")
    data_new = [[1.0, []]]
    for idx, x in enumerate(range(train_num)):
        data_new[0][1].append(idx)

    with open(data_fn_new, "w") as f:
        json.dump(data_new, f, indent=4)


if __name__ == "__main__":
    main()
