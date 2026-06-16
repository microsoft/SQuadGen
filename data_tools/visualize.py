import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export visualization files from a CDF patch npz sample."
    )
    parser.add_argument("--input", "-i", required=True, help="Input .npz file.")
    parser.add_argument(
        "--outdir",
        "-d",
        default="data_tools/results/visualize",
        help="Folder for visualization output. Defaults to data_tools/results/visualize.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    input_file = Path(args.input).expanduser()
    if not input_file.is_absolute():
        input_file = (repo_root / input_file).resolve()
    else:
        input_file = input_file.resolve()

    if input_file.suffix.lower() != ".npz":
        raise ValueError(f"Input must be a .npz file: {input_file}")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = (repo_root / outdir).resolve()
    else:
        outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    from data_tools.test_load_new_format import load_patch_data_reformat
    from squadgen.network.utils import save_all

    data = load_patch_data_reformat(
        str(input_file),
        num_surface=50000,
        is_add_noise=0,
        fps_return_type="first",
        debug=0,
        verbose=True,
    )
    # print keys and value shape of data
    for key, value in data.items():
        print(f"{key}: {value.shape}")
    save_all(data, str(outdir))
    print(f"Visualization output: {outdir}")


if __name__ == "__main__":
    main()