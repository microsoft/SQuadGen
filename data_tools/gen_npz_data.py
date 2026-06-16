import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate CDF patch npz data via the QuadTools CDFGen executable. "
            "Input can be either a single .ply/.obj file, or a .txt file listing "
            "one .ply path per line for batch processing."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input mesh (.ply/.obj) for single mode, or .txt file listing .ply paths for batch mode.",
    )
    parser.add_argument(
        "--outdir",
        "-d",
        default="data_tools/results/output",
        help="Folder for generated npz data. Defaults to data_tools/results/output.",
    )
    parser.add_argument("--num-points", "-n", type=int, default=50000)
    parser.add_argument(
        "--start",
        "-s",
        type=int,
        default=0,
        help="Batch mode only: start index in the txt file, inclusive and 0-based.",
    )
    parser.add_argument(
        "--end",
        "-e",
        type=int,
        default=None,
        help="Batch mode only: end index in the txt file, exclusive and 0-based. Defaults to the end of the list.",
    )
    return parser.parse_args()


def resolve_path(path, repo_root):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def run_command(cmd, cwd):
    print(" ".join(str(item) for item in cmd), flush=True)
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")

    if result.returncode != 0:
        print(f"CDFGen failed with exit code {result.returncode}. Skip this input.")
        return False
    return True


def load_input_files(input_list, repo_root):
    files = []
    with input_list.open("r") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            path_text = line.strip()
            if not path_text:
                continue

            input_file = resolve_path(path_text, repo_root)
            if input_file.suffix.lower() != ".ply":
                print(f"Line {line_no}: input is not a .ply file, skip: {input_file}")
                continue
            files.append(input_file)
    return files


def run_one(executable, repo_root, input_file, outdir, num_points):
    return run_command(
        [
            str(executable.relative_to(repo_root)),
            "-i",
            str(input_file),
            "-d",
            str(outdir),
            "-n",
            str(num_points),
        ],
        cwd=repo_root,
    )


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    input_path = resolve_path(args.input, repo_root)
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    outdir = resolve_path(args.outdir, repo_root)
    outdir.mkdir(parents=True, exist_ok=True)

    executable = repo_root / "QuadTools" / "build" / "CDFGen"
    if not executable.exists():
        raise FileNotFoundError(
            f"CDFGen executable does not exist: {executable}. Build QuadTools first "
            f"(see QuadTools/README.md or the Installation Steps in the project README)."
        )

    suffix = input_path.suffix.lower()
    if suffix == ".txt":
        if args.start < 0:
            raise ValueError(f"--start must be >= 0, got {args.start}")
        if args.end is not None and args.end < args.start:
            raise ValueError(f"--end must be >= --start, got start={args.start}, end={args.end}")

        input_files = load_input_files(input_path, repo_root)
        selected_files = input_files[args.start : args.end]
        if not selected_files:
            print(
                f"No input files selected from {input_path}. "
                f"List size={len(input_files)}, start={args.start}, end={args.end}."
            )
            return

        end_for_log = args.end if args.end is not None else len(input_files)
        print(
            f"Run CDFGen for {len(selected_files)} files from {input_path} "
            f"with range [{args.start}, {end_for_log})."
        )

        success_count = 0
        failed_files = []
        for index, input_file in enumerate(selected_files, start=args.start):
            print(f"\n[{index}] Processing: {input_file}", flush=True)
            if not input_file.exists():
                print(f"Input file does not exist. Skip this input: {input_file}")
                failed_files.append(input_file)
                continue

            if run_one(executable, repo_root, input_file, outdir, args.num_points):
                success_count += 1
            else:
                failed_files.append(input_file)

        print(f"\nDone. Success: {success_count}, failed: {len(failed_files)}.")
        if failed_files:
            print("Failed files:")
            for failed_file in failed_files:
                print(failed_file)
        return

    if suffix not in {".ply", ".obj"}:
        raise ValueError(
            f"Input must be a .ply/.obj mesh or a .txt listing .ply paths: {input_path}"
        )

    if not run_one(executable, repo_root, input_path, outdir, args.num_points):
        return

    raw_npz = outdir / f"{input_path.stem}_0.npz"
    if not raw_npz.exists():
        print(f"Expected npz was not generated: {raw_npz}. Skip this input.")
        return

    print(f"Generated npz: {raw_npz}")


if __name__ == "__main__":
    main()
