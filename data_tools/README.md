# data_tools

Python utilities for preparing SQuadGen training data. The scripts here are thin
wrappers around the `CDFGen` executable from [QuadTools](https://github.com/xueyuhanlang/QuadTools4SquadGen)
submodule, plus helpers for building training file lists and visualizing CDF
NPZ samples.

## Requirements

Build the `CDFGen` executable from the QuadTools submodule (see
[QuadTools/README.md](https://github.com/xueyuhanlang/QuadTools4SquadGen) or the Installation Steps in the
project [README.md](../README.md)). After building, the executable should be
available at `QuadTools/build/CDFGen`.

## Generate NPZ Data

Use `data_tools.gen_npz_data` to generate CDF patch NPZ samples. The `--input`
argument auto-detects the mode:

- A single `.ply` / `.obj` mesh -> single-file mode.
- A `.txt` file (one `.ply` path per non-empty line) -> batch mode. `--start` /
  `--end` select a 0-based `[start, end)` slice of the list and are useful for
  splitting a large dataset across jobs.

Single-file example:

```bash
python -m data_tools.gen_npz_data \
  --input examples/59a7e911d0ed408f89bfd3010564c03e.ply
```

This example mesh contains 20 parts, so the generated files match:

```text
data_tools/results/output/59a7e911d0ed408f89bfd3010564c03e_*.npz
```

Batch example:

```bash
python -m data_tools.gen_npz_data \
  --input path/to/ply_filelist.txt \
  --start 0 \
  --end 1000 \
  --outdir data_tools/results/output
```

By default, outputs are written to `data_tools/results/output`. Use `--outdir`
for a different folder and `--num-points` to change the number of sampled
points. In batch mode, a failed input is skipped and the script continues with
the remaining files.

The script invokes the QuadTools `CDFGen` executable per input as:

```bash
QuadTools/build/CDFGen \
  -i <input_mesh> \
  -d <output_folder> \
  -n 50000
```

## Create Training File Lists

After generating NPZ files, use `data_tools.create_dataset_info` to create file
lists for training. It recursively collects all `.npz` files from an input
folder and writes the split files to the output folder:

```bash
python -m data_tools.create_dataset_info \
  --input-folder data_tools/results/output \
  --outdir data_tools/results/dataset_info
```

The output folder contains:

```text
_all_filelist.txt
_train_filelist.txt
_test_filelist.txt
_rest_filelist.txt
labeling_all.json
```

By default, `test` contains 128 examples, `train` contains 95% of the remaining
examples, and `rest` contains the final 5%. Use `--test-num`, `--train-ratio`,
and `--seed` to change these defaults.

`labeling_all.json` is used by the SQuadGen training dataloader for balanced
sampling. The training scripts pass it with `--labeling_fn=labeling_all.json`.
The dataloader loads it from `dataset_folder/data_filter_name/labeling_all.json`
when `data_filter_name` is set, or from `dataset_folder/labeling_all.json`
otherwise.

The file is a list of sampling groups:

```json
[
  [1.0, [0, 1, 2, 3]]
]
```

Each group has two fields:

- The first value is the sampling ratio for that group. All ratios should sum to
  `1.0`.
- The second value is a list of indices into `_train_filelist.txt`.

During training, the dataloader first selects a group according to the ratios,
then randomly samples one training file index from that group. The default
`labeling_all.json` produced by `create_dataset_info` puts all training files
into one group with ratio `1.0`, which keeps all training samples available
while using the same balanced-sampling code path.

## Visualize an NPZ Sample

Use `data_tools.visualize` to export visualization files from a generated NPZ
sample:

```bash
python -m data_tools.visualize \
  --input examples/59a7e911d0ed408f89bfd3010564c03e_3.npz
```

By default, files written by `save_all` are saved to:

```text
data_tools/results/visualize
```

Use `--outdir` to choose a different visualization folder:

```bash
python -m data_tools.visualize \
  --input examples/59a7e911d0ed408f89bfd3010564c03e_3.npz \
  --outdir /path/to/visualize_folder
```
