# NGVM

Research code for NPR-guided VMamba deepfake detection. The model implementation is in `networks/`; `train.py` trains it, `validate.py` provides the data-set evaluation and pretrained-weight mapping, and `test.py` evaluates a saved checkpoint across the configured benchmarks.

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/*.py` | Data loading and image transforms (tracked source code) |
| `networks/` | Model architecture and training wrapper |
| `options/` | Command-line options |
| `train.py` | Training and per-epoch diagnostics |
| `validate.py` | Validation metrics and VMamba weight mapping |
| `test.py` | Benchmark evaluation and CSV export |
| `requirements.txt` | Current Python package snapshot; see environment note below |

Datasets, pretrained weights, checkpoints, logs, and compiled wheels are external artifacts and are not stored in this repository.

## Environment

The reported AutoDL environment uses Python 3.10, PyTorch `2.9.0+cu128`, and the CUDA 12.8 runtime. For a fresh environment, install the matching PyTorch build first, then the pinned Python packages:

```bash
conda create -n NGVM python=3.10 -y
conda activate NGVM
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The PyTorch command follows the [official instructions for version 2.9.0 with CUDA 12.8](https://pytorch.org/get-started/previous-versions/). Verify the resulting environment:

```bash
python -c "import sys, torch; print(sys.version); print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
python -m pip show mamba-ssm
```

`requirements.txt` pins `torch==2.9.0` but does not include the locally installed `mamba-ssm` wheel. A wheel previously used with this project is labeled for **PyTorch 2.2**, so it is not a verified match for the reported PyTorch 2.9.0 environment. Obtain or build compiled selective-scan/Mamba extensions for the actual Python, PyTorch, CUDA, and C++ ABI combination before training. The model calls `selective_scan_cuda_oflex`; installing `mamba-ssm` alone does not establish that this backend is available. Record the wheel filenames and SHA-256 hashes alongside the experiment. Do not reuse the old wheel solely because `pip freeze` recorded its local path.

After installing those matching extensions, check that the VMamba CUDA backend imports:

```bash
python -c "import torch, selective_scan_cuda_oflex; print(torch.__version__, torch.version.cuda, 'oflex OK')"
```

## Data and pretrained weights

`--dataroot` must point to a dataset root containing `train/`, `val/`, and `test/`. The training script evaluates the eight test subsets named in `train.py` (ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN, and Deepfake); prepare them under `test/` using the layout expected by `data/datasets.py`. Keep image files outside the Git repository.

Fresh training and standalone `validate.py` load the VMamba Small checkpoint from `--pretrain_path`. The default location is:

```text
<NGVM repository>/pretrained/vssmsmall_dp03_ckpt_epoch_238.pth
```

Place the checkpoint there, pass another absolute path with `--pretrain_path`, or set `NGVM_PRETRAIN_PATH`. The checkpoint is a local artifact and is excluded from Git.

## Training

Run from the NGVM repository root in the configured environment. Set `DATA_ROOT` to the dataset root described above and `RUN_ROOT` to a location with enough space for logs and checkpoints. Supply `--classes` when your training layout requires an explicit comma-separated class list.

```bash
cd ~/autodl-tmp/NGVM
DATA_ROOT=/absolute/path/to/dataset_root
RUN_ROOT=/root/autodl-tmp/ngvm-runs
PRETRAIN=/absolute/path/to/vssmsmall_dp03_ckpt_epoch_238.pth

python train.py \
  --dataroot "$DATA_ROOT" \
  --pretrain_path "$PRETRAIN" \
  --name ngvm \
  --niter 50 \
  --batch_size 64 \
  --checkpoints_dir "$RUN_ROOT/checkpoints"
```

The run name receives a timestamp. Training writes TensorBoard logs and **only one checkpoint**, `model_epoch_last.pth`, under `RUN_ROOT/checkpoints/<run-name>/`. This file contains model weights only. `--continue_train` is intentionally disabled because exact resumption would require optimizer, scheduler, and epoch state. Training still reports validation and test-subset metrics every epoch; those test-domain reports are diagnostic and should not be treated as an untouched final test set when selecting experiments.

## Evaluation

Pass the final checkpoint explicitly with `--model_path`; run names contain timestamps, so the script does not guess which run to evaluate.

```bash
cd ~/autodl-tmp/NGVM
CHECKPOINT=/absolute/path/to/model_epoch_last.pth
BENCHMARK_ROOT=/absolute/path/to/Generalization_Test
RESULTS_CSV=/root/autodl-tmp/ngvm-runs/results/ngvm_tables.csv

python test.py \
  --model_path "$CHECKPOINT" \
  --benchmark_root "$BENCHMARK_ROOT" \
  --results_csv "$RESULTS_CSV" \
  --batch_size 64
```

The default benchmark directory is `<NGVM repository>/data/datasets/Generalization_Test`, and the default CSV path is `<NGVM repository>/results/ngvm_tables.csv`. Override them with command-line arguments or `NGVM_BENCHMARK_ROOT` and `NGVM_RESULTS_CSV`.
