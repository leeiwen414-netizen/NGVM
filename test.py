"""Evaluate a frozen NPR/VMamba checkpoint on NPR Tables 1--5.

Table 1 extends the NPR paper's original eight ForenSynths subsets
with CRN, IMLE, SITD, SAN, and WFIR. Consequently, the final macro average is
computed over 43 sub-tests instead of the paper's original 38-sub-test Table 6.

Cloud example:
    python test.py \
        --model_path /root/autodl-tmp/NGVM/checkpoints/YOUR_RUN_NAME/model_epoch_last.pth \
        --benchmark_root /root/autodl-tmp/NGVM/data/datasets/Generalization_Test \
        --batch_size 64 \
        --results_csv ./results/ngvm_tables.csv

``--benchmark_root`` and ``--results_csv`` are parsed locally in this script;
the remaining arguments are handled by TestOptions. This keeps the file
compatible with the older cloud copy of options/test_options.py.
"""

import argparse
import csv
import inspect
import os
import random
import time
from collections.abc import Mapping

import numpy as np
import torch

from networks.NVPR import NPRVMamba
from options.test_options import TestOptions
from util import printSet
from validate import validate


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BENCHMARK_ROOT = os.path.join(PROJECT_ROOT, "data", "datasets", "Generalization_Test")
DEFAULT_RESULTS_CSV = os.path.join(PROJECT_ROOT, "results", "ngvm_tables.csv")


# The first eight Table 1 entries and Tables 2--5 follow the NPR paper. The
# five additional ForenSynths subsets are appended to Table 1. Directory names
# are kept separate from paper labels where the public dataset uses aliases.
BENCHMARKS = [
    {
        "table": "Table 1 (Extended)",
        "dataset": "ForenSynths",
        "root_parts": ("ForenSynths_test", "test"),
        "no_resize": False,
        "no_crop": True,
        "subsets": [
            ("progan", "ProGAN"),
            ("stylegan", "StyleGAN"),
            ("stylegan2", "StyleGAN2"),
            ("biggan", "BigGAN"),
            ("cyclegan", "CycleGAN"),
            ("stargan", "StarGAN"),
            ("gaugan", "GauGAN"),
            ("deepfake", "Deepfake"),
            ("crn", "CRN"),
            ("imle", "IMLE"),
            ("seeingdark", "SITD"),
            ("san", "SAN"),
            ("whichfaceisreal", "WFIR"),
        ],
    },
    {
        "table": "Table 2",
        "dataset": "GANGen-Detection",
        "root_parts": ("GANGen-Detection",),
        "no_resize": True,
        "no_crop": True,
        "subsets": [
            ("AttGAN", "AttGAN"),
            ("BEGAN", "BEGAN"),
            ("CramerGAN", "CramerGAN"),
            ("InfoMaxGAN", "InfoMaxGAN"),
            ("MMDGAN", "MMDGAN"),
            ("RelGAN", "RelGAN"),
            ("S3GAN", "S3GAN"),
            ("SNGAN", "SNGAN"),
            ("STGAN", "STGAN"),
        ],
    },
    {
        "table": "Table 3",
        "dataset": "DiffusionForensics/DIRE",
        "root_parts": ("DiffusionForensics",),
        "no_resize": False,
        "no_crop": True,
        "subsets": [
            ("adm", "ADM"),
            ("ddpm", "DDPM"),
            ("iddpm", "IDDPM"),
            ("ldm", "LDM"),
            ("pndm", "PNDM"),
            ("vqdiffusion", "VQ-Diffusion"),
            ("sdv1_new", "Stable Diffusion v1"),
            ("sdv2", "Stable Diffusion v2"),
        ],
    },
    {
        "table": "Table 4",
        "dataset": "UniversalFakeDetect/Ojha",
        "root_parts": ("UniversalFakeDetect",),
        "no_resize": False,
        "no_crop": True,
        "subsets": [
            ("dalle", "DALLE"),
            ("glide_100_10", "Glide 100/10"),
            ("glide_100_27", "Glide 100/27"),
            ("glide_50_27", "Glide 50/27"),
            ("guided", "ADM"),
            ("ldm_100", "LDM 100"),
            ("ldm_200", "LDM 200"),
            ("ldm_200_cfg", "LDM 200 cfg"),
        ],
    },
    {
        "table": "Table 5",
        "dataset": "Diffusion1kStep",
        "root_parts": ("Diffusion1kStep",),
        "no_resize": False,
        "no_crop": True,
        "subsets": [
            ("ddpm", "DDPM"),
            ("improved-diffusion", "IDDPM"),
            ("guided-diffusion", "ADM"),
            ("midjourney", "Midjourney"),
            ("DALLE", "DALLE"),
        ],
    },
]

EXPECTED_SUBTESTS = 43
CSV_FIELDS = [
    "kind",
    "table",
    "dataset",
    "subset",
    "directory",
    "acc_percent",
    "ap_percent",
    "real_acc_percent",
    "fake_acc_percent",
    "num_samples",
    "num_real",
    "num_fake",
]


def parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--benchmark_root",
        default=os.environ.get("NGVM_BENCHMARK_ROOT", DEFAULT_BENCHMARK_ROOT),
    )
    parser.add_argument(
        "--results_csv",
        default=os.environ.get("NGVM_RESULTS_CSV", DEFAULT_RESULTS_CSV),
    )
    args, _ = parser.parse_known_args()
    return args


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(opt):
    """Instantiate both the legacy TM3 model and the newer platform model."""
    parameters = inspect.signature(NPRVMamba.__init__).parameters
    kwargs = {"num_classes": 1}
    if "artifact_dim" in parameters:
        kwargs["artifact_dim"] = int(getattr(opt, "artifact_dim", 16))
    return NPRVMamba(**kwargs)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a model state dictionary.")

    # DataParallel checkpoints commonly add this prefix.
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(model, model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n" "Pass its actual cloud path with --model_path."
        )

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "The checkpoint and networks/NVPR.py are not the same architecture. "
            "Use the model code that produced the supplied checkpoint."
        ) from exc


def benchmark_root(spec, root):
    return os.path.join(root, *spec["root_parts"])


def check_benchmark_layout(root):
    missing = []
    for spec in BENCHMARKS:
        dataset_root = benchmark_root(spec, root)
        for directory, _ in spec["subsets"]:
            subset_root = os.path.join(dataset_root, directory)
            if not os.path.isdir(subset_root):
                archive = subset_root + ".tar.gz"
                suffix = " (archive exists but is not extracted)" if os.path.isfile(archive) else ""
                missing.append(subset_root + suffix)

    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Benchmark directories are incomplete. Extract every dataset archive "
            f"before testing:\n{details}"
        )

    total = sum(len(spec["subsets"]) for spec in BENCHMARKS)
    if total != EXPECTED_SUBTESTS:
        raise RuntimeError(f"Configured {total} sub-tests; expected {EXPECTED_SUBTESTS}.")


def percent(value):
    return float(value) * 100.0


def make_row(kind, table, dataset, subset, directory, metrics):
    acc, ap, real_acc, fake_acc, y_true, _ = metrics
    y_true = np.asarray(y_true)
    return {
        "kind": kind,
        "table": table,
        "dataset": dataset,
        "subset": subset,
        "directory": directory,
        "acc_percent": f"{percent(acc):.6f}",
        "ap_percent": f"{percent(ap):.6f}",
        "real_acc_percent": f"{percent(real_acc):.6f}",
        "fake_acc_percent": f"{percent(fake_acc):.6f}",
        "num_samples": int(y_true.size),
        "num_real": int(np.sum(y_true == 0)),
        "num_fake": int(np.sum(y_true == 1)),
    }


def mean_row(kind, table, dataset, subset_rows):
    def mean_of(field):
        return float(np.mean([float(row[field]) for row in subset_rows]))

    return {
        "kind": kind,
        "table": table,
        "dataset": dataset,
        "subset": "Mean",
        "directory": "",
        "acc_percent": f"{mean_of('acc_percent'):.6f}",
        "ap_percent": f"{mean_of('ap_percent'):.6f}",
        "real_acc_percent": f"{mean_of('real_acc_percent'):.6f}",
        "fake_acc_percent": f"{mean_of('fake_acc_percent'):.6f}",
        "num_samples": sum(int(row["num_samples"]) for row in subset_rows),
        "num_real": sum(int(row["num_real"]) for row in subset_rows),
        "num_fake": sum(int(row["num_fake"]) for row in subset_rows),
    }


def print_result(index, label, row):
    print(
        f"({index:02d} {label:20}) "
        f"acc: {float(row['acc_percent']):6.2f}; "
        f"ap: {float(row['ap_percent']):6.2f}; "
        f"real: {float(row['real_acc_percent']):6.2f}; "
        f"fake: {float(row['fake_acc_percent']):6.2f}; "
        f"n: {row['num_samples']}"
    )


if __name__ == "__main__":
    script_args = parse_script_args()
    opt = TestOptions().parse(print_options=False)
    if not opt.model_path:
        raise ValueError(
            "Pass --model_path /absolute/path/to/model_epoch_last.pth "
            "for the training run you want to evaluate."
        )
    opt.seed = int(getattr(opt, "seed", 100))
    opt.serial_batches = True
    opt.class_bal = False
    opt.no_flip = True

    seed_torch(opt.seed)
    check_benchmark_layout(script_args.benchmark_root)

    print("=" * 88)
    print("FINAL CHECKPOINT EVALUATION: one frozen model_epoch_last.pth")
    print("Training reports test-domain metrics each epoch; keep that in mind when reporting results.")
    print(f"Checkpoint     : {opt.model_path}")
    print(f"Benchmark root : {script_args.benchmark_root}")
    print(f"Results CSV    : {script_args.results_csv}")
    print(f"Seed           : {opt.seed}")
    print("=" * 88)

    model = build_model(opt)
    load_checkpoint(model, opt.model_path)
    device = torch.device(f"cuda:{opt.gpu_ids[0]}") if opt.gpu_ids else torch.device("cpu")
    model.to(device)
    model.eval()

    output_dir = os.path.dirname(os.path.abspath(script_args.results_csv))
    os.makedirs(output_dir, exist_ok=True)

    all_subset_rows = []
    with open(script_args.results_csv, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for spec in BENCHMARKS:
            title = f"{spec['table']} - {spec['dataset']}"
            printSet(title)
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
            dataset_root = benchmark_root(spec, script_args.benchmark_root)
            table_rows = []

            for subset_id, (directory, paper_label) in enumerate(spec["subsets"]):
                opt.dataroot = os.path.join(dataset_root, directory)
                opt.classes = []
                opt.no_resize = spec["no_resize"]
                opt.no_crop = spec["no_crop"]

                metrics = validate(model, opt)
                row = make_row(
                    "subset",
                    spec["table"],
                    spec["dataset"],
                    paper_label,
                    directory,
                    metrics,
                )
                table_rows.append(row)
                all_subset_rows.append(row)
                writer.writerow(row)
                csv_file.flush()
                print_result(subset_id, paper_label, row)

            table_mean = mean_row("table_mean", spec["table"], spec["dataset"], table_rows)
            writer.writerow(table_mean)
            csv_file.flush()
            print_result(len(table_rows), "Mean", table_mean)
            print("*" * 88)

        if len(all_subset_rows) != EXPECTED_SUBTESTS:
            raise RuntimeError(
                f"Evaluated {len(all_subset_rows)} sub-tests; expected {EXPECTED_SUBTESTS}."
            )

        overall = mean_row(
            "overall_macro",
            "Table 6 (Extended)",
            "All 43 sub-tests",
            all_subset_rows,
        )
        writer.writerow(overall)
        csv_file.flush()

    printSet("Table 6 (Extended) - Macro average over all 43 sub-tests")
    print_result(EXPECTED_SUBTESTS, "Overall Macro Mean", overall)
    print(f"Completed. CSV saved to: {script_args.results_csv}")
