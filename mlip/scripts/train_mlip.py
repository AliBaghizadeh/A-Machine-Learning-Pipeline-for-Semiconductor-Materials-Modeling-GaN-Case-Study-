"""
Train MACE MLIP for GaN (Wurtzite)
==================================
Runs real MACE training via mace_run_train with strict CUDA mode.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.model_config import MACE_CONFIG, TRAINING_CONFIG, GPU_CONFIG


PROJECT_ROOT = Path(__file__).parent.parent.parent
MLIP_DATA_DIR = PROJECT_ROOT / "mlip" / "data"
MLIP_MODELS_DIR = PROJECT_ROOT / "mlip" / "models"
MLIP_RESULTS_DIR = PROJECT_ROOT / "mlip" / "results"


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def ensure_cuda(strict_cuda: bool = True):
    print_header("GPU CHECK")
    has_cuda = torch.cuda.is_available()
    print(f"  CUDA available: {has_cuda}")
    if not has_cuda:
        msg = "CUDA is not available. This trainer requires NVIDIA GPU."
        if strict_cuda:
            raise RuntimeError(msg)
        print(f"  WARNING: {msg}")
        return torch.device("cpu")

    device_str = GPU_CONFIG.get("device", "cuda:0")
    device = torch.device(device_str)
    torch.cuda.set_device(device)
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Device: {device}")
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    return device


def find_training_files():
    # Prefer versioned datasets written by extract_dft_data.py.
    latest_ptr = MLIP_DATA_DIR / "LATEST_DATASET.txt"
    if latest_ptr.exists():
        try:
            latest_dir = Path(latest_ptr.read_text(encoding="utf-8").strip())
            if latest_dir.exists():
                train_file = latest_dir / "train.xyz"
                val_file = latest_dir / "val.xyz"
                test_file = latest_dir / "test.xyz"
                all_file = latest_dir / "all_data.xyz"
                if train_file.exists() or all_file.exists():
                    # Use this dataset directory if it contains expected files.
                    if train_file.exists():
                        use_train = train_file
                        use_val = val_file if val_file.exists() else None
                        use_test = test_file if test_file.exists() else None
                    else:
                        use_train = all_file
                        use_val = None
                        use_test = None
                    return use_train, use_val, use_test
        except Exception:
            pass

    # Legacy fallback: unversioned files directly under mlip/data/
    train_file = MLIP_DATA_DIR / "train.xyz"
    val_file = MLIP_DATA_DIR / "val.xyz"
    test_file = MLIP_DATA_DIR / "test.xyz"
    all_file = MLIP_DATA_DIR / "all_data.xyz"

    if train_file.exists():
        use_train = train_file
        use_val = val_file if val_file.exists() else None
        use_test = test_file if test_file.exists() else None
    elif all_file.exists():
        # Fallback for small datasets without split files
        use_train = all_file
        use_val = None
        use_test = None
    else:
        raise FileNotFoundError(
            "No training data found. Expected mlip/data/train.xyz or mlip/data/all_data.xyz"
        )

    return use_train, use_val, use_test


def count_structures(path: Path):
    if path is None or not path.exists():
        return 0
    count = 0
    with open(path, "r") as f:
        for line in f:
            if line.strip().isdigit():
                count += 1
    return count


def _is_mnt_path(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return str(path.resolve()).startswith("/mnt/")
    except Exception:
        return str(path).startswith("/mnt/")


def maybe_stage_training_files(train_file: Path, val_file: Path | None, test_file: Path | None,
                               enable_stage: bool, stage_root: Path, run_tag: str):
    """Stage dataset files from /mnt/* to Linux scratch for faster dataloader I/O."""
    if not enable_stage:
        return train_file, val_file, test_file, None

    files = [train_file, val_file, test_file]
    needs_stage = any(_is_mnt_path(p) for p in files if p is not None)
    if not needs_stage:
        return train_file, val_file, test_file, None

    stage_dir = stage_root / f"dataset_{run_tag}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    def _copy_one(path: Path | None):
        if path is None:
            return None
        target = stage_dir / path.name
        shutil.copy2(path, target)
        return target

    staged_train = _copy_one(train_file)
    staged_val = _copy_one(val_file)
    staged_test = _copy_one(test_file)
    print_header("I/O STAGING")
    print(f"  Source appears on /mnt/* -> staging to fast path: {stage_dir}")
    return staged_train, staged_val, staged_test, stage_dir


def run_torch_profiler_probe(device: torch.device, out_dir: Path, num_workers: int, pin_memory: bool,
                             persistent_workers: bool, prefetch_factor: int, steps: int = 20):
    """
    Run a short PyTorch profiler probe to identify input-pipeline vs GPU bottlenecks.
    This profiles a synthetic dataloader + H2D copies with your worker/pin settings.
    """
    print_header("PYTORCH PROFILER PROBE")
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "torch_profile_trace.json"
    summary_path = out_dir / "torch_profile_summary.txt"

    samples = 4096
    batch_size = 64
    x = torch.randn(samples, 1024)
    y = torch.randn(samples, 64)
    dataset = TensorDataset(x, y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,
    )

    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 64),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=max(2, steps - 2), repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )

    iterator = iter(loader)
    with prof:
        for _ in range(steps):
            try:
                xb, yb = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                xb, yb = next(iterator)
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            prof.step()

    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
    with open(summary_path, "w") as f:
        f.write(table)
    print(f"  Saved profiler trace: {trace_path}")
    print(f"  Saved profiler summary: {summary_path}")
    return {"trace": str(trace_path), "summary": str(summary_path)}


def build_mace_command(
    train_file: Path,
    val_file: Path | None,
    run_dir: Path,
    training_overrides: dict | None = None,
    restart_latest: bool = False,
):
    train_cfg = dict(TRAINING_CONFIG)
    if training_overrides:
        train_cfg.update(training_overrides)
    mace_cfg = MACE_CONFIG
    num_workers = int(train_cfg.get("num_workers", 12))
    if num_workers not in (8, 12):
        # Enforce project rule for performance/repro.
        print(f"WARNING: TRAINING_CONFIG['num_workers']={num_workers} not in {{8,12}}; using 12.")
        num_workers = 12
    pin_memory = bool(train_cfg.get("pin_memory", True))
    persistent_workers = bool(train_cfg.get("persistent_workers", True))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 4))
    if persistent_workers or prefetch_factor:
        print("NOTE: mace_run_train CLI does not expose persistent_workers/prefetch_factor directly.")
        print(f"      Requested policy: persistent_workers={persistent_workers}, prefetch_factor={prefetch_factor}")

    cmd = [
        "mace_run_train",
        "--name", "gan_mace",
        "--work_dir", str(run_dir),
        "--model_dir", str(MLIP_MODELS_DIR),
        "--results_dir", str(MLIP_RESULTS_DIR),
        "--checkpoints_dir", str(run_dir / "checkpoints"),
        "--log_dir", str(run_dir / "logs"),
        "--device", "cuda",
        "--default_dtype", "float32",
        "--model", "MACE",
        "--train_file", str(train_file),
        # For small custom datasets, let MACE estimate per-element reference
        # atomic energies from data instead of requiring manual E0 inputs.
        "--E0s", "average",
        "--energy_key", "energy",
        "--forces_key", "forces",
        "--stress_key", "stress",
        "--num_workers", str(num_workers),
        "--pin_memory", str(pin_memory),
        "--batch_size", str(int(train_cfg.get("batch_size", 4))),
        "--max_num_epochs", str(int(train_cfg.get("max_num_epochs", 500))),
        "--patience", str(int(train_cfg.get("patience", 100))),
        "--eval_interval", str(int(train_cfg.get("eval_interval", 10))),
        "--lr", str(float(train_cfg.get("learning_rate", 1e-4))),
        "--weight_decay", str(float(train_cfg.get("weight_decay", 1e-8))),
        "--energy_weight", str(float(train_cfg.get("energy_weight", 1.0))),
        "--forces_weight", str(float(train_cfg.get("forces_weight", 10.0))),
        "--stress_weight", str(float(train_cfg.get("stress_weight", 0.1))),
        "--r_max", str(float(mace_cfg.get("r_max", 5.0))),
        "--num_radial_basis", str(int(mace_cfg.get("num_radial_basis", 8))),
        "--num_cutoff_basis", str(int(mace_cfg.get("num_cutoff_basis", 5))),
        "--max_ell", str(int(mace_cfg.get("max_ell", 3))),
        "--correlation", str(int(mace_cfg.get("correlation", 3))),
        "--num_interactions", str(int(mace_cfg.get("num_interactions", 2))),
        "--MLP_irreps", str(mace_cfg.get("mlp_irreps", "16x0e")),
        "--seed", "42",
    ]

    # Prefer num_channels/max_L for compatibility across MACE CLI versions.
    if mace_cfg.get("num_channels") is not None and mace_cfg.get("max_L") is not None:
        cmd += [
            "--num_channels", str(int(mace_cfg["num_channels"])),
            "--max_L", str(int(mace_cfg["max_L"])),
        ]
    else:
        cmd += ["--hidden_irreps", str(mace_cfg.get("hidden_irreps", "128x0e + 128x1o + 64x2e + 32x3o"))]

    if val_file is not None:
        cmd += ["--valid_file", str(val_file)]
    else:
        cmd += ["--valid_fraction", "0.1"]

    atomic_numbers = mace_cfg.get("atomic_numbers")
    if atomic_numbers:
        # MACE CLI compatibility:
        # This environment's mace_run_train expects --atomic_numbers to parse into
        # a Python list. Comma-separated plain strings (e.g. "31,7") can be
        # interpreted as str and trigger assertion failures.
        atomic_numbers_arg = "[" + ",".join(str(int(z)) for z in atomic_numbers) + "]"
        cmd += ["--atomic_numbers", atomic_numbers_arg]

    if bool(train_cfg.get("amsgrad", True)):
        cmd += ["--amsgrad"]

    if bool(train_cfg.get("ema", True)):
        cmd += ["--ema", "--ema_decay", str(float(train_cfg.get("ema_decay", 0.99)))]

    if restart_latest:
        cmd += ["--restart_latest"]

    return cmd


def _gpu_snapshot(device_index: int):
    alloc = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 2)
    return {"allocated_mb": float(alloc), "reserved_mb": float(reserved)}


def run_mace_training(cmd, run_dir: Path, device: torch.device, run_tag: str):
    print_header("TRAINING MACE MODEL")
    print("Command:")
    print("  " + shlex.join(cmd))

    run_dir.mkdir(parents=True, exist_ok=True)
    MLIP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MLIP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stdout_log = run_dir / "mace_stdout.log"
    epoch_log = run_dir / "epoch_device_log.jsonl"
    # Preserve prior logs when resuming into an existing directory.
    if stdout_log.exists():
        stdout_log = run_dir / f"mace_stdout__{run_tag}.log"
    if epoch_log.exists():
        epoch_log = run_dir / f"epoch_device_log__{run_tag}.jsonl"

    epoch_re = re.compile(r"\b[Ee]poch\s*[:=]?\s*(\d+)\b")
    last_epoch = None
    device_index = device.index if device.index is not None else 0
    saw_training_complete = False
    saw_pickle_export_error = False
    saw_model_save_line = False

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    with open(stdout_log, "w") as out, open(epoch_log, "w") as elog:
        for line in process.stdout:
            print(line, end="")
            out.write(line)
            if "Training complete" in line:
                saw_training_complete = True
            if "Saving model to" in line:
                saw_model_save_line = True
            if "ScriptFunction cannot be pickled" in line:
                saw_pickle_export_error = True

            m = epoch_re.search(line)
            if not m:
                continue

            epoch = int(m.group(1))
            if epoch == last_epoch:
                continue
            last_epoch = epoch

            snap = _gpu_snapshot(device_index)
            event = {
                "time": datetime.now().isoformat(),
                "epoch": epoch,
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device_index),
                **snap,
            }
            elog.write(json.dumps(event) + "\n")
            elog.flush()
            print(
                f"[epoch-device] epoch={epoch} device={device} "
                f"alloc={snap['allocated_mb']:.1f}MB reserved={snap['reserved_mb']:.1f}MB"
            )

    rc = process.wait()
    if rc != 0:
        # Known MACE post-training export issue in some torch/mace combinations:
        # training finishes and checkpoints are written, but final .model export fails.
        if saw_training_complete and saw_model_save_line and saw_pickle_export_error:
            ckpts = sorted((run_dir / "checkpoints").glob("*.pt"))
            if ckpts:
                latest_ckpt = str(ckpts[-1])
                warn = (
                    "MACE finished training and wrote checkpoints, but final .model export failed "
                    "with ScriptFunction pickling error. Proceeding with latest checkpoint."
                )
                print(f"WARNING: {warn}")
                marker = run_dir / "TRAINING_COMPLETED_WITH_EXPORT_WARNING.txt"
                marker.write_text(warn + "\n" + f"checkpoint={latest_ckpt}\n", encoding="utf-8")
                return {
                    "stdout_log": str(stdout_log),
                    "epoch_device_log": str(epoch_log),
                    "warning": warn,
                    "checkpoint": latest_ckpt,
                    "warning_marker": str(marker),
                }
        raise RuntimeError(
            f"MACE training failed with exit code {rc}. "
            f"See {stdout_log} for details."
        )

    return {
        "stdout_log": str(stdout_log),
        "epoch_device_log": str(epoch_log),
    }


def write_summary(run_dir: Path, train_file: Path, val_file: Path | None, cmd, logs):
    summary = {
        "started": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "train_file": str(train_file),
        "val_file": str(val_file) if val_file else None,
        "n_train_structures": count_structures(train_file),
        "n_val_structures": count_structures(val_file) if val_file else None,
        "command": cmd,
        "logs": logs,
    }
    out = MLIP_RESULTS_DIR / "training_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining summary saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Train real MACE model on GPU")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback (default is strict CUDA-only mode).",
    )
    parser.add_argument(
        "--restart-latest",
        action="store_true",
        help="Resume optimizer from latest checkpoint in the run directory (passes --restart_latest to mace_run_train).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max epochs (default comes from config).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early-stopping patience (default comes from config).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Override eval interval (default comes from config).",
    )
    parser.add_argument(
        "--energy-weight",
        type=float,
        default=None,
        help="Override MACE energy loss weight (default comes from config).",
    )
    parser.add_argument(
        "--forces-weight",
        type=float,
        default=None,
        help="Override MACE forces loss weight (default comes from config).",
    )
    parser.add_argument(
        "--stress-weight",
        type=float,
        default=None,
        help="Override MACE stress loss weight (default comes from config).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Faster iteration preset: max_epochs=150, patience=30, eval_interval=20.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Reuse an existing run directory (for resume). Preserves existing logs by suffixing new log filenames.",
    )
    parser.add_argument(
        "--profile-before-train",
        action="store_true",
        help="Run a short PyTorch profiler probe before launching MACE training.",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Run profiler probe only and exit (no MACE training).",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=20,
        help="Profiler probe steps (default: 20).",
    )
    parser.add_argument(
        "--no-stage-io",
        action="store_true",
        help="Disable automatic staging of training data from /mnt/* to Linux scratch.",
    )
    parser.add_argument(
        "--stage-dir",
        type=str,
        default=None,
        help="Override scratch directory for staged training data.",
    )
    args = parser.parse_args()

    print_header("MACE MLIP TRAINING FOR GaN")
    device = ensure_cuda(strict_cuda=not args.allow_cpu)

    if device.type != "cuda":
        raise RuntimeError("This training path is intended for NVIDIA CUDA GPU.")

    run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_file, val_file, test_file = find_training_files()
    n_train = count_structures(train_file)
    n_val = count_structures(val_file) if val_file else 0
    n_total_supervised = n_train + n_val
    if val_file is None and n_train < 2:
        raise RuntimeError(
            f"Too few training structures for MACE split (train={n_train}, val=0). "
            "Add more Tier-B completed entries, then rerun extract_data. "
            "Recommended: run Stage B2b (extra single-point structures)."
        )
    if n_total_supervised < 4:
        print(
            f"WARNING: Very small supervised dataset (train={n_train}, val={n_val}). "
            "Training may be unstable; add more Tier-B data for meaningful results."
        )
    train_cfg = TRAINING_CONFIG
    num_workers = int(train_cfg.get("num_workers", 12))
    if num_workers not in (8, 12):
        num_workers = 12
    pin_memory = bool(train_cfg.get("pin_memory", True))
    stage_root = Path(args.stage_dir) if args.stage_dir else Path(train_cfg.get("io_stage_dir", "/tmp/mlip_fast_io"))
    io_stage_enabled = bool(train_cfg.get("io_stage_from_mnt", True)) and (not args.no_stage_io)

    train_file, val_file, test_file, staged_dir = maybe_stage_training_files(
        train_file, val_file, test_file, io_stage_enabled, stage_root, run_tag
    )
    run_dir = Path(args.run_dir) if args.run_dir else (MLIP_RESULTS_DIR / f"mace_run_{run_tag}")

    profiler_logs = {}
    if args.profile_before_train or args.profile_only:
        profiler_logs = run_torch_profiler_probe(
            device=device,
            out_dir=run_dir / "profiler",
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=bool(train_cfg.get("persistent_workers", True)),
            prefetch_factor=int(train_cfg.get("prefetch_factor", 4)),
            steps=max(5, int(args.profile_steps)),
        )
    if args.profile_only:
        print_header("PROFILE COMPLETE")
        print("Profiler-only mode finished.")
        return

    # Optional overrides to accelerate iteration without editing config.
    overrides = {}
    if args.fast:
        overrides.update({"max_num_epochs": 150, "patience": 30, "eval_interval": 20})
    if args.max_epochs is not None:
        overrides["max_num_epochs"] = int(args.max_epochs)
    if args.patience is not None:
        overrides["patience"] = int(args.patience)
    if args.eval_interval is not None:
        overrides["eval_interval"] = int(args.eval_interval)
    if args.energy_weight is not None:
        overrides["energy_weight"] = float(args.energy_weight)
    if args.forces_weight is not None:
        overrides["forces_weight"] = float(args.forces_weight)
    if args.stress_weight is not None:
        overrides["stress_weight"] = float(args.stress_weight)

    cmd = build_mace_command(
        train_file,
        val_file,
        run_dir,
        training_overrides=overrides if overrides else None,
        restart_latest=bool(args.restart_latest),
    )
    logs = run_mace_training(cmd, run_dir, device, run_tag=run_tag)
    if profiler_logs:
        logs["profiler"] = profiler_logs
    if staged_dir is not None:
        logs["staged_data_dir"] = str(staged_dir)
    write_summary(run_dir, train_file, val_file, cmd, logs)

    print_header("TRAINING COMPLETE")
    print("MACE training finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc!r}")
        raise SystemExit(1)
