#!/usr/bin/env python3
"""Clean concatenated per-run HDF5 files without modifying originals.

Some run_i.hdf5 files can contain leftover samples from earlier successful runs.
This script uses each run's log_i_env0.json as the source of truth for success
and valid episode length, then writes a cleaned zip under output/cleandata with the same top-level layout.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import zipfile
from pathlib import Path

import h5py


DEFAULT_MAX_STEPS = 100
ROOT_FILES_TO_COPY = ("episode_results.jsonl", "episode_results.json")


def run_idx_from_name(name: str) -> int:
    stem = Path(name).stem
    return int(stem.split("_")[1])


def numeric_demo_key(name: str) -> int:
    return int(name.split("_")[-1])


def copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def infer_num_samples(demo_group: h5py.Group) -> int:
    if "num_samples" in demo_group.attrs:
        return int(demo_group.attrs["num_samples"])
    if "actions" in demo_group and getattr(demo_group["actions"], "shape", None):
        return int(demo_group["actions"].shape[0])

    sizes = []

    def collect_sizes(group: h5py.Group):
        for value in group.values():
            if isinstance(value, h5py.Dataset) and value.shape:
                sizes.append(int(value.shape[0]))
            elif isinstance(value, h5py.Group):
                collect_sizes(value)

    collect_sizes(demo_group)
    return max(sizes) if sizes else 0


def copy_dataset(src: h5py.Dataset, dst_group: h5py.Group, name: str, original_samples: int, keep_samples: int):
    if src.shape and src.shape[0] == original_samples and keep_samples < original_samples:
        data = src[-keep_samples:]
    else:
        data = src[()]

    dst = dst_group.create_dataset(name, data=data)
    copy_attrs(src, dst)


def copy_group(src_group: h5py.Group, dst_group: h5py.Group, original_samples: int, keep_samples: int):
    copy_attrs(src_group, dst_group)
    for name, value in src_group.items():
        if isinstance(value, h5py.Dataset):
            copy_dataset(value, dst_group, name, original_samples, keep_samples)
        elif isinstance(value, h5py.Group):
            child = dst_group.create_group(name)
            copy_group(value, child, original_samples, keep_samples)


def clean_hdf5_file(src_file, dst_path: Path, valid_steps: int) -> tuple[int, int]:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_file, "r") as src, h5py.File(dst_path, "w") as dst:
        copy_attrs(src, dst)
        src_data = src["data"]
        dst_data = dst.create_group("data")
        copy_attrs(src_data, dst_data)

        total_samples = 0
        original_samples_report = 0
        for demo_name in sorted(src_data.keys(), key=numeric_demo_key):
            src_demo = src_data[demo_name]
            original_samples = infer_num_samples(src_demo)
            keep_samples = min(valid_steps, original_samples)
            original_samples_report = max(original_samples_report, original_samples)

            dst_demo = dst_data.create_group(demo_name)
            copy_group(src_demo, dst_demo, original_samples, keep_samples)
            dst_demo.attrs["num_samples"] = keep_samples
            total_samples += keep_samples

        dst_data.attrs["total"] = total_samples

    return original_samples_report, valid_steps


def valid_steps_from_log(log_data: dict, default_max_steps: int) -> int:
    if log_data.get("success") is True:
        final_step = log_data.get("final_step")
        if final_step is not None:
            return int(final_step)
    return default_max_steps


def copy_directory_root_files(src_root: Path, dst_root: Path):
    dst_root.mkdir(parents=True, exist_ok=True)
    for filename in ROOT_FILES_TO_COPY:
        src_file = src_root / filename
        if src_file.exists():
            shutil.copy2(src_file, dst_root / filename)


def copy_zip_root_files(zf: zipfile.ZipFile, dst_root: Path, top_level: str):
    dst_root.mkdir(parents=True, exist_ok=True)
    names = set(zf.namelist())
    for filename in ROOT_FILES_TO_COPY:
        src_name = f"{top_level}/{filename}"
        if src_name in names:
            (dst_root / filename).write_bytes(zf.read(src_name))


def make_clean_zip(dst_root: Path, overwrite: bool) -> Path | None:
    zip_path = dst_root.with_suffix(".zip")
    if zip_path.exists() and not overwrite:
        print(f"skip existing {zip_path}")
        return zip_path

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in dst_root.rglob("*") if p.is_file()):
            arcname = Path(dst_root.name) / path.relative_to(dst_root)
            zf.write(path, arcname.as_posix())

    print(f"clean zip written to {zip_path}")
    return zip_path


def clean_task_dir(task_dir: Path, dst_task_dir: Path, default_max_steps: int, overwrite: bool):
    for log_path in sorted(task_dir.glob("log_*_env0.json"), key=lambda p: run_idx_from_name(p.name)):
        run_idx = run_idx_from_name(log_path.name)
        hdf5_path = task_dir / f"run_{run_idx}.hdf5"
        if not hdf5_path.exists():
            print(f"missing {hdf5_path}")
            continue

        dst_hdf5 = dst_task_dir / hdf5_path.name
        if dst_hdf5.exists() and not overwrite:
            print(f"skip existing {dst_hdf5}")
            continue

        log_data = json.loads(log_path.read_text())
        valid_steps = valid_steps_from_log(log_data, default_max_steps)
        original_samples, kept_samples = clean_hdf5_file(hdf5_path, dst_hdf5, valid_steps)

        dst_task_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_path, dst_task_dir / log_path.name)
        env_cfg = task_dir / "env_cfg.json"
        if env_cfg.exists():
            shutil.copy2(env_cfg, dst_task_dir / env_cfg.name)

        print(f"{task_dir.name}/run_{run_idx}.hdf5: {original_samples} -> {kept_samples} samples")


def clean_directory(src_root: Path, dst_root: Path, default_max_steps: int, overwrite: bool):
    copy_directory_root_files(src_root, dst_root)
    for task_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        if not list(task_dir.glob("log_*_env0.json")):
            continue
        clean_task_dir(task_dir, dst_root / task_dir.name, default_max_steps, overwrite)


def clean_zip(src_zip: Path, dst_root: Path, default_max_steps: int, overwrite: bool):
    with zipfile.ZipFile(src_zip, "r") as zf:
        names = zf.namelist()
        top_levels = sorted({parts[0] for name in names if (parts := name.split("/")) and parts[0]})
        top_level = src_zip.stem if src_zip.stem in top_levels else top_levels[0]
        copy_zip_root_files(zf, dst_root, top_level)

        task_names = sorted({parts[1] for name in names if len(parts := name.split("/")) >= 3 and parts[0] == top_level})
        for task_name in task_names:
            log_names = sorted(
                [name for name in names if name.startswith(f"{top_level}/{task_name}/log_") and name.endswith("_env0.json")],
                key=lambda name: run_idx_from_name(Path(name).name),
            )
            if not log_names:
                continue

            dst_task_dir = dst_root / task_name
            for log_name in log_names:
                run_idx = run_idx_from_name(Path(log_name).name)
                hdf5_name = str(Path(log_name).parent / f"run_{run_idx}.hdf5")
                if hdf5_name not in names:
                    print(f"missing {hdf5_name}")
                    continue

                dst_hdf5 = dst_task_dir / f"run_{run_idx}.hdf5"
                if dst_hdf5.exists() and not overwrite:
                    print(f"skip existing {dst_hdf5}")
                    continue

                log_data = json.loads(zf.read(log_name))
                valid_steps = valid_steps_from_log(log_data, default_max_steps)
                src_bytes = io.BytesIO(zf.read(hdf5_name))
                original_samples, kept_samples = clean_hdf5_file(src_bytes, dst_hdf5, valid_steps)

                dst_task_dir.mkdir(parents=True, exist_ok=True)
                (dst_task_dir / Path(log_name).name).write_bytes(zf.read(log_name))
                env_cfg_name = str(Path(log_name).parent / "env_cfg.json")
                if env_cfg_name in names:
                    (dst_task_dir / "env_cfg.json").write_bytes(zf.read(env_cfg_name))

                print(f"{task_name}/run_{run_idx}.hdf5: {original_samples} -> {kept_samples} samples")


def main():
    parser = argparse.ArgumentParser(description="Clean concatenated RoboLab run_i.hdf5 files into output/cleandata.")
    parser.add_argument("input", type=Path, help="Experiment directory or .zip file")
    parser.add_argument("--output-root", type=Path, default=Path("output/cleandata"))
    parser.add_argument("--default-max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-folder", action="store_true", help="Keep the temporary cleaned folder after writing the zip")
    args = parser.parse_args()

    src = args.input
    clean_name = src.stem if src.suffix == ".zip" else src.name
    dst_root = args.output_root / clean_name

    if src.suffix == ".zip":
        clean_zip(src, dst_root, args.default_max_steps, args.overwrite)
    else:
        clean_directory(src, dst_root, args.default_max_steps, args.overwrite)

    zip_path = make_clean_zip(dst_root, args.overwrite)
    if zip_path is not None and not args.keep_folder:
        shutil.rmtree(dst_root)
        print(f"removed temporary folder {dst_root}")


if __name__ == "__main__":
    main()
