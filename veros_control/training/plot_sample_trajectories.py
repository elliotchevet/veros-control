import argparse
import csv
import glob
import os
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot diagnostics from sampled JVP/VJP u trajectories saved by ad_benchmark.py."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="HDF5 file(s), glob(s), or folder(s) containing *_steps_*.h5 files.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Folder for plots and CSV diagnostics. Default: <input-parent>/trajectory_plots.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative tolerance for allclose checks. Default: 0 for strict numerical equality.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for allclose checks. Default: 0 for strict numerical equality.",
    )
    parser.add_argument(
        "--no-per-file-plots",
        action="store_true",
        help="Only write summary plots and CSV files.",
    )
    parser.add_argument(
        "--skip-steps",
        nargs="*",
        type=int,
        default=(3, 9),
        help=(
            "n_steps values to ignore in plots and CSV diagnostics. "
            "Default: 3 9. Pass --skip-steps with no values to include all steps."
        ),
    )
    return parser.parse_args()


def resolve_inputs(inputs):
    paths = []
    for item in inputs:
        item_path = Path(item)
        if item_path.is_dir():
            paths.extend(sorted(item_path.glob("*.h5")))
            continue

        matches = [Path(path) for path in sorted(glob.glob(item))]
        if matches:
            paths.extend(matches)
        elif item_path.exists():
            paths.append(item_path)

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(resolved)

    if not unique_paths:
        raise FileNotFoundError(f"No HDF5 files matched: {inputs}")

    return sorted(unique_paths, key=lambda path: (parse_steps_from_name(path), str(path)))


def parse_steps_from_name(path):
    match = re.search(r"_steps_(\d+)\.h5$", path.name)
    if match:
        return int(match.group(1))
    return -1


def attr_float(attrs, key):
    if key not in attrs:
        return np.nan
    return float(attrs[key])


def mean_u2(u):
    return float(np.mean(np.asarray(u, dtype=np.float64) ** 2))


def finite_rms_and_max(diff):
    diff = np.asarray(diff, dtype=np.float64)
    finite = np.isfinite(diff)
    if not np.any(finite):
        return np.nan, np.nan
    finite_diff = diff[finite]
    return (
        float(np.sqrt(np.mean(finite_diff**2))),
        float(np.max(np.abs(finite_diff))),
    )


def two_dimensional_field(field):
    field = np.asarray(field)
    if field.ndim == 0:
        return field.reshape(1, 1)
    if field.ndim == 1:
        return field.reshape(1, -1)
    if field.ndim == 2:
        return field
    return np.nanmean(field, axis=tuple(range(2, field.ndim)))


def compute_file_metrics(path, rtol=0.0, atol=0.0):
    with h5py.File(path, "r") as h5file:
        steps = int(h5file.attrs.get("iterations", parse_steps_from_name(path)))
        sample_steps = int(h5file.attrs.get("sample_steps", 0))
        sample_indices = np.asarray(h5file["sample_step_indices"][:], dtype=np.int64)

        jvp_u = h5file["jvp/u"]
        vjp_u = h5file["vjp/u"]
        jvp_stored_loss = np.asarray(h5file["jvp/manual_mean_u2"][:], dtype=np.float64)
        vjp_stored_loss = np.asarray(h5file["vjp/manual_mean_u2"][:], dtype=np.float64)
        jvp_write_count = np.asarray(h5file["jvp/write_count"][:], dtype=np.int64)
        vjp_write_count = np.asarray(h5file["vjp/write_count"][:], dtype=np.int64)

        sample_rows = []
        total_squared_diff = 0.0
        total_value_count = 0
        total_max_abs_diff = 0.0

        for sample_index, step_index in enumerate(sample_indices):
            u_jvp = np.asarray(jvp_u[sample_index])
            u_vjp = np.asarray(vjp_u[sample_index])
            diff = np.asarray(u_vjp, dtype=np.float64) - np.asarray(u_jvp, dtype=np.float64)
            sample_rms, sample_max = finite_rms_and_max(diff)
            sample_written = (
                int(jvp_write_count[sample_index]) > 0
                and int(vjp_write_count[sample_index]) > 0
            )
            finite = np.isfinite(diff)
            if np.any(finite):
                total_squared_diff += float(np.sum(diff[finite] ** 2))
                total_value_count += int(np.sum(finite))
                total_max_abs_diff = max(total_max_abs_diff, sample_max)

            jvp_recomputed_loss = mean_u2(u_jvp)
            vjp_recomputed_loss = mean_u2(u_vjp)
            strict_equal = sample_written and bool(np.array_equal(u_jvp, u_vjp))
            allclose_equal = sample_written and bool(
                np.allclose(u_jvp, u_vjp, rtol=rtol, atol=atol, equal_nan=True)
            )

            sample_rows.append(
                {
                    "path": str(path),
                    "file": path.name,
                    "steps": steps,
                    "sample_index": sample_index,
                    "step_index": int(step_index),
                    "jvp_write_count": int(jvp_write_count[sample_index]),
                    "vjp_write_count": int(vjp_write_count[sample_index]),
                    "jvp_stored_mean_u2": float(jvp_stored_loss[sample_index]),
                    "vjp_stored_mean_u2": float(vjp_stored_loss[sample_index]),
                    "jvp_recomputed_mean_u2": jvp_recomputed_loss,
                    "vjp_recomputed_mean_u2": vjp_recomputed_loss,
                    "jvp_recomputed_minus_stored": jvp_recomputed_loss - float(jvp_stored_loss[sample_index]),
                    "vjp_recomputed_minus_stored": vjp_recomputed_loss - float(vjp_stored_loss[sample_index]),
                    "manual_loss_diff": vjp_recomputed_loss - jvp_recomputed_loss,
                    "u_rms_diff": sample_rms,
                    "u_max_abs_diff": sample_max,
                    "strict_equal": strict_equal,
                    "allclose": allclose_equal,
                }
            )

        jvp_recomputed_loss_sum = float(np.sum([row["jvp_recomputed_mean_u2"] for row in sample_rows]))
        vjp_recomputed_loss_sum = float(np.sum([row["vjp_recomputed_mean_u2"] for row in sample_rows]))
        strict_equal_fraction = float(np.mean([row["strict_equal"] for row in sample_rows])) if sample_rows else 1.0
        allclose_fraction = float(np.mean([row["allclose"] for row in sample_rows])) if sample_rows else 1.0
        total_rms_diff = (
            float(np.sqrt(total_squared_diff / total_value_count))
            if total_value_count
            else np.nan
        )

        metrics = {
            "path": str(path),
            "file": path.name,
            "steps": steps,
            "sample_steps": sample_steps,
            "n_samples": len(sample_indices),
            "u_shape": "x".join(str(size) for size in jvp_u.shape[1:]),
            "jvp_ad_loss": attr_float(h5file.attrs, "jvp_ad_loss"),
            "vjp_ad_loss": attr_float(h5file.attrs, "vjp_ad_loss"),
            "jvp_stored_manual_loss_sum": attr_float(h5file.attrs, "jvp_manual_loss_sum"),
            "vjp_stored_manual_loss_sum": attr_float(h5file.attrs, "vjp_manual_loss_sum"),
            "jvp_recomputed_manual_loss_sum": jvp_recomputed_loss_sum,
            "vjp_recomputed_manual_loss_sum": vjp_recomputed_loss_sum,
            "jvp_recomputed_minus_ad_loss": jvp_recomputed_loss_sum - attr_float(h5file.attrs, "jvp_ad_loss"),
            "vjp_recomputed_minus_ad_loss": vjp_recomputed_loss_sum - attr_float(h5file.attrs, "vjp_ad_loss"),
            "jvp_stored_minus_ad_loss": attr_float(h5file.attrs, "jvp_manual_minus_ad_loss"),
            "vjp_stored_minus_ad_loss": attr_float(h5file.attrs, "vjp_manual_minus_ad_loss"),
            "loss_abs_diff_ad": abs(attr_float(h5file.attrs, "vjp_ad_loss") - attr_float(h5file.attrs, "jvp_ad_loss")),
            "loss_abs_diff_manual": abs(vjp_recomputed_loss_sum - jvp_recomputed_loss_sum),
            "u_rms_diff": total_rms_diff,
            "u_max_abs_diff": total_max_abs_diff,
            "strict_equal_fraction": strict_equal_fraction,
            "allclose_fraction": allclose_fraction,
            "jvp_min_write_count": int(np.min(jvp_write_count)) if len(jvp_write_count) else 0,
            "jvp_max_write_count": int(np.max(jvp_write_count)) if len(jvp_write_count) else 0,
            "vjp_min_write_count": int(np.min(vjp_write_count)) if len(vjp_write_count) else 0,
            "vjp_max_write_count": int(np.max(vjp_write_count)) if len(vjp_write_count) else 0,
        }

    return metrics, sample_rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def set_step_axis(ax, steps):
    ax.set_xscale("log", base=3)
    ax.set_xticks(steps)
    labels = []
    for step in steps:
        power = round(np.log(step) / np.log(3)) if step > 0 else None
        labels.append(rf"$3^{power}$" if power is not None and 3**power == step else str(step))
    ax.set_xticklabels(labels)


def symlog_nonnegative(ax, values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    linthresh = float(np.min(finite) / 10.0) if finite.size else 1e-30
    ax.set_yscale("symlog", linthresh=linthresh)


def plot_summary(metrics_rows, output_folder):
    steps = np.asarray([row["steps"] for row in metrics_rows], dtype=int)
    order = np.argsort(steps)
    metrics_rows = [metrics_rows[index] for index in order]
    steps = steps[order]

    def values(key):
        return np.asarray([row[key] for row in metrics_rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(steps, values("jvp_ad_loss"), "o-", label="JVP AD loss")
    axes[0].plot(steps, values("jvp_recomputed_manual_loss_sum"), "o--", label="JVP manual from stored u")
    axes[0].plot(steps, values("vjp_ad_loss"), "s-", label="VJP AD loss")
    axes[0].plot(steps, values("vjp_recomputed_manual_loss_sum"), "s--", label="VJP manual from stored u")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    jvp_abs = np.abs(values("jvp_recomputed_minus_ad_loss"))
    vjp_abs = np.abs(values("vjp_recomputed_minus_ad_loss"))
    axes[1].plot(steps, jvp_abs, "o-", label="|JVP manual - AD|")
    axes[1].plot(steps, vjp_abs, "s-", label="|VJP manual - AD|")
    axes[1].set_ylabel("Absolute loss error")
    axes[1].set_xlabel("n_steps")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    symlog_nonnegative(axes[1], np.concatenate([jvp_abs, vjp_abs]))
    set_step_axis(axes[1], steps)

    fig.suptitle("AD Loss Versus Manual Loss From Stored u")
    fig.tight_layout()
    fig.savefig(Path(output_folder) / "loss_manual_vs_ad.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(steps, values("u_rms_diff"), "o-", label="RMS(u_VJP - u_JVP)")
    axes[0].plot(steps, values("u_max_abs_diff"), "s-", label="max |u_VJP - u_JVP|")
    axes[0].set_ylabel("u difference")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    symlog_nonnegative(axes[0], np.concatenate([values("u_rms_diff"), values("u_max_abs_diff")]))

    axes[1].plot(steps, values("strict_equal_fraction"), "o-", label="strict equal fraction")
    axes[1].plot(steps, values("allclose_fraction"), "s-", label="allclose fraction")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("Fraction of samples")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].plot(steps, values("jvp_max_write_count"), "o-", label="JVP max write count")
    axes[2].plot(steps, values("vjp_max_write_count"), "s-", label="VJP max write count")
    axes[2].plot(steps, values("jvp_min_write_count"), "o--", label="JVP min write count")
    axes[2].plot(steps, values("vjp_min_write_count"), "s--", label="VJP min write count")
    axes[2].set_ylabel("Callback writes/sample")
    axes[2].set_xlabel("n_steps")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)
    set_step_axis(axes[2], steps)

    fig.suptitle("JVP/VJP Trajectory Equality Diagnostics")
    fig.tight_layout()
    fig.savefig(Path(output_folder) / "trajectory_distance_summary.png", dpi=160)
    plt.close(fig)


def plot_per_file(path, sample_rows, output_folder):
    if not sample_rows:
        return

    stem = Path(path).stem
    step_indices = np.asarray([row["step_index"] for row in sample_rows], dtype=int)

    def values(key):
        return np.asarray([row[key] for row in sample_rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(step_indices, values("jvp_recomputed_mean_u2"), "o-", label="JVP manual mean(u^2)")
    axes[0].plot(step_indices, values("vjp_recomputed_mean_u2"), "s--", label="VJP manual mean(u^2)")
    axes[0].set_ylabel("Sample loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(step_indices, values("manual_loss_diff"), "o-", label="VJP - JVP")
    axes[1].set_ylabel("Sample loss difference")
    axes[1].set_xlabel("Veros step")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    symlog_nonnegative(axes[1], np.abs(values("manual_loss_diff")))

    fig.suptitle(f"Manual Sample Losses: {stem}")
    fig.tight_layout()
    fig.savefig(Path(output_folder) / f"{stem}_sample_losses.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(step_indices, values("u_rms_diff"), "o-", label="RMS difference")
    axes[0].plot(step_indices, values("u_max_abs_diff"), "s-", label="max abs difference")
    axes[0].set_ylabel("u_VJP - u_JVP")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    symlog_nonnegative(axes[0], np.concatenate([values("u_rms_diff"), values("u_max_abs_diff")]))

    axes[1].plot(step_indices, values("strict_equal"), "o-", label="strict equal")
    axes[1].plot(step_indices, values("allclose"), "s--", label="allclose")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("Boolean")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].plot(step_indices, values("jvp_write_count"), "o-", label="JVP write count")
    axes[2].plot(step_indices, values("vjp_write_count"), "s-", label="VJP write count")
    axes[2].set_ylabel("Writes/sample")
    axes[2].set_xlabel("Veros step")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.suptitle(f"Sampled u Trajectory Distance: {stem}")
    fig.tight_layout()
    fig.savefig(Path(output_folder) / f"{stem}_u_distance.png", dpi=160)
    plt.close(fig)


def plot_final_maps(path, output_folder):
    with h5py.File(path, "r") as h5file:
        if h5file["jvp/u"].shape[0] == 0:
            return
        u_jvp = np.asarray(h5file["jvp/u"][-1])
        u_vjp = np.asarray(h5file["vjp/u"][-1])

    jvp_map = two_dimensional_field(u_jvp)
    vjp_map = two_dimensional_field(u_vjp)
    diff_map = vjp_map - jvp_map

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    panels = [
        (jvp_map, "JVP final sampled u"),
        (vjp_map, "VJP final sampled u"),
        (diff_map, "VJP - JVP"),
    ]
    for ax, (field, title) in zip(axes, panels):
        image = ax.imshow(np.asarray(field).T, origin="lower", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        fig.colorbar(image, ax=ax, shrink=0.85)

    fig.suptitle(f"Final Sample 2D Mean Field: {Path(path).stem}")
    fig.savefig(Path(output_folder) / f"{Path(path).stem}_final_u_maps.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    paths = resolve_inputs(args.inputs)
    skip_steps = set(args.skip_steps)
    if skip_steps:
        kept_paths = [path for path in paths if parse_steps_from_name(path) not in skip_steps]
        skipped_paths = [path for path in paths if parse_steps_from_name(path) in skip_steps]
        for path in skipped_paths:
            print(f"Skipping steps={parse_steps_from_name(path)} file: {path}")
        paths = kept_paths
        if not paths:
            raise ValueError(f"All input files were skipped by --skip-steps={sorted(skip_steps)}")

    if args.output_folder is None:
        args.output_folder = str(paths[0].parent / "trajectory_plots")
    os.makedirs(args.output_folder, exist_ok=True)

    metrics_rows = []
    sample_rows_all = []
    for path in paths:
        print(f"Reading {path}")
        metrics, sample_rows = compute_file_metrics(path, rtol=args.rtol, atol=args.atol)
        metrics_rows.append(metrics)
        sample_rows_all.extend(sample_rows)
        if not args.no_per_file_plots:
            plot_per_file(path, sample_rows, args.output_folder)
            plot_final_maps(path, args.output_folder)

    write_csv(Path(args.output_folder) / "trajectory_summary.csv", metrics_rows)
    write_csv(Path(args.output_folder) / "trajectory_samples.csv", sample_rows_all)
    plot_summary(metrics_rows, args.output_folder)

    print(f"\nWrote diagnostics to {args.output_folder}")
    print("\nSummary:")
    for row in sorted(metrics_rows, key=lambda item: item["steps"]):
        strict = row["strict_equal_fraction"] == 1.0
        close = row["allclose_fraction"] == 1.0
        print(
            f"steps={row['steps']:>5} "
            f"manual-ad(JVP)={row['jvp_recomputed_minus_ad_loss']:.3e} "
            f"manual-ad(VJP)={row['vjp_recomputed_minus_ad_loss']:.3e} "
            f"u_rms={row['u_rms_diff']:.3e} "
            f"u_max={row['u_max_abs_diff']:.3e} "
            f"strict={strict} allclose={close} "
            f"writes JVP={row['jvp_min_write_count']}..{row['jvp_max_write_count']} "
            f"VJP={row['vjp_min_write_count']}..{row['vjp_max_write_count']}"
        )


if __name__ == "__main__":
    main()

