import argparse
import glob
import os
import pickle

import numpy as np


GLORYS_VAR_MAP = {
    "temp": "thetao",
    "salt": "so",
    "u": "uo",
    "v": "vo",
}
FEATURE_NAMES = ("temp", "salt", "u_center", "v_center")
IGNORED_INPUT_SUFFIXES = (".lock",)


def resolve_files(patterns):
    files = []
    for pattern in patterns:
        matched = glob.glob(pattern)
        files.extend(matched if matched else [pattern])

    unique = []
    seen = set()
    for filename in sorted(files):
        if str(filename).endswith(IGNORED_INPUT_SUFFIXES):
            continue
        if filename not in seen:
            unique.append(filename)
            seen.add(filename)

    if not unique:
        raise FileNotFoundError(f"No GLORYS files matched: {patterns}")

    missing = [filename for filename in unique if not os.path.exists(filename)]
    if missing:
        raise FileNotFoundError(f"Missing GLORYS files: {missing}")

    return unique


def _replace_fill_values(data, attrs):
    data = np.asarray(data, dtype=np.float64)
    for attr_name in ("_FillValue", "missing_value"):
        if attr_name not in attrs:
            continue
        fill_values = np.asarray(attrs[attr_name]).reshape(-1)
        for fill_value in fill_values:
            if np.isfinite(fill_value):
                data[data == fill_value] = np.nan
    data[~np.isfinite(data)] = np.nan
    return data


def read_glorys_field(filename, key, time_index=0):
    import h5netcdf

    if key not in GLORYS_VAR_MAP:
        raise KeyError(f"No GLORYS variable mapping for feature key {key!r}")

    var_name = GLORYS_VAR_MAP[key]
    with h5netcdf.File(filename, "r") as infile:
        if var_name not in infile.variables:
            raise KeyError(f"{filename} has no variable {var_name!r}")
        var = infile.variables[var_name]
        data = _replace_fill_values(np.asarray(var), dict(var.attrs))

    # Match the orientation used by train.py: after transpose, time is last.
    data = data.T
    if data.ndim == 4:
        data = data[..., time_index]
    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D field for {filename}:{var_name}, got {data.shape}"
        )
    return data


def destagger_uv(u_raw, v_raw):
    u_center = 0.5 * (u_raw + np.roll(u_raw, shift=1, axis=0))
    v_center = 0.5 * (v_raw + np.roll(v_raw, shift=1, axis=1))
    return u_center, v_center


def _empty_accumulators(nz):
    shape = (nz, len(FEATURE_NAMES))
    return {
        "count": np.zeros(shape, dtype=np.float64),
        "sum": np.zeros(shape, dtype=np.float64),
        "sumsq": np.zeros(shape, dtype=np.float64),
        "min": np.full(shape, np.inf, dtype=np.float64),
        "max": np.full(shape, -np.inf, dtype=np.float64),
    }


def _accumulate_channel(acc, channel_index, field):
    finite = np.isfinite(field)
    count = finite.sum(axis=(0, 1), dtype=np.float64)
    values = np.where(finite, field, 0.0)
    mins = np.where(finite, field, np.inf).min(axis=(0, 1))
    maxs = np.where(finite, field, -np.inf).max(axis=(0, 1))
    valid = count > 0

    acc["count"][:, channel_index] += count
    acc["sum"][:, channel_index] += values.sum(axis=(0, 1))
    acc["sumsq"][:, channel_index] += np.square(values).sum(axis=(0, 1))
    acc["min"][valid, channel_index] = np.minimum(
        acc["min"][valid, channel_index],
        mins[valid],
    )
    acc["max"][valid, channel_index] = np.maximum(
        acc["max"][valid, channel_index],
        maxs[valid],
    )


def compute_feature_stats(files, time_index=0, method="meanstd", std_floor=1e-6):
    acc = None

    for file_index, filename in enumerate(files, start=1):
        print(f"[{file_index}/{len(files)}] reading {filename}")
        temp = read_glorys_field(filename, "temp", time_index=time_index)
        salt = read_glorys_field(filename, "salt", time_index=time_index)
        u_raw = read_glorys_field(filename, "u", time_index=time_index)
        v_raw = read_glorys_field(filename, "v", time_index=time_index)
        u_center, v_center = destagger_uv(u_raw, v_raw)

        features = (temp, salt, u_center, v_center)
        nz = temp.shape[2]
        if acc is None:
            acc = _empty_accumulators(nz)

        for feature in features:
            if feature.shape != temp.shape:
                raise ValueError(
                    f"All feature fields must have the same shape; "
                    f"got {feature.shape} and {temp.shape} in {filename}"
                )
            if feature.shape[2] != acc["count"].shape[0]:
                raise ValueError(
                    f"Inconsistent nz in {filename}: got {feature.shape[2]}, "
                    f"expected {acc['count'].shape[0]}"
                )

        for channel_index, feature in enumerate(features):
            _accumulate_channel(acc, channel_index, feature)

    if acc is None:
        raise ValueError("No files were accumulated")

    count = acc["count"]
    valid = count > 0
    mean = np.zeros_like(acc["sum"])
    mean[valid] = acc["sum"][valid] / count[valid]

    variance = np.zeros_like(acc["sum"])
    variance[valid] = acc["sumsq"][valid] / count[valid] - np.square(mean[valid])
    std = np.sqrt(np.maximum(variance, 0.0))
    std[valid] = np.maximum(std[valid], std_floor)
    std[~valid] = 1.0

    min_val = np.where(valid, acc["min"], 0.0)
    max_val = np.where(valid, acc["max"], 1.0)

    if method == "meanstd":
        stats_features = (mean, std)
    elif method == "minmax":
        stats_features = (min_val, max_val)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return stats_features, {
        "feature_names": FEATURE_NAMES,
        "count": count,
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "empty_levels": np.where(~valid),
    }


def load_feature_stats_file(path):
    with open(path, "rb") as infile:
        payload = pickle.load(infile)

    if isinstance(payload, dict) and "stats_features" in payload:
        stats = payload["stats_features"]
        method = payload.get("scaling_method_features")
        return stats, method, payload

    if isinstance(payload, (tuple, list)) and len(payload) == 2:
        return tuple(payload), None, {"stats_features": tuple(payload)}

    raise ValueError(
        f"{path} must contain either a stats_features payload or a (stat_a, stat_b) tuple"
    )


def save_feature_stats(path, stats_features, method, metadata, files, args):
    dtype = np.float32 if args.output_dtype == "float32" else np.float64
    stats_features = tuple(np.asarray(item, dtype=dtype) for item in stats_features)

    payload = {
        "stats_features": stats_features,
        "normalize_inputs": True,
        "scaling_method_features": method,
        "feature_names": FEATURE_NAMES,
        "source": "glorys",
        "source_files": list(files),
        "time_index": args.time_index,
        "std_floor": args.std_floor,
        "output_dtype": args.output_dtype,
        "metadata": metadata,
    }

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as outfile:
        pickle.dump(payload, outfile)
    return payload


def inject_stats_into_checkpoint(checkpoint_path, output_path, stats_payload, overwrite=False):
    if output_path is None:
        if not overwrite:
            raise ValueError(
                "Refusing to overwrite checkpoint without --overwrite-checkpoint. "
                "Pass --checkpoint-output or --overwrite-checkpoint."
            )
        output_path = checkpoint_path

    if os.path.abspath(output_path) == os.path.abspath(checkpoint_path) and not overwrite:
        raise ValueError(
            "Refusing to overwrite checkpoint without --overwrite-checkpoint."
        )

    with open(checkpoint_path, "rb") as infile:
        checkpoint = pickle.load(infile)

    checkpoint["stats_features"] = stats_payload["stats_features"]
    checkpoint["normalize_inputs"] = True
    checkpoint["scaling_method_features"] = stats_payload["scaling_method_features"]
    checkpoint["feature_stats_metadata"] = {
        key: value
        for key, value in stats_payload.items()
        if key not in ("stats_features",)
    }

    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "wb") as outfile:
        pickle.dump(checkpoint, outfile)

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build Veros-control NN input normalization stats from GLORYS files. "
            "The stats have shape (nz, 4) for temp, salt, centered u, centered v."
        )
    )
    parser.add_argument(
        "glorys_files",
        nargs="+",
        help="GLORYS .nc files or glob patterns. Files ending in .lock are ignored.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output pickle path for the feature stats payload.",
    )
    parser.add_argument(
        "--method",
        choices=("meanstd", "minmax"),
        default="meanstd",
        help="Normalization method to save.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Time index to read from each GLORYS file.",
    )
    parser.add_argument(
        "--std-floor",
        type=float,
        default=1e-6,
        help="Minimum std used for mean/std normalization.",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Dtype used to store the stats arrays.",
    )
    parser.add_argument(
        "--inject-checkpoint",
        default=None,
        help="Optional existing NN checkpoint to augment with these stats.",
    )
    parser.add_argument(
        "--checkpoint-output",
        default=None,
        help="Output checkpoint path when using --inject-checkpoint.",
    )
    parser.add_argument(
        "--overwrite-checkpoint",
        action="store_true",
        help="Allow --inject-checkpoint to overwrite the checkpoint in place.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    files = resolve_files(args.glorys_files)
    print(f"Using {len(files)} GLORYS files")

    stats_features, metadata = compute_feature_stats(
        files,
        time_index=args.time_index,
        method=args.method,
        std_floor=args.std_floor,
    )
    payload = save_feature_stats(
        args.output,
        stats_features,
        args.method,
        metadata,
        files,
        args,
    )
    print(f"Saved feature stats -> {args.output}")
    print(f"method={args.method}, feature_names={FEATURE_NAMES}")
    print(f"stats shapes={[item.shape for item in payload['stats_features']]}")

    if args.inject_checkpoint:
        checkpoint_output = inject_stats_into_checkpoint(
            args.inject_checkpoint,
            args.checkpoint_output,
            payload,
            overwrite=args.overwrite_checkpoint,
        )
        print(f"Saved checkpoint with feature stats -> {checkpoint_output}")


if __name__ == "__main__":
    main()

