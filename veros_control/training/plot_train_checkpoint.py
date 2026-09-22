import argparse
import csv
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def _as_float(value):
    try:
        return float(np.asarray(value))
    except (TypeError, ValueError):
        return np.nan


def _flatten_numeric_dict(prefix, value):
    if not isinstance(value, dict):
        return {}
    return {
        f"{prefix}_{key}": _as_float(item)
        for key, item in value.items()
    }


def load_train_history(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    history = checkpoint.get("train_history")
    if history:
        rows = []
        for i, item in enumerate(history):
            rows.append(
                {
                    "update": int(item.get("update", i + 1)),
                    "loss": _as_float(item.get("loss")),
                    "lr": _as_float(item.get("lr", np.nan)),
                    "epsilon": _as_float(item.get("epsilon", np.nan)),
                    "grad_norm": _as_float(item.get("grad_norm", np.nan)),
                    "grad_max_abs": _as_float(item.get("grad_max_abs", np.nan)),
                    "update_norm": _as_float(item.get("update_norm", np.nan)),
                    "update_max_abs": _as_float(item.get("update_max_abs", np.nan)),
                    "param_norm": _as_float(item.get("param_norm", np.nan)),
                    "param_max_abs": _as_float(item.get("param_max_abs", np.nan)),
                    "selected_grad": _as_float(item.get("selected_grad", np.nan)),
                    "time": _as_float(item.get("time", np.nan)),
                    **_flatten_numeric_dict("loss_term", item.get("loss_terms")),
                    **_flatten_numeric_dict("normalized_loss_term", item.get("normalized_loss_terms")),
                    **_flatten_numeric_dict("tendency_ratio_rms", item.get("tendency_ratio_rms")),
                    **_flatten_numeric_dict("tendency_ratio_max", item.get("tendency_ratio_max")),
                }
            )
        return rows, checkpoint

    losses = checkpoint.get("losses", [])
    rows = [
        {
            "update": i + 1,
            "loss": _as_float(loss),
            "lr": np.nan,
            "epsilon": np.nan,
            "grad_norm": np.nan,
            "grad_max_abs": np.nan,
            "update_norm": np.nan,
            "update_max_abs": np.nan,
            "param_norm": np.nan,
            "param_max_abs": np.nan,
            "selected_grad": np.nan,
            "time": np.nan,
        }
        for i, loss in enumerate(losses)
    ]
    return rows, checkpoint


def write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _array_from_rows(rows, key):
    return np.asarray([row.get(key, np.nan) for row in rows], dtype=float)


def _loss_term_keys(rows):
    return sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("loss_term_")
        }
    )


def _title_bits(checkpoint):
    title_bits = ["NN train checkpoint"]
    if "n_steps" in checkpoint:
        title_bits.append(f"n_steps={checkpoint['n_steps']}")
    if "epsilon" in checkpoint:
        title_bits.append(f"epsilon={checkpoint['epsilon']}")
    if checkpoint.get("epsilon_final") is not None:
        title_bits.append(f"epsilon_final={checkpoint['epsilon_final']}")
    if "loss_mode" in checkpoint:
        title_bits.append(str(checkpoint["loss_mode"]))
    return title_bits


def plot_history(rows, checkpoint, output_path):
    if not rows:
        raise ValueError("Checkpoint has no train_history or losses to plot")

    updates = np.asarray([row["update"] for row in rows], dtype=float)
    loss = _array_from_rows(rows, "loss")
    lr = _array_from_rows(rows, "lr")
    epsilon = _array_from_rows(rows, "epsilon")
    grad_norm = _array_from_rows(rows, "grad_norm")
    update_norm = _array_from_rows(rows, "update_norm")
    selected_grad = _array_from_rows(rows, "selected_grad")

    has_extra = np.any(np.isfinite(grad_norm)) or np.any(np.isfinite(update_norm))
    if has_extra:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
        axes = np.asarray([axes])

    fig.suptitle(" | ".join(_title_bits(checkpoint)), fontsize=12)

    ax = axes[0]
    ax.plot(updates, loss, marker="o")
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    if has_extra:
        ax = axes[1]
        if np.any(np.isfinite(lr)):
            ax.plot(updates, lr, marker="o", color="tab:green")
            ax.set_yscale("log")
        ax.set_xlabel("Update")
        ax.set_ylabel("Learning rate")
        ax.set_title("LR / Epsilon Schedule")
        ax.grid(True, which="both", alpha=0.3)
        if np.any(np.isfinite(epsilon)):
            ax_eps = ax.twinx()
            ax_eps.plot(updates, epsilon, marker="s", color="tab:purple")
            ax_eps.set_ylabel("Epsilon")
            ax_eps.set_yscale("log")

        ax = axes[2]
        if np.any(np.isfinite(grad_norm)):
            ax.plot(updates, grad_norm, marker="o", label="grad_norm")
        if np.any(np.isfinite(update_norm)):
            ax.plot(updates, update_norm, marker="s", label="update_norm")
        ax.set_yscale("log")
        ax.set_xlabel("Update")
        ax.set_ylabel("Norm")
        ax.set_title("Gradient / Update Norm")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

        ax = axes[3]
        if np.any(np.isfinite(selected_grad)):
            ax.plot(updates, selected_grad, marker="o", color="tab:red")
            finite_abs = np.abs(selected_grad[np.isfinite(selected_grad)])
            finite_abs = finite_abs[finite_abs > 0]
            linthresh = float(np.min(finite_abs) / 10.0) if finite_abs.size else 1e-30
            ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_xlabel("Update")
        ax.set_ylabel("Selected gradient")
        ax.set_title("Selected Scalar Gradient")
        ax.grid(True, which="both", alpha=0.3)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_loss_terms(rows, checkpoint, output_path):
    loss_term_keys = _loss_term_keys(rows)
    if not loss_term_keys:
        return False

    updates = np.asarray([row["update"] for row in rows], dtype=float)
    normalized_keys = sorted({key for row in rows for key in row if key.startswith("normalized_loss_term_")})
    if normalized_keys:
        fig, (ax, normalized_ax) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for key in normalized_keys:
            normalized_ax.plot(updates, _array_from_rows(rows, key), marker="o",
                               label=key.removeprefix("normalized_loss_term_"))
        normalized_ax.set(xlabel="Update", ylabel="Loss / initial loss", title="Normalized Loss Terms")
        normalized_ax.legend()
        normalized_ax.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
    fig.suptitle(" | ".join(_title_bits(checkpoint)), fontsize=12)

    plotted = False
    for key in loss_term_keys:
        term = _array_from_rows(rows, key)
        if np.any(np.isfinite(term)):
            ax.plot(
                updates,
                term,
                marker="o",
                label=key.removeprefix("loss_term_"),
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Update")
    ax.set_ylabel("Loss contribution")
    ax.set_title("Temp / Salt Loss Terms")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def default_output_path(checkpoint_path):
    root, _ = os.path.splitext(checkpoint_path)
    return root + "_loss.png"


def default_terms_output_path(checkpoint_path):
    root, _ = os.path.splitext(checkpoint_path)
    return root + "_loss_terms.png"


def default_csv_path(checkpoint_path):
    root, _ = os.path.splitext(checkpoint_path)
    return root + "_history.csv"


def main():
    parser = argparse.ArgumentParser(description="Plot training loss from a Veros-control checkpoint.")
    parser.add_argument("checkpoint", help="Training checkpoint .pkl file.")
    parser.add_argument("--output", default=None, help="Output PNG path. Default: <checkpoint>_loss.png.")
    parser.add_argument(
        "--terms-output",
        default=None,
        help="Output PNG path for temp/salt loss terms. Default: <checkpoint>_loss_terms.png.",
    )
    parser.add_argument("--csv", default=None, help="Optional CSV path. Default: <checkpoint>_history.csv.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write a CSV summary.")
    args = parser.parse_args()

    output_path = args.output or default_output_path(args.checkpoint)
    terms_output_path = args.terms_output or default_terms_output_path(args.checkpoint)
    csv_path = args.csv or default_csv_path(args.checkpoint)

    rows, checkpoint = load_train_history(args.checkpoint)
    plot_history(rows, checkpoint, output_path)
    print(f"Saved loss plot -> {output_path}")
    if plot_loss_terms(rows, checkpoint, terms_output_path):
        print(f"Saved loss-terms plot -> {terms_output_path}")

    if not args.no_csv:
        write_csv(rows, csv_path)
        print(f"Saved history CSV -> {csv_path}")

    if rows:
        first_loss = rows[0]["loss"]
        last_loss = rows[-1]["loss"]
        print(
            f"updates={len(rows)} "
            f"loss_start={first_loss:.12e} "
            f"loss_end={last_loss:.12e} "
            f"delta={last_loss - first_loss:.12e}"
        )


if __name__ == "__main__":
    main()

