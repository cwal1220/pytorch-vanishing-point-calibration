from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training history CSV.")
    parser.add_argument("--history-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.history_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise SystemExit("History CSV is empty.")

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_normdist = [float(row["train_normdist"]) for row in rows]
    val_normdist = [float(row["val_normdist"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, val_loss, marker="o", label="val")
    axes[0].set_title("RMSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_normdist, marker="o", label="train")
    axes[1].plot(epochs, val_normdist, marker="o", label="val")
    axes[1].set_title("NormDist")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("NormDist")
    axes[1].legend()

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"saved_plot={args.output}")


if __name__ == "__main__":
    main()
