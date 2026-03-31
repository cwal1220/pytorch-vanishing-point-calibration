from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed 90/10 train/val split for VP GT rows.")
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.gt_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows or fieldnames is None:
        raise SystemExit("No rows found in GT CSV.")

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(rows), generator=generator).tolist()
    train_size = int(len(rows) * args.train_ratio)
    train_indices = set(permutation[:train_size])
    val_indices = set(permutation[train_size:])

    train_rows = [rows[index] for index in range(len(rows)) if index in train_indices]
    val_rows = [rows[index] for index in range(len(rows)) if index in val_indices]

    train_csv = args.output_dir / "train.csv"
    val_csv = args.output_dir / "val.csv"
    train_list = args.output_dir / "train_image_paths.txt"
    val_list = args.output_dir / "val_image_paths.txt"

    for output_path, split_rows in [(train_csv, train_rows), (val_csv, val_rows)]:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)

    train_list.write_text("\n".join(row["image_path"] for row in train_rows) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(row["image_path"] for row in val_rows) + "\n", encoding="utf-8")

    metadata = {
        "source_gt_csv": str(args.gt_csv),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
    }
    (args.output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"train_size={len(train_rows)}")
    print(f"val_size={len(val_rows)}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
