from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.vp_regression import VPDatasetConfig, VPRegressionDataset, split_indices
from src.models.paper_vp_cnn import PaperVPCNN
from src.training.device import resolve_device
from src.training.vp_metrics import normdist, rmse_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ICCP 2017 VP CNN on the available CULane subset.")
    parser.add_argument("--culane-root", type=Path, required=True)
    parser.add_argument("--gt-csv", type=Path, default=None)
    parser.add_argument("--train-gt-csv", type=Path, default=None)
    parser.add_argument("--val-gt-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--input-width", type=int, default=160)
    parser.add_argument("--input-height", type=int, default=48)
    parser.add_argument("--crop-mode", type=str, default="paper_aspect_bottom", choices=["paper_aspect_bottom", "paper_aspect_center", "full_frame"])
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, int, int]:
    if args.train_gt_csv is not None and args.val_gt_csv is not None:
        train_dataset = VPRegressionDataset(
            culane_root=args.culane_root,
            gt_csv=args.train_gt_csv,
            config=VPDatasetConfig(
                input_width=args.input_width,
                input_height=args.input_height,
                crop_mode=args.crop_mode,
                horizontal_flip_prob=args.flip_prob,
            ),
        )
        val_dataset = VPRegressionDataset(
            culane_root=args.culane_root,
            gt_csv=args.val_gt_csv,
            config=VPDatasetConfig(
                input_width=args.input_width,
                input_height=args.input_height,
                crop_mode=args.crop_mode,
                horizontal_flip_prob=0.0,
            ),
        )
        if args.max_train_samples is not None:
            train_dataset.indices = train_dataset.indices[: args.max_train_samples]
        if args.max_val_samples is not None:
            val_dataset.indices = val_dataset.indices[: args.max_val_samples]
        train_size = len(train_dataset)
        val_size = len(val_dataset)
    else:
        if args.gt_csv is None:
            raise ValueError("Provide either --gt-csv or both --train-gt-csv and --val-gt-csv.")
        base_dataset = VPRegressionDataset(
            culane_root=args.culane_root,
            gt_csv=args.gt_csv,
            config=VPDatasetConfig(
                input_width=args.input_width,
                input_height=args.input_height,
                crop_mode=args.crop_mode,
                horizontal_flip_prob=0.0,
            ),
        )
        train_indices, val_indices = split_indices(len(base_dataset.records), train_ratio=args.train_ratio, seed=args.seed)
        if args.max_train_samples is not None:
            train_indices = train_indices[: args.max_train_samples]
        if args.max_val_samples is not None:
            val_indices = val_indices[: args.max_val_samples]

        train_dataset = VPRegressionDataset(
            culane_root=args.culane_root,
            gt_csv=args.gt_csv,
            config=VPDatasetConfig(
                input_width=args.input_width,
                input_height=args.input_height,
                crop_mode=args.crop_mode,
                horizontal_flip_prob=args.flip_prob,
            ),
            indices=train_indices,
        )
        val_dataset = VPRegressionDataset(
            culane_root=args.culane_root,
            gt_csv=args.gt_csv,
            config=VPDatasetConfig(
                input_width=args.input_width,
                input_height=args.input_height,
                crop_mode=args.crop_mode,
                horizontal_flip_prob=0.0,
            ),
            indices=val_indices,
        )
        train_size = len(train_indices)
        val_size = len(val_indices)

    pin_memory = False
    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, train_size, val_size


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam | None,
    device: torch.device,
    input_width: int,
    input_height: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_rmse = 0.0
    total_normdist = 0.0
    total_examples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            predictions = model(images)
            loss = rmse_loss(predictions, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = images.shape[0]
            total_examples += batch_size
            total_loss += float(loss.item()) * batch_size
            total_rmse += float(loss.item()) * batch_size
            total_normdist += float(normdist(predictions, targets, input_width, input_height).item()) * batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "rmse": total_rmse / max(total_examples, 1),
        "normdist": total_normdist / max(total_examples, 1),
    }


def save_history(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_loader, val_loader, train_size, val_size = build_loaders(args)

    model = PaperVPCNN()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    best_val_loss = math.inf
    epochs_without_improvement = 0

    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metadata = {
        "paper": {
            "epochs": 20,
            "batch_size": 256,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "loss": "RMSE",
            "augmentation": "horizontal flip",
            "input_size": [160, 48],
        },
        "implementation_notes": {
            "dataset": "Training uses VP labels provided through the GT CSV.",
            "crop_mode": args.crop_mode,
            "stride_choice": "Conv stride=2 is an implementation inference because the paper text omits stride/padding.",
        },
        "run_args": serializable_args,
        "dataset_sizes": {"train": train_size, "val": val_size},
        "device": str(device),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            input_width=args.input_width,
            input_height=args.input_height,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            input_width=args.input_width,
            input_height=args.input_height,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "train_normdist": train_metrics["normdist"],
            "val_loss": val_metrics["loss"],
            "val_rmse": val_metrics["rmse"],
            "val_normdist": val_metrics["normdist"],
        }
        history.append(row)
        save_history(history, args.output_dir / "history.csv")

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_rmse={train_metrics['rmse']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_rmse={val_metrics['rmse']:.4f} "
            f"val_normdist={val_metrics['normdist']:.5f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "metadata": metadata,
                },
                args.output_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"early_stop=1 epoch={epoch}")
                break

    torch.save(model.state_dict(), args.output_dir / "last_model_state_dict.pt")
    print(f"best_val_loss={best_val_loss:.6f}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
