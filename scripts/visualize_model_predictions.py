from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.vp_regression import VPDatasetConfig, compute_crop_box
from src.models.paper_vp_cnn import PaperVPCNN
from src.training.device import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GT vs predicted VP on validation images.")
    parser.add_argument("--culane-root", type=Path, required=True)
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--input-width", type=int, default=160)
    parser.add_argument("--input-height", type=int, default=48)
    parser.add_argument("--crop-mode", type=str, default="paper_aspect_bottom", choices=["paper_aspect_bottom", "paper_aspect_center", "full_frame"])
    return parser.parse_args()


def load_rows(gt_csv: Path) -> list[dict[str, str]]:
    with gt_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row["valid"] == "1"]


def load_model(checkpoint_path: Path, device: torch.device) -> PaperVPCNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PaperVPCNN()
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device=device)
    rows = load_rows(args.gt_csv)
    if not rows:
        raise SystemExit("No valid rows found.")

    rng = np.random.default_rng(args.seed)
    sample_count = min(args.num_samples, len(rows))
    indices = np.sort(rng.choice(len(rows), size=sample_count, replace=False))
    selected = [rows[int(index)] for index in indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for out_index, row in enumerate(selected):
        image_path = args.culane_root / row["image_path"].lstrip("/")
        image = Image.open(image_path).convert("RGB")
        crop_box = compute_crop_box(
            width=image.width,
            height=image.height,
            config=VPDatasetConfig(
                input_width=args.input_width,
                input_height=args.input_height,
                crop_mode=args.crop_mode,
            ),
        )
        cropped = image.crop(crop_box).resize((args.input_width, args.input_height), resample=Image.BILINEAR)
        image_tensor = TF.to_tensor(cropped).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(image_tensor)[0].cpu().numpy()

        gt_x = float(row["vp_x"])
        gt_y = float(row["vp_y"])
        left, top, right, bottom = crop_box
        gt_x = (gt_x - left) * args.input_width / (right - left)
        gt_y = (gt_y - top) * args.input_height / (bottom - top)

        fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
        ax.imshow(cropped)
        ax.scatter([gt_x], [gt_y], c="#22aa55", s=90, marker="o", label="GT")
        ax.scatter([prediction[0]], [prediction[1]], c="#ffffff", edgecolors="#111111", s=90, marker="x", linewidths=2, label="Pred")
        ax.set_title(
            f"{row['image_path']} | GT=({gt_x:.1f}, {gt_y:.1f}) | Pred=({prediction[0]:.1f}, {prediction[1]:.1f})",
            fontsize=9,
        )
        ax.axis("off")
        ax.legend(loc="upper right")
        fig.tight_layout()
        output_path = args.output_dir / f"{out_index:02d}_{row['image_path'].strip('/').replace('/', '__').replace('.jpg', '')}.png"
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    print(f"saved_dir={args.output_dir}")
    print(f"num_visualized={sample_count}")


if __name__ == "__main__":
    main()
