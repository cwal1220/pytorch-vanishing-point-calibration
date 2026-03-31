from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.culane import resolve_culane_path
from src.gt.vp_from_lanes import fit_lane_top_segment, load_lane_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CULane VP GT samples.")
    parser.add_argument("--culane-root", type=Path, required=True)
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--poly-degree", type=int, default=2, choices=[2, 3])
    parser.add_argument("--only-valid", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-width", type=int, default=1640)
    parser.add_argument("--image-height", type=int, default=590)
    return parser.parse_args()


def load_rows(path: Path, only_valid: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if only_valid and row["valid"] != "1":
                continue
            rows.append(row)
    return rows


def draw_sample(
    row: dict[str, str],
    culane_root: Path,
    output_path: Path,
    image_width: int,
    image_height: int,
    poly_degree: int,
) -> None:
    image_path = resolve_culane_path(culane_root, row["image_path"])
    line_path = resolve_culane_path(culane_root, row["line_path"])
    lanes = load_lane_points(line_path)

    fig, ax = plt.subplots(figsize=(16.4, 5.9), dpi=100)

    if image_path.exists():
        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)
    else:
        blank = np.full((image_height, image_width, 3), 245, dtype=np.uint8)
        ax.imshow(blank)

    lane_colors = ["#d7263d", "#f49d37", "#2a9d8f", "#457b9d", "#6a4c93", "#1982c4"]

    for idx, lane in enumerate(lanes):
        color = lane_colors[idx % len(lane_colors)]
        ax.plot(lane[:, 0], lane[:, 1], "o", ms=3, color=color, alpha=0.8)

        fit = fit_lane_top_segment(lane, polynomial_degree=poly_degree)
        if fit is None:
            continue

        y_curve = np.linspace(fit.y_min, fit.y_max, 80)
        x_curve = np.polyval(np.asarray(fit.coefficients), y_curve)
        ax.plot(x_curve, y_curve, "-", lw=2.2, color=color)

        y_tangent = np.linspace(-image_height * 0.25, fit.tangent_y + 40.0, 60)
        x_tangent = fit.tangent_slope * y_tangent + fit.tangent_intercept
        ax.plot(x_tangent, y_tangent, "--", lw=1.5, color=color, alpha=0.8)
        ax.plot([fit.tangent_x], [fit.tangent_y], "o", ms=6, color=color, mec="black", mew=0.6)

    if row["valid"] == "1" and row["vp_x"] and row["vp_y"]:
        vp_x = float(row["vp_x"])
        vp_y = float(row["vp_y"])
        ax.scatter([vp_x], [vp_y], s=120, c="#111111", marker="x", linewidths=2.5)
        ax.scatter([vp_x], [vp_y], s=180, facecolors="none", edgecolors="#111111", linewidths=1.5)

    title = (
        f"{row['image_path']} | valid={row['valid']} | reason={row['reason']} | "
        f"conf={float(row['confidence']):.3f} | spread={float(row['spread']) if row['spread'] else math.nan:.2f}"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, -image_height * 0.35)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def draw_distribution(rows: list[dict[str, str]], output_path: Path, image_width: int, image_height: int) -> None:
    valid_rows = [row for row in rows if row["valid"] == "1" and row["vp_x"] and row["vp_y"]]
    if not valid_rows:
        return

    vp_x = np.asarray([float(row["vp_x"]) for row in valid_rows])
    vp_y = np.asarray([float(row["vp_y"]) for row in valid_rows])
    confidence = np.asarray([float(row["confidence"]) for row in valid_rows])

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    scatter = ax.scatter(vp_x, vp_y, c=confidence, s=9, cmap="viridis", alpha=0.65)
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, -image_height * 0.35)
    ax.set_xlabel("VP x")
    ax.set_ylabel("VP y")
    ax.set_title("VP Distribution")
    fig.colorbar(scatter, ax=ax, label="confidence")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.gt_csv, only_valid=args.only_valid)
    if not rows:
        raise SystemExit("No rows available to visualize.")

    rng = np.random.default_rng(args.seed)
    sample_count = min(args.num_samples, len(rows))
    sample_indices = rng.choice(len(rows), size=sample_count, replace=False)
    selected_rows = [rows[int(index)] for index in np.sort(sample_indices)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_distribution(
        rows=rows,
        output_path=args.output_dir / "vp_distribution.png",
        image_width=args.image_width,
        image_height=args.image_height,
    )

    for index, row in enumerate(selected_rows):
        stem = row["line_path"].strip("/").replace("/", "__").replace(".lines.txt", "")
        output_path = args.output_dir / f"{index:02d}_{stem}.png"
        draw_sample(
            row=row,
            culane_root=args.culane_root,
            output_path=output_path,
            image_width=args.image_width,
            image_height=args.image_height,
            poly_degree=args.poly_degree,
        )

    print(f"saved_dir={args.output_dir}")
    print(f"num_rows={len(rows)}")
    print(f"num_visualized={sample_count}")


if __name__ == "__main__":
    main()
