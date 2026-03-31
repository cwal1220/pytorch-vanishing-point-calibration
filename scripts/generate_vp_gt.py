from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.culane import image_path, iter_split_file, line_annotation_path
from src.gt.vp_from_lanes import estimate_vp_from_lanes, load_lane_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate vanishing-point GT from CULane lane annotations.")
    parser.add_argument("--culane-root", type=Path, required=True, help="Root directory of extracted CULane data.")
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="Path to a CULane split file such as list/train_gt.txt or list/val_gt.txt.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for quick dry-runs.")
    parser.add_argument("--poly-degree", type=int, default=2, choices=[2, 3], help="Polynomial degree for lane fitting.")
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Keep invalid samples in the output CSV for debugging and threshold tuning.",
    )
    parser.add_argument("--image-width", type=int, default=1640)
    parser.add_argument("--image-height", type=int, default=590)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records = iter_split_file(args.split_file)
    if args.max_samples is not None:
        from itertools import islice

        records = islice(records, args.max_samples)

    fieldnames = [
        "image_path",
        "line_path",
        "image_exists",
        "line_exists",
        "lane_0",
        "lane_1",
        "lane_2",
        "lane_3",
        "valid",
        "vp_x",
        "vp_y",
        "vp_x_norm",
        "vp_y_norm",
        "spread",
        "confidence",
        "num_lanes",
        "num_intersections",
        "num_inliers",
        "mean_lane_rmse",
        "reason",
    ]

    stats = Counter()

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for sample in tqdm(records, desc=f"Generating {args.split_file.name}"):
            img_path = image_path(args.culane_root, sample)
            line_path = line_annotation_path(args.culane_root, sample)
            stats["total"] += 1

            row = {
                "image_path": sample.image_path,
                "line_path": sample.line_annotation_path,
                "image_exists": int(img_path.exists()),
                "line_exists": int(line_path.exists()),
                "lane_0": sample.lane_exists[0],
                "lane_1": sample.lane_exists[1],
                "lane_2": sample.lane_exists[2],
                "lane_3": sample.lane_exists[3],
            }

            if not line_path.exists():
                row.update(
                    {
                        "valid": 0,
                        "vp_x": "",
                        "vp_y": "",
                        "vp_x_norm": "",
                        "vp_y_norm": "",
                        "spread": "",
                        "confidence": 0.0,
                        "num_lanes": 0,
                        "num_intersections": 0,
                        "num_inliers": 0,
                        "mean_lane_rmse": "",
                        "reason": "missing_line_annotation",
                    }
                )
                stats["missing_line_annotation"] += 1
                if args.include_invalid:
                    writer.writerow(row)
                continue

            lanes = load_lane_points(line_path)
            result = estimate_vp_from_lanes(
                lanes,
                image_width=args.image_width,
                image_height=args.image_height,
                polynomial_degree=args.poly_degree,
            )

            row.update(
                {
                    "valid": int(result.valid),
                    "vp_x": result.vp_x if result.valid or args.include_invalid else "",
                    "vp_y": result.vp_y if result.valid or args.include_invalid else "",
                    "vp_x_norm": (result.vp_x / args.image_width) if result.valid or args.include_invalid else "",
                    "vp_y_norm": (result.vp_y / args.image_height) if result.valid or args.include_invalid else "",
                    "spread": result.spread if result.valid or args.include_invalid else "",
                    "confidence": result.confidence,
                    "num_lanes": result.num_lanes,
                    "num_intersections": result.num_intersections,
                    "num_inliers": result.num_inliers,
                    "mean_lane_rmse": result.mean_lane_rmse if result.mean_lane_rmse < float("inf") else "",
                    "reason": result.reason,
                }
            )

            stats[result.reason] += 1
            if result.valid:
                stats["valid"] += 1

            if result.valid or args.include_invalid:
                writer.writerow(row)

    valid = stats.get("valid", 0)
    total = stats.get("total", 0)
    ratio = valid / total if total else 0.0

    print(f"split_file={args.split_file}")
    print(f"output={args.output}")
    print(f"valid={valid} total={total} ratio={ratio:.4f}")
    for key, value in sorted(stats.items()):
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
