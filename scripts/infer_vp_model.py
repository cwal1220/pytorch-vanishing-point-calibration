from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.vp_regression import VPDatasetConfig, compute_crop_box, vp_to_image_space
from src.geometry.vp_to_extrinsics import compute_pitch_yaw_from_vp, project_forward_axis_to_vp
from src.models.paper_vp_cnn import PaperVPCNN
from src.training.device import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VP inference and optional pitch/yaw conversion.")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--input-width", type=int, default=160)
    parser.add_argument("--input-height", type=int, default=48)
    parser.add_argument("--crop-mode", type=str, default="paper_aspect_bottom", choices=["paper_aspect_bottom", "paper_aspect_center", "full_frame"])
    parser.add_argument("--focal-length-px", type=float, default=None)
    parser.add_argument("--original-vp-x", type=float, default=None)
    parser.add_argument("--original-vp-y", type=float, default=None)
    parser.add_argument("--original-pitch", type=float, default=0.0)
    parser.add_argument("--original-yaw", type=float, default=0.0)
    parser.add_argument("--angles-in-degrees", action="store_true")
    return parser.parse_args()


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> PaperVPCNN:
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
    model = load_checkpoint_model(args.checkpoint, device=device)

    image = Image.open(args.image_path).convert("RGB")
    config = VPDatasetConfig(
        input_width=args.input_width,
        input_height=args.input_height,
        crop_mode=args.crop_mode,
    )
    crop_box = compute_crop_box(image.width, image.height, config)
    cropped = image.crop(crop_box).resize((args.input_width, args.input_height), resample=Image.BILINEAR)
    tensor = TF.to_tensor(cropped).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(tensor)[0].cpu()

    predicted_vp_image_x, predicted_vp_image_y = vp_to_image_space(
        vp_x=float(prediction[0].item()),
        vp_y=float(prediction[1].item()),
        crop_box=crop_box,
        input_width=args.input_width,
        input_height=args.input_height,
    )

    result: dict[str, object] = {
        "image_path": str(args.image_path),
        "predicted_vp_x": float(prediction[0].item()),
        "predicted_vp_y": float(prediction[1].item()),
        "predicted_vp_image_x": float(predicted_vp_image_x),
        "predicted_vp_image_y": float(predicted_vp_image_y),
        "crop_box": list(crop_box),
        "model_input_size": [args.input_width, args.input_height],
        "original_image_size": [image.width, image.height],
        "crop_mode": args.crop_mode,
    }

    if args.focal_length_px is not None:
        if args.original_vp_x is None or args.original_vp_y is None:
            inferred_original_vp_x, inferred_original_vp_y = project_forward_axis_to_vp(
                focal_length_px=args.focal_length_px,
                image_width=image.width,
                image_height=image.height,
                pitch=args.original_pitch,
                yaw=args.original_yaw,
                roll=0.0,
            )
        else:
            inferred_original_vp_x = args.original_vp_x
            inferred_original_vp_y = args.original_vp_y

        pitch_yaw = compute_pitch_yaw_from_vp(
            predicted_vp_x=float(predicted_vp_image_x),
            predicted_vp_y=float(predicted_vp_image_y),
            original_vp_x=float(inferred_original_vp_x),
            original_vp_y=float(inferred_original_vp_y),
            focal_length_px=args.focal_length_px,
            original_pitch=args.original_pitch,
            original_yaw=args.original_yaw,
            angles_in_degrees=args.angles_in_degrees,
        )
        result["pitch_yaw"] = pitch_yaw.to_dict()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
