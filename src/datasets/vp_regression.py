from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class VPDatasetConfig:
    input_width: int = 160
    input_height: int = 48
    crop_mode: str = "paper_aspect_bottom"
    horizontal_flip_prob: float = 0.0


@dataclass(frozen=True)
class VPRecord:
    image_path: str
    line_path: str
    vp_x: float
    vp_y: float


def load_vp_records(gt_csv: Path) -> list[VPRecord]:
    records: list[VPRecord] = []
    with gt_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["valid"] != "1":
                continue
            records.append(
                VPRecord(
                    image_path=row["image_path"],
                    line_path=row["line_path"],
                    vp_x=float(row["vp_x"]),
                    vp_y=float(row["vp_y"]),
                )
            )
    return records


def compute_crop_box(width: int, height: int, config: VPDatasetConfig) -> tuple[int, int, int, int]:
    target_aspect_height = round(width * config.input_height / config.input_width)
    crop_height = min(height, target_aspect_height)

    if config.crop_mode == "paper_aspect_bottom":
        top = max(0, height - crop_height)
        return 0, top, width, height

    if config.crop_mode == "paper_aspect_center":
        top = max(0, (height - crop_height) // 2)
        bottom = min(height, top + crop_height)
        return 0, top, width, bottom

    if config.crop_mode == "full_frame":
        return 0, 0, width, height

    raise ValueError(f"Unsupported crop mode: {config.crop_mode}")


def vp_to_model_space(
    vp_x: float,
    vp_y: float,
    crop_box: tuple[int, int, int, int],
    output_width: int,
    output_height: int,
) -> tuple[float, float]:
    left, top, right, bottom = crop_box
    crop_width = right - left
    crop_height = bottom - top

    vp_x = (vp_x - left) * output_width / crop_width
    vp_y = (vp_y - top) * output_height / crop_height
    return vp_x, vp_y


def vp_to_image_space(
    vp_x: float,
    vp_y: float,
    crop_box: tuple[int, int, int, int],
    input_width: int,
    input_height: int,
) -> tuple[float, float]:
    left, top, right, bottom = crop_box
    crop_width = right - left
    crop_height = bottom - top

    image_x = left + float(vp_x) * crop_width / input_width
    image_y = top + float(vp_y) * crop_height / input_height
    return image_x, image_y


class VPRegressionDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        culane_root: Path,
        gt_csv: Path,
        config: VPDatasetConfig,
        indices: list[int] | None = None,
    ) -> None:
        self.culane_root = culane_root
        self.records = load_vp_records(gt_csv)
        self.config = config
        self.indices = indices if indices is not None else list(range(len(self.records)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[self.indices[index]]
        image_path = self.culane_root / record.image_path.lstrip("/")
        image = Image.open(image_path).convert("RGB")
        crop_box = compute_crop_box(image.width, image.height, self.config)
        image = image.crop(crop_box)
        image = image.resize((self.config.input_width, self.config.input_height), resample=Image.BILINEAR)

        target_x, target_y = vp_to_model_space(
            vp_x=record.vp_x,
            vp_y=record.vp_y,
            crop_box=crop_box,
            output_width=self.config.input_width,
            output_height=self.config.input_height,
        )

        image_tensor = TF.to_tensor(image)
        target = torch.tensor([target_x, target_y], dtype=torch.float32)

        if self.config.horizontal_flip_prob > 0.0:
            if torch.rand(1).item() < self.config.horizontal_flip_prob:
                image_tensor = torch.flip(image_tensor, dims=[2])
                target[0] = float(self.config.input_width - 1) - target[0]

        return {
            "image": image_tensor,
            "target": target,
            "image_path": record.image_path,
        }


def split_indices(num_samples: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=generator).tolist()
    train_size = int(num_samples * train_ratio)
    train_indices = perm[:train_size]
    val_indices = perm[train_size:]
    return train_indices, val_indices
