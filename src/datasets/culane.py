from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


@dataclass(frozen=True)
class CULaneSample:
    image_path: str
    label_path: Optional[str]
    lane_exists: tuple[int, int, int, int]

    @property
    def line_annotation_path(self) -> str:
        return self.image_path.replace(".jpg", ".lines.txt")


def parse_split_line(line: str) -> CULaneSample:
    tokens = line.strip().split()
    if not tokens:
        raise ValueError("empty split line")

    image_path = tokens[0]
    label_path = None
    lane_exists = (0, 0, 0, 0)

    if len(tokens) == 6:
        label_path = tokens[1]
        lane_exists = tuple(int(x) for x in tokens[2:6])  # type: ignore[assignment]
    elif len(tokens) == 5:
        lane_exists = tuple(int(x) for x in tokens[1:5])  # type: ignore[assignment]

    return CULaneSample(
        image_path=image_path,
        label_path=label_path,
        lane_exists=lane_exists,
    )


def iter_split_file(split_file: Path) -> Iterator[CULaneSample]:
    with split_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            yield parse_split_line(raw_line)


def resolve_culane_path(culane_root: Path, relative_path: str) -> Path:
    return culane_root / relative_path.lstrip("/")


def line_annotation_path(culane_root: Path, sample: CULaneSample) -> Path:
    return resolve_culane_path(culane_root, sample.line_annotation_path)


def image_path(culane_root: Path, sample: CULaneSample) -> Path:
    return resolve_culane_path(culane_root, sample.image_path)
