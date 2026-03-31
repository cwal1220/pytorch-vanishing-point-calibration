from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


FILES = {
    "annotations_new": "1QbB1TOk9Fy6Sk0CoOsR3V8R56_eG6Xnu",
    "list": "18alVEPAMBA9Hpr3RDAAchqSj5IxZNRKd",
    "driver_23_30frame": "14Gi1AXbgkqvSysuoLyq1CsjFSypvoLVL",
    "driver_161_90frame": "1AQjQZwOAkeBTSG_1I9fYn8KBcxBBbYyk",
    "driver_182_30frame": "1PH7UdmtZOK3Qi3SBqtYOkWSH2dpbfmkL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the train/val portion of CULane with gdown.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--files",
        nargs="+",
        default=["annotations_new", "list", "driver_23_30frame", "driver_161_90frame", "driver_182_30frame"],
        choices=sorted(FILES.keys()),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gdown = shutil.which("gdown")
    if gdown is None:
        raise SystemExit("gdown is required. Install it with `python3 -m pip install --user gdown`.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.files:
        file_id = FILES[name]
        target = args.output_dir / f"{name}.tar.gz"
        cmd = [gdown, "--fuzzy", f"https://drive.google.com/open?id={file_id}", "-O", str(target)]
        print("Running:", " ".join(cmd))
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            print(f"Download failed for {name}. Google Drive quota is a common cause; rerun later if needed.")
            break


if __name__ == "__main__":
    main()
