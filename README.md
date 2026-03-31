# PyTorch Vanishing Point Calibration

PyTorch implementation of the vanishing-point regression model from:

- Razvan Itu, Diana Borza, Radu Danescu, *Automatic extrinsic camera parameters calibration using Convolutional Neural Networks*, ICCP 2017

This repository focuses on:

- reproducing the paper's CNN architecture and training setup
- generating VP labels from CULane lane annotations for local experiments
- training and evaluating the model on Apple Silicon with `mps`

## What Is Included

- paper-style VP regression CNN
- CULane VP ground-truth generation pipeline
- fixed train/validation split generation
- training, inference, and prediction-visualization scripts
- a tracked pretrained checkpoint

## What Is Not Included

- the original OpenStreetCam dataset used in the paper
- raw dataset files from CULane

`data/` and most generated artifacts under `outputs/` are ignored by git.

## Setup

```bash
uv sync
```

## Dataset

Download CULane helper files:

```bash
uv run python scripts/download_culane.py
```

If you only want split files and annotations first:

```bash
uv run python scripts/download_culane.py --files annotations_new list
```

## Generate VP Ground Truth

CULane does not provide VP labels, so this repository generates them from lane annotations.
This part is a CULane adaptation and is not part of the original paper dataset.

```bash
uv run python scripts/generate_vp_gt.py \
  --culane-root data/CULane \
  --split-file data/CULane/list/train_gt.txt \
  --output outputs/vp_gt_train.csv \
  --poly-degree 2
```

## Train

Paper-aligned defaults:

- input size: `160x48`
- optimizer: `Adam`
- learning rate: `1e-3`
- batch size: `256`
- max epochs: `20`
- augmentation: horizontal flip
- loss: RMSE

Create a fixed 90/10 split:

```bash
uv run python scripts/create_vp_split.py \
  --gt-csv outputs/vp_gt_train.csv \
  --output-dir outputs/splits/paper_90_10
```

Train on the fixed split:

```bash
uv run python scripts/train_vp_model.py \
  --culane-root data/CULane \
  --train-gt-csv outputs/splits/paper_90_10/train.csv \
  --val-gt-csv outputs/splits/paper_90_10/val.csv \
  --output-dir outputs/train_runs/paper_split_v1
```

Quick smoke test:

```bash
uv run python scripts/train_vp_model.py \
  --culane-root data/CULane \
  --gt-csv outputs/vp_gt_train.csv \
  --output-dir outputs/train_smoke \
  --epochs 1 \
  --batch-size 32 \
  --num-workers 0 \
  --max-train-samples 128 \
  --max-val-samples 32
```

## Pretrained

Tracked checkpoint:

- `pretrained/paper_split_v1_best_model.pt`

Example inference:

```bash
uv run python scripts/infer_vp_model.py \
  --image-path data/CULane/driver_161_90frame/06031222_0836.MP4/00000.jpg \
  --checkpoint pretrained/paper_split_v1_best_model.pt \
  --focal-length-px 800 \
  --angles-in-degrees
```

## Results

Current preserved CULane experiment:

| Item | Value |
| --- | --- |
| GT CSV | `outputs/vp_gt_train.csv` |
| Valid VP labels | `15,905` |
| Train split | `14,314` |
| Validation split | `1,591` |
| Epochs run | `17` |
| Best epoch | `12` |
| Best validation RMSE | `3.3975` |
| Best validation NormDist | `0.02221` |
| Checkpoint | `pretrained/paper_split_v1_best_model.pt` |

## Visualization

Plot training history:

```bash
uv run python scripts/plot_training_history.py \
  --history-csv outputs/train_runs/paper_split_v1/history.csv \
  --output outputs/train_runs/paper_split_v1/history_plot.png
```

Visualize validation predictions:

```bash
uv run python scripts/visualize_model_predictions.py \
  --culane-root data/CULane \
  --gt-csv outputs/splits/paper_90_10/val.csv \
  --checkpoint pretrained/paper_split_v1_best_model.pt \
  --output-dir outputs/train_runs/paper_split_v1/prediction_viz
```

Visualize generated VP GT:

```bash
uv run python scripts/visualize_vp_gt.py \
  --culane-root data/CULane \
  --gt-csv outputs/vp_gt_train.csv \
  --output-dir outputs/viz/vp_gt
```

## Pitch And Yaw

The repository also includes the VP-to-pitch/yaw conversion described in the paper:

- `pitch = original_pitch + delta_pitch`
- `delta_pitch = (original_vp_y - predicted_vp_y) / focal_length`
- yaw is computed analogously from the horizontal offset

Implementation:

- `src/geometry/vp_to_extrinsics.py`

## Project Structure

```text
scripts/
  create_vp_split.py
  download_culane.py
  generate_vp_gt.py
  infer_vp_model.py
  plot_training_history.py
  train_vp_model.py
  visualize_model_predictions.py
  visualize_vp_gt.py
src/
  datasets/
  geometry/
  gt/
  models/
  training/
pretrained/
```

## Notes

- The paper specifies the channel counts and kernel sizes, but not convolution stride or padding in the text.
- This implementation uses stride-2 convolutions as a practical downsampling choice before the fully connected layers.
- The model and training recipe are paper-aligned, but the CULane VP labels are generated locally rather than coming from the original paper dataset.
