# Automatic Extrinsic Camera Calibration Using CNNs

`uv` based environment for a paper-aligned PyTorch implementation of:

- Razvan Itu, Diana Borza, Radu Danescu, "Automatic extrinsic camera parameters calibration using Convolutional Neural Networks", ICCP 2017

This repository keeps the CNN architecture, training recipe, and pitch/yaw conversion aligned with the paper.
For local experiments, VP labels can be generated from CULane lane annotations, but the original OpenStreetCam dataset from the paper is not bundled here.

## Environment

```bash
uv sync
```

## Download

Official CULane files are hosted on Google Drive, so large archives can occasionally fail because of quota or permission limits.

```bash
uv run python scripts/download_culane.py
```

If only the annotation and split files are needed first:

```bash
uv run python scripts/download_culane.py --files annotations_new list
```

The repository does not include dataset files or generated experiment artifacts.
Keep local data under `data/` and generated results under `outputs/`; both are ignored by git.

## Generate VP GT

This step is a local dataset adaptation for CULane and is not part of the original paper dataset release.

Generate VP GT from raw lane annotations:

```bash
uv run python scripts/generate_vp_gt.py \
  --culane-root data/CULane \
  --split-file data/CULane/list/train_gt.txt \
  --output outputs/vp_gt_train.csv \
  --poly-degree 2
```

## Train

Paper-aligned training choices:

- input size: `160x48`
- optimizer: `Adam`
- learning rate: `1e-3`
- batch size: `256`
- max epochs: `20`
- augmentation: horizontal flip
- loss: RMSE on VP coordinates

Run training:

```bash
uv run python scripts/train_vp_model.py \
  --culane-root data/CULane \
  --gt-csv outputs/vp_gt_train.csv \
  --output-dir outputs/train_runs/paper_v1
```

Create a fixed paper-style 90/10 split:

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
| Checkpoint | `outputs/train_runs/paper_split_v1/best_model.pt` |

## Pretrained

Tracked checkpoint included in this repository:

- `pretrained/paper_split_v1_best_model.pt`

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

Note:

- The paper text explicitly specifies kernel sizes and channel counts, but does not state convolution stride or padding in text.
- This implementation uses stride-2 convolutions as a practical inference so the CNN can downsample before the fully connected layers.

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
  --checkpoint outputs/train_runs/paper_split_v1/best_model.pt \
  --output-dir outputs/train_runs/paper_split_v1/prediction_viz
```

## Pitch And Yaw

The paper computes camera pitch and yaw from VP offsets using:

- `pitch = original_pitch + delta_pitch`
- `delta_pitch = (original_vp_y - predicted_vp_y) / focal_length`
- yaw uses the analogous `x` equation

This is implemented in:

- `src/geometry/vp_to_extrinsics.py`

Example inference:

```bash
uv run python scripts/infer_vp_model.py \
  --image-path data/CULane/driver_161_90frame/06031222_0836.MP4/00000.jpg \
  --checkpoint outputs/train_smoke/best_model.pt \
  --focal-length-px 800 \
  --angles-in-degrees
```
