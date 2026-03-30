# Dataset

This project uses a Clash Royale dataset derived from the KataCR framework.

Due to size and licensing constraints, the dataset is **not included** in this repository.

---

## Source

KataCR:  
https://github.com/wty-yy/KataCR

---

## Purpose

The dataset is used **only for training the perception module (YOLO object detector)**.

The reinforcement learning agent operates on detected features and does **not require direct access to the dataset** during inference.

---

## Expected Format

The dataset must be converted into standard YOLO format:

```
data/
  images/
    train/
    val/
  labels/
    train/
    val/
```

- `images/` → gameplay frames
- `labels/` → corresponding YOLO annotations

Each image must have a matching `.txt` file in the labels directory.

---

## YOLO Label Format

Each label file follows:

```
<class_id> <x_center> <y_center> <width> <height>
```

- All values are normalized between 0 and 1
- One line per object instance

---

## Training the Perception Model

Example training command using Ultralytics YOLO:

```bash
yolo detect train \
  data=data.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=640
```

After training, move the best checkpoint:

```
runs/detect/train/weights/best.pt → models/yolo.pt
```

---

## Notes

- Only object detection labels are used in this project
- Additional metadata from KataCR (e.g., OCR, classification, game state) is not required
- Dataset quality directly impacts detection performance, particularly recall

---

## Reproducibility

To reproduce the perception model:

1. Download dataset from KataCR
2. Convert to YOLO format
3. Train using the command above
4. Save weights to:
   ```
   models/yolo.pt
   ```

Pretrained weights are already provided in this repository for convenience.