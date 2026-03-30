This repository does not include the full dataset due to size and licensing constraints.

The perception model is trained on a Clash Royale dataset derived from the KataCR framework, consisting of labeled gameplay frames and synthetic compositions.

---

## Dataset Source

KataCR:  
https://github.com/wty-yy/KataCR

---

## Expected Structure

After downloading or preparing the dataset, organize it as:

```text
data/
  images/
  labels/
```

- `images/` → gameplay frames
- `labels/` → YOLO-format annotations

---

## Usage

- The dataset is required only for training the perception model
- Pretrained detection weights (`best.pt`) are already provided for inference

---

## Notes

- The dataset includes both manually labeled and synthetic images
- Labels follow the YOLO object detection format
- Dataset quality directly affects perception performance, particularly recall

---

## Reproducing the Perception Model

1. Download dataset from KataCR
2. Ensure YOLO-compatible directory structure
3. Train or fine-tune a YOLO model
4. Save weights as `best.pt` for use in the pipeline