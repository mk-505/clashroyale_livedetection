# Data

## Overview
This project does not include the full dataset due to size and licensing constraints.

The perception model is trained on a Clash Royale dataset derived from the KataCR framework.

## Dataset Source
KataCR:  
https://github.com/wty-yy/KataCR

## Expected Structure
Place your dataset in the following format:

```text
data/
images/
labels/
```

- `images/` -> gameplay frames
- `labels/` -> YOLO-format annotations

## Notes
- The dataset includes both manually labeled and synthetic images
- Dataset preparation should follow the YOLO training format

## Reproducing Perception Model
1. Download dataset from KataCR
2. Ensure YOLO-compatible structure
3. Train or fine-tune the detection model
