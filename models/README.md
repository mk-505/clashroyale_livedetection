# Models

## Overview
This project uses two main models:

1. YOLO object detection model
2. PPO policy model

## YOLO Model
The perception module uses a YOLO-based detector trained on Clash Royale gameplay frames.

To use:
- Download or train YOLO weights
- Place weights in this directory

Example:

```text
models/yolo.pt
```

## PPO Model
The PPO agent is trained using live gameplay data.

To use a trained model:
- Place checkpoint file in this directory

Example:

```text
models/ppo.pt
```

## Notes
- Pretrained models are not included due to size constraints
- Users can train models using:
  - `train_ppo.py` for policy
  - the YOLO training pipeline for detection

## Optional
You may provide a download link for pretrained models here if available.
