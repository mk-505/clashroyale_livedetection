# ClashBotPro

## Overview

ClashBotPro is a **vision-based reinforcement learning system** for Clash Royale that learns directly from **live gameplay**, without access to a simulator or structured game state.

The system integrates:
- A **YOLO-based perception module** for real-time object detection
- A **Proximal Policy Optimization (PPO)** agent trained on live interactions

This repository provides the **full pipeline** for:
- Live detection
- Data collection
- Training
- Evaluation

---

## Repository Structure

```
src/cr_rl_live/   # Core RL logic and runtime system
scripts/          # Helper utilities
data/             # Dataset instructions
models/           # Model checkpoints and weights
assets/           # Example visuals and figures
```

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Place required model files in `models/`:

```
models/yolov8.pt   # Base YOLO weights
models/best.pt     # Fine-tuned detection model
models/ppo.pt      # (Optional) trained PPO checkpoint
```

---

## Dataset

This project uses a Clash Royale dataset derived from the KataCR framework. See:

```
data/README.md
```

---

## Models

Required components:
- YOLO detector weights (`best.pt` for inference)
- PPO policy checkpoint (`ppo.pt`, optional)

See:

```
models/README.md
```

---

## Running the System

### Quick Test (Recommended)

Run live detection to verify setup:

```bash
python3 live_detection.py --model models/best.pt --source "window:MuMu"
```

### 1. Live Detection (YOLO only)

```bash
python3 live_detection.py --model models/best.pt --source "window:MuMu"
```

### 2. Random Baseline Agent

```bash
python3 live_random.py --model models/best.pt --source "window:MuMu" --episode-dir episodes_random
```

### 3. Trained PPO Agent (Inference)

```bash
python3 live_infer.py --model models/best.pt --checkpoint models/ppo.pt --source "window:MuMu" --episode-dir episodes
```

### 4. Train PPO from Collected Episodes

```bash
python3 train_ppo.py --episode-dir episodes --checkpoint models/ppo.pt
```

### 5. Validate Rollout Data

```bash
python3 episode_sanity_check.py --episode-dir episodes --last 5
```

---

## Runtime Notes

- Designed for Windows + MuMu emulator (`window:MuMu`)
- Alternative input sources:
  ```
  --source 2   # webcam or capture device
  ```
- Real-time inference is required; performance depends on system latency
- `live_random.py` → random baseline (no learning)
- `live_infer.py` → trained PPO agent
- Both use a 13-action discrete control space

---

## Reproducing Results

1. Collect gameplay data using live episodes
2. Train PPO on collected transitions
3. Evaluate the trained agent against human opponents

> **Important:** Training is performed on real gameplay (no simulator). Results may vary due to stochastic learning and real-time environment noise.

---

## License

See `LICENSE` for details.