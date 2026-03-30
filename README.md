# ClashBotPro

## Overview
ClashBotPro is a vision-based reinforcement learning system for Clash Royale that learns directly from live gameplay without access to a simulator or structured game state. The system combines a YOLO-based perception module with a PPO agent trained on real-time gameplay interactions.

This repository contains the full pipeline for live detection, data collection, training, and evaluation.

## Repository Structure
```text
src/cr_rl_live/   # core RL + runtime logic
scripts/          # helper utilities
data/             # dataset instructions
models/           # model requirements
assets/           # example visuals
```

## Setup
```bash
pip install -r requirements.txt
```

Place model files in `models/`:
- `models/yolo.pt` for the detector
- `models/ppo.pt` for an optional trained PPO checkpoint

## Dataset
This project uses a Clash Royale dataset derived from the KataCR framework.

See:
- [data/README.md](data/README.md)

## Models
The system requires:
- YOLO detector weights
- PPO policy checkpoint (optional)

See:
- [models/README.md](models/README.md)

## Running The System
1. Run live detection (YOLO only)
```bash
python3 live_detection.py --model models/yolo.pt --source "window:MuMu"
```

2. Run random baseline agent
```bash
python3 live_random.py --model models/yolo.pt --source "window:MuMu" --episode-dir episodes_random
```

3. Run trained PPO agent
```bash
python3 live_infer.py --model models/yolo.pt --checkpoint models/ppo.pt --source "window:MuMu" --episode-dir episodes
```

4. Train PPO from collected episodes
```bash
python3 train_ppo.py --episode-dir episodes --checkpoint models/ppo.pt
```

5. Validate rollout data
```bash
python3 episode_sanity_check.py --episode-dir episodes --last 5
```

## Runtime Notes
- `window:MuMu` capture is intended for Windows with a visible MuMu emulator window
- If you are not using window capture, `--source` can also be a camera index such as `2`
- `live_infer.py` and `live_random.py` use the same 13-action live control interface
- `live_random.py` is the no-learning random baseline
- `live_infer.py` requires a PPO checkpoint if you want a trained policy instead of a freshly initialized one

## Reproducing Results
- Collect gameplay data by running live episodes
- Train the PPO model using collected transitions
- Evaluate the trained agent against human opponents

Note:
- Training is performed using live gameplay with manual interaction
- Results may vary due to stochastic learning and real-time constraints

## Paper
See the full report for details on system design, experiments, and results:  
[INSERT PAPER LINK]

## License
See [LICENSE](LICENSE).
