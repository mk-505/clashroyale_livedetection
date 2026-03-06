# Clash Royale Live Detection Dataset for RL Agent Training

> This dataset does not contain any video data. Video data is sourced from YouTube or self-recorded videos.

This repository is based on the original Clash Royale Image Dataset created by [wty-yy](https://github.com/wty-yy/KataCR). We extend their work by reconfiguring it for live object detection on the MuMu Android emulator and integrating it into a reinforcement learning (RL) environment.

## Credits
- **Original Dataset and KataCR Framework**: Created by [wty-yy](https://github.com/wty-yy). All rights to the original dataset, images, and core processing scripts belong to them.
- **Our Modifications**: Live detection script for MuMu emulator integration and RL agent development.

## Overview
### Sliced Dataset
This dataset collects a total of 154 categories (see [`label_list.py`](https://github.com/wty-yy/KataCR/blob/master/katacr/constants/label_list.py) for all sliced names), totaling 4,654 slices, used to create generative datasets.
![Slice size distribution](./asserts/segment_size.png)

<div align="center">
    <img src="images/segment/archer/archer_1_0000007.png" width="10%"/>
    <img src="images/segment/archer/archer_1_0000009.png" width="10%"/>
    <img src="images/segment/archer/archer_1_0000010.png" width="10%"/>
    <img src="images/segment/archer/archer_1_0000028.png" width="10%"/>
    <img src="images/segment/archer/archer_1_0000057.png" width="10%"/>
    <img src="images/segment/archer/archer_1_0000060.png" width="10%"/>
    <img src="images/segment/archer/archer_1_0000094.png" width="10%"/>
    <img src="images/segment/archer/archer_1_0000168.png" width="10%"/>
    <img src="images/segment/archer/archer_1_0000176.png" width="10%"/>
</div>
<div align="center">
    <img src="images/segment/hog-rider/hog-rider_0_0000004.png" width="10%"/>
    <img src="images/segment/hog-rider/hog-rider_0_0000027.png" width="10%"/>
    <img src="images/segment/hog-rider/hog-rider_0_0000053.png" width="10%"/>
    <img src="images/segment/hog-rider/hog-rider_0_0000059.png" width="10%"/>
    <img src="images/segment/hog-rider/hog-rider_0_0000062.png" width="10%"/>
    <img src="images/segment/hog-rider/hog-rider_0_0000054.png" width="10%"/>
    <img src="images/segment/hog-rider/hog-rider_1_0000493.png" width="10%"/>
    <img src="images/segment/hog-rider/hog-rider_1_0000557.png" width="10%"/>
    <img src="images/segment/hog-rider/hog-rider_1_0000496.png" width="10%"/>
</div>
<div align="center">
    <img src="images/segment/queen-tower/queen-tower_0_0000000.png" width="10%"/>
    <img src="images/segment/queen-tower/queen-tower_0_0000007.png" width="10%"/>
    <img src="images/segment/queen-tower/queen-tower_0_0006331.png" width="10%"/>
    <img src="images/segment/queen-tower/queen-tower_0_0006335.png" width="10%"/>
    <img src="images/segment/queen-tower/queen-tower_0_0006380.png" width="10%"/>
    <img src="images/segment/queen-tower/queen-tower_1_attack_929.png" width="10%"/>
    <img src="images/segment/queen-tower/queen-tower_1_006330.png" width="10%"/>
    <img src="images/segment/queen-tower/queen-tower_1_0009264.png" width="10%"/>
    <img src="images/segment/queen-tower/queen-tower_1_0007320.png" width="10%"/>
</div>

### Generative Dataset
Generated using [`generator.py`](https://github.com/wty-yy/KataCR/blob/master/katacr/build_dataset/generator.py), for training with [YOLOv8](https://github.com/ultralytics/ultralytics).
<div style="display: flex; flex-wrap: nowrap; justify-content: space-between;">
    <img src="asserts/generation1.jpg" alt="Generation 1" width="49%" />
    <img src="asserts/generation2.jpg" alt="Generation 2" width="49%" />
</div>

### Card Dataset
This dataset only collects images of the 2.6 Hog Cycle deck.
<div align="center">
    <img src="images/card_classification/cannon/00030_2.jpg" width="9%"/>
    <img src="images/card_classification/fireball/00285_1.jpg" width="9%"/>
    <img src="images/card_classification/hog-rider/00045_3.jpg" width="9%"/>
    <img src="images/card_classification/ice-golem/00450_1.jpg" width="9%"/>
    <img src="images/card_classification/ice-spirit-evolution/04425_4.jpg" width="9%"/>
    <img src="images/card_classification/ice-spirit/00105_4.jpg" width="9%"/>
    <img src="images/card_classification/musketeer/00480_3.jpg" width="9%"/>
    <img src="images/card_classification/skeletons/00780_1.jpg" width="9%"/>
    <img src="images/card_classification/skeletons-evolution/04875_2.jpg" width="9%"/>
    <img src="images/card_classification/the-log/00210_3.jpg" width="9%"/>
</div>

### Elixir Dataset
This dataset only collects images of 5 different elixir numbers, used for further classification of target recognition results.
<div align="center">
    <img src="images/elixir_classification/-1/101.jpg" width="19%"/>
    <img src="images/elixir_classification/-2/11.jpg" width="19%"/>
    <img src="images/elixir_classification/-3/302.jpg" width="19%"/>
    <img src="images/elixir_classification/-4/10.jpg" width="19%"/>
    <img src="images/elixir_classification/1/105.jpg" width="19%"/>
</div>

## Our Modifications and Usage
We have reconfigured this dataset for live object detection on the MuMu Android emulator. The dataset is used to train YOLOv8 models for real-time detection of Clash Royale game elements.

### Live Detection on MuMu
- **Script**: `live_detection.py` - Performs real-time object detection on MuMu screen streams.
- **Setup**:
  1. Install dependencies: `pip install -r requirements.txt`
  2. Install scrcpy: `brew install scrcpy`
  3. Start MuMu and stream screen: `scrcpy --v4l2-sink=/dev/video2 --no-video-playback`
  4. Run detection: `python live_detection.py --model path/to/model.pt`

### Training YOLO Models
1. Use the manually labeled data in `images/part2/` for direct YOLO training.
2. Generate synthetic data using KataCR's `generator.py` for augmentation.
3. Train with YOLOv8: `model.train(data='images/part2/ClashRoyale_detection.yaml')`

## Future Plans
Our goal is to integrate this detection system into a reinforcement learning environment for creating an AI agent that can play Clash Royale autonomously. This will involve:
- Combining object detection with game state understanding
- Implementing RL algorithms (e.g., PPO, DQN) for decision-making
- Training the agent to make strategic moves based on detected game elements

## Dataset Structure
1. **Manually Labeled Images** (`images/part2/`): Real gameplay frames with YOLO annotations.
2. **Sliced Elements** (`images/segment/`): Individual game object images for data generation.
3. **Card Classification** (`images/card_classification/`): Card recognition images.
4. **Elixir Classification** (`images/elixir_classification/`): Elixir value recognition.

## License
See [LICENSE](LICENSE) for details. Original dataset credits to wty-yy.
