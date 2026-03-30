# Models

## Overview

This directory contains the models used in the ClashBotPro system.

## Included Models

### 1. YOLO Base Model
- File: `yolov8.pt`
- Description: Pretrained YOLOv8 model used as the base for object detection.

### 2. Fine-Tuned Detection Model
- File: `best.pt`
- Description: YOLO model fine-tuned on Clash Royale gameplay data for live detection.

### 3. PPO Policy Model
- File: `ppo.pt`
- Description: Trained PPO policy used for decision-making during gameplay.

## Usage

- `best.pt` is used for live perception (object detection)
- `ppo.pt` is used for action selection
- `yolov8.pt` is used as the base model for training

## Notes

- Models are trained on a limited dataset collected through live gameplay
- Performance may vary due to stochastic training and real-time constraints