#!/usr/bin/env python3
"""
Clash Royale Reinforcement Learning Environment

This module defines the state and action spaces for a Clash Royale RL agent.
The environment interfaces with the game through screen capture and detection.

State Space:
- Elixir count (0-10)
- Player tower healths (3 towers, 0-100 each)
- Opponent tower healths (3 towers, 0-100 each)
- Cards in hand (4 cards, each represented by type ID)
- Detected units on field (positions and types)
- Game phase (early/mid/late game)

Action Space:
- Play card 1-4 at position (x,y)
- Do nothing (pass turn)

Prerequisites:
- gymnasium
- torch
- numpy
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import cv2
from ultralytics import YOLO
import time


class ClashRoyaleEnv(gym.Env):
    """
    Custom Gym environment for Clash Royale game playing.
    """

    def __init__(self,
                 model_path: str = 'path/to/model.pt',
                 device: int = 2,
                 detection_conf: float = 0.5):
        super().__init__()

        # Load detection model
        self.model = YOLO(model_path)
        self.device = device
        self.detection_conf = detection_conf

        # Initialize video capture
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video device {device}")

        # Define card types (example - would need to map to actual game cards)
        self.card_types = {
            0: 'empty',
            1: 'knight',
            2: 'archers',
            3: 'giant',
            4: 'minions',
            5: 'balloon',
            6: 'wizard',
            7: 'hog_rider',
            8: 'valkyrie',
            9: 'golem',
            10: 'lava_hound',
            # Add more cards as needed
        }

        # State space dimensions
        self.max_elixir = 10
        self.max_tower_health = 100
        self.num_towers = 3
        self.hand_size = 4
        self.max_units_on_field = 20  # Maximum units to track
        self.field_width = 16  # Game field grid width
        self.field_height = 9   # Game field grid height

        # Define observation space
        self.observation_space = spaces.Dict({
            'elixir': spaces.Discrete(self.max_elixir + 1),  # 0-10
            'player_towers': spaces.MultiDiscrete([self.max_tower_health + 1] * self.num_towers),
            'opponent_towers': spaces.MultiDiscrete([self.max_tower_health + 1] * self.num_towers),
            'hand_cards': spaces.MultiDiscrete([len(self.card_types)] * self.hand_size),
            'field_units': spaces.MultiDiscrete([len(self.card_types), self.field_width, self.field_height] * self.max_units_on_field),
            'game_phase': spaces.Discrete(3),  # 0: early, 1: mid, 2: late
        })

        # Define action space
        # Actions: (card_index, x_position, y_position)
        # card_index: 0-3 for hand cards, 4 for pass
        # positions: discretized field positions
        self.action_space = spaces.MultiDiscrete([
            self.hand_size + 1,  # card selection (0-3) + pass (4)
            self.field_width,     # x position
            self.field_height     # y position
        ])

        # Initialize state
        self.state = self._get_initial_state()

    def _get_initial_state(self) -> Dict:
        """Get initial game state"""
        return {
            'elixir': 0,
            'player_towers': [100, 100, 100],  # Full health
            'opponent_towers': [100, 100, 100],
            'hand_cards': [0, 0, 0, 0],  # Empty hand
            'field_units': [],  # No units initially
            'game_phase': 0,  # Early game
        }

    def _capture_screen(self) -> np.ndarray:
        """Capture current game screen"""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture screen")
        return frame

    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Run object detection on frame"""
        results = self.model(frame, conf=self.detection_conf, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': cls,
                    'label': self.card_types.get(cls, 'unknown')
                })

        return detections

    def _extract_game_state(self, detections: List[Dict]) -> Dict:
        """Extract game state from detections"""
        # This is a simplified extraction - would need actual game UI parsing
        state = self.state.copy()

        # Extract elixir (placeholder - would need OCR or specific detection)
        state['elixir'] = min(self.max_elixir, state['elixir'] + 1)  # Simple increment

        # Extract tower healths (placeholder)
        # In reality, would detect health bars or use OCR

        # Extract hand cards (placeholder)
        # Would detect card icons in hand area

        # Extract field units
        field_units = []
        for det in detections:
            if det['label'] != 'empty':  # Assuming 'empty' is background
                # Convert bbox to grid position
                x_center = (det['bbox'][0] + det['bbox'][2]) / 2
                y_center = (det['bbox'][1] + det['bbox'][3]) / 2

                # Normalize to field coordinates (simplified)
                grid_x = int((x_center / 1920) * self.field_width)  # Assuming 1920p screen
                grid_y = int((y_center / 1080) * self.field_height)

                field_units.append({
                    'type': det['class'],
                    'x': grid_x,
                    'y': grid_y
                })

        state['field_units'] = field_units[:self.max_units_on_field]

        # Determine game phase based on time or elixir
        if state['elixir'] < 4:
            state['game_phase'] = 0  # Early
        elif state['elixir'] < 8:
            state['game_phase'] = 1  # Mid
        else:
            state['game_phase'] = 2  # Late

        return state

    def step(self, action: Tuple[int, int, int]) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        card_idx, x_pos, y_pos = action

        # Capture screen and detect
        frame = self._capture_screen()
        detections = self._detect_objects(frame)
        self.state = self._extract_game_state(detections)

        reward = 0
        terminated = False
        truncated = False

        # Execute action
        if card_idx < self.hand_size:  # Play a card
            card_type = self.state['hand_cards'][card_idx]
            if card_type != 0:  # Valid card
                # Simulate playing card (placeholder)
                print(f"Playing card {card_type} at position ({x_pos}, {y_pos})")
                reward += 0.1  # Small reward for playing a card

                # Remove card from hand (simplified)
                self.state['hand_cards'][card_idx] = 0
            else:
                reward -= 0.1  # Penalty for invalid action
        else:  # Pass turn
            print("Passing turn")
            reward -= 0.05  # Small penalty for passing

        # Check win/lose conditions (simplified)
        if all(h <= 0 for h in self.state['opponent_towers']):
            reward += 10  # Win reward
            terminated = True
        elif all(h <= 0 for h in self.state['player_towers']):
            reward -= 10  # Lose penalty
            terminated = True

        # Add elixir
        self.state['elixir'] = min(self.max_elixir, self.state['elixir'] + 1)

        info = {'detections': detections}

        return self.state, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        self.state = self._get_initial_state()
        info = {}
        return self.state, info

    def render(self, mode: str = 'human'):
        """Render the environment"""
        frame = self._capture_screen()
        detections = self._detect_objects(frame)

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, det['label'], (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Clash Royale RL Environment', frame)
        cv2.waitKey(1)

    def close(self):
        """Close the environment"""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    # Create environment
    env = ClashRoyaleEnv(model_path='path/to/your/model.pt')

    # Reset environment
    obs, info = env.reset()

    print("Initial state:")
    print(f"Elixir: {obs['elixir']}")
    print(f"Player towers: {obs['player_towers']}")
    print(f"Opponent towers: {obs['opponent_towers']}")
    print(f"Hand cards: {obs['hand_cards']}")
    print(f"Game phase: {obs['game_phase']}")

    # Example step
    action = (0, 8, 4)  # Play first card at center
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nAfter action - Reward: {reward}, Terminated: {terminated}")
    print(f"New elixir: {obs['elixir']}")

    env.close()