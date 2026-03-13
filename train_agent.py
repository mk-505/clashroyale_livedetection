#!/usr/bin/env python3
"""
Clash Royale RL Agent Training Script

This script demonstrates how to train an RL agent using the ClashRoyaleEnv.

Uses PPO algorithm from stable-baselines3 for training.
"""

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
from clash_royale_env import ClashRoyaleEnv


class TrainingCallback(BaseCallback):
    """Custom callback for logging training progress"""

    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Save model checkpoint
            self.model.save(os.path.join(self.save_path, f"clash_royale_ppo_{self.n_calls}"))
            if self.verbose > 0:
                print(f"Saved model at step {self.n_calls}")
        return True


def make_env(model_path: str, device: int = 2):
    """Create environment factory function"""
    def _init():
        env = ClashRoyaleEnv(model_path=model_path, device=device)
        return env
    return _init


def train_agent(model_path: str, total_timesteps: int = 100000, device: int = 2):
    """Train the RL agent"""

    # Create vectorized environment
    env = DummyVecEnv([make_env(model_path, device)])

    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./clash_royale_tensorboard/"
    )

    # Create callback for saving checkpoints
    callback = TrainingCallback(
        check_freq=10000,
        save_path="./models/",
        verbose=1
    )

    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)

    # Train the agent
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    # Save final model
    model.save("./models/clash_royale_ppo_final")
    print("Training completed! Model saved to ./models/clash_royale_ppo_final")

    return model


def evaluate_agent(model_path: str, env_model_path: str, num_episodes: int = 10, device: int = 2):
    """Evaluate trained agent"""

    # Load model
    model = PPO.load(model_path)

    # Create environment
    env = ClashRoyaleEnv(model_path=env_model_path, device=device)

    total_rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print("
Evaluation Results:")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")

    env.close()
    return avg_reward, std_reward


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or evaluate Clash Royale RL agent")
    parser.add_argument('--mode', choices=['train', 'eval'], default='train',
                       help='Mode: train or evaluate')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to YOLO detection model (.pt file)')
    parser.add_argument('--rl-model', type=str, default='./models/clash_royale_ppo_final.zip',
                       help='Path to RL model (for eval) or save path (for train)')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--device', type=int, default=2,
                       help='Video device ID')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')

    args = parser.parse_args()

    if args.mode == 'train':
        train_agent(args.model, args.timesteps, args.device)
    elif args.mode == 'eval':
        evaluate_agent(args.rl_model, args.model, args.episodes, args.device)