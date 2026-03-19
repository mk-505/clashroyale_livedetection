#!/usr/bin/env python3
import argparse

import torch

from cr_rl_live.ppo import PPOConfig, load_or_create_model, save_model, train_ppo_epoch
from cr_rl_live.storage import iter_episode_files, load_transitions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train minimal PPO from completed live-match transitions")
    parser.add_argument("--episode-dir", default="episodes")
    parser.add_argument("--checkpoint", default="checkpoints/policy_latest.pt")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    args = parser.parse_args()

    files = list(iter_episode_files(args.episode_dir))
    if not files:
        print(f"No episode files found in {args.episode_dir}")
        return

    transitions = load_transitions(args.episode_dir)
    if len(transitions) < 32:
        print(f"Not enough transitions for PPO update: {len(transitions)} (need >= 32)")
        return

    cfg = PPOConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model, optimizer = load_or_create_model(args.checkpoint, cfg)
    model.train()

    stats = train_ppo_epoch(model, optimizer, transitions, cfg)
    save_model(args.checkpoint, model, optimizer, step=len(transitions))

    print(
        f"Trained PPO on {int(stats['transitions'])} transitions "
        f"| loss_pi={stats['loss_policy']:.4f} "
        f"| loss_v={stats['loss_value']:.4f} "
        f"| entropy={stats['entropy']:.4f}"
    )
    print(f"Saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()

