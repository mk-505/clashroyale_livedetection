#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def load_episode(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty file: {path}")
    meta = json.loads(lines[0])
    rows = [json.loads(x) for x in lines[1:]]
    return meta, rows


def summarize(path: Path) -> dict:
    meta, rows = load_episode(path)
    if not rows:
        return {
            "file": path.name,
            "valid": False,
            "reason": "no transitions",
        }

    rewards = np.array([float(r["reward"]) for r in rows], dtype=np.float32)
    dones = np.array([bool(r["done"]) for r in rows], dtype=bool)
    states = np.array([r["state"] for r in rows], dtype=np.float32)

    nonzero_reward = int(np.count_nonzero(np.abs(rewards) > 1e-9))
    state_std = states.std(axis=0)

    # Key feature indices in 13-d state vector.
    idx = {
        "elixir": 0,
        "own_left_hp": 1,
        "own_right_hp": 2,
        "enemy_left_hp": 3,
        "enemy_right_hp": 4,
        "time_remaining": 5,
        "pressure_left": 10,
        "pressure_right": 11,
    }

    signal_changes = {
        k: float(state_std[v])
        for k, v in idx.items()
    }
    hp_signal = signal_changes["own_left_hp"] + signal_changes["own_right_hp"] + signal_changes["enemy_left_hp"] + signal_changes["enemy_right_hp"]
    pressure_signal = signal_changes["pressure_left"] + signal_changes["pressure_right"]
    has_terminal = bool(np.count_nonzero(dones) == 1 and dones[-1])

    issues: List[str] = []
    if nonzero_reward == 0:
        issues.append("all rewards are zero")
    if hp_signal < 1e-6:
        issues.append("tower hp features never changed")
    if pressure_signal < 1e-6:
        issues.append("pressure features never changed")
    if not has_terminal:
        issues.append("no valid terminal transition (expect exactly one, at end)")

    return {
        "file": path.name,
        "steps_meta": int(meta.get("steps", -1)),
        "steps_rows": len(rows),
        "outcome": int(meta.get("outcome", 0)),
        "model": str(meta.get("model", "")),
        "reward_sum": float(rewards.sum()),
        "reward_nonzero": nonzero_reward,
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "done_count": int(np.count_nonzero(dones)),
        "last_done": bool(dones[-1]),
        "signal": signal_changes,
        "valid": len(issues) == 0,
        "issues": issues,
    }


def print_summary(s: dict) -> None:
    print(f"File: {s['file']}")
    print(f"Model: {s['model']}")
    print(f"Steps: meta={s['steps_meta']} rows={s['steps_rows']}")
    print(f"Outcome: {s['outcome']} | Done count: {s['done_count']} | Last done: {s['last_done']}")
    print(
        "Reward: "
        f"sum={s['reward_sum']:.4f} "
        f"nonzero={s['reward_nonzero']} "
        f"min={s['reward_min']:.4f} "
        f"max={s['reward_max']:.4f}"
    )
    sig = s["signal"]
    print(
        "Signal std: "
        f"elixir={sig['elixir']:.6f} "
        f"ownL={sig['own_left_hp']:.6f} ownR={sig['own_right_hp']:.6f} "
        f"enemyL={sig['enemy_left_hp']:.6f} enemyR={sig['enemy_right_hp']:.6f} "
        f"pressureL={sig['pressure_left']:.6f} pressureR={sig['pressure_right']:.6f}"
    )
    if s["valid"]:
        print("Result: OK for training (basic sanity checks passed)")
    else:
        print("Result: NOT OK for training")
        for issue in s["issues"]:
            print(f"- {issue}")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick sanity check for saved live RL episodes")
    parser.add_argument("--episode", default=None, help="Path to one episode_*.jsonl file")
    parser.add_argument("--episode-dir", default="episodes", help="Directory containing episode_*.jsonl")
    parser.add_argument("--last", type=int, default=1, help="How many latest episodes to check from --episode-dir")
    args = parser.parse_args()

    if args.episode:
        paths = [Path(args.episode)]
    else:
        paths = sorted(Path(args.episode_dir).glob("episode_*.jsonl"))
        if not paths:
            print(f"No episodes found in: {args.episode_dir}")
            return
        paths = paths[-max(1, args.last):]

    for p in paths:
        s = summarize(p)
        print_summary(s)


if __name__ == "__main__":
    main()

