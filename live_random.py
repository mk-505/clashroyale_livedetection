#!/usr/bin/env python3
import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cr_rl_live.runtime import (
    MatchEndDetector,
    RuntimeConfig,
    StateBuilder,
    ActionExecutor,
    LiveCapture,
    build_model,
    extract_detections,
    parse_anchor_string,
    reward_from_features,
)
from cr_rl_live.storage import save_episode


def terminal_from_detections(detections, state_features, elapsed: float) -> tuple[bool, int]:
    for det in detections:
        name = str(det["name"]).lower()
        if "victory" in name or "win" in name:
            return True, 1
        if "defeat" in name or "lose" in name or "loss" in name:
            return True, -1
    if float(state_features["time_remaining"]) <= 0.0 and elapsed > 120:
        own = float(state_features["own_left_hp"]) + float(state_features["own_right_hp"])
        enemy = float(state_features["enemy_left_hp"]) + float(state_features["enemy_right_hp"])
        if enemy < own:
            return True, 1
        if enemy > own:
            return True, -1
        return True, 0
    return False, 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Clash Royale random-policy baseline + rollout collection")
    parser.add_argument("--model", required=True, help="YOLO model path")
    parser.add_argument("--source", default="window:MuMu", help='Capture source, e.g. "window:MuMu" or "2"')
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--episode-dir", default="episodes_random")
    parser.add_argument("--anchors", default="210,520;280,430;360,520")
    parser.add_argument("--action-interval", type=float, default=0.75)
    parser.add_argument("--max-seconds", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for action sampling")
    parser.add_argument("--show", action="store_true", help="Show live detection window")
    parser.add_argument("--debug-state", action="store_true", help="Overlay parsed state on video and print periodic debug")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cfg = RuntimeConfig(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        action_interval_sec=args.action_interval,
        max_match_seconds=args.max_seconds,
        anchors=parse_anchor_string(args.anchors),
        episode_dir=args.episode_dir,
    )

    yolo = build_model(cfg.model_path)
    capture = LiveCapture(cfg.source, window_refresh=cfg.window_refresh)
    executor = ActionExecutor(cfg.anchors)
    state_builder = StateBuilder(yolo.names)
    end_detector = MatchEndDetector(max_seconds=cfg.max_match_seconds)

    print(f"YOLO model: {Path(cfg.model_path).resolve()}")
    print(f"YOLO classes: {len(yolo.names)}")
    print(f"Capture: {capture.describe_source()}")
    print(f"Random baseline started. Seed={args.seed}. Manual flow: start match yourself, bot plays one match, then exits.")
    print("Policy: uniform random over actions 0..12, no learning, same environment and action space.")
    print("Hotkeys: q=abort early, e=force episode end as draw, v=force win, l=force loss, p=print source info, d=print top detections")

    transitions = []
    last_action = 0
    prev_state: Optional[np.ndarray] = None
    prev_features = None
    last_action_t = 0.0
    last_action_prev = 0
    started = time.time()
    forced_terminal = None
    last_reward = 0.0
    zero_det_streak = 0
    step_idx = 0

    try:
        while True:
            frame, rect = capture.read()
            results = yolo(frame, conf=cfg.conf, verbose=False)
            detections = extract_detections(results[0])
            step_idx += 1
            if len(detections) == 0:
                zero_det_streak += 1
            else:
                zero_det_streak = 0
            elapsed = time.time() - started

            state_data = state_builder.build(
                detections=detections,
                frame_shape=frame.shape[:2],
                elapsed_seconds=elapsed,
                last_action=last_action,
            )
            state = state_data["vector"]
            features = state_data["features"]

            now = time.time()
            if now - last_action_t >= cfg.action_interval_sec:
                action = rng.randint(0, 12)
                executor.execute(action, rect)
                last_action = action
                last_action_t = now

                if prev_state is not None and prev_features is not None:
                    done, outcome = terminal_from_detections(detections, features, elapsed)
                    if forced_terminal is not None:
                        done = True
                        outcome = forced_terminal
                    timeout_done, timeout_outcome = end_detector.update(features)
                    if timeout_done:
                        done = True
                        outcome = timeout_outcome
                    reward = reward_from_features(prev_features, features, done=done, outcome=outcome)
                    last_reward = reward
                    transitions.append(
                        {
                            "state": prev_state,
                            "action": int(last_action_prev),
                            "reward": float(reward),
                            "next_state": state,
                            "done": bool(done),
                            "log_prob": 0.0,
                        }
                    )
                    if done:
                        break

                prev_state = state.copy()
                prev_features = dict(features)
                last_action_prev = action

            if args.show:
                annotated = results[0].plot()
                if args.debug_state:
                    own_total = float(features["own_left_hp"]) + float(features["own_right_hp"])
                    enemy_total = float(features["enemy_left_hp"]) + float(features["enemy_right_hp"])
                    lines = [
                        f"det={len(detections)} zero_det_streak={zero_det_streak}",
                        f"elixir={float(features['elixir']):.2f} time={float(features['time_remaining']):.1f}s",
                        f"own_hp={own_total:.3f} enemy_hp={enemy_total:.3f} reward={last_reward:.4f}",
                        f"random_action={last_action} hand={features['hand_ids']}",
                    ]
                    y0 = 26
                    for i, line in enumerate(lines):
                        y = y0 + i * 24
                        cv2.putText(annotated, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 255, 20), 2)
                cv2.imshow("CR Live Random Baseline", annotated)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Abort requested.")
                break
            if key == ord("e"):
                forced_terminal = 0
            if key == ord("v"):
                forced_terminal = 1
            if key == ord("l"):
                forced_terminal = -1
            if key == ord("d"):
                c = Counter([str(x["name"]) for x in detections])
                top = c.most_common(10)
                print(f"[debug] detections={len(detections)} top={top}")
            if key == ord("p"):
                print(f"[debug] capture: {capture.describe_source()}")

            if args.debug_state and step_idx % 120 == 0:
                own_total = float(features["own_left_hp"]) + float(features["own_right_hp"])
                enemy_total = float(features["enemy_left_hp"]) + float(features["enemy_right_hp"])
                print(
                    "[random] "
                    f"det={len(detections)} elixir={float(features['elixir']):.2f} "
                    f"own_hp={own_total:.3f} enemy_hp={enemy_total:.3f} "
                    f"time={float(features['time_remaining']):.1f} action={last_action}"
                )
            if zero_det_streak == 180:
                print("[warn] 180 consecutive frames with zero detections. Check source/model/conf/window.")
                print(f"[warn] capture: {capture.describe_source()}")

    finally:
        capture.close()
        cv2.destroyAllWindows()

    if len(transitions) == 0:
        print("No transitions collected. Episode not saved.")
        return

    own = float(prev_features["own_left_hp"]) + float(prev_features["own_right_hp"]) if prev_features else 0.0
    enemy = float(prev_features["enemy_left_hp"]) + float(prev_features["enemy_right_hp"]) if prev_features else 0.0
    if forced_terminal is not None:
        outcome = int(forced_terminal)
    elif enemy < own:
        outcome = 1
    elif enemy > own:
        outcome = -1
    else:
        outcome = 0

    path = save_episode(
        cfg.episode_dir,
        transitions,
        meta={
            "type": "meta",
            "steps": len(transitions),
            "outcome": outcome,
            "source": cfg.source,
            "model": cfg.model_path,
            "policy": "random_uniform_13",
            "seed": args.seed,
        },
    )
    print(f"Saved episode: {path} ({len(transitions)} transitions)")
    print("Random baseline match complete.")


if __name__ == "__main__":
    main()
