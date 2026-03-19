import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def save_episode(episode_dir: str, transitions: List[Dict[str, object]], meta: Dict[str, object]) -> Path:
    out_dir = Path(episode_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"episode_{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", **meta}) + "\n")
        for t in transitions:
            row = dict(t)
            row["state"] = np.asarray(row["state"], dtype=np.float32).tolist()
            row["next_state"] = np.asarray(row["next_state"], dtype=np.float32).tolist()
            f.write(json.dumps(row) + "\n")
    return path


def load_transitions(episode_dir: str) -> List[Dict[str, object]]:
    all_rows: List[Dict[str, object]] = []
    for path in sorted(Path(episode_dir).glob("episode_*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("type") == "meta":
                    continue
                row["state"] = np.asarray(row["state"], dtype=np.float32)
                row["next_state"] = np.asarray(row["next_state"], dtype=np.float32)
                row["action"] = int(row["action"])
                row["reward"] = float(row["reward"])
                row["done"] = bool(row["done"])
                row["log_prob"] = float(row.get("log_prob", 0.0))
                all_rows.append(row)
    return all_rows


def iter_episode_files(episode_dir: str) -> Iterable[Path]:
    yield from sorted(Path(episode_dir).glob("episode_*.jsonl"))

