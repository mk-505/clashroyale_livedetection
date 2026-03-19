import ctypes
import ctypes.wintypes
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import mss
except ImportError:
    mss = None


STATE_SIZE = 13
ACTION_SIZE = 13


@dataclass
class RuntimeConfig:
    model_path: str
    source: str = "window:MuMu"
    conf: float = 0.5
    window_refresh: int = 60
    action_interval_sec: float = 0.75
    no_op_bias: float = 0.2
    max_match_seconds: int = 220
    # 3 fixed anchors (x, y) in emulator window coordinates.
    anchors: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (210, 520),
        (280, 430),
        (360, 520),
    )
    # Optional stop key for immediate manual override.
    emergency_stop_key: str = "q"
    episode_dir: str = "episodes"
    checkpoint_path: str = "checkpoints/policy_latest.pt"


def parse_anchor_string(value: str) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    parts = [x.strip() for x in value.split(";") if x.strip()]
    if len(parts) != 3:
        raise ValueError("anchors must be exactly 3 points, format: x1,y1;x2,y2;x3,y3")
    pts = []
    for p in parts:
        x_str, y_str = [s.strip() for s in p.split(",")]
        pts.append((int(x_str), int(y_str)))
    return tuple(pts)  # type: ignore[return-value]


def find_window_rect_windows(title_substring: str) -> Optional[Dict[str, int]]:
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def enum_handler(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value
        if title_substring.lower() in title.lower():
            rect = ctypes.wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            if width > 0 and height > 0:
                found.append({
                    "left": rect.left,
                    "top": rect.top,
                    "width": width,
                    "height": height,
                    "title": title,
                })
                return False
        return True

    user32.EnumWindows(enum_handler, 0)
    return found[0] if found else None


class LiveCapture:
    def __init__(self, source: str, window_refresh: int = 60) -> None:
        self.source = source
        self.window_refresh = max(1, window_refresh)
        self.frame_count = 0
        self.window_mode = source.lower().startswith("window:")
        self.window_query = source.split(":", 1)[1].strip() if self.window_mode else None
        self.cap = None
        self.rect = None
        self.sct = None
        self.window_title = None

        if self.window_mode:
            if platform.system() != "Windows":
                raise RuntimeError("window:<title> source is supported on Windows only.")
            if mss is None:
                raise RuntimeError("mss is required for window capture. Install with: pip install mss")
            rect_info = find_window_rect_windows(self.window_query)  # type: ignore[arg-type]
            if rect_info is None:
                raise RuntimeError(f'No visible window found matching "{self.window_query}"')
            self.rect = {k: rect_info[k] for k in ("left", "top", "width", "height")}
            self.window_title = str(rect_info["title"])
            self.sct = mss.mss()
        else:
            src = int(source) if source.isdigit() else source
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open source: {source}")

    def read(self) -> Tuple[np.ndarray, Dict[str, int]]:
        self.frame_count += 1
        if self.window_mode:
            if self.frame_count % self.window_refresh == 0:
                rect_info = find_window_rect_windows(self.window_query)  # type: ignore[arg-type]
                if rect_info is None:
                    raise RuntimeError(f'Window not found: "{self.window_query}"')
                self.rect = {k: rect_info[k] for k in ("left", "top", "width", "height")}
                self.window_title = str(rect_info["title"])
            sct_img = self.sct.grab(self.rect)  # type: ignore[arg-type]
            frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            return frame, self.rect  # type: ignore[return-value]

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from source.")
        h, w = frame.shape[:2]
        return frame, {"left": 0, "top": 0, "width": w, "height": h}

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if self.sct is not None:
            self.sct.close()

    def describe_source(self) -> str:
        if self.window_mode:
            t = self.window_title or "(unknown)"
            return f'window query="{self.window_query}" resolved_title="{t}" rect={self.rect}'
        return f"video source={self.source}"


def extract_detections(result) -> List[Dict[str, float]]:
    boxes = result.boxes
    names = result.names
    out = []
    if boxes is None:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        c = int(cls[i])
        out.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": (x1 + x2) * 0.5,
            "cy": (y1 + y2) * 0.5,
            "conf": float(conf[i]),
            "cls": c,
            "name": str(names[c]),
        })
    return out


def _extract_number_by_patterns(name: str, patterns: List[str]) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def _team_from_name(name: str) -> Optional[int]:
    m = re.search(r"[_\-](0|1)(?:$|[_\-])", name)
    if not m:
        return None
    return int(m.group(1))


class StateBuilder:
    def __init__(self, class_names: Dict[int, str]) -> None:
        self.class_names = class_names
        self.card_to_id = {n: i + 1 for i, n in enumerate(sorted(set(class_names.values())))}
        self.prev_towers = {
            "own_left": 1.0,
            "own_right": 1.0,
            "enemy_left": 1.0,
            "enemy_right": 1.0,
        }
        self.prev_elixir = 5.0
        self.prev_time = 180.0
        self.prev_hand = [0, 0, 0, 0]
        self.max_bar_width = {
            "own_left": 1.0,
            "own_right": 1.0,
            "enemy_left": 1.0,
            "enemy_right": 1.0,
        }
        self.ui_names = {
            "king-tower",
            "queen-tower",
            "cannoneer-tower",
            "dagger-duchess-tower",
            "dagger-duchess-tower-bar",
            "tower-bar",
            "king-tower-bar",
            "bar",
            "bar-level",
            "clock",
            "emote",
            "text",
            "elixir",
            "selected",
            "dirt",
            "evolution-symbol",
            "ice-spirit-evolution-symbol",
            "pad_belong",
        }

    def build(
        self,
        detections: List[Dict[str, float]],
        frame_shape: Tuple[int, int],
        elapsed_seconds: float,
        last_action: int,
    ) -> Dict[str, object]:
        h, w = frame_shape
        cx_mid = w * 0.5
        own_left_hp = None
        own_right_hp = None
        enemy_left_hp = None
        enemy_right_hp = None
        elixir = None
        timer = None
        hand: List[Tuple[float, int]] = []
        pressure_left = 0.0
        pressure_right = 0.0

        for det in detections:
            name = str(det["name"])
            conf = float(det["conf"])
            cx = float(det["cx"])
            cy = float(det["cy"])
            bw = float(det["x2"] - det["x1"])
            team = _team_from_name(name)

            val_elixir = _extract_number_by_patterns(
                name,
                [r"elixir[_\-: ]?(\d+(?:\.\d+)?)", r"^e(\d+)$"],
            )
            if val_elixir is not None:
                elixir = val_elixir

            val_time = _extract_number_by_patterns(
                name,
                [r"time[_\-: ]?(\d+(?:\.\d+)?)", r"clock[_\-: ]?(\d+(?:\.\d+)?)"],
            )
            if val_time is not None:
                timer = val_time

            # HP proxy from detected tower-bar widths (works with this dataset labels).
            if name in {"tower-bar", "dagger-duchess-tower-bar", "king-tower-bar"} and bw > 2.0:
                if cy < h * 0.5:
                    if cx < cx_mid:
                        slot = "enemy_left"
                    else:
                        slot = "enemy_right"
                else:
                    if cx < cx_mid:
                        slot = "own_left"
                    else:
                        slot = "own_right"
                self.max_bar_width[slot] = max(self.max_bar_width[slot], bw)
                hp_norm = max(0.0, min(1.0, bw / max(1.0, self.max_bar_width[slot])))
                if slot == "own_left":
                    own_left_hp = hp_norm
                elif slot == "own_right":
                    own_right_hp = hp_norm
                elif slot == "enemy_left":
                    enemy_left_hp = hp_norm
                elif slot == "enemy_right":
                    enemy_right_hp = hp_norm

            # Pressure proxy: non-UI troops on our half of arena.
            is_pad = name.startswith("pad_")
            is_ui = (name in self.ui_names) or is_pad
            if not is_ui and h * 0.35 < cy < h * 0.85:
                if cx < cx_mid:
                    pressure_left += conf
                else:
                    pressure_right += conf

            # Approximate hand from detections near bottom bar.
            if cy > h * 0.75 and not is_ui:
                card_id = self.card_to_id.get(name, 0)
                hand.append((conf, card_id))

        if own_left_hp is None:
            own_left_hp = self.prev_towers["own_left"]
        if own_right_hp is None:
            own_right_hp = self.prev_towers["own_right"]
        if enemy_left_hp is None:
            enemy_left_hp = self.prev_towers["enemy_left"]
        if enemy_right_hp is None:
            enemy_right_hp = self.prev_towers["enemy_right"]
        if elixir is None:
            # Fallback for this dataset: estimate from count of "elixir" detections near bottom bar.
            elixir_count = 0
            for det in detections:
                if str(det["name"]) == "elixir" and float(det["cy"]) > h * 0.70:
                    elixir_count += 1
            if elixir_count > 0:
                elixir = float(min(10, elixir_count))
            else:
                elixir = self.prev_elixir
        if timer is None:
            timer = max(0.0, 180.0 - elapsed_seconds)

        hand.sort(key=lambda x: x[0], reverse=True)
        hand_ids = [c for _, c in hand[:4]]
        while len(hand_ids) < 4:
            hand_ids.append(0)
        if hand_ids == [0, 0, 0, 0]:
            hand_ids = self.prev_hand.copy()

        self.prev_elixir = float(max(0.0, min(10.0, elixir)))
        self.prev_time = float(max(0.0, timer))
        self.prev_hand = hand_ids
        self.prev_towers = {
            "own_left": float(own_left_hp),
            "own_right": float(own_right_hp),
            "enemy_left": float(enemy_left_hp),
            "enemy_right": float(enemy_right_hp),
        }

        state_vec = np.array(
            [
                self.prev_elixir / 10.0,
                self.prev_towers["own_left"],
                self.prev_towers["own_right"],
                self.prev_towers["enemy_left"],
                self.prev_towers["enemy_right"],
                self.prev_time / 180.0,
                hand_ids[0] / 256.0,
                hand_ids[1] / 256.0,
                hand_ids[2] / 256.0,
                hand_ids[3] / 256.0,
                min(1.0, pressure_left / 10.0),
                min(1.0, pressure_right / 10.0),
                last_action / (ACTION_SIZE - 1),
            ],
            dtype=np.float32,
        )
        return {
            "vector": state_vec,
            "features": {
                "elixir": self.prev_elixir,
                "own_left_hp": self.prev_towers["own_left"],
                "own_right_hp": self.prev_towers["own_right"],
                "enemy_left_hp": self.prev_towers["enemy_left"],
                "enemy_right_hp": self.prev_towers["enemy_right"],
                "time_remaining": self.prev_time,
                "hand_ids": hand_ids,
                "pressure_left": min(1.0, pressure_left / 10.0),
                "pressure_right": min(1.0, pressure_right / 10.0),
                "last_action": int(last_action),
            },
        }


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG)),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG)),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.wintypes.DWORD), ("union", INPUT_UNION)]


class ActionExecutor:
    INPUT_KEYBOARD = 1
    INPUT_MOUSE = 0
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_SCANCODE = 0x0008
    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_ABSOLUTE = 0x8000
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004

    def __init__(self, anchors: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> None:
        if platform.system() != "Windows":
            raise RuntimeError("ActionExecutor currently supports Windows only.")
        self.anchors = anchors
        self.user32 = ctypes.windll.user32

    def _send_input(self, inp):
        self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def _focus_window_from_rect(self, window_rect: Dict[str, int]) -> None:
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.wintypes.LONG), ("y", ctypes.wintypes.LONG)]

        cx = int(window_rect["left"] + window_rect["width"] * 0.5)
        cy = int(window_rect["top"] + max(10, min(40, window_rect["height"] * 0.1)))
        hwnd = self.user32.WindowFromPoint(POINT(cx, cy))
        if hwnd:
            self.user32.SetForegroundWindow(hwnd)
            time.sleep(0.01)

    def _press_vk(self, vk: int):
        scan = self.user32.MapVirtualKeyW(vk, 0)

        down = INPUT(type=self.INPUT_KEYBOARD, union=INPUT_UNION(ki=KEYBDINPUT(vk, scan, 0, 0, None)))
        up = INPUT(type=self.INPUT_KEYBOARD, union=INPUT_UNION(ki=KEYBDINPUT(vk, scan, self.KEYEVENTF_KEYUP, 0, None)))
        scan_down = INPUT(
            type=self.INPUT_KEYBOARD,
            union=INPUT_UNION(ki=KEYBDINPUT(0, scan, self.KEYEVENTF_SCANCODE, 0, None)),
        )
        scan_up = INPUT(
            type=self.INPUT_KEYBOARD,
            union=INPUT_UNION(ki=KEYBDINPUT(0, scan, self.KEYEVENTF_SCANCODE | self.KEYEVENTF_KEYUP, 0, None)),
        )
        self._send_input(down)
        self._send_input(scan_down)
        time.sleep(0.02)
        self._send_input(up)
        self._send_input(scan_up)

    def _click_screen(self, x: int, y: int):
        screen_w = self.user32.GetSystemMetrics(0)
        screen_h = self.user32.GetSystemMetrics(1)
        abs_x = int(x * 65535 / max(1, screen_w - 1))
        abs_y = int(y * 65535 / max(1, screen_h - 1))
        move = INPUT(
            type=self.INPUT_MOUSE,
            union=INPUT_UNION(
                mi=MOUSEINPUT(abs_x, abs_y, 0, self.MOUSEEVENTF_MOVE | self.MOUSEEVENTF_ABSOLUTE, 0, None)
            ),
        )
        down = INPUT(
            type=self.INPUT_MOUSE,
            union=INPUT_UNION(mi=MOUSEINPUT(0, 0, 0, self.MOUSEEVENTF_LEFTDOWN, 0, None)),
        )
        up = INPUT(
            type=self.INPUT_MOUSE,
            union=INPUT_UNION(mi=MOUSEINPUT(0, 0, 0, self.MOUSEEVENTF_LEFTUP, 0, None)),
        )
        self._send_input(move)
        time.sleep(0.02)
        self._send_input(down)
        time.sleep(0.02)
        self._send_input(up)

    def execute(self, action_idx: int, window_rect: Dict[str, int]) -> None:
        if action_idx <= 0:
            return
        card_slot = (action_idx - 1) // 3
        anchor_idx = (action_idx - 1) % 3

        if card_slot < 0 or card_slot > 3:
            return

        self._focus_window_from_rect(window_rect)
        key_code = ord("1") + card_slot
        self._press_vk(key_code)

        rel_x, rel_y = self.anchors[anchor_idx]
        x = int(window_rect["left"] + rel_x)
        y = int(window_rect["top"] + rel_y)
        self._click_screen(x, y)


def action_to_slot_anchor(action_idx: int) -> Tuple[int, int]:
    if action_idx == 0:
        return -1, -1
    return (action_idx - 1) // 3, (action_idx - 1) % 3


def reward_from_features(prev_features: Dict[str, object], next_features: Dict[str, object], done: bool, outcome: int) -> float:
    prev_enemy = float(prev_features["enemy_left_hp"]) + float(prev_features["enemy_right_hp"])
    next_enemy = float(next_features["enemy_left_hp"]) + float(next_features["enemy_right_hp"])
    prev_own = float(prev_features["own_left_hp"]) + float(prev_features["own_right_hp"])
    next_own = float(next_features["own_left_hp"]) + float(next_features["own_right_hp"])
    reward = (prev_enemy - next_enemy) - (prev_own - next_own)
    if done:
        if outcome > 0:
            reward += 1.0
        elif outcome < 0:
            reward -= 1.0
    return float(reward)


class MatchEndDetector:
    def __init__(self, max_seconds: int = 220) -> None:
        self.max_seconds = max_seconds
        self.start = time.time()
        self.enemy_hp_hist = []
        self.own_hp_hist = []

    def update(self, features: Dict[str, object]) -> Tuple[bool, int]:
        own_total = float(features["own_left_hp"]) + float(features["own_right_hp"])
        enemy_total = float(features["enemy_left_hp"]) + float(features["enemy_right_hp"])
        self.enemy_hp_hist.append(enemy_total)
        self.own_hp_hist.append(own_total)

        elapsed = time.time() - self.start
        if elapsed < self.max_seconds:
            return False, 0

        # Outcome from final visible hp when forced by timeout.
        if enemy_total < own_total:
            return True, 1
        if enemy_total > own_total:
            return True, -1
        return True, 0


def build_model(model_path: str) -> YOLO:
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)
    return YOLO(model_path)
