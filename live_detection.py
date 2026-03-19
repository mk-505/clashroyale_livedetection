#!/usr/bin/env python3
"""
Live Clash Royale Object Detection on MuMu Player

This script performs real-time object detection on a MuMu emulator screen stream.
It uses YOLOv8 for inference and displays results with bounding boxes.

Prerequisites:
1. Install dependencies: pip install ultralytics opencv-python
2. Set up scrcpy: brew install scrcpy (on macOS)
3. Start screen streaming: scrcpy --v4l2-sink=/dev/video2 --no-video-playback
4. Train or download a YOLO model compatible with the dataset

Usage:
python live_detection.py --model path/to/model.pt --source 2

Press 'q' to quit.
"""

import argparse
import cv2
from ultralytics import YOLO
import time
import platform
import ctypes
import ctypes.wintypes
import numpy as np

try:
    import mss
except ImportError:
    mss = None


def find_window_rect_windows(title_substring):
    """Find first visible window whose title contains title_substring."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Live Clash Royale Detection on MuMu")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained YOLO model (.pt file)')
    parser.add_argument(
        '--source',
        type=str,
        default='2',
        help='Capture source. Integer camera index or stream path/URL (default: 2)'
    )
    parser.add_argument('--device', type=int, default=None,
                        help='Backward-compatible alias for --source')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--show-fps', action='store_true',
                        help='Display FPS counter')
    parser.add_argument(
        '--window-refresh',
        type=int,
        default=60,
        help='Refresh window bounds every N frames when using window capture (default: 60)'
    )

    args = parser.parse_args()

    if args.device is not None:
        args.source = str(args.device)

    source = int(args.source) if args.source.isdigit() else args.source
    window_mode = isinstance(source, str) and source.lower().startswith("window:")
    window_query = source.split(":", 1)[1].strip() if window_mode else None

    # Load YOLO model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    cap = None
    rect = None
    window_title = None
    sct = None

    if window_mode:
        if platform.system() != "Windows":
            print("Error: window:<title> mode currently supports Windows only.")
            return
        if mss is None:
            print("Error: mss is required for window capture. Install with: pip install mss")
            return
        rect_info = find_window_rect_windows(window_query)
        if rect_info is None:
            print(f'Error: No visible window found matching title substring "{window_query}".')
            print("Tip: open MuMu first, then try: --source \"window:MuMu\"")
            return
        rect = {k: rect_info[k] for k in ("left", "top", "width", "height")}
        window_title = rect_info["title"]
        sct = mss.mss()
        print(f'Capturing window: "{window_title}"')
    else:
        # Open video capture from camera index or stream path/URL.
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {args.source}")
            print("Use --source window:MuMu for direct MuMu window capture on Windows.")
            print("Or use a webcam index / virtual camera source.")
            return

    print("Starting live detection... Press 'q' to quit")

    fps_start_time = time.time()
    frame_count = 0

    while True:
        if window_mode:
            # Re-acquire window bounds periodically in case window moved.
            if frame_count % max(1, args.window_refresh) == 0:
                rect_info = find_window_rect_windows(window_query)
                if rect_info is None:
                    print(f'Error: Window "{window_query}" not found anymore.')
                    break
                rect = {k: rect_info[k] for k in ("left", "top", "width", "height")}
                window_title = rect_info["title"]
            sct_img = sct.grab(rect)
            # mss returns BGRA; convert to BGR for OpenCV/YOLO
            frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break

        # Run inference
        results = model(frame, conf=args.conf, verbose=False)

        # Get annotated frame
        annotated_frame = results[0].plot()

        # Calculate and display FPS if requested
        if args.show_fps:
            frame_count += 1
            if frame_count % 30 == 0:  # Update every 30 frames
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display result
        cv2.imshow('Clash Royale Live Detection', annotated_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if cap is not None:
        cap.release()
    if sct is not None:
        sct.close()
    cv2.destroyAllWindows()
    print("Detection stopped")


if __name__ == "__main__":
    main()
