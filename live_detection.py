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
python live_detection.py --model path/to/model.pt --device 2

Press 'q' to quit.
"""

import argparse
import cv2
from ultralytics import YOLO
import time


def main():
    parser = argparse.ArgumentParser(
        description="Live Clash Royale Detection on MuMu")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--device', type=int, default=2,
                        help='Video device ID (default: 2 for /dev/video2)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--show-fps', action='store_true',
                        help='Display FPS counter')

    args = parser.parse_args()

    # Load YOLO model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Open video capture from scrcpy stream
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"Error: Could not open video device {args.device}")
        print("Make sure scrcpy is running: scrcpy --v4l2-sink=/dev/video2 --no-video-playback")
        return

    print("Starting live detection... Press 'q' to quit")

    fps_start_time = time.time()
    frame_count = 0

    while True:
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
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")


if __name__ == "__main__":
    main() < /content >
<parameter name = "filePath" > /Users/manroopkalsi/Documents/projects/Clash-Royale-Detection-Dataset/live_detection.py
