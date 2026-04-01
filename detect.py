"""
YOLOv8 Object Detection on Video

Usage:
    python detect.py restaurant_video.mp4
    python detect.py restaurant_video.mp4 --model yolov8n.pt --conf 0.4
    python detect.py restaurant_video.mp4 --model yolov8m.pt --device cpu
"""

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Detect objects in a video using YOLOv8")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="YOLOv8 model to use (default: yolov8s.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--output", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--device", default="",
                        help="Device to run on (default: auto-detect, e.g. 'cpu', 'mps')")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    if not os.path.isfile(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Open video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {args.video_path}")
        sys.exit(1)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    video_name = Path(args.video_path).stem
    output_path = os.path.join(args.output, f"{video_name}_detected.mp4")

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing: {args.video_path}")
    print(f"  Resolution: {width}x{height} | FPS: {fps} | Frames: {total_frames} | Duration: {duration_sec:.1f}s")
    print(f"  Model: {args.model} | Confidence: {args.conf}")
    print()

    object_counter = Counter()
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        device_arg = {"device": args.device} if args.device else {}
        results = model(frame, conf=args.conf, verbose=False, **device_arg)

        # Draw annotations
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # Count detections
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            object_counter[class_name] += 1

        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed
            print(f"  Processed {frame_count}/{total_frames} frames ({fps_processing:.1f} fps)")

    cap.release()
    out.release()

    elapsed = time.time() - start_time

    # Build summary
    summary_lines = [
        "=" * 50,
        "DETECTION SUMMARY",
        "=" * 50,
        f"Video: {args.video_path}",
        f"Duration: {duration_sec:.1f}s | Frames: {total_frames} | FPS: {fps}",
        f"Model: {args.model} | Confidence threshold: {args.conf}",
        f"Processing time: {elapsed:.1f}s ({frame_count / elapsed:.1f} fps)",
        "",
        "Objects detected (total detections across all frames):",
    ]

    for obj, count in object_counter.most_common():
        summary_lines.append(f"  {obj:<20s}: {count:>8,}")

    summary_lines.extend([
        "",
        f"Unique object classes found: {len(object_counter)}",
        f"Total detections: {sum(object_counter.values()):,}",
        "",
        f"Output video saved to: {output_path}",
    ])

    report_path = os.path.join(args.output, f"{video_name}_report.txt")
    summary_lines.append(f"Report saved to: {report_path}")

    summary = "\n".join(summary_lines)

    # Print and save
    print()
    print(summary)

    with open(report_path, "w") as f:
        f.write(summary + "\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
