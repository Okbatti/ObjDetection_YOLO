"""
YOLOv8 Detailed Video Analysis

Produces timeline CSV, peak frame snapshot, periodic snapshots, and per-class statistics.

Usage:
    python analyze.py restaurant_video.mp4
    python analyze.py restaurant_video.mp4 --model yolov8m.pt --snapshot-interval 5
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Detailed object detection analysis on video")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="YOLOv8 model to use (default: yolov8s.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--output", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--device", default="",
                        help="Device to run on (default: auto-detect)")
    parser.add_argument("--snapshot-interval", type=int, default=10,
                        help="Save a snapshot every N seconds (default: 10)")
    return parser.parse_args()


class FrameAnalyzer:
    def __init__(self):
        self.total_counter = Counter()
        self.timeline = []  # list of (frame_num, timestamp, {class: count})
        self.class_confidences = defaultdict(list)  # class -> [confidence scores]
        self.class_first_frame = {}  # class -> first frame number
        self.class_last_frame = {}   # class -> last frame number
        self.peak_frame_num = 0
        self.peak_detection_count = 0

    def process_frame(self, frame_num, timestamp, results, model_names):
        frame_counts = Counter()

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model_names[class_id]
            confidence = float(box.conf[0])

            frame_counts[class_name] += 1
            self.total_counter[class_name] += 1
            self.class_confidences[class_name].append(confidence)

            if class_name not in self.class_first_frame:
                self.class_first_frame[class_name] = frame_num
            self.class_last_frame[class_name] = frame_num

        self.timeline.append((frame_num, timestamp, dict(frame_counts)))

        total_in_frame = sum(frame_counts.values())
        if total_in_frame > self.peak_detection_count:
            self.peak_detection_count = total_in_frame
            self.peak_frame_num = frame_num

        return frame_counts


def main():
    args = parse_args()

    if not os.path.isfile(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    video_name = Path(args.video_path).stem
    os.makedirs(args.output, exist_ok=True)
    snapshots_dir = os.path.join(args.output, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

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

    output_video_path = os.path.join(args.output, f"{video_name}_analyzed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Analyzing: {args.video_path}")
    print(f"  Resolution: {width}x{height} | FPS: {fps} | Frames: {total_frames} | Duration: {duration_sec:.1f}s")
    print(f"  Snapshots every {args.snapshot_interval}s")
    print()

    analyzer = FrameAnalyzer()
    snapshot_interval_frames = args.snapshot_interval * fps
    frame_count = 0
    peak_annotated_frame = None
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps if fps > 0 else 0
        device_arg = {"device": args.device} if args.device else {}
        results = model(frame, conf=args.conf, verbose=False, **device_arg)

        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        analyzer.process_frame(frame_count, timestamp, results, model.names)

        # Save peak frame
        if frame_count == analyzer.peak_frame_num:
            peak_annotated_frame = annotated_frame.copy()

        # Save periodic snapshots
        if snapshot_interval_frames > 0 and frame_count % snapshot_interval_frames == 0:
            snap_path = os.path.join(snapshots_dir, f"{video_name}_t{int(timestamp)}s.jpg")
            cv2.imwrite(snap_path, annotated_frame)

        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {frame_count}/{total_frames} frames ({frame_count / elapsed:.1f} fps)")

    cap.release()
    out.release()

    elapsed = time.time() - start_time

    # Save peak frame
    if peak_annotated_frame is not None:
        peak_path = os.path.join(args.output, f"{video_name}_peak_frame.jpg")
        cv2.imwrite(peak_path, peak_annotated_frame)
        print(f"\nPeak frame saved: {peak_path} (frame {analyzer.peak_frame_num}, {analyzer.peak_detection_count} detections)")

    # Write timeline CSV
    csv_path = os.path.join(args.output, f"{video_name}_timeline.csv")
    all_classes = sorted(analyzer.total_counter.keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp_sec"] + all_classes)
        for frame_num, timestamp, counts in analyzer.timeline:
            row = [frame_num, f"{timestamp:.2f}"]
            for cls in all_classes:
                row.append(counts.get(cls, 0))
            writer.writerow(row)

    print(f"Timeline CSV saved: {csv_path}")

    # Build detailed report
    report_lines = [
        "=" * 60,
        "DETAILED ANALYSIS REPORT",
        "=" * 60,
        f"Video: {args.video_path}",
        f"Duration: {duration_sec:.1f}s | Frames: {total_frames} | FPS: {fps}",
        f"Model: {args.model} | Confidence threshold: {args.conf}",
        f"Processing time: {elapsed:.1f}s ({frame_count / elapsed:.1f} fps)",
        "",
        f"Peak frame: #{analyzer.peak_frame_num} with {analyzer.peak_detection_count} simultaneous detections",
        "",
        "-" * 60,
        "PER-CLASS STATISTICS",
        "-" * 60,
        f"{'Class':<20s} {'Total':>8s} {'Avg/Frame':>10s} {'Avg Conf':>10s} {'First':>8s} {'Last':>8s}",
        "-" * 60,
    ]

    for cls in all_classes:
        total = analyzer.total_counter[cls]
        avg_per_frame = total / total_frames if total_frames > 0 else 0
        avg_conf = sum(analyzer.class_confidences[cls]) / len(analyzer.class_confidences[cls])
        first = analyzer.class_first_frame[cls]
        last = analyzer.class_last_frame[cls]
        report_lines.append(
            f"{cls:<20s} {total:>8,} {avg_per_frame:>10.2f} {avg_conf:>10.2f} {first:>8} {last:>8}"
        )

    report_lines.extend([
        "",
        f"Unique object classes found: {len(analyzer.total_counter)}",
        f"Total detections: {sum(analyzer.total_counter.values()):,}",
        "",
        "Output files:",
        f"  Annotated video: {output_video_path}",
        f"  Timeline CSV:    {csv_path}",
        f"  Snapshots:       {snapshots_dir}/",
    ])

    if peak_annotated_frame is not None:
        report_lines.append(f"  Peak frame:      {peak_path}")

    report_path = os.path.join(args.output, f"{video_name}_analysis_report.txt")
    report_lines.append(f"  This report:     {report_path}")

    report = "\n".join(report_lines)
    print()
    print(report)

    with open(report_path, "w") as f:
        f.write(report + "\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
