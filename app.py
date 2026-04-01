"""
YOLOv8 Object Detection Web App

Upload a video, run YOLO detection, and view/download the annotated result.

Usage:
    python app.py
    # Then open http://localhost:5000
"""

import os
import time
import uuid
import json
from collections import Counter
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
from ultralytics import YOLO

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "output"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# Global progress tracking
processing_jobs = {}

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    job_id = str(uuid.uuid4())[:8]
    ext = file.filename.rsplit(".", 1)[1].lower()
    input_filename = f"{job_id}_input.{ext}"
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_filename)
    file.save(input_path)

    # Get video info
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    processing_jobs[job_id] = {
        "status": "uploaded",
        "progress": 0,
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
        "input_path": input_path,
        "input_filename": input_filename,
    }

    return jsonify({"job_id": job_id, "total_frames": total_frames, "duration": f"{duration:.1f}s",
                     "resolution": f"{width}x{height}", "fps": fps})


@app.route("/detect/<job_id>", methods=["POST"])
def run_detection(job_id):
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404

    job = processing_jobs[job_id]
    if job["status"] == "processing":
        return jsonify({"error": "Already processing"}), 400

    job["status"] = "processing"
    job["progress"] = 0

    conf = float(request.json.get("conf", 0.25)) if request.json else 0.25
    model_name = request.json.get("model", "yolov8s.pt") if request.json else "yolov8s.pt"

    input_path = job["input_path"]
    output_filename = f"{job_id}_detected.mp4"
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)

    try:
        model = YOLO(model_name)

        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, job["fps"], (job["width"], job["height"]))

        object_counter = Counter()
        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf, verbose=False)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                object_counter[class_name] += 1

            frame_count += 1
            job["progress"] = int((frame_count / job["total_frames"]) * 100)

        cap.release()
        out.release()

        elapsed = time.time() - start_time

        summary = {
            "total_detections": sum(object_counter.values()),
            "unique_classes": len(object_counter),
            "objects": [{"name": name, "count": count} for name, count in object_counter.most_common()],
            "processing_time": f"{elapsed:.1f}s",
            "processing_fps": f"{frame_count / elapsed:.1f}",
            "output_video": output_filename,
        }

        job["status"] = "done"
        job["progress"] = 100
        job["summary"] = summary

        return jsonify(summary)

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        return jsonify({"error": str(e)}), 500


@app.route("/progress/<job_id>")
def get_progress(job_id):
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
    job = processing_jobs[job_id]
    return jsonify({"status": job["status"], "progress": job["progress"]})


@app.route("/output/<filename>")
def serve_output(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
