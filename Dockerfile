FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (use headless OpenCV for server)
COPY requirements.txt .
RUN pip install --no-cache-dir opencv-python-headless gunicorn && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python 2>/dev/null || true

# Copy application code
COPY . .

# Create directories
RUN mkdir -p uploads output output/snapshots

# Download YOLO model at build time so it's cached
RUN python -c "import cv2; print('OpenCV OK'); from ultralytics import YOLO; YOLO('yolov8s.pt'); print('YOLO OK')"

EXPOSE ${PORT:-5000}

CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --timeout 300 --workers 1 --threads 2
