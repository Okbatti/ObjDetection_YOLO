FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch first (saves ~4GB vs CUDA version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
RUN pip install --no-cache-dir ultralytics numpy gunicorn flask

# Force replace opencv-python with headless version LAST
# (ultralytics installs opencv-python as dependency, but we need headless on server)
RUN pip uninstall -y opencv-python && \
    pip install --no-cache-dir opencv-python-headless && \
    python -c "import cv2; import torch; from ultralytics import YOLO; print('All imports OK')"

# Copy application code
COPY . .

# Create directories
RUN mkdir -p uploads output output/snapshots

# Download YOLO model at build time
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt'); print('YOLO model OK')"

EXPOSE ${PORT:-5000}

CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --timeout 300 --workers 1 --threads 2
