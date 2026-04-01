FROM python:3.11-slim

# Install system dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# 1. Install everything from requirements
# 2. Force replace opencv-python with headless version (no GUI needed on server)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn && \
    pip uninstall -y opencv-python 2>/dev/null; \
    pip install --no-cache-dir opencv-python-headless && \
    python -c "import cv2; print(f'OpenCV {cv2.__version__} OK')"

# Copy application code
COPY . .

# Create directories
RUN mkdir -p uploads output output/snapshots

# Download YOLO model at build time so it's cached
RUN python -c "from ultralytics import YOLO; model = YOLO('yolov8s.pt'); print('YOLO model downloaded OK')"

EXPOSE ${PORT:-5000}

CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --timeout 300 --workers 1 --threads 2
