FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn opencv-python-headless && \
    pip uninstall -y opencv-python 2>/dev/null || true

# Copy application code
COPY . .

# Create directories
RUN mkdir -p uploads output output/snapshots

# Download YOLO model at build time so it's cached
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"

EXPOSE ${PORT:-5000}

CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --timeout 300 --workers 1 --threads 2
