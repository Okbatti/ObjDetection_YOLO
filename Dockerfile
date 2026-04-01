FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch first (saves ~4GB vs CUDA version)
# Then install remaining dependencies with headless OpenCV
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir opencv-python-headless gunicorn && \
    pip install --no-cache-dir ultralytics numpy && \
    python -c "import cv2; import torch; print(f'OpenCV {cv2.__version__} OK, PyTorch {torch.__version__} OK')"

# Copy application code
COPY . .

# Create directories
RUN mkdir -p uploads output output/snapshots

# Download YOLO model at build time
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt'); print('YOLO model OK')"

EXPOSE ${PORT:-5000}

CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --timeout 300 --workers 1 --threads 2
