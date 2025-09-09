FROM python:3.13-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy .env (if using repo option)
COPY .env /app/.env

# Copy all files
COPY . /app

# Install Python dependencies
RUN pip install fastapi uvicorn opencv-python-headless numpy ultralytics

# Preload models (fixed to avoid GUI issues)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').load(); YOLO('yolov8n-pose.pt').load()"

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
