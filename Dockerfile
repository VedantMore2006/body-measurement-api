FROM python:3.13-slim
WORKDIR /app
COPY . /app
RUN pip install fastapi uvicorn opencv-python-headless numpy ultralytics
# Download YOLO models if not included
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt')"
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
