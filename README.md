# Object Dimension Measurement API

Yo, fam! This repo is your go-to for a dope FastAPI app that measures human dimensions (height, shoulder width, etc.) using YOLOv8 models. Built for Android integration—let’s get it poppin’! 🚀

## What’s Inside?
- `api.py`: The main FastAPI app with YOLO-based detection and pose estimation.
- `yolov8n.pt` & `yolov8n-pose.pt`: Pre-trained models for person and pose detection.
- `Dockerfile`: Container setup to run the app anywhere.
- Test images (`s1.jpg`, `s2.jpg`, etc.): Sample data to vibe with.

## How It Works
1. Upload an image with an A4 sheet for scale.
2. API detects the person, estimates pose, and returns measurements (cm) like:
   - Total Height
   - Shoulder Width
   - Arm Length
   - And more!
3. Secured with an API key (`x-api-key` header).

## Setup & Run Locally
### Prerequisites
- Docker installed (`sudo pacman -S docker` on Arch).
- Python 3.13 env (e.g., `yolox_env`).

### Steps
1. Clone this repo:
   ```bash
   git clone <your-repo-url>
   cd all_parameters
