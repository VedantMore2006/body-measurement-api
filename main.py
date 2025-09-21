from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from ultralytics import YOLO
import io
import os

app = FastAPI()
person_model = None
pose_model = None

@app.on_event("startup")
async def startup_event():
    print("FastAPI application startup: Models are loading...")
    global person_model, pose_model
    person_model = YOLO("yolov8n.pt")  # Person detection model
    pose_model = YOLO("yolov8n-pose.pt")  # Pose estimation model
    print("Models loaded successfully.")

@app.get("/healthz")
async def health_check():
    print("Health check hit!")
    return {"status": "healthy"}
#

@app.post("/detect/")
async def detect_measurements(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    # A4 detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    a4_height_pixels = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w)
        if 1.3 < aspect_ratio < 1.5 and 400 < h < 700 and y < img.shape[0] * 0.6:
            a4_height_pixels = h
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break

    if not a4_height_pixels:
        return {"error": "A4 not detected!"}

    scale_factor = round(29.7 / a4_height_pixels, 4)

    # Person detection
    results = person_model(img)
    person_bbox = None
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            person_bbox = (x1, y1, x2, y2)
            bbox_height = y2 - y1
            total_height_cm = round(bbox_height * scale_factor, 2) - 12
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                img,
                f"Height: {total_height_cm:.2f} cm",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )
            break

    if person_bbox is None:
        return {"error": "Person not detected!"}

    # Pose estimation
    pose_results = pose_model(img)

    def dist(p1, p2):
        return round(np.linalg.norm(np.array(p1) - np.array(p2)) * scale_factor, 2)

    for r in pose_results:
        keypoints = r.keypoints.xy.cpu().numpy()[0]
        ls, rs = keypoints[5], keypoints[6]  # shoulders
        lh, rh = keypoints[11], keypoints[12]  # hips
        le, re = keypoints[7], keypoints[8]  # elbows
        lw, rw = keypoints[9], keypoints[10]  # wrists
        lk, rk = keypoints[13], keypoints[14]  # knees
        la, ra = keypoints[15], keypoints[16]  # ankles

        ShoulderWidth = round(float(dist(ls, rs)), 2)
        ChestWidth = round(float(ShoulderWidth * 0.9), 2)
        Waist = round(float(dist(lh, rh)), 2)
        Hips = round(float(Waist * 1.05), 2)
        ArmLength = round(float(max(dist(ls, lw), dist(rs, rw))), 2)
        ShoulderToWaist = round(float(np.mean([dist(ls, lh), dist(rs, rh)])), 2)
        WaistToKnee = round(float(np.mean([dist(lh, lk), dist(rh, rk)])), 2)
        LegLength = round(float(np.mean([dist(lh, la), dist(rh, ra)])), 2)

        params = {
            "Gender": "Needs classifier",
            "Age": "Needs classifier",
            "ShoulderWidth_cm": ShoulderWidth,
            "ChestWidth_cm": ChestWidth,
            "Waist_cm": Waist,
            "Hips_cm": Hips,
            "ArmLength_cm": ArmLength,
            "ShoulderToWaist_cm": ShoulderToWaist,
            "WaistToKnee_cm": WaistToKnee,
            "LegLength_cm": LegLength,
            "TotalHeight_cm": total_height_cm,
        }
        return params  # Fixed indent: aligned with 'for r in pose_results'

    return {"error": "No pose data detected!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)