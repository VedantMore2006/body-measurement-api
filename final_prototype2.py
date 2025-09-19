import cv2
import numpy as np
from ultralytics import YOLO

# ============================
# Step 1: Detect A4 with OpenCV
# ============================

# Load the image
img = cv2.imread("s4.jpg")

# Convert to grayscale and apply edge detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

# Find contours in the image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

a4_height_pixels = None
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = h / float(w)

    # Filter by aspect ratio, size, and position (to avoid false detections like shoes)
    if 1.3 < aspect_ratio < 1.5 and 400 < h < 700 and y < img.shape[0] * 0.6:
        a4_height_pixels = h
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw A4 box
        break

if not a4_height_pixels:
    raise Exception("A4 not detected!")

# Compute scale factor (real cm per pixel)
scale_factor = 29.7 / a4_height_pixels
print(
    f"[INFO] A4 detected. Pixel height={a4_height_pixels}, Scale={scale_factor:.2f} cm/pixel"
)

# =================================
# Step 2: Detect Person with YOLOv8n
# =================================

person_model = YOLO("yolov8n.pt")
results = person_model(img)

person_bbox = None
for r in results:
    for box in r.boxes.xyxy:  # bounding box coordinates
        x1, y1, x2, y2 = map(int, box[:4])
        person_bbox = (x1, y1, x2, y2)

        # Person height in pixels
        bbox_height = y2 - y1

        # Convert to cm (removed manual -15 offset—let’s test raw scale first)
        total_height_cm = (round(bbox_height * scale_factor, 2)) - 12

        # Draw bounding box and height on image
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
    raise Exception("Person not detected!")

# ====================================
# Step 3: Pose Estimation with YOLOv8n-pose
# ====================================

pose_model = YOLO("yolov8n-pose.pt")
pose_results = pose_model(img)


# Helper function: distance between two keypoints (scaled to cm)
def dist(p1, p2):
    return round(np.linalg.norm(np.array(p1) - np.array(p2)) * scale_factor, 2)


for r in pose_results:
    keypoints = r.keypoints.xy.cpu().numpy()[0]  # 17 keypoints (x, y)

    # Key landmark indices (COCO format)
    ls, rs = keypoints[5], keypoints[6]  # left/right shoulder
    lh, rh = keypoints[11], keypoints[12]  # left/right hip
    le, re = keypoints[7], keypoints[8]  # elbows
    lw, rw = keypoints[9], keypoints[10]  # wrists
    lk, rk = keypoints[13], keypoints[14]  # knees
    la, ra = keypoints[15], keypoints[16]  # ankles

    # Measurements (converted to cm)
    ShoulderWidth = dist(ls, rs)
    ChestWidth = float(round(ShoulderWidth * 0.9, 2))  # rough approx
    Waist = float(dist(lh, rh))
    Hips = float(round(Waist * 1.05, 2))  # approx
    ArmLength = float(round(max(dist(ls, lw), dist(rs, rw)), 2))
    ShoulderToWaist = float(round(np.mean([dist(ls, lh), dist(rs, rh)]), 2))
    WaistToKnee = float(round(np.mean([dist(lh, lk), dist(rh, rk)]), 2))
    LegLength = float(round(np.mean([dist(lh, la), dist(rh, ra)]), 2))

    # Final results dictionary
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

    # Print results
    print("\n[MEASURED PARAMETERS]")
    for k, v in params.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
