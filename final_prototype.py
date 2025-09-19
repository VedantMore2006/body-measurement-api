import cv2
import numpy as np
from ultralytics import YOLO

# --- Step 1: Detect A4 with OpenCV ---
img = cv2.imread("s1.jpg")
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
    raise Exception("A4 not detected!")

scale_factor = 29.7 / a4_height_pixels  # cm per pixel
print(f"[INFO] A4 pixels={a4_height_pixels}, scale={scale_factor:.4f} cm/pixel")

# --- Step 2: YOLO person + pose ---
pose_model = YOLO("yolov8n-pose.pt")
results = pose_model(img)

keypoints = results[0].keypoints.xy.cpu().numpy()[0]


def dist(p1, p2):  # Euclidean distance
    return np.linalg.norm(p1 - p2)


# --- Step 3: Extract measurements ---
nose = keypoints[0]
l_shoulder, r_shoulder = keypoints[5], keypoints[6]
l_hip, r_hip = keypoints[11], keypoints[12]
l_wrist, r_wrist = keypoints[9], keypoints[10]
l_knee, r_knee = keypoints[13], keypoints[14]
l_ankle, r_ankle = keypoints[15], keypoints[16]

# person height
ankle = l_ankle if l_ankle[1] > r_ankle[1] else r_ankle
height_px = dist(nose, ankle)
height_cm = height_px * scale_factor

# shoulder width
shoulder_cm = dist(l_shoulder, r_shoulder) * scale_factor

# waist/hip width
waist_cm = dist(l_hip, r_hip) * scale_factor

# arm length
arm_cm = dist(l_shoulder, l_wrist) * scale_factor

# leg length
leg_cm = dist(l_hip, ankle) * scale_factor

print(f"Height: {height_cm:.2f} cm")
print(f"Shoulder Width: {shoulder_cm:.2f} cm")
print(f"Waist Width: {waist_cm:.2f} cm")
print(f"Arm Length: {arm_cm:.2f} cm")
print(f"Leg Length: {leg_cm:.2f} cm")

# visualize
results[0].plot()
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
