import cv2
import numpy as np
from ultralytics import YOLO

# Load image
img = cv2.imread("1.jpg")
orig = img.copy()

# --- Step 1: Detect A4 Paper ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


a4_height_pixels = None
img_h, img_w = gray.shape

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = h / float(w)

    # 1. Aspect ratio check
    if not (1.3 < aspect_ratio < 1.5):
        continue

    # 2. Position filter (ignore bottom part of image)
    if y > img_h * 0.6:
        continue

    # 3. Size filter
    if not (400 < h < 700):
        continue

    # 4. Brightness check
    roi = gray[y : y + h, x : x + w]
    mean_brightness = cv2.mean(roi)[0]
    if mean_brightness < 150:  # adjust threshold
        continue

    # If all filters pass → likely A4
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    a4_height_pixels = h
    print(
        f"Detected A4: (x={x}, y={y}, w={w}, h={h}), Aspect={aspect_ratio:.2f}, Brightness={mean_brightness:.1f}"
    )
    break

if a4_height_pixels is None:
    print("A4 not detected!")
    exit()

# --- Step 2: Calculate scale factor ---
real_a4_height_cm = 29.7
scale_factor = real_a4_height_cm / a4_height_pixels
print(
    f"A4 Height Pixels: {a4_height_pixels}, Scale factor: {scale_factor:.4f} cm/pixel"
)

# --- Step 3: YOLO Pose Detection ---
model = YOLO("yolov8n-pose.pt")  # make sure model is downloaded
results = model(orig)

# Extract keypoints
keypoints = results[0].keypoints.xy.cpu().numpy()[0]

# Top (nose) and bottom (ankle) landmarks
nose = keypoints[0]  # nose
ankle = (
    keypoints[15] if keypoints[15][1] > keypoints[16][1] else keypoints[16]
)  # lower ankle

person_height_px = abs(ankle[1] - nose[1])
person_height_cm = person_height_px * scale_factor

print(f"Person height in pixels: {person_height_px:.2f}")
print(f"Estimated real height: {person_height_cm:.2f} cm")

# Show results
cv2.imshow("Detected A4 + Person", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
