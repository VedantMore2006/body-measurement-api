# from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Header
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import io
#
# app = FastAPI()
#
# # Load models once on startup
# person_model = YOLO("yolov8n.pt")
# pose_model = YOLO("yolov8n-pose.pt")
#
#
# # API Key Verification Function
# async def verify_api_key(x_api_key: str = Header(None)):
#     if (
#         x_api_key != "gY5pR3L8aB1nS9eK4dH6cJ7mF2qV4oX"
#     ):  # Replace with a strong key (e.g., from secrets module)
#         raise HTTPException(status_code=401, detail="Invalid API key")
#     return x_api_key
#
#
# @app.post("/detect/", dependencies=[Depends(verify_api_key)])
# async def detect_measurements(file: UploadFile = File(...)):
#     # Read image from uploaded file
#     contents = await file.read()
#     img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
#
#     # Step 1: Detect A4 with OpenCV
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     a4_height_pixels = None
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         aspect_ratio = h / float(w)
#         if 1.3 < aspect_ratio < 1.5 and 400 < h < 700 and y < img.shape[0] * 0.6:
#             a4_height_pixels = h
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             break
#
#     if not a4_height_pixels:
#         return {"error": "A4 not detected!"}
#
#     scale_factor = round(29.7 / a4_height_pixels, 4)
#
#     # Step 2: Detect Person with YOLOv8n
#     results = person_model(img)
#     person_bbox = None
#     for r in results:
#         for box in r.boxes.xyxy:
#             x1, y1, x2, y2 = map(int, box[:4])
#             person_bbox = (x1, y1, x2, y2)
#             bbox_height = y2 - y1
#             total_height_cm = round(bbox_height * scale_factor, 2) - 12
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(
#                 img,
#                 f"Height: {total_height_cm:.2f} cm",
#                 (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (255, 0, 0),
#                 2,
#             )
#             break
#
#     if person_bbox is None:
#         return {"error": "Person not detected!"}
#
#     # Step 3: Pose Estimation with YOLOv8n-pose
#     pose_results = pose_model(img)
#
#     def dist(p1, p2):
#         return round(np.linalg.norm(np.array(p1) - np.array(p2)) * scale_factor, 2)
#
#     for r in pose_results:
#         keypoints = r.keypoints.xy.cpu().numpy()[0]
#         ls, rs = keypoints[5], keypoints[6]  # shoulders
#         lh, rh = keypoints[11], keypoints[12]  # hips
#         le, re = keypoints[7], keypoints[8]  # elbows
#         lw, rw = keypoints[9], keypoints[10]  # wrists
#         lk, rk = keypoints[13], keypoints[14]  # knees
#         la, ra = keypoints[15], keypoints[16]  # ankles
#
#         ShoulderWidth = round(float(dist(ls, rs)), 2)
#         ChestWidth = round(float(ShoulderWidth * 0.9), 2)
#         Waist = round(float(dist(lh, rh)), 2)
#         Hips = round(float(Waist * 1.05), 2)
#         ArmLength = round(float(max(dist(ls, lw), dist(rs, rw))), 2)
#         ShoulderToWaist = round(float(np.mean([dist(ls, lh), dist(rs, rh)])), 2)
#         WaistToKnee = round(float(np.mean([dist(lh, lk), dist(rh, rk)])), 2)
#         LegLength = round(float(np.mean([dist(lh, la), dist(rh, ra)])), 2)
#
#         params = {
#             "Gender": "Needs classifier",
#             "Age": "Needs classifier",
#             "ShoulderWidth_cm": ShoulderWidth,
#             "ChestWidth_cm": ChestWidth,
#             "Waist_cm": Waist,
#             "Hips_cm": Hips,
#             "ArmLength_cm": ArmLength,
#             "ShoulderToWaist_cm": ShoulderToWaist,
#             "WaistToKnee_cm": WaistToKnee,
#             "LegLength_cm": LegLength,
#             "TotalHeight_cm": total_height_cm,
#         }
#
#         return params  # make sure this is only returned once per valid pose
#
#     return {"error": "No pose data detected!"}
#
#
# # Run with uvicorn
# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from ultralytics import YOLO
import io

app = FastAPI()

# Load models once on startup
person_model = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt")


@app.post("/detect/")
async def detect_measurements(file: UploadFile = File(...)):
    # Read image from uploaded file
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    # Step 1: Detect A4 with OpenCV
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

    # Step 2: Detect Person with YOLOv8n
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

    # Step 3: Pose Estimation with YOLOv8n-pose
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

        return params  # make sure this is only returned once per valid pose

    return {"error": "No pose data detected!"}


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
#
# from fastapi import FastAPI, UploadFile, File
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import io
#
# app = FastAPI()
#
# # Load models once on startup
# person_model = YOLO("yolov8n.pt")
# pose_model = YOLO("yolov8n-pose.pt")
#
# @app.post("/detect/")
# async def detect_measurements(file: UploadFile = File(...)):
#     # Read image from uploaded file
#     contents = await file.read()
#     img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
#
#     # Step 1: Detect A4 with OpenCV
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     a4_height_pixels = None
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         aspect_ratio = h / float(w)
#         if 1.3 < aspect_ratio < 1.5 and 400 < h < 700 and y < img.shape[0] * 0.6:
#             a4_height_pixels = h
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             break
#
#     if not a4_height_pixels:
#         return {"error": "A4 not detected!"}
#
#     scale_factor = round(29.7 / a4_height_pixels, 4)
#
#     # Step 2: Detect Person with YOLOv8n
#     results = person_model(img)
#     person_bbox = None
#     for r in results:
#         for box in r.boxes.xyxy:
#             x1, y1, x2, y2 = map(int, box[:4])
#             person_bbox = (x1, y1, x2, y2)
#             bbox_height = y2 - y1
#             total_height_cm = round(bbox_height * scale_factor, 2) - 12
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(
#                 img,
#                 f"Height: {total_height_cm:.2f} cm",
#                 (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (255, 0, 0),
#                 2,
#             )
#             break
#
#     if person_bbox is None:
#         return {"error": "Person not detected!"}
#
#     # Step 3: Pose Estimation with YOLOv8n-pose
#     pose_results = pose_model(img)
#
#     def dist(p1, p2):
#         return round(np.linalg.norm(np.array(p1) - np.array(p2)) * scale_factor, 2)
#
#     for r in pose_results:
#         keypoints = r.keypoints.xy.cpu().numpy()[0]
#         ls, rs = keypoints[5], keypoints[6]  # shoulders
#         lh, rh = keypoints[11], keypoints[12]  # hips
#         le, re = keypoints[7], keypoints[8]  # elbows
#         lw, rw = keypoints[9], keypoints[10]  # wrists
#         lk, rk = keypoints[13], keypoints[14]  # knees
#         la, ra = keypoints[15], keypoints[16]  # ankles
#
#         # Convert all measurements from cm to inches (1 cm = 0.393701 inches)
#         ShoulderWidth = round(float(dist(ls, rs)) * 0.393701, 2)
#         ChestWidth = round(float(ShoulderWidth * 0.9), 2)
#         Waist = round(float(dist(lh, rh)) * 0.393701, 2)
#         Hips = round(float(Waist * 1.05), 2)
#         ArmLength = round(float(max(dist(ls, lw), dist(rs, rw))) * 0.393701, 2)
#         ShoulderToWaist = round(float(np.mean([dist(ls, lh), dist(rs, rh)]) * 0.393701), 2)
#         WaistToKnee = round(float(np.mean([dist(lh, lk), dist(rh, rk)]) * 0.393701), 2)
#         LegLength = round(float(np.mean([dist(lh, la), dist(rh, ra)]) * 0.393701), 2)
#         total_height_in = round((total_height_cm * 0.393701), 2)
#
#         params = {
#             "Gender": "Needs classifier",
#             "Age": "Needs classifier",
#             "ShoulderWidth_in": ShoulderWidth,
#             "ChestWidth_in": ChestWidth,
#             "Waist_in": Waist,
#             "Hips_in": Hips,
#             "ArmLength_in": ArmLength,
#             "ShoulderToWaist_in": ShoulderToWaist,
#             "WaistToKnee_in": WaistToKnee,
#             "LegLength_in": LegLength,
#             "TotalHeight_in": total_height_in,
#         }
#
#         return params  # make sure this is only returned once per valid pose
#
#     return {"error": "No pose data detected!"}
#
# # Run with uvicorn
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
