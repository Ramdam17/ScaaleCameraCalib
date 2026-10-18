"""Debug: test YOLO + MediaPipe pipeline on a single frame."""
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Test full pipeline on one frame from cam02
cap = cv2.VideoCapture("data/02/0202.MP4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 600)
ok, bgr = cap.read()
cap.release()
print(f"Frame: {bgr.shape}")

small = cv2.resize(bgr, (1920, 1080))
rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

# YOLO
yolo = YOLO("yolov8m.pt", verbose=False)
res = yolo(small, verbose=False, classes=[0], conf=0.25)
boxes = res[0].boxes
print(f"YOLO: {len(boxes)} persons")

# MediaPipe
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="models/pose_landmarker_heavy.task"),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)
landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

# Process each YOLO detection
for i, (xyxy, conf) in enumerate(zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy())):
    bx1, by1, bx2, by2 = xyxy.astype(int)
    bw, bh = bx2 - bx1, by2 - by1
    pad = 0.25
    dx, dy = int(bw * pad), int(bh * pad)
    ex1 = max(0, bx1 - dx)
    ey1 = max(0, by1 - dy)
    ex2 = min(1920, bx2 + dx)
    ey2 = min(1080, by2 + dy)

    crop = rgb[ey1:ey2, ex1:ex2]
    print(f"  Person {i}: bbox=[{bx1},{by1},{bx2},{by2}] conf={conf:.2f} crop={crop.shape} dtype={crop.dtype}")

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
    result = landmarker.detect(mp_image)
    n_poses = len(result.pose_landmarks) if result.pose_landmarks else 0
    print(f"    MediaPipe: {n_poses} pose(s) detected")

landmarker.close()
print("Done")
