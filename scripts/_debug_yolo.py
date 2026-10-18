"""Quick debug: test YOLO + MediaPipe on a single frame from cam02."""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent

# Load one frame from cam02
cap = cv2.VideoCapture(str(ROOT / "data/02/0203.MP4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 600)  # ~5 seconds in at 120fps
ok, bgr = cap.read()
cap.release()
print(f"Frame shape: {bgr.shape}")  # (2160, 3840, 3)

# Downscale
scale = 1920 / bgr.shape[1]
small = cv2.resize(bgr, (1920, int(bgr.shape[0] * scale)))
rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
print(f"Downscaled: {rgb.shape}")

# YOLO person detection
yolo = YOLO("yolov8n.pt", verbose=False)
results = yolo(rgb, verbose=False, classes=[0], conf=0.3)
boxes = results[0].boxes
print(f"YOLO detections: {len(boxes) if boxes is not None else 0}")
if boxes is not None and len(boxes) > 0:
    for i, (xyxy, conf) in enumerate(zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy())):
        x1, y1, x2, y2 = xyxy.astype(int)
        print(f"  Person {i}: [{x1},{y1}]-[{x2},{y2}] conf={conf:.2f} size={x2-x1}x{y2-y1}")
        
        # Expand bbox
        bw, bh = x2 - x1, y2 - y1
        pad = 0.25
        dx, dy = int(bw * pad), int(bh * pad)
        ex1 = max(0, x1 - dx)
        ey1 = max(0, y1 - dy)
        ex2 = min(rgb.shape[1], x2 + dx)
        ey2 = min(rgb.shape[0], y2 + dy)
        crop = rgb[ey1:ey2, ex1:ex2]
        print(f"  Crop: [{ex1},{ey1}]-[{ex2},{ey2}] = {crop.shape}")
        
        # MediaPipe on crop
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(ROOT / "models/pose_landmarker_heavy.task")),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
        result = landmarker.detect(mp_image)
        landmarker.close()
        
        n_poses = len(result.pose_landmarks) if result.pose_landmarks else 0
        print(f"  MediaPipe poses in crop: {n_poses}")
        if n_poses > 0:
            visible = sum(1 for lm in result.pose_landmarks[0] if lm.visibility > 0.5)
            print(f"  Visible landmarks: {visible}/33")
else:
    print("NO YOLO DETECTIONS!")
