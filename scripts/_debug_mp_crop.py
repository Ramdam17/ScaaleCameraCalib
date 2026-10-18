"""Debug: test why MediaPipe fails on YOLO crops."""
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Load frame
cap = cv2.VideoCapture("data/02/0202.MP4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 600)
ok, bgr = cap.read()
cap.release()

small = cv2.resize(bgr, (1920, 1080))
rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

# YOLO
yolo = YOLO("yolov8m.pt", verbose=False)
res = yolo(small, verbose=False, classes=[0], conf=0.25)
boxes = res[0].boxes

# MediaPipe 
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="models/pose_landmarker_heavy.task"),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)
landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

# Test 1: Full frame (should work)
mp_full = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
res_full = landmarker.detect(mp_full)
print(f"Full frame: {len(res_full.pose_landmarks)} pose(s)")

# Test 2: Crop without contiguous copy
xyxy = boxes.xyxy.cpu().numpy()[0].astype(int)
bx1, by1, bx2, by2 = xyxy
bw, bh = bx2 - bx1, by2 - by1
pad = 0.25
dx, dy = int(bw * pad), int(bh * pad)
ex1 = max(0, bx1 - dx)
ey1 = max(0, by1 - dy)
ex2 = min(1920, bx2 + dx)
ey2 = min(1080, by2 + dy)

crop_view = rgb[ey1:ey2, ex1:ex2]
print(f"\nCrop view: shape={crop_view.shape} contiguous={crop_view.flags['C_CONTIGUOUS']} strides={crop_view.strides}")
mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_view)
r = landmarker.detect(mp_img)
print(f"  Crop view result: {len(r.pose_landmarks)} pose(s)")

# Test 3: Contiguous copy
crop_copy = np.ascontiguousarray(crop_view)
print(f"\nCrop copy: shape={crop_copy.shape} contiguous={crop_copy.flags['C_CONTIGUOUS']} strides={crop_copy.strides}")
mp_img2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_copy)
r2 = landmarker.detect(mp_img2)
print(f"  Crop copy result: {len(r2.pose_landmarks)} pose(s)")

# Test 4: Resize crop to bigger
crop_resized = cv2.resize(crop_copy, (640, 640))
print(f"\nCrop resized 640x640: shape={crop_resized.shape}")
mp_img3 = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_resized)
r3 = landmarker.detect(mp_img3)
print(f"  Crop resized result: {len(r3.pose_landmarks)} pose(s)")

# Test 5: Lower detection confidence
options2 = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="models/pose_landmarker_heavy.task"),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.1,
    min_tracking_confidence=0.1,
)
landmarker2 = mp.tasks.vision.PoseLandmarker.create_from_options(options2)
mp_img4 = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_copy)
r4 = landmarker2.detect(mp_img4)
print(f"\nCrop with low conf (0.1): {len(r4.pose_landmarks)} pose(s)")

landmarker.close()
landmarker2.close()
print("\nDone")
