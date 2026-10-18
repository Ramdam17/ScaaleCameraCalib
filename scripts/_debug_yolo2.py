"""Debug: test YOLO detection without class filter."""
import cv2
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

cap = cv2.VideoCapture(str(ROOT / "data/02/0203.MP4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 600)
ok, bgr = cap.read()
cap.release()

# Try at full 4K
rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# Try downscaled
scale = 1920 / bgr.shape[1]
small = cv2.resize(bgr, (1920, int(bgr.shape[0] * scale)))
rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

for model_name in ["yolov8n.pt", "yolov8m.pt"]:
    print(f"\n=== {model_name} ===")
    yolo = YOLO(model_name, verbose=False)
    
    # Full resolution, no class filter
    results = yolo(rgb_full, verbose=False, conf=0.1)
    boxes = results[0].boxes
    n = len(boxes) if boxes else 0
    print(f"  Full 4K, no filter: {n} detections")
    if boxes and n > 0:
        for i in range(min(n, 5)):
            cls = int(boxes.cls[i].item())
            conf = boxes.conf[i].item()
            print(f"    class={cls} ({results[0].names[cls]}) conf={conf:.2f}")
    
    # Downscaled, no class filter
    results = yolo(rgb_small, verbose=False, conf=0.1)
    boxes = results[0].boxes
    n = len(boxes) if boxes else 0
    print(f"  1920px, no filter: {n} detections")
    if boxes and n > 0:
        for i in range(min(n, 5)):
            cls = int(boxes.cls[i].item())
            conf = boxes.conf[i].item()
            print(f"    class={cls} ({results[0].names[cls]}) conf={conf:.2f}")
    
    # Downscaled, person class only
    results = yolo(rgb_small, verbose=False, classes=[0], conf=0.1)
    boxes = results[0].boxes
    n = len(boxes) if boxes else 0
    print(f"  1920px, person only: {n} detections")
