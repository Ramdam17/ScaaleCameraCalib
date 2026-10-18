"""Debug: extract a sample frame from each camera and analyze person positions."""
import json
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "config.json") as f:
    cfg = json.load(f)

# Check where the detected person is in each camera
for cam_id in ["cam01", "cam02", "cam03"]:
    with open(f"output/poses/{cam_id}_poses.json") as f:
        data = json.load(f)
    
    img_w, img_h = data["image_size"]
    
    # Analyze center X of detected persons
    centers_x = [[] for _ in range(3)]  # per person_index
    for frame in data["frames"]:
        for pose in frame.get("poses", []):
            bb = pose.get("bounding_box")
            if bb:
                cx = (bb["x_min"] + bb["x_max"]) / 2.0
                pi = pose["person_index"]
                if pi < 3:
                    centers_x[pi].append(cx)
    
    print(f"\n{cam_id} ({img_w}x{img_h}):")
    for pi in range(3):
        if centers_x[pi]:
            arr = np.array(centers_x[pi])
            print(f"  person_index {pi}: {len(arr)} detections, "
                  f"center_x: min={arr.min():.0f} mean={arr.mean():.0f} max={arr.max():.0f} "
                  f"({arr.mean()/img_w*100:.0f}% from left)")

# Also extract and save a sample frame from each camera for visual inspection
for cam_id in ["cam01", "cam02", "cam03"]:
    video_path = str(ROOT / cfg["cameras"][cam_id]["skeleton_video"])
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Jump to 5 seconds in
    target_frame = int(5 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, bgr = cap.read()
    if ok:
        out_path = ROOT / "output" / "visualization" / f"debug_{cam_id}_frame5s.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Save at half res
        small = cv2.resize(bgr, (bgr.shape[1] // 2, bgr.shape[0] // 2))
        cv2.imwrite(str(out_path), small)
        print(f"\nSaved debug frame: {out_path}")
    cap.release()
