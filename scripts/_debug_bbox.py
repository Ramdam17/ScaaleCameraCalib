import json
import numpy as np

for cam_id in ["cam01", "cam02", "cam03"]:
    with open(f"output/poses/{cam_id}_poses.json") as f:
        data = json.load(f)
    
    widths = []
    heights = []
    for frame in data["frames"]:
        for pose in frame.get("poses", []):
            bb = pose.get("bounding_box")
            if bb:
                widths.append(bb["x_max"] - bb["x_min"])
                heights.append(bb["y_max"] - bb["y_min"])
    
    if widths:
        w = np.array(widths)
        h = np.array(heights)
        print(f"{cam_id}: {len(widths)} bboxes")
        print(f"  width:  min={w.min():.0f} median={np.median(w):.0f} max={w.max():.0f}")
        print(f"  height: min={h.min():.0f} median={np.median(h):.0f} max={h.max():.0f}")
        pct_w = np.median(w) / 1920 * 100
        pct_h = np.median(h) / 1080 * 100
        print(f"  person occupies: {pct_w:.1f}% width, {pct_h:.1f}% height")
    else:
        print(f"{cam_id}: no bounding boxes")
