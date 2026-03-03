"""
05_gaze.py — Head orientation from 3D skeleton landmarks.

All cameras are overhead/elevated: MediaPipe FaceLandmarker detects 0%
of faces from these angles. Instead, head orientation is computed from
the triangulated 3D face landmarks (nose + eyes → face normal vector),
giving stable yaw/pitch/roll in the world coordinate system.

Steps:
  1. Load skeleton_3d.json and camera calibration
  2. For each video frame, find the matching 3D skeleton by synced timestamp
  3. Compute head orientation: face normal from (nose, left_eye, right_eye)
  4. Project 3D gaze arrow onto the camera image
  5. Save overlay video + JSON with angles per frame

Usage:
  python scripts/05_gaze.py
"""

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config.json") as f:
        return json.load(f)


def load_skeleton_3d() -> dict:
    with open(ROOT / "output" / "skeleton3d" / "skeleton_3d.json") as f:
        return json.load(f)


def load_calibration() -> dict:
    with open(ROOT / "output" / "calibration" / "calibration.json") as f:
        return json.load(f)


def extract_points(frame: dict) -> np.ndarray:
    """Extract (33, 3) array from a skeleton frame. NaN for missing landmarks."""
    pts = np.full((33, 3), np.nan)
    for lm in frame["landmarks"]:
        if lm["x"] is not None:
            pts[lm["id"]] = [lm["x"], lm["y"], lm["z"]]
    return pts


def clean_output():
    out_dir = ROOT / "output" / "gaze"
    if out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"Cleaned {out_dir}")


# ---------------------------------------------------------------------------
# Head pose from 3D skeleton
# ---------------------------------------------------------------------------


def compute_head_pose_3d(pts: np.ndarray) -> dict | None:
    """Compute head orientation from triangulated 3D face landmarks.

    World coordinate system (cam01 origin, overhead camera):
      X: right, Y: forward (on floor), Z: downward toward floor.

    Landmarks used:
      0  = nose tip
      2  = left_eye
      5  = right_eye

    Returns:
      face_normal: unit vector pointing outward from face (gaze direction)
      yaw_deg:   rotation in floor plane (0 = facing +Y, +90 = facing +X)
      pitch_deg: upward tilt from floor plane (positive = tilted up)
      roll_deg:  lateral head tilt (positive = right ear down)
    """
    nose = pts[0]
    left_eye = pts[2]
    right_eye = pts[5]

    if np.isnan(nose).any() or np.isnan(left_eye).any() or np.isnan(right_eye).any():
        return None

    eye_mid = (left_eye + right_eye) / 2.0

    # Face right vector: from left eye to right eye
    face_right = right_eye - left_eye
    r_norm = np.linalg.norm(face_right)
    if r_norm < 1e-6:
        return None
    face_right /= r_norm

    # Nose direction: from eye midpoint toward nose tip
    nose_dir = nose - eye_mid
    n_norm = np.linalg.norm(nose_dir)
    if n_norm < 1e-6:
        return None
    nose_dir /= n_norm

    # Face normal: outward from face plane (cross product)
    face_normal = np.cross(face_right, nose_dir)
    fn_norm = np.linalg.norm(face_normal)
    if fn_norm < 1e-6:
        return None
    face_normal /= fn_norm

    # Yaw: direction in horizontal (XY) floor plane
    yaw = float(np.degrees(np.arctan2(face_normal[0], face_normal[1])))

    # Pitch: tilt out of floor plane (Z is downward in world)
    horiz = np.sqrt(face_normal[0] ** 2 + face_normal[1] ** 2)
    pitch = float(np.degrees(np.arctan2(-face_normal[2], horiz)))

    # Roll: lateral head tilt (Z component of face_right vector)
    roll = float(np.degrees(np.arctan2(
        face_right[2], np.sqrt(face_right[0] ** 2 + face_right[1] ** 2)
    )))

    return {
        "face_normal": face_normal,
        "yaw_deg": round(yaw, 2),
        "pitch_deg": round(pitch, 2),
        "roll_deg": round(roll, 2),
    }


# ---------------------------------------------------------------------------
# Gaze arrow projection
# ---------------------------------------------------------------------------


def project_gaze_arrow(
    pts_3d: np.ndarray,
    head_pose: dict,
    K: np.ndarray,
    dist: np.ndarray,
    R_cam: np.ndarray,
    T_cam: np.ndarray,
    video_scale: float,
    arrow_len_m: float = 0.3,
) -> tuple[tuple | None, tuple | None]:
    """Project 3D nose position and gaze endpoint onto the camera image.

    Returns (nose_px, gaze_px) in video-resolution pixel coords, or (None, None).
    """
    nose_3d = pts_3d[0]
    if np.isnan(nose_3d).any():
        return None, None

    gaze_end_3d = nose_3d + head_pose["face_normal"] * arrow_len_m
    rvec, _ = cv2.Rodrigues(R_cam)

    nose_2d, _ = cv2.projectPoints(
        nose_3d.reshape(1, 1, 3), rvec, T_cam, K, dist)
    gaze_2d, _ = cv2.projectPoints(
        gaze_end_3d.reshape(1, 1, 3), rvec, T_cam, K, dist)

    # Scale from calibration space to video space
    nose_px = tuple((nose_2d.reshape(2) * video_scale).astype(int).tolist())
    gaze_px = tuple((gaze_2d.reshape(2) * video_scale).astype(int).tolist())

    return nose_px, gaze_px


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = load_config()
    gaze_cfg = cfg["gaze_estimation"]
    cam_id = gaze_cfg.get("camera", "cam02")

    t_start = time.time()
    print("=== Head Orientation from 3D Skeleton ===\n")
    clean_output()

    # Load data
    print("Loading skeleton + calibration...", flush=True)
    skeleton_data = load_skeleton_3d()
    calibration = load_calibration()

    cam = calibration["cameras"][cam_id]
    K = np.array(cam["camera_matrix"])
    dist = np.array(cam["dist_coeffs"])
    R_cam = np.array(cam["R"])
    T_cam = np.array(cam["T"])

    # Sync offset for this camera
    sync_path = ROOT / "output" / "poses" / "sync_offsets.json"
    with open(sync_path) as f:
        sync_offsets = json.load(f)
    sync_offset = sync_offsets.get(cam_id, 0.0)

    # Open video
    video_path = str(ROOT / cfg["cameras"][cam_id]["skeleton_video"])
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_fps = gaze_cfg["subsample_fps"]
    step = max(1, int(round(source_fps / target_fps)))
    expected = total_frames // step

    # K is calibrated at calib_w; video frames are orig_w
    calib_w = cam["image_size"][0]
    video_scale = orig_w / calib_w  # e.g. 3840/1920 = 2.0

    # Output at half resolution
    out_w, out_h = orig_w // 2, orig_h // 2

    print(f"  Camera: {cam_id} | {orig_w}x{orig_h} @ {source_fps:.0f}fps")
    print(f"  step={step} → ~{expected} frames | output: {out_w}x{out_h}")
    print(f"  video_scale: {video_scale:.2f} (K at {calib_w}px, video at {orig_w}px)")
    print(f"  sync_offset: {sync_offset:.4f}s")
    print(f"  {skeleton_data['total_frames']} 3D skeleton frames loaded\n")

    # Skeleton timestamp lookup
    skel_timestamps = np.array([f["timestamp_s"] for f in skeleton_data["frames"]])
    skel_frames = skeleton_data["frames"]

    output_dir = ROOT / cfg["output_dir"] / "gaze"
    output_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_dir / "gaze_overlay.mp4"), fourcc, target_fps, (out_w, out_h)
    )

    # Process frames
    results = []
    frame_count = 0
    valid_pose_count = 0
    frame_pos = 0
    t0 = time.time()

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        if frame_pos % step == 0:
            synced_ts = frame_pos / source_fps + sync_offset

            # Find nearest skeleton frame
            skeleton_frame = None
            if len(skel_timestamps) > 0:
                nearest_idx = np.argmin(np.abs(skel_timestamps - synced_ts))
                if abs(skel_timestamps[nearest_idx] - synced_ts) < 0.1:
                    skeleton_frame = skel_frames[nearest_idx]

            entry = {
                "frame_index": frame_pos,
                "timestamp_s": round(frame_pos / source_fps, 4),
                "synced_ts": round(synced_ts, 4),
                "head_pose_valid": False,
            }

            if skeleton_frame is not None:
                pts = extract_points(skeleton_frame)
                head_pose = compute_head_pose_3d(pts)

                if head_pose is not None:
                    valid_pose_count += 1
                    entry.update({
                        "head_pose_valid": True,
                        "yaw_deg": head_pose["yaw_deg"],
                        "pitch_deg": head_pose["pitch_deg"],
                        "roll_deg": head_pose["roll_deg"],
                    })

                    # Project and draw gaze arrow
                    nose_px, gaze_px = project_gaze_arrow(
                        pts, head_pose, K, dist, R_cam, T_cam,
                        video_scale=video_scale, arrow_len_m=0.3,
                    )
                    if nose_px is not None and gaze_px is not None:
                        cv2.arrowedLine(bgr, nose_px, gaze_px,
                                        (0, 0, 255), max(3, int(video_scale * 3)),
                                        tipLength=0.3)

                    # Text at video scale
                    font_scale = max(1.0, video_scale * 0.6)
                    thickness = max(2, int(video_scale * 2))
                    cv2.putText(bgr,
                                f"Yaw:{head_pose['yaw_deg']:.0f}  "
                                f"Pitch:{head_pose['pitch_deg']:.0f}  "
                                f"Roll:{head_pose['roll_deg']:.0f}",
                                (int(30 * video_scale), int(80 * video_scale)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (0, 0, 255), thickness)

            results.append(entry)
            small = cv2.resize(bgr, (out_w, out_h))
            out.write(small)
            frame_count += 1

            if frame_count % 100 == 0:
                elapsed = time.time() - t0
                fps = frame_count / max(elapsed, 0.01)
                remaining = (expected - frame_count) / max(fps, 0.01)
                pct = 100 * frame_count // expected
                print(f"  {frame_count}/{expected} ({pct}%) "
                      f"| {valid_pose_count} valid | {fps:.1f} fr/s "
                      f"| ~{remaining:.0f}s left", flush=True)

        frame_pos += 1

    cap.release()
    out.release()

    elapsed = time.time() - t0
    print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s")
    print(f"Head pose: {valid_pose_count}/{frame_count} valid "
          f"({100*valid_pose_count//max(frame_count,1)}%)")
    print(f"Overlay: {output_dir/'gaze_overlay.mp4'}")

    # Save JSON
    output = {
        "method": "3d_skeleton_head_orientation",
        "camera": cam_id,
        "coordinate_system": "cam01_origin",
        "total_frames": len(results),
        "valid_poses": valid_pose_count,
        "validation_rate": round(valid_pose_count / max(len(results), 1), 4),
        "frames": results,
    }
    json_path = output_dir / "gaze.json"
    with open(json_path, "w") as f:
        json.dump(output, f)
    print(f"JSON: {json_path}")
    print(f"Total time: {time.time()-t_start:.1f}s")
    print("\nDone!")


if __name__ == "__main__":
    main()
