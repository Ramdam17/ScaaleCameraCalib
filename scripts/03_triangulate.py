"""
03_triangulate.py — 3D skeleton reconstruction from multi-view 2D poses.

Steps:
  1. Load calibration data (projection matrices) and 2D poses
  2. Align frames across cameras by synchronized timestamp
  3. For each aligned frame, triangulate each landmark using DLT
  4. Filter by reprojection error and save 3D skeleton data

Usage:
  python scripts/03_triangulate.py
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def load_config() -> dict:
    with open(ROOT / "config.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_calibration() -> dict:
    """Load calibration.json and rebuild numpy arrays."""
    with open(ROOT / "output" / "calibration" / "calibration.json") as f:
        data = json.load(f)

    cameras = {}
    for cam_id, cam in data["cameras"].items():
        cameras[cam_id] = {
            "K": np.array(cam["camera_matrix"]),
            "dist": np.array(cam["dist_coeffs"]),
            "R": np.array(cam["R"]),
            "T": np.array(cam["T"]),
            "P": np.array(cam["projection_matrix"]),
            "image_size": cam["image_size"],
        }
    return cameras


def load_poses(cam_id: str) -> dict:
    with open(ROOT / "output" / "poses" / f"{cam_id}_poses.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Frame alignment
# ---------------------------------------------------------------------------


def align_frames(all_poses: dict[str, dict], tolerance_s: float = 0.05) -> list[dict]:
    """Match frames across cameras by synced timestamp.

    Groups frames from different cameras that fall within tolerance_s of each other.
    Returns list of frame groups, each containing camera data for that instant.
    """
    # Build (synced_ts, cam_id, frame_data) list for all cameras
    entries = []
    for cam_id, pose_data in all_poses.items():
        for frame in pose_data["frames"]:
            entries.append((frame["synced_timestamp_s"], cam_id, frame))

    # Sort by timestamp
    entries.sort(key=lambda x: x[0])

    # Group nearby timestamps
    groups = []
    used = set()

    for i, (ts, cam_id, frame) in enumerate(entries):
        if i in used:
            continue

        group = {"timestamp_s": ts, "cameras": {cam_id: frame}}
        used.add(i)

        # Find other cameras within tolerance
        for j in range(i + 1, len(entries)):
            if j in used:
                continue
            ts_j, cam_j, frame_j = entries[j]
            if ts_j - ts > tolerance_s:
                break
            if cam_j not in group["cameras"]:
                group["cameras"][cam_j] = frame_j
                used.add(j)

        groups.append(group)

    return groups


# ---------------------------------------------------------------------------
# Triangulation (DLT)
# ---------------------------------------------------------------------------


def triangulate_point_dlt(observations: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, float]:
    """Triangulate a 3D point from 2+ observations using DLT.

    Args:
        observations: list of (point_2d, projection_matrix) tuples
            point_2d: shape (2,) in pixels (undistorted)
            projection_matrix: shape (3, 4)

    Returns:
        (point_3d, reprojection_error)
    """
    A = []
    for pt, P in observations:
        x, y = pt
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)

    # Solve Ax = 0 via SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]  # dehomogenize

    # Reprojection error
    errors = []
    for pt, P in observations:
        projected = P @ np.append(X, 1.0)
        projected = projected[:2] / projected[2]
        errors.append(np.linalg.norm(projected - pt))

    return X, float(np.mean(errors))


def triangulate_frame(
    frame_group: dict,
    cameras: dict,
    confidence_threshold: float,
    max_reproj_error: float,
    scale_factors: dict | None = None,
) -> dict | None:
    """Triangulate all landmarks for one frame.

    Single-person mode: take person_index=0 from each camera.
    Batch-undistorts all 33 landmarks per camera in one call (33× fewer Python calls).

    scale_factors: per-camera (sx, sy) to convert pose pixel coordinates to the
    calibration coordinate system (e.g. 0.5 if poses were saved at 4K but K is for 1080p).
    """
    cam_ids = list(frame_group["cameras"].keys())

    # Get the single person's landmarks from each camera (if detected)
    per_cam_landmarks = {}
    for cam_id in cam_ids:
        frame_data = frame_group["cameras"][cam_id]
        if frame_data["num_poses"] > 0:
            per_cam_landmarks[cam_id] = frame_data["poses"][0]["landmarks"]

    if len(per_cam_landmarks) < 2:
        return None  # need at least 2 cameras

    # Batch-undistort all 33 landmarks per camera (1 call instead of 33)
    undistorted = {}
    for cam_id, landmarks in per_cam_landmarks.items():
        K = cameras[cam_id]["K"]
        dist = cameras[cam_id]["dist"]
        sx, sy = (scale_factors[cam_id] if scale_factors and cam_id in scale_factors
                  else (1.0, 1.0))
        pts = np.array([[lm["x"] * sx, lm["y"] * sy] for lm in landmarks], dtype=np.float64)
        undistorted[cam_id] = cv2.undistortPoints(
            pts.reshape(-1, 1, 2), K, dist, P=K
        ).reshape(-1, 2)

    # Triangulate each of the 33 landmarks
    landmarks_3d = []
    for lm_idx in range(33):
        observations = []
        for cam_id, landmarks in per_cam_landmarks.items():
            lm = landmarks[lm_idx]
            if lm["visibility"] < confidence_threshold:
                continue
            observations.append((undistorted[cam_id][lm_idx], cameras[cam_id]["P"]))

        if len(observations) < 2:
            landmarks_3d.append({
                "id": lm_idx,
                "name": LANDMARK_NAMES[lm_idx],
                "x": None, "y": None, "z": None,
                "reprojection_error": None,
                "num_cameras": 0,
            })
            continue

        point_3d, reproj_err = triangulate_point_dlt(observations)

        if reproj_err > max_reproj_error:
            landmarks_3d.append({
                "id": lm_idx,
                "name": LANDMARK_NAMES[lm_idx],
                "x": None, "y": None, "z": None,
                "reprojection_error": round(reproj_err, 2),
                "num_cameras": len(observations),
            })
        else:
            landmarks_3d.append({
                "id": lm_idx,
                "name": LANDMARK_NAMES[lm_idx],
                "x": round(float(point_3d[0]), 4),
                "y": round(float(point_3d[1]), 4),
                "z": round(float(point_3d[2]), 4),
                "reprojection_error": round(reproj_err, 2),
                "num_cameras": len(observations),
            })

    valid_count = sum(1 for lm in landmarks_3d if lm["x"] is not None)
    if valid_count == 0:
        return None

    return {
        "timestamp_s": frame_group["timestamp_s"],
        "num_cameras_used": len(per_cam_landmarks),
        "landmarks": landmarks_3d,
        "num_valid_landmarks": valid_count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = load_config()
    cam_ids = sorted(cfg["cameras"].keys())
    tri_cfg = cfg["triangulation"]

    t_start = time.time()
    print("=== 3D Triangulation ===\n")

    # Load data
    t0 = time.time()
    cameras = load_calibration()
    all_poses = {cam_id: load_poses(cam_id) for cam_id in cam_ids}
    print(f"Loaded calibration + poses in {time.time()-t0:.1f}s")
    print(f"  {len(cameras)} cameras calibrated")
    for cam_id, poses in all_poses.items():
        n_detected = sum(1 for f in poses["frames"] if f["num_poses"] > 0)
        print(f"  {cam_id}: {n_detected}/{len(poses['frames'])} frames with detections")

    # Compute scale factors: pose landmarks may be at a different resolution than K
    # (e.g. poses at 3840x2160 but K calibrated for 1920x1080)
    scale_factors = {}
    for cam_id in cam_ids:
        pose_w, pose_h = all_poses[cam_id]["image_size"]
        calib_w, calib_h = cameras[cam_id]["image_size"]
        sx, sy = calib_w / pose_w, calib_h / pose_h
        scale_factors[cam_id] = (sx, sy)
        if abs(sx - 1.0) > 0.01:
            print(f"  Scale correction {cam_id}: {pose_w}x{pose_h} → {calib_w}x{calib_h} "
                  f"(sx={sx:.3f}, sy={sy:.3f})")

    # Align frames
    t0 = time.time()
    frame_groups = align_frames(all_poses)
    multi_cam = [g for g in frame_groups if len(g["cameras"]) >= 2]
    by_ncam = {n: sum(1 for g in multi_cam if len(g["cameras"]) == n) for n in range(2, len(cam_ids)+1)}
    print(f"\nFrame alignment in {time.time()-t0:.2f}s: "
          f"{len(frame_groups)} groups total, {len(multi_cam)} with 2+ cameras {by_ncam}")

    # Triangulate
    frames_3d = []
    t_tri = time.time()
    for i, fg in enumerate(multi_cam):
        result = triangulate_frame(
            fg, cameras,
            confidence_threshold=tri_cfg["confidence_threshold"],
            max_reproj_error=tri_cfg["max_reprojection_error_px"],
            scale_factors=scale_factors,
        )
        if result is not None:
            frames_3d.append(result)
        if (i + 1) % 100 == 0 or (i + 1) == len(multi_cam):
            elapsed = time.time() - t_tri
            fps = (i + 1) / max(elapsed, 0.01)
            remaining = (len(multi_cam) - i - 1) / max(fps, 0.01)
            print(f"  {i+1}/{len(multi_cam)} ({100*(i+1)//len(multi_cam)}%) "
                  f"| {len(frames_3d)} valid | {fps:.0f} fr/s | ~{remaining:.0f}s left")

    print(f"\nTriangulated {len(frames_3d)} frames in {time.time()-t_tri:.1f}s")

    # Stats
    if frames_3d:
        avg_landmarks = np.mean([f["num_valid_landmarks"] for f in frames_3d])
        all_errors = [
            lm["reprojection_error"]
            for f in frames_3d for lm in f["landmarks"]
            if lm["reprojection_error"] is not None and lm["x"] is not None
        ]
        avg_error = np.mean(all_errors) if all_errors else 0
        print(f"  Avg valid landmarks per frame: {avg_landmarks:.1f} / 33")
        print(f"  Avg reprojection error: {avg_error:.2f} px")

    # Save
    output = {
        "coordinate_system": "cam01_origin",
        "units": "meters",
        "fps": cfg["pose_detection"]["subsample_fps"],
        "num_persons": 1,
        "landmark_names": LANDMARK_NAMES,
        "total_frames": len(frames_3d),
        "frames": frames_3d,
    }

    output_dir = ROOT / cfg["output_dir"] / "skeleton3d"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "skeleton_3d.json"
    with open(output_path, "w") as f:
        json.dump(output, f)
    print(f"\nSaved to {output_path}")
    print(f"Total time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
