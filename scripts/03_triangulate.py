"""
03_triangulate.py — 3D skeleton reconstruction from multi-view 2D poses.

Supports 1 or N persons (driven by config.json → pose_detection.num_poses).

For multi-person, cross-camera person identity is resolved by exhaustive
assignment search: for each frame, all 2^(K-1) camera-assignment combinations
are scored by triangulation reprojection error on proxy landmarks, and the
best assignment is used to triangulate all 33 landmarks per person.

Steps:
  1. Load calibration data (projection matrices) and 2D poses
  2. Align frames across cameras by synchronized timestamp
  3. For each aligned frame:
     a. Undistort all persons' landmarks per camera
     b. Find best cross-camera person assignment (minimize reprojection error)
     c. Triangulate all 33 landmarks per person with the winning assignment
  4. Filter by reprojection error and save 3D skeleton data

Usage:
  python scripts/03_triangulate.py
"""

import json
import time
from itertools import product as iter_product
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

# Landmarks used to score cross-camera assignments (robust, large body parts)
PROXY_LM_INDICES = [0, 11, 12, 23, 24]  # nose, shoulders, hips


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
    """Match frames across cameras by synced timestamp."""
    entries = []
    for cam_id, pose_data in all_poses.items():
        for frame in pose_data["frames"]:
            entries.append((frame["synced_timestamp_s"], cam_id, frame))

    entries.sort(key=lambda x: x[0])

    groups = []
    used = set()
    for i, (ts, cam_id, frame) in enumerate(entries):
        if i in used:
            continue
        group = {"timestamp_s": ts, "cameras": {cam_id: frame}}
        used.add(i)
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


def triangulate_point_dlt(
    observations: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, float]:
    """Triangulate a 3D point from 2+ (point_2d, projection_matrix) pairs via DLT."""
    A = []
    for pt, P in observations:
        x, y = pt
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]

    errors = []
    for pt, P in observations:
        projected = P @ np.append(X, 1.0)
        projected = projected[:2] / projected[2]
        errors.append(np.linalg.norm(projected - pt))

    return X, float(np.mean(errors))


# ---------------------------------------------------------------------------
# Multi-person cross-camera matching
# ---------------------------------------------------------------------------


def undistort_all_persons(
    per_cam_detections: dict[str, list],
    cameras: dict,
    scale_factors: dict,
) -> dict[str, list[np.ndarray]]:
    """Undistort landmarks for every detected person in every camera.

    Returns: {cam_id: [ndarray(33, 2) for person 0, ndarray(33, 2) for person 1, ...]}
    """
    result = {}
    for cam_id, persons_lm in per_cam_detections.items():
        K = cameras[cam_id]["K"]
        dist = cameras[cam_id]["dist"]
        sx, sy = scale_factors.get(cam_id, (1.0, 1.0))

        undist_list = []
        for landmarks in persons_lm:
            pts = np.array(
                [[lm["x"] * sx, lm["y"] * sy] for lm in landmarks], dtype=np.float64
            )
            undistorted = cv2.undistortPoints(
                pts.reshape(-1, 1, 2), K, dist, P=K
            ).reshape(-1, 2)
            undist_list.append(undistorted)
        result[cam_id] = undist_list

    return result


def _score_assignment(
    assignment: dict[str, list[int]],
    per_cam_undistorted: dict[str, list[np.ndarray]],
    cameras: dict,
) -> float:
    """Score a cross-camera assignment by triangulating proxy landmarks.

    Lower is better. Returns inf if no landmark has 2+ observations.
    """
    # Collect observations per (global_person_id, lm_idx)
    obs: dict[tuple[int, int], list] = {}
    for cam_id, local_to_global in assignment.items():
        P = cameras[cam_id]["P"]
        persons_undist = per_cam_undistorted.get(cam_id, [])
        for local_idx, global_idx in enumerate(local_to_global):
            if local_idx >= len(persons_undist):
                continue
            undist = persons_undist[local_idx]
            for lm_idx in PROXY_LM_INDICES:
                key = (global_idx, lm_idx)
                obs.setdefault(key, []).append((undist[lm_idx], P))

    total_error = 0.0
    count = 0
    for obs_list in obs.values():
        if len(obs_list) < 2:
            continue
        _, err = triangulate_point_dlt(obs_list)
        total_error += err
        count += 1

    return total_error / count if count > 0 else float("inf")


def find_person_assignment(
    per_cam_undistorted: dict[str, list[np.ndarray]],
    cameras: dict,
    num_persons: int,
) -> dict[str, list[int]]:
    """Find the best cross-camera person assignment.

    Fixes the reference camera's ordering and tries all 2^(K-1) combinations
    for the remaining cameras, picking the one with lowest reprojection error.

    Returns: {cam_id: [global_person_id for local_0, global_person_id for local_1, ...]}
    """
    cam_ids = list(per_cam_undistorted.keys())
    if not cam_ids:
        return {}

    ref_cam = cam_ids[0]
    ref_n = len(per_cam_undistorted[ref_cam])
    ref_assignment = list(range(min(ref_n, num_persons)))

    other_cams = cam_ids[1:]

    def candidates(cam_id: str) -> list[list[int]]:
        n = len(per_cam_undistorted[cam_id])
        if n == 0:
            return [[]]
        elif n == 1:
            # Single detection: try assigning it to each global person
            return [[gp] for gp in range(num_persons)]
        else:
            # Two detections: try both orderings
            base = list(range(num_persons))
            return [base, list(reversed(base))]

    best_assignment: dict[str, list[int]] = {ref_cam: ref_assignment}
    best_error = float("inf")

    for combo in iter_product(*[candidates(c) for c in other_cams]):
        candidate = {ref_cam: ref_assignment, **dict(zip(other_cams, combo))}
        error = _score_assignment(candidate, per_cam_undistorted, cameras)
        if error < best_error:
            best_error = error
            best_assignment = candidate

    return best_assignment


# ---------------------------------------------------------------------------
# Per-person triangulation
# ---------------------------------------------------------------------------


def triangulate_person(
    global_person_id: int,
    assignment: dict[str, list[int]],
    per_cam_undistorted: dict[str, list[np.ndarray]],
    per_cam_detections: dict[str, list],
    cameras: dict,
    confidence_threshold: float,
    max_reproj_error: float,
) -> tuple[list[dict], int]:
    """Triangulate all 33 landmarks for one global person given a camera assignment."""
    landmarks_3d = []

    for lm_idx in range(33):
        observations = []

        for cam_id, local_to_global in assignment.items():
            if global_person_id not in local_to_global:
                continue
            local_idx = local_to_global.index(global_person_id)

            persons_lm = per_cam_detections.get(cam_id, [])
            persons_undist = per_cam_undistorted.get(cam_id, [])
            if local_idx >= len(persons_lm) or local_idx >= len(persons_undist):
                continue

            lm = persons_lm[local_idx][lm_idx]
            if lm["visibility"] < confidence_threshold:
                continue

            observations.append((persons_undist[local_idx][lm_idx], cameras[cam_id]["P"]))

        if len(observations) < 2:
            landmarks_3d.append({
                "id": lm_idx, "name": LANDMARK_NAMES[lm_idx],
                "x": None, "y": None, "z": None,
                "reprojection_error": None, "num_cameras": 0,
            })
            continue

        point_3d, reproj_err = triangulate_point_dlt(observations)

        if reproj_err > max_reproj_error:
            landmarks_3d.append({
                "id": lm_idx, "name": LANDMARK_NAMES[lm_idx],
                "x": None, "y": None, "z": None,
                "reprojection_error": round(reproj_err, 2),
                "num_cameras": len(observations),
            })
        else:
            landmarks_3d.append({
                "id": lm_idx, "name": LANDMARK_NAMES[lm_idx],
                "x": round(float(point_3d[0]), 4),
                "y": round(float(point_3d[1]), 4),
                "z": round(float(point_3d[2]), 4),
                "reprojection_error": round(reproj_err, 2),
                "num_cameras": len(observations),
            })

    valid_count = sum(1 for lm in landmarks_3d if lm["x"] is not None)
    return landmarks_3d, valid_count


# ---------------------------------------------------------------------------
# Frame-level triangulation
# ---------------------------------------------------------------------------


def triangulate_frame(
    frame_group: dict,
    cameras: dict,
    confidence_threshold: float,
    max_reproj_error: float,
    num_persons: int,
    scale_factors: dict | None = None,
) -> dict | None:
    """Triangulate all persons' landmarks for one multi-camera frame group.

    Returns a frame dict with a 'persons' list, one entry per global person.
    Returns None if no valid 3D points could be computed.
    """
    cam_ids = list(frame_group["cameras"].keys())

    # Collect raw detections per camera: {cam_id: [[lm, ...] per person]}
    per_cam_detections: dict[str, list] = {}
    for cam_id in cam_ids:
        frame_data = frame_group["cameras"][cam_id]
        if frame_data["num_poses"] > 0:
            per_cam_detections[cam_id] = [
                pose["landmarks"] for pose in frame_data["poses"]
            ]

    if len(per_cam_detections) < 2:
        return None

    # Undistort in bulk (1 undistortPoints call per person per camera)
    per_cam_undistorted = undistort_all_persons(
        per_cam_detections, cameras, scale_factors or {}
    )

    # Find best cross-camera person assignment
    assignment = find_person_assignment(per_cam_undistorted, cameras, num_persons)

    # Triangulate each global person
    persons_3d = []
    for global_person_id in range(num_persons):
        # Check if this person appears in at least 2 cameras
        cams_with_person = sum(
            1 for cam_id, local_to_global in assignment.items()
            if global_person_id in local_to_global
            and local_to_global.index(global_person_id) < len(per_cam_detections.get(cam_id, []))
        )
        if cams_with_person < 2:
            continue

        landmarks_3d, valid_count = triangulate_person(
            global_person_id, assignment,
            per_cam_undistorted, per_cam_detections,
            cameras, confidence_threshold, max_reproj_error,
        )

        if valid_count > 0:
            persons_3d.append({
                "person_id": global_person_id,
                "num_valid_landmarks": valid_count,
                "landmarks": landmarks_3d,
            })

    if not persons_3d:
        return None

    return {
        "timestamp_s": frame_group["timestamp_s"],
        "num_cameras_used": len(per_cam_detections),
        "num_persons": len(persons_3d),
        "persons": persons_3d,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = load_config()
    cam_ids = sorted(cfg["cameras"].keys())
    tri_cfg = cfg["triangulation"]
    num_persons = cfg["pose_detection"]["num_poses"]

    t_start = time.time()
    print(f"=== 3D Triangulation ({num_persons} person(s)) ===\n")

    # Load data
    t0 = time.time()
    cameras = load_calibration()
    all_poses = {cam_id: load_poses(cam_id) for cam_id in cam_ids}
    print(f"Loaded calibration + poses in {time.time()-t0:.1f}s")
    print(f"  {len(cameras)} cameras calibrated")
    for cam_id, poses in all_poses.items():
        n_detected = sum(1 for f in poses["frames"] if f["num_poses"] > 0)
        n_full = sum(1 for f in poses["frames"] if f["num_poses"] >= num_persons)
        print(f"  {cam_id}: {n_detected}/{len(poses['frames'])} frames with detections "
              f"({n_full} with {num_persons} person(s))")

    # Compute scale factors: pose landmarks may be at a different resolution than K
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
    by_ncam = {n: sum(1 for g in multi_cam if len(g["cameras"]) == n)
               for n in range(2, len(cam_ids) + 1)}
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
            num_persons=num_persons,
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

    # Stats per person
    if frames_3d:
        for person_id in range(num_persons):
            person_frames = [
                p for f in frames_3d
                for p in f["persons"] if p["person_id"] == person_id
            ]
            if person_frames:
                avg_lm = np.mean([p["num_valid_landmarks"] for p in person_frames])
                all_errors = [
                    lm["reprojection_error"]
                    for p in person_frames
                    for lm in p["landmarks"]
                    if lm["reprojection_error"] is not None and lm["x"] is not None
                ]
                avg_err = np.mean(all_errors) if all_errors else 0
                print(f"  Person {person_id}: {len(person_frames)} frames, "
                      f"{avg_lm:.1f}/33 valid landmarks, {avg_err:.2f}px reprojection error")

    # Save
    output = {
        "coordinate_system": "cam01_origin",
        "units": "meters",
        "fps": cfg["pose_detection"]["subsample_fps"],
        "num_persons": num_persons,
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
