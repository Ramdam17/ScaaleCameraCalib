"""
01_calibrate.py — Multi-camera calibration using ChArUco board.

Steps:
  1. Extract frames from calibration videos (subsampled)
  2. Detect ChArUco corners in each frame
  3. Calibrate intrinsics per camera
  4. Find common frames across camera pairs for stereo calibration
  5. Build unified multi-camera system (cam01 = world origin)

Usage:
  python scripts/01_calibrate.py
"""

import json
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import numpy as np

# Force unbuffered stdout so progress prints appear immediately
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config.json") as f:
        return json.load(f)


def log(msg: str = ""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# ChArUco board setup
# ---------------------------------------------------------------------------


def create_charuco_detector(cfg: dict):
    """Create the CharucoBoard and CharucoDetector from config."""
    dict_name = cfg["charuco"]["dictionary"]  # e.g. "DICT_6X6_250"
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))

    board = cv2.aruco.CharucoBoard(
        size=(cfg["charuco"]["squares_x"], cfg["charuco"]["squares_y"]),
        squareLength=cfg["charuco"]["square_length_m"],
        markerLength=cfg["charuco"]["marker_length_m"],
        dictionary=dictionary,
    )
    board.setLegacyPattern(True)

    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True

    detector_params = cv2.aruco.DetectorParameters()
    detector_params.minMarkerPerimeterRate = 0.01
    detector_params.adaptiveThreshWinSizeMin = 3
    detector_params.adaptiveThreshWinSizeMax = 53
    detector_params.adaptiveThreshWinSizeStep = 4

    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    return board, detector


# ---------------------------------------------------------------------------
# Frame extraction + detection (streaming, memory-safe)
# ---------------------------------------------------------------------------


def extract_and_detect(
    video_path: str, target_fps: float, detector, min_corners: int,
    cam_id: str = "", keep_bgr_every: int = 100, downscale_width: int = 0,
) -> tuple[list[dict], int, float, tuple[int, int]]:
    """Extract frames and detect ChArUco in a single streaming pass.

    Reads one frame at a time — never holds more than one 4K frame in memory.
    If downscale_width > 0, each frame is resized before detection (faster, less RAM).
    Corner coordinates are in the downscaled space.
    Returns (detections, total_frames_sampled, source_fps, proc_image_size).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(source_fps / target_fps)))
    expected = total_frames // step
    duration_s = total_frames / source_fps

    # Compute processing resolution
    if downscale_width > 0 and w > downscale_width:
        proc_w = downscale_width
        proc_h = int(h * downscale_width / w)
    else:
        proc_w, proc_h = w, h

    log(f"  Video: {w}x{h} @ {source_fps:.0f}fps, {total_frames} frames ({duration_s:.1f}s)")
    log(f"  Processing at: {proc_w}x{proc_h} (downscale={downscale_width > 0 and w > downscale_width})")
    log(f"  Sampling: step={step} → ~{expected} frames to process")

    detections = []
    n_sampled = 0
    frame_pos = 0
    t0 = time.time()

    # Sequential read (no seeking) — much faster on compressed video
    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        # Only process every `step`-th frame
        if frame_pos % step == 0:
            n_sampled += 1

            if proc_w != w:
                bgr = cv2.resize(bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

            if charuco_corners is not None and len(charuco_corners) >= min_corners:
                det = {
                    "frame_idx": frame_pos,
                    "charuco_corners": charuco_corners,
                    "charuco_ids": charuco_ids,
                    "marker_corners": marker_corners,
                    "marker_ids": marker_ids,
                }
                if keep_bgr_every > 0 and len(detections) % keep_bgr_every == 0:
                    det["bgr"] = bgr
                detections.append(det)

            if n_sampled % 100 == 0:
                elapsed = time.time() - t0
                fps_proc = n_sampled / max(elapsed, 0.01)
                remaining = (expected - n_sampled) / max(fps_proc, 0.01)
                pct = 100 * n_sampled / max(1, expected)
                log(f"  [{cam_id}] {n_sampled}/{expected} ({pct:.0f}%) "
                    f"| {len(detections)} detected | {fps_proc:.1f} fr/s | ~{remaining:.0f}s left")

        frame_pos += 1

    elapsed = time.time() - t0
    cap.release()
    log(f"  [{cam_id}] DONE: {len(detections)}/{n_sampled} frames with ChArUco "
        f"({100*len(detections)/max(1,n_sampled):.0f}%) in {elapsed:.1f}s")
    return detections, n_sampled, source_fps, (proc_w, proc_h)


# ---------------------------------------------------------------------------
# Intrinsic calibration
# ---------------------------------------------------------------------------


def calibrate_intrinsics(board, detections: list[dict], image_size: tuple, max_frames: int = 150) -> dict:
    """Calibrate a single camera from its ChArUco detections.

    Caps at max_frames (uniformly subsampled) — accuracy plateaus past ~150 frames.
    """
    all_obj = []
    all_img = []

    for det in detections:
        obj_pts, img_pts = board.matchImagePoints(det["charuco_corners"], det["charuco_ids"])
        if obj_pts is not None and len(obj_pts) >= 4:
            all_obj.append(obj_pts)
            all_img.append(img_pts)

    if len(all_obj) < 5:
        raise RuntimeError(f"Only {len(all_obj)} usable frames — need at least 5 for calibration")

    n_total = len(all_obj)
    if n_total > max_frames:
        indices = np.linspace(0, n_total - 1, max_frames, dtype=int)
        all_obj = [all_obj[i] for i in indices]
        all_img = [all_img[i] for i in indices]
        log(f"  Subsampled {n_total} → {max_frames} frames for calibration")

    log(f"  Calibrating intrinsics from {len(all_obj)} frames...")
    t0 = time.time()
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_obj,
        all_img,
        image_size,
        None,
        None,
        flags=cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_ASPECT_RATIO,
    )
    elapsed = time.time() - t0
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    log(f"  Intrinsics done in {elapsed:.1f}s: RMS={rms:.4f}px, fx={fx:.1f}, fy={fy:.1f}")

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "rms": rms,
        "num_frames": len(all_obj),
        "image_size": image_size,
    }


# ---------------------------------------------------------------------------
# Stereo calibration
# ---------------------------------------------------------------------------


def find_common_frames(
    det_a: list[dict], det_b: list[dict], source_fps: float, max_frame_diff: int = 3
) -> list[tuple[dict, dict]]:
    """Find frame pairs where the board is detected in both cameras at ~same time."""
    pairs = []
    b_idx = 0
    for da in det_a:
        while b_idx < len(det_b) and det_b[b_idx]["frame_idx"] < da["frame_idx"] - max_frame_diff:
            b_idx += 1
        for j in range(b_idx, len(det_b)):
            db = det_b[j]
            if db["frame_idx"] > da["frame_idx"] + max_frame_diff:
                break
            ids_a = set(da["charuco_ids"].flatten())
            ids_b = set(db["charuco_ids"].flatten())
            common_ids = ids_a & ids_b
            if len(common_ids) >= 6:
                pairs.append((da, db))
                break
    return pairs


def calibrate_stereo(
    board,
    pairs: list[tuple[dict, dict]],
    intrinsics_a: dict,
    intrinsics_b: dict,
) -> dict | None:
    """Stereo calibration from common frame pairs."""
    if len(pairs) < 5:
        log(f"  WARNING: only {len(pairs)} common frames — stereo may be unreliable")
        if len(pairs) < 3:
            return None

    all_obj = []
    all_img_a = []
    all_img_b = []

    for da, db in pairs:
        ids_a = set(da["charuco_ids"].flatten())
        ids_b = set(db["charuco_ids"].flatten())
        common_ids = sorted(ids_a & ids_b)

        if len(common_ids) < 6:
            continue

        mask_a = np.isin(da["charuco_ids"].flatten(), common_ids)
        mask_b = np.isin(db["charuco_ids"].flatten(), common_ids)

        corners_a = da["charuco_corners"][mask_a]
        ids_a_filtered = da["charuco_ids"][mask_a]
        corners_b = db["charuco_corners"][mask_b]

        obj_pts, img_pts_a = board.matchImagePoints(corners_a, ids_a_filtered)
        _, img_pts_b = board.matchImagePoints(corners_b, db["charuco_ids"][mask_b])

        if obj_pts is not None and len(obj_pts) >= 6:
            all_obj.append(obj_pts)
            all_img_a.append(img_pts_a)
            all_img_b.append(img_pts_b)

    if len(all_obj) < 3:
        return None

    log(f"  Running stereoCalibrate with {len(all_obj)} frame pairs...")
    image_size = intrinsics_a["image_size"]
    t0 = time.time()
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        all_obj,
        all_img_a,
        all_img_b,
        intrinsics_a["camera_matrix"],
        intrinsics_a["dist_coeffs"],
        intrinsics_b["camera_matrix"],
        intrinsics_b["dist_coeffs"],
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-6),
    )
    elapsed = time.time() - t0
    t_vec = T.flatten()
    log(f"  stereoCalibrate done in {elapsed:.1f}s: RMS={rms:.4f}px")
    log(f"  Translation: [{t_vec[0]:.4f}, {t_vec[1]:.4f}, {t_vec[2]:.4f}] m")

    return {
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "rms": rms,
        "num_common_frames": len(all_obj),
    }


# ---------------------------------------------------------------------------
# Fallback: per-camera board pose estimation
# ---------------------------------------------------------------------------


def estimate_camera_poses_from_board(board, detections: list[dict], intrinsics: dict) -> list[dict]:
    """Estimate (rvec, tvec) of the board relative to the camera for each frame."""
    poses = []
    for det in detections:
        obj_pts, img_pts = board.matchImagePoints(det["charuco_corners"], det["charuco_ids"])
        if obj_pts is None or len(obj_pts) < 6:
            continue
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, intrinsics["camera_matrix"], intrinsics["dist_coeffs"]
        )
        if ok:
            poses.append({"frame_idx": det["frame_idx"], "rvec": rvec, "tvec": tvec})
    return poses


def average_camera_pose(poses: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Average multiple board-relative poses (median, robust to outliers)."""
    rvecs = np.array([p["rvec"].flatten() for p in poses])
    tvecs = np.array([p["tvec"].flatten() for p in poses])
    rvec_median = np.median(rvecs, axis=0).reshape(3, 1)
    tvec_median = np.median(tvecs, axis=0).reshape(3, 1)
    return rvec_median, tvec_median


# ---------------------------------------------------------------------------
# Build multi-camera system
# ---------------------------------------------------------------------------


def _board_pose_extrinsics(
    board, ref_id, cam_id, all_detections, intrinsics
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute relative extrinsics via per-camera board pose (solvePnP)."""
    log(f"  Computing board poses for {ref_id}...")
    poses_ref = estimate_camera_poses_from_board(board, all_detections[ref_id], intrinsics[ref_id])
    log(f"  Computing board poses for {cam_id}...")
    poses_cam = estimate_camera_poses_from_board(board, all_detections[cam_id], intrinsics[cam_id])

    log(f"  Board poses: {len(poses_ref)} ({ref_id}) + {len(poses_cam)} ({cam_id})")

    if not poses_ref or not poses_cam:
        return None

    rvec_ref, tvec_ref = average_camera_pose(poses_ref)
    rvec_cam, tvec_cam = average_camera_pose(poses_cam)
    R_board_ref, _ = cv2.Rodrigues(rvec_ref)
    R_board_cam, _ = cv2.Rodrigues(rvec_cam)
    R = R_board_ref.T @ R_board_cam
    T = R_board_ref.T @ (tvec_cam - tvec_ref)
    return R, T


def _is_stereo_reliable(stereo: dict | None, min_frames: int, max_rms: float) -> bool:
    return (
        stereo is not None
        and stereo["num_common_frames"] >= min_frames
        and stereo["rms"] < max_rms
    )


def build_camera_system(
    intrinsics: dict[str, dict],
    stereo_pairs: dict[str, dict | None],
    board,
    all_detections: dict[str, list[dict]],
    min_stereo_frames: int = 15,
    max_stereo_rms: float = 5.0,
) -> dict:
    """Build unified coordinate system. cam01 is the world origin.

    Strategy per pair (ref→cam):
    1. Direct stereo if reliable (≥min_stereo_frames, RMS<max_stereo_rms)
    2. Transitive stereo via intermediate camera (chain two reliable stereos)
    3. Board-pose fallback (solvePnP per camera, median averaging)
    """
    cam_ids = sorted(intrinsics.keys())
    system = {}

    ref = cam_ids[0]
    system[ref] = _make_camera_entry(intrinsics[ref], np.eye(3), np.zeros((3, 1)))
    log(f"  {ref}: world origin")

    for cam_id in cam_ids[1:]:
        pair_key = f"{ref}_{cam_id}"
        stereo = stereo_pairs.get(pair_key)

        if _is_stereo_reliable(stereo, min_stereo_frames, max_stereo_rms):
            R, T = stereo["R"], stereo["T"]
            log(f"  {pair_key}: DIRECT STEREO (RMS={stereo['rms']:.3f}, {stereo['num_common_frames']} frames)")

        else:
            # Try transitive stereo
            transitive_found = False
            for mid in cam_ids:
                if mid == ref or mid == cam_id:
                    continue
                s_ref_mid = stereo_pairs.get(f"{ref}_{mid}")
                s_mid_cam = stereo_pairs.get(f"{mid}_{cam_id}")
                if (_is_stereo_reliable(s_ref_mid, min_stereo_frames, max_stereo_rms)
                        and _is_stereo_reliable(s_mid_cam, min_stereo_frames, max_stereo_rms)):
                    # X_cam = R_mid_cam @ (R_ref_mid @ X_ref + T_ref_mid) + T_mid_cam
                    R = s_mid_cam["R"] @ s_ref_mid["R"]
                    T = s_mid_cam["R"] @ s_ref_mid["T"] + s_mid_cam["T"]
                    log(f"  {pair_key}: TRANSITIVE via {mid} "
                        f"({ref}→{mid}: RMS={s_ref_mid['rms']:.2f}/{s_ref_mid['num_common_frames']}fr, "
                        f"{mid}→{cam_id}: RMS={s_mid_cam['rms']:.2f}/{s_mid_cam['num_common_frames']}fr)")
                    transitive_found = True
                    break

            if not transitive_found:
                reason = "no stereo result"
                if stereo is not None:
                    reason = f"stereo unreliable (RMS={stereo['rms']:.2f}, {stereo['num_common_frames']} frames)"
                log(f"  {pair_key}: {reason} → BOARD-POSE FALLBACK")

                result = _board_pose_extrinsics(board, ref, cam_id, all_detections, intrinsics)
                if result is not None:
                    R, T = result
                else:
                    log(f"  WARNING: cannot compute extrinsics for {cam_id}")
                    R, T = np.eye(3), np.zeros((3, 1))

        t = T.flatten()
        log(f"  {cam_id} position: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
        system[cam_id] = _make_camera_entry(intrinsics[cam_id], R, T)

    return system


def _make_camera_entry(intrinsics: dict, R: np.ndarray, T: np.ndarray) -> dict:
    K = intrinsics["camera_matrix"]
    P = K @ np.hstack([R, T])
    return {
        "camera_matrix": K,
        "dist_coeffs": intrinsics["dist_coeffs"],
        "R": R,
        "T": T,
        "projection_matrix": P,
        "intrinsic_rms": intrinsics["rms"],
        "num_calibration_frames": intrinsics["num_frames"],
        "image_size": list(intrinsics["image_size"]),
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def save_detection_images(detections: list[dict], cam_id: str, output_dir: Path, max_images: int = 10):
    """Save a sample of frames with detected ChArUco corners drawn."""
    det_dir = output_dir / "detections" / cam_id
    det_dir.mkdir(parents=True, exist_ok=True)

    with_bgr = [d for d in detections if "bgr" in d]
    if not with_bgr:
        log(f"  [{cam_id}] No BGR images stored, skipping detection images")
        return

    step = max(1, len(with_bgr) // max_images)
    saved = 0
    for i in range(0, len(with_bgr), step):
        det = with_bgr[i]
        vis = det["bgr"].copy()
        if det["marker_corners"] is not None:
            cv2.aruco.drawDetectedMarkers(vis, det["marker_corners"], det["marker_ids"])
        if det["charuco_corners"] is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis, det["charuco_corners"], det["charuco_ids"])

        h, w = vis.shape[:2]
        if w > 1920:
            scale = 1920 / w
            vis = cv2.resize(vis, (1920, int(h * scale)))

        fname = det_dir / f"frame_{det['frame_idx']:05d}.jpg"
        cv2.imwrite(str(fname), vis)
        saved += 1

    log(f"  [{cam_id}] Saved {saved} detection images to {det_dir}")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def save_calibration(system: dict, stereo_pairs: dict, cfg: dict, output_path: Path):
    result = {
        "cameras": {},
        "stereo_pairs": {},
        "coordinate_system": "cam01_origin",
        "units": "meters",
        "charuco_config": cfg["charuco"],
    }

    for cam_id, cam in system.items():
        result["cameras"][cam_id] = {
            "camera_matrix": _to_list(cam["camera_matrix"]),
            "dist_coeffs": _to_list(cam["dist_coeffs"]),
            "R": _to_list(cam["R"]),
            "T": _to_list(cam["T"]),
            "projection_matrix": _to_list(cam["projection_matrix"]),
            "intrinsic_rms": cam["intrinsic_rms"],
            "num_calibration_frames": cam["num_calibration_frames"],
            "image_size": cam["image_size"],
        }

    for pair_key, stereo in stereo_pairs.items():
        if stereo is not None:
            result["stereo_pairs"][pair_key] = {
                "rms": stereo["rms"],
                "num_common_frames": stereo["num_common_frames"],
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Calibration saved to {output_path}")


def generate_report(system: dict, stereo_pairs: dict) -> str:
    lines = ["=" * 60, "CALIBRATION REPORT", "=" * 60, ""]

    for cam_id, cam in sorted(system.items()):
        K = cam["camera_matrix"]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        lines.append(f"--- {cam_id} ---")
        lines.append(f"  Focal length:  fx={fx:.1f}  fy={fy:.1f} px")
        lines.append(f"  Principal pt:  cx={cx:.1f}  cy={cy:.1f} px")
        lines.append(f"  Distortion:    {cam['dist_coeffs'].flatten()[:5]}")
        lines.append(f"  Intrinsic RMS: {cam['intrinsic_rms']:.4f} px")
        lines.append(f"  Frames used:   {cam['num_calibration_frames']}")
        if not np.allclose(cam["T"], 0):
            t = cam["T"].flatten()
            lines.append(f"  Translation:   [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
        lines.append("")

    for pair_key, stereo in stereo_pairs.items():
        if stereo is not None:
            quality = "EXCELLENT" if stereo["rms"] < 1.0 else ("GOOD" if stereo["rms"] < 2.0 else "CHECK")
            lines.append(f"Stereo {pair_key}: RMS={stereo['rms']:.4f} px ({quality}), {stereo['num_common_frames']} frames")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------


def _camera_worker(cam_id, video_path, charuco_cfg, target_fps, min_corners,
                   downscale_width, output_dir_str, max_frames, result_queue):
    """Runs in a separate process: detect ChArUco + calibrate intrinsics for one camera."""
    sys.stdout.reconfigure(line_buffering=True)
    board, detector = create_charuco_detector({"charuco": charuco_cfg})
    detections, n_sampled, source_fps, proc_size = extract_and_detect(
        video_path, target_fps, detector, min_corners,
        cam_id=cam_id, downscale_width=downscale_width
    )
    intrinsics = None
    if detections:
        intrinsics = calibrate_intrinsics(board, detections, proc_size, max_frames=max_frames)
        log(f"  [{cam_id}] Intrinsic RMS: {intrinsics['rms']:.4f}px ({intrinsics['num_frames']} frames)")
        save_detection_images(detections, cam_id, Path(output_dir_str))
    else:
        log(f"  [{cam_id}] WARNING: No ChArUco detected!")
    result_queue.put({
        "cam_id": cam_id, "detections": detections,
        "intrinsics": intrinsics, "image_size": proc_size, "source_fps": source_fps,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t_start = time.time()
    cfg = load_config()
    cam_ids = sorted(cfg["cameras"].keys())
    output_dir = ROOT / cfg["output_dir"] / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    board, detector = create_charuco_detector(cfg)
    target_fps = cfg["calibration"]["frame_subsample_fps"]
    min_corners = cfg["calibration"]["min_corners_per_frame"]
    max_frames_per_camera = cfg["calibration"].get("max_frames_per_camera", 150)
    downscale_width = cfg["calibration"].get("downscale_width", 0)

    log(f"Config: {len(cam_ids)} cameras, target_fps={target_fps}, "
        f"min_corners={min_corners}, max_frames={max_frames_per_camera}, "
        f"downscale_width={downscale_width or 'none'}")

    # --- Step 1: Detect per camera (PARALLEL — 3 processes simultaneously) ---
    all_detections: dict[str, list[dict]] = {}
    intrinsics: dict[str, dict] = {}
    image_size = None
    source_fps = None

    log(f"\n{'='*60}")
    log(f"[Step 1] ChArUco Detection — {len(cam_ids)} cameras in parallel")
    log(f"{'='*60}")

    result_queue: Queue = Queue()
    processes = [
        Process(target=_camera_worker, args=(
            cam_id,
            str(ROOT / cfg["cameras"][cam_id]["calibration_video"]),
            cfg["charuco"], target_fps, min_corners, downscale_width,
            str(output_dir), max_frames_per_camera, result_queue,
        ))
        for cam_id in cam_ids
    ]
    for p in processes:
        p.start()
        log(f"  Started process for {cam_ids[processes.index(p)]}")

    # Drain queue BEFORE p.join() to avoid deadlock:
    # large payloads fill the OS pipe buffer, blocking the worker from exiting,
    # while the parent waits in p.join() → deadlock.
    for _ in cam_ids:
        res = result_queue.get()
        all_detections[res["cam_id"]] = res["detections"]
        if res["intrinsics"] is not None:
            intrinsics[res["cam_id"]] = res["intrinsics"]
        if image_size is None and res["image_size"] is not None:
            image_size = res["image_size"]
        if source_fps is None and res["source_fps"] is not None:
            source_fps = res["source_fps"]

    for p in processes:
        p.join()

    if len(intrinsics) < 2:
        log("\nERROR: Need at least 2 calibrated cameras for stereo.")
        sys.exit(1)

    # --- Step 1b: Shared intrinsics ---
    log(f"\n{'='*60}")
    log(f"[Step 1b] Shared Intrinsics (pooling all cameras)")
    log(f"{'='*60}")
    all_obj_pooled = []
    all_img_pooled = []
    for cam_id in cam_ids:
        cam_obj = []
        cam_img = []
        for det in all_detections.get(cam_id, []):
            obj_pts, img_pts = board.matchImagePoints(det["charuco_corners"], det["charuco_ids"])
            if obj_pts is not None and len(obj_pts) >= 4:
                cam_obj.append(obj_pts)
                cam_img.append(img_pts)
        n_total = len(cam_obj)
        if n_total > max_frames_per_camera:
            indices = np.linspace(0, n_total - 1, max_frames_per_camera, dtype=int)
            cam_obj = [cam_obj[i] for i in indices]
            cam_img = [cam_img[i] for i in indices]
            log(f"  {cam_id}: {n_total} → {max_frames_per_camera} frames (subsampled)")
        else:
            log(f"  {cam_id}: {n_total} usable frames")
        all_obj_pooled.extend(cam_obj)
        all_img_pooled.extend(cam_img)

    log(f"  Total pooled frames: {len(all_obj_pooled)}")
    log(f"  Running calibrateCamera on pooled data...")
    t0 = time.time()
    rms_shared, K_shared, dist_shared, _, _ = cv2.calibrateCamera(
        all_obj_pooled,
        all_img_pooled,
        image_size,
        None,
        None,
        flags=cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_ASPECT_RATIO,
    )
    elapsed = time.time() - t0
    log(f"  Done in {elapsed:.1f}s")
    log(f"  Shared intrinsics: RMS={rms_shared:.4f}px, fx=fy={K_shared[0,0]:.1f}, "
        f"cx={K_shared[0,2]:.1f}, cy={K_shared[1,2]:.1f}")
    log(f"  Distortion: {dist_shared.flatten()[:5]}")

    for cam_id in intrinsics:
        intrinsics[cam_id]["camera_matrix"] = K_shared.copy()
        intrinsics[cam_id]["dist_coeffs"] = dist_shared.copy()
        intrinsics[cam_id]["rms"] = rms_shared

    # --- Step 2: Stereo calibration ---
    log(f"\n{'='*60}")
    log(f"[Step 2] Stereo Calibration")
    log(f"{'='*60}")
    stereo_pairs: dict[str, dict | None] = {}

    pairs_to_try = [(cam_ids[0], cam_ids[1]), (cam_ids[0], cam_ids[2])]
    if len(cam_ids) > 2:
        pairs_to_try.append((cam_ids[1], cam_ids[2]))

    for cam_a, cam_b in pairs_to_try:
        if cam_a not in intrinsics or cam_b not in intrinsics:
            continue
        pair_key = f"{cam_a}_{cam_b}"
        log(f"\n  [{pair_key}]")
        log(f"  Searching common frames (max_frame_diff=3)...")
        log(f"  {cam_a}: {len(all_detections[cam_a])} detections, "
            f"{cam_b}: {len(all_detections[cam_b])} detections")
        common = find_common_frames(all_detections[cam_a], all_detections[cam_b],
                                    source_fps)
        log(f"  Found {len(common)} common frame pairs")
        if common:
            log(f"  Frame indices of first 5 pairs: "
                f"{[(da['frame_idx'], db['frame_idx']) for da, db in common[:5]]}")
        stereo = calibrate_stereo(board, common, intrinsics[cam_a], intrinsics[cam_b])
        stereo_pairs[pair_key] = stereo
        if stereo:
            log(f"  Stereo RMS: {stereo['rms']:.4f}px ({stereo['num_common_frames']} frames)")
        else:
            log(f"  Stereo calibration FAILED — will use fallback")

    # --- Step 3: Build system ---
    log(f"\n{'='*60}")
    log(f"[Step 3] Building Camera System")
    log(f"{'='*60}")
    system = build_camera_system(intrinsics, stereo_pairs, board, all_detections)

    # --- Step 4: Save results ---
    log(f"\n{'='*60}")
    log(f"[Step 4] Saving Results")
    log(f"{'='*60}")
    save_calibration(system, stereo_pairs, cfg, output_dir / "calibration.json")

    report = generate_report(system, stereo_pairs)
    log(f"\n{report}")
    with open(output_dir / "calibration_report.txt", "w") as f:
        f.write(report)

    total_time = time.time() - t_start
    log(f"\nTotal calibration time: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
