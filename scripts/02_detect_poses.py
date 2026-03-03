"""
02_detect_poses.py — 2D pose detection with MediaPipe + audio synchronization.

Steps:
  1. Compute audio-based temporal sync offsets between cameras
  2. Detect 2D poses (up to 2 people) per camera using MediaPipe Pose Landmarker
  3. Save per-camera pose data as JSON

Usage:
  python scripts/02_detect_poses.py
"""

import json
import subprocess
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import fftconvolve

# Force unbuffered stdout so progress prints appear immediately in subprocesses
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

# MediaPipe landmark names (33 total)
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
# Audio sync
# ---------------------------------------------------------------------------


def extract_audio_mono(video_path: str, sample_rate: int = 48000) -> np.ndarray:
    """Extract mono audio from video using ffmpeg, return as float32 numpy array."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio


def compute_audio_offset(ref_audio: np.ndarray, target_audio: np.ndarray,
                         sample_rate: int = 48000) -> float:
    """Compute temporal offset between two audio tracks via cross-correlation.

    Returns offset in seconds: positive means target starts AFTER ref.
    """
    # Use first 15 seconds for correlation (enough with music)
    max_samples = 15 * sample_rate
    ref = ref_audio[:max_samples].astype(np.float64)
    target = target_audio[:max_samples].astype(np.float64)

    # Normalize
    ref = ref - ref.mean()
    target = target - target.mean()
    ref_norm = np.sqrt(np.sum(ref**2))
    target_norm = np.sqrt(np.sum(target**2))
    if ref_norm < 1e-10 or target_norm < 1e-10:
        return 0.0

    # Cross-correlation via FFT
    corr = fftconvolve(ref, target[::-1], mode="full")
    peak_idx = np.argmax(np.abs(corr))

    # Convert to offset: peak at len(target)-1 means zero lag
    offset_samples = peak_idx - (len(target) - 1)
    offset_seconds = offset_samples / sample_rate
    return offset_seconds


def compute_sync_offsets(cfg: dict) -> dict:
    """Compute audio sync offsets for all cameras relative to cam01."""
    cam_ids = sorted(cfg["cameras"].keys())
    ref_cam = cam_ids[0]
    sample_rate = 48000

    print(f"Extracting audio from {ref_cam}...")
    ref_audio = extract_audio_mono(str(ROOT / cfg["cameras"][ref_cam]["skeleton_video"]), sample_rate)

    offsets = {ref_cam: 0.0}
    for cam_id in cam_ids[1:]:
        print(f"Extracting audio from {cam_id}...")
        target_audio = extract_audio_mono(
            str(ROOT / cfg["cameras"][cam_id]["skeleton_video"]), sample_rate
        )
        offset = compute_audio_offset(ref_audio, target_audio, sample_rate)
        offsets[cam_id] = round(offset, 4)
        print(f"  {cam_id} offset: {offset:.4f}s")

    return offsets


# ---------------------------------------------------------------------------
# Pose detection
# ---------------------------------------------------------------------------


def create_pose_detector(cfg_pose: dict):
    """Create MediaPipe PoseLandmarker from pose_detection config section."""
    model_path = str(ROOT / cfg_pose["model_path"])
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=cfg_pose["num_poses"],
        min_pose_detection_confidence=cfg_pose["min_detection_confidence"],
        min_tracking_confidence=cfg_pose["min_tracking_confidence"],
        output_segmentation_masks=False,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def detect_poses_in_video(
    landmarker, video_path: str, target_fps: float, sync_offset: float,
    cam_id: str = "", downscale_width: int | None = None,
) -> dict:
    """Detect 2D poses in a video at target_fps using sequential read (no seeking).

    Sequential read avoids H.264 keyframe-seek overhead (4-5× faster than cap.set).
    Landmarks are normalized [0,1] and scaled back to original resolution.
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

    print(f"  [{cam_id}] {w}x{h} @ {source_fps:.0f}fps, {total_frames} frames ({duration_s:.1f}s), "
          f"step={step} → ~{expected} frames", flush=True)

    # Resolution used for detection (= calibration coordinate system)
    if downscale_width and w > downscale_width:
        lm_w = downscale_width
        lm_h = int(h * downscale_width / w)
    else:
        lm_w, lm_h = w, h

    frames_data = []
    processed = 0
    frame_pos = 0
    t0 = time.time()

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        if frame_pos % step == 0:
            # Optionally downscale for faster inference
            if downscale_width and w > downscale_width:
                scale = downscale_width / w
                small = cv2.resize(bgr, (downscale_width, int(h * scale)))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamps: computed from frame_pos for monotonically increasing guarantee
            timestamp_s = frame_pos / source_fps
            synced_timestamp_s = timestamp_s + sync_offset
            timestamp_ms = int(timestamp_s * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Landmarks are normalized [0,1] — scale to detection resolution
            # (matches calibration coordinate system when downscale_width is used)
            poses = []
            for i, landmarks in enumerate(result.pose_landmarks):
                lm_data = []
                xs, ys = [], []
                for j, lm in enumerate(landmarks):
                    px = lm.x * lm_w
                    py = lm.y * lm_h
                    lm_data.append({
                        "id": j,
                        "name": LANDMARK_NAMES[j],
                        "x": round(px, 1),
                        "y": round(py, 1),
                        "z": round(lm.z, 4),
                        "visibility": round(lm.visibility, 3),
                        "presence": round(lm.presence, 3),
                    })
                    if lm.visibility > 0.5:
                        xs.append(px)
                        ys.append(py)

                bbox = None
                if xs and ys:
                    bbox = {
                        "x_min": round(min(xs)),
                        "y_min": round(min(ys)),
                        "x_max": round(max(xs)),
                        "y_max": round(max(ys)),
                    }

                poses.append({
                    "person_index": i,
                    "landmarks": lm_data,
                    "bounding_box": bbox,
                })

            frames_data.append({
                "frame_index": frame_pos,
                "timestamp_s": round(timestamp_s, 4),
                "synced_timestamp_s": round(synced_timestamp_s, 4),
                "num_poses": len(poses),
                "poses": poses,
            })

            processed += 1
            if processed % 50 == 0:
                elapsed = time.time() - t0
                fps_proc = processed / max(elapsed, 0.01)
                remaining = (expected - processed) / max(fps_proc, 0.01)
                n_detected = sum(1 for f in frames_data if f["num_poses"] > 0)
                pct = 100 * processed / max(1, expected)
                print(f"  [{cam_id}] {processed}/{expected} ({pct:.0f}%) "
                      f"| {n_detected} detected | {fps_proc:.1f} fr/s | ~{remaining:.0f}s left", flush=True)

        frame_pos += 1

    cap.release()

    elapsed = time.time() - t0
    n_detected = sum(1 for f in frames_data if f["num_poses"] > 0)
    print(f"  [{cam_id}] DONE: {processed} frames, {n_detected} with pose "
          f"({100*n_detected//max(1,processed)}%) in {elapsed:.1f}s", flush=True)

    return {
        "source_fps": source_fps,
        "detection_fps": target_fps,
        "image_size": [lm_w, lm_h],
        "total_frames_processed": processed,
        "frames": frames_data,
    }


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------


def _pose_worker(cam_id, video_path, cfg_pose, sync_offset, output_dir_str, result_queue):
    """Runs in a separate process: detect poses for one camera, save JSON."""
    sys.stdout.reconfigure(line_buffering=True)
    landmarker = create_pose_detector(cfg_pose)
    downscale = cfg_pose.get("downscale_width")
    result = detect_poses_in_video(
        landmarker, video_path, cfg_pose["subsample_fps"],
        sync_offset, cam_id=cam_id, downscale_width=downscale,
    )
    landmarker.close()
    result["camera_id"] = cam_id
    result["video_path"] = video_path
    result["sync_offset_s"] = sync_offset

    # Save JSON inside the worker — avoids sending large payload through queue
    out_path = Path(output_dir_str) / f"{cam_id}_poses.json"
    with open(out_path, "w") as f:
        json.dump(result, f)

    # Only send lightweight stats through queue
    n_frames = result["total_frames_processed"]
    pose_counts = [fr["num_poses"] for fr in result["frames"]]
    result_queue.put({
        "cam_id": cam_id,
        "n_frames": n_frames,
        "avg_poses": float(np.mean(pose_counts)) if pose_counts else 0.0,
        "pose_distribution": {int(k): int(v) for k, v in
                              zip(*np.unique(pose_counts, return_counts=True))} if pose_counts else {},
        "out_path": str(out_path),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = load_config()
    cam_ids = sorted(cfg["cameras"].keys())
    output_dir = ROOT / cfg["output_dir"] / "poses"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Audio sync ---
    print("=== Audio Synchronization ===\n")
    sync_offsets = compute_sync_offsets(cfg)
    print(f"\nSync offsets: {sync_offsets}")

    # Ensure JSON-serializable types
    sync_offsets = {k: float(v) for k, v in sync_offsets.items()}

    sync_path = output_dir / "sync_offsets.json"
    with open(sync_path, "w") as f:
        json.dump(sync_offsets, f, indent=2)
    print(f"Saved to {sync_path}\n")

    # --- Step 2: Pose detection (parallel — 3 cameras simultaneously) ---
    print(f"=== Pose Detection — {len(cam_ids)} cameras in parallel ===\n")

    result_queue: Queue = Queue()
    processes = [
        Process(target=_pose_worker, args=(
            cam_id,
            str(ROOT / cfg["cameras"][cam_id]["skeleton_video"]),
            cfg["pose_detection"],
            sync_offsets[cam_id],
            str(output_dir),
            result_queue,
        ))
        for cam_id in cam_ids
    ]
    for p in processes:
        p.start()
        print(f"  Started process for {cam_ids[processes.index(p)]}")

    # Drain queue BEFORE join() to avoid pipe deadlock
    num_persons = cfg["pose_detection"]["num_poses"]
    for _ in cam_ids:
        stats = result_queue.get()
        cam_id = stats["cam_id"]
        dist = stats["pose_distribution"]
        # Frames where all expected persons are detected
        full_det = dist.get(num_persons, 0)
        full_pct = 100 * full_det // max(1, stats["n_frames"])
        print(f"\n[{cam_id}] {stats['n_frames']} frames | avg {stats['avg_poses']:.2f}/{num_persons} persons")
        print(f"  Distribution: {dist} | {full_pct}% frames with {num_persons} persons")
        print(f"  Saved to {stats['out_path']}")

    for p in processes:
        p.join()

    print("\nDone.")


if __name__ == "__main__":
    main()
