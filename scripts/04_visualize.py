"""
04_visualize.py — 3D skeleton visualization.

Outputs:
  1. Static 3D plot of a single frame
  2. Animated 3D skeleton video (MP4)
  3. Overlay on original camera video (reprojected 3D → 2D)

Usage:
  python scripts/04_visualize.py
"""

import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

# Per-person colors: BGR
PERSON_COLORS = [
    (0, 255, 0),      # person 0: green
    (0, 165, 255),    # person 1: orange
]

# MediaPipe POSE_CONNECTIONS (subset of the 33 landmarks that form skeleton edges)
CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32),
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
]


def load_config() -> dict:
    with open(ROOT / "config.json") as f:
        return json.load(f)


def load_skeleton_3d() -> dict:
    with open(ROOT / "output" / "skeleton3d" / "skeleton_3d.json") as f:
        return json.load(f)


def load_calibration() -> dict:
    with open(ROOT / "output" / "calibration" / "calibration.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_points(frame: dict, person_id: int = 0) -> np.ndarray:
    """Extract (33, 3) array for one person from a frame. NaN for missing landmarks."""
    pts = np.full((33, 3), np.nan)
    # New multi-person format: frame["persons"][i]["landmarks"]
    persons = frame.get("persons")
    if persons is not None:
        for person in persons:
            if person["person_id"] == person_id:
                for lm in person["landmarks"]:
                    if lm["x"] is not None:
                        pts[lm["id"]] = [lm["x"], lm["y"], lm["z"]]
                break
    else:
        # Legacy single-person format: frame["landmarks"]
        for lm in frame.get("landmarks", []):
            if lm["x"] is not None:
                pts[lm["id"]] = [lm["x"], lm["y"], lm["z"]]
    return pts


def get_num_persons(skeleton_data: dict) -> int:
    return skeleton_data.get("num_persons", 1)


def get_data_bounds(skeleton_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Get min/max bounds across all valid frames and all persons."""
    num_persons = get_num_persons(skeleton_data)
    all_pts = []
    for frame in skeleton_data["frames"]:
        for person_id in range(num_persons):
            pts = extract_points(frame, person_id)
            valid = pts[~np.isnan(pts[:, 0])]
            if len(valid) > 0:
                all_pts.append(valid)
    if not all_pts:
        return np.array([-1, -1, -1]), np.array([1, 1, 1])
    all_pts = np.vstack(all_pts)
    margin = 0.3
    return all_pts.min(axis=0) - margin, all_pts.max(axis=0) + margin


def draw_skeleton(ax, pts: np.ndarray, color="tab:blue", lw=2, ms=4, flip_z=True):
    """Draw skeleton on a 3D axes."""
    valid = ~np.isnan(pts[:, 0])
    z_sign = -1 if flip_z else 1
    ax.scatter(pts[valid, 0], pts[valid, 1], z_sign * pts[valid, 2],
               c=color, s=ms**2, depthshade=True)

    for i, j in CONNECTIONS:
        if valid[i] and valid[j]:
            ax.plot([pts[i, 0], pts[j, 0]],
                    [pts[i, 1], pts[j, 1]],
                    [z_sign * pts[i, 2], z_sign * pts[j, 2]],
                    c=color, lw=lw)


# ---------------------------------------------------------------------------
# 1. Static 3D plot
# ---------------------------------------------------------------------------


def plot_static(skeleton_data: dict, frame_idx: int = 0):
    """Plot a single frame's 3D skeleton (all persons)."""
    num_persons = get_num_persons(skeleton_data)
    # Find a frame with enough landmarks across any person
    for i in range(frame_idx, len(skeleton_data["frames"])):
        frame = skeleton_data["frames"][i]
        persons = frame.get("persons", [])
        if any(p["num_valid_landmarks"] >= 10 for p in persons):
            break
        # Legacy format
        if frame.get("num_valid_landmarks", 0) >= 10:
            break
    else:
        print("No frame with enough landmarks found.")
        return

    bounds_min, bounds_max = get_data_bounds(skeleton_data)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    persons = frame.get("persons", [])
    for person in persons:
        person_id = person["person_id"]
        pts = extract_points(frame, person_id)
        color = ["tab:blue", "tab:orange"][person_id % 2]
        draw_skeleton(ax, pts, color=color, ms=6)
    # Legacy single-person format
    if not persons:
        pts = extract_points(frame)
        draw_skeleton(ax, pts, color="tab:blue", ms=6)

    ax.set_xlim(bounds_min[0], bounds_max[0])
    ax.set_ylim(bounds_min[1], bounds_max[1])
    ax.set_zlim(-bounds_max[2], -bounds_min[2])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    total_valid = sum(p["num_valid_landmarks"] for p in frame.get("persons", []))
    ax.set_title(f"3D Skeleton — t={frame['timestamp_s']:.2f}s "
                 f"({total_valid} landmarks, {frame.get('num_persons', 1)} persons)")

    output_path = ROOT / "output" / "visualization" / "skeleton_frame.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Static plot saved to {output_path}")


# ---------------------------------------------------------------------------
# 2. Animated 3D video
# ---------------------------------------------------------------------------


def create_animation(skeleton_data: dict, fps: int = 15):
    """Create animated 3D skeleton video."""
    frames = skeleton_data["frames"]
    if not frames:
        print("No frames to animate.")
        return

    bounds_min, bounds_max = get_data_bounds(skeleton_data)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    frame_count_ref = [0]
    t0 = time.time()

    num_persons = get_num_persons(skeleton_data)

    def update(frame_idx):
        frame_count_ref[0] += 1
        if frame_count_ref[0] % 50 == 0:
            elapsed = time.time() - t0
            fps_ren = frame_count_ref[0] / max(elapsed, 0.01)
            remaining = (len(frames) - frame_count_ref[0]) / max(fps_ren, 0.01)
            print(f"  Animation {frame_count_ref[0]}/{len(frames)} "
                  f"| {fps_ren:.1f} fr/s | ~{remaining:.0f}s left", flush=True)

        ax.clear()
        frame = frames[frame_idx]
        total_valid = 0
        persons = frame.get("persons", [])
        for person in persons:
            person_id = person["person_id"]
            pts = extract_points(frame, person_id)
            color = ["tab:blue", "tab:orange"][person_id % 2]
            draw_skeleton(ax, pts, color=color, ms=5)
            total_valid += person["num_valid_landmarks"]
        # Legacy single-person format
        if not persons:
            pts = extract_points(frame)
            draw_skeleton(ax, pts, color="tab:blue", ms=5)
            total_valid = frame.get("num_valid_landmarks", 0)

        ax.set_xlim(bounds_min[0], bounds_max[0])
        ax.set_ylim(bounds_min[1], bounds_max[1])
        ax.set_zlim(-bounds_max[2], -bounds_min[2])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"t={frame['timestamp_s']:.2f}s — "
                     f"{total_valid} landmarks, {frame.get('num_persons', 1)} persons, "
                     f"{frame['num_cameras_used']} cams")
        ax.view_init(elev=20, azim=30 + frame_idx * 0.5)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)

    output_path = ROOT / "output" / "visualization" / "skeleton_3d.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer="ffmpeg", fps=fps, dpi=100)
    plt.close(fig)
    print(f"Animation saved to {output_path} ({len(frames)} frames, {time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# 3. Overlay on original video
# ---------------------------------------------------------------------------


def create_overlay(skeleton_data: dict, calibration: dict, cfg: dict,
                   cam_id: str = "cam02"):
    """Reproject 3D skeleton onto original camera video.

    Uses cam02 by default (best detection rate).
    """
    cam = calibration["cameras"][cam_id]
    K = np.array(cam["camera_matrix"])
    dist = np.array(cam["dist_coeffs"])
    R = np.array(cam["R"])
    T = np.array(cam["T"])
    rvec, _ = cv2.Rodrigues(R)

    video_path = str(ROOT / cfg["cameras"][cam_id]["skeleton_video"])
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale factor: K is calibrated at calib_w×calib_h, video is w×h
    calib_w = cam["image_size"][0]
    video_scale = w / calib_w  # e.g. 3840/1920 = 2.0

    # Output at half resolution for reasonable file size
    out_w, out_h = w // 2, h // 2

    output_path = ROOT / "output" / "visualization" / f"overlay_{cam_id}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, 15, (out_w, out_h))

    # Load sync offset for this camera
    sync_path = ROOT / "output" / "poses" / "sync_offsets.json"
    with open(sync_path) as f:
        sync_offsets = json.load(f)
    sync_offset = sync_offsets.get(cam_id, 0.0)

    target_fps = cfg["pose_detection"]["subsample_fps"]
    step = max(1, int(round(source_fps / target_fps)))

    # Build sorted timestamp array for nearest-neighbor lookup
    skel_timestamps = np.array([f["timestamp_s"] for f in skeleton_data["frames"]])
    skel_frames = skeleton_data["frames"]

    frame_count = 0
    frame_pos = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected = total // step
    t0 = time.time()

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        if frame_pos % step == 0:
            # Compute synced timestamp for this video frame
            synced_ts = frame_pos / source_fps + sync_offset

            # Find nearest skeleton frame within tolerance
            skeleton_frame = None
            if len(skel_timestamps) > 0:
                nearest_idx = np.argmin(np.abs(skel_timestamps - synced_ts))
                if abs(skel_timestamps[nearest_idx] - synced_ts) < 0.1:
                    skeleton_frame = skel_frames[nearest_idx]

            if skeleton_frame is not None:
                for person in skeleton_frame.get("persons", []):
                    person_id = person["person_id"]
                    pts_3d = extract_points(skeleton_frame, person_id)
                    valid = ~np.isnan(pts_3d[:, 0])
                    if not valid.any():
                        continue
                    color = PERSON_COLORS[person_id % len(PERSON_COLORS)]
                    valid_pts = pts_3d[valid].astype(np.float64)
                    pts_2d, _ = cv2.projectPoints(valid_pts, rvec, T, K, dist)
                    pts_2d = (pts_2d.reshape(-1, 2) * video_scale).astype(int)
                    valid_indices = np.where(valid)[0]
                    all_2d = {vi: pt2d for vi, pt2d in zip(valid_indices, pts_2d)}
                    for i, j in CONNECTIONS:
                        if i in all_2d and j in all_2d:
                            cv2.line(bgr, tuple(all_2d[i]), tuple(all_2d[j]), color, 3)
                    for pt in all_2d.values():
                        cv2.circle(bgr, tuple(pt), 5, (0, 0, 255), -1)
                # Legacy single-person format fallback
                if not skeleton_frame.get("persons"):
                    pts_3d = extract_points(skeleton_frame)
                    valid = ~np.isnan(pts_3d[:, 0])
                    if valid.any():
                        valid_pts = pts_3d[valid].astype(np.float64)
                        pts_2d, _ = cv2.projectPoints(valid_pts, rvec, T, K, dist)
                        pts_2d = (pts_2d.reshape(-1, 2) * video_scale).astype(int)
                        valid_indices = np.where(valid)[0]
                        all_2d = {vi: pt2d for vi, pt2d in zip(valid_indices, pts_2d)}
                        for i, j in CONNECTIONS:
                            if i in all_2d and j in all_2d:
                                cv2.line(bgr, tuple(all_2d[i]), tuple(all_2d[j]), PERSON_COLORS[0], 3)
                        for pt in all_2d.values():
                            cv2.circle(bgr, tuple(pt), 5, (0, 0, 255), -1)

            small = cv2.resize(bgr, (out_w, out_h))
            out.write(small)
            frame_count += 1

            if frame_count % 100 == 0:
                elapsed = time.time() - t0
                fps = frame_count / max(elapsed, 0.01)
                print(f"  [{cam_id}] {frame_count}/{expected} "
                      f"| {fps:.1f} fr/s | ~{(expected-frame_count)/max(fps,0.01):.0f}s left",
                      flush=True)

        frame_pos += 1

    cap.release()
    out.release()
    print(f"Overlay saved to {output_path} ({frame_count} frames, {time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# 4. Composite 3-camera overlay
# ---------------------------------------------------------------------------


def create_composite_overlay(skeleton_data: dict, calibration: dict, cfg: dict):
    """2x2 grid: 3 camera overlays + skeleton 3D animation, with audio."""
    cam_ids = sorted(cfg["cameras"].keys())  # cam01, cam02, cam03

    # Per-camera calibration data
    cam_calib = {}
    for cam_id in cam_ids:
        cam = calibration["cameras"][cam_id]
        K = np.array(cam["camera_matrix"])
        dist = np.array(cam["dist_coeffs"])
        R = np.array(cam["R"])
        T = np.array(cam["T"])
        rvec, _ = cv2.Rodrigues(R)
        cam_calib[cam_id] = {"K": K, "dist": dist, "rvec": rvec, "T": T}

    # Load sync offsets
    sync_path = ROOT / "output" / "poses" / "sync_offsets.json"
    with open(sync_path) as f:
        sync_offsets = json.load(f)

    # Open all camera video captures
    caps = {}
    source_fps = None
    orig_w, orig_h = None, None
    for cam_id in cam_ids:
        video_path = str(ROOT / cfg["cameras"][cam_id]["skeleton_video"])
        cap = cv2.VideoCapture(video_path)
        caps[cam_id] = cap
        if source_fps is None:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Open skeleton 3D animation as 4th video source
    skel3d_path = ROOT / "output" / "visualization" / "skeleton_3d.mp4"
    skel3d_cap = cv2.VideoCapture(str(skel3d_path))
    skel3d_total = int(skel3d_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Scale factor: K is calibrated at calib_w, video is orig_w
    calib_w = calibration["cameras"][cam_ids[0]]["image_size"][0]
    video_scale = orig_w / calib_w  # e.g. 3840/1920 = 2.0

    # Panel size: quarter resolution per camera → 2x2 grid
    panel_w, panel_h = orig_w // 4, orig_h // 4  # 960x540
    composite_w = panel_w * 2   # 1920
    composite_h = panel_h * 2   # 1080

    output_path = ROOT / "output" / "visualization" / "composite_overlay.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = 15
    out = cv2.VideoWriter(str(output_path), fourcc, out_fps,
                          (composite_w, composite_h))

    target_fps = cfg["pose_detection"]["subsample_fps"]
    step = max(1, int(round(source_fps / target_fps)))

    # Skeleton timestamp index
    skel_timestamps = np.array([f["timestamp_s"] for f in skeleton_data["frames"]])
    skel_frames = skeleton_data["frames"]

    total = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps.values())
    expected = total // step
    frame_count = 0
    frame_pos = 0
    t0 = time.time()

    while True:
        # Read one frame from each camera sequentially (no seek — lockstep)
        bgrs = {}
        done = False
        for cam_id in cam_ids:
            ok, bgr = caps[cam_id].read()
            if not ok:
                done = True
                break
            bgrs[cam_id] = bgr
        if done:
            break

        if frame_pos % step == 0:
            # Find matching skeleton frame (use cam01 as time reference)
            ref_synced_ts = frame_pos / source_fps + sync_offsets.get(cam_ids[0], 0.0)
            skeleton_frame = None
            if len(skel_timestamps) > 0:
                nearest_idx = np.argmin(np.abs(skel_timestamps - ref_synced_ts))
                if abs(skel_timestamps[nearest_idx] - ref_synced_ts) < 0.1:
                    skeleton_frame = skel_frames[nearest_idx]

            # Build camera panels
            panels = []
            for cam_id in cam_ids:
                bgr = bgrs[cam_id]

                if skeleton_frame is not None:
                    cc = cam_calib[cam_id]
                    for person in skeleton_frame.get("persons", []):
                        person_id = person["person_id"]
                        pts_3d = extract_points(skeleton_frame, person_id)
                        valid_mask = ~np.isnan(pts_3d[:, 0])
                        if not valid_mask.any():
                            continue
                        color = PERSON_COLORS[person_id % len(PERSON_COLORS)]
                        valid_pts = pts_3d[valid_mask].astype(np.float64)
                        pts_2d, _ = cv2.projectPoints(
                            valid_pts, cc["rvec"], cc["T"], cc["K"], cc["dist"]
                        )
                        pts_2d = (pts_2d.reshape(-1, 2) * video_scale).astype(int)
                        valid_indices = np.where(valid_mask)[0]
                        all_2d = {vi: pt2d for vi, pt2d in zip(valid_indices, pts_2d)}
                        for i, j in CONNECTIONS:
                            if i in all_2d and j in all_2d:
                                cv2.line(bgr, tuple(all_2d[i]), tuple(all_2d[j]), color, 4)
                        for pt in all_2d.values():
                            cv2.circle(bgr, tuple(pt), 8, (0, 0, 255), -1)
                    # Legacy single-person format fallback
                    if not skeleton_frame.get("persons"):
                        pts_3d = extract_points(skeleton_frame)
                        valid_mask = ~np.isnan(pts_3d[:, 0])
                        if valid_mask.any():
                            valid_pts = pts_3d[valid_mask].astype(np.float64)
                            pts_2d, _ = cv2.projectPoints(
                                valid_pts, cc["rvec"], cc["T"], cc["K"], cc["dist"]
                            )
                            pts_2d = (pts_2d.reshape(-1, 2) * video_scale).astype(int)
                            valid_indices = np.where(valid_mask)[0]
                            all_2d = {vi: pt2d for vi, pt2d in zip(valid_indices, pts_2d)}
                            for i, j in CONNECTIONS:
                                if i in all_2d and j in all_2d:
                                    cv2.line(bgr, tuple(all_2d[i]), tuple(all_2d[j]), PERSON_COLORS[0], 4)
                            for pt in all_2d.values():
                                cv2.circle(bgr, tuple(pt), 8, (0, 0, 255), -1)

                panel = cv2.resize(bgr, (panel_w, panel_h))
                cv2.putText(panel, cam_id.upper(), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                panels.append(panel)

            # 4th panel: skeleton 3D — read sequentially (already at 15fps, 1:1 with output)
            ok_skel, skel3d_bgr = skel3d_cap.read()
            if ok_skel:
                skel3d_panel = cv2.resize(skel3d_bgr, (panel_w, panel_h))
            else:
                skel3d_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

            # Add metrics overlay on skeleton panel
            if skeleton_frame is not None:
                num_valid = sum(p["num_valid_landmarks"] for p in skeleton_frame.get("persons", []))
                num_cams = skeleton_frame["num_cameras_used"]
                n_persons = skeleton_frame.get("num_persons", 1)
            else:
                num_valid = num_cams = n_persons = 0
            cv2.putText(skel3d_panel,
                        f"t={ref_synced_ts:.2f}s  {num_valid}lm  {n_persons}p  {num_cams}cams",
                        (10, panel_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1)

            # 2x2 grid: top row = cam01, cam02 / bottom row = cam03, skeleton3D
            top_row = np.hstack([panels[0], panels[1]])
            bottom_row = np.hstack([panels[2], skel3d_panel])
            composite = np.vstack([top_row, bottom_row])

            out.write(composite)
            frame_count += 1

            if frame_count % 50 == 0:
                elapsed = time.time() - t0
                fps = frame_count / max(elapsed, 0.01)
                print(f"  Composite {frame_count}/{expected} "
                      f"| {fps:.1f} fr/s | ~{(expected-frame_count)/max(fps,0.01):.0f}s left",
                      flush=True)

        frame_pos += 1

    for cap in caps.values():
        cap.release()
    skel3d_cap.release()
    out.release()
    print(f"Composite 2x2: {frame_count} frames written in {time.time()-t0:.1f}s")

    # Mux audio from cam02 onto the composite video
    audio_source = str(ROOT / cfg["cameras"]["cam02"]["skeleton_video"])
    final_path = output_path.with_name("composite_overlay_audio.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(output_path),
        "-i", audio_source,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        str(final_path),
    ]
    subprocess.run(cmd, capture_output=True)
    # Remove silent intermediate file
    output_path.unlink(missing_ok=True)
    print(f"Composite overlay saved to {final_path} (with audio)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def clean_output():
    """Remove previous visualization output before regenerating."""
    out_dir = ROOT / "output" / "visualization"
    if out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"Cleaned {out_dir}")


def main():
    cfg = load_config()
    skeleton_data = load_skeleton_3d()
    calibration = load_calibration()

    print("=== Visualization ===\n")

    # Clean previous output
    clean_output()

    print(f"{skeleton_data['total_frames']} 3D frames loaded\n")

    # 1. Static plot
    print("--- Static 3D Plot ---")
    plot_static(skeleton_data)

    # 2. Animation (needed as input for composite)
    print("\n--- 3D Animation ---")
    create_animation(skeleton_data, fps=15)

    # 3. Overlay on cam02 (best detection rate)
    print("\n--- Video Overlay ---")
    create_overlay(skeleton_data, calibration, cfg, cam_id="cam02")

    # 4. Composite 2x2 (3 cameras + skeleton 3D) with audio
    print("\n--- Composite 2x2 Overlay ---")
    create_composite_overlay(skeleton_data, calibration, cfg)

    print("\nDone!")


if __name__ == "__main__":
    main()
