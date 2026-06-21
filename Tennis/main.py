import sys
import os
import subprocess
import importlib
import argparse
import logging
import time
import threading
import queue

# =====================================================================
# AUTO-INSTALL DEPENDENCIES
# =====================================================================
REQUIRED_PACKAGES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "filterpy": "filterpy",
    "torch": "torch",
    "ultralytics": "ultralytics",
    "tqdm": "tqdm",
    "colorama": "colorama",
}


def _bootstrap_dependencies():
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        return

    print(f"[Hawk-Eye Setup] Installing {len(missing)} missing package(s): {', '.join(missing)}")
    for pkg in missing:
        print(f"  -> pip install {pkg} ...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", pkg],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  [WARNING] Failed to install {pkg}: {result.stderr[-300:]}")
        else:
            print(f"  [OK] {pkg} installed")
    print("[Hawk-Eye Setup] Dependency bootstrap complete.\n")


_bootstrap_dependencies()

import cv2
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc="", **kwargs):
        return iterable if iterable is not None else range(total or 0)

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    C_OK, C_WARN, C_ERR, C_INFO, C_RESET = Fore.GREEN, Fore.YELLOW, Fore.RED, Fore.CYAN, Style.RESET_ALL
except ImportError:
    C_OK = C_WARN = C_ERR = C_INFO = C_RESET = ""

from ultralytics import YOLO

import analytics
import visualization

# =====================================================================
# LOGGING
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format=f"{C_INFO}%(asctime)s{C_RESET} [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hawkeye.main")

BANNER = r"""
 _   _                _      _   _              _      ______           
| | | |              | |    | | | |            | |     |  ___|          
| |_| | __ ___      _| | __ | |_| | __ _      _| |     | |__ _   _  ___ 
|  _  |/ _` \ \ /\ / / |/ / |  _  |/ _` | __ / / |     |  __| | | |/ _ \
| | | | (_| |\ V  V /|   <  | | | | (_| | / //  |____  | |__| |_| |  __/
\_| |_/\__,_| \_/\_/ |_|\_\ \_| |_/\__,_|/_/    \____/  \____/\__, |\___|
                                                                __/ |    
        TENNIS HAWK-EYE 3D INTELLIGENCE SYSTEM                |___/  v1.0
                  dev: tubakhxn
"""


def detect_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info(f"{C_OK}CUDA available -> using GPU: {name}{C_RESET}")
        return "cuda"
    logger.info(f"{C_WARN}CUDA not available -> falling back to CPU{C_RESET}")
    return "cpu"


def load_yolo_model(device):
    """Loads YOLOv8 (nano) for player detection, with a clear error path
    if weights cannot be auto-downloaded (e.g. restricted network)."""
    try:
        model = YOLO("yolov8n.pt")
        model.to(device)
        logger.info(f"{C_OK}YOLOv8 player-detection model loaded ({device}){C_RESET}")
        return model
    except Exception as e:
        logger.error(
            f"{C_ERR}Failed to load YOLOv8 weights ('yolov8n.pt'). "
            f"This usually means the weights could not be auto-downloaded "
            f"(no internet access or a restricted network). "
            f"Download manually from "
            f"https://github.com/ultralytics/assets/releases and place "
            f"'yolov8n.pt' in this folder, then re-run.{C_RESET}\nDetails: {e}"
        )
        raise SystemExit(1)


# =====================================================================
# THREADED VIDEO READER (decouples disk I/O from processing loop)
# =====================================================================
class ThreadedVideoReader:
    def __init__(self, path, queue_size=64):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.q.put(None)
                break
            self.q.put(frame)
        self.cap.release()

    def read(self):
        return self.q.get()

    def stop(self):
        self.stopped = True


# =====================================================================
# PLAYER DETECTION (YOLOv8 wrapper, filtered to 'person' class)
# =====================================================================
def detect_players(model, frame, conf_thresh=0.35, device="cpu"):
    results = model.predict(frame, classes=[0], conf=conf_thresh, verbose=False, device=device)
    dets = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            # filter tiny spectator/ball-kid boxes far in background by area heuristic
            if (y2 - y1) > frame.shape[0] * 0.08:
                dets.append((x1, y1, x2, y2, conf))
    return dets


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def run_pipeline(video_path, output_dir, display=True, max_frames=None):
    os.makedirs(output_dir, exist_ok=True)
    print(BANNER)

    device = detect_device()
    model = load_yolo_model(device)

    logger.info(f"Opening video: {video_path}")
    reader = ThreadedVideoReader(video_path).start()
    fps, w, h, total_frames = reader.fps, reader.w, reader.h, reader.total_frames
    logger.info(f"Video: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")
    if max_frames:
        total_frames = min(total_frames, max_frames)

    # --- module initialization ---
    calibrator = analytics.CourtCalibrator(w, h)
    player_tracker = analytics.PlayerTracker(fps=fps)
    ball_tracker = analytics.BallKalmanTracker(fps=fps)
    ball_detector = analytics.BallCandidateDetector()
    physics = analytics.PhysicsEngine(fps=fps)
    stats = analytics.MatchStatistics(fps=fps)

    overlay = visualization.BroadcastOverlay(w, h)
    hawkeye3d = visualization.HawkEye3DRenderer()

    out_video_path = os.path.join(output_dir, "output_match.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    ball_height_history = []
    ball_court_history = []
    bounce_world_points = []
    recent_heights = []
    recent_court_pts = []
    last_bounce_frame = -999

    frame_idx = 0
    t_start = time.time()
    fps_smooth = fps

    pbar = tqdm(total=total_frames, desc="Processing match", unit="frame",
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (C_OK, C_RESET))

    while True:
        frame = reader.read()
        if frame is None or (max_frames and frame_idx >= max_frames):
            break

        loop_t0 = time.time()

        if frame_idx == 0:
            calibrator.calibrate(frame)

        # ---- player detection & tracking ----
        detections = detect_players(model, frame, device=device)
        players = player_tracker.update(detections, calibrator)

        # ---- ball detection & tracking ----
        roi = None
        if ball_tracker.initialized:
            px, py = ball_tracker.predict()
            roi = (int(px - 90), int(py - 90), int(px + 90), int(py + 90))
        measurement = ball_detector.detect(frame, roi=roi)

        if measurement is not None:
            if not ball_tracker.initialized:
                ball_tracker.init(*measurement)
            bx, by = ball_tracker.update(measurement)
        elif ball_tracker.initialized:
            bx, by = ball_tracker.update(None)
        else:
            bx, by = None, None

        is_bounce = False
        ball_speed_kmh = 0.0
        ball_wx = ball_wy = None
        ball_height = 0.0

        if bx is not None and ball_tracker.initialized:
            ball_wx, ball_wy = calibrator.image_to_court(bx, by)
            ball_tracker.history_px.append((bx, by))
            ball_tracker.history_court.append((ball_wx, ball_wy))

            # crude relative height proxy: vertical pixel velocity damped by
            # distance-from-net (perspective), smoothed for visual plausibility
            vy = ball_tracker.velocity_px[1]
            height_proxy = max(0.0, 1.2 - abs(vy) * 0.01)
            recent_heights.append(height_proxy)
            recent_court_pts.append((ball_wx, ball_wy))
            if len(recent_heights) > 12:
                recent_heights.pop(0)
                recent_court_pts.pop(0)
            ball_height = height_proxy
            ball_height_history.append(ball_height)
            ball_court_history.append((ball_wx, ball_wy))

            ball_speed_kmh = physics.estimate_speed_kmh(ball_tracker.history_court, fps)
            if ball_speed_kmh > 5:
                stats.shot_speeds.append(ball_speed_kmh)

            bounces = physics.detect_bounce(recent_heights, recent_court_pts, fps)
            if bounces and (frame_idx - last_bounce_frame) > fps * 0.4:
                is_bounce = True
                last_bounce_frame = frame_idx
                bounce_world_points.append((ball_wx, ball_wy))
                stats.log_bounce(ball_wx, ball_wy, frame_idx)
                if ball_speed_kmh > 80:
                    stats.serve_speeds.append(ball_speed_kmh)

        stats.log_frame(players, ball_speed_kmh)

        # ---- 2D broadcast overlay render ----
        frame = overlay.draw_court_lines(frame, calibrator)
        frame = overlay.draw_players(frame, players)
        ball_px = (bx, by) if bx is not None else None
        frame = overlay.update_ball(frame, ball_px, speed_kmh=ball_speed_kmh,
                                     is_bounce=is_bounce, calibrator=calibrator)

        elapsed = max(time.time() - loop_t0, 1e-6)
        inst_fps = 1.0 / elapsed
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps

        panel_lines = [
            f"Players Tracked: {len(players)}",
            f"Ball Speed: {ball_speed_kmh:5.1f} km/h",
            f"Bounces: {len(bounce_world_points)}",
        ]
        frame = overlay.draw_hud(frame, panel_lines, fps_smooth, stats.rally_count, frame_idx, total_frames)

        writer.write(frame)

        # ---- live 3D Hawk-Eye window (rendered periodically for perf) ----
        if display and frame_idx % 2 == 0:
            players_court = {tid: (t.positions_court[-1][0], t.positions_court[-1][1], t.color_label)
                              for tid, t in players.items() if t.positions_court}
            img3d = hawkeye3d.render(
                frame_idx, total_frames, players_court,
                ball_court_history[-150:], ball_height_history[-150:],
                bounce_world_points,
                current_ball_pos=(ball_wx, ball_wy) if ball_wx is not None else None,
                current_ball_height=ball_height,
            )
            try:
                cv2.imshow("Stacked HawkEye Preview (Live Rendering)", frame)
                cv2.imshow("HawkEye 3D Trajectory", img3d)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested early stop (q pressed)")
                    break
            except cv2.error:
                # headless environment (no GUI backend) - disable further display attempts
                display = False
                logger.warning(f"{C_WARN}No display backend available, continuing headless{C_RESET}")

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    reader.stop()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    total_time = time.time() - t_start
    logger.info(f"{C_OK}Processed {frame_idx} frames in {total_time:.1f}s "
                f"({frame_idx / max(total_time, 1e-6):.1f} fps average){C_RESET}")
    logger.info(f"{C_OK}Saved {out_video_path}{C_RESET}")

    stats.log_rally_end()

    # ---- generate the four post-match analytics graphics ----
    logger.info("Generating analytics outputs...")
    summary = stats.player_summary(player_tracker.tracks)
    visualization.generate_heatmaps(stats, player_tracker.tracks,
                                     os.path.join(output_dir, "player_heatmaps.png"))
    visualization.generate_dashboard(stats, player_tracker.tracks, summary,
                                      os.path.join(output_dir, "match_dashboard.png"))
    visualization.generate_trajectory_analysis(stats, ball_court_history, ball_height_history,
                                                os.path.join(output_dir, "trajectory_analysis.png"))
    visualization.generate_serve_statistics(stats, os.path.join(output_dir, "serve_statistics.png"))

    logger.info(f"{C_OK}All outputs saved to: {output_dir}{C_RESET}")
    _print_summary(stats, summary)


def _print_summary(stats, summary):
    print(f"\n{C_INFO}{'=' * 60}{C_RESET}")
    print(f"{C_INFO}MATCH ANALYTICS SUMMARY{C_RESET}")
    print(f"{C_INFO}{'=' * 60}{C_RESET}")
    for tid, s in summary.items():
        print(f"  Player {tid} ({s['color']}): {s['distance_m']}m covered, "
              f"avg {s['avg_speed_kmh']} km/h, sprint {s['sprint_speed_kmh']} km/h, "
              f"coverage {s['coverage_pct']}%")
    serve = stats.serve_stats()
    shot = stats.shot_stats()
    print(f"  Rallies: {stats.rally_count}  |  Shots: {shot['count']}  |  "
          f"Serves: {serve['count']} (avg {serve['avg_kmh']} km/h, max {serve['max_kmh']} km/h)")
    print(f"  Bounces detected: {len(stats.bounce_locations)}")
    print(f"{C_INFO}{'=' * 60}{C_RESET}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Tennis Hawk-Eye 3D Intelligence System")
    parser.add_argument("--video", type=str, required=True, help="Path to input tennis match video")
    parser.add_argument("--output", type=str, default="./output", help="Output directory for results")
    parser.add_argument("--no-display", action="store_true", help="Disable live preview windows")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit processing to N frames (testing)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_pipeline(args.video, args.output, display=not args.no_display, max_frames=args.max_frames)
    except KeyboardInterrupt:
        print(f"\n{C_WARN}Interrupted by user.{C_RESET}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"{C_ERR}Pipeline failed: {e}{C_RESET}")
        raise
