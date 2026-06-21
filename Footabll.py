import sys
import os
import subprocess
import importlib
import argparse
import logging
import time
import threading
import queue
import math
from collections import defaultdict, deque


REQUIRED_PACKAGES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
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
    print(f"[Football-Intel Setup] Installing {len(missing)} missing package(s): {', '.join(missing)}")
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
    print("[Football-Intel Setup] Dependency bootstrap complete.\n")


_bootstrap_dependencies()

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

logging.basicConfig(
    level=logging.INFO,
    format=f"{C_INFO}%(asctime)s{C_RESET} [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("football.main")

WINDOW_NAME = "Football Intelligence | Stacked Tactical Preview"

BANNER = r"""
 _____           _   _           _ _   _____       _       _
|  ___|         | | | |         | | | |_   _|     | |     | |
| |__ ___   ___ | |_| |__   __ _| | |   | | _ __ | |_ ___| |
|  __/ _ \ / _ \| __| '_ \ / _` | | |   | || '_ \| __/ _ \ |
| | | (_) | (_) | |_| |_) | (_| | | |  _| || | | | ||  __/ |
\_|  \___/ \___/ \__|_.__/ \__,_|_|_|  \___/_| |_|\__\___|_|

        FOOTBALL / FIFA MATCH INTELLIGENCE SYSTEM        v1.0
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
    try:
        model = YOLO("yolov8n.pt")
        model.to(device)
        logger.info(f"{C_OK}YOLOv8 model loaded ({device}){C_RESET}")
        return model
    except Exception as e:
        logger.error(
            f"{C_ERR}Failed to load YOLOv8 weights ('yolov8n.pt'). "
            f"Download manually from https://github.com/ultralytics/assets/releases "
            f"and place 'yolov8n.pt' in this folder, then re-run.{C_RESET}\nDetails: {e}"
        )
        raise SystemExit(1)


def _gui_available():
    import platform
    system = platform.system()
    if system in ("Windows", "Darwin"):
        return True
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        probe = (
            "import cv2, sys\n"
            "try:\n"
            "    cv2.namedWindow('probe', cv2.WINDOW_NORMAL)\n"
            "    cv2.destroyAllWindows()\n"
            "    sys.exit(0)\n"
            "except Exception:\n"
            "    sys.exit(1)\n"
        )
        try:
            result = subprocess.run([sys.executable, "-c", probe], capture_output=True, timeout=8)
            return result.returncode == 0
        except Exception:
            return False
    return False


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


class TeamClassifier:
    """Classifies a player crop into Team A / Team B / Referee using
    dominant HSV color clustering against two running reference colors."""

    def __init__(self):
        self.team_colors = {}  # team_id -> running mean HSV
        self.initialized = False
        self.samples = []

    def _dominant_hsv(self, crop):
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        torso = crop[int(h * 0.15):int(h * 0.55), int(w * 0.2):int(w * 0.8)]
        if torso.size == 0:
            torso = crop
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)
        # drop near-grass-green and near-black/white pixels (pitch/shadow bleed)
        mask = ~((pixels[:, 1] < 40) | (pixels[:, 2] < 40) | (pixels[:, 2] > 245))
        filtered = pixels[mask]
        if filtered.shape[0] < 5:
            filtered = pixels
        return np.median(filtered, axis=0)

    def bootstrap(self, crops):
        """Seed two team color clusters from early-frame crops via k-means."""
        feats = []
        for c in crops:
            f = self._dominant_hsv(c)
            if f is not None:
                feats.append(f)
        if len(feats) < 4:
            return False
        feats = np.float32(feats)
        k = 2
        _, labels, centers = cv2.kmeans(
            feats, k, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
            5, cv2.KMEANS_PP_CENTERS,
        )
        self.team_colors = {0: centers[0], 1: centers[1]}
        self.initialized = True
        return True

    def classify(self, crop):
        if not self.initialized:
            return -1
        feat = self._dominant_hsv(crop)
        if feat is None:
            return -1
        best_team, best_dist = -1, 1e9
        for tid, ref in self.team_colors.items():
            dh = min(abs(feat[0] - ref[0]), 180 - abs(feat[0] - ref[0]))
            dist = dh * 2.0 + abs(feat[1] - ref[1]) * 0.5 + abs(feat[2] - ref[2]) * 0.3
            if dist < best_dist:
                best_dist, best_team = dist, tid
        return best_team


# =====================================================================
# SIMPLE IOU TRACKER (player identity persistence)
# =====================================================================
class Track:
    __slots__ = ("id", "box", "team", "age", "missed", "trail", "ball_touches")

    def __init__(self, tid, box, team):
        self.id = tid
        self.box = box
        self.team = team
        self.age = 0
        self.missed = 0
        self.trail = deque(maxlen=600)
        self.ball_touches = 0


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


class PlayerTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, classifier, frame):
        unmatched = list(range(len(detections)))
        matched_ids = set()
        for tid, tr in self.tracks.items():
            best_j, best_iou = -1, 0.3
            for j in unmatched:
                v = iou(tr.box, detections[j][:4])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                x1, y1, x2, y2 = detections[best_j][:4]
                tr.box = (x1, y1, x2, y2)
                tr.age += 1
                tr.missed = 0
                cx, cy = (x1 + x2) / 2, y2
                tr.trail.append((cx, cy))
                unmatched.remove(best_j)
                matched_ids.add(tid)

        for tid, tr in self.tracks.items():
            if tid not in matched_ids:
                tr.missed += 1

        for j in unmatched:
            x1, y1, x2, y2 = detections[j][:4]
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            team = classifier.classify(crop)
            tr = Track(self.next_id, (x1, y1, x2, y2), team)
            tr.trail.append(((x1 + x2) / 2, y2))
            self.tracks[self.next_id] = tr
            self.next_id += 1

        dead = [tid for tid, tr in self.tracks.items() if tr.missed > 40]
        for tid in dead:
            del self.tracks[tid]

        return self.tracks


# =====================================================================
# BALL DETECTION + KALMAN TRACKING
# =====================================================================
class BallTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.initialized = False
        self.last_pos = None
        self.history = deque(maxlen=300)

    def detect_candidate(self, frame, player_boxes, roi=None):
        """Detect a small, fast-moving, roughly-circular bright/white blob
        that is NOT inside a player bounding box."""
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = 0, 0, w, h
        if roi is not None:
            x0, y0, x1, y1 = [int(max(0, v)) for v in roi]
            x1, y1 = min(w, x1), min(h, y1)
        sub = frame[y0:y1, x0:x1]
        if sub.size == 0:
            return None
        gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1
        for c in contours:
            area = cv2.contourArea(c)
            if area < 4 or area > 250:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            circularity = area / (math.pi * radius * radius + 1e-6)
            if circularity < 0.5:
                continue
            gx, gy = cx + x0, cy + y0
            inside_player = any(bx1 <= gx <= bx2 and by1 <= gy <= by2
                                 for (bx1, by1, bx2, by2) in player_boxes)
            if inside_player:
                continue
            score = circularity
            if score > best_score:
                best_score, best = score, (gx, gy)
        return best

    def init(self, x, y):
        self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.initialized = True
        self.last_pos = (x, y)

    def predict(self):
        pred = self.kf.predict()
        pred = np.ravel(pred)
        return float(pred[0]), float(pred[1])

    def update(self, measurement):
        if measurement is not None:
            mx, my = measurement
            corrected = self.kf.correct(np.array([[mx], [my]], np.float32))
        else:
            corrected = self.kf.statePost
        x, y = float(np.ravel(corrected)[0]), float(np.ravel(corrected)[1])
        self.last_pos = (x, y)
        self.history.append((x, y))
        return x, y


# =====================================================================
# OFFSIDE LINE ESTIMATION
# =====================================================================
def estimate_offside_line(tracks, ball_pos, attacking_team, frame_w):
    """Very lightweight offside proxy: among defending-team outfield
    players, find the 2nd-deepest (last defender excluding keeper) and
    draw a vertical line through that player's position. attacking_team
    is inferred as the team currently nearest the ball; defending team
    is the other one."""
    defending_team = 1 - attacking_team if attacking_team in (0, 1) else None
    if defending_team is None:
        return None, None

    defenders = [tr for tr in tracks.values() if tr.team == defending_team]
    if len(defenders) < 2:
        return None, None

    moving_right = True
    if ball_pos is not None:
        moving_right = ball_pos[0] < frame_w / 2

    sorted_defs = sorted(defenders, key=lambda t: t.box[0], reverse=not moving_right)
    last_def = sorted_defs[1] if len(sorted_defs) > 1 else sorted_defs[0]
    line_x = (last_def.box[0] + last_def.box[2]) / 2
    return line_x, last_def


def nearest_team_to_ball(tracks, ball_pos):
    if ball_pos is None:
        return None
    bx, by = ball_pos
    best_team, best_dist = None, 1e9
    for tr in tracks.values():
        if tr.team not in (0, 1):
            continue
        cx, cy = (tr.box[0] + tr.box[2]) / 2, tr.box[3]
        d = (cx - bx) ** 2 + (cy - by) ** 2
        if d < best_dist:
            best_dist, best_team = d, tr.team
    return best_team


# =====================================================================
# MATCH STATISTICS
# =====================================================================
class MatchStats:
    def __init__(self):
        self.possession_frames = {0: 0, 1: 0}
        self.passes = defaultdict(int)  # (team, from_id, to_id) -> count
        self.last_possessor = None
        self.shots = []  # (x, y, team, frame_idx)
        self.touches = defaultdict(int)

    def log_frame(self, tracks, ball_pos, possessing_team, possessor_id):
        if possessing_team in (0, 1):
            self.possession_frames[possessing_team] += 1
        if possessor_id is not None:
            self.touches[possessor_id] += 1
            if self.last_possessor is not None and self.last_possessor != possessor_id:
                a, b = self.last_possessor, possessor_id
                ta = tracks[a].team if a in tracks else None
                tb = tracks[b].team if b in tracks else None
                if ta == tb and ta is not None:
                    self.passes[(ta, a, b)] += 1
            self.last_possessor = possessor_id

    def log_shot(self, x, y, team, frame_idx):
        self.shots.append((x, y, team, frame_idx))

    def possession_pct(self):
        total = sum(self.possession_frames.values())
        if total == 0:
            return {0: 0.0, 1: 0.0}
        return {k: round(v / total * 100, 1) for k, v in self.possession_frames.items()}


def closest_player_to_ball(tracks, ball_pos, max_dist=60):
    if ball_pos is None:
        return None
    bx, by = ball_pos
    best_id, best_dist = None, max_dist
    for tid, tr in tracks.items():
        cx, cy = (tr.box[0] + tr.box[2]) / 2, tr.box[3]
        d = math.hypot(cx - bx, cy - by)
        if d < best_dist:
            best_dist, best_id = d, tid
    return best_id


# =====================================================================
# VISUAL OVERLAYS
# =====================================================================
TEAM_COLORS = {0: (60, 60, 230), 1: (230, 180, 30), -1: (170, 170, 170)}
TEAM_NAMES = {0: "TEAM A", 1: "TEAM B", -1: "UNK"}


def _glow_circle(img, center, radius, color, intensity=1.0):
    """Draws a soft additive glow behind a small circle, FIFA-broadcast style."""
    overlay = np.zeros_like(img)
    cv2.circle(overlay, center, radius * 3, color, -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), radius * 1.1)
    img[:] = cv2.addWeighted(img, 1.0, overlay, 0.55 * intensity, 0)
    cv2.circle(img, center, radius, color, -1)
    cv2.circle(img, center, radius, (255, 255, 255), 1, cv2.LINE_AA)


def _rounded_tag(img, x1, y1, text, color, scale=0.48):
    """Small filled pill-shaped tag above a player box, FIFA-graphics style."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad_x, pad_y = 6, 4
    x0, y0 = int(x1), int(y1 - th - 2 * pad_y - 4)
    x2, y2 = x0 + tw + 2 * pad_x, y0 + th + 2 * pad_y
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x2, y2), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, dst=img)
    cv2.putText(img, text, (x0 + pad_x, y2 - pad_y - 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_broadcast_overlay(frame, tracks, ball_pos, offside_x, offside_player,
                            stats, frame_idx, total_frames, fps_smooth):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # subtle vignette for cinematic broadcast feel
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    vig = np.clip(1.0 - 0.35 * (dist - 0.35), 0.72, 1.0).astype(np.float32)
    overlay = (overlay.astype(np.float32) * vig[..., None]).clip(0, 255).astype(np.uint8)

    # ---- offside line: glowing red, with shaded "offside zone" ----
    if offside_x is not None:
        ox = int(offside_x)
        zone = overlay.copy()
        zone_x0, zone_x1 = (0, ox) if ox < w / 2 else (ox, w)
        cv2.rectangle(zone, (zone_x0, 0), (zone_x1, h), (40, 40, 230), -1)
        overlay = cv2.addWeighted(overlay, 0.94, zone, 0.06, 0)
        glow = np.zeros_like(overlay)
        cv2.line(glow, (ox, 0), (ox, h), (40, 40, 255), 8)
        glow = cv2.GaussianBlur(glow, (0, 0), 6)
        overlay = cv2.addWeighted(overlay, 1.0, glow, 0.9, 0)
        cv2.line(overlay, (ox, 0), (ox, h), (60, 60, 255), 2, cv2.LINE_AA)
        _rounded_tag(overlay, ox - 60, 34, "OFFSIDE LINE", (30, 30, 220), scale=0.55)

    # ---- player boxes: corner brackets + ghost trail + jersey tag ----
    for tid, tr in tracks.items():
        x1, y1, x2, y2 = [int(v) for v in tr.box]
        color = TEAM_COLORS.get(tr.team, TEAM_COLORS[-1])

        if len(tr.trail) > 2:
            pts = list(tr.trail)[-18:]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                p1 = tuple(map(int, pts[i - 1]))
                p2 = tuple(map(int, pts[i]))
                fade_color = tuple(int(c * alpha) for c in color)
                cv2.line(overlay, p1, p2, fade_color, 2, cv2.LINE_AA)

        corner = max(6, min(16, (x2 - x1) // 5))
        thick = 2
        for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(overlay, (px, py), (px + dx * corner, py), color, thick, cv2.LINE_AA)
            cv2.line(overlay, (px, py), (px, py + dy * corner), color, thick, cv2.LINE_AA)

        cv2.ellipse(overlay, ((x1 + x2) // 2, y2), (max(12, (x2 - x1) // 2), 6),
                    0, 0, 360, color, 2, cv2.LINE_AA)
        _rounded_tag(overlay, x1, y1, f"{TEAM_NAMES.get(tr.team, 'UNK')} #{tid}", color)

    # ---- ball: glow + comet trail ----
    if ball_pos is not None:
        bx, by = int(ball_pos[0]), int(ball_pos[1])
        _glow_circle(overlay, (bx, by), 6, (40, 230, 255), intensity=1.2)

    # ---- scoreboard panel: dark gradient, possession bar, pass counts ----
    panel_w, panel_h = 300, 100
    panel = overlay[0:panel_h, 0:panel_w].copy()
    grad = np.zeros_like(panel, dtype=np.float32)
    for i in range(panel_h):
        shade = 22 - int(14 * (i / panel_h))
        grad[i, :] = (shade, shade, shade)
    panel = cv2.addWeighted(panel, 0.15, grad.astype(np.uint8), 0.85, 0)
    overlay[0:panel_h, 0:panel_w] = panel
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (90, 90, 90), 1, cv2.LINE_AA)

    poss = stats.possession_pct()
    bar_total = panel_w - 24
    a_w = int(bar_total * poss.get(0, 50) / 100)
    cv2.rectangle(overlay, (12, 14), (12 + bar_total, 24), (50, 50, 50), -1)
    cv2.rectangle(overlay, (12, 14), (12 + a_w, 24), TEAM_COLORS[0], -1)
    cv2.rectangle(overlay, (12 + a_w, 14), (12 + bar_total, 24), TEAM_COLORS[1], -1)
    cv2.putText(overlay, f"TEAM A {poss.get(0, 0):.0f}%", (12, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEAM_COLORS[0], 1, cv2.LINE_AA)
    cv2.putText(overlay, f"TEAM B {poss.get(1, 0):.0f}%", (160, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEAM_COLORS[1], 1, cv2.LINE_AA)
    cv2.putText(overlay, f"PASSES  A {sum(c for (t, a, b), c in stats.passes.items() if t == 0)}"
                          f"   B {sum(c for (t, a, b), c in stats.passes.items() if t == 1)}",
                (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
    progress = frame_idx / max(total_frames, 1)
    cv2.rectangle(overlay, (12, 84), (12 + bar_total, 88), (60, 60, 60), -1)
    cv2.rectangle(overlay, (12, 84), (12 + int(bar_total * progress), 88), (0, 220, 255), -1)
    cv2.putText(overlay, f"{fps_smooth:.0f} FPS", (panel_w - 78, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.putText(overlay, "tubakhxn", (w - 140, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, "tubakhxn", (w - 140, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return overlay


class TacticalMinimap:
    """Renders a top-down pitch with player dots + ball, styled like a
    broadcast tactical-cam graphic (striped turf, glow dots, jersey tags)."""

    def __init__(self, frame_w, panel_h, pitch_w_m=105, pitch_h_m=68):
        self.w = frame_w
        self.h = panel_h
        self.pitch_w_m = pitch_w_m
        self.pitch_h_m = pitch_h_m

    def _draw_pitch(self, img, margin, pw, ph):
        stripes = 12
        stripe_w = pw / stripes
        for i in range(stripes):
            shade = (28, 96, 28) if i % 2 == 0 else (24, 84, 24)
            x0 = int(margin + i * stripe_w)
            x1 = int(margin + (i + 1) * stripe_w)
            cv2.rectangle(img, (x0, margin), (x1, margin + ph), shade, -1)
        line_color = (235, 235, 235)
        cv2.rectangle(img, (margin, margin), (margin + pw, margin + ph), line_color, 2, cv2.LINE_AA)
        cv2.line(img, (margin + pw // 2, margin), (margin + pw // 2, margin + ph), line_color, 2, cv2.LINE_AA)
        cv2.circle(img, (margin + pw // 2, margin + ph // 2), 42, line_color, 2, cv2.LINE_AA)
        cv2.circle(img, (margin + pw // 2, margin + ph // 2), 3, line_color, -1, cv2.LINE_AA)
        box_w, box_h = int(pw * 0.14), int(ph * 0.5)
        cv2.rectangle(img, (margin, margin + (ph - box_h) // 2),
                      (margin + box_w, margin + (ph + box_h) // 2), line_color, 2, cv2.LINE_AA)
        cv2.rectangle(img, (margin + pw - box_w, margin + (ph - box_h) // 2),
                      (margin + pw, margin + (ph + box_h) // 2), line_color, 2, cv2.LINE_AA)

    def render(self, tracks, ball_pos, frame_w, frame_h, offside_x=None):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        margin = 36
        pw, ph = self.w - 2 * margin, self.h - 2 * margin
        self._draw_pitch(img, margin, pw, ph)

        def to_minimap(px, py):
            nx = px / max(frame_w, 1)
            ny = py / max(frame_h, 1)
            return int(margin + nx * pw), int(margin + ny * ph)

        if offside_x is not None:
            ox, _ = to_minimap(offside_x, 0)
            glow = np.zeros_like(img)
            cv2.line(glow, (ox, margin), (ox, margin + ph), (40, 40, 255), 6)
            glow = cv2.GaussianBlur(glow, (0, 0), 5)
            img = cv2.addWeighted(img, 1.0, glow, 0.9, 0)
            cv2.line(img, (ox, margin), (ox, margin + ph), (60, 60, 255), 2, cv2.LINE_AA)

        for tid, tr in tracks.items():
            cx, cy = (tr.box[0] + tr.box[2]) / 2, tr.box[3]
            mx, my = to_minimap(cx, cy)
            color = TEAM_COLORS.get(tr.team, TEAM_COLORS[-1])
            cv2.circle(img, (mx, my), 7, color, -1, cv2.LINE_AA)
            cv2.circle(img, (mx, my), 7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(tid), (mx - 4, my + 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (0, 0, 0), 1, cv2.LINE_AA)

        if ball_pos is not None:
            mx, my = to_minimap(*ball_pos)
            _glow_circle(img, (mx, my), 5, (40, 230, 255), intensity=1.0)

        cv2.putText(img, "TACTICAL VIEW", (margin, margin - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        legend_y = self.h - 14
        cv2.circle(img, (margin + 8, legend_y), 6, TEAM_COLORS[0], -1)
        cv2.putText(img, "Team A", (margin + 20, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.circle(img, (margin + 110, legend_y), 6, TEAM_COLORS[1], -1)
        cv2.putText(img, "Team B", (margin + 122, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (230, 230, 230), 1, cv2.LINE_AA)
        return img


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

    classifier = TeamClassifier()
    tracker = PlayerTracker()
    ball_tracker = BallTracker()
    stats = MatchStats()

    panel_h = int(w * 0.45)
    stacked_h = h + panel_h
    minimap = TacticalMinimap(w, panel_h)

    out_path = os.path.join(output_dir, "output_match.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, stacked_h))

    if display and not _gui_available():
        display = False
        logger.warning(f"{C_WARN}No display backend detected — running headless.{C_RESET}")

    if display:
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, min(w, 900), int(min(w, 900) * stacked_h / w))
            cv2.moveWindow(WINDOW_NAME, 60, 30)
            logger.info(f"{C_OK}Live preview open — press 'q' to stop early.{C_RESET}")
        except cv2.error as e:
            display = False
            logger.warning(f"{C_WARN}Could not open preview window ({e}); continuing headless.{C_RESET}")

    bootstrap_crops = []
    bootstrap_frames_needed = 25
    frame_idx = 0
    fps_smooth = fps
    t_start = time.time()
    heatmap_points = defaultdict(list)
    in_possession_prev = None

    pbar = tqdm(total=total_frames, desc="Processing match", unit="frame",
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (C_OK, C_RESET))

    while True:
        frame = reader.read()
        if frame is None or (max_frames and frame_idx >= max_frames):
            break
        loop_t0 = time.time()

        results = model.predict(frame, classes=[0], conf=0.35, verbose=False, device=device)
        boxes = []
        if results and len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if (y2 - y1) > h * 0.04:
                    boxes.append((x1, y1, x2, y2))

        if not classifier.initialized:
            for (x1, y1, x2, y2) in boxes:
                bootstrap_crops.append(frame[int(y1):int(y2), int(x1):int(x2)])
            if frame_idx >= bootstrap_frames_needed and len(bootstrap_crops) >= 6:
                ok = classifier.bootstrap(bootstrap_crops)
                if ok:
                    logger.info(f"{C_OK}Team color clusters bootstrapped.{C_RESET}")

        tracks = tracker.update(boxes, classifier, frame)
        player_boxes = [tr.box for tr in tracks.values()]

        roi = None
        if ball_tracker.initialized:
            px, py = ball_tracker.predict()
            roi = (px - 100, py - 100, px + 100, py + 100)
        cand = ball_tracker.detect_candidate(frame, player_boxes, roi=roi)
        if cand is not None and not ball_tracker.initialized:
            ball_tracker.init(*cand)
        bx_by = ball_tracker.update(cand) if ball_tracker.initialized else None

        possessor_id = closest_player_to_ball(tracks, bx_by) if bx_by else None
        possessing_team = tracks[possessor_id].team if possessor_id in tracks else None
        stats.log_frame(tracks, bx_by, possessing_team, possessor_id)

        attacking_team = nearest_team_to_ball(tracks, bx_by)
        offside_x, offside_player = estimate_offside_line(tracks, bx_by, attacking_team, w)

        for tid, tr in tracks.items():
            cx, cy = (tr.box[0] + tr.box[2]) / 2, tr.box[3]
            heatmap_points[tid].append((cx, cy))

        broadcast = draw_broadcast_overlay(frame, tracks, bx_by, offside_x, offside_player,
                                            stats, frame_idx, total_frames, fps_smooth)
        minimap_img = minimap.render(tracks, bx_by, w, h, offside_x=offside_x)

        stacked = np.vstack([broadcast, minimap_img])
        writer.write(stacked)

        elapsed = max(time.time() - loop_t0, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / elapsed)

        if display:
            try:
                cv2.imshow(WINDOW_NAME, stacked)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User requested early stop (q pressed)")
                    break
            except cv2.error as e:
                display = False
                logger.warning(f"{C_WARN}Live display unavailable ({e}); continuing headless.{C_RESET}")

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    reader.stop()
    writer.release()
    if display:
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error:
            pass

    total_time = time.time() - t_start
    logger.info(f"{C_OK}Processed {frame_idx} frames in {total_time:.1f}s "
                f"({frame_idx / max(total_time, 1e-6):.1f} fps avg){C_RESET}")
    logger.info(f"{C_OK}Saved {out_path}{C_RESET}")

    logger.info("Generating analytics outputs...")
    generate_heatmaps(heatmap_points, tracker.tracks, w, h, os.path.join(output_dir, "player_heatmaps.png"))
    generate_pass_network(stats, tracker.tracks, heatmap_points, w, h, os.path.join(output_dir, "pass_network.png"))
    generate_dashboard(stats, tracker.tracks, os.path.join(output_dir, "match_dashboard.png"))
    generate_shot_map(stats, w, h, os.path.join(output_dir, "shot_map.png"))

    logger.info(f"{C_OK}All outputs saved to: {output_dir}{C_RESET}")
    _print_summary(stats)


# =====================================================================
# POST-MATCH ANALYTICS GRAPHICS
# =====================================================================
def generate_heatmaps(heatmap_points, tracks, w, h, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for team_idx, ax in enumerate(axes):
        ax.set_facecolor("#1c5c1c")
        ax.set_title(f"Team {'A' if team_idx == 0 else 'B'} Heatmap")
        xs, ys = [], []
        for tid, pts in heatmap_points.items():
            tr = tracks.get(tid)
            team = tr.team if tr else -1
            if team == team_idx:
                for (x, y) in pts:
                    xs.append(x)
                    ys.append(y)
        if xs:
            ax.hist2d(xs, ys, bins=40, range=[[0, w], [0, h]], cmap="hot", alpha=0.85)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, facecolor="#0d0d0d")
    plt.close(fig)


def generate_pass_network(stats, tracks, heatmap_points, w, h, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#1c5c1c")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    avg_pos = {}
    for tid, pts in heatmap_points.items():
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            avg_pos[tid] = (sum(xs) / len(xs), sum(ys) / len(ys))

    for (team, a, b), count in stats.passes.items():
        if a in avg_pos and b in avg_pos:
            x1, y1 = avg_pos[a]
            x2, y2 = avg_pos[b]
            color = TEAM_COLORS.get(team, (180, 180, 180))
            color_norm = tuple(c / 255 for c in color[::-1])
            ax.plot([x1, x2], [y1, y2], color=color_norm, alpha=min(1.0, 0.2 + count * 0.05),
                    linewidth=min(5, 1 + count * 0.3))

    for tid, (x, y) in avg_pos.items():
        tr = tracks.get(tid)
        team = tr.team if tr else -1
        color = TEAM_COLORS.get(team, (180, 180, 180))
        color_norm = tuple(c / 255 for c in color[::-1])
        ax.scatter([x], [y], s=180, color=color_norm, edgecolors="black", zorder=5)
        ax.text(x, y, str(tid), ha="center", va="center", fontsize=8, color="white", zorder=6)

    ax.set_title("Pass Network")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, facecolor="#0d0d0d")
    plt.close(fig)


def generate_dashboard(stats, tracks, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor("#0d0d0d")

    poss = stats.possession_pct()
    ax = axes[0, 0]
    ax.pie([poss.get(0, 0), poss.get(1, 0)], labels=["Team A", "Team B"],
           colors=["#3c3ce6", "#e6b41e"], autopct="%1.1f%%")
    ax.set_title("Possession", color="white")

    ax = axes[0, 1]
    touch_ids = sorted(stats.touches, key=lambda k: -stats.touches[k])[:10]
    ax.bar([str(t) for t in touch_ids], [stats.touches[t] for t in touch_ids], color="#5cd65c")
    ax.set_title("Top 10 Touches by Player", color="white")
    ax.tick_params(colors="white")

    ax = axes[1, 0]
    pass_counts = defaultdict(int)
    for (team, a, b), c in stats.passes.items():
        pass_counts[team] += c
    ax.bar(["Team A", "Team B"], [pass_counts.get(0, 0), pass_counts.get(1, 0)],
           color=["#3c3ce6", "#e6b41e"])
    ax.set_title("Total Passes", color="white")
    ax.tick_params(colors="white")

    ax = axes[1, 1]
    shot_counts = defaultdict(int)
    for (_, _, team, _) in stats.shots:
        shot_counts[team] += 1
    ax.bar(["Team A", "Team B"], [shot_counts.get(0, 0), shot_counts.get(1, 0)],
           color=["#3c3ce6", "#e6b41e"])
    ax.set_title("Shots", color="white")
    ax.tick_params(colors="white")

    for row in axes:
        for a in row:
            a.set_facecolor("#0d0d0d")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, facecolor="#0d0d0d")
    plt.close(fig)


def generate_shot_map(stats, w, h, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#1c5c1c")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    for (x, y, team, _) in stats.shots:
        color = TEAM_COLORS.get(team, (180, 180, 180))
        color_norm = tuple(c / 255 for c in color[::-1])
        ax.scatter([x], [y], s=120, color=color_norm, edgecolors="white", marker="*")
    ax.set_title("Shot Map")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, facecolor="#0d0d0d")
    plt.close(fig)


def _print_summary(stats):
    print(f"\n{C_INFO}{'=' * 60}{C_RESET}")
    print(f"{C_INFO}MATCH ANALYTICS SUMMARY{C_RESET}")
    print(f"{C_INFO}{'=' * 60}{C_RESET}")
    poss = stats.possession_pct()
    total_passes = sum(stats.passes.values())
    total_shots = len(stats.shots)
    print(f"  Possession -> Team A: {poss.get(0, 0)}%  Team B: {poss.get(1, 0)}%")
    print(f"  Total passes detected: {total_passes}")
    print(f"  Total shots detected: {total_shots}")
    print(f"{C_INFO}{'=' * 60}{C_RESET}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Football/FIFA Match Intelligence System")
    parser.add_argument("video_positional", nargs="?", default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    video = args.video or args.video_positional
    if not video:
        parser.error("a video path is required, e.g. 'python football_intelligence.py match.mp4'")
    args.video = video
    return args


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