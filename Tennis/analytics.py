import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from filterpy.kalman import KalmanFilter
import logging

logger = logging.getLogger("hawkeye.analytics")


COURT_LENGTH = 23.77
COURT_WIDTH_DOUBLES = 10.97
COURT_WIDTH_SINGLES = 8.23
SERVICE_LINE_DIST = 6.40         
NET_HEIGHT_CENTER = 0.914
NET_HEIGHT_POST = 1.07
GRAVITY = 9.81                   


COURT_KEYPOINTS_WORLD = {
    "baseline_far_left":     (-COURT_WIDTH_DOUBLES / 2, COURT_LENGTH / 2),
    "baseline_far_right":    (COURT_WIDTH_DOUBLES / 2, COURT_LENGTH / 2),
    "baseline_near_left":    (-COURT_WIDTH_DOUBLES / 2, -COURT_LENGTH / 2),
    "baseline_near_right":   (COURT_WIDTH_DOUBLES / 2, -COURT_LENGTH / 2),
    "service_far_left":      (-COURT_WIDTH_SINGLES / 2, SERVICE_LINE_DIST),
    "service_far_right":     (COURT_WIDTH_SINGLES / 2, SERVICE_LINE_DIST),
    "service_near_left":     (-COURT_WIDTH_SINGLES / 2, -SERVICE_LINE_DIST),
    "service_near_right":    (COURT_WIDTH_SINGLES / 2, -SERVICE_LINE_DIST),
    "net_left":               (-COURT_WIDTH_DOUBLES / 2, 0.0),
    "net_right":               (COURT_WIDTH_DOUBLES / 2, 0.0),
}



class CourtCalibrator:


    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.H = None            # image -> world (meters)
        self.H_inv = None        # world -> image
        self.calibrated = False
        self.court_corners_img = None

    def detect_court_lines(self, frame):
 
        h, w = frame.shape[:2]
        hsv = cv2_safe_import().cvtColor(frame, cv2_safe_import().COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 170])
        upper_white = np.array([180, 60, 255])
        mask = cv2_safe_import().inRange(hsv, lower_white, upper_white)

        cv2 = cv2_safe_import()
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                 minLineLength=int(w * 0.15), maxLineGap=20)
        if lines is None or len(lines) < 4:
            logger.warning("Court line detection low-confidence, using default calibration")
            return None

        horiz, vert = [], []
        for l in lines[:, 0]:
            x1, y1, x2, y2 = l
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            length = np.hypot(x2 - x1, y2 - y1)
            if abs(angle) < 15 or abs(abs(angle) - 180) < 15:
                horiz.append((x1, y1, x2, y2, length))
            elif abs(abs(angle) - 90) < 25:
                vert.append((x1, y1, x2, y2, length))

        if len(horiz) < 2 or len(vert) < 2:
            return None

        horiz.sort(key=lambda l: min(l[1], l[3]))
        top = horiz[0]
        bottom = horiz[-1]

        def intersect(l1, l2):
            x1, y1, x2, y2 = l1[:4]
            x3, y3, x4, y4 = l2[:4]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                return None
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return (px, py)

        vert.sort(key=lambda l: min(l[0], l[2]))
        left_v = vert[0]
        right_v = vert[-1]

        fl = intersect(top, left_v)
        fr = intersect(top, right_v)
        nl = intersect(bottom, left_v)
        nr = intersect(bottom, right_v)

        if None in (fl, fr, nl, nr):
            return None

        corners = np.array([fl, fr, nr, nl], dtype=np.float32)
        if np.any(corners[:, 0] < -w * 0.5) or np.any(corners[:, 0] > w * 1.5):
            return None

        return corners

    def calibrate(self, frame):
        cv2 = cv2_safe_import()
        corners = self.detect_court_lines(frame)
        if corners is None:

            h, w = frame.shape[:2]
            corners = np.array([
                [w * 0.32, h * 0.30],   # far left
                [w * 0.68, h * 0.30],   # far right
                [w * 0.90, h * 0.95],   # near right
                [w * 0.10, h * 0.95],   # near left
            ], dtype=np.float32)
            self.calibrated = False
        else:
            self.calibrated = True

        self.court_corners_img = corners
        world_pts = np.array([
            COURT_KEYPOINTS_WORLD["baseline_far_left"],
            COURT_KEYPOINTS_WORLD["baseline_far_right"],
            COURT_KEYPOINTS_WORLD["baseline_near_right"],
            COURT_KEYPOINTS_WORLD["baseline_near_left"],
        ], dtype=np.float32)

        self.H = cv2.getPerspectiveTransform(corners, world_pts)
        self.H_inv = cv2.getPerspectiveTransform(world_pts, corners)
        logger.info(f"Court calibration {'(auto-detected)' if self.calibrated else '(default fallback)'} complete")
        return self.calibrated

    def image_to_court(self, px, py):
        """Project an image pixel (ground contact point) to court meters."""
        if self.H is None:
            return (0.0, 0.0)
        pt = np.array([[[px, py]]], dtype=np.float32)
        out = cv2_safe_import().perspectiveTransform(pt, self.H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def court_to_image(self, wx, wy):
        if self.H_inv is None:
            return (0, 0)
        pt = np.array([[[wx, wy]]], dtype=np.float32)
        out = cv2_safe_import().perspectiveTransform(pt, self.H_inv)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def is_in_court(self, wx, wy, margin=0.5):
        return (-COURT_WIDTH_DOUBLES / 2 - margin <= wx <= COURT_WIDTH_DOUBLES / 2 + margin and
                -COURT_LENGTH / 2 - margin <= wy <= COURT_LENGTH / 2 + margin)


_cv2_module = None
def cv2_safe_import():
    global _cv2_module
    if _cv2_module is None:
        import cv2
        _cv2_module = cv2
    return _cv2_module

@dataclass
class PlayerTrack:
    track_id: int
    bbox: tuple
    color_label: str
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    positions_court: deque = field(default_factory=lambda: deque(maxlen=2000))
    positions_px: deque = field(default_factory=lambda: deque(maxlen=2000))
    speeds: deque = field(default_factory=lambda: deque(maxlen=500))
    total_distance: float = 0.0


class PlayerTracker:
    """
    Two-stage IOU-based tracker inspired by ByteTrack: high-confidence
    detections are matched first via Hungarian assignment on IOU cost,
    then remaining low-confidence detections are matched to any still
    unmatched tracks, improving robustness during occlusion swings.
    Caps at two persistent tracks (the two players) once locked in,
    which is the standard simplification for singles broadcast feeds.
    """

    def __init__(self, max_age=30, iou_threshold=0.25, fps=30):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.fps = fps
        self.locked_ids = []  # the two main player IDs once established

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections, calibrator):
        """
        detections: list of (x1, y1, x2, y2, conf)
        Returns dict {track_id: PlayerTrack} of currently visible tracks.
        """
        det_boxes = [d[:4] for d in detections]

        if not self.tracks:
            # Bootstrap: take up to 2 highest-confidence detections (court-side players)
            sorted_dets = sorted(detections, key=lambda d: -d[4])[:2]
            for det in sorted_dets:
                self._spawn_track(det[:4])
        else:
            active_ids = list(self.tracks.keys())
            if det_boxes and active_ids:
                cost = np.zeros((len(active_ids), len(det_boxes)))
                for i, tid in enumerate(active_ids):
                    for j, db in enumerate(det_boxes):
                        cost[i, j] = 1.0 - self._iou(self.tracks[tid].bbox, db)
                row_ind, col_ind = linear_sum_assignment(cost)

                matched_tracks, matched_dets = set(), set()
                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] <= (1.0 - self.iou_threshold):
                        tid = active_ids[r]
                        self._update_track(tid, det_boxes[c], calibrator)
                        matched_tracks.add(tid)
                        matched_dets.add(c)

                # age-out unmatched tracks
                for tid in active_ids:
                    if tid not in matched_tracks:
                        self.tracks[tid].time_since_update += 1
                        self.tracks[tid].age += 1

                # if a slot is free (player briefly lost) and we have a strong
                # unmatched detection, and we're below 2 locked tracks, spawn it
                if len(self.tracks) < 2:
                    for j, db in enumerate(det_boxes):
                        if j not in matched_dets:
                            self._spawn_track(db)
                            break

        # prune stale tracks (only if we'd still have a chance to recover —
        # keep at least the locked identities alive across brief occlusion)
        to_remove = [tid for tid, t in self.tracks.items() if t.time_since_update > self.max_age]
        for tid in to_remove:
            del self.tracks[tid]

        return self.tracks

    def _spawn_track(self, bbox):
        color = "cyan" if len(self.locked_ids) == 0 else "magenta"
        t = PlayerTrack(track_id=self.next_id, bbox=bbox, color_label=color)
        self.tracks[self.next_id] = t
        self.locked_ids.append(self.next_id)
        self.next_id += 1

    def _update_track(self, tid, bbox, calibrator):
        t = self.tracks[tid]
        prev_px = t.positions_px[-1] if t.positions_px else None
        t.bbox = bbox
        t.hits += 1
        t.age += 1
        t.time_since_update = 0

        foot_x = (bbox[0] + bbox[2]) / 2.0
        foot_y = bbox[3]
        t.positions_px.append((foot_x, foot_y))
        wx, wy = calibrator.image_to_court(foot_x, foot_y)
        t.positions_court.append((wx, wy))

        if prev_px is not None:
            dpx = np.hypot(foot_x - prev_px[0], foot_y - prev_px[1])
            if len(t.positions_court) >= 2:
                wx0, wy0 = t.positions_court[-2]
                dmeters = np.hypot(wx - wx0, wy - wy0)
                # reject implausible teleports (mis-association) above ~12 m/s
                inst_speed = dmeters * self.fps
                if inst_speed < 12.0:
                    t.total_distance += dmeters
                    t.speeds.append(inst_speed)


# =====================================================================
# BALL TRACKING (Kalman filter, constant-acceleration model w/ gravity)
# =====================================================================
class BallKalmanTracker:
    """
    Constant-acceleration Kalman filter over image-plane (x, y) with a
    learned downward acceleration prior (gravity projected into image
    space), used to bridge occlusions and smooth noisy raw detections
    from the ball candidate detector.
    """

    def __init__(self, fps=30):
        self.fps = fps
        dt = 1.0 / fps
        self.kf = KalmanFilter(dim_x=6, dim_z=2)
        self.kf.F = np.array([
            [1, 0, dt, 0, 0.5 * dt ** 2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt ** 2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ])
        self.kf.R *= 6.0
        self.kf.P *= 80.0
        self.kf.Q = np.eye(6) * 0.6
        self.initialized = False
        self.missed = 0
        self.max_missed = 18
        self.history_px = deque(maxlen=600)
        self.history_court = deque(maxlen=600)
        self.history_z = deque(maxlen=600)  # estimated height above ground (m)

    def init(self, x, y):
        self.kf.x = np.array([x, y, 0, 0, 0, 0], dtype=float)
        self.initialized = True
        self.missed = 0

    def predict(self):
        self.kf.predict()
        return self.kf.x[0], self.kf.x[1]

    def update(self, measurement):
        if measurement is None:
            self.missed += 1
            if self.missed > self.max_missed:
                self.initialized = False
            return self.kf.x[0], self.kf.x[1]
        self.missed = 0
        self.kf.update(np.array(measurement))
        return self.kf.x[0], self.kf.x[1]

    @property
    def velocity_px(self):
        return self.kf.x[2], self.kf.x[3]


class BallCandidateDetector:
    """
    Small, fast-moving object detector specialised for the tennis ball:
    HSV color thresholding (high-vis yellow/lime) intersected with
    frame-differencing motion mask, then Hough-circle / contour shape
    validation. Designed as the measurement source feeding the Kalman
    tracker above, since the ball typically occupies <0.05% of frame
    area and standard YOLO anchors are unreliable at that scale.
    """

    def __init__(self):
        self.prev_gray = None
        self.lower = np.array([22, 60, 140])
        self.upper = np.array([42, 255, 255])

    def detect(self, frame, roi=None):
        cv2 = cv2_safe_import()
        h, w = frame.shape[:2]
        x_off, y_off = 0, 0
        search = frame
        if roi is not None:
            x1, y1, x2, y2 = roi
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                search = frame[y1:y2, x1:x2]
                x_off, y_off = x1, y1

        gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.lower, self.upper)

        motion_mask = None
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, motion_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.dilate(motion_mask, None, iterations=2)

        if roi is None:
            self.prev_gray = gray

        if motion_mask is not None:
            combined = cv2.bitwise_and(color_mask, motion_mask)
        else:
            combined = color_mask

        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if 2 <= area <= 400:
                (cx, cy), radius = cv2.minEnclosingCircle(c)
                circularity = area / (np.pi * radius ** 2 + 1e-6)
                if circularity > 0.35 and 1.0 <= radius <= 14:
                    candidates.append((cx + x_off, cy + y_off, circularity * area))

        if not candidates:
            # fall back to pure color blobs (helps on slow/served balls with low motion)
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if 2 <= area <= 250:
                    (cx, cy), radius = cv2.minEnclosingCircle(c)
                    if 1.0 <= radius <= 12:
                        candidates.append((cx + x_off, cy + y_off, area * 0.5))

        if not candidates:
            return None
        candidates.sort(key=lambda c: -c[2])
        return candidates[0][0], candidates[0][1]


# =====================================================================
# PHYSICS ENGINE
# =====================================================================
class PhysicsEngine:
    """
    Fits parabolic (projectile) motion to short windows of the ball's
    court-space trajectory, detects bounces via local-minima + velocity
    sign-change in vertical motion, and derives speed / launch / impact
    angles from the fitted model. Court-space z (height) is estimated
    from apparent ball size / vertical pixel offset against the
    calibrated ground plane, smoothed into a relative scale.
    """

    def __init__(self, fps=30):
        self.fps = fps

    def fit_parabola(self, t, y):
        """Fit y = a*t^2 + b*t + c, returns (a, b, c) or None."""
        if len(t) < 4:
            return None
        try:
            coeffs = np.polyfit(t, y, 2)
            return tuple(coeffs)
        except Exception:
            return None

    def detect_bounce(self, height_series, court_xy_series, fps):
        """
        Scans a rolling buffer of estimated heights for local minima
        near ground level with a velocity sign reversal -> bounce event.
        Returns list of bounce indices.
        """
        bounces = []
        if len(height_series) < 5:
            return bounces
        h = np.array(height_series)
        for i in range(2, len(h) - 2):
            if h[i] < h[i - 1] and h[i] < h[i + 1] and h[i] < 0.35:
                # confirm a genuine velocity reversal, not sensor noise
                v_before = h[i - 1] - h[i - 2]
                v_after = h[i + 2] - h[i + 1]
                if v_before < 0 and v_after > 0:
                    bounces.append(i)
        return bounces

    def estimate_speed_kmh(self, court_xy_series, fps, window=3):
        """Estimate instantaneous speed (km/h) from last `window` court-space points."""
        if len(court_xy_series) < window + 1:
            return 0.0
        pts = list(court_xy_series)[-(window + 1):]
        dists = [np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
        avg_dist = np.mean(dists)
        mps = avg_dist * fps
        return float(mps * 3.6)

    def launch_angle(self, court_xy_series, height_series, fps, n=5):
        """Estimate launch angle (degrees from horizontal) at the start of a shot."""
        if len(court_xy_series) < n or len(height_series) < n:
            return 0.0
        xy = list(court_xy_series)[:n]
        hz = list(height_series)[:n]
        horiz_dist = np.hypot(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1])
        vert_dist = hz[-1] - hz[0]
        if horiz_dist < 1e-3:
            return 90.0 if vert_dist > 0 else -90.0
        return float(np.degrees(np.arctan2(vert_dist, horiz_dist)))

    def impact_angle(self, court_xy_series, height_series, fps, n=5):
        if len(court_xy_series) < n or len(height_series) < n:
            return 0.0
        xy = list(court_xy_series)[-n:]
        hz = list(height_series)[-n:]
        horiz_dist = np.hypot(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1])
        vert_dist = hz[0] - hz[-1]
        if horiz_dist < 1e-3:
            return 90.0
        return float(np.degrees(np.arctan2(vert_dist, horiz_dist)))


# =====================================================================
# STATISTICS AGGREGATION
# =====================================================================
class MatchStatistics:
    """
    Aggregates per-player movement statistics, ball/shot statistics,
    rally segmentation, and derived aggression/defensive scoring used
    by the dashboard, heatmaps and serve-statistics outputs.
    """

    def __init__(self, fps=30):
        self.fps = fps
        self.rally_count = 0
        self.current_rally_shots = 0
        self.rally_lengths = []
        self.shot_speeds = []
        self.serve_speeds = []
        self.bounce_locations = []     # list of (wx, wy)
        self.ball_speed_log = []
        self.frame_count = 0
        self.player_heatpoints = defaultdict(list)   # track_id -> [(wx,wy), ...]
        self.player_aggression = defaultdict(list)   # track_id -> shot-speed samples while striking
        self.last_bounce_frame = -999
        self.in_rally = False

    def log_frame(self, players, ball_speed_kmh):
        self.frame_count += 1
        for tid, track in players.items():
            if track.positions_court:
                self.player_heatpoints[tid].append(track.positions_court[-1])
        if ball_speed_kmh and ball_speed_kmh > 8:
            self.ball_speed_log.append(ball_speed_kmh)
            if not self.in_rally:
                self.in_rally = True
                self.rally_count += 1
                self.current_rally_shots = 0

    def log_bounce(self, wx, wy, frame_idx):
        self.bounce_locations.append((wx, wy))
        if frame_idx - self.last_bounce_frame > self.fps * 0.3:
            self.current_rally_shots += 1
        self.last_bounce_frame = frame_idx

    def log_rally_end(self):
        if self.in_rally:
            self.rally_lengths.append(self.current_rally_shots)
        self.in_rally = False

    def player_summary(self, players):
        summary = {}
        for tid, t in players.items():
            speeds = list(t.speeds)
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            sprint_speed = float(np.max(speeds)) if speeds else 0.0
            pts = self.player_heatpoints.get(tid, [])
            coverage = self._estimate_coverage(pts)
            aggression = self._aggression_score(speeds)
            defensive = self._defensive_score(pts)
            summary[tid] = {
                "color": t.color_label,
                "distance_m": round(t.total_distance, 1),
                "avg_speed_kmh": round(avg_speed * 3.6, 1),
                "sprint_speed_kmh": round(sprint_speed * 3.6, 1),
                "coverage_pct": round(coverage, 1),
                "aggression_score": round(aggression, 1),
                "defensive_score": round(defensive, 1),
            }
        return summary

    @staticmethod
    def _estimate_coverage(points, court_w=COURT_WIDTH_DOUBLES, court_l=COURT_LENGTH, cell=1.0):
        if not points:
            return 0.0
        cells = set()
        for wx, wy in points:
            cells.add((int(wx // cell), int(wy // cell)))
        total_cells = (court_w / cell) * (court_l / 2 / cell)
        return min(100.0, 100.0 * len(cells) / max(total_cells, 1))

    @staticmethod
    def _aggression_score(speeds):
        if not speeds:
            return 0.0
        fast_moves = [s for s in speeds if s > 3.5]
        return min(100.0, 100.0 * len(fast_moves) / max(len(speeds), 1))

    @staticmethod
    def _defensive_score(points):
        if len(points) < 10:
            return 0.0
        ys = [p[1] for p in points]
        # players spending more time deep behind the baseline area score
        # higher on "defensive positioning"
        deep_count = sum(1 for y in ys if abs(y) > COURT_LENGTH / 2 * 0.6)
        deep_frac = deep_count / len(ys)
        return float(min(100.0, deep_frac * 100))

    def serve_stats(self):
        if not self.serve_speeds:
            return {"count": 0, "avg_kmh": 0.0, "max_kmh": 0.0, "min_kmh": 0.0}
        arr = np.array(self.serve_speeds)
        return {
            "count": len(arr),
            "avg_kmh": round(float(arr.mean()), 1),
            "max_kmh": round(float(arr.max()), 1),
            "min_kmh": round(float(arr.min()), 1),
        }

    def shot_stats(self):
        if not self.shot_speeds:
            return {"count": 0, "avg_kmh": 0.0, "max_kmh": 0.0}
        arr = np.array(self.shot_speeds)
        return {
            "count": len(arr),
            "avg_kmh": round(float(arr.mean()), 1),
            "max_kmh": round(float(arr.max()), 1),
        }
