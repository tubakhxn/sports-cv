import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import logging

from analytics import COURT_LENGTH, COURT_WIDTH_DOUBLES, COURT_WIDTH_SINGLES, \
    SERVICE_LINE_DIST, NET_HEIGHT_CENTER

logger = logging.getLogger("hawkeye.visualization")

# =====================================================================
# COLOR PALETTE (BGR for OpenCV)
# =====================================================================
NEON_CYAN = (255, 255, 0)
NEON_MAGENTA = (255, 0, 255)
NEON_YELLOW = (0, 255, 255)
NEON_GREEN = (60, 255, 120)
NEON_ORANGE = (0, 140, 255)
COURT_LINE_COLOR = (60, 230, 255)
HUD_BG = (25, 20, 15)
HUD_TEXT = (235, 235, 235)
WHITE = (255, 255, 255)

PLAYER_COLORS = {"cyan": NEON_CYAN, "magenta": NEON_MAGENTA}


def _alpha_blend_rect(img, pt1, pt2, color, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _put_text(img, text, org, scale=0.55, color=HUD_TEXT, thickness=1, font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


# =====================================================================
# LIVE 2D BROADCAST OVERLAY
# =====================================================================
class BroadcastOverlay:
    """
    Renders the cinematic Hawk-Eye style HUD directly onto each video
    frame: court line overlay, tracked player boxes with motion trails,
    ball trail + speed tag, bounce markers, rally counter, FPS counter
    and a live match-stats panel.
    """

    def __init__(self, frame_w, frame_h):
        self.w = frame_w
        self.h = frame_h
        self.ball_trail = []
        self.bounce_markers = []   # (px, py, ttl)
        self.max_trail = 28

    def draw_court_lines(self, frame, calibrator):
        if calibrator.court_corners_img is None:
            return frame
        pts = calibrator.court_corners_img.astype(int)
        overlay = frame.copy()
        cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], True, COURT_LINE_COLOR, 2, cv2.LINE_AA)

        def proj(wx, wy):
            x, y = calibrator.court_to_image(wx, wy)
            return (int(x), int(y))

        sl = proj(-COURT_WIDTH_SINGLES / 2, SERVICE_LINE_DIST)
        sr = proj(COURT_WIDTH_SINGLES / 2, SERVICE_LINE_DIST)
        nl = proj(-COURT_WIDTH_SINGLES / 2, -SERVICE_LINE_DIST)
        nr = proj(COURT_WIDTH_SINGLES / 2, -SERVICE_LINE_DIST)
        cv2.line(overlay, sl, sr, COURT_LINE_COLOR, 2, cv2.LINE_AA)
        cv2.line(overlay, nl, nr, COURT_LINE_COLOR, 2, cv2.LINE_AA)
        cmid_far = proj(0, SERVICE_LINE_DIST)
        cmid_near = proj(0, -SERVICE_LINE_DIST)
        cv2.line(overlay, cmid_far, cmid_near, COURT_LINE_COLOR, 2, cv2.LINE_AA)

        net_l = proj(-COURT_WIDTH_DOUBLES / 2, 0)
        net_r = proj(COURT_WIDTH_DOUBLES / 2, 0)
        cv2.line(overlay, net_l, net_r, (220, 255, 255), 3, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        return frame

    def draw_players(self, frame, players):
        for tid, t in players.items():
            if t.time_since_update > 0:
                continue
            color = PLAYER_COLORS.get(t.color_label, NEON_CYAN)
            x1, y1, x2, y2 = [int(v) for v in t.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            pts = list(t.positions_px)[-40:]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pt1, pt2, trail_color, 2, cv2.LINE_AA)

            speed_kmh = (t.speeds[-1] * 3.6) if t.speeds else 0.0
            label = f"P{tid} {speed_kmh:4.1f} km/h"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            _alpha_blend_rect(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), (0, 0, 0), 0.5)
            _put_text(frame, label, (x1 + 5, y1 - 6), scale=0.5, color=color)
        return frame

    def update_ball(self, frame, ball_px, speed_kmh=None, is_bounce=False, calibrator=None):
        if ball_px is not None:
            self.ball_trail.append(ball_px)
            if len(self.ball_trail) > self.max_trail:
                self.ball_trail.pop(0)
            if is_bounce:
                self.bounce_markers.append([ball_px[0], ball_px[1], 24])

        for i in range(1, len(self.ball_trail)):
            alpha = i / len(self.ball_trail)
            pt1 = tuple(int(v) for v in self.ball_trail[i - 1])
            pt2 = tuple(int(v) for v in self.ball_trail[i])
            c = tuple(int(ch * alpha) for ch in NEON_YELLOW)
            cv2.line(frame, pt1, pt2, c, max(1, int(3 * alpha)), cv2.LINE_AA)

        if self.ball_trail:
            bx, by = self.ball_trail[-1]
            cv2.circle(frame, (int(bx), int(by)), 7, NEON_YELLOW, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(bx), int(by)), 10, (255, 255, 255), 1, cv2.LINE_AA)
            if speed_kmh:
                _put_text(frame, f"{speed_kmh:.0f} km/h", (int(bx) + 12, int(by) - 8),
                           scale=0.5, color=NEON_YELLOW)

        for m in self.bounce_markers:
            px, py, ttl = m
            radius = int(22 * (1 - ttl / 24)) + 4
            alpha_t = ttl / 24
            color = tuple(int(c * alpha_t) for c in NEON_ORANGE)
            cv2.circle(frame, (int(px), int(py)), radius, color, 2, cv2.LINE_AA)
            m[2] -= 1
        self.bounce_markers = [m for m in self.bounce_markers if m[2] > 0]
        return frame

    def draw_hud(self, frame, stats_panel, fps, rally_count, frame_idx, total_frames):
        h, w = frame.shape[:2]

        panel_w, panel_h = 300, 92
        _alpha_blend_rect(frame, (10, 10), (10 + panel_w, 10 + panel_h), HUD_BG, 0.6)
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (90, 90, 90), 1)
        _put_text(frame, "HAWK-EYE 3D INTELLIGENCE", (20, 32), scale=0.55, color=NEON_GREEN, thickness=1)
        y = 54
        for line in stats_panel:
            _put_text(frame, line, (20, y), scale=0.45, color=HUD_TEXT)
            y += 18

        tr_w = 230
        _alpha_blend_rect(frame, (w - tr_w - 10, 10), (w - 10, 70), HUD_BG, 0.6)
        cv2.rectangle(frame, (w - tr_w - 10, 10), (w - 10, 70), (90, 90, 90), 1)
        _put_text(frame, f"FPS: {fps:5.1f}", (w - tr_w + 5, 32), scale=0.5, color=NEON_GREEN)
        _put_text(frame, f"RALLY #{rally_count}", (w - tr_w + 5, 54), scale=0.5, color=NEON_ORANGE)

        bar_y = h - 18
        cv2.rectangle(frame, (10, bar_y), (w - 10, bar_y + 6), (60, 60, 60), -1)
        prog = int((w - 20) * (frame_idx / max(total_frames, 1)))
        cv2.rectangle(frame, (10, bar_y), (10 + prog, bar_y + 6), NEON_GREEN, -1)
        return frame


# =====================================================================
# LIVE 3D HAWK-EYE WINDOW
# =====================================================================
class HawkEye3DRenderer:
    """
    Renders the secondary 'HawkEye 3D Trajectory' window: a 3D wireframe
    court, net plane, player markers projected onto the ground plane,
    and the ball's live + historical 3D trajectory (with bounce points),
    matching the look of professional broadcast Hawk-Eye replay graphics.
    """

    def __init__(self, fig_w=5.0, fig_h=5.5, dpi=120):
        self.fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.patch.set_facecolor("#0c0f0c")
        self._setup_axes()

    def _setup_axes(self):
        ax = self.ax
        ax.set_facecolor("#0c0f0c")
        ax.set_xlim(-COURT_WIDTH_DOUBLES / 2 - 1, COURT_WIDTH_DOUBLES / 2 + 1)
        ax.set_ylim(-COURT_LENGTH / 2 - 1, COURT_LENGTH / 2 + 1)
        ax.set_zlim(0, 4.0)
        ax.set_box_aspect((COURT_WIDTH_DOUBLES + 2, COURT_LENGTH + 2, 8))
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor((0.03, 0.05, 0.03, 0.9))
        ax.grid(False)
        ax.tick_params(colors="#7fdc7f", labelsize=6)
        ax.view_init(elev=22, azim=-60)

    def _draw_court(self):
        ax = self.ax
        hw_d, hw_s, hl, sd = COURT_WIDTH_DOUBLES / 2, COURT_WIDTH_SINGLES / 2, COURT_LENGTH / 2, SERVICE_LINE_DIST
        lines = [
            [(-hw_d, -hl, 0), (hw_d, -hl, 0)], [(-hw_d, hl, 0), (hw_d, hl, 0)],
            [(-hw_d, -hl, 0), (-hw_d, hl, 0)], [(hw_d, -hl, 0), (hw_d, hl, 0)],
            [(-hw_s, -hl, 0), (-hw_s, hl, 0)], [(hw_s, -hl, 0), (hw_s, hl, 0)],
            [(-hw_s, -sd, 0), (hw_s, -sd, 0)], [(-hw_s, sd, 0), (hw_s, sd, 0)],
            [(0, -sd, 0), (0, sd, 0)],
        ]
        for (x1, y1, z1), (x2, y2, z2) in lines:
            ax.plot([x1, x2], [y1, y2], [z1, z2], color="#cfeccf", linewidth=1.1, alpha=0.85)

        net_x = np.linspace(-hw_d, hw_d, 2)
        net_z = np.linspace(0, NET_HEIGHT_CENTER, 2)
        NX, NZ = np.meshgrid(net_x, net_z)
        NY = np.zeros_like(NX)
        ax.plot_surface(NX, NY, NZ, color="#3fd0c9", alpha=0.25, linewidth=0)
        ax.plot([-hw_d, hw_d], [0, 0], [NET_HEIGHT_CENTER, NET_HEIGHT_CENTER], color="#3fd0c9", linewidth=1.5)

    def render(self, frame_idx, total_frames, players_court, ball_court_history, ball_height_history,
               bounce_points, current_ball_pos=None, current_ball_height=0.0):
        ax = self.ax
        ax.cla()
        self._setup_axes()
        self._draw_court()
        ax.set_title(f"Hawkeye 3D Trajectory | Frame {frame_idx} / {total_frames}",
                      color="#dfffe0", fontsize=9, pad=2)

        for tid, (wx, wy, color_label) in players_court.items():
            c = "#00e5ff" if color_label == "cyan" else "#ff35e0"
            ax.scatter([wx], [wy], [0], color=c, s=70, marker="o", edgecolor="white", linewidth=0.6)

        if len(ball_court_history) > 1:
            xs = [p[0] for p in ball_court_history]
            ys = [p[1] for p in ball_court_history]
            zs = list(ball_height_history)
            ax.plot(xs, ys, zs, color="#ffe600", linewidth=1.6, alpha=0.85)

        for bx, by in bounce_points[-12:]:
            ax.scatter([bx], [by], [0], color="#ff8a00", s=45, marker="x")

        if current_ball_pos is not None:
            ax.scatter([current_ball_pos[0]], [current_ball_pos[1]], [current_ball_height],
                       color="#ffe600", s=90, edgecolor="white", linewidth=0.8)

        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        return img


# =====================================================================
# POST-MATCH OUTPUT GRAPHICS
# =====================================================================
def _draw_static_court(ax, fc="#0e3d1e"):
    hw_d, hw_s, hl, sd = COURT_WIDTH_DOUBLES / 2, COURT_WIDTH_SINGLES / 2, COURT_LENGTH / 2, SERVICE_LINE_DIST
    ax.set_facecolor(fc)
    ax.add_patch(Rectangle((-hw_d, -hl), COURT_WIDTH_DOUBLES, COURT_LENGTH,
                            fill=False, edgecolor="white", linewidth=1.8))
    ax.add_patch(Rectangle((-hw_s, -hl), COURT_WIDTH_SINGLES, COURT_LENGTH,
                            fill=False, edgecolor="white", linewidth=1.2))
    ax.plot([-hw_s, hw_s], [-sd, -sd], color="white", linewidth=1.2)
    ax.plot([-hw_s, hw_s], [sd, sd], color="white", linewidth=1.2)
    ax.plot([0, 0], [-sd, sd], color="white", linewidth=1.2)
    ax.plot([-hw_d, hw_d], [0, 0], color="#3fd0c9", linewidth=2.4)
    ax.set_xlim(-hw_d - 1.5, hw_d + 1.5)
    ax.set_ylim(-hl - 1.5, hl + 1.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def generate_heatmaps(stats, players, out_path="player_heatmaps.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), facecolor="#0a0a0a")
    titles = {"cyan": "Player 1 (Neon Cyan)", "magenta": "Player 2 (Neon Magenta)"}
    cmaps = {"cyan": "cool", "magenta": "spring"}

    for ax, (tid, t) in zip(axes, players.items()):
        _draw_static_court(ax)
        pts = stats.player_heatpoints.get(tid, [])
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.hexbin(xs, ys, gridsize=28, cmap=cmaps.get(t.color_label, "cool"),
                      alpha=0.75, mincnt=1, extent=[-COURT_WIDTH_DOUBLES/2-1.5, COURT_WIDTH_DOUBLES/2+1.5,
                                                     -COURT_LENGTH/2-1.5, COURT_LENGTH/2+1.5])
        ax.set_title(titles.get(t.color_label, f"Player {tid}"), color="white", fontsize=13)

    fig.suptitle("Court Positioning Heatmaps", color="white", fontsize=16, y=0.98)
    fig.text(0.5, 0.02, "Tennis Hawk-Eye 3D Intelligence System  |  dev: tubakhxn",
              color="#888888", ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def generate_dashboard(stats, players, summary, out_path="match_dashboard.png"):
    fig = plt.figure(figsize=(15, 9), facecolor="#0a0a0a")
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    ax_court = fig.add_subplot(gs[0:2, 0])
    _draw_static_court(ax_court)
    for tid, t in players.items():
        pts = list(t.positions_court)[-300:]
        if pts:
            xs, ys = zip(*pts)
            c = "#00e5ff" if t.color_label == "cyan" else "#ff35e0"
            ax_court.plot(xs, ys, color=c, alpha=0.5, linewidth=1)
            ax_court.scatter(xs[-1], ys[-1], color=c, s=60, edgecolor="white", zorder=5)
    if stats.bounce_locations:
        bx, by = zip(*stats.bounce_locations)
        ax_court.scatter(bx, by, color="#ff8a00", marker="x", s=30, label="Bounces")
    ax_court.set_title("Rally Movement & Bounce Map", color="white", fontsize=11)

    ax_bars = fig.add_subplot(gs[0, 1])
    metrics = ["distance_m", "avg_speed_kmh", "sprint_speed_kmh"]
    labels = ["Distance (m)", "Avg Spd (km/h)", "Sprint (km/h)"]
    x = np.arange(len(metrics))
    width = 0.35
    colors = {"cyan": "#00e5ff", "magenta": "#ff35e0"}
    for i, (tid, s) in enumerate(summary.items()):
        vals = [s[m] for m in metrics]
        ax_bars.bar(x + (i - 0.5) * width, vals, width, label=f"P{tid}", color=colors.get(s["color"], "gray"))
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(labels, color="white", fontsize=8)
    ax_bars.tick_params(colors="white")
    ax_bars.set_facecolor("#111111")
    ax_bars.legend(facecolor="#222222", labelcolor="white", fontsize=8)
    ax_bars.set_title("Player Movement Metrics", color="white", fontsize=11)

    ax_scores = fig.add_subplot(gs[0, 2])
    score_metrics = ["aggression_score", "defensive_score", "coverage_pct"]
    score_labels = ["Aggression", "Defensive", "Coverage %"]
    xs = np.arange(len(score_metrics))
    for i, (tid, s) in enumerate(summary.items()):
        vals = [s[m] for m in score_metrics]
        ax_scores.bar(xs + (i - 0.5) * width, vals, width, color=colors.get(s["color"], "gray"))
    ax_scores.set_xticks(xs)
    ax_scores.set_xticklabels(score_labels, color="white", fontsize=8)
    ax_scores.tick_params(colors="white")
    ax_scores.set_facecolor("#111111")
    ax_scores.set_ylim(0, 100)
    ax_scores.set_title("Playing Style Scores", color="white", fontsize=11)

    ax_speed = fig.add_subplot(gs[1, 1:3])
    if stats.ball_speed_log:
        ax_speed.plot(stats.ball_speed_log, color="#ffe600", linewidth=1)
        ax_speed.fill_between(range(len(stats.ball_speed_log)), stats.ball_speed_log, color="#ffe600", alpha=0.2)
    ax_speed.set_facecolor("#111111")
    ax_speed.tick_params(colors="white")
    ax_speed.set_title("Ball Speed Over Match (km/h)", color="white", fontsize=11)

    ax_rally = fig.add_subplot(gs[2, 0])
    if stats.rally_lengths:
        ax_rally.hist(stats.rally_lengths, bins=range(0, max(stats.rally_lengths) + 2),
                      color="#3fd0c9", edgecolor="white")
    ax_rally.set_facecolor("#111111")
    ax_rally.tick_params(colors="white")
    ax_rally.set_title("Rally Length Distribution (shots)", color="white", fontsize=10)

    ax_text = fig.add_subplot(gs[2, 1:3])
    ax_text.axis("off")
    serve = stats.serve_stats()
    shot = stats.shot_stats()
    summary_txt = (
        f"MATCH SUMMARY\n\n"
        f"Total Rallies: {stats.rally_count}\n"
        f"Total Shots Tracked: {shot['count']}\n"
        f"Avg Shot Speed: {shot['avg_kmh']} km/h   Max: {shot['max_kmh']} km/h\n"
        f"Serve Count: {serve['count']}   Avg: {serve['avg_kmh']} km/h   Max: {serve['max_kmh']} km/h\n"
        f"Bounces Detected: {len(stats.bounce_locations)}"
    )
    ax_text.text(0.02, 0.5, summary_txt, color="white", fontsize=11, va="center", family="monospace")

    fig.suptitle("Tennis Hawk-Eye 3D Intelligence System — Match Dashboard", color="#3fd0c9", fontsize=16)
    fig.text(0.5, 0.01, "dev: tubakhxn", color="#888888", ha="center", fontsize=9)
    plt.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def generate_trajectory_analysis(stats, ball_court_history, ball_height_history, out_path="trajectory_analysis.png"):
    fig = plt.figure(figsize=(14, 8), facecolor="#0a0a0a")
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.set_facecolor("#0a0a0a")
    if len(ball_court_history) > 1:
        xs = [p[0] for p in ball_court_history]
        ys = [p[1] for p in ball_court_history]
        zs = list(ball_height_history)
        ax3d.scatter(xs, ys, zs, c=range(len(xs)), cmap="autumn", s=6)
        ax3d.plot(xs, ys, zs, color="#ffe600", alpha=0.4, linewidth=0.8)
    if stats.bounce_locations:
        bx, by = zip(*stats.bounce_locations)
        ax3d.scatter(bx, by, [0] * len(bx), color="#ff8a00", marker="x", s=40)
    ax3d.set_title("Full Match Ball Trajectory (3D)", color="white", fontsize=11)
    ax3d.tick_params(colors="white", labelsize=6)

    ax_height = fig.add_subplot(gs[0, 1])
    if ball_height_history:
        ax_height.plot(ball_height_history, color="#ffe600", linewidth=1)
    ax_height.set_facecolor("#111111")
    ax_height.tick_params(colors="white")
    ax_height.set_title("Ball Arc Height Over Time", color="white", fontsize=10)

    ax_top = fig.add_subplot(gs[1, 1])
    _draw_static_court(ax_top)
    if len(ball_court_history) > 1:
        xs = [p[0] for p in ball_court_history]
        ys = [p[1] for p in ball_court_history]
        ax_top.scatter(xs, ys, c=range(len(xs)), cmap="autumn", s=4)
    if stats.bounce_locations:
        bx, by = zip(*stats.bounce_locations)
        ax_top.scatter(bx, by, color="#ff5050", marker="x", s=35)
    ax_top.set_title("Top-Down Bounce Map", color="white", fontsize=10)

    fig.suptitle("Trajectory & Physics Analysis", color="#3fd0c9", fontsize=15)
    fig.text(0.5, 0.01, "dev: tubakhxn", color="#888888", ha="center", fontsize=9)
    plt.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def generate_serve_statistics(stats, out_path="serve_statistics.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="#0a0a0a")
    serve = stats.serve_stats()

    ax1 = axes[0]
    ax1.set_facecolor("#111111")
    if stats.serve_speeds:
        ax1.plot(stats.serve_speeds, marker="o", color="#ffe600", linewidth=1.2, markersize=4)
    ax1.tick_params(colors="white")
    ax1.set_title("Serve Speed Progression (km/h)", color="white", fontsize=11)

    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_facecolor("#0a0a0a")
    txt = (
        f"SERVE STATISTICS\n\n"
        f"Total Serves: {serve['count']}\n"
        f"Average Speed: {serve['avg_kmh']} km/h\n"
        f"Fastest Serve: {serve['max_kmh']} km/h\n"
        f"Slowest Serve: {serve['min_kmh']} km/h"
    )
    ax2.text(0.05, 0.5, txt, color="white", fontsize=13, va="center", family="monospace")

    fig.suptitle("Serve Analytics", color="#3fd0c9", fontsize=15)
    fig.text(0.5, 0.01, "dev: tubakhxn", color="#888888", ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    plt.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")
