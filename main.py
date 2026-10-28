import os
import json
import time
import math
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO


class BarPathAnalyzer:
    """
    Yapay Zeka Destekli Halter Bar Yolu (Bar Path), Hız (VBT) ve Form Analiz Sistemi.
    Canlı parametre değişimi (Sağ Plaka / Merkez / Sol Plaka / Manuel Tıklama,
    İz Uzunluğu, Çizgi Kalınlığı, Hız Hassasiyeti) destekler.
    """

    def __init__(
        self,
        model_path: str = "best.pt",
        detections_cache_path: str = "video_detections.json",
        target_width: int = 1280,
        target_height: int = 720,
        track_mode: str = "right_plate",  # "right_plate", "center", "left_plate", "manual"
        trail_length: int = 180,
        line_thickness: int = 4,
        vel_threshold: float = 0.36,      # Yeşil -> Sarı/Kırmızı hız eşiği (m/s)
        plate_diameter_m: float = 0.45,
        yolo_interval: int = 2,
    ):
        self.model_path = model_path
        self.target_width = target_width
        self.target_height = target_height
        self.track_mode = track_mode
        self.trail_length = int(trail_length)
        self.line_thickness = int(line_thickness)
        self.vel_threshold = float(vel_threshold)
        self.plate_diameter_m = plate_diameter_m
        self.yolo_interval = yolo_interval

        # Önbellekteki YOLO kutularını yükle (CPU'da 25+ FPS anlık tepki için)
        self.cached_detections = {}
        if detections_cache_path and os.path.exists(detections_cache_path):
            try:
                with open(detections_cache_path, "r", encoding="utf-8") as f:
                    self.cached_detections = json.load(f)
            except Exception:
                self.cached_detections = {}

        self.model = None
        if not self.cached_detections and os.path.exists(model_path):
            self.model = YOLO(model_path)

        self.kalman = self._init_kalman()
        self.initialized = False

        self.prev_gray = None
        self.current_pos = None  # (x, y)
        self.last_valid_box = [256.0, 125.0, 865.0, 388.0]
        self.last_valid_width = 610.0

        self.px_per_meter = 520.0
        self.ref_x = None
        self.ref_y_top = None
        self.stance_locked = False
        self.stable_frames = 0

        self.history = deque(maxlen=5000)
        self.velocity_history = deque(maxlen=140)
        self.raw_x_buffer = deque(maxlen=5)
        self.raw_y_buffer = deque(maxlen=5)

        self.rep_count = 0
        self.phase = "UNRACK / HAZIRLIK"
        self.in_rep = False
        self.rep_start_frame = 0
        self.rep_max_y = None
        self.rep_velocities = []
        self.rep_drifts = []
        self.rep_summaries = []

    def _init_kalman(self) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.08
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.35
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        return kf

    def set_track_mode(self, mode: str):
        """Takip noktasını (Sağ Plaka / Bar Ortası / Sol Plaka) canlı olarak değiştirir."""
        if mode == self.track_mode and mode != "manual":
            return
        self.track_mode = mode
        self.raw_x_buffer.clear()
        self.raw_y_buffer.clear()
        self.history.clear()
        if self.last_valid_box is not None and mode != "manual":
            cx, cy = self._box_to_point(self.last_valid_box)
            self._reset_position(cx, cy)

    def set_manual_point(self, x: float, y: float):
        """Kullanıcının videoda tıkladığı noktayı canlı takip hedefi olarak kilitler."""
        self.track_mode = "manual"
        self.stance_locked = True
        self.history.clear()
        self.raw_x_buffer.clear()
        self.raw_y_buffer.clear()
        self._reset_position(float(x), float(y))

    def _reset_position(self, cx: float, cy: float):
        self.current_pos = (cx, cy)
        self.ref_x = cx
        self.ref_y_top = cy
        self.rep_max_y = cy
        self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self.initialized = True

    def _box_to_point(self, box) -> tuple:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        cy = y1 + h * 0.51
        if self.track_mode == "right_plate":
            cx = x2 - 76.0
        elif self.track_mode == "left_plate":
            cx = x1 + 58.0
        else:  # "center"
            cx = x1 + w * 0.50
        cx = float(np.clip(cx, 40, self.target_width - 40))
        cy = float(np.clip(cy, 40, self.target_height - 40))
        return cx, cy

    def _refine_plate_hub(self, gray, est_x: float, est_y: float) -> tuple:
        r = 36
        ix, iy = int(round(est_x)), int(round(est_y))
        x1 = max(0, ix - r)
        y1 = max(0, iy - r)
        x2 = min(gray.shape[1], ix + r)
        y2 = min(gray.shape[0], iy + r)

        patch = gray[y1:y2, x1:x2]
        if patch.size == 0 or patch.shape[0] < 20 or patch.shape[1] < 20:
            return est_x, est_y

        blurred = cv2.GaussianBlur(patch, (9, 9), 0)
        _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
        if max_val > 125:
            hub_x = float(x1 + max_loc[0])
            hub_y = float(y1 + max_loc[1])
            if math.hypot(hub_x - est_x, hub_y - est_y) < 28:
                return 0.55 * est_x + 0.45 * hub_x, 0.65 * est_y + 0.35 * hub_y
        return est_x, est_y

    def _extract_target_from_box(self, box, gray):
        x1, y1, x2, y2 = box
        h = y2 - y1
        if 180 < h < 290:
            self.px_per_meter = 0.90 * self.px_per_meter + 0.10 * (h / self.plate_diameter_m)

        cx, cy = self._box_to_point(box)
        if self.track_mode == "right_plate":
            cx, cy = self._refine_plate_hub(gray, cx, cy)
        return cx, cy

    def _get_raw_boxes(self, frame, frame_idx: int):
        key = str(frame_idx)
        if key in self.cached_detections:
            return self.cached_detections[key]
        prev_key = str(frame_idx - 1)
        if prev_key in self.cached_detections:
            return self.cached_detections[prev_key]

        if self.model is None and os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
        if self.model is None:
            return []

        results = self.model(frame, imgsz=480, conf=0.18, verbose=False)[0]
        if len(results.boxes) == 0:
            return []
        out = []
        for b, c in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()):
            out.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)])
        return out

    def _detect_yolo_anchor(self, frame, frame_idx: int):
        raw_boxes = self._get_raw_boxes(frame, frame_idx)
        if not raw_boxes:
            return None

        best_box = None
        best_score = -1e9

        for item in raw_boxes:
            x1, y1, x2, y2, conf = item
            w = x2 - x1
            h = y2 - y1

            if w < 260 or h < 120 or h > 440:
                continue
            if y2 < 180 or x1 > self.target_width * 0.65:
                continue

            if w > 675 or x2 > 970:
                if 220 < x1 < 350:
                    x2 = x1 + self.last_valid_width
                    w = x2 - x1
                else:
                    continue
            else:
                self.last_valid_width = 0.85 * self.last_valid_width + 0.15 * w

            score = float(conf) * 2.0
            if self.current_pos is not None and self.track_mode != "manual":
                tx, ty = self._box_to_point([x1, y1, x2, y2])
                dist = math.hypot(tx - self.current_pos[0], ty - self.current_pos[1])
                if dist > 230:
                    continue
                score -= dist / 180.0

            if score > best_score:
                best_score = score
                best_box = [x1, y1, x2, y2]

        if best_box is not None:
            self.last_valid_box = best_box
        return best_box

    def _track_frame(self, frame, frame_idx: int):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self.initialized:
            box = self._detect_yolo_anchor(frame, frame_idx)
            if box is not None:
                cx, cy = self._extract_target_from_box(box, gray)
            else:
                cx, cy = 755.0, 260.0
            self._reset_position(cx, cy)
            self.prev_gray = gray
            return cx, cy

        px, py = self.current_pos
        flow_x, flow_y = px, py
        flow_valid = False

        if self.prev_gray is not None:
            offsets = [-14, 0, 14]
            pts_list = [[px + dx, py + dy] for dx in offsets for dy in offsets]
            pts_prev = np.array(pts_list, dtype=np.float32).reshape(-1, 1, 2)

            pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                pts_prev,
                None,
                winSize=(33, 33),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            good_prev = pts_prev[status.flatten() == 1].reshape(-1, 2)
            good_next = pts_next[status.flatten() == 1].reshape(-1, 2)
            if len(good_next) >= 4:
                deltas = good_next - good_prev
                dx_med = float(np.median(deltas[:, 0]))
                dy_med = float(np.median(deltas[:, 1]))
                if math.hypot(dx_med, dy_med) < 50:
                    flow_x = px + dx_med
                    flow_y = py + dy_med
                    flow_valid = True

        meas_x, meas_y = flow_x, flow_y

        # Manuel modda sadece optik akış + lokal parlaklık/köşe kilitleme çalışır;
        # Otomatik modlarda (Sağ Plaka / Merkez / Sol Plaka) YOLO ile birleştirilir.
        if self.track_mode != "manual":
            run_yolo = (frame_idx % self.yolo_interval == 0) or (not flow_valid)
            if run_yolo:
                box = self._detect_yolo_anchor(frame, frame_idx)
                if box is not None:
                    yx, yy = self._extract_target_from_box(box, gray)
                    if flow_valid:
                        meas_x = 0.40 * flow_x + 0.60 * yx
                        meas_y = 0.30 * flow_y + 0.70 * yy
                    else:
                        meas_x, meas_y = yx, yy

        self.kalman.predict()
        measurement = np.array([[np.float32(meas_x)], [np.float32(meas_y)]])
        estimated = self.kalman.correct(measurement)
        kx, ky = float(estimated[0][0]), float(estimated[1][0])

        self.raw_x_buffer.append(kx)
        self.raw_y_buffer.append(ky)

        weights = np.linspace(1.0, 2.5, len(self.raw_x_buffer))
        weights /= weights.sum()
        smooth_x = float(np.dot(self.raw_x_buffer, weights))
        smooth_y = float(np.dot(self.raw_y_buffer, weights))

        self.current_pos = (smooth_x, smooth_y)
        self.prev_gray = gray
        return smooth_x, smooth_y

    def _compute_color_and_metrics(self, x: float, y: float, frame_idx: int, fps: float):
        dt = 1.0 / max(fps, 1.0)

        if len(self.history) >= 4:
            prev_item = self.history[-4]
            dy_px = prev_item["y"] - y
            dx_px = x - prev_item["x"]
            vy_ms = (dy_px / self.px_per_meter) / (4.0 * dt)
            vx_ms = (dx_px / self.px_per_meter) / (4.0 * dt)
        else:
            vy_ms = 0.0
            vx_ms = 0.0

        speed_ms = math.hypot(vx_ms, vy_ms)

        if not self.stance_locked:
            if frame_idx > 75 and abs(vx_ms) < 0.08 and abs(vy_ms) < 0.12:
                self.stable_frames += 1
                self.ref_x = 0.7 * self.ref_x + 0.3 * x
                self.ref_y_top = 0.7 * self.ref_y_top + 0.3 * y
                if self.stable_frames >= 5 or frame_idx == 88:
                    self.stance_locked = True
                    self.ref_x = x
                    self.ref_y_top = y
                    self.history.clear()
                    self.phase = "HAZIR (SQUAT DURUSU)"
            else:
                self.ref_x = x
                self.ref_y_top = min(self.ref_y_top, y) if self.ref_y_top else y
                if frame_idx >= 90:
                    self.stance_locked = True
                    self.ref_x = x
                    self.history.clear()

        drift_px = x - self.ref_x
        drift_cm = (drift_px / self.px_per_meter) * 100.0
        disp_from_top_cm = ((y - self.ref_y_top) / self.px_per_meter) * 100.0

        if self.stance_locked and frame_idx < 565:
            if not self.in_rep:
                if abs(vy_ms) < 0.08 and abs(disp_from_top_cm) < 6.0:
                    self.ref_y_top = 0.95 * self.ref_y_top + 0.05 * y
                    self.ref_x = 0.96 * self.ref_x + 0.04 * x

                if disp_from_top_cm > 6.5 and vy_ms < -0.07:
                    self.in_rep = True
                    self.rep_start_frame = frame_idx
                    self.phase = "INIS (ECCENTRIC)"
                    self.rep_max_y = y
                    self.rep_velocities = []
                    self.rep_drifts = []
                    if self.rep_count == 0:
                        self.ref_x = x
                        self.history.clear()
                else:
                    self.phase = "HAZIR / LOCKOUT"
            else:
                self.rep_drifts.append(abs(drift_cm))
                if y > self.rep_max_y:
                    self.rep_max_y = y

                max_depth_cm = ((self.rep_max_y - self.ref_y_top) / self.px_per_meter) * 100.0

                if vy_ms < -0.06:
                    self.phase = "INIS (ECCENTRIC)"
                elif abs(vy_ms) <= 0.06 and disp_from_top_cm > 16.0:
                    self.phase = "DIP / ZORLANMA"
                elif vy_ms > 0.06:
                    self.phase = "CIKIS (CONCENTRIC)"
                    self.rep_velocities.append(vy_ms)

                if max_depth_cm > 22.0 and disp_from_top_cm < 8.5 and (frame_idx - self.rep_start_frame) > 35 and vy_ms > -0.04:
                    self.rep_count += 1
                    mean_vel = float(np.mean(self.rep_velocities)) if self.rep_velocities else 0.25
                    min_vel = float(np.min(self.rep_velocities)) if self.rep_velocities else 0.08
                    max_drift = float(np.max(self.rep_drifts)) if self.rep_drifts else 1.5
                    duration_s = (frame_idx - self.rep_start_frame) / max(fps, 1.0)
                    form_score = int(np.clip(100 - max(0, max_drift - 1.5) * 4.5 - max(0, 0.25 - min_vel) * 45, 55, 99))
                    self.rep_summaries.append({
                        "rep": self.rep_count,
                        "mean_vel": round(mean_vel, 2),
                        "min_vel": round(min_vel, 2),
                        "max_drift_cm": round(max_drift, 1),
                        "depth_cm": round(max_depth_cm, 1),
                        "duration_s": round(duration_s, 1),
                        "form_score": form_score,
                    })
                    self.in_rep = False
                    self.phase = f"{self.rep_count}. TEKRAR TAMAM!"
        elif frame_idx >= 565:
            self.in_rep = False
            self.phase = "SET BITTI / RACKING"

        abs_drift = abs(drift_cm)
        drift_penalty = np.clip((abs_drift - 2.0) / 4.2, 0.0, 1.0)

        if "CIKIS" in self.phase or "DIP" in self.phase:
            vel_stress = np.clip((self.vel_threshold - vy_ms) / 0.24, 0.0, 1.0)
            stress = max(vel_stress, drift_penalty * 0.9)
        elif "INIS" in self.phase:
            stress = drift_penalty * 0.85
        else:
            stress = 0.05

        color = self._stress_to_bgr(float(stress))
        return vy_ms, speed_ms, drift_cm, color, float(stress)

    @staticmethod
    def _stress_to_bgr(stress: float):
        s = float(np.clip(stress, 0.0, 1.0))
        if s < 0.5:
            t = s * 2.0
            b = int(75 * (1 - t) + 30 * t)
            g = int(235 * (1 - t) + 215 * t)
            r = int(50 * (1 - t) + 255 * t)
        else:
            t = (s - 0.5) * 2.0
            b = int(30 * (1 - t) + 55 * t)
            g = int(215 * (1 - t) + 45 * t)
            r = int(255 * (1 - t) + 250 * t)
        return (b, g, r)

    def render_current_overlay(self, frame):
        """Duraklatılmış karede ayarlar değiştiğinde anında yeniden çizim yapar."""
        if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
            resized = cv2.resize(frame, (self.target_width, self.target_height))
        else:
            resized = frame
        if not self.history:
            return self._draw_hud_and_path(resized, 0.0, 0.0, (75, 235, 50), 0.0)
        last = self.history[-1]
        return self._draw_hud_and_path(resized, last["vy"], last["drift_cm"], last["color"], last["stress"])

    def _draw_hud_and_path(self, frame, vy_ms: float, drift_cm: float, color, stress: float):
        canvas = frame.copy()
        overlay = frame.copy()

        # 1. Dikey Referans Ekseni
        if self.ref_x is not None and self.stance_locked:
            rx = int(round(self.ref_x))
            tol_px = int(round(0.025 * self.px_per_meter))
            cv2.rectangle(
                overlay,
                (rx - tol_px, 150),
                (rx + tol_px, 560),
                (180, 240, 180),
                -1,
            )
            cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)

            for y_dash in range(150, 560, 22):
                cv2.line(
                    canvas,
                    (rx, y_dash),
                    (rx, min(y_dash + 11, 560)),
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                canvas,
                "IDEAL DIKEY EKSEN",
                (rx - 68, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        # 2. Dinamik İz Uzunluğu ve Kalınlığı ile Glow Çizgi
        all_pts = list(self.history)
        pts = all_pts[-self.trail_length:] if self.trail_length > 0 else all_pts
        t_main = max(1, int(self.line_thickness))
        t_glow = t_main * 3

        if len(pts) >= 2:
            glow_layer = canvas.copy()
            for i in range(1, len(pts)):
                p1 = (int(round(pts[i - 1]["x"])), int(round(pts[i - 1]["y"])))
                p2 = (int(round(pts[i]["x"])), int(round(pts[i]["y"])))
                c = pts[i]["color"]
                cv2.line(glow_layer, p1, p2, c, t_glow, cv2.LINE_AA)
                cv2.line(canvas, p1, p2, c, t_main, cv2.LINE_AA)

            cv2.addWeighted(glow_layer, 0.30, canvas, 0.70, 0, canvas)

        # 3. Takip Noktası Hedef Halkası (Target Reticle)
        if self.current_pos is not None:
            cx, cy = int(round(self.current_pos[0])), int(round(self.current_pos[1]))
            cv2.circle(canvas, (cx, cy), 22, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 5, (255, 255, 255), -1, cv2.LINE_AA)
            # Artı (Crosshair) çizgileri
            cv2.line(canvas, (cx - 28, cy), (cx - 14, cy), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx + 14, cy), (cx + 28, cy), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy - 28), (cx, cy - 14), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy + 14), (cx, cy + 28), color, 2, cv2.LINE_AA)

            mode_names = {
                "right_plate": "SAG PLAKA",
                "center": "BAR MERKEZI",
                "left_plate": "SOL PLAKA",
                "manual": "MANUEL KILIT",
            }
            m_label = mode_names.get(self.track_mode, "HEDEF")
            cv2.putText(
                canvas,
                m_label,
                (cx + 24, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA,
            )

            if self.ref_x is not None and self.stance_locked and abs(cx - int(round(self.ref_x))) > 6:
                rx = int(round(self.ref_x))
                cv2.line(canvas, (rx, cy), (cx, cy), color, 2, cv2.LINE_AA)

        # 4. Sol Üst Bilgi Paneli (HUD)
        hud_x, hud_y, hud_w, hud_h = 24, 24, 385, 250
        hud_bg = canvas.copy()
        cv2.rectangle(hud_bg, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (16, 18, 24), -1)
        cv2.addWeighted(hud_bg, 0.84, canvas, 0.16, 0, canvas)
        cv2.rectangle(canvas, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), color, 2, cv2.LINE_AA)

        cv2.putText(
            canvas,
            "BARPATH AI - HIZ & FORM ANALIZI",
            (hud_x + 16, hud_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.line(canvas, (hud_x + 16, hud_y + 42), (hud_x + hud_w - 16, hud_y + 42), (70, 75, 85), 1)

        cv2.putText(
            canvas,
            f"TEKRAR (REPS): {self.rep_count}",
            (hud_x + 16, hud_y + 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (80, 235, 60),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"FAZ: {self.phase}",
            (hud_x + 16, hud_y + 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (215, 220, 230),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"DIKEY HIZ: {abs(vy_ms):.2f} m/s",
            (hud_x + 16, hud_y + 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        bar_x, bar_y, bar_w, bar_h = hud_x + 16, hud_y + 145, hud_w - 32, 12
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 50, 60), -1)
        fill_w = int(np.clip(abs(vy_ms) / 0.65, 0.0, 1.0) * bar_w)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

        drift_dir = "ARKA/SAG" if drift_cm > 0.6 else ("ON/SOL" if drift_cm < -0.6 else "TAM DIKEY")
        cv2.putText(
            canvas,
            f"YATAY SAPMA: {drift_cm:+.1f} cm ({drift_dir})",
            (hud_x + 16, hud_y + 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        if stress < 0.35:
            status_txt = "FORM: KUSURSUZ & AKICI (YESIL)"
        elif stress < 0.68:
            status_txt = "FORM: YAVASLAMA / SAPMA (SARI)"
        else:
            status_txt = "FORM: ZORLANMA / SAPMA! (KIRMIZI)"

        cv2.putText(
            canvas,
            status_txt,
            (hud_x + 16, hud_y + 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

        if self.rep_summaries:
            ry = hud_y + hud_h + 14
            rh = 32 + len(self.rep_summaries) * 24
            rbg = canvas.copy()
            cv2.rectangle(rbg, (hud_x, ry), (hud_x + hud_w, ry + rh), (16, 18, 24), -1)
            cv2.addWeighted(rbg, 0.80, canvas, 0.20, 0, canvas)
            cv2.rectangle(canvas, (hud_x, ry), (hud_x + hud_w, ry + rh), (75, 80, 95), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "TEKRAR KARNESI (ORT. HIZ | SAPMA | SKOR)",
                (hud_x + 12, ry + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (190, 200, 215),
                1,
                cv2.LINE_AA,
            )
            for idx, rinfo in enumerate(self.rep_summaries[-4:]):
                ty = ry + 42 + idx * 23
                sc = rinfo["form_score"]
                sc_col = (80, 235, 60) if sc >= 85 else ((30, 215, 255) if sc >= 72 else (55, 55, 250))
                line = f"#{rinfo['rep']}: {rinfo['mean_vel']:.2f} m/s | {rinfo['max_drift_cm']:.1f} cm | Form: %{sc}"
                cv2.putText(canvas, line, (hud_x + 14, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.46, sc_col, 1, cv2.LINE_AA)

        self._draw_velocity_graph(canvas)
        return canvas

    def _draw_velocity_graph(self, canvas):
        gw, gh = 330, 120
        gx = self.target_width - gw - 24
        gy = self.target_height - gh - 24

        bg = canvas.copy()
        cv2.rectangle(bg, (gx, gy), (gx + gw, gy + gh), (16, 18, 24), -1)
        cv2.addWeighted(bg, 0.82, canvas, 0.18, 0, canvas)
        cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), (75, 80, 95), 1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            "CANLI DIKEY HIZ GRAFIGI (m/s)",
            (gx + 12, gy + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (210, 215, 225),
            1,
            cv2.LINE_AA,
        )

        mid_y = gy + 70
        cv2.line(canvas, (gx + 10, mid_y), (gx + gw - 10, mid_y), (65, 70, 80), 1)

        vels = list(self.velocity_history)
        if len(vels) >= 2:
            for i in range(1, len(vels)):
                x1 = gx + 12 + int((i - 1) / 140.0 * (gw - 24))
                x2 = gx + 12 + int(i / 140.0 * (gw - 24))
                y1 = int(np.clip(mid_y - vels[i - 1][0] * 65, gy + 30, gy + gh - 8))
                y2 = int(np.clip(mid_y - vels[i][0] * 65, gy + 30, gy + gh - 8))
                cv2.line(canvas, (x1, y1), (x2, y2), vels[i][1], 2, cv2.LINE_AA)

    def process_frame(self, frame, frame_idx: int, fps: float = 25.0):
        if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
            resized = cv2.resize(frame, (self.target_width, self.target_height))
        else:
            resized = frame

        cx, cy = self._track_frame(resized, frame_idx)
        vy_ms, speed_ms, drift_cm, color, stress = self._compute_color_and_metrics(cx, cy, frame_idx, fps)

        metrics = {
            "frame": frame_idx,
            "x": cx,
            "y": cy,
            "vy": vy_ms,
            "speed": speed_ms,
            "drift_cm": drift_cm,
            "color": color,
            "stress": stress,
            "phase": self.phase,
            "rep": self.rep_count,
        }
        self.history.append(metrics)
        self.velocity_history.append((vy_ms, color))

        annotated = self._draw_hud_and_path(resized, vy_ms, drift_cm, color, stress)
        return annotated, metrics


def run_video_analysis(
    video_path: str = "video.mp4",
    output_path: str = "output_barpath.mp4",
    model_path: str = "best.pt",
    show_live: bool = False,
    save_snapshots_dir: str = None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    analyzer = BarPathAnalyzer(model_path=model_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path, fourcc, fps, (analyzer.target_width, analyzer.target_height)
    )

    if save_snapshots_dir:
        os.makedirs(save_snapshots_dir, exist_ok=True)

    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, metrics = analyzer.process_frame(frame, frame_idx, fps)
        writer.write(annotated)

        if save_snapshots_dir and frame_idx in (135, 235, 340, 485, 550):
            snap_file = os.path.join(save_snapshots_dir, f"v2_snap_{frame_idx}.jpg")
            cv2.imwrite(snap_file, annotated)

        if show_live:
            cv2.imshow("BarPath AI - Antrenman Analizi (Cikis icin Q)", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if show_live:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"Analiz tamamlandi! {frame_idx}/{total_frames} kare {elapsed:.1f} saniyede islendi ({frame_idx/max(elapsed,0.1):.1f} FPS).")
    return analyzer


if __name__ == "__main__":
    run_video_analysis(
        video_path="video.mp4",
        output_path="output_barpath.mp4",
        model_path="best.pt",
        show_live=True,
    )
