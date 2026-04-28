import cv2
import numpy as np
import json
import threading
import queue
import easyocr
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from scipy import stats as scipy_stats
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
from tqdm import tqdm
import argparse
import sys
import signal
import atexit
import time
import csv
import re
import torch

from config import ATESConfig


# ════════════════════════════════════════════════════════════════════════════
# DEVICE DETECTION
# ════════════════════════════════════════════════════════════════════════════

def get_device():
    """Auto detect best available device"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"[Device] ✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print(f"[Device] Using CPU (no GPU detected)")
    return device

DEVICE = get_device()


# ════════════════════════════════════════════════════════════════════════════
# 1. SPEED ESTIMATOR
# ════════════════════════════════════════════════════════════════════════════

class SpeedEstimator:
    """Multi-method speed estimator with ensemble approach"""

    def __init__(self, fps, config=None, pixel_to_meter=None):
        self.config = config or ATESConfig()
        self.fps = fps
        self.pixel_to_meter = pixel_to_meter or self.config.DEFAULT_PIXEL_TO_METER
        self.frame_height = None
        self.frame_width = None

        self.speed_history = defaultdict(
            lambda: deque(maxlen=self.config.SPEED_BUFFER_SIZE)
        )
        self.last_speeds = {}
        self.position_history = defaultdict(lambda: deque(maxlen=50))
        self.kalman_filters = {}
        self.raw_speed_history = defaultdict(
            lambda: deque(maxlen=self.config.SPEED_BUFFER_SIZE)
        )

        print(f"[SpeedEstimator] FPS: {fps}")
        print(f"[SpeedEstimator] Pixel-to-Meter: {self.pixel_to_meter:.6f}")
        print(f"[SpeedEstimator] Method: {self.config.SPEED_CALCULATION_METHOD}")

    def set_frame_dimensions(self, height, width):
        self.frame_height = height
        self.frame_width = width

    def _init_kalman_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0 / self.fps
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        kf.P *= 100
        kf.R = np.eye(2) * 1.0
        kf.Q = np.eye(4) * 0.01
        return kf

    def _apply_perspective_correction(self, position):
        if not self.config.USE_PERSPECTIVE_CORRECTION or self.frame_height is None:
            return 1.0
        y = position[1]
        normalized_y = y / self.frame_height
        strength = self.config.PERSPECTIVE_CORRECTION_STRENGTH
        correction = 1.0 + (1.0 - normalized_y) * strength
        correction = np.clip(correction, 0.8, 2.5)
        return correction

    def _calculate_speed_kalman(self, track_id, trajectory):
        if len(trajectory) < 5:
            return None
        if track_id not in self.kalman_filters:
            self.kalman_filters[track_id] = self._init_kalman_filter()
            pos = trajectory[0]
            self.kalman_filters[track_id].x = np.array(
                [pos[0], pos[1], 0, 0]
            )
        kf = self.kalman_filters[track_id]
        current_pos = np.array(trajectory[-1])
        kf.predict()
        kf.update(current_pos)
        vx, vy = kf.x[2], kf.x[3]
        return np.sqrt(vx**2 + vy**2)

    def _calculate_speed_linear(self, trajectory):
        if len(trajectory) < 5:
            return None
        points = trajectory[-self.config.TRAJECTORY_ANALYSIS_POINTS:]
        if len(points) < 5:
            return None
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        frames = np.arange(len(points))
        try:
            slope_x, *_ = scipy_stats.linregress(frames, xs)
            slope_y, *_ = scipy_stats.linregress(frames, ys)
            return np.sqrt(slope_x**2 + slope_y**2)
        except:
            return None

    def _calculate_speed_polynomial(self, trajectory):
        if len(trajectory) < 8:
            return None
        points = trajectory[-self.config.TRAJECTORY_ANALYSIS_POINTS:]
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        frames = np.arange(len(points))
        try:
            poly_x = np.polyfit(frames, xs, 2)
            poly_y = np.polyfit(frames, ys, 2)
            vel_x = 2 * poly_x[0] * len(points) + poly_x[1]
            vel_y = 2 * poly_y[0] * len(points) + poly_y[1]
            return np.sqrt(vel_x**2 + vel_y**2)
        except:
            return None

    def _calculate_speed_ensemble(self, track_id, trajectory):
        speeds = []
        k = self._calculate_speed_kalman(track_id, trajectory)
        if k and k > 0.01:
            speeds.append(k)
        l = self._calculate_speed_linear(trajectory)
        if l and l > 0.01:
            speeds.append(l)
        p = self._calculate_speed_polynomial(trajectory)
        if p and p > 0.01:
            speeds.append(p)
        if not speeds:
            return None
        return np.median(speeds)

    def _reject_outliers(self, track_id, new_speed):
        if not self.config.USE_SPEED_OUTLIER_REJECTION:
            return new_speed
        history = list(self.raw_speed_history[track_id])
        if len(history) < 5:
            return new_speed
        mean_speed = np.mean(history)
        std_speed = np.std(history)
        if std_speed == 0:
            return new_speed
        z_score = abs((new_speed - mean_speed) / std_speed)
        if z_score > self.config.SPEED_OUTLIER_THRESHOLD:
            return self.last_speeds.get(track_id, mean_speed)
        return new_speed

    def _smooth_speed(self, track_id, raw_speed):
        alpha = self.config.SPEED_SMOOTHING_ALPHA
        if track_id in self.last_speeds:
            smoothed = alpha * raw_speed + (1 - alpha) * self.last_speeds[track_id]
        else:
            smoothed = raw_speed
        self.last_speeds[track_id] = smoothed
        return smoothed

    def estimate_speed(self, track_id, trajectory):
        if len(trajectory) < self.config.MIN_TRACK_LENGTH:
            return None

        self.position_history[track_id].extend(trajectory[-5:])
        method = self.config.SPEED_CALCULATION_METHOD.lower()

        if "ensemble" in method:
            pixel_velocity = self._calculate_speed_ensemble(track_id, trajectory)
        elif "kalman" in method:
            pixel_velocity = self._calculate_speed_kalman(track_id, trajectory)
        elif "polynomial" in method:
            pixel_velocity = self._calculate_speed_polynomial(trajectory)
        elif "linear" in method:
            pixel_velocity = self._calculate_speed_linear(trajectory)
        else:
            pixel_velocity = self._calculate_speed_kalman(track_id, trajectory)

        if pixel_velocity is None or pixel_velocity < 0.05:
            return None

        perspective_factor = self._apply_perspective_correction(trajectory[-1])
        corrected_pv = pixel_velocity * perspective_factor

        meters_per_frame = corrected_pv * self.pixel_to_meter
        meters_per_second = meters_per_frame * self.fps
        kmh = meters_per_second * 3.6

        if (kmh < self.config.MIN_REALISTIC_SPEED or
                kmh > self.config.MAX_REALISTIC_SPEED):
            if track_id in self.last_speeds:
                return self.last_speeds[track_id]
            return None

        self.raw_speed_history[track_id].append(kmh)
        kmh = self._reject_outliers(track_id, kmh)
        smoothed_kmh = self._smooth_speed(track_id, kmh)
        self.speed_history[track_id].append(smoothed_kmh)

        return smoothed_kmh

    def get_average_speed(self, track_id):
        if (track_id not in self.speed_history or
                len(self.speed_history[track_id]) == 0):
            return None
        return np.median(list(self.speed_history[track_id]))

    def cleanup_track(self, track_id):
        for attr in ['speed_history', 'last_speeds', 'kalman_filters',
                     'position_history', 'raw_speed_history']:
            container = getattr(self, attr)
            if track_id in container:
                del container[track_id]


# ════════════════════════════════════════════════════════════════════════════
# 2. VEHICLE DETECTOR (CPU compatible, no gmc_method)
# ════════════════════════════════════════════════════════════════════════════

class VehicleDetector:
    def __init__(self, model_path=None, config=None):
        self.config = config or ATESConfig()
        model_path = model_path or self.config.MODEL_PATH

        print(f"[Detector] Loading model: {model_path}")
        self.model = YOLO(model_path)

        # ── Core tracking state ──────────────────────────────────────────
        self.active_tracks   = {}   # track_id -> detection dict
        self.track_history   = defaultdict(list)  # track_id -> [(cx,cy)]
        self.track_bboxes    = defaultdict(list)  # track_id -> [bbox]
        self.counted_ids     = set()

        # ── Track lifecycle ───────────────────────────────────────────────
        self.track_first_seen  = {}   # track_id -> frame_num
        self.track_last_seen   = {}   # track_id -> frame_num
        self.track_class       = {}   # track_id -> class_name
        self.track_miss_count  = defaultdict(int)  # frames missed

        self.current_frame = 0
        self.next_id       = 1

        # ── Tuning parameters ─────────────────────────────────────────────
        self.MAX_MATCH_DIST    = 300   # pixels - centroid match distance
        self.IOU_MATCH_THRESH  = 0.30  # IoU threshold for box matching
        self.MAX_MISS_FRAMES   = 25    # frames before track is dropped
        self.MIN_FRAMES_COUNT  = 30    # min frames before counting
        self.MIN_DIST_COUNT    = 80    # min pixels moved before counting

        # ── Statistics ────────────────────────────────────────────────────
        self.vehicle_counts = defaultdict(int)
        self.total_vehicles = 0

        # ── Marker persistence ────────────────────────────────────────────
        self.track_markers       = {}
        self.marker_display_time = 60

        print(f"[Detector] ✓ Model loaded")
        print(f"[Detector] Conf={self.config.CONFIDENCE_THRESHOLD} "
              f"IOU={self.config.IOU_THRESHOLD}")
        print(f"[Detector] Device={DEVICE}")
        print(f"[Detector] MAX_MATCH_DIST={self.MAX_MATCH_DIST}px")
        print(f"[Detector] MAX_MISS_FRAMES={self.MAX_MISS_FRAMES}")
        print(f"[Detector] MIN_FRAMES_COUNT={self.MIN_FRAMES_COUNT}")

    # ─────────────────────────────────────────────────────────────────────
    # IoU CALCULATION
    # ─────────────────────────────────────────────────────────────────────

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter   = inter_w * inter_h

        if inter == 0:
            return 0.0

        areaA = ((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = ((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        union = areaA + areaB - inter

        return inter / union if union > 0 else 0.0

    # ─────────────────────────────────────────────────────────────────────
    # MATCHING LOGIC
    # ─────────────────────────────────────────────────────────────────────

    def _match_detections(self, raw_detections):
        if not raw_detections:
            return []

        # Only match against RECENTLY seen tracks
        active_track_ids = [
            tid for tid in self.track_last_seen
            if (self.current_frame - self.track_last_seen[tid])
               <= self.MAX_MISS_FRAMES
        ]

        # Build cost matrix (detections x active_tracks)
        n_det   = len(raw_detections)
        n_tracks = len(active_track_ids)

        matched_detections = []
        used_track_ids     = set()

        if n_tracks == 0:
            # No existing tracks - assign new IDs
            for det in raw_detections:
                det['track_id'] = self.next_id
                self.next_id += 1
                matched_detections.append(det)
            return matched_detections

        # ── Build score matrix ────────────────────────────────────────────
        # Score = combined IoU + inverse centroid distance
        scores = np.zeros((n_det, n_tracks))

        for i, det in enumerate(raw_detections):
            det_center = np.array(det['center'], dtype=float)
            det_bbox   = det['bbox']

            for j, tid in enumerate(active_track_ids):
                # Centroid distance score
                if self.track_history[tid]:
                    prev_center = np.array(
                        self.track_history[tid][-1], dtype=float
                    )
                    dist = np.linalg.norm(det_center - prev_center)
                    dist_score = max(0, 1.0 - dist / self.MAX_MATCH_DIST)
                else:
                    dist_score = 0.0

                # IoU score
                if self.track_bboxes[tid]:
                    prev_bbox = self.track_bboxes[tid][-1]
                    iou_score = self._iou(det_bbox, prev_bbox)
                else:
                    iou_score = 0.0

                # Combined score (IoU weighted higher)
                scores[i, j] = 0.4 * dist_score + 0.6 * iou_score

        # ── Greedy matching (best score first) ────────────────────────────
        # Sort all (det, track) pairs by score descending
        pairs = []
        for i in range(n_det):
            for j in range(n_tracks):
                if scores[i, j] > 0.1:  # Minimum threshold
                    pairs.append((scores[i, j], i, j))

        pairs.sort(reverse=True)

        matched_det_ids = set()

        for score, i, j in pairs:
            det = raw_detections[i]
            tid = active_track_ids[j]

            # Skip if already matched
            if i in matched_det_ids or tid in used_track_ids:
                continue

            # Accept match
            det['track_id'] = tid
            matched_det_ids.add(i)
            used_track_ids.add(tid)

        # ── Assign new IDs to unmatched detections ────────────────────────
        for i, det in enumerate(raw_detections):
            if i not in matched_det_ids:
                det['track_id'] = self.next_id
                self.next_id += 1

        matched_detections = raw_detections
        return matched_detections

    # ─────────────────────────────────────────────────────────────────────
    # UPDATE MARKERS
    # ─────────────────────────────────────────────────────────────────────

    def _update_markers(self):
        expired = [
            tid for tid, m in self.track_markers.items()
            if m['frames_left'] <= 0
        ]
        for tid in expired:
            del self.track_markers[tid]
        for m in self.track_markers.values():
            m['frames_left'] -= 1

    # ─────────────────────────────────────────────────────────────────────
    # COUNTING CRITERIA
    # ─────────────────────────────────────────────────────────────────────

    def _calculate_distance_moved(self, track_id):
        """Total distance moved along trajectory"""
        traj = self.track_history.get(track_id, [])
        if len(traj) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(traj)):
            p1 = np.array(traj[i - 1], dtype=float)
            p2 = np.array(traj[i],     dtype=float)
            total += np.linalg.norm(p2 - p1)
        return total

    def _should_count_vehicle(self, track_id):
        if track_id in self.counted_ids:
            return False

        if track_id not in self.track_first_seen:
            return False

        frames_tracked = (self.current_frame
                          - self.track_first_seen[track_id])
        if frames_tracked < self.MIN_FRAMES_COUNT:
            return False

        distance = self._calculate_distance_moved(track_id)
        if distance < self.MIN_DIST_COUNT:
            return False

        return True

    # ─────────────────────────────────────────────────────────────────────
    # MAIN DETECTION + TRACKING
    # ─────────────────────────────────────────────────────────────────────

    def detect_and_track(self, frame):
        self.current_frame += 1
        self._update_markers()

        # ── YOLO predict (no tracker - avoids gmc_method error) ───────────
        try:
            results = self.model.predict(
                frame,
                conf=self.config.CONFIDENCE_THRESHOLD,
                iou=self.config.IOU_THRESHOLD,
                verbose=False
            )
        except Exception as e:
            print(f"[Detector] Predict error: {e}")
            return [], []

        detections = []

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes       = result.boxes.xyxy.cpu().numpy()
                classes     = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()

                # ── Build raw detections ──────────────────────────────────
                raw_detections = []
                for box, cls, conf in zip(boxes, classes, confidences):
                    x1, y1, x2, y2 = box

                    # Box size validation
                    w = x2 - x1
                    h = y2 - y1
                    if (w < self.config.MIN_BOX_WIDTH  or
                        h < self.config.MIN_BOX_HEIGHT or
                        w > self.config.MAX_BOX_WIDTH  or
                        h > self.config.MAX_BOX_HEIGHT):
                        continue

                    vehicle_class = self.config.VEHICLE_CLASSES.get(
                        cls, "unknown"
                    )
                    if vehicle_class == "unknown":
                        continue

                    center_x     = int((x1 + x2) / 2)
                    center_y     = int((y1 + y2) / 2)
                    bottom_center = (center_x, int(y2))

                    raw_detections.append({
                        'track_id':     None,
                        'bbox':         (int(x1), int(y1), int(x2), int(y2)),
                        'center':       (center_x, center_y),
                        'bottom_center': bottom_center,
                        'class':        vehicle_class,
                        'class_id':     int(cls),
                        'confidence':   float(conf)
                    })

                # ── Match to existing tracks ──────────────────────────────
                matched = self._match_detections(raw_detections)

                # ── Update track state ────────────────────────────────────
                current_frame_ids = set()

                for det in matched:
                    tid = det['track_id']
                    current_frame_ids.add(tid)

                    # First time seeing this track
                    if tid not in self.track_first_seen:
                        self.track_first_seen[tid] = self.current_frame
                        self.track_class[tid]      = det['class']

                    # Update last seen + miss count
                    self.track_last_seen[tid] = self.current_frame
                    self.track_miss_count[tid] = 0

                    # Update trajectory (bottom center for speed calc)
                    self.track_history[tid].append(det['bottom_center'])
                    if len(self.track_history[tid]) > 200:
                        self.track_history[tid] = \
                            self.track_history[tid][-200:]

                    # Update bbox history for IoU matching
                    self.track_bboxes[tid].append(det['bbox'])
                    if len(self.track_bboxes[tid]) > 10:
                        self.track_bboxes[tid] = \
                            self.track_bboxes[tid][-10:]

                    self.active_tracks[tid] = det

                    self.track_markers[tid] = {
                        'position':   det['bottom_center'],
                        'frames_left': self.marker_display_time
                    }

                    # ── Strict counting ───────────────────────────────────
                    if self._should_count_vehicle(tid):
                        self.counted_ids.add(tid)
                        self.vehicle_counts[det['class']] += 1
                        self.total_vehicles += 1
                        print(
                            f"[COUNT] Track {tid:4d} | "
                            f"{det['class']:12s} | "
                            f"frames={self.current_frame - self.track_first_seen[tid]} | "
                            f"Total={self.total_vehicles}"
                        )

                # ── Increment miss count for lost tracks ──────────────────
                for tid in list(self.active_tracks.keys()):
                    if tid not in current_frame_ids:
                        self.track_miss_count[tid] += 1

                detections = matched

        return results, detections

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def get_trajectory(self, track_id, n_points=None):
        traj = self.track_history.get(track_id, [])
        if n_points and len(traj) > n_points:
            return traj[-n_points:]
        return traj

    def cleanup_lost(self, current_frame_ids):
        """
        Remove tracks that have been missing too long
        """
        to_remove = []
        for tid in list(self.active_tracks.keys()):
            if self.track_miss_count[tid] > self.MAX_MISS_FRAMES:
                to_remove.append(tid)

        for tid in to_remove:
            self.active_tracks.pop(tid, None)

    def get_statistics(self):
        return {
            'total_vehicles': self.total_vehicles,
            'vehicle_counts': dict(self.vehicle_counts),
            'active_tracks':  len(self.active_tracks),
            'counted_unique': len(self.counted_ids)
        }


# ════════════════════════════════════════════════════════════════════════════
# 3. SMART CALIBRATOR
# ════════════════════════════════════════════════════════════════════════════

class SmartCalibrator:
    """Smart calibration with AI and manual modes"""

    def __init__(self, video_path, model_path=None, config=None):
        self.video_path = video_path
        self.config = config or ATESConfig()
        self.model_path = model_path or self.config.MODEL_PATH

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.original_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps             = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames    = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.display_width, self.display_height = \
            self._calculate_display_size()
        self.scale_x = self.original_width  / self.display_width
        self.scale_y = self.original_height / self.display_height

        print(f"\n[SmartCalibrator] Initializing...")
        print(f"  Video      : {self.original_width}x"
              f"{self.original_height} @ {self.fps}fps")
        print(f"  Display    : {self.display_width}x{self.display_height}")

        print(f"  ✓ Model loaded")
        self.model = YOLO(self.model_path)

    def _calculate_display_size(self):
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            root.destroy()
            max_w = int(screen_w * 0.95)
            max_h = int(screen_h * 0.90)
        except:
            max_w = 1280
            max_h = 720

        scale = min(max_w / self.original_width,
                    max_h / self.original_height, 1.0)
        return (int(self.original_width  * scale),
                int(self.original_height * scale))

    def _display_to_original(self, dx, dy):
        return int(dx * self.scale_x), int(dy * self.scale_y)

    def _original_to_display(self, ox, oy):
        return int(ox / self.scale_x), int(oy / self.scale_y)

    def auto_calibrate(self):
        """AI Auto-Calibration"""
        print("\n[AUTO-CALIBRATION] Scanning video...")

        calibrations = []
        frame_count  = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 5 != 0:
                continue

            results = self.model(frame, conf=0.4, verbose=False)

            if results[0].boxes is not None:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, cls in zip(boxes, classes):
                    x1, y1, x2, y2 = box
                    bbox_height = y2 - y1
                    vehicle_class = self.config.VEHICLE_CLASSES.get(
                        cls, None
                    )
                    if vehicle_class and vehicle_class in \
                            self.config.VEHICLE_DIMENSIONS:
                        real_h = \
                            self.config.VEHICLE_DIMENSIONS[vehicle_class] * 0.7
                        if bbox_height > 0:
                            calibrations.append(real_h / bbox_height)

            if frame_count % 30 == 0:
                print(f"  Frame {frame_count}/{self.total_frames} - "
                      f"Found {len(calibrations)} candidates")
            if frame_count > self.config.CALIBRATION_AUTO_FRAMES:
                break

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if not calibrations:
            print("[AUTO-CALIBRATION] No vehicles found!")
            return None

        median_val = float(np.median(calibrations))
        print(f"[AUTO-CALIBRATION] ✓ Result: {median_val:.6f} m/pixel")

        return {
            'mode':          'auto',
            'pixel_to_meter': median_val,
            'confidence':    0.85,
            'method':        'auto_detection',
            'is_valid':      True,
            'timestamp':     datetime.now().isoformat(),
        }

    def manual_calibrate(self):
        """Manual Calibration"""
        print("\n[MANUAL-CALIBRATION] Starting...")

        ret, frame = self.cap.read()
        if not ret:
            print("[ERROR] Cannot read video frame")
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        print(f"  Frame loaded  : {self.original_width}x{self.original_height}")
        print(f"  Display size  : {self.display_width}x{self.display_height}")

        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                orig_x, orig_y = self._display_to_original(x, y)
                points.append((orig_x, orig_y))
                print(f"\n  ✓ Point {len(points)}: "
                      f"disp=({x},{y})  orig=({orig_x},{orig_y})")

        print("\n" + "=" * 60)
        print("  MANUAL CALIBRATION")
        print("=" * 60)
        print("  Click TWO points with KNOWN real distance:")
        refs = {
            'lane_width':       3.65,
            'double_lane':      7.3,
            'car_length':       4.5,
            'truck_length':     12.0,
            'road_marking_gap': 6.0,
            'road_marking_len': 3.0,
            'parking_space':    5.5,
        }
        for k, v in refs.items():
            print(f"    {k:<20s}: {v} m")
        print("\n  SPACE/ENTER = confirm  |  R = reset  |  ESC = cancel")
        print("=" * 60)
        print("\n  Waiting for clicks...")

        win = "Manual Calibration"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.display_width, self.display_height)
        cv2.setMouseCallback(win, mouse_callback)

        while True:
            disp = cv2.resize(
                frame.copy(), (self.display_width, self.display_height)
            )

            # Instructions overlay
            cv2.rectangle(disp, (10, 10),
                          (self.display_width - 10, 75),
                          (20, 20, 20), -1)
            cv2.rectangle(disp, (10, 10),
                          (self.display_width - 10, 75),
                          (0, 255, 255), 2)
            cv2.putText(
                disp,
                "Click 2 points | ENTER=confirm | R=reset | ESC=cancel",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2
            )
            cv2.putText(
                disp, f"Points: {len(points)}/2",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2
            )

            # Draw points
            for i, (ox, oy) in enumerate(points):
                dx, dy = self._original_to_display(ox, oy)
                cv2.circle(disp, (dx, dy), 8, (0, 0, 255), -1)
                cv2.circle(disp, (dx, dy), 10, (0, 255, 0), 2)
                cv2.putText(
                    disp, f"P{i+1}", (dx + 15, dy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            # Draw line between points
            if len(points) == 2:
                dp1 = self._original_to_display(*points[0])
                dp2 = self._original_to_display(*points[1])
                cv2.line(disp, dp1, dp2, (0, 255, 0), 3)

                px_dist = np.linalg.norm(
                    np.array(points[0]) - np.array(points[1])
                )
                mid_x = (dp1[0] + dp2[0]) // 2
                mid_y = (dp1[1] + dp2[1]) // 2

                cv2.rectangle(disp,
                              (mid_x - 110, mid_y - 25),
                              (mid_x + 110, mid_y + 15),
                              (20, 20, 20), -1)
                cv2.putText(
                    disp, f"{px_dist:.1f} px",
                    (mid_x - 100, mid_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2
                )
                cv2.putText(
                    disp, "Ready! Press ENTER",
                    (self.display_width - 320,
                     self.display_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2
                )

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                cv2.destroyAllWindows()
                print("\n  Cancelled.")
                return None
            elif key == ord('r'):
                points = []
                print("\n  Reset.")
            elif key in (13, 32) and len(points) == 2:  # ENTER or SPACE
                cv2.destroyAllWindows()

                px_dist = np.linalg.norm(
                    np.array(points[0]) - np.array(points[1])
                )
                print(f"\n  P1: {points[0]}")
                print(f"  P2: {points[1]}")
                print(f"  Distance: {px_dist:.2f} px")

                print("\n  Enter real distance:")
                for k, v in refs.items():
                    print(f"    {k:<20s}: {v} m")

                while True:
                    try:
                        real_dist = float(
                            input("\n  Distance (meters): ")
                        )
                        if real_dist <= 0:
                            print("  Must be positive!")
                            continue
                        break
                    except ValueError:
                        print("  Enter a number.")

                pixel_to_meter = real_dist / px_dist
                speed_ref = 100 * pixel_to_meter * self.fps * 3.6

                print(f"\n[MANUAL-CALIBRATION] Result:")
                print(f"  Distance    : {real_dist:.3f} m / {px_dist:.2f} px")
                print(f"  pixel_to_m  : {pixel_to_meter:.10f}")
                print(f"  Speed ref   : {speed_ref:.1f} km/h @ 100px/frame")
                print(f"  Validation  : Valid ({speed_ref:.1f} km/h @ 100px/frame)")

                return {
                    'mode':          'manual',
                    'pixel_to_meter': pixel_to_meter,
                    'confidence':    0.97,
                    'method':        'manual_2point',
                    'is_valid':      True,
                    'timestamp':     datetime.now().isoformat(),
                }

    def run(self):
        """Main calibration flow"""
        print("\n" + "=" * 80)
        print("CALIBRATION METHOD SELECTION")
        print("=" * 80)
        print("\n[1] AUTO-CALIBRATION (Recommended)")
        print("    Scans video, detects vehicles, calculates ratio")
        print("\n[2] MANUAL-CALIBRATION")
        print("    Click 2 points, enter real distance")
        print("\n[3] SKIP (Use Default 0.025 m/pixel)")
        print("    Fast but likely inaccurate")

        while True:
            choice = input("\nYour choice (1/2/3): ").strip()

            if choice == '1':
                result = self.auto_calibrate()
                if result:
                    return result
            elif choice == '2':
                result = self.manual_calibrate()
                if result:
                    return result
            elif choice == '3':
                return {
                    'mode':          'default',
                    'pixel_to_meter': 0.025,
                    'confidence':    0.20,
                    'method':        'default',
                    'is_valid':      True,
                    'timestamp':     datetime.now().isoformat(),
                }
            else:
                print("Enter 1, 2, or 3")

    def close(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


# ════════════════════════════════════════════════════════════════════════════
# 4. UNIVERSAL PLATE READER
# ════════════════════════════════════════════════════════════════════════════

class UniversalPlateReader:
    """Universal plate reader with EasyOCR"""

    def __init__(self, ocr_languages=['en'], upscale_factor=2, config=None):
        self.upscale_factor = upscale_factor
        self.ocr_languages  = ocr_languages
        self.config         = config or ATESConfig()

        print(f"\n[PlateReader] Initializing (upscale: {upscale_factor}x)...")

        try:
            self.reader = easyocr.Reader(
                ocr_languages,
                gpu=(DEVICE == 'cuda'),
                verbose=False
            )
            print(f"[PlateReader] ✓ EasyOCR loaded")
        except Exception as e:
            print(f"[PlateReader] ✗ EasyOCR failed: {e}")
            self.reader = None

        self.plate_readings = defaultdict(list)
        self._lock = threading.Lock()

        self.char_confusion = {
            '0': 'O', 'O': '0',
            '1': 'I', 'I': '1', 'L': '1',
            '5': 'S', 'S': '5',
            '8': 'B', 'B': '8',
            '2': 'Z', 'Z': '2',
            '6': 'G', 'G': '6',
        }

    def _upscale_image(self, img, factor=2):
        h, w = img.shape[:2]
        return cv2.resize(
            img, (w * factor, h * factor),
            interpolation=cv2.INTER_LANCZOS4
        )

    def _preprocess_adaptive(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
                   if len(img.shape) == 3 else img
            denoised = cv2.fastNlMeansDenoising(gray, h=8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            binary = cv2.adaptiveThreshold(
                enhanced, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 13, 3
            )
            return binary, enhanced
        except:
            return None, None

    def _preprocess_gamma(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
                   if len(img.shape) == 3 else img
            gamma = 1.3
            lut = np.array(
                [((i / 255) ** (1 / gamma) * 255) for i in range(256)],
                dtype=np.uint8
            )
            corrected = cv2.LUT(gray, lut)
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5, 5))
            enhanced = clahe.apply(corrected)
            _, binary = cv2.threshold(
                enhanced, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary, enhanced
        except:
            return None, None

    def _run_ocr(self, img):
        if self.reader is None:
            return []
        try:
            return self.reader.readtext(
                img,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=1,
            )
        except:
            return []

    def _validate_plate(self, text):
        patterns = [
            r'^[A-Z]{2,3}[0-9]{4,5}[A-Z]?$',
            r'^[A-Z]{2}[0-9]{2}\s?[A-Z]{3}$',
            r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$',
            r'^[A-Z0-9]{4,12}$',
        ]
        text = text.upper().strip()
        text = ''.join(c for c in text if c.isalnum())
        if not text or len(text) < 4 or len(text) > 12:
            return False
        if not (any(c.isdigit() for c in text)
                and any(c.isalpha() for c in text)):
            return False
        for p in patterns:
            if re.match(p, text):
                return True
        return False

    def _correct_ocr_errors(self, text):
        corrected = list(text)
        for i, char in enumerate(corrected):
            if char in self.char_confusion:
                alt = self.char_confusion[char]
                if i < len(text) // 2 and alt.isalpha():
                    corrected[i] = alt
                elif i >= len(text) // 2 and alt.isdigit():
                    corrected[i] = alt
        return ''.join(corrected)

    def read_plate(self, frame, vehicle_bbox):
        x1, y1, x2, y2 = vehicle_bbox
        crop_top = y1 + int((y2 - y1) * 0.60)
        crop = frame[crop_top:y2, x1:x2].copy()

        if crop.size == 0:
            return None, 0.0

        h, w = crop.shape[:2]
        upscaled = self._upscale_image(crop, self.upscale_factor) \
                   if w < 80 else crop

        all_readings = []

        for method in [self._preprocess_adaptive, self._preprocess_gamma]:
            binary, gray = method(upscaled)
            if binary is not None:
                for img in [binary, gray]:
                    if img is not None:
                        for (_, text, conf) in self._run_ocr(img):
                            if conf >= 0.45:
                                corrected = self._correct_ocr_errors(text)
                                if self._validate_plate(corrected):
                                    all_readings.append((corrected, conf))

        if all_readings:
            best_reading, best_conf = max(all_readings, key=lambda x: x[1])
            return best_reading, best_conf

        return None, 0.0

    def record_plate_for_id(self, track_id, plate, confidence):
        with self._lock:
            if plate and plate not in ("SCANNING", "N/A"):
                self.plate_readings[track_id].append((plate, confidence))

    def get_best_plate_for_id(self, track_id):
        with self._lock:
            readings = self.plate_readings.get(track_id, [])
        if not readings:
            return None, 0.0
        plates_only = [p for p, _ in readings if p]
        if not plates_only:
            return None, 0.0
        return self._vote_best_plate(plates_only)

    def _vote_best_plate(self, readings):
        if not readings:
            return None, 0.0
        from collections import Counter
        length_counter = Counter(len(r) for r in readings)
        most_common_len = length_counter.most_common(1)[0][0]
        filtered = [r for r in readings if len(r) == most_common_len] \
                   or readings
        result = []
        conf_sum = 0.0
        for pos in range(most_common_len):
            chars = [r[pos] for r in filtered if pos < len(r)]
            if chars:
                char_votes = Counter(chars)
                best_char, vote_count = char_votes.most_common(1)[0]
                result.append(best_char)
                conf_sum += vote_count / len(filtered)
        best = ''.join(result)
        final_conf = conf_sum / len(result) if result else 0.0
        return best, float(final_conf)


# ════════════════════════════════════════════════════════════════════════════
# 5. TRAFFIC ANALYZER
# ════════════════════════════════════════════════════════════════════════════

class TrafficAnalyzer:
    """Analyze traffic flow and density"""

    def __init__(self):
        self.track_history   = defaultdict(list)
        self.flow_directions = deque(maxlen=100)
        self.density_history = deque(maxlen=30)
        self.last_flow_text  = "Unknown"

    def update_flow(self, track_history):
        self.track_history = track_history
        if track_history:
            for track_id, positions in list(track_history.items())[-10:]:
                if len(positions) > 5:
                    start = np.array(positions[0])
                    end   = np.array(positions[-1])
                    delta = end - start
                    if np.linalg.norm(delta) > 20:
                        direction = delta / np.linalg.norm(delta)
                        angle = np.degrees(
                            np.arctan2(direction[1], direction[0])
                        )
                        self.flow_directions.append(angle)

    def calculate_density(self, active_count):
        self.density_history.append(active_count)
        if active_count <= 3:
            level = "Low"
        elif active_count <= 7:
            level = "Medium"
        elif active_count <= 12:
            level = "High"
        else:
            level = "Very High"
        return level, active_count

    def get_flow_text(self):
        if not self.flow_directions:
            return "No flow"
        avg_angle = np.mean(list(self.flow_directions)) % 360
        if avg_angle < 45 or avg_angle >= 315:
            self.last_flow_text = "East"
        elif 45 <= avg_angle < 135:
            self.last_flow_text = "South"
        elif 135 <= avg_angle < 225:
            self.last_flow_text = "West"
        else:
            self.last_flow_text = "North"
        return self.last_flow_text


# ════════════════════════════════════════════════════════════════════════════
# 6. VIOLATION CHECKER
# ════════════════════════════════════════════════════════════════════════════

class ViolationChecker:
    """Check traffic violations"""

    def __init__(self, config=None):
        self.config      = config or ATESConfig()
        self.SPEED_LIMITS = self.config.SPEED_LIMITS

    def check_speed(self, vehicle_type, speed):
        if speed is None or speed < self.config.OVERSPEED_MIN_CHECK:
            return None, 0.0
        vtype   = str(vehicle_type).lower()
        matched = next(
            (k for k in self.SPEED_LIMITS if k in vtype), 'car'
        )
        limit = self.SPEED_LIMITS[matched]
        if speed > limit:
            excess = speed - limit
            confidence = min(excess / 20.0, 1.0)
            message = f"OVERSPEED {int(speed)}kmh (limit:{limit})"
            return message, float(confidence)
        return None, 0.0


# ════════════════════════════════════════════════════════════════════════════
# 7. DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

class Dashboard:
    """4K-aware dashboard"""

    COLOR_WHITE      = (255, 255, 255)
    COLOR_GREEN      = (0,   255,   0)
    COLOR_RED        = (0,     0, 255)
    COLOR_YELLOW     = (0,   255, 255)
    COLOR_BLUE       = (255, 100,   0)
    COLOR_BLACK      = (0,     0,   0)
    COLOR_ORANGE     = (0,   165, 255)
    COLOR_CYAN       = (255, 255,   0)
    COLOR_GRAY       = (128, 128, 128)
    COLOR_ACCENT     = (255, 140,   0)
    COLOR_LIGHT_GRAY = (200, 200, 200)
    COLOR_DARK_RED   = (0,     0, 180)
    COLOR_DARK_GREEN = (0,   180,   0)

    def __init__(self, width, height):
        self.width  = width
        self.height = height

        self.scale = max(min(width / 1920.0, height / 1080.0), 1.0)

        self.left_panel_w  = int(420 * self.scale)
        self.right_panel_w = int(550 * self.scale)
        self.panel_margin  = int(20  * self.scale)

        self.font_title  = max(0.9,  0.9  * self.scale)
        self.font_header = max(0.75, 0.75 * self.scale)
        self.font_normal = max(0.65, 0.65 * self.scale)
        self.font_small  = max(0.55, 0.55 * self.scale)

        self.line_title  = int(50 * self.scale)
        self.line_header = int(40 * self.scale)
        self.line_normal = int(35 * self.scale)
        self.line_small  = int(30 * self.scale)
        self.line_row    = int(48 * self.scale)

        self.thick_title  = max(2, int(2 * self.scale))
        self.thick_normal = max(1, int(1 * self.scale))
        self.thick_bold   = max(2, int(2 * self.scale))
        self.box_thickness = max(2, int(2 * self.scale))

        self.vehicle_history   = {}
        self.vehicle_last_seen = {}
        self.PERSIST_FRAMES    = 120

        print(f"\n[Dashboard] Initialized: {width}x{height}")
        print(f"[Dashboard] Scale factor: {self.scale:.2f}x")
        print(f"[Dashboard] Left panel:  {self.left_panel_w}px wide")
        print(f"[Dashboard] Right panel: {self.right_panel_w}px wide")
        print(f"[Dashboard] Font scale:  {self.font_normal:.2f}")

    def _panel(self, frame, x, y, w, h,
               bg=(20, 20, 20), alpha=0.82,
               border=(100, 100, 100), bth=2):
        x  = max(0, x)
        y  = max(0, y)
        x2 = min(frame.shape[1] - 1, x + w)
        y2 = min(frame.shape[0] - 1, y + h)
        if x2 <= x or y2 <= y:
            return
        ov = frame.copy()
        cv2.rectangle(ov, (x, y), (x2, y2), bg, -1)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x2, y2), border, bth)

    def _header(self, frame, x, y, w, h, color):
        x2 = min(frame.shape[1] - 1, x + w)
        y2 = min(frame.shape[0] - 1, y + h)
        ov = frame.copy()
        cv2.rectangle(ov, (x, y), (x2, y2), color, -1)
        cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)

    def _text(self, frame, text, x, y,
              fs=None, color=(255, 255, 255), th=None, shadow=True):
        if fs is None:
            fs = self.font_normal
        if th is None:
            th = self.thick_normal
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = int(x), int(y)
        if shadow:
            cv2.putText(frame, text, (x + 2, y + 2), font, fs,
                        (0, 0, 0), th + 1)
        cv2.putText(frame, text, (x, y), font, fs, color, th)

    def _line(self, frame, x1, y1, x2, y2,
              color=(80, 80, 80), th=1):
        cv2.line(frame,
                 (int(x1), int(y1)), (int(x2), int(y2)),
                 color, th)

    def _row_bg(self, frame, x, y, w, h, color, alpha=0.4):
        x2 = min(frame.shape[1] - 1, x + w)
        y2 = min(frame.shape[0] - 1, y + h)
        if x2 <= x or y2 <= y:
            return
        ov = frame.copy()
        cv2.rectangle(ov, (x, y), (x2, y2), color, -1)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

    def update_vehicle_history(self, vehicles, frame_num):
        for v in vehicles:
            tid = v.get('track_id')
            if tid is None:
                continue
            self.vehicle_history[tid]   = v
            self.vehicle_last_seen[tid] = frame_num
        expired = [
            tid for tid, last in self.vehicle_last_seen.items()
            if frame_num - last > self.PERSIST_FRAMES
        ]
        for tid in expired:
            self.vehicle_history.pop(tid, None)
            self.vehicle_last_seen.pop(tid, None)

    def draw_box(self, frame, bbox, track_id, info, violations=False):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = self.COLOR_RED if violations else self.COLOR_GREEN
        th    = self.box_thickness + 1 if violations else self.box_thickness
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, th)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, t_h), _ = cv2.getTextSize(info, font, self.font_normal,
                                       self.thick_bold)
        lbl_y1 = max(0, y1 - t_h - int(14 * self.scale))
        cv2.rectangle(frame, (x1, lbl_y1),
                      (x1 + tw + int(10 * self.scale), y1),
                      color, -1)
        cv2.putText(frame, info,
                    (x1 + int(5 * self.scale),
                     y1 - int(6 * self.scale)),
                    font, self.font_normal, self.COLOR_WHITE,
                    self.thick_bold)

    def draw_plate_label(self, frame, x, y, plate_text, high_conf=True):
        if not plate_text:
            return
        if plate_text == "SCANNING":
            color = (80, 80, 80)
        elif high_conf:
            color = self.COLOR_BLUE
        else:
            color = (120, 120, 120)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, t_h), _ = cv2.getTextSize(
            plate_text, font, self.font_normal, self.thick_bold
        )
        x = max(0, min(x, self.width  - tw - 10))
        y = max(t_h + 5, min(y, self.height - 5))
        pad = int(5 * self.scale)
        cv2.rectangle(frame,
                      (x - pad, y - t_h - pad),
                      (x + tw + pad, y + pad),
                      color, -1)
        cv2.putText(frame, plate_text, (x, y),
                    font, self.font_normal, self.COLOR_WHITE,
                    self.thick_bold)

    def draw_violation_label(self, frame, x, y, message):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, t_h), _ = cv2.getTextSize(
            message, font, self.font_normal, self.thick_bold
        )
        x = max(0, min(x, self.width  - tw - 10))
        y = max(t_h + 5, min(y, self.height - 5))
        pad = int(5 * self.scale)
        cv2.rectangle(frame,
                      (x - pad, y - t_h - pad),
                      (x + tw + pad, y + pad),
                      self.COLOR_RED, -1)
        cv2.putText(frame, message, (x, y),
                    font, self.font_normal, self.COLOR_WHITE,
                    self.thick_bold)

    def draw_left_panel(self, frame, stats, density_info, flow_text):
        m  = self.panel_margin
        pw = self.left_panel_w
        ph = min(int(900 * self.scale), self.height - m * 2)
        x  = m
        y  = m

        self._panel(frame, x, y, pw, ph,
                    bg=(15, 15, 35), border=(60, 120, 200))

        bar_h = self.line_title + int(10 * self.scale)
        self._header(frame, x, y, pw, bar_h, (40, 40, 120))

        title = "TRAFFIC MONITOR"
        (tw, _), _ = cv2.getTextSize(
            title, cv2.FONT_HERSHEY_SIMPLEX,
            self.font_title, self.thick_title
        )
        self._text(frame, title,
                   x + pw // 2 - tw // 2,
                   y + bar_h - int(10 * self.scale),
                   fs=self.font_title,
                   color=self.COLOR_YELLOW,
                   th=self.thick_title)

        y_cur = y + bar_h + int(15 * self.scale)

        # Timestamp
        ts = datetime.now().strftime("%H:%M:%S   %d/%m/%Y")
        self._text(frame, ts,
                   x + int(15 * self.scale), y_cur,
                   fs=self.font_small, color=self.COLOR_LIGHT_GRAY)
        y_cur += self.line_small

        self._line(frame, x + 10, y_cur, x + pw - 10, y_cur,
                   color=(60, 60, 100), th=2)
        y_cur += int(18 * self.scale)

        # Vehicle counts
        self._text(frame, "VEHICLE COUNTS",
                   x + int(15 * self.scale), y_cur,
                   fs=self.font_header, color=self.COLOR_ACCENT,
                   th=self.thick_bold)
        y_cur += self.line_header

        total  = stats.get('total_vehicles', 0)
        active = stats.get('active_tracks', 0)

        self._text(frame, f"  Total Counted : {total:4d}",
                   x + int(20 * self.scale), y_cur,
                   fs=self.font_normal, color=self.COLOR_GREEN,
                   th=self.thick_bold)
        y_cur += self.line_normal

        self._text(frame, f"  Active Now    : {active:4d}",
                   x + int(20 * self.scale), y_cur,
                   fs=self.font_normal, color=self.COLOR_WHITE)
        y_cur += self.line_normal

        self._line(frame, x + 20, y_cur, x + pw - 20, y_cur,
                   color=(50, 50, 80))
        y_cur += int(12 * self.scale)

        # Per-class
        class_colors = {
            'car': self.COLOR_GREEN,
            'truck': self.COLOR_ORANGE,
            'bus': self.COLOR_YELLOW,
            'motorbike': self.COLOR_CYAN,
            'threewheel': (255, 80, 255),
            'van': (180, 180, 255),
        }
        vehicle_counts = stats.get('vehicle_counts', {})
        if vehicle_counts:
            for vclass, count in sorted(vehicle_counts.items()):
                color = class_colors.get(vclass, self.COLOR_WHITE)
                self._text(
                    frame,
                    f"  {vclass.upper():<12s}: {count:4d}",
                    x + int(20 * self.scale), y_cur,
                    fs=self.font_normal, color=color
                )
                y_cur += self.line_normal
        else:
            self._text(frame, "  No vehicles yet",
                       x + int(20 * self.scale), y_cur,
                       fs=self.font_normal, color=self.COLOR_GRAY)
            y_cur += self.line_normal

        y_cur += int(8 * self.scale)
        self._line(frame, x + 10, y_cur, x + pw - 10, y_cur,
                   color=(60, 60, 100), th=2)
        y_cur += int(18 * self.scale)

        # Traffic status
        self._text(frame, "TRAFFIC STATUS",
                   x + int(15 * self.scale), y_cur,
                   fs=self.font_header, color=self.COLOR_ACCENT,
                   th=self.thick_bold)
        y_cur += self.line_header

        density_level, density_count = density_info
        density_color = {
            'Low':       self.COLOR_GREEN,
            'Medium':    self.COLOR_YELLOW,
            'High':      self.COLOR_ORANGE,
            'Very High': self.COLOR_RED,
        }.get(density_level, self.COLOR_WHITE)

        self._text(frame,
                   f"  Density : {density_level} ({density_count})",
                   x + int(20 * self.scale), y_cur,
                   fs=self.font_normal, color=density_color,
                   th=self.thick_bold)
        y_cur += self.line_normal

        self._text(frame, f"  Flow    : {flow_text}",
                   x + int(20 * self.scale), y_cur,
                   fs=self.font_normal, color=self.COLOR_WHITE)
        y_cur += self.line_normal

        # Violations summary
        y_cur += int(8 * self.scale)
        self._line(frame, x + 10, y_cur, x + pw - 10, y_cur,
                   color=(60, 60, 100), th=2)
        y_cur += int(18 * self.scale)

        self._text(frame, "VIOLATIONS",
                   x + int(15 * self.scale), y_cur,
                   fs=self.font_header, color=self.COLOR_ACCENT,
                   th=self.thick_bold)
        y_cur += self.line_header

        total_v = sum(
            1 for v in self.vehicle_history.values()
            if v.get('violations')
        )
        speed_v = sum(
            1 for v in self.vehicle_history.values()
            if any(vv.get('type') == 'OVERSPEED'
                   for vv in v.get('violations', []))
        )

        v_color = self.COLOR_RED if total_v > 0 else self.COLOR_GREEN
        self._text(frame, f"  Total : {total_v:3d}",
                   x + int(20 * self.scale), y_cur,
                   fs=self.font_normal, color=v_color,
                   th=self.thick_bold)
        y_cur += self.line_normal

        self._text(frame, f"  Overspeed : {speed_v:3d}",
                   x + int(20 * self.scale), y_cur,
                   fs=self.font_normal,
                   color=self.COLOR_ORANGE if speed_v > 0 else self.COLOR_WHITE)

    def draw_right_panel(self, frame, frame_num):
        m  = self.panel_margin
        pw = self.right_panel_w
        ph = self.height - m * 2
        x  = self.width - pw - m
        y  = m

        self._panel(frame, x, y, pw, ph,
                    bg=(15, 30, 15), border=(60, 180, 60))

        bar_h = self.line_title + int(10 * self.scale)
        self._header(frame, x, y, pw, bar_h, (20, 80, 20))

        n_v     = len(self.vehicle_history)
        title   = f"ACTIVE VEHICLES  [{n_v}]"
        (tw, _), _ = cv2.getTextSize(
            title, cv2.FONT_HERSHEY_SIMPLEX,
            self.font_title, self.thick_title
        )
        self._text(frame, title,
                   x + pw // 2 - tw // 2,
                   y + bar_h - int(10 * self.scale),
                   fs=self.font_title, color=self.COLOR_YELLOW,
                   th=self.thick_title)

        y_cur = y + bar_h + int(10 * self.scale)

        # Column headers
        s = self.scale
        col_id    = int(12  * s)
        col_type  = int(75  * s)
        col_speed = int(200 * s)
        col_plate = int(300 * s)
        col_stat  = int(450 * s)

        hdr_y = y_cur + self.line_small
        for label, col in [("ID", col_id), ("TYPE", col_type),
                            ("SPEED", col_speed), ("PLATE", col_plate),
                            ("STATUS", col_stat)]:
            self._text(frame, label, x + col, hdr_y,
                       fs=self.font_small, color=self.COLOR_ACCENT,
                       th=self.thick_bold, shadow=False)

        y_cur = hdr_y + int(10 * s)
        self._line(frame, x + 8, y_cur, x + pw - 8, y_cur,
                   color=(60, 150, 60), th=2)
        y_cur += int(10 * s)

        row_h     = self.line_row
        available = (y + ph) - y_cur - int(30 * s)
        max_rows  = max(1, available // row_h)

        vehicles = sorted(
            self.vehicle_history.values(),
            key=lambda v: (
                0 if v.get('violations') else 1,
                -self.vehicle_last_seen.get(v.get('track_id', 0), 0)
            )
        )

        displayed = 0
        for vehicle in vehicles:
            if displayed >= max_rows:
                break
            if y_cur + row_h > y + ph - int(30 * s):
                break

            try:
                track_id   = vehicle.get('track_id', 0)
                vclass     = str(vehicle.get('class', 'unknown'))
                speed      = float(vehicle.get('speed', 0.0) or 0.0)
                plate      = str(vehicle.get('plate', '') or '')
                plate_conf = float(vehicle.get('plate_conf', 0.0) or 0.0)
                violations = vehicle.get('violations', []) or []

                row_top    = y_cur - int(5 * s)
                row_bottom = y_cur + row_h - int(8 * s)
                text_y     = y_cur + int(row_h * 0.55)

                if violations:
                    self._row_bg(frame, x + 4, row_top,
                                 pw - 8, row_bottom - row_top,
                                 (100, 0, 0), alpha=0.45)
                elif displayed % 2 == 0:
                    self._row_bg(frame, x + 4, row_top,
                                 pw - 8, row_bottom - row_top,
                                 (0, 40, 0), alpha=0.30)

                # ID
                id_color = self.COLOR_RED if violations else self.COLOR_WHITE
                self._text(frame, f"{track_id:3d}",
                           x + col_id, text_y,
                           fs=self.font_normal, color=id_color,
                           th=self.thick_bold, shadow=False)

                # TYPE
                type_color_map = {
                    'car': self.COLOR_GREEN,
                    'truck': self.COLOR_ORANGE,
                    'bus': self.COLOR_YELLOW,
                    'motorbike': self.COLOR_CYAN,
                    'threewheel': (255, 80, 255),
                    'van': (180, 180, 255),
                }
                type_color = type_color_map.get(
                    vclass.lower(), self.COLOR_WHITE
                )
                self._text(frame, vclass[:8].upper(),
                           x + col_type, text_y,
                           fs=self.font_normal, color=type_color,
                           th=self.thick_normal, shadow=False)

                # SPEED
                if speed > 1.0:
                    spd_txt = f"{speed:5.1f}"
                    spd_col = (self.COLOR_RED if speed > 50 else
                               self.COLOR_ORANGE if speed > 35 else
                               self.COLOR_GREEN)
                else:
                    spd_txt = "  ---"
                    spd_col = self.COLOR_GRAY

                self._text(frame, spd_txt,
                           x + col_speed, text_y,
                           fs=self.font_normal, color=spd_col,
                           th=self.thick_bold, shadow=False)

                # PLATE
                clean_plate = plate.strip()
                if (clean_plate and
                        clean_plate not in ('SCANNING', 'N/A', '--') and
                        len(clean_plate) >= 3):
                    plt_txt = clean_plate[:10]
                    plt_col = (self.COLOR_CYAN if plate_conf > 0.6
                               else self.COLOR_LIGHT_GRAY)
                else:
                    plt_txt = "SCAN..."
                    plt_col = self.COLOR_GRAY

                self._text(frame, plt_txt,
                           x + col_plate, text_y,
                           fs=self.font_small, color=plt_col,
                           th=self.thick_normal, shadow=False)

                # STATUS
                if violations:
                    vtype   = violations[0].get('type', 'VIO')[:7]
                    s_txt   = f"!{vtype}"
                    s_color = self.COLOR_RED
                else:
                    s_txt   = "OK"
                    s_color = self.COLOR_DARK_GREEN

                self._text(frame, s_txt,
                           x + col_stat, text_y,
                           fs=self.font_small, color=s_color,
                           th=self.thick_bold, shadow=False)

                # Row separator
                sep_y = row_bottom + int(3 * s)
                self._line(frame, x + 8, sep_y, x + pw - 8, sep_y,
                           color=(40, 80, 40))

                y_cur += row_h
                displayed += 1

            except Exception as e:
                print(f"[Dashboard] Row error: {e}")
                y_cur += row_h
                displayed += 1

        # Footer
        footer_y = y + ph - int(25 * s)
        if n_v == 0:
            self._text(frame, "  Waiting for vehicles...",
                       x + int(15 * s), y + int(200 * s),
                       fs=self.font_header, color=self.COLOR_GRAY)
        elif n_v > max_rows:
            self._text(frame,
                       f"  ... and {n_v - max_rows} more",
                       x + int(15 * s), footer_y,
                       fs=self.font_small, color=self.COLOR_GRAY)
        else:
            self._text(frame,
                       f"  Showing all {n_v} vehicle(s)",
                       x + int(15 * s), footer_y,
                       fs=self.font_small, color=self.COLOR_GRAY)


# ════════════════════════════════════════════════════════════════════════════
# 8. VIDEO HANDLER
# ════════════════════════════════════════════════════════════════════════════

class VideoHandler:
    """Handle video I/O"""

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.width        = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps          = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.out         = None
        self.output_path = None

        print(f"\n[VideoHandler] {Path(video_path).name}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")

    def init_writer(self, output_path):
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            output_path, fourcc, self.fps,
            (self.width, self.height)
        )
        if not self.out.isOpened():
            raise ValueError(f"Cannot create output: {output_path}")
        print(f"[VideoHandler] Output video initialized: {output_path}")

    def read_frame(self):
        return self.cap.read()

    def write_frame(self, frame):
        if self.out is not None:
            self.out.write(frame)

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.out is not None:
            self.out.release()
        print(f"[VideoHandler] Released all resources")


# ════════════════════════════════════════════════════════════════════════════
# 9. CSV EXPORTER
# ════════════════════════════════════════════════════════════════════════════

class CSVExporter:
    """Complete CSV export"""

    def __init__(self, output_dir):
        self.output_dir    = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_start = datetime.now()
        self.session_id    = self.session_start.strftime("%Y%m%d_%H%M%S")

        self.detections  = []
        self.violations  = []
        self.speeds      = []
        self.plates      = defaultdict(list)
        self.frame_stats = []

        self._lock = threading.Lock()

        print(f"\n[CSVExporter] Output: {self.output_dir}")
        print(f"[CSVExporter] Session: {self.session_id}")

    def add_detection(self, frame, track_id, vehicle_class,
                      bbox, confidence, speed=None, plate=None):
        with self._lock:
            try:
                self.detections.append({
                    'track_id':   int(track_id),
                    'frame':      int(frame),
                    'class':      str(vehicle_class),
                    'bbox_x1':    int(bbox[0]),
                    'bbox_y1':    int(bbox[1]),
                    'bbox_x2':    int(bbox[2]),
                    'bbox_y2':    int(bbox[3]),
                    'bbox_area':  int((bbox[2]-bbox[0])*(bbox[3]-bbox[1])),
                    'confidence': float(confidence),
                    'speed_kmh':  float(speed) if speed is not None else 0.0,
                    'plate_raw':  str(plate) if plate else 'UNKNOWN',
                })
            except Exception as e:
                print(f"[CSVExporter] add_detection error: {e}")

    def add_violation(self, frame, track_id, vehicle_class,
                      violation_type, message, confidence, speed=None):
        with self._lock:
            try:
                self.violations.append({
                    'track_id':       int(track_id),
                    'frame':          int(frame),
                    'class':          str(vehicle_class),
                    'violation_type': str(violation_type),
                    'message':        str(message),
                    'confidence':     float(confidence),
                    'speed_kmh':      float(speed) if speed is not None else 0.0,
                    'timestamp':      datetime.now().isoformat(),
                })
            except Exception as e:
                print(f"[CSVExporter] add_violation error: {e}")

    def add_speed_measurement(self, frame, track_id, speed_kmh,
                              method, confidence):
        with self._lock:
            try:
                self.speeds.append({
                    'track_id':   int(track_id),
                    'frame':      int(frame),
                    'speed_kmh':  float(speed_kmh),
                    'method':     str(method),
                    'confidence': float(confidence),
                })
            except Exception as e:
                print(f"[CSVExporter] add_speed error: {e}")

    def record_plate(self, track_id, plate_text, confidence):
        with self._lock:
            if plate_text and plate_text not in ("UNKNOWN", "SCANNING"):
                self.plates[int(track_id)].append({
                    'plate':      str(plate_text),
                    'confidence': float(confidence),
                })

    def add_frame_stats(self, frame_num, active_vehicles,
                        density_level, flow_direction,
                        violations_count=0):
        with self._lock:
            try:
                self.frame_stats.append({
                    'frame':           int(frame_num),
                    'active_vehicles': int(active_vehicles),
                    'density':         str(density_level),
                    'flow_direction':  str(flow_direction),
                    'violations':      int(violations_count),
                    'timestamp':       datetime.now().isoformat(),
                })
            except Exception as e:
                print(f"[CSVExporter] add_frame_stats error: {e}")

    def _vote_best_plate(self, readings):
        if not readings:
            return None, 0.0
        from collections import Counter
        length_counter = Counter(len(p) for p in readings)
        most_common_len = length_counter.most_common(1)[0][0]
        filtered = ([r for r in readings if len(r) == most_common_len]
                    or readings)
        result = []
        conf_sum = 0.0
        for pos in range(most_common_len):
            chars = [r[pos] for r in filtered if pos < len(r)]
            if chars:
                char_votes = Counter(chars)
                best_char, vote_count = char_votes.most_common(1)[0]
                result.append(best_char)
                conf_sum += vote_count / len(filtered)
        best = ''.join(result)
        final_conf = conf_sum / len(result) if result else 0.0
        return best, float(final_conf)

    def save_all(self, prefix="ates_traffic"):
        print("\n" + "=" * 80)
        print("SAVING CSV EXPORTS (GROUPED BY TRACK ID)")
        print("=" * 80)

        files = []

        with self._lock:

            # 1. Detections
            if self.detections:
                df = pd.DataFrame(self.detections).sort_values(
                    ['track_id', 'frame']
                )
                fp = self.output_dir / \
                     f"{prefix}_{self.session_id}_detections.csv"
                df.to_csv(fp, index=False)
                print(f"  ✓ {fp.name} ({len(df)} rows)")
                files.append(fp)

            # 2. Violations
            if self.violations:
                df = pd.DataFrame(self.violations).sort_values(
                    ['track_id', 'frame']
                )
                fp = self.output_dir / \
                     f"{prefix}_{self.session_id}_violations.csv"
                df.to_csv(fp, index=False)
                print(f"  ✓ {fp.name} ({len(df)} rows)")
                files.append(fp)

            # 3. Plates raw
            plates_data = []
            for tid, readings in self.plates.items():
                for r in readings:
                    plates_data.append({
                        'track_id':   int(tid),
                        'plate_raw':  r['plate'],
                        'confidence': float(r['confidence']),
                    })
            if plates_data:
                df = pd.DataFrame(plates_data).sort_values('track_id')
                fp = self.output_dir / \
                     f"{prefix}_{self.session_id}_plates_raw.csv"
                df.to_csv(fp, index=False)
                print(f"  ✓ {fp.name} ({len(df)} rows)")
                files.append(fp)

            # 4. Speed summary with best plate
            if self.speeds:
                df_speeds = pd.DataFrame(self.speeds)
                grouped = df_speeds.groupby('track_id').agg(
                    avg_speed=('speed_kmh', 'mean'),
                    min_speed=('speed_kmh', 'min'),
                    max_speed=('speed_kmh', 'max'),
                    std_speed=('speed_kmh', 'std'),
                    sample_count=('speed_kmh', 'count'),
                ).reset_index()

                best_plates = {}
                for tid, readings in self.plates.items():
                    plates_only = [r['plate'] for r in readings]
                    plate, _ = self._vote_best_plate(plates_only)
                    best_plates[tid] = plate if plate else "UNKNOWN"

                grouped['best_plate'] = grouped['track_id'].map(
                    lambda tid: best_plates.get(tid, "UNKNOWN")
                )

                vehicle_class_map = {}
                for det in self.detections:
                    tid = det['track_id']
                    if tid not in vehicle_class_map:
                        vehicle_class_map[tid] = det['class']
                grouped['vehicle_class'] = grouped['track_id'].map(
                    lambda tid: vehicle_class_map.get(tid, "unknown")
                )

                for col in ['avg_speed', 'min_speed',
                            'max_speed', 'std_speed']:
                    grouped[col] = grouped[col].round(2)

                fp = self.output_dir / \
                     f"{prefix}_{self.session_id}_speed_summary.csv"
                grouped.to_csv(fp, index=False)
                print(f"  ✓ {fp.name} ({len(grouped)} rows)")
                files.append(fp)

            # 5. Frame stats
            if self.frame_stats:
                df = pd.DataFrame(self.frame_stats)
                fp = self.output_dir / \
                     f"{prefix}_{self.session_id}_frame_stats.csv"
                df.to_csv(fp, index=False)
                print(f"  ✓ {fp.name} ({len(df)} rows)")
                files.append(fp)

            # 6. Summary
            summary = self._generate_summary()
            if summary:
                fp = self.output_dir / \
                     f"{prefix}_{self.session_id}_summary.csv"
                pd.DataFrame([summary]).to_csv(fp, index=False)
                print(f"  ✓ {fp.name}")
                files.append(fp)

        print("=" * 80)
        print(f"✅ Saved {len(files)} files\n")
        return files

    def _generate_summary(self):
        try:
            total_det = len(self.detections)
            total_vio = len(self.violations)
            unique_v  = (len(set(d['track_id'] for d in self.detections))
                         if self.detections else 0)

            summary = {
                'session_id':       self.session_id,
                'start_time':       self.session_start.isoformat(),
                'end_time':         datetime.now().isoformat(),
                'total_detections': total_det,
                'total_violations': total_vio,
                'unique_vehicles':  unique_v,
                'unique_plates':    len(self.plates),
            }

            if self.speeds:
                speeds_list = [s['speed_kmh'] for s in self.speeds
                               if s['speed_kmh'] > 0]
                if speeds_list:
                    summary['avg_speed_all'] = float(np.mean(speeds_list))
                    summary['max_speed_all'] = float(np.max(speeds_list))
                    summary['min_speed_all'] = float(np.min(speeds_list))

            return summary
        except Exception as e:
            print(f"[CSVExporter] Summary error: {e}")
            return None

    def print_summary(self):
        summary = self._generate_summary()
        if not summary:
            return
        print("\n" + "=" * 80)
        print("SESSION SUMMARY")
        print("=" * 80)
        print(f"Session ID:        {summary.get('session_id')}")
        print(f"Start Time:        {summary.get('start_time')}")
        print(f"End Time:          {summary.get('end_time')}")
        print(f"\nDetection Statistics:")
        print(f"  Total Detections:  {summary.get('total_detections', 0)}")
        print(f"  Unique Vehicles:   {summary.get('unique_vehicles', 0)}")
        print(f"  Unique Plates:     {summary.get('unique_plates', 0)}")
        print(f"\nViolations:")
        print(f"  Total Violations:  {summary.get('total_violations', 0)}")
        print(f"\nSpeed Statistics (All Vehicles):")
        print(f"  Average:           {summary.get('avg_speed_all', 0.0):.1f} km/h")
        print(f"  Maximum:           {summary.get('max_speed_all', 0.0):.1f} km/h")
        print(f"  Minimum:           {summary.get('min_speed_all', 0.0):.1f} km/h")
        print("=" * 80 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# 10. MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

class ATESPipelineIntegrated:
    """Complete ATES pipeline - ALL FILES MERGED"""

    def __init__(self, video_path):
        self.config      = ATESConfig()
        self._cleanup_done = False

        print("\n" + "=" * 80)
        print("  ATES PHASE-1 INTEGRATED SYSTEM V2")
        print("  Vehicle Detection + Speed + Plate + Dashboard + CSV")
        print("=" * 80 + "\n")

        self.output_dir = Path("ATES_output")
        self.output_dir.mkdir(exist_ok=True)

        # 1. VIDEO
        print("[STEP 1/8] Loading video...")
        self.video = VideoHandler(video_path)

        # 2. CALIBRATION
        print("[STEP 2/8] Smart calibration...")
        calibrator = SmartCalibrator(
            video_path,
            self.config.MODEL_PATH,
            self.config
        )
        calibration = calibrator.run()
        calibrator.close()

        if calibration is None:
            print("[ERROR] Calibration failed")
            sys.exit(1)

        self.pixel_to_meter = calibration['pixel_to_meter']
        print(f"\nPixel-to-meter: {self.pixel_to_meter:.6f} m/pixel")
        print(f"Calibration method: {calibration['method']}")
        print(f"Confidence: {calibration['confidence']:.2%}")

        # 3. DETECTOR
        print("[STEP 3/8] Loading detector...")
        self.detector = VehicleDetector(self.config.MODEL_PATH, self.config)

        # 4. SPEED ESTIMATOR
        print("[STEP 4/8] Initializing speed estimator...")
        self.speed_estimator = SpeedEstimator(
            self.video.fps, self.config, self.pixel_to_meter
        )
        self.speed_estimator.set_frame_dimensions(
            self.video.height, self.video.width
        )

        # 5. PLATE READER
        print("[STEP 5/8] Loading plate reader...")
        self.plate_reader = UniversalPlateReader(
            self.config.PLATE_OCR_LANGUAGES,
            self.config.PLATE_UPSCALE,
            self.config
        )

        # 6. ANALYTICS
        print("[STEP 6/8] Initializing analytics...")
        self.analyzer          = TrafficAnalyzer()
        self.violation_checker = ViolationChecker(self.config)

        # 7. DASHBOARD
        print("[STEP 7/8] Initializing dashboard...")
        self.dashboard = Dashboard(self.video.width, self.video.height)

        # 8. CSV EXPORTER
        print("[STEP 8/8] Initializing CSV exporter...")
        csv_dir = self.output_dir / "csv"
        csv_dir.mkdir(exist_ok=True)
        self.csv_exporter = CSVExporter(str(csv_dir))

        # Output video
        output_video = str(
            self.output_dir /
            f"ates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        self.video.init_writer(output_video)

        # State
        self.frame_count     = 0
        self.unique_vehicles = set()

        print("\n" + "=" * 80)
        print("✅ INITIALIZATION COMPLETE")
        print("=" * 80)
        print(f"Video:       {Path(video_path).name}")
        print(f"Resolution:  {self.video.width}x"
              f"{self.video.height} @ {self.video.fps:.1f}fps")
        print(f"Calibration: {self.pixel_to_meter:.6f} m/pixel")
        print(f"Output:      {output_video}")
        print("\n[CONTROLS]")
        print("  Q      - Quit")
        print("  P      - Pause/Resume")
        print("  Ctrl+C - Emergency stop")
        print("=" * 80 + "\n")

        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        self.should_stop = False

    def _signal_handler(self, signum, frame):
        if not self.should_stop:
            print("\n[INFO] Ctrl+C - stopping...")
            self.should_stop = True

    def process_frame(self, frame):
        """Process single frame"""
        self.frame_count += 1

        # Detect & track
        _, detections = self.detector.detect_and_track(frame)

        output_frame    = frame.copy()
        current_ids     = set()
        vehicle_display = []
        violation_data  = {}

        for detection in detections:
            track_id   = detection['track_id']
            bbox       = detection['bbox']
            vclass     = detection['class']
            confidence = detection['confidence']

            current_ids.add(track_id)
            self.unique_vehicles.add(track_id)

            trajectory = self.detector.get_trajectory(track_id)

            # Speed
            speed = self.speed_estimator.estimate_speed(track_id, trajectory)
            if speed is None:
                speed = 0.0

            # Plate (every N frames)
            plate      = "SCANNING"
            plate_conf = 0.0

            if self.frame_count % self.config.PLATE_DETECTION_INTERVAL == 0:
                plate_read, conf = self.plate_reader.read_plate(frame, bbox)
                if plate_read:
                    plate      = plate_read
                    plate_conf = conf
                    self.plate_reader.record_plate_for_id(
                        track_id, plate, conf
                    )
                    self.csv_exporter.record_plate(
                        track_id, plate, conf
                    )

            # Get best voted plate
            best_plate, best_conf = self.plate_reader.get_best_plate_for_id(
                track_id
            )
            if best_plate and best_conf > 0.5:
                plate      = best_plate
                plate_conf = best_conf

            plate_display = plate if plate else "SCANNING"

            # Violations
            violations = []
            speed_msg, speed_conf = self.violation_checker.check_speed(
                vclass, speed
            )
            if speed_msg:
                violations.append({
                    'type': 'OVERSPEED', 'message': speed_msg
                })
                self.csv_exporter.add_violation(
                    self.frame_count, track_id, vclass,
                    'OVERSPEED', speed_msg, speed_conf, speed
                )

            violation_data[track_id] = violations

            # CSV exports
            self.csv_exporter.add_detection(
                self.frame_count, track_id, vclass,
                bbox, confidence, speed, plate_display
            )
            if speed > 0:
                self.csv_exporter.add_speed_measurement(
                    self.frame_count, track_id, speed, 'ensemble', 1.0
                )

            # Build vehicle info dict
            vehicle_info = {
                'track_id':   track_id,
                'class':      vclass,
                'bbox':       bbox,
                'speed':      speed,
                'plate':      plate_display,
                'plate_conf': plate_conf,
                'violations': violations,
                'confidence': confidence,
            }
            vehicle_display.append(vehicle_info)

            # Draw detection box
            info_text = (f"ID:{track_id} {vclass} "
                         f"{speed:.1f}km/h")
            self.dashboard.draw_box(
                output_frame, bbox, track_id, info_text,
                violations=len(violations) > 0
            )

            # Draw plate label
            x1, _, _, y2 = bbox
            self.dashboard.draw_plate_label(
                output_frame, x1,
                int(y2 + 30 * self.dashboard.scale),
                f"[{plate_display}]",
                high_conf=(plate_conf > 0.6)
            )

            # Draw violation labels
            for i, v in enumerate(violations):
                viol_y = int(
                    y2
                    + 60 * self.dashboard.scale
                    + i * 30 * self.dashboard.scale
                )
                self.dashboard.draw_violation_label(
                    output_frame, x1, viol_y,
                    f"! {v['type']}"
                )

        # Traffic analysis
        self.analyzer.update_flow(self.detector.track_history)
        density_info = self.analyzer.calculate_density(len(current_ids))
        flow_text    = self.analyzer.get_flow_text()

        stats = self.detector.get_statistics()
        stats['active_tracks'] = len(current_ids)

        # Sync dashboard history
        self.dashboard.update_vehicle_history(
            vehicle_display, self.frame_count
        )

        # Frame stats
        self.csv_exporter.add_frame_stats(
            self.frame_count, len(current_ids),
            density_info[0], flow_text,
            len(violation_data)
        )

        # Draw panels
        self.dashboard.draw_left_panel(
            output_frame, stats, density_info, flow_text
        )
        self.dashboard.draw_right_panel(output_frame, self.frame_count)

        # Cleanup lost tracks
        self.detector.cleanup_lost(current_ids)

        return output_frame

    def run(self):
        """Run pipeline"""
        print(f"\n[INFO] Starting video processing...\n")

        # Window manager
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            sw = int(root.winfo_screenwidth()  * 0.90)
            sh = int(root.winfo_screenheight() * 0.85)
            root.destroy()
        except:
            sw, sh = 1280, 720

        scale = min(sw / self.video.width, sh / self.video.height, 1.0)
        disp_w = int(self.video.width  * scale)
        disp_h = int(self.video.height * scale)
        print(f"[INFO] Display: {disp_w}x{disp_h} (scale={scale:.2f}x)")

        win_name = "ATES Phase-1 Integrated"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, disp_w, disp_h)

        pbar = tqdm(
            total=self.video.total_frames,
            desc="Processing",
            unit="frames",
            ncols=100
        )

        paused = False

        try:
            while not self.should_stop:
                ret, frame = self.video.read_frame()
                if not ret:
                    break

                try:
                    output_frame = self.process_frame(frame)
                    self.video.write_frame(output_frame)

                    # Display (resized)
                    if scale < 1.0:
                        disp_frame = cv2.resize(
                            output_frame, (disp_w, disp_h),
                            interpolation=cv2.INTER_LINEAR
                        )
                    else:
                        disp_frame = output_frame

                    cv2.imshow(win_name, disp_frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord('q'), ord('Q'), 27):
                        print("\n[INFO] Quit pressed")
                        break
                    elif key in (ord('p'), ord('P')):
                        paused = not paused
                        print(f"\n[INFO] {'PAUSED' if paused else 'RESUMED'}")

                    while paused:
                        key2 = cv2.waitKey(100) & 0xFF
                        if key2 in (ord('p'), ord('P'), ord('q'), 27):
                            paused = False
                            break

                except Exception as e:
                    print(f"\n[ERROR] Frame {self.frame_count}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                pbar.update(1)

        finally:
            pbar.close()
            cv2.destroyAllWindows()
            self.cleanup()

    def cleanup(self):
        """Cleanup - runs only once"""
        if self._cleanup_done:
            return
        self._cleanup_done = True

        print("\n[INFO] Cleaning up...")

        try:
            cv2.destroyAllWindows()
        except:
            pass

        try:
            self.video.release()
        except Exception as e:
            print(f"[WARNING] Video release: {e}")

        try:
            self.csv_exporter.save_all()
            self.csv_exporter.print_summary()
        except Exception as e:
            print(f"[WARNING] CSV save error: {e}")
            import traceback
            traceback.print_exc()

        try:
            stats = self.detector.get_statistics()
            print("\n" + "=" * 80)
            print("✅ PROCESSING COMPLETE")
            print("=" * 80)
            print(f"  Frames processed : {self.frame_count}")
            print(f"  Total counted    : {stats.get('total_vehicles', 0)}")
            print(f"  Unique IDs       : {len(self.unique_vehicles)}")
            print(f"\n  Vehicle Breakdown:")
            for vtype, count in sorted(
                stats.get('vehicle_counts', {}).items()
            ):
                print(f"    {vtype:<15s}: {count:3d}")
            print("=" * 80 + "\n")
        except Exception as e:
            print(f"[WARNING] Summary error: {e}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ATES Phase-1 Integrated - Vehicle Detection + Speed + Plate"
    )
    parser.add_argument("video_path", help="Path to input video file")
    args = parser.parse_args()

    if not Path(args.video_path).exists():
        print(f"\n[ERROR] Video not found: {args.video_path}\n")
        sys.exit(1)

    try:
        pipeline = ATESPipelineIntegrated(args.video_path)
        pipeline.run()
        print("\n[SUCCESS] All tasks completed!\n")
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
