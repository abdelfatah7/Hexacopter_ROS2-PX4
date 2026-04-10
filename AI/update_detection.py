#!/usr/bin/env python3
"""
Project Nebula - Threat Detector Node (v2.4.1)
===============================================
Target Hardware: Dell Precision 5560 (i7-11th, 32GB, RTX A2000 4GB)

v2.4.1 - "Patient but Faster":
- MULTI-FACTOR REQUIREMENT: At least 3 of 5 factors must be above their
  individual thresholds to produce a non-zero score.
- SUSTAINED SCORING WINDOW: Score over last 10 frames, 50% must pass.
- CONFIRM FRAMES: 8 frames (~0.27s at 30fps).
- POST-THREAT COOLDOWN: 15 frames (~0.5s) after clearing.
- AFFINE CAMERA COMPENSATION: Full rotation+translation compensation.
- GPU WARMUP: First inference runs at startup to avoid initial lag.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time
import torch
from collections import deque
from ultralytics import YOLO


class ThreatDetector(Node):
    def __init__(self):
        super().__init__('nebula_threat_detector')
        self.bridge = CvBridge()

        # ==================== YOLO ====================
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Loading YOLOv8 on {device.upper()}...")
        self.model = YOLO('yolov8s.pt')
        if device == 'cuda':
            self.get_logger().info("GPU Warmup...")
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, classes=[0], conf=0.25, imgsz=640, verbose=False)
        self.get_logger().info("Model Loaded & Warmed Up!")

        # ==================== Topics ====================
        self.declare_parameter('image_topic',
            '/world/default/model/x500_0/link/cgo3_camera_link/sensor/camera/image')
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        self.subscription = self.create_subscription(
            Image, image_topic, self.image_callback, 10)

        self.threat_pub = self.create_publisher(Bool, '/nebula/threat_status', 10)
        self.target_pub = self.create_publisher(PoseStamped, '/nebula/target_pose', 10)
        self.debug_image_pub = self.create_publisher(Image, '/nebula/cv_debug_image', 10)
        self.threat_score_pub = self.create_publisher(Float32, '/nebula/threat_score', 10)

        # ==================== Tracking ====================
        self.person_history = {}
        self.next_id = 0
        self.MAX_HISTORY = 30
        self.missing_frames = {}

        # ==================== Detection ====================
        self.CONFIDENCE = 0.25
        self.YOLO_IOU = 0.25
        self.YOLO_IMGSZ = 640
        self.TRACKING_THRESHOLD = 100
        self.IOU_WEIGHT = 60.0
        self.MIN_PERSONS_FOR_THREAT = 2
        self.GRACE_PERIOD = 5

        # ==================== Scoring Weights ====================
        self.W_OVERLAP = 0.25
        self.W_PROXIMITY = 0.20
        self.W_AGITATION = 0.25
        self.W_APPROACH = 0.15
        self.W_VERTICAL = 0.15

        # ==================== Individual Factor Thresholds ====================
        # ✨ v2.4: كل عامل عنده threshold فردي
        # لازم العامل يعدي الـ threshold بتاعه عشان "يتحسب" كعامل نشط
        self.FACTOR_THRESHOLDS = {
            'overlap':   0.15,   # minimum overlap ratio
            'proximity': 0.20,   # minimum proximity score
            'agitation': 0.25,   # minimum agitation
            'approach':  0.15,   # minimum approach score
            'vertical':  0.20,   # minimum vertical overlap
        }

        # ✨ v2.4: MULTI-FACTOR REQUIREMENT
        # لازم عدد العوامل النشطة يكون >= العدد ده
        self.MIN_ACTIVE_FACTORS = 3

        # ==================== Distance & Interaction Gates ====================
        self.PROXIMITY_FACTOR = 2.5         # ⬆ من 1.8 - يمسك الأشخاص القريبين من بعض حتى لو بعيدين عن الدرون
        self.APPROACH_SPEED = 1.5
        self.MAX_DISTANCE_FACTOR = 4.0      # ⬆ من 2.5 - يسمح بتحليل أشخاص أبعد عن بعض
        self.MIN_AGITATION_GATE = 0.22
        self.MIN_INTERACTION_SCORE = 0.10   # ⬇ من 0.15 - يسمح بحساب الـ score لأشخاص بعيدين عن الدرون

        # ==================== Confirmation (MORE PATIENT) ====================
        self.THREAT_SCORE_THRESHOLD = 0.48
        self.THREAT_CONFIRM_FRAMES = 15     # ~0.67s @ 30fps
        self.NO_THREAT_FRAMES = 40

        # ✨ v2.4: SUSTAINED SCORING WINDOW
        # بنحسب متوسط الـ score على آخر N فريم
        # spike واحد مش هيعدي الـ threshold
        self.score_history = deque(maxlen=30)
        self.MIN_SUSTAINED_RATIO = 0.60     # 60% من آخر 30 فريم (~1 ثانية)

        # ✨ v2.4: POST-THREAT COOLDOWN
        # بعد ما الـ threat يتلغي، بننتظر قبل ما نسمح بـ threat جديد
        self.POST_THREAT_COOLDOWN = 15      # ~0.5 ثانية (كان 30)
        self.cooldown_counter = 0
        self.in_cooldown = False

        # Startup
        self.STARTUP_COOLDOWN_FRAMES = 45
        self.CAMERA_MOTION_THRESHOLD = 4.0
        self.smooth_cam_motion = 0.0
        self.CAM_MOTION_ALPHA = 0.3

        # Pair consistency
        self.pair_threat_history = {}
        self.PAIR_HISTORY_LEN = 15          # ⬆ أطول عشان أدق
        self.MIN_PAIR_CONSISTENCY = 0.4

        # ==================== State ====================
        self.threat_frame_count = 0
        self.threat_active = False
        self.threat_log_counter = 0
        self.THREAT_LOG_INTERVAL = 15

        # ✨ v2.4.3: THREAT LOCK SYSTEM
        # لما نتأكد من الخناقة، بنحفظ الـ IDs ونفضل متتبعينهم
        # الـ threat بيتلغي بس لما الأشخاص يختفوا من الصورة
        self.locked_threat = False          # هل في threat مقفول عليه؟
        self.locked_p1_id = None            # ID الشخص الأول
        self.locked_p2_id = None            # ID الشخص التاني
        self.locked_missing_frames = 0      # كام فريم الأشخاص مش موجودين
        self.LOCK_RELEASE_FRAMES = 45       # ~1.5 ثانية لازم يختفوا عشان نلغي

        self.prev_frame = None
        self.camera_vx = 0.0
        self.camera_vy = 0.0
        self.camera_affine = None

        self.last_threat_p1 = None
        self.last_threat_p2 = None
        self.last_threat_score = 0.0

        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = None
        self.current_fps = 0.0

        self.get_logger().info("=" * 60)
        self.get_logger().info("Nebula Threat Detector v2.4.1 - Patient but Faster")
        self.get_logger().info(f"  Multi-factor: >= {self.MIN_ACTIVE_FACTORS} of 5 required")
        self.get_logger().info(f"  Confirm frames: {self.THREAT_CONFIRM_FRAMES} (~{self.THREAT_CONFIRM_FRAMES/30:.1f}s)")
        self.get_logger().info(f"  Sustained ratio: {self.MIN_SUSTAINED_RATIO:.0%} of window")
        self.get_logger().info(f"  Post-threat cooldown: {self.POST_THREAT_COOLDOWN} frames")
        self.get_logger().info(f"  Factor thresholds: {self.FACTOR_THRESHOLDS}")
        self.get_logger().info("=" * 60)

    # ============================================================
    #                     Performance
    # ============================================================

    def update_fps(self):
        now = time.time()
        if self.last_frame_time is not None:
            dt = now - self.last_frame_time
            if dt > 0:
                self.fps_history.append(1.0 / dt)
                self.current_fps = sum(self.fps_history) / len(self.fps_history)
        self.last_frame_time = now

    # ============================================================
    #                 Affine Camera Compensation
    # ============================================================

    def estimate_camera_motion(self, current_frame):
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            self.camera_affine = None
            return 0.0, 0.0

        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_frame, maxCorners=40, qualityLevel=0.3,
            minDistance=15, blockSize=7)

        if prev_pts is None or len(prev_pts) < 6:
            self.prev_frame = curr_gray
            self.camera_affine = None
            return 0.0, 0.0

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, curr_gray, prev_pts, None,
            winSize=(15, 15), maxLevel=2)

        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 6:
            self.prev_frame = curr_gray
            self.camera_affine = None
            return 0.0, 0.0

        affine_matrix, inliers = cv2.estimateAffinePartial2D(
            good_prev, good_curr,
            method=cv2.RANSAC, ransacReprojThreshold=3.0)

        if affine_matrix is not None:
            self.camera_affine = affine_matrix
            vx = float(affine_matrix[0, 2])
            vy = float(affine_matrix[1, 2])
        else:
            self.camera_affine = None
            motion = good_curr - good_prev
            vx = float(motion[:, 0].mean())
            vy = float(motion[:, 1].mean())

        raw_motion = math.sqrt(vx**2 + vy**2)
        self.smooth_cam_motion = (
            self.CAM_MOTION_ALPHA * raw_motion +
            (1.0 - self.CAM_MOTION_ALPHA) * self.smooth_cam_motion)

        self.prev_frame = curr_gray
        return vx, vy

    def compensate_point(self, cx, cy):
        if self.camera_affine is None:
            return self.camera_vx, self.camera_vy
        moved_x = (self.camera_affine[0, 0] * cx +
                    self.camera_affine[0, 1] * cy +
                    self.camera_affine[0, 2])
        moved_y = (self.camera_affine[1, 0] * cx +
                    self.camera_affine[1, 1] * cy +
                    self.camera_affine[1, 2])
        return moved_x - cx, moved_y - cy

    # ============================================================
    #                     Preprocessing
    # ============================================================

    def preprocess_image(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.convertScaleAbs(enhanced, alpha=1.2, beta=15)

    # ============================================================
    #                     Geometry
    # ============================================================

    def _compute_iou(self, box1, box2):
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter_area
        return inter_area / union if union > 0 else 0.0

    def compute_overlap_ratio(self, box1, box2):
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        min_area = min(
            (box1[2] - box1[0]) * (box1[3] - box1[1]),
            (box2[2] - box2[0]) * (box2[3] - box2[1]))
        return inter_area / min_area if min_area > 0 else 0.0

    def compute_vertical_overlap(self, box1, box2):
        y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
        min_h = min(box1[3] - box1[1], box2[3] - box2[1])
        return y_overlap / min_h if min_h > 0 else 0.0

    # ============================================================
    #                     Tracking
    # ============================================================

    def get_person_id(self, cx, cy, box):
        best_id = None
        best_score = float('inf')
        for pid, history in self.person_history.items():
            if self.missing_frames.get(pid, 0) > 0:
                continue
            last = history[-1]
            dist = math.sqrt((cx - last['cx'])**2 + (cy - last['cy'])**2)
            if dist > self.TRACKING_THRESHOLD * 1.5:
                continue
            last_box = (last['x1'], last['y1'], last['x2'], last['y2'])
            iou = self._compute_iou(box, last_box)
            score = dist - (iou * self.IOU_WEIGHT)
            if score < self.TRACKING_THRESHOLD and score < best_score:
                best_score = score
                best_id = pid
        if best_id is None:
            best_id = self.next_id
            self.next_id += 1
            self.person_history[best_id] = deque(maxlen=self.MAX_HISTORY)
        return best_id

    # ============================================================
    #                     Motion Analysis
    # ============================================================

    def get_relative_velocity(self, pid):
        """Velocity relative to camera - simple translation-compensated"""
        history = self.person_history.get(pid, [])
        if len(history) < 4:
            return 0.0, 0.0
        raw_vx = (history[-1]['cx'] - history[-4]['cx']) / 3.0
        raw_vy = (history[-1]['cy'] - history[-4]['cy']) / 3.0
        return raw_vx - self.camera_vx, raw_vy - self.camera_vy

    def get_agitation(self, pid):
        """
        ✨ v2.4.2: Agitation محسوبة بطريقة IMMUNE لحركة الكاميرا
        
        بدل ما نحسب direction changes (اللي بتتأثر بالـ camera motion)
        بنحسب مؤشرين مش بيتأثروا بحركة الكاميرا:
        
        1. SIZE VARIANCE: لو الشخص بيتخانق، حجمه بيتغير بسرعة
           (بيتحرك قدام وورا، بينحني، بيوقع). شخص ماشي عادي حجمه ثابت.
        
        2. ASPECT RATIO CHANGES: الشخص بيتغير شكله أثناء الخناقة
           (إيده ممدودة، جسمه مايل). ماشي عادي = شكل ثابت.
        
        الميزة: الكاميرا لما بتتحرك أو بتلف، حجم الشخص مش بيتغير!
        فمفيش false agitation من حركة الدرون.
        """
        history = self.person_history.get(pid, [])
        if len(history) < 10:
            return 0.0

        # 1. SIZE VARIANCE - تغير حجم الشخص
        areas = []
        for h in history:
            w = h.get('w', 0)
            ht = h.get('h', 0)
            if w > 0 and ht > 0:
                areas.append(w * ht)

        if len(areas) < 6:
            return 0.0

        avg_area = sum(areas) / len(areas)
        if avg_area == 0:
            return 0.0

        # Normalized variance of area (0 = ثابت، 1+ = بيتغير)
        area_variance = sum((a - avg_area)**2 for a in areas) / len(areas)
        area_cv = math.sqrt(area_variance) / avg_area  # coefficient of variation

        # 2. ASPECT RATIO CHANGES
        ratios = []
        for h in history:
            w = h.get('w', 1)
            ht = h.get('h', 1)
            if w > 0 and ht > 0:
                ratios.append(w / ht)

        ratio_changes = 0
        for i in range(1, len(ratios)):
            if abs(ratios[i] - ratios[i-1]) > 0.05:
                ratio_changes += 1
        ratio_score = ratio_changes / len(ratios) if ratios else 0.0

        # 3. INTER-FRAME JITTER (size-normalized)
        # حتى لو الكاميرا بتتحرك، الـ jitter النسبي لحجم الشخص بيكون مختلف
        # شخص بيتخانق: حركات مفاجئة وغير متساقة
        # شخص ماشي: حركة smooth
        avg_width = sum(h.get('w', 30) for h in history) / len(history)
        jitter_count = 0
        jitter_total = 0

        for i in range(2, len(history)):
            # بنحسب acceleration (تغير السرعة) مش velocity
            # acceleration أقل تأثراً بحركة الكاميرا من velocity
            dx1 = history[i-1]['cx'] - history[i-2]['cx']
            dy1 = history[i-1]['cy'] - history[i-2]['cy']
            dx2 = history[i]['cx'] - history[i-1]['cx']
            dy2 = history[i]['cy'] - history[i-1]['cy']

            # acceleration = change in velocity
            ax = dx2 - dx1
            ay = dy2 - dy1
            accel = math.sqrt(ax**2 + ay**2)

            # normalize by person size
            if avg_width > 0:
                norm_accel = accel / avg_width
                jitter_total += 1
                if norm_accel > 0.25:  # ⬆ من 0.15 - أقل حساسية لاهتزاز الكاميرا
                    jitter_count += 1

        jitter_score = jitter_count / jitter_total if jitter_total > 0 else 0.0

        # Combined agitation score
        # area_cv: 0.0-0.3 typical range
        # ratio_score: 0.0-1.0
        # jitter_score: 0.0-1.0
        agitation = (
            0.3 * min(area_cv / 0.15, 1.0) +      # size variance
            0.2 * min(ratio_score / 0.3, 1.0) +    # aspect ratio changes
            0.5 * min(jitter_score / 0.4, 1.0)      # acceleration jitter
        )

        return min(agitation, 1.0)

    def get_approach_score(self, p1_id, p2_id, p1, p2):
        """
        ✨ v2.4.2: Approach based on DISTANCE CHANGE over time
        بدل velocity projection (اللي بتتأثر بالكاميرا)
        بنشوف هل المسافة بين الشخصين بتقل ولا لأ
        """
        h1 = self.person_history.get(p1_id, [])
        h2 = self.person_history.get(p2_id, [])

        if len(h1) < 5 or len(h2) < 5:
            return 0.0

        # مسافة دلوقتي
        dist_now = math.sqrt(
            (h1[-1]['cx'] - h2[-1]['cx'])**2 +
            (h1[-1]['cy'] - h2[-1]['cy'])**2)

        # مسافة من 4 فريمات
        dist_prev = math.sqrt(
            (h1[-4]['cx'] - h2[-4]['cx'])**2 +
            (h1[-4]['cy'] - h2[-4]['cy'])**2)

        # لو المسافة بتقل = approaching
        if dist_prev <= 0:
            return 0.0

        closing_rate = (dist_prev - dist_now) / dist_prev  # نسبة الاقتراب

        # closing_rate > 0 = بيقربوا
        # closing_rate > 0.1 = بيقربوا بسرعة
        if closing_rate <= 0:
            return 0.0

        return min(closing_rate / 0.3, 1.0)

    # ============================================================
    #                     Pair Consistency
    # ============================================================

    def update_pair_history(self, id1, id2, score):
        key = (min(id1, id2), max(id1, id2))
        if key not in self.pair_threat_history:
            self.pair_threat_history[key] = deque(maxlen=self.PAIR_HISTORY_LEN)
        self.pair_threat_history[key].append(score)

    def get_pair_consistency(self, id1, id2):
        key = (min(id1, id2), max(id1, id2))
        history = self.pair_threat_history.get(key, [])
        if len(history) < 3:
            return 1.0
        above = sum(1 for s in history if s > self.THREAT_SCORE_THRESHOLD * 0.5)
        return above / len(history)

    def cleanup_pair_history(self, active_ids):
        to_delete = [k for k in self.pair_threat_history
                     if k[0] not in active_ids and k[1] not in active_ids]
        for key in to_delete:
            del self.pair_threat_history[key]

    # ============================================================
    #          Threat Scoring v2.4 - Multi-Factor + Patient
    # ============================================================

    def compute_threat_score(self, p1, p2, p1_id, p2_id):
        box1 = (p1['x1'], p1['y1'], p1['x2'], p1['y2'])
        box2 = (p2['x1'], p2['y1'], p2['x2'], p2['y2'])

        # GATE 0: Distance (relative to person size)
        distance = math.sqrt((p1['cx'] - p2['cx'])**2 + (p1['cy'] - p2['cy'])**2)
        avg_width = (p1['w'] + p2['w']) / 2.0
        max_dist = avg_width * self.MAX_DISTANCE_FACTOR

        if distance > max_dist:
            return 0.0, self._make_details(gate='TOO_FAR')

        # ✨ v2.4.1: Size-aware factor thresholds
        # لما الأشخاص بعيدين عن الدرون (صغيرين في الصورة)
        # الـ proximity threshold بينخفض عشان المسافة بالبيكسل بتكون صغيرة
        # بس الـ agitation threshold بيفضل ثابت عشان مهم للدقة
        is_far = avg_width < 40  # أشخاص صغيرين = بعيدين عن الدرون

        # Compute raw factor scores
        overlap_raw = self.compute_overlap_ratio(box1, box2)
        overlap_score = min(overlap_raw / 0.8, 1.0)

        dynamic_threshold = avg_width * self.PROXIMITY_FACTOR
        if distance < dynamic_threshold:
            proximity_score = 1.0 - (distance / dynamic_threshold)
        else:
            # ✨ لما يكونوا أبعد من dynamic_threshold بس لسه جوا max_dist
            # نديهم proximity score صغير بدل 0 - خصوصاً لو بعيدين عن الدرون
            if is_far and distance < max_dist:
                proximity_score = 0.1 * (1.0 - distance / max_dist)
            else:
                proximity_score = 0.0

        agitation1 = self.get_agitation(p1_id)
        agitation2 = self.get_agitation(p2_id)
        max_agitation = max(agitation1, agitation2)
        agitation_score = min(max_agitation / 0.8, 1.0)

        approach_score = self.get_approach_score(p1_id, p2_id, p1, p2)

        v_overlap = self.compute_vertical_overlap(box1, box2)
        vertical_score = min(v_overlap / 0.7, 1.0)

        scores = {
            'overlap':   overlap_score,
            'proximity': proximity_score,
            'agitation': agitation_score,
            'approach':  approach_score,
            'vertical':  vertical_score,
        }

        # GATE 1: Agitation minimum (ثابت - مش بيتغير بالمسافة)
        if max_agitation < self.MIN_AGITATION_GATE:
            return 0.0, self._make_details(scores=scores, gate='LOW_AGITATION')

        # GATE 2: Interaction minimum
        # ✨ لو بعيدين عن الدرون، نخفف الـ interaction gate
        effective_interaction_min = self.MIN_INTERACTION_SCORE
        if is_far:
            effective_interaction_min *= 0.5  # نص الحد لو بعيدين

        interaction = (
            (self.W_OVERLAP * overlap_score + self.W_PROXIMITY * proximity_score) /
            (self.W_OVERLAP + self.W_PROXIMITY))
        if interaction < effective_interaction_min:
            return 0.0, self._make_details(scores=scores, gate='LOW_INTERACTION')

        # GATE 3: MULTI-FACTOR CHECK
        # ✨ لو بعيدين عن الدرون، 2 factors كفاية بدل 3
        effective_min_factors = self.MIN_ACTIVE_FACTORS
        if is_far:
            effective_min_factors = 2

        active_factors = 0
        active_list = []
        for name, val in scores.items():
            threshold = self.FACTOR_THRESHOLDS[name]
            # ✨ proximity و overlap thresholds أقل لو بعيدين
            if is_far and name in ('proximity', 'overlap', 'vertical'):
                threshold *= 0.6
            if val >= threshold:
                active_factors += 1
                active_list.append(name[:3].upper())

        if active_factors < effective_min_factors:
            return 0.0, self._make_details(
                scores=scores, gate=f'LOW_FACTORS({active_factors}/{effective_min_factors})',
                active=active_list)

        # All gates passed - compute weighted score
        total_score = (
            self.W_OVERLAP   * overlap_score +
            self.W_PROXIMITY * proximity_score +
            self.W_AGITATION * agitation_score +
            self.W_APPROACH  * approach_score +
            self.W_VERTICAL  * vertical_score
        )

        # Pair consistency
        consistency = self.get_pair_consistency(p1_id, p2_id)
        if consistency < self.MIN_PAIR_CONSISTENCY:
            total_score *= 0.5

        self.update_pair_history(p1_id, p2_id, total_score)

        return total_score, self._make_details(
            scores=scores, gate='PASSED', total=total_score,
            active=active_list, active_count=active_factors)

    def _make_details(self, scores=None, gate='', total=0.0, active=None, active_count=0):
        d = {
            'overlap': 0.0, 'proximity': 0.0, 'agitation': 0.0,
            'approach': 0.0, 'vertical': 0.0, 'total': total,
            'gate': gate, 'active_factors': active_count,
            'active_list': active or []
        }
        if scores:
            d.update(scores)
            d['total'] = total
        return d

    # ============================================================
    #              Sustained Scoring Check (v2.4)
    # ============================================================

    def is_sustained_threat(self, current_score):
        """
        ✨ v2.4.2: بنشيك إن الـ score فضل عالي لفترة
        لازم 15 فريم على الأقل في الـ history عشان نحكم
        وده ~0.5 ثانية - false threat لمدة أقل من كده مش هيعدي
        """
        self.score_history.append(current_score)

        # لازم نكون شوفنا 15 فريم على الأقل عشان نحكم
        if len(self.score_history) < 15:
            return False

        above_count = sum(1 for s in self.score_history
                          if s >= self.THREAT_SCORE_THRESHOLD)
        ratio = above_count / len(self.score_history)
        return ratio >= self.MIN_SUSTAINED_RATIO

    # ============================================================
    #                 Threat Lock System (v2.4.3)
    # ============================================================

    def _engage_lock(self, p1_id, p2_id):
        """تفعيل الـ lock على شخصين بيتخانقوا"""
        self.locked_threat = True
        self.locked_p1_id = p1_id
        self.locked_p2_id = p2_id
        self.locked_missing_frames = 0
        self.get_logger().warn(f"LOCKED on P{p1_id} & P{p2_id}")

    def _release_lock(self):
        """تحرير الـ lock"""
        old_p1 = self.locked_p1_id
        old_p2 = self.locked_p2_id
        self.locked_threat = False
        self.locked_p1_id = None
        self.locked_p2_id = None
        self.locked_missing_frames = 0
        self.threat_frame_count = 0
        self.threat_active = False
        self.last_threat_p1 = None
        self.last_threat_p2 = None
        self.last_threat_score = 0.0
        self.score_history.clear()
        self.in_cooldown = True
        self.cooldown_counter = 0
        self.get_logger().info(f"LOCK RELEASED (P{old_p1} & P{old_p2} disappeared)")

    # ============================================================
    #                     Stability Checks
    # ============================================================

    def is_camera_stable(self):
        return self.smooth_cam_motion < self.CAMERA_MOTION_THRESHOLD

    def is_startup_done(self):
        return self.frame_count > self.STARTUP_COOLDOWN_FRAMES

    # ============================================================
    #                     Main Callback
    # ============================================================

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        self.update_fps()
        self.frame_count += 1

        img_height, img_width = cv_image.shape[:2]
        img_center_x = img_width // 2
        img_center_y = img_height // 2

        self.camera_vx, self.camera_vy = self.estimate_camera_motion(cv_image)

        can_score = self.is_startup_done() and self.is_camera_stable()

        # Post-threat cooldown
        if self.in_cooldown:
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.POST_THREAT_COOLDOWN:
                self.in_cooldown = False
                self.cooldown_counter = 0
            else:
                can_score = False

        # YOLO
        enhanced = self.preprocess_image(cv_image)
        results = self.model(
            enhanced, classes=[0], conf=self.CONFIDENCE,
            iou=self.YOLO_IOU, imgsz=self.YOLO_IMGSZ, verbose=False)

        persons = []
        current_ids = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                width = x2 - x1
                conf = float(box.conf[0])

                pid = self.get_person_id(cx, cy, (x1, y1, x2, y2))
                height = y2 - y1

                self.person_history[pid].append({
                    'cx': cx, 'cy': cy,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'w': width, 'h': height
                })
                current_ids.append(pid)
                persons.append({
                    'cx': cx, 'cy': cy, 'w': width, 'id': pid,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'conf': conf
                })

                # ✨ لو الشخص ده مقفول عليه = أحمر، غير كده = أخضر
                if self.locked_threat and pid in (self.locked_p1_id, self.locked_p2_id):
                    color = (0, 0, 255)
                    thickness = 3
                else:
                    color = (0, 255, 0) if conf > 0.5 else (0, 200, 100)
                    thickness = 2
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(cv_image, f"P{pid} {conf:.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Grace period
        for pid in list(self.person_history.keys()):
            if pid not in current_ids:
                self.missing_frames[pid] = self.missing_frames.get(pid, 0) + 1
                if self.missing_frames[pid] > self.GRACE_PERIOD:
                    del self.person_history[pid]
                    if pid in self.missing_frames:
                        del self.missing_frames[pid]
            else:
                self.missing_frames[pid] = 0

        self.cleanup_pair_history(set(self.person_history.keys()))

        # ==================== Threat Scoring ====================
        best_score = 0.0
        best_details = None
        target_cx, target_cy = img_center_x, img_center_y
        threat_p1 = None
        threat_p2 = None

        if can_score and len(persons) >= self.MIN_PERSONS_FOR_THREAT:
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    p1 = persons[i]
                    p2 = persons[j]
                    score, details = self.compute_threat_score(
                        p1, p2, p1['id'], p2['id'])
                    if score > best_score:
                        best_score = score
                        best_details = details
                        threat_p1 = p1
                        threat_p2 = p2
                        target_cx = (p1['cx'] + p2['cx']) // 2
                        target_cy = (p1['cy'] + p2['cy']) // 2

        # ✨ v2.4: Sustained check
        sustained = self.is_sustained_threat(best_score)
        raw_threat = best_score >= self.THREAT_SCORE_THRESHOLD and sustained

        # Publish score
        score_msg = Float32()
        score_msg.data = float(best_score)
        self.threat_score_pub.publish(score_msg)

        # ============================================================
        #          ✨ v2.4.3: THREAT LOCK SYSTEM
        # ============================================================
        # مرحلة 1: قبل التأكيد - counter عادي
        # مرحلة 2: بعد التأكيد - LOCK على الأشخاص
        #          الـ threat بيتلغي بس لما الأشخاص يختفوا

        if self.locked_threat:
            # === LOCKED MODE ===
            # بنشوف هل الأشخاص المقفول عليهم لسه موجودين في الصورة
            p1_found = self.locked_p1_id in current_ids
            p2_found = self.locked_p2_id in current_ids

            if p1_found or p2_found:
                # لسه موجودين - نفضل مقفولين عليهم
                self.locked_missing_frames = 0
                threat_detected = True

                # نحدث الـ target position لو موجودين
                for p in persons:
                    if p['id'] == self.locked_p1_id:
                        self.last_threat_p1 = p
                    if p['id'] == self.locked_p2_id:
                        self.last_threat_p2 = p

                if self.last_threat_p1 and self.last_threat_p2:
                    target_cx = (self.last_threat_p1['cx'] + self.last_threat_p2['cx']) // 2
                    target_cy = (self.last_threat_p1['cy'] + self.last_threat_p2['cy']) // 2
            else:
                # الأشخاص اختفوا - بنستنى شوية قبل ما نلغي
                self.locked_missing_frames += 1
                if self.locked_missing_frames >= self.LOCK_RELEASE_FRAMES:
                    # اختفوا لفترة كافية - نلغي الـ lock
                    threat_detected = False
                    self._release_lock()
                else:
                    # لسه بنستنى - نفضل على آخر position
                    threat_detected = True
        else:
            # === NORMAL MODE (pre-confirmation) ===
            if raw_threat:
                self.threat_frame_count = min(
                    self.threat_frame_count + 1,
                    self.THREAT_CONFIRM_FRAMES + self.NO_THREAT_FRAMES)
                self.last_threat_p1 = threat_p1
                self.last_threat_p2 = threat_p2
                self.last_threat_score = best_score
            else:
                self.threat_frame_count = max(self.threat_frame_count - 3, 0)

            threat_detected = self.threat_frame_count >= self.THREAT_CONFIRM_FRAMES

            # لو تأكدنا - ندخل LOCK MODE
            if threat_detected and threat_p1 and threat_p2:
                self._engage_lock(threat_p1['id'], threat_p2['id'])

        # ==================== Visualization ====================
        if threat_detected:
            p1 = self.last_threat_p1
            p2 = self.last_threat_p2
            if p1 and p2:
                avg_w = (p1['w'] + p2['w']) / 2.0
                dynamic_r = int(avg_w * self.PROXIMITY_FACTOR)
                cv2.line(cv_image, (p1['cx'], p1['cy']),
                         (p2['cx'], p2['cy']), (0, 165, 255), 3)
                cv2.circle(cv_image, (target_cx, target_cy),
                           dynamic_r, (0, 0, 255), 2)
                cv2.rectangle(cv_image,
                              (p1['x1'], p1['y1']), (p1['x2'], p1['y2']),
                              (0, 0, 255), 3)
                cv2.rectangle(cv_image,
                              (p2['x1'], p2['y1']), (p2['x2'], p2['y2']),
                              (0, 0, 255), 3)

        # Score bar
        bar_x = img_width - 30
        bar_h = int(best_score * 150)
        bar_color = (0, 0, 255) if best_score >= self.THREAT_SCORE_THRESHOLD else (0, 255, 255)
        cv2.rectangle(cv_image, (bar_x, 10), (bar_x + 20, 160), (50, 50, 50), -1)
        cv2.rectangle(cv_image, (bar_x, 160 - bar_h), (bar_x + 20, 160), bar_color, -1)
        cv2.putText(cv_image, f"{best_score:.2f}", (bar_x - 10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, bar_color, 1)

        # Score breakdown
        if best_details:
            y_offset = img_height - 120
            gate = best_details.get('gate', '')
            active_list = best_details.get('active_list', [])
            af = best_details.get('active_factors', 0)

            for key in ['overlap', 'proximity', 'agitation', 'approach', 'vertical']:
                val = best_details.get(key, 0.0)
                thresh = self.FACTOR_THRESHOLDS[key]
                is_active = key[:3].upper() in active_list
                fc = (0, 255, 0) if is_active else (100, 100, 100)
                marker = "*" if is_active else " "
                label = f"{marker}{key[:3].upper()}: {val:.2f} (>{thresh:.2f})"
                cv2.putText(cv_image, label, (img_width - 180, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, fc, 1)
                y_offset += 14

            gate_color = (0, 255, 0) if gate == 'PASSED' else (0, 165, 255)
            cv2.putText(cv_image, f"GATE: {gate}", (img_width - 180, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, gate_color, 1)
            y_offset += 14
            cv2.putText(cv_image, f"Factors: {af}/{self.MIN_ACTIVE_FACTORS}",
                        (img_width - 180, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

        # ==================== Logging ====================
        if threat_detected and not self.threat_active:
            self.threat_active = True
            self.threat_log_counter = 0
            self.get_logger().warn("=" * 60)
            self.get_logger().warn("THREAT CONFIRMED! FIGHT DETECTED!")
            if best_details:
                al = ','.join(best_details.get('active_list', []))
                self.get_logger().warn(
                    f"Score: {self.last_threat_score:.3f} | "
                    f"Active: [{al}] ({best_details.get('active_factors',0)}/5)")
                self.get_logger().warn(
                    f"OVR={best_details.get('overlap',0):.2f} "
                    f"PRX={best_details.get('proximity',0):.2f} "
                    f"AGT={best_details.get('agitation',0):.2f} "
                    f"APR={best_details.get('approach',0):.2f} "
                    f"VRT={best_details.get('vertical',0):.2f}")
            self.get_logger().warn(
                f"Offset: X={target_cx - img_center_x}, Y={target_cy - img_center_y}")
            self.get_logger().warn("=" * 60)

        elif threat_detected and self.threat_active:
            self.threat_log_counter += 1
            if self.threat_log_counter >= self.THREAT_LOG_INTERVAL:
                self.threat_log_counter = 0
                self.get_logger().warn(
                    f"THREAT ACTIVE | Score: {self.last_threat_score:.3f} | "
                    f"Offset: X={target_cx - img_center_x}, Y={target_cy - img_center_y}")

        elif not threat_detected and self.threat_active:
            self.threat_active = False
            self.threat_log_counter = 0
            self.last_threat_p1 = None
            self.last_threat_p2 = None
            self.last_threat_score = 0.0
            # ✨ v2.4: Start cooldown
            self.in_cooldown = True
            self.cooldown_counter = 0
            self.score_history.clear()
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"Threat cleared! Cooldown {self.POST_THREAT_COOLDOWN} frames...")
            self.get_logger().info("=" * 60)

        # ==================== HUD ====================
        scoring_suspended = not can_score
        color = (0, 0, 255) if threat_detected else (0, 255, 255)

        if scoring_suspended:
            if not self.is_startup_done():
                remaining = self.STARTUP_COOLDOWN_FRAMES - self.frame_count
                status = f"STARTING UP... ({remaining})"
                color = (128, 128, 128)
            elif self.in_cooldown:
                remaining = self.POST_THREAT_COOLDOWN - self.cooldown_counter
                status = f"COOLDOWN... ({remaining})"
                color = (0, 200, 200)
            else:
                status = f"CAM UNSTABLE ({self.smooth_cam_motion:.1f})"
                color = (0, 165, 255)
        elif threat_detected:
            if self.locked_threat:
                status = f"THREAT LOCKED! P{self.locked_p1_id}&P{self.locked_p2_id}"
            else:
                status = "THREAT CONFIRMED!"
        else:
            status = f"Monitoring... ({self.threat_frame_count}/{self.THREAT_CONFIRM_FRAMES})"

        cv2.putText(cv_image, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(cv_image, f"Persons: {len(persons)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(cv_image, f"Cam: vx={self.camera_vx:.1f} vy={self.camera_vy:.1f}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(cv_image, f"Score: {best_score:.3f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(cv_image, f"FPS: {self.current_fps:.1f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        stab_color = (0, 255, 0) if self.is_camera_stable() else (0, 0, 255)
        cv2.putText(cv_image, f"Stab: {self.smooth_cam_motion:.1f}",
                    (10, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.40, stab_color, 1)

        # Sustained indicator
        if len(self.score_history) >= 5:
            above = sum(1 for s in self.score_history if s >= self.THREAT_SCORE_THRESHOLD)
            ratio = above / len(self.score_history)
            sus_color = (0, 255, 0) if ratio >= self.MIN_SUSTAINED_RATIO else (100, 100, 100)
            cv2.putText(cv_image, f"Sus: {ratio:.0%}", (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, sus_color, 1)

        aff = "AFF" if self.camera_affine is not None else "TRN"
        cv2.putText(cv_image, aff, (10, 172),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # ==================== Publish ====================
        msg_threat = Bool()
        msg_threat.data = threat_detected
        self.threat_pub.publish(msg_threat)

        if threat_detected:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "camera_link"
            pose_msg.pose.position.x = float(target_cx - img_center_x)
            pose_msg.pose.position.y = float(target_cy - img_center_y)
            pose_msg.pose.position.z = 0.0
            self.target_pub.publish(pose_msg)

        cv2.drawMarker(cv_image, (img_center_x, img_center_y),
                       (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
        debug_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.debug_image_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    detector = ThreatDetector()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
