import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import VehicleLocalPosition
from cv_bridge import CvBridge
import cv2, numpy as np, math, time, torch
from collections import deque
from ultralytics import YOLO


class ThreatDetector(Node):

    def __init__(self):
        super().__init__('nebula_threat_detector')
        self.bridge = CvBridge()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Loading YOLOv8 on {device.upper()}...")
        self.model = YOLO('yolov8s.pt')
        if device == 'cuda':
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, classes=[0], conf=0.25, imgsz=640, verbose=False)
        self.get_logger().info("Model ready!")

        self.declare_parameter('image_topic',
            '/world/default/model/x500_0/link/cgo3_camera_link/sensor/camera/image')
        itopic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.subscription = self.create_subscription(
            Image, itopic, self.image_callback, 10)

        self.threat_pub = self.create_publisher(Bool,        '/nebula/threat_status',  10)
        self.target_pub = self.create_publisher(PoseStamped, '/nebula/target_pose',    10)
        self.debug_pub  = self.create_publisher(Image,       '/nebula/cv_debug_image', 10)
        self.score_pub  = self.create_publisher(Float32,     '/nebula/threat_score',   10)
        self.pair_pub   = self.create_publisher(String,      '/nebula/locked_pair_id', 10)

        self.create_subscription(String, '/nebula/blacklist_pair',
                                 self._blacklist_cb, 10)
        self.create_subscription(VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self._pos_cb, qos_profile_sensor_data)
        self.create_subscription(String, '/nebula/mission_state',
                                 self._state_cb, 10)

        # ── Tracking ────────────────────────────────────────────────
        self.person_history  = {}
        self.next_id         = 0
        self.MAX_HISTORY     = 30
        self.missing_frames  = {}
        self.GRACE_PERIOD    = 8

        # ── Detection ───────────────────────────────────────────────
        self.CONFIDENCE  = 0.28
        self.YOLO_IOU    = 0.25
        self.YOLO_IMGSZ  = 640
        self.TRACK_THRESH= 110
        self.IOU_W       = 60.0
        self.MIN_PERSONS = 2

        # ── Scoring weights ─────────────────────────────────────────
        self.W_OVL = 0.20
        self.W_PRX = 0.25
        self.W_AGI = 0.35
        self.W_APP = 0.10
        self.W_VER = 0.10

        # ── Scoring thresholds ──────────────────────────────────────
        self.FACTOR_THR = {
            'overlap'  : 0.15,
            'proximity': 0.20,
            'agitation': 0.28,
            'approach' : 0.15,
            'vertical' : 0.18
        }
        self.MIN_FACTORS      = 3
        self.PROXIMITY_K      = 2.5
        self.MAX_DIST_K       = 4.5

        
        self.MIN_AGIT_GATE    = 0.28   # at least ONE
        self.MIN_BOTH_AGIT    = 0.15   # BOTH must exceed
        self.MIN_MOTION_SIG   = 0.15   # at least ONE
        self.MIN_BOTH_MOTION  = 0.08   # BOTH must exceed
        self.MIN_INTERACT     = 0.10

        self.MAX_AGIT_RATIO   = 3.5
        self.MAX_MOTION_RATIO = 3.5

        # ── Confirmation ─────────────────────────────────────────────
        self.SCORE_THRESH     = 0.48
        self.CONFIRM_FRAMES   = 8
        self.NO_THREAT_FRAMES = 35
        self.score_history    = deque(maxlen=15)
        self.MIN_SUS_RATIO    = 0.50
        self.COOLDOWN_FRAMES  = 25
        self.cooldown_counter = 0
        self.in_cooldown      = False
        self.STARTUP_FRAMES   = 30
        self.CAM_MOTION_THR   = 6.0
        self.smooth_cam_motion= 0.0
        self.CAM_ALPHA        = 0.3
        self.pair_hist        = {}
        self.PAIR_HIST_LEN    = 8
        self.MIN_CONSISTENCY  = 0.25

        # ── Lock ─────────────────────────────────────────────────────
        self.locked_threat   = False
        self.locked_p1       = None
        self.locked_p2       = None
        self.miss_frames     = 0
        self.LOCK_RELEASE_FR = 60
        self.MIN_LOCK_SEC    = 8.0
        self.lock_t          = None
        self.live_scores     = deque(maxlen=100)
        self.VAL_AFTER_SEC   = 2.0
        self.VAL_FRAMES      = 40
        self.VAL_THRESH      = 0.14
        self.fight_cx        = 0
        self.fight_cy        = 0
        self.fight_avg_w     = 50
        self.MIN_REC_MOTION  = 0.04
        self.REC_RADIUS_K    = 2.0

        # ── Blacklists ───────────────────────────────────────────────
        self.bl_pairs    : set  = set()
        self.geo_bl      : list = []
        self.BL_RADIUS   = 6.0       # ★ زودنا شوية
        self.BL_DURATION = 600.0
        self.CAM_FOV_DEG = 80.0
        self.img_w = 0
        self.img_h = 0

        # ── Drone state ──────────────────────────────────────────────
        self.drone_x   = 0.
        self.drone_y   = 0.
        self.drone_z   = 0.
        self.drone_yaw = 0.
        self.drone_ready = False

        # ── Mission state ─────────────────────────────────────────────
        self.SUPPRESS_STATES = {"RETURNING", "COOLDOWN"}
        self.mission_state   = "MISSION"

        # ── Optical flow ─────────────────────────────────────────────
        self.prev_frame = None
        self.cam_vx     = 0.
        self.cam_vy     = 0.
        self.cam_aff    = None

        # ── General state ─────────────────────────────────────────────
        self.last_p1       = None
        self.last_p2       = None
        self.last_score    = 0.
        self.threat_cnt    = 0
        self.threat_active = False
        self.log_cnt       = 0
        self.LOG_INTERVAL  = 10
        self.frame_cnt     = 0
        self.fps_hist      = deque(maxlen=30)
        self.last_t        = None
        self.fps           = 0.

        self.get_logger().info("=" * 55)
        self.get_logger().info("Nebula Threat Detector v7.0 — FALSE PAIR + REDETECT FIXED")
        self.get_logger().info("=" * 55)

    # ────────────────────────────────────────────────────────────────
    # External callbacks
    # ────────────────────────────────────────────────────────────────
    def _pos_cb(self, msg):
        try:
            self.drone_x    = msg.x
            self.drone_y    = msg.y
            self.drone_z    = msg.z
            self.drone_yaw  = msg.heading
            self.drone_ready = True
        except Exception as e:
            self.get_logger().error(f"_pos_cb: {e}")

    def _state_cb(self, msg):
        try:
            self.mission_state = msg.data
        except Exception as e:
            self.get_logger().error(f"_state_cb: {e}")

    def _blacklist_cb(self, msg):
        """
     
        """
        try:
            key = msg.data.strip()
            if not key:
                return
            self.bl_pairs.add(key)

            if self.drone_ready:
                # ★ project fight centroid to world coords
                fight_world = self._project(self.fight_cx, self.fight_cy)
                if fight_world is not None:
                    fx, fy = fight_world
                else:
                    fx, fy = self.drone_x, self.drone_y

                # radius scales with how big the fighters appear in frame
                dynamic_radius = max(
                    self.BL_RADIUS,
                    self.fight_avg_w * 0.05 + 3.0
                )

                self.geo_bl.append({
                    'x'     : fx,
                    'y'     : fy,
                    'expiry': time.time() + self.BL_DURATION,
                    'pair'  : key,
                    'radius': dynamic_radius
                })
                self.get_logger().warn(
                    f"🚫 GEO-BL fight_pos=({fx:.1f},{fy:.1f})"
                    f" r={dynamic_radius:.1f}m pair={key}")

            if self.locked_threat:
                self._release_lock("mission blacklisted")

        except Exception as e:
            self.get_logger().error(f"_blacklist_cb error: {e}")

    # ────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────
    def _pk(self, a, b):
        x, y = sorted([a, b])
        return f"{x}_{y}"

    def _id_bl(self, a, b):
        return self._pk(a, b) in self.bl_pairs

    def _project(self, px, py):
        """Pinhole projection: image pixel → world (NED)."""
        if not self.drone_ready or self.img_w <= 0:
            return None
        alt = max(abs(self.drone_z), 1.0)
        fov = math.radians(self.CAM_FOV_DEG)
        hw  = alt * math.tan(fov / 2)
        mpp = hw / (self.img_w / 2)
        dx  =  (px - self.img_w / 2) * mpp
        dy  = -(py - self.img_h / 2) * mpp
        cy_ = math.cos(self.drone_yaw)
        sy_ = math.sin(self.drone_yaw)
        return (self.drone_x + dy * cy_ - dx * sy_,
                self.drone_y + dy * sy_ + dx * cy_)

    def _pair_geo_bl(self, mpx, mpy):
        """
        """
        if not self.geo_bl:
            return False
        now = time.time()
        self.geo_bl = [e for e in self.geo_bl if e['expiry'] > now]
        if not self.geo_bl:
            return False

        world = self._project(mpx, mpy)
        if world is None:
            # fallback: use drone position
            if not self.drone_ready:
                return False
            for e in self.geo_bl:
                r = e.get('radius', self.BL_RADIUS)
                if math.sqrt((self.drone_x - e['x']) ** 2 +
                             (self.drone_y - e['y']) ** 2) < r:
                    return True
            return False

        fx, fy = world
        for e in self.geo_bl:
            r = e.get('radius', self.BL_RADIUS)
            if math.sqrt((fx - e['x']) ** 2 + (fy - e['y']) ** 2) < r:
                return True
        return False

    # ── Performance ──────────────────────────────────────────────────
    def _upd_fps(self):
        now = time.time()
        if self.last_t and now > self.last_t:
            self.fps_hist.append(1. / (now - self.last_t))
            self.fps = sum(self.fps_hist) / len(self.fps_hist)
        self.last_t = now

    # ── Camera motion ─────────────────────────────────────────────────
    def _cam_motion(self, frame):
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.cam_aff = None
            return 0., 0.
        cg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pp = cv2.goodFeaturesToTrack(self.prev_frame, 40, 0.3, 15, blockSize=7)
        if pp is None or len(pp) < 6:
            self.prev_frame = cg
            self.cam_aff = None
            return 0., 0.
        cp, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, cg, pp, None,
            winSize=(15, 15), maxLevel=2)
        gp = pp[st == 1]
        gc = cp[st == 1]
        if len(gp) < 6:
            self.prev_frame = cg
            self.cam_aff = None
            return 0., 0.
        am, _ = cv2.estimateAffinePartial2D(
            gp, gc, method=cv2.RANSAC, ransacReprojThreshold=3.)
        if am is not None:
            self.cam_aff = am
            vx = float(am[0, 2])
            vy = float(am[1, 2])
        else:
            self.cam_aff = None
            m  = gc - gp
            vx = float(m[:, 0].mean())
            vy = float(m[:, 1].mean())
        raw = math.sqrt(vx * vx + vy * vy)
        self.smooth_cam_motion = (self.CAM_ALPHA * raw +
                                  (1 - self.CAM_ALPHA) * self.smooth_cam_motion)
        self.prev_frame = cg
        return vx, vy

    # ── Preprocessing ─────────────────────────────────────────────────
    def _pre(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(2., (4, 4))
        l  = cl.apply(l)
        e  = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return cv2.convertScaleAbs(e, alpha=1.2, beta=15)

    # ── Geometry ──────────────────────────────────────────────────────
    def _iou(self, b1, b2):
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        i   = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        u   = ((b1[2]-b1[0])*(b1[3]-b1[1]) +
               (b2[2]-b2[0])*(b2[3]-b2[1]) - i)
        return i / u if u > 0 else 0.

    def _ovl(self, b1, b2):
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        i   = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ma  = min((b1[2]-b1[0])*(b1[3]-b1[1]),
                  (b2[2]-b2[0])*(b2[3]-b2[1]))
        return i / ma if ma > 0 else 0.

    def _vovl(self, b1, b2):
        yo = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        mh = min(b1[3]-b1[1], b2[3]-b2[1])
        return yo / mh if mh > 0 else 0.

    # ── Tracking ──────────────────────────────────────────────────────
    def _get_id(self, cx, cy, box, cur):
        bid = None
        bs  = float('inf')
        for pid, hist in self.person_history.items():
            if self.missing_frames.get(pid, 0) > 0 or pid in cur:
                continue
            last = hist[-1]
            d    = math.sqrt((cx - last['cx'])**2 + (cy - last['cy'])**2)
            if d > self.TRACK_THRESH * 1.5:
                continue
            lb = (last['x1'], last['y1'], last['x2'], last['y2'])
            sc = d - self._iou(box, lb) * self.IOU_W
            if sc < self.TRACK_THRESH and sc < bs:
                bs  = sc
                bid = pid
        if bid is None:
            bid = self.next_id
            self.next_id += 1
            self.person_history[bid] = deque(maxlen=self.MAX_HISTORY)
        return bid

    # ── Motion analysis ───────────────────────────────────────────────
    def _agit(self, pid):
        h = self.person_history.get(pid, [])
        if len(h) < 6:
            return 0.
        areas = [f.get('w', 0) * f.get('h', 0) for f in h if f.get('w', 0) > 0]
        if len(areas) < 4:
            return 0.
        avg = sum(areas) / len(areas)
        if avg == 0:
            return 0.
        acv = math.sqrt(sum((a - avg)**2 for a in areas) / len(areas)) / avg
        ra  = [f.get('w', 1) / max(f.get('h', 1), 1) for f in h]
        rc  = sum(1 for i in range(1, len(ra))
                  if abs(ra[i] - ra[i-1]) > 0.04) / len(ra)
        aw  = sum(f.get('w', 30) for f in h) / len(h)
        jc  = jt = 0
        for i in range(2, len(h)):
            dx1 = h[i-1]['cx'] - h[i-2]['cx']
            dy1 = h[i-1]['cy'] - h[i-2]['cy']
            dx2 = h[i]['cx']   - h[i-1]['cx']
            dy2 = h[i]['cy']   - h[i-1]['cy']
            ax  = (dx2 - self.cam_vx) - (dx1 - self.cam_vx)
            ay  = (dy2 - self.cam_vy) - (dy1 - self.cam_vy)
            jt += 1
            if aw > 0 and math.sqrt(ax*ax + ay*ay) / aw > 0.20:
                jc += 1
        js = jc / jt if jt > 0 else 0.
        return min(
            0.3 * min(acv / 0.08, 1.) +
            0.2 * min(rc  / 0.15, 1.) +
            0.5 * min(js  / 0.20, 1.), 1.)

    def _motion(self, pid):
        h = self.person_history.get(pid, [])
        if len(h) < 5:
            return 0.
        aw = sum(f.get('w', 30) for f in h) / len(h)
        if aw <= 0:
            return 0.
        peak = 0.
        for i in range(2, len(h)):
            dx1 = h[i-1]['cx'] - h[i-2]['cx']
            dy1 = h[i-1]['cy'] - h[i-2]['cy']
            dx2 = h[i]['cx']   - h[i-1]['cx']
            dy2 = h[i]['cy']   - h[i-1]['cy']
            ax  = (dx2 - self.cam_vx) - (dx1 - self.cam_vx)
            ay  = (dy2 - self.cam_vy) - (dy1 - self.cam_vy)
            v   = math.sqrt(ax*ax + ay*ay) / aw
            if v > peak:
                peak = v
        return peak

    def _approach(self, id1, id2):
        h1 = self.person_history.get(id1, [])
        h2 = self.person_history.get(id2, [])
        if len(h1) < 4 or len(h2) < 4:
            return 0.
        dn = math.sqrt((h1[-1]['cx']-h2[-1]['cx'])**2 +
                       (h1[-1]['cy']-h2[-1]['cy'])**2)
        dp = math.sqrt((h1[-3]['cx']-h2[-3]['cx'])**2 +
                       (h1[-3]['cy']-h2[-3]['cy'])**2)
        if dp <= 0:
            return 0.
        return max(0., min((dp - dn) / dp / 0.25, 1.))

    # ── Pair consistency ──────────────────────────────────────────────
    def _upd_pair_hist(self, a, b, sc):
        k = (min(a, b), max(a, b))
        if k not in self.pair_hist:
            self.pair_hist[k] = deque(maxlen=self.PAIR_HIST_LEN)
        self.pair_hist[k].append(sc)

    def _consistency(self, a, b):
        k = (min(a, b), max(a, b))
        h = self.pair_hist.get(k, [])
        if len(h) < 2:
            return 1.
        return sum(1 for s in h
                   if s > self.SCORE_THRESH * 0.4) / len(h)

    def _clean_pair_hist(self, ids):
        dead = [k for k in list(self.pair_hist.keys())
                if k[0] not in ids and k[1] not in ids]
        for k in dead:
            del self.pair_hist[k]

    # ── Scoring ───────────────────────────────────────────────────────
    def _score(self, p1, p2, id1, id2):
        b1   = (p1['x1'], p1['y1'], p1['x2'], p1['y2'])
        b2   = (p2['x1'], p2['y1'], p2['x2'], p2['y2'])
        dist = math.sqrt((p1['cx']-p2['cx'])**2 + (p1['cy']-p2['cy'])**2)
        aw   = (p1['w'] + p2['w']) / 2.
        md   = aw * self.MAX_DIST_K
        if dist > md:
            return 0., self._mk(gate='TOO_FAR')

        # ── Motion gates ─────────────────────────────────────────────
        m1 = self._motion(id1)
        m2 = self._motion(id2)
        mx_motion = max(m1, m2)
        mn_motion = min(m1, m2)

        if mx_motion < self.MIN_MOTION_SIG:
            return 0., self._mk(gate='STATIC_PAIR')
        if mn_motion < self.MIN_BOTH_MOTION:
            return 0., self._mk(gate='UNILATERAL_MOTION')

        # ★ FIX #1 — motion symmetry
        if mn_motion > 0:
            if mx_motion / mn_motion > self.MAX_MOTION_RATIO:
                return 0., self._mk(gate='ASYMMETRIC_MOTION')
        else:
            return 0., self._mk(gate='ASYMMETRIC_MOTION')

        far = (aw < 35)

        # ── Factors ──────────────────────────────────────────────────
        ov  = min(self._ovl(b1, b2) / 0.7, 1.)
        sd  = aw * 1.2
        dt  = aw * self.PROXIMITY_K
        if dist <= sd:
            pr = 1.
        elif dist < dt:
            pr = 1. - (dist - sd) / (dt - sd)
        elif far:
            pr = 0.12 * (1. - dist / md)
        else:
            pr = 0.

        a1  = self._agit(id1)
        a2  = self._agit(id2)
        mxa = max(a1, a2)
        mna = min(a1, a2)
        ag  = min(mxa / 0.7, 1.)

        if mxa < self.MIN_AGIT_GATE:
            return 0., self._mk(gate='LOW_AGITATION')
        if mna < self.MIN_BOTH_AGIT:
            return 0., self._mk(gate='UNILATERAL_AGIT')

        if mna > 0:
            if mxa / mna > self.MAX_AGIT_RATIO:
                return 0., self._mk(gate='ASYMMETRIC_AGIT')
        else:
            return 0., self._mk(gate='ASYMMETRIC_AGIT')

        ap = self._approach(id1, id2)
        if dist <= sd:
            ap = max(ap, 0.70)

        ve = min(self._vovl(b1, b2) / 0.6, 1.)

        sc = {
            'overlap'  : ov,
            'proximity': pr,
            'agitation': ag,
            'approach' : ap,
            'vertical' : ve
        }

        # ── Interaction gate ─────────────────────────────────────────
        ei    = self.MIN_INTERACT * (0.5 if far else 1.)
        inter = (self.W_OVL * ov + self.W_PRX * pr) / (self.W_OVL + self.W_PRX)
        if inter < ei:
            return 0., self._mk(sc, gate='LOW_INTERACTION')

        if ap < 0.15 and ov < 0.10 and pr < 0.40:
            return 0., self._mk(sc, gate='NO_ENGAGEMENT')

        # ── Factor count gate ─────────────────────────────────────────
        em = 2 if far else self.MIN_FACTORS
        ac = 0
        al = []
        for n, v in sc.items():
            thr = self.FACTOR_THR[n]
            if far and n in ('proximity', 'overlap', 'vertical'):
                thr *= 0.55
            if v >= thr:
                ac += 1
                al.append(n[:3].upper())
        if ac < em:
            return 0., self._mk(sc, gate=f'LOW_F({ac}/{em})',
                                 active=al, mr=em)

        # ── Final score ───────────────────────────────────────────────
        tot  = (self.W_OVL * ov + self.W_PRX * pr +
                self.W_AGI * ag + self.W_APP * ap + self.W_VER * ve)
        cons = self._consistency(id1, id2)
        if cons < self.MIN_CONSISTENCY:
            tot *= 0.6
        self._upd_pair_hist(id1, id2, tot)
        return tot, self._mk(sc, gate='PASSED', total=tot,
                              active=al, ac=ac, mr=em)

    def _mk(self, sc=None, gate='', total=0.,
             active=None, ac=0, mr=3):
        d = {
            'overlap': 0., 'proximity': 0., 'agitation': 0.,
            'approach': 0., 'vertical': 0.,
            'total': total, 'gate': gate,
            'active_factors': ac,
            'active_list': active or [],
            'min_req': mr
        }
        if sc:
            d.update(sc)
            d['total'] = total
        return d

    # ── Sustained check ───────────────────────────────────────────────
    def _sustained(self, sc):
        self.score_history.append(sc)
        if len(self.score_history) < 5:
            return False
        ratio = sum(1 for s in self.score_history
                    if s >= self.SCORE_THRESH) / len(self.score_history)
        return ratio >= self.MIN_SUS_RATIO

    # ────────────────────────────────────────────────────────────────
    # Lock management
    # ────────────────────────────────────────────────────────────────
    def _engage(self, id1, id2, p1, p2):
        self.locked_threat   = True
        self.locked_p1       = id1
        self.locked_p2       = id2
        self.miss_frames     = 0
        self.lock_t          = time.time()
        self.live_scores.clear()
        self.fight_cx    = (p1['cx'] + p2['cx']) // 2
        self.fight_cy    = (p1['cy'] + p2['cy']) // 2
        self.fight_avg_w = (p1['w']  + p2['w'])  // 2
        self.get_logger().warn(
            f"🔒 LOCKED P{id1}&P{id2}  key={self._pk(id1, id2)}")
        m = String()
        m.data = self._pk(id1, id2)
        self.pair_pub.publish(m)

    def _release(self, reason: str = ""):
        """Core release logic."""
        op1, op2 = self.locked_p1, self.locked_p2
        self.locked_threat    = False
        self.locked_p1        = None
        self.locked_p2        = None
        self.miss_frames      = 0
        self.lock_t           = None
        self.live_scores.clear()
        self.threat_cnt       = 0
        self.threat_active    = False
        self.last_p1          = None
        self.last_p2          = None
        self.last_score       = 0.
        self.score_history.clear()
        self.in_cooldown      = True
        self.cooldown_counter = 0
        self.get_logger().info(
            f"🔓 RELEASED P{op1}&P{op2} reason={reason}")

    def _release_lock(self, reason: str = ""):
        """
        """
        self._release(reason)

    # ── Recovery ──────────────────────────────────────────────────────
    def _recover(self, persons, missing_id, exclude=None):
        if not persons:
            return None
        if exclude is None:
            exclude = set()
        partner = (self.locked_p2 if missing_id == self.locked_p1
                   else self.locked_p1)
        radius  = self.fight_avg_w * self.REC_RADIUS_K
        best = None
        bs   = float('inf')
        for p in persons:
            if p['id'] == partner or p['id'] in exclude:
                continue
            d  = math.sqrt((p['cx'] - self.fight_cx)**2 +
                           (p['cy'] - self.fight_cy)**2)
            if d >= radius:
                continue
            mo = self._motion(p['id'])
            if mo < self.MIN_REC_MOTION:
                continue
            ag = self._agit(p['id'])
            sc = d - (mo * 80.) - (ag * 80.)
            if sc < bs:
                bs   = sc
                best = p['id']
        return best

    def _live_score(self, persons):
        if not self.locked_threat or self.locked_p1 is None:
            return None
        p1 = next((p for p in persons if p['id'] == self.locked_p1), None)
        p2 = next((p for p in persons if p['id'] == self.locked_p2), None)
        if not p1 or not p2:
            return None
        s, _ = self._score(p1, p2, self.locked_p1, self.locked_p2)
        return s

    # ── Stability ─────────────────────────────────────────────────────
    def _stable(self):
        return (self.locked_threat or
                self.smooth_cam_motion < self.CAM_MOTION_THR)

    def _started(self):
        return self.frame_cnt > self.STARTUP_FRAMES

    # ────────────────────────────────────────────────────────────────
    # Main image callback
    # ────────────────────────────────────────────────────────────────
    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return
        try:
            self._process_frame(img)
        except Exception as e:
            self.get_logger().error(f"Frame processing error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _process_frame(self, img):
        self._upd_fps()
        self.frame_cnt += 1
        ih, iw = img.shape[:2]
        self.img_w = iw
        self.img_h = ih
        icx = iw // 2
        icy = ih // 2
        self.cam_vx, self.cam_vy = self._cam_motion(img)

        # ── Scoring gate ─────────────────────────────────────────────
        can_score = self._started() and self._stable()
        if self.mission_state in self.SUPPRESS_STATES and not self.locked_threat:
            can_score = False
        if self.in_cooldown:
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.COOLDOWN_FRAMES:
                self.in_cooldown      = False
                self.cooldown_counter = 0
            elif not self.locked_threat:
                can_score = False

        # ── YOLO ─────────────────────────────────────────────────────
        enh = self._pre(img)
        res = self.model(enh, classes=[0],
                         conf=self.CONFIDENCE,
                         iou=self.YOLO_IOU,
                         imgsz=self.YOLO_IMGSZ,
                         verbose=False)
        persons = []
        cur     = []
        for r in res:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1+x2)//2, (y1+y2)//2
                w, h   = x2-x1, y2-y1
                cf     = float(box.conf[0])
                pid    = self._get_id(cx, cy, (x1, y1, x2, y2), cur)
                self.person_history[pid].append({
                    'cx': cx, 'cy': cy,
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'w': w, 'h': h
                })
                cur.append(pid)
                persons.append({
                    'cx': cx, 'cy': cy, 'w': w, 'id': pid,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'conf': cf
                })
                # box colour
                if (self.locked_threat and
                        pid in (self.locked_p1, self.locked_p2)):
                    col, th = (0, 0, 255), 3
                elif cf > 0.5:
                    col, th = (0, 255, 0), 2
                else:
                    col, th = (0, 200, 100), 1
                cv2.rectangle(img, (x1, y1), (x2, y2), col, th)
                cv2.putText(img, f"P{pid} {cf:.2f}",
                            (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, col, 1)

        # ── Tracking cleanup ─────────────────────────────────────────
        for pid in list(self.person_history.keys()):
            if pid not in cur:
                self.missing_frames[pid] = \
                    self.missing_frames.get(pid, 0) + 1
                if self.missing_frames[pid] > self.GRACE_PERIOD:
                    del self.person_history[pid]
                    self.missing_frames.pop(pid, None)
            else:
                self.missing_frames[pid] = 0
        self._clean_pair_hist(set(self.person_history.keys()))

        # ── Scoring loop ─────────────────────────────────────────────
        best = 0.
        bdet = None
        tcx  = icx
        tcy  = icy
        tp1  = tp2 = None

        if can_score and len(persons) >= self.MIN_PERSONS:
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    p1, p2 = persons[i], persons[j]
                    if p1['id'] == p2['id']:
                        continue
                    if self._id_bl(p1['id'], p2['id']):
                        continue
                    if self._ovl(
                            (p1['x1'],p1['y1'],p1['x2'],p1['y2']),
                            (p2['x1'],p2['y1'],p2['x2'],p2['y2'])) > 0.85:
                        continue
                    mpx = (p1['cx'] + p2['cx']) / 2.
                    mpy = (p1['cy'] + p2['cy']) / 2.
                    if self._pair_geo_bl(mpx, mpy):
                        continue
                    sc, det = self._score(p1, p2, p1['id'], p2['id'])
                    if sc > best:
                        best = sc
                        bdet = det
                        tp1  = p1
                        tp2  = p2
                        tcx  = int(mpx)
                        tcy  = int(mpy)

        sus = self._sustained(best)
        raw = best >= self.SCORE_THRESH and sus
        sf  = Float32()
        sf.data = float(best)
        self.score_pub.publish(sf)

        # ── Lock system ───────────────────────────────────────────────
        threat = False

        if self.locked_threat:
            for p in persons:
                if p['id'] == self.locked_p1:
                    self.last_p1 = p
                if p['id'] == self.locked_p2:
                    self.last_p2 = p

            f1   = self.locked_p1 in cur
            f2   = self.locked_p2 in cur
            age  = time.time() - self.lock_t if self.lock_t else 0.
            hold = age < self.MIN_LOCK_SEC

            close = False
            if f1 and f2 and self.last_p1 and self.last_p2:
                d   = math.sqrt(
                    (self.last_p1['cx']-self.last_p2['cx'])**2 +
                    (self.last_p1['cy']-self.last_p2['cy'])**2)
                aw  = (self.last_p1['w'] + self.last_p2['w']) / 2.
                close = d < aw * self.MAX_DIST_K * 1.8
                self.fight_cx = \
                    (self.last_p1['cx'] + self.last_p2['cx']) // 2
                self.fight_cy = \
                    (self.last_p1['cy'] + self.last_p2['cy']) // 2
                self.fight_avg_w = int(aw)

            # Path A — both visible & close
            if f1 and f2 and close:
                self.miss_frames = 0
                threat = True
                tcx = self.fight_cx
                tcy = self.fight_cy
                m   = String()
                m.data = self._pk(self.locked_p1, self.locked_p2)
                self.pair_pub.publish(m)
                lv = self._live_score(persons)
                if lv is not None:
                    self.live_scores.append(lv)
                if (self.mission_state in ("MISSION", "SERVOING") and
                        age > self.VAL_AFTER_SEC and
                        len(self.live_scores) >= self.VAL_FRAMES):
                    rec = list(self.live_scores)[-self.VAL_FRAMES:]
                    avg = sum(rec) / len(rec)
                    if avg < self.VAL_THRESH:
                        self._release(
                            f"score collapsed avg={avg:.2f}")
                        threat = False

            # Path B — one visible
            elif (f1 or f2) and \
                    self.miss_frames < self.LOCK_RELEASE_FR:
                self.miss_frames += 1
                threat = True
                if self.last_p1 and self.last_p2:
                    tcx = self.fight_cx
                    tcy = self.fight_cy
                if not f1:
                    ni = self._recover(persons, self.locked_p1)
                    if ni:
                        self.get_logger().info(
                            f"🔄 P{self.locked_p1}→P{ni}")
                        self.locked_p1 = ni
                if not f2:
                    ni = self._recover(persons, self.locked_p2)
                    if ni:
                        self.get_logger().info(
                            f"🔄 P{self.locked_p2}→P{ni}")
                        self.locked_p2 = ni

            # Path C — both lost, forced hold
            elif hold:
                self.miss_frames += 1
                threat = True
                if self.last_p1 and self.last_p2:
                    tcx = self.fight_cx
                    tcy = self.fight_cy
                if persons and self.miss_frames % 10 == 0:
                    c1 = self._recover(persons, self.locked_p1,
                                       exclude=set())
                    c2 = self._recover(persons, self.locked_p2,
                                       exclude={c1} if c1 is not None
                                       else set())
                    if (c1 is not None and
                            c2 is not None and c1 != c2):
                        self.get_logger().info(
                            f"🔄 Dual P{self.locked_p1}→P{c1},"
                            f" P{self.locked_p2}→P{c2}")
                        self.locked_p1 = c1
                        self.locked_p2 = c2

            # Path D — release
            else:
                self._release("pair lost or separated")

        else:
            # No active lock
            if raw:
                self.threat_cnt = min(
                    self.threat_cnt + 1,
                    self.CONFIRM_FRAMES + self.NO_THREAT_FRAMES)
                self.last_p1    = tp1
                self.last_p2    = tp2
                self.last_score = best
            else:
                self.threat_cnt = max(self.threat_cnt - 2, 0)
            threat = self.threat_cnt >= self.CONFIRM_FRAMES
            if threat and tp1 and tp2:
                if not self._id_bl(tp1['id'], tp2['id']):
                    self._engage(tp1['id'], tp2['id'], tp1, tp2)
                else:
                    threat = False
                    self.threat_cnt = 0

        # ── Draw ─────────────────────────────────────────────────────
        if threat and self.last_p1 and self.last_p2:
            p1o = self.last_p1
            p2o = self.last_p2
            aw  = (p1o['w'] + p2o['w']) / 2.
            cv2.line(img,
                     (p1o['cx'], p1o['cy']),
                     (p2o['cx'], p2o['cy']),
                     (0, 165, 255), 3)
            cv2.circle(img, (tcx, tcy),
                       int(aw * self.PROXIMITY_K),
                       (0, 0, 255), 2)
            cv2.rectangle(img,
                          (p1o['x1'], p1o['y1']),
                          (p1o['x2'], p1o['y2']),
                          (0, 0, 255), 3)
            cv2.rectangle(img,
                          (p2o['x1'], p2o['y1']),
                          (p2o['x2'], p2o['y2']),
                          (0, 0, 255), 3)

        # ── Score bar ─────────────────────────────────────────────────
        bx = iw - 30
        bh = int(best * 150)
        bc = (0, 0, 255) if best >= self.SCORE_THRESH else (0, 255, 255)
        cv2.rectangle(img, (bx, 10),     (bx+20, 160), (50,50,50), -1)
        cv2.rectangle(img, (bx, 160-bh), (bx+20, 160), bc,         -1)
        cv2.putText(img, f"{best:.2f}",
                    (bx-10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bc, 1)

        # ── Factor overlay ────────────────────────────────────────────
        if bdet:
            y0 = ih - 120
            g  = bdet.get('gate', '')
            al = bdet.get('active_list', [])
            af = bdet.get('active_factors', 0)
            mr = bdet.get('min_req', self.MIN_FACTORS)
            for k in ['overlap','proximity','agitation',
                      'approach','vertical']:
                v  = bdet.get(k, 0.)
                t  = self.FACTOR_THR[k]
                ia = k[:3].upper() in al
                fc = (0, 255, 0) if ia else (100, 100, 100)
                mk = "*" if ia else " "
                cv2.putText(img,
                            f"{mk}{k[:3].upper()}:{v:.2f}(>{t:.2f})",
                            (iw-185, y0), cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, fc, 1)
                y0 += 14
            gc = (0,255,0) if g == 'PASSED' else (0,165,255)
            cv2.putText(img, f"GATE:{g}",
                        (iw-185, y0), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, gc, 1)
            y0 += 14
            cv2.putText(img, f"F:{af}/{mr}",
                        (iw-185, y0), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (200,200,200), 1)

        # ── Logging ───────────────────────────────────────────────────
        if threat and not self.threat_active:
            self.threat_active = True
            self.log_cnt       = 0
            self.get_logger().warn("=" * 55)
            self.get_logger().warn("🚨 THREAT CONFIRMED!")
            if self.locked_threat:
                self.get_logger().warn(
                    f"   Pair:P{self.locked_p1}&P{self.locked_p2}"
                    f"  Score:{self.last_score:.3f}")
            self.get_logger().warn("=" * 55)
        elif threat and self.threat_active:
            self.log_cnt += 1
            if self.log_cnt >= self.LOG_INTERVAL:
                self.log_cnt = 0
                la = time.time() - self.lock_t if self.lock_t else 0
                self.get_logger().warn(
                    f"🚨 ACTIVE | Score:{self.last_score:.3f} | "
                    f"Pair:{self._pk(self.locked_p1 or 0, self.locked_p2 or 0)}"
                    f" | {la:.1f}s")
        elif not threat and self.threat_active:
            self.threat_active    = False
            self.log_cnt          = 0
            self.last_p1          = None
            self.last_p2          = None
            self.last_score       = 0.
            self.in_cooldown      = True
            self.cooldown_counter = 0
            self.score_history.clear()

        # ── HUD ───────────────────────────────────────────────────────
        cl = (0, 0, 255) if threat else (0, 255, 255)
        if not can_score and not self.locked_threat:
            if not self._started():
                st = f"STARTUP ({self.STARTUP_FRAMES - self.frame_cnt})"
                cl = (128, 128, 128)
            elif self.mission_state in self.SUPPRESS_STATES:
                st = f"WAIT [{self.mission_state}]"
                cl = (255, 200, 0)
            elif self.in_cooldown:
                st = (f"COOLDOWN "
                      f"({self.COOLDOWN_FRAMES - self.cooldown_counter})")
                cl = (0, 200, 200)
            else:
                st = f"CAM UNSTABLE ({self.smooth_cam_motion:.1f})"
                cl = (0, 165, 255)
        elif threat:
            if self.locked_threat:
                la = time.time() - self.lock_t if self.lock_t else 0
                st = (f"LOCKED P{self.locked_p1}"
                      f"&P{self.locked_p2} ({la:.1f}s)")
            else:
                st = "THREAT CONFIRMED!"
        else:
            st = (f"Monitoring ({self.threat_cnt}/{self.CONFIRM_FRAMES})"
                  f" GeoBL={len(self.geo_bl)} [{self.mission_state}]")

        cv2.putText(img, st,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cl, 2)
        cv2.putText(img, f"Persons:{len(persons)}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,0), 2)
        cv2.putText(img, f"Score:{best:.3f}",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cl, 1)
        cv2.putText(img, f"FPS:{self.fps:.1f}",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (150,150,150), 1)
        if self.drone_ready:
            cv2.putText(img,
                        f"Drone:({self.drone_x:.1f},{self.drone_y:.1f})",
                        (10, 112), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (200,200,200), 1)
        if self.live_scores:
            rec = list(self.live_scores)[-self.VAL_FRAMES:]
            avg = sum(rec) / len(rec)
            vc  = (0,0,255) if avg < self.VAL_THRESH else (0,255,0)
            cv2.putText(img, f"ValAvg:{avg:.2f}",
                        (10, 128), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, vc, 1)

        # ── Publish ───────────────────────────────────────────────────
        bt = Bool()
        bt.data = threat
        self.threat_pub.publish(bt)
        if threat:
            pm = PoseStamped()
            pm.header.stamp    = self.get_clock().now().to_msg()
            pm.header.frame_id = "camera_link"
            pm.pose.position.x = float(tcx - icx)
            pm.pose.position.y = float(tcy - icy)
            pm.pose.position.z = 0.
            self.target_pub.publish(pm)
        cv2.drawMarker(img, (icx, icy),
                       (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
        self.debug_pub.publish(
            self.bridge.cv2_to_imgmsg(img, "bgr8"))


def main(args=None):
    rclpy.init(args=args)
    n = ThreatDetector()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()