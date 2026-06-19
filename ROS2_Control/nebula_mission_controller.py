import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from px4_msgs.msg import (VehicleCommand, TrajectorySetpoint,
                           VehicleLocalPosition, VehicleStatus,
                           OffboardControlMode)
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped
import math


class DS:
    MISSION  = "MISSION"
    SERVOING = "SERVOING"
    HOVERING = "HOVERING"
    RETURNING= "RETURNING"
    COOLDOWN = "COOLDOWN"


class MissionManager(Node):

    # ── Tunable ───────────────────────────────────────────────────────
    HOVER_SEC          = 5.0
    CENTRED_PX         = 40
    CENTRED_FRAMES     = 15
    TARGET_ALT_NED     = -4.0
    ALT_TOL            = 0.3
    MAX_SPEED          = 0.4
    KP_XY              = 0.005
    KP_Z               = 0.5
    WARMUP_TICKS       = 20
    THREAT_CNF         = 5
    NO_THREAT_CNF      = 50
    RESUME_WD_SEC      = 3.0
    MAX_SERVO_SEC      = 15.0
    STALE_TGT_SEC      = 3.0
    POST_HOVER_CD_SEC  = 4.0
    STATE_PUB_HZ       = 5
    SERVO_LOST_THRESH  = 20
    MIN_THREAT_FOR_HOVER = 8

    def __init__(self):
        super().__init__('nebula_mission_manager')
        qs = qos_profile_sensor_data
        qp = qos_profile_system_default

        self.cmd_pub   = self.create_publisher(
            VehicleCommand,      '/fmu/in/vehicle_command',       qp)
        self.offb_pub  = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qp)
        self.traj_pub  = self.create_publisher(
            TrajectorySetpoint,  '/fmu/in/trajectory_setpoint',   qp)
        self.bl_pub    = self.create_publisher(
            String,              '/nebula/blacklist_pair',        10)
        self.state_pub = self.create_publisher(
            String,              '/nebula/mission_state',         10)

        self.create_subscription(
            Bool,                '/nebula/threat_status',
            self.thr_cb,  10)
        self.create_subscription(
            PoseStamped,         '/nebula/target_pose',
            self.tgt_cb,  10)
        self.create_subscription(
            VehicleLocalPosition,'/fmu/out/vehicle_local_position_v1',
            self.pos_cb,  qs)
        self.create_subscription(
            VehicleStatus,       '/fmu/out/vehicle_status_v2',
            self.stat_cb, qs)
        self.create_subscription(
            String,              '/nebula/locked_pair_id',
            self.pair_cb, 10)

        # ── State ─────────────────────────────────────────────────────
        self.state     = DS.MISSION
        self.nav_state = 0

        # ── Position ──────────────────────────────────────────────────
        self.px = self.py = self.pz = 0.
        self.yaw      = 0.
        self.lock_yaw = 0.
        self.local_rdy = False

        # ── Hover ─────────────────────────────────────────────────────
        self.hover_x = self.hover_y = self.hover_z = 0.

        # ── Threat debounce ───────────────────────────────────────────
        self.raw_thr    = False
        self.conf_thr   = False
        self.thr_cnt    = 0
        self.no_thr_cnt = 0

        # ── Servo ─────────────────────────────────────────────────────
        self.err_x         = 0.
        self.err_y         = 0.
        self.last_tgt_t    = None
        self.centred_cnt   = 0
        self.offb_tick     = 0
        self.servo_start_t = None
        # ★ v7.0
        self.servo_lost_cnt      = 0
        self.threat_seen_in_servo = 0

        # ── Hover ─────────────────────────────────────────────────────
        self.hover_start_t = None
        self.hover_rdy     = False

        # ── Return / Cooldown ─────────────────────────────────────────
        self.resume_cmd_t = None
        self.cd_start_t   = None

        # ── Pair tracking ─────────────────────────────────────────────
        self.pair_id: str | None = None

        self.timer      = self.create_timer(0.1,  self.tick)
        self.spub_timer = self.create_timer(
            1. / self.STATE_PUB_HZ, self._pub_state)

        self.get_logger().info("=" * 55)
        self.get_logger().info("NEBULA Mission Manager v7.0")
        self.get_logger().info("=" * 55)

    # ── Subscribers ───────────────────────────────────────────────────
    def thr_cb(self, m):
        self.raw_thr = m.data

    def tgt_cb(self, m):
        self.err_x      = m.pose.position.x
        self.err_y      = m.pose.position.y
        self.last_tgt_t = self.get_clock().now()

    def pos_cb(self, m):
        self.px = m.x; self.py = m.y; self.pz = m.z
        self.yaw = m.heading
        self.local_rdy = True

    def stat_cb(self, m):
        self.nav_state = m.nav_state

    def pair_cb(self, m):
        if m.data:
            self.pair_id = m.data

    def _pub_state(self):
        m = String()
        m.data = self.state
        self.state_pub.publish(m)

    # ── Main loop ─────────────────────────────────────────────────────
    def tick(self):
        if not self.local_rdy:
            return
        try:
            if self.state in (DS.SERVOING, DS.HOVERING):
                self._pub_offb()
            self._debounce()
            if   self.state == DS.SERVOING:  self._servo()
            elif self.state == DS.HOVERING:  self._hover()
            elif self.state == DS.RETURNING: self._ret()
            elif self.state == DS.COOLDOWN:  self._cd()
        except Exception as e:
            self.get_logger().error(f"tick error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    # ── Debounce (MISSION only) ───────────────────────────────────────
    def _debounce(self):
        if self.state != DS.MISSION:
            return
        if self.raw_thr:
            self.thr_cnt    += 1
            self.no_thr_cnt  = 0
            if self.thr_cnt >= self.THREAT_CNF and not self.conf_thr:
                self.conf_thr = True
                self.thr_cnt  = 0
                self._enter_servo()
        else:
            self.no_thr_cnt += 1
            self.thr_cnt     = 0
            if self.no_thr_cnt >= self.NO_THREAT_CNF:
                self.conf_thr   = False
                self.no_thr_cnt = 0

    # ── SERVOING ──────────────────────────────────────────────────────
    def _enter_servo(self):
        self.get_logger().warn("⚠️  THREAT → SERVOING")
        self.lock_yaw              = self.yaw
        self.centred_cnt           = 0
        self.offb_tick             = 0
        self.servo_start_t         = self.get_clock().now()
        self.servo_lost_cnt        = 0    # ★ v7.0
        self.threat_seen_in_servo  = 0    # ★ v7.0
        self.state                 = DS.SERVOING
        self._pub_offb()

    def _abort_servo(self, reason: str):
        """
        """
        self.get_logger().warn(f"⚠️  SERVO ABORT ({reason}) → RETURNING")
        self.pair_id      = None          # don't blacklist false positive
        self.state        = DS.RETURNING
        self.resume_cmd_t = self.get_clock().now()
        self._mode(4, 4)

    def _servo(self):
        # ── Threat tracking ──────────────────────────────────────────
        if self.raw_thr:
            self.servo_lost_cnt       = 0
            self.threat_seen_in_servo += 1   # ★ count confirmed frames
        else:
            self.servo_lost_cnt += 1
            if self.servo_lost_cnt > self.SERVO_LOST_THRESH:
                self._abort_servo(
                    f"threat gone {self.SERVO_LOST_THRESH/10:.0f}s")
                return

        # ── Warmup ───────────────────────────────────────────────────
        if self.offb_tick < self.WARMUP_TICKS:
            self._pub_offb()
            self._vel(0., 0., 0.)
            self.offb_tick += 1
            if self.offb_tick == self.WARMUP_TICKS // 2:
                self._mode(6)
            return

        # ── Hard cap ─────────────────────────────────────────────────
        if self.servo_start_t:
            el = (self.get_clock().now() -
                  self.servo_start_t).nanoseconds / 1e9
            if el > self.MAX_SERVO_SEC:
                # ★ v7.0 — only hover if threat was confirmed enough
                if self.threat_seen_in_servo >= self.MIN_THREAT_FOR_HOVER:
                    self.get_logger().warn("⚠️  Servo cap → forcing hover")
                    self._enter_hover()
                else:
                    self._abort_servo(
                        f"cap reached, threat_seen={self.threat_seen_in_servo}")
                return

        # ── Stale target check ────────────────────────────────────────
        stale = (self.last_tgt_t is None or
                 (self.get_clock().now() -
                  self.last_tgt_t).nanoseconds / 1e9 > 0.5)
        if stale and self.last_tgt_t:
            sd = (self.get_clock().now() -
                  self.last_tgt_t).nanoseconds / 1e9
            if sd > self.STALE_TGT_SEC:
                self.get_logger().warn(
                    f"⚠️  Target stale {sd:.1f}s → hover")
                self._enter_hover()
                return

        # ── Velocity control ──────────────────────────────────────────
        vxb = -self.KP_XY * self.err_y if not stale else 0.
        vyb =  self.KP_XY * self.err_x if not stale else 0.
        sp  = math.sqrt(vxb*vxb + vyb*vyb)
        if sp > self.MAX_SPEED:
            vxb = vxb / sp * self.MAX_SPEED
            vyb = vyb / sp * self.MAX_SPEED
        cy  = math.cos(self.lock_yaw)
        sy  = math.sin(self.lock_yaw)
        vn  = vxb * cy - vyb * sy
        ve  = vxb * sy + vyb * cy
        vz  = max(min(self.KP_Z * (self.TARGET_ALT_NED - self.pz),
                      0.5), -0.5)
        self._vel(vn, ve, vz)
        self.get_logger().info(
            f"🎥 SERVOING | err=({self.err_x:.0f},{self.err_y:.0f})"
            f" | V=({vn:.2f},{ve:.2f},{vz:.2f})"
            f" | seen={self.threat_seen_in_servo}",
            throttle_duration_sec=0.5)

        # ── Centring check ────────────────────────────────────────────
        em = math.sqrt(self.err_x**2 + self.err_y**2)
        if em < self.CENTRED_PX and not stale:
            self.centred_cnt += 1
        else:
            self.centred_cnt = 0

        if self.centred_cnt >= self.CENTRED_FRAMES:
            # ★ v7.0 — only hover if threat was confirmed enough
            if self.threat_seen_in_servo >= self.MIN_THREAT_FOR_HOVER:
                self.get_logger().info(
                    f"✅ Centred + threat_seen={self.threat_seen_in_servo}"
                    f" → HOVERING")
                self._enter_hover()
            else:
                self._abort_servo(
                    f"centred but threat_seen={self.threat_seen_in_servo}"
                    f" < {self.MIN_THREAT_FOR_HOVER}")

    # ── HOVERING ──────────────────────────────────────────────────────
    def _enter_hover(self):
        self.get_logger().warn(f"🔴 HOVERING {self.HOVER_SEC:.0f}s")
        self.hover_x     = self.px
        self.hover_y     = self.py
        self.hover_z     = self.TARGET_ALT_NED
        self.hover_start_t = None
        self.hover_rdy   = False
        self.state       = DS.HOVERING

    def _hover(self):
        self._pos(self.hover_x, self.hover_y,
                  self.hover_z, self.lock_yaw)
        ae = abs(self.pz - self.hover_z)
        if not self.hover_rdy and ae < self.ALT_TOL:
            self.hover_rdy     = True
            self.hover_start_t = self.get_clock().now()
            self.get_logger().info("⏱️  Alt confirmed — hover timer start")
        if not self.hover_rdy:
            self.get_logger().info(
                f"⏳ Alt lock err={ae:.2f}m",
                throttle_duration_sec=1.)
            return
        el = (self.get_clock().now() -
              self.hover_start_t).nanoseconds / 1e9
        self.get_logger().info(
            f"⏱️  HOVERING {self.HOVER_SEC - el:.1f}s remaining",
            throttle_duration_sec=1.)
        if el >= self.HOVER_SEC:
            self._finish_hover()

    def _finish_hover(self):
        self.get_logger().info("✅ Hover done → blacklist → RETURNING")
        if self.pair_id:
            m      = String()
            m.data = self.pair_id
            self.bl_pub.publish(m)
            self.get_logger().warn(
                f"🚫 Blacklisted {self.pair_id}"
                f" @ ({self.px:.1f},{self.py:.1f})")
            self.pair_id = None
        else:
            self.get_logger().warn("⚠️  No pair_id to blacklist")
        self.state        = DS.RETURNING
        self.resume_cmd_t = self.get_clock().now()
        self._mode(4, 4)
        self.get_logger().info("▶️  AUTO.MISSION sent")

    # ── RETURNING ─────────────────────────────────────────────────────
    def _ret(self):
        if self.nav_state in (3, 4):
            self.get_logger().info(
                "✅ PX4 AUTO.MISSION confirmed → COOLDOWN")
            self.state      = DS.COOLDOWN
            self.cd_start_t = self.get_clock().now()
            return
        if self.resume_cmd_t:
            el = (self.get_clock().now() -
                  self.resume_cmd_t).nanoseconds / 1e9
            if el > self.RESUME_WD_SEC:
                self.get_logger().warn("⚠️  Watchdog: re-send AUTO.MISSION")
                self._mode(4, 4)
                self.resume_cmd_t = self.get_clock().now()

    # ── COOLDOWN ──────────────────────────────────────────────────────
    def _cd(self):
        if self.cd_start_t is None:
            self.cd_start_t = self.get_clock().now()
        el = (self.get_clock().now() -
              self.cd_start_t).nanoseconds / 1e9
        if el >= self.POST_HOVER_CD_SEC:
            self.get_logger().info(
                "✅ Cooldown done → MISSION (ready for next fight)")
            self.state      = DS.MISSION
            self.conf_thr   = False
            self.thr_cnt    = 0
            self.no_thr_cnt = 0
            self.cd_start_t = None
        else:
            self.get_logger().info(
                f"⏸️  POST-HOVER CD {self.POST_HOVER_CD_SEC - el:.1f}s",
                throttle_duration_sec=1.)

    # ── PX4 helpers ───────────────────────────────────────────────────
    def _ts(self):
        return self.get_clock().now().nanoseconds // 1000

    def _pub_offb(self):
        m = OffboardControlMode()
        m.position     = True
        m.velocity     = True
        m.acceleration = m.attitude = m.body_rate = False
        m.timestamp    = self._ts()
        self.offb_pub.publish(m)

    def _vel(self, vx, vy, vz):
        m = TrajectorySetpoint()
        m.velocity = [float(vx), float(vy), float(vz)]
        m.position = [float('nan')] * 3
        m.yaw      = float(self.lock_yaw)
        m.timestamp= self._ts()
        self.traj_pub.publish(m)

    def _pos(self, x, y, z, yaw):
        m = TrajectorySetpoint()
        m.position = [float(x), float(y), float(z)]
        m.velocity = [float('nan')] * 3
        m.yaw      = float(yaw)
        m.timestamp= self._ts()
        self.traj_pub.publish(m)

    def _mode(self, cm, sm=0):
        m = VehicleCommand()
        m.command          = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        m.param1           = 1.
        m.param2           = float(cm)
        m.param3           = float(sm)
        m.target_system    = m.source_system    = 1
        m.target_component = m.source_component = 1
        m.from_external    = True
        m.timestamp        = self._ts()
        self.cmd_pub.publish(m)
        lbl = {(4, 4): "AUTO.MISSION",
               (6, 0): "OFFBOARD"}.get((cm, sm), f"mode({cm},{sm})")
        self.get_logger().warn(f"🔧 → {lbl}")


def main(args=None):
    rclpy.init(args=args)
    n = MissionManager()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()