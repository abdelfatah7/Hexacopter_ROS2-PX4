#!/usr/bin/env python3
#created by: Abdelfattah Ahmed
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data, qos_profile_system_default
from px4_msgs.msg import (
    VehicleCommand,
    TrajectorySetpoint,
    VehicleLocalPosition,
    VehicleGlobalPosition,
    VehicleStatus,
    OffboardControlMode
)
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
import math

class DroneState:
    MISSION   = "MISSION"
    SERVOING  = "SERVOING"    
    RETURNING = "RETURNING"

class MissionManager(Node):
    def __init__(self):
        super().__init__('nebula_mission_manager')

        # ==================== QoS Profiles ====================
        qos_sub = qos_profile_sensor_data
        qos_pub = qos_profile_system_default

        # ==================== Publishers ====================
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_pub)
        self.offboard_control_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_pub)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_pub)
        self.spotlight_pub = self.create_publisher(Bool, '/nebula/spotlight', 10)

        # ==================== Subscribers ====================
        self.create_subscription(Bool, '/nebula/threat_status', self.threat_callback, 10)
        self.create_subscription(PoseStamped, '/nebula/target_pose', self.target_callback, 10) # 👈 Offset الكاميرا
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.local_position_callback, qos_sub)
        self.create_subscription(VehicleGlobalPosition, '/fmu/out/vehicle_global_position', self.global_position_callback, qos_sub)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v2', self.status_callback, qos_sub)

        # ==================== State ====================
        self.state       = DroneState.MISSION
        self.nav_state   = 0
        self.local_ready = False

        # ==================== Debounce ====================
        self.raw_threat              = False
        self.confirmed_threat        = False
        self.threat_confirm_count    = 0
        self.no_threat_confirm_count = 0
        self.THREAT_CONFIRM_THRESHOLD    = 5
        self.NO_THREAT_CONFIRM_THRESHOLD = 50

        # ==================== Drone State ====================
        self.local_x = 0.0
        self.local_y = 0.0
        self.local_z = 0.0
        self.current_yaw = 0.0
        self.lock_yaw = 0.0 # 👈 زاوية الوش الثابتة وقت الهجوم

        # ==================== Visual Servoing Targets ====================
        self.pixel_err_x = 0.0
        self.pixel_err_y = 0.0
        self.last_target_time = None
        # ==================== State ====================

        # ==================== Parameters ====================
        self.MAX_SPEED       = 0.4  
        self.TARGET_ALTITUDE = -4.0  # 4 متر (NED: سالب = فوق)
        
        # Proportional Controllers (PID)
        self.KP_XY = 0.005  # تحويل البيكسل لسرعة م/ث
        self.KP_Z  = 0.5    # تحويل خطأ الارتفاع لسرعة م/ث

        self.offboard_counter = 0
        self.OFFBOARD_WARMUP  = 20

        self.spotlight_state   = False
        self.spotlight_counter = 0
        self.FLASH_INTERVAL    = 10

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("✅ Mission Manager: Visual Servoing (IBVS) Active!")

    # ============================================================
    #                     Callbacks
    # ============================================================

    def threat_callback(self, msg):
        self.raw_threat = msg.data

    def target_callback(self, msg):
        """استقبال مسافة البيكسل من مركز الصورة"""
        self.pixel_err_x = msg.pose.position.x
        self.pixel_err_y = msg.pose.position.y
        self.last_target_time = self.get_clock().now()

    def local_position_callback(self, msg):
        self.local_x = msg.x
        self.local_y = msg.y
        self.local_z = msg.z
        self.current_yaw = msg.heading
        self.local_ready = True

    def global_position_callback(self, msg):
        pass

    def status_callback(self, msg):
        self.nav_state = msg.nav_state

    # ============================================================
    #                     Debounce & Transitions
    # ============================================================

    def _update_debounce(self):
        if self.raw_threat:
            self.threat_confirm_count += 1
            self.no_threat_confirm_count = 0
            if self.threat_confirm_count >= self.THREAT_CONFIRM_THRESHOLD and not self.confirmed_threat:
                self.confirmed_threat = True
                self.threat_confirm_count = 0
                self._on_threat_confirmed()
        else:
            self.no_threat_confirm_count += 1
            self.threat_confirm_count = 0
            if self.no_threat_confirm_count >= self.NO_THREAT_CONFIRM_THRESHOLD and self.confirmed_threat:
                self.confirmed_threat = False
                self.no_threat_confirm_count = 0
                self._on_threat_cleared()

    def _on_threat_confirmed(self):
        self.get_logger().warn("⚠️ THREAT CONFIRMED! Locking visual target...")
        
        # 🌟 نثبت اتجاه الطيارة (Yaw) عشان متهتزش، وهتتحرك بالجنب ورا وقدام
        self.lock_yaw = self.current_yaw 
        
        if self.state == DroneState.MISSION:
            self.state = DroneState.SERVOING
            self.offboard_counter = 0
            self.activate_spotlight(True)

    def _on_threat_cleared(self):
        self.get_logger().info("✅ Threat CLEARED! Returning to Mission...")
        if self.state == DroneState.SERVOING:
            self.state = DroneState.RETURNING
            self.activate_spotlight(False)
            self._resume_mission()

    # ============================================================
    #                     Main Loop
    # ============================================================

    def timer_callback(self):
        if not self.local_ready:
            return

        self._update_debounce()

        if self.state == DroneState.SERVOING:
            self._handle_servoing()
            
            # Spotlight Flashing
            self.spotlight_counter += 1
            if self.spotlight_counter >= self.FLASH_INTERVAL:
                self.spotlight_counter = 0
                self.spotlight_state = not self.spotlight_state
                self.activate_spotlight(self.spotlight_state)

        elif self.state == DroneState.RETURNING:
            if self.nav_state == 4:
                self.state = DroneState.MISSION
                self.get_logger().info("✅ State -> MISSION")

    # ============================================================
    #                     Visual Servoing Logic
    # ============================================================

    def _handle_servoing(self):
        if self.offboard_counter < self.OFFBOARD_WARMUP:
            self._publish_offboard_mode()
            self._publish_hold_setpoint()
            self.offboard_counter += 1
            if self.offboard_counter == self.OFFBOARD_WARMUP // 2:
                self._set_offboard_mode()
            return

        # 1. فحص تحديث الكاميرا (Hover لو الهدف اختفى لحظياً)
        if self.last_target_time is None or (self.get_clock().now() - self.last_target_time).nanoseconds / 1e9 > 0.5:
            vx_body = 0.0
            vy_body = 0.0
        else:
            # 2. حساب السرعة بناءً على البيكسل
            # offset_y موجب (الخناقة تحت في الصورة) -> الطيارة لازم ترجع لورا (سالب)
            # offset_x موجب (الخناقة يمين في الصورة) -> الطيارة لازم تروح يمين (موجب)
            vx_body = -self.KP_XY * self.pixel_err_y
            vy_body =  self.KP_XY * self.pixel_err_x

        # 3. تحجيم السرعة القصوى لـ 0.4 م/ث
        speed = math.sqrt(vx_body**2 + vy_body**2)
        if speed > self.MAX_SPEED:
            vx_body = (vx_body / speed) * self.MAX_SPEED
            vy_body = (vy_body / speed) * self.MAX_SPEED

        # 4. تحويل السرعة من إطار الطيارة (Body) لإطار العالم (NED) عشان PX4 يفهمها
        vn = vx_body * math.cos(self.lock_yaw) - vy_body * math.sin(self.lock_yaw)
        ve = vx_body * math.sin(self.lock_yaw) + vy_body * math.cos(self.lock_yaw)

        # 5. التحكم في الارتفاع لـ 4 متر بـ P-Controller
        err_z = self.TARGET_ALTITUDE - self.local_z
        vz = self.KP_Z * err_z
        vz = max(min(vz, 0.5), -0.5)  # حد أقصى للنزول/الطلوع 0.5 م/ث

        # 6. إرسال الأوامر
        self._publish_offboard_mode()
        self._publish_velocity_setpoint(vn, ve, vz, self.lock_yaw)

        self.get_logger().info(
            f"🎥 SERVOING | offset:({self.pixel_err_x:.0f},{self.pixel_err_y:.0f}) | V:({vn:.2f},{ve:.2f},{vz:.2f})",
            throttle_duration_sec=0.5
        )

    # ============================================================
    #                     PX4 Commands
    # ============================================================

    def _publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position, msg.velocity, msg.acceleration, msg.attitude, msg.body_rate = False, True, False, False, False
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.offboard_control_pub.publish(msg)

    def _publish_velocity_setpoint(self, vx, vy, vz, yaw):
        msg = TrajectorySetpoint()
        msg.velocity = [float(vx), float(vy), float(vz)]
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.yaw = float(yaw)
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.trajectory_pub.publish(msg)

    def _publish_hold_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position = [self.local_x, self.local_y, self.local_z]
        msg.velocity = [0.0, 0.0, 0.0]
        msg.yaw = float(self.current_yaw)
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.trajectory_pub.publish(msg)

    def _set_offboard_mode(self):
        msg = VehicleCommand()
        msg.command, msg.param1, msg.param2 = VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0
        msg.target_system = msg.target_component = msg.source_system = msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.vehicle_command_pub.publish(msg)
        self.get_logger().warn("🔧 → OFFBOARD mode")

    def _resume_mission(self):
        msg = VehicleCommand()
        msg.command, msg.param1, msg.param2 = VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0
        msg.target_system = msg.target_component = msg.source_system = msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.vehicle_command_pub.publish(msg)

    def activate_spotlight(self, state: bool):
        msg = Bool()
        msg.data = state
        self.spotlight_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()