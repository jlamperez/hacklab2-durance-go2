"""Follow node: where the target is in, sport-mode requests out. Move, stop, look.

  subscribes  /reflex/target        reflex_msgs/Target   (target node, or the SNN's adapter), input:=target
              /reflex/decision      reflex_msgs/ReflexDecision (fpga_perception, perception), input:=decision
              /durance/ball_zone    std_msgs/String      (the Kria's ball_node: the FPGA zones network),
                                                         input:=zone. Default of ros-look
  publishes   /api/sport/request    unitree_api/Request  (BalanceStand, Move, StopMove, Pose, Euler, Damp)
              /reflex/debug/follow  std_msgs/String, JSON: state, last target, last command

    pixi run -e ros ros-follow                 # walk mode, keyboard in this terminal
    pixi run -e ros ros-follow mode:=gaze      # trunk only (Euler yaw), no walking
    pixi run -e ros ros-look                   # = mode:=look input:=zone: nine gestures, Euler yaw + pitch,
                                               #   from the FPGA's ball zone on /durance/ball_zone
    pixi run -e ros ros-look input:=target     # look mode from /reflex/target instead
    pixi run -e ros ros-look input:=decision   # look mode from /reflex/decision (left / center / right,
                                               #   e.g. fpga_perception: zones3 on the FPGA)

input:=zone. ball_node publishes upper_left ... lower_right or none, only when the zone
changes (transient-local). The node keeps the latest and feeds it to the controller at
zone_rate_hz as the zone's centre (reflex/runtime/zone_target.py) with confidence
zone_confidence, because the controller counts consecutive sightings. If ball_node leaves
the graph, the zone counts as "none" (nothing seen) rather than being repeated.

Starts DISABLED. Keys:  f start following   SPACE stop (StopMove, disabled)
                        ! Damp (motors limp, the robot drops: emergency only)   q quit
look mode only, a gesture by hand (numpad layout), following or not:
                        7 left_up    8 up      9 right_up
                        4 left       5 center  6 right
                        1 left_down  2 down    3 right_down
Without a terminal (launch files), `autostart:=true` starts it, and
`ros2 service call /reflex/follow/enable std_srvs/srv/SetBool "{data: false}"` stops it.

All the logic and the safety limits are in reflex/runtime/follow.py (FollowController).
This node adds: a 50 Hz tick so bursts end on time even if targets stop arriving, and
StopMove on shutdown. Keep a person on the Unitree remote anyway.
"""
import json
import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import SetBool

from reflex.runtime import sport_api as api
from reflex.runtime.follow import FollowConfig, FollowController
from reflex.runtime.target import Target
from reflex.runtime.zone_target import target_from_decision, target_from_zone
try:
    from reflex_msgs.msg import ReflexDecision
    from reflex_msgs.msg import Target as TargetMsg
except ImportError:      # input:=zone needs no reflex_msgs: runs on a board without our build
    ReflexDecision = TargetMsg = None
from unitree_api.msg import Request

# Hard limits: whatever the parameters say, the node never commands more than this.
HARD_MAX_VX = 0.5     # m/s
HARD_MAX_WZ = 1.0     # rad/s
HARD_MAX_YAW = 0.5    # rad
HARD_MAX_PITCH = 0.35 # rad

# look mode: numpad layout -> gesture
KEY_GESTURES = {"7": "left_up", "8": "up", "9": "right_up", "4": "left", "5": "center", "6": "right",
                "1": "left_down", "2": "down", "3": "right_down"}


class FollowNode(Node):
    def __init__(self):
        super().__init__("reflex_follow")
        p = self.declare_parameter
        d = FollowConfig()
        cfg = FollowConfig(
            mode=p("mode", d.mode).value,
            min_confidence=float(p("min_confidence", d.min_confidence).value),
            consecutive=int(p("consecutive", d.consecutive).value),
            hfov_deg=float(p("hfov_deg", d.hfov_deg).value),
            move_s=float(p("move_s", d.move_s).value),
            settle_s=float(p("settle_s", d.settle_s).value),
            turn_gain=float(p("turn_gain", d.turn_gain).value),
            max_wz=min(float(p("max_wz", d.max_wz).value), HARD_MAX_WZ),
            max_vx=min(float(p("max_vx", d.max_vx).value), HARD_MAX_VX),
            walk_cone_deg=float(p("walk_cone_deg", d.walk_cone_deg).value),
            v_near=float(p("v_near", d.v_near).value),
            v_slow=float(p("v_slow", d.v_slow).value),
            max_travel_m=float(p("max_travel_m", d.max_travel_m).value),
            max_yaw=min(float(p("max_yaw", d.max_yaw).value), HARD_MAX_YAW),
            lost_s=float(p("lost_s", d.lost_s).value),
            look_yaw=min(float(p("look_yaw", d.look_yaw).value), HARD_MAX_YAW),
            look_pitch=min(float(p("look_pitch", d.look_pitch).value), HARD_MAX_PITCH),
            vfov_deg=float(p("vfov_deg", d.vfov_deg).value),
            v_level=float(p("v_level", d.v_level).value),
            look_h_deg=float(p("look_h_deg", d.look_h_deg).value),
            look_v_deg=float(p("look_v_deg", d.look_v_deg).value),
            look_hyst_deg=float(p("look_hyst_deg", d.look_hyst_deg).value),
            camera_on_trunk=bool(p("camera_on_trunk", d.camera_on_trunk).value))
        # The sign of Euler pitch (y) on the real Go2 is not verified: if "up" (key 8) tilts the
        # nose down, set invert_pitch:=true.
        self.pitch_sign = -1.0 if bool(p("invert_pitch", False).value) else 1.0
        self.dry_run = bool(p("dry_run", False).value)      # log the requests, publish nothing
        self.ctrl = FollowController(cfg)
        self.pub = self.create_publisher(Request, api.SPORT_REQUEST_TOPIC, 10)
        self.debug_pub = self.create_publisher(String, "/reflex/debug/follow", 10)
        self.input = p("input", "target").value
        if self.input == "zone":
            self.zone_topic = p("zone_topic", "/durance/ball_zone").value
            self.zone_confidence = float(p("zone_confidence", 1.0).value)   # ball_node sends no probability
            self._zone = None
            self._zone_gone_said = False
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)   # as ball_node publishes
            self.create_subscription(String, self.zone_topic, self.on_zone, qos)
            self.create_timer(1.0 / float(p("zone_rate_hz", 20.0).value), self.on_zone_tick)
            self.source = self.zone_topic
        elif self.input == "target":
            if TargetMsg is None:
                raise RuntimeError("input:=target needs reflex_msgs (build ros2_ws); input:=zone does not")
            self.create_subscription(TargetMsg, "/reflex/target", self.on_target, 10)
            self.source = "/reflex/target"
        elif self.input == "decision":
            if ReflexDecision is None:
                raise RuntimeError("input:=decision needs reflex_msgs (build ros2_ws)")
            self.decision_v = float(p("decision_v", 0.0).value)     # image row for the decisions: level
            self.create_subscription(ReflexDecision, "/reflex/decision", self.on_decision, 10)
            self.source = "/reflex/decision"
        else:
            raise ValueError(f"input:={self.input}: use target, zone or decision")
        self.get_logger().info(f"targets from {self.source}")
        self.create_service(SetBool, "/reflex/follow/enable", self.on_enable)
        self.create_timer(0.02, self.on_tick)
        self._stamp_us = 0
        self._logged_burst = None
        # Posture mode as last commanded. On the real Go2 Euler acts only after Pose(true) and
        # Move does not walk until Pose(false) (sport_api.POSE, reflex/runtime/sim_sport.py).
        self._posture = False
        self._last_target_t = None
        self._silent_warned = False
        if cfg.mode == "look":
            self.get_logger().info(f"mode look: yaw {cfg.look_yaw:.2f} rad, pitch {cfg.look_pitch:.2f} rad"
                                   f"{' (inverted)' if self.pitch_sign < 0 else ''}, camera "
                                   f"{'on the trunk' if cfg.camera_on_trunk else 'fixed'}"
                                   + ("  [DRY RUN: nothing is published]" if self.dry_run else ""))
        else:
            self.get_logger().info(f"mode {cfg.mode}: max {cfg.max_vx:.2f} m/s, {cfg.max_wz:.2f} rad/s, "
                                   f"bursts of {cfg.move_s:.1f} s, travel budget {cfg.max_travel_m:.1f} m"
                                   + ("  [DRY RUN: nothing is published]" if self.dry_run else ""))
        if bool(p("autostart", False).value):
            self.start()
        else:
            self.say("DISABLED. f = start following, SPACE = stop, ! = Damp, q = quit"
                     + (", 1-9 = gesture by hand" if cfg.mode == "look" else ""))

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def say(self, text: str) -> None:
        print(f"\r[follow] {text}\r", flush=True)

    # --- switches ------------------------------------------------------------------------
    def start(self) -> None:
        self.send_all(self.ctrl.enable(self.now()))
        self.say(f"FOLLOWING ({self.ctrl.cfg.mode}). SPACE to stop")

    def stop(self, reason: str = "stopped") -> None:
        self.send_all(self.ctrl.disable(self.now(), reason))
        self.say(f"STOPPED ({reason}). f to start again")

    def gesture(self, name: str) -> None:
        self.send_all(self.ctrl.look(name, self.now()))
        self.say(f"LOOK {name}" + (" (following goes on from here)" if self.ctrl.enabled else ""))

    def damp(self) -> None:
        self.ctrl.disable(self.now(), "damp")
        self.publish(api.DAMP)
        self._posture = False
        self.say("DAMP sent: motors limp")

    def on_enable(self, request, response):
        if request.data:
            self.start()
        else:
            self.stop("service call")
        response.success, response.message = True, self.ctrl.state
        return response

    # --- input ---------------------------------------------------------------------------
    def on_target(self, msg: TargetMsg) -> None:
        self._stamp_us = msg.header.stamp.sec * 1_000_000 + msg.header.stamp.nanosec // 1000
        self.feed(Target(float(msg.u), float(msg.v), float(msg.confidence), self._stamp_us))

    def on_decision(self, msg) -> None:
        self._stamp_us = msg.header.stamp.sec * 1_000_000 + msg.header.stamp.nanosec // 1000
        try:
            target = target_from_decision(msg.label, float(msg.confidence), self._stamp_us, self.decision_v)
        except ValueError as e:
            self.get_logger().warn(str(e))
            return
        self.feed(target)

    def feed(self, target: Target) -> None:
        self._last_target_t, self._silent_warned = self.now(), False
        self.send_all(self.ctrl.update(target, self.now()))

    def on_zone(self, msg: String) -> None:
        try:
            target_from_zone(msg.data)
        except ValueError as e:
            self.get_logger().warn(str(e))
            return
        if msg.data != self._zone:
            self.get_logger().info(f"ball zone: {msg.data}")
        self._zone = msg.data

    def on_zone_tick(self) -> None:
        if self._zone is None:
            return                           # nothing heard yet: the controller keeps waiting
        alive = self.count_publishers(self.zone_topic) > 0
        if not alive and not self._zone_gone_said:
            self.get_logger().warn(f"no publisher on {self.zone_topic} any more: treating it as 'none'")
        self._zone_gone_said = not alive
        self._stamp_us = self.get_clock().now().nanoseconds // 1000
        self.feed(target_from_zone(self._zone if alive else "none", self._stamp_us, self.zone_confidence))

    def on_tick(self) -> None:
        t = self.now()
        if self.ctrl.enabled and self._last_target_t is not None and t - self._last_target_t > 1.0 \
                and not self._silent_warned:
            self.get_logger().warn(f"{self.source} is silent: no new bursts until it comes back")
            self._silent_warned = True
        was = self.ctrl.state
        self.send_all(self.ctrl.update(None, t))
        if was != "halted" and self.ctrl.state == "halted":
            self.say("HALTED: travel budget used up. f to start again")
        if self.debug_pub.get_subscription_count() > 0:
            tg, b = self.ctrl.last_target, self.ctrl.burst
            self.debug_pub.publish(String(data=json.dumps({
                "state": self.ctrl.state, "mode": self.ctrl.cfg.mode, "travel_m": round(self.ctrl.travel_m, 2),
                "yaw": round(self.ctrl.yaw, 3), "pitch": round(self.ctrl.pitch, 3),
                "gesture": self.ctrl.gesture if self.ctrl.cfg.mode == "look" else None,
                "target": None if tg is None else {"u": round(tg.u, 3), "v": round(tg.v, 3),
                                                   "confidence": round(tg.confidence, 3)},
                "burst": None if b is None else {"kind": b.kind, "vx": round(b.vx, 3), "wz": round(b.wz, 3),
                                                 "yaw": round(b.yaw, 3), "pitch": round(b.pitch, 3),
                                                 "reason": b.reason}})))

    # --- output --------------------------------------------------------------------------
    def send_all(self, commands) -> None:
        for c in commands:
            if c.kind == "move":
                vx = max(0.0, min(HARD_MAX_VX, c.vx))
                wz = max(-HARD_MAX_WZ, min(HARD_MAX_WZ, c.wz))
                repeat = c is self._logged_burst          # a burst repeats its Move every resend_s
                self._logged_burst = c
                self.set_posture(False, "Move only walks outside posture mode")
                self.publish(api.MOVE, api.move_params(vx, 0.0, wz), c.reason, log=not repeat)
            elif c.kind == "stop":
                self.publish(api.STOP_MOVE, "", c.reason)
            elif c.kind == "euler":
                yaw = max(-HARD_MAX_YAW, min(HARD_MAX_YAW, c.yaw))
                pitch = self.pitch_sign * max(-HARD_MAX_PITCH, min(HARD_MAX_PITCH, c.pitch)) + 0.0   # no -0.0
                straight = yaw == 0.0 and pitch == 0.0
                if not straight:
                    self.set_posture(True, "Euler only acts in posture mode")
                self.publish(api.EULER, api.euler_params(0.0, pitch, yaw), c.reason)
                if straight:
                    self.set_posture(False, "trunk straight again")
            elif c.kind == "balance_stand":
                self.publish(api.BALANCE_STAND, "", c.reason)

    def set_posture(self, on: bool, reason: str) -> None:
        if self._posture != on:
            self.publish(api.POSE, api.pose_params(on), reason)
            self._posture = on

    def publish(self, cmd, param: str = "", reason: str = "", log: bool = True) -> None:
        if log:
            self.get_logger().info(f"{cmd.name} {param} {('(' + reason + ')') if reason else ''}".rstrip())
        if self.dry_run:
            return
        req = Request()
        req.header.identity.api_id = cmd.api_id
        req.header.identity.id = self._stamp_us     # newest event behind it: latency of the chain
        req.parameter = param
        self.pub.publish(req)


def keyboard(node: FollowNode) -> None:
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            key = sys.stdin.read(1)
            if key == "f":
                node.start()
            elif key == " ":
                node.stop("space bar")
            elif key == "!":
                node.damp()
            elif key in KEY_GESTURES and node.ctrl.cfg.mode == "look":
                node.gesture(KEY_GESTURES[key])
            elif key == "q":
                node.stop("quit")
                rclpy.shutdown()
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    rclpy.init()
    node = FollowNode()
    if sys.stdin.isatty():
        threading.Thread(target=keyboard, args=(node,), daemon=True).start()
    else:
        node.say("no keyboard (not a terminal): use autostart:=true or the /reflex/follow/enable service")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.send_all(node.ctrl.disable(node.now(), "node shutting down"))


if __name__ == "__main__":
    main()
