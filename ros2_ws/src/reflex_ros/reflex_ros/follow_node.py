"""Follow node: where the target is in, sport-mode requests out. Move, stop, look.

  subscribes  /reflex/target        reflex_msgs/Target   (target node, or the SNN's adapter)
  publishes   /api/sport/request    unitree_api/Request  (BalanceStand, Move, StopMove, Pose, Euler, Damp)
              /reflex/debug/follow  std_msgs/String, JSON: state, last target, last command

    pixi run -e ros ros-follow                 # walk mode, keyboard in this terminal
    pixi run -e ros ros-follow mode:=gaze      # trunk only (Euler yaw), no walking

Starts DISABLED. Keys:  f start following   SPACE stop (StopMove, disabled)
                        ! Damp (motors limp, the robot drops: emergency only)   q quit
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
from std_msgs.msg import String
from std_srvs.srv import SetBool

from reflex.runtime import sport_api as api
from reflex.runtime.follow import FollowConfig, FollowController
from reflex.runtime.target import Target
from reflex_msgs.msg import Target as TargetMsg
from unitree_api.msg import Request

# Hard limits: whatever the parameters say, the node never commands more than this.
HARD_MAX_VX = 0.5     # m/s
HARD_MAX_WZ = 1.0     # rad/s
HARD_MAX_YAW = 0.5    # rad


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
            lost_s=float(p("lost_s", d.lost_s).value))
        self.dry_run = bool(p("dry_run", False).value)      # log the requests, publish nothing
        self.ctrl = FollowController(cfg)
        self.pub = self.create_publisher(Request, api.SPORT_REQUEST_TOPIC, 10)
        self.debug_pub = self.create_publisher(String, "/reflex/debug/follow", 10)
        self.create_subscription(TargetMsg, "/reflex/target", self.on_target, 10)
        self.create_service(SetBool, "/reflex/follow/enable", self.on_enable)
        self.create_timer(0.02, self.on_tick)
        self._stamp_us = 0
        self._logged_burst = None
        # Posture mode as last commanded. On the real Go2 Euler acts only after Pose(true) and
        # Move does not walk until Pose(false) (sport_api.POSE, reflex/runtime/sim_sport.py).
        self._posture = False
        self._last_target_t = None
        self._silent_warned = False
        self.get_logger().info(f"mode {cfg.mode}: max {cfg.max_vx:.2f} m/s, {cfg.max_wz:.2f} rad/s, "
                               f"bursts of {cfg.move_s:.1f} s, travel budget {cfg.max_travel_m:.1f} m"
                               + ("  [DRY RUN: nothing is published]" if self.dry_run else ""))
        if bool(p("autostart", False).value):
            self.start()
        else:
            self.say("DISABLED. f = start following, SPACE = stop, ! = Damp, q = quit")

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
        self._last_target_t, self._silent_warned = self.now(), False
        target = Target(float(msg.u), float(msg.v), float(msg.confidence), self._stamp_us)
        self.send_all(self.ctrl.update(target, self.now()))

    def on_tick(self) -> None:
        t = self.now()
        if self.ctrl.enabled and self._last_target_t is not None and t - self._last_target_t > 1.0 \
                and not self._silent_warned:
            self.get_logger().warn("/reflex/target is silent: no new bursts until it comes back")
            self._silent_warned = True
        was = self.ctrl.state
        self.send_all(self.ctrl.update(None, t))
        if was != "halted" and self.ctrl.state == "halted":
            self.say("HALTED: travel budget used up. f to start again")
        if self.debug_pub.get_subscription_count() > 0:
            tg, b = self.ctrl.last_target, self.ctrl.burst
            self.debug_pub.publish(String(data=json.dumps({
                "state": self.ctrl.state, "mode": self.ctrl.cfg.mode, "travel_m": round(self.ctrl.travel_m, 2),
                "yaw": round(self.ctrl.yaw, 3),
                "target": None if tg is None else {"u": round(tg.u, 3), "v": round(tg.v, 3),
                                                   "confidence": round(tg.confidence, 3)},
                "burst": None if b is None else {"kind": b.kind, "vx": round(b.vx, 3), "wz": round(b.wz, 3),
                                                 "yaw": round(b.yaw, 3), "reason": b.reason}})))

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
                if yaw != 0.0:
                    self.set_posture(True, "Euler only acts in posture mode")
                self.publish(api.EULER, api.euler_params(0.0, 0.0, yaw), c.reason)
                if yaw == 0.0:
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
