"""Behavior node: what the camera saw in, Unitree sport-mode requests out.

  subscribes  /reflex/decision      reflex_msgs/ReflexDecision  (mode keeper / dodge)
              /durance/ball_zone    std_msgs/String             (mode look: six zones, fpga_perception)
  publishes   /api/sport/request    unitree_api/Request
              /reflex/debug/behavior   std_msgs/String, JSON: state of the reflex and last request

It publishes the same message whether a simulator or a real Go2 is listening.

Modes:
  keeper, dodge  reflexes from left / center / right (ReflexFSM): lean or step, hold, back.
                 A six-zone network reaches them with the row dropped (zones_readout.reflex_label).
  look           the trunk points at the ball's zone, column -> yaw, row -> pitch, and stays
                 there; straight again after lost_s without the ball (reflex/runtime/zone_look.py,
                 the follow controller's look mode, the same gestures as `ros-look`).
                 Parameters: look_yaw (0.15 rad), look_pitch (0.1 rad), camera_on_trunk (false),
                 invert_pitch (false), lost_s (2.0), consecutive, zone_topic, zone_rate_hz (20).

Safety, all here and nowhere else:
  - a reflex needs several agreeing, confident decisions in a row (ReflexFSM)
  - one reaction at a time, then a rest period
  - lateral speed is clamped
  - if perception goes silent, StopMove is sent once
  - `enabled:=false` makes the node log what it would do without publishing

`header.identity.id` carries the time (microseconds) of the newest event behind the
decision, so whoever receives the request can measure the latency of the whole chain.
"""
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from reflex.runtime import sport_api
from reflex.runtime.detectors import Decision
from reflex.runtime.follow import FollowConfig
from reflex.runtime.reflex_fsm import FSMConfig, ReflexFSM
from reflex.runtime.zone_look import ZoneLook
from reflex_msgs.msg import ReflexDecision
from unitree_api.msg import Request

MAX_VY = 0.5      # m/s, hard limit on lateral speed
MAX_ROLL = 0.5    # rad, hard limit on lean


class BehaviorNode(Node):
    def __init__(self):
        super().__init__("reflex_behavior")
        p = self.declare_parameter
        self.mode = p("mode", "keeper").value
        self.watchdog_s = p("watchdog_s", 0.5).value
        self.enabled = p("enabled", True).value
        self._stamp_us = 0
        self._last_request, self._reactions, self._ticks = "-", 0, 0
        self.pub = self.create_publisher(Request, sport_api.SPORT_REQUEST_TOPIC, 10)
        self.debug_pub = self.create_publisher(String, "/reflex/debug/behavior", 10)
        if self.mode == "look":
            self.init_look(p)
            return
        cfg = FSMConfig(mode=self.mode, min_confidence=p("min_confidence", 0.6).value,
                        consecutive=p("consecutive", 2).value, hold_s=p("hold_s", 0.6).value,
                        refractory_s=p("refractory_s", 0.8).value)
        self.fsm = ReflexFSM(cfg)
        # How a sideways reaction is commanded: "euler" leans the trunk, "move" steps sideways.
        self.reaction = p("reaction", "euler").value
        self.roll = min(abs(p("euler_roll", 0.3).value), MAX_ROLL)
        self.vy = min(abs(p("move_vy", 0.3).value), MAX_VY)
        self._last_pose, self._last_decision_t, self._silent = "stand", None, False
        self.sub = self.create_subscription(ReflexDecision, "/reflex/decision", self.on_decision, 10)
        self.timer = self.create_timer(0.02, self.on_timer)
        self.get_logger().info(f"mode {cfg.mode}, reaction {self.reaction}, enabled {self.enabled}")

    # --- mode look: six zones -> trunk gestures ---------------------------------------------
    def init_look(self, p) -> None:
        d = FollowConfig()
        cfg = FollowConfig(
            mode="look", consecutive=int(p("consecutive", d.consecutive).value),
            lost_s=float(p("lost_s", d.lost_s).value),
            look_yaw=float(p("look_yaw", 0.15).value), look_pitch=float(p("look_pitch", 0.1).value),
            camera_on_trunk=bool(p("camera_on_trunk", False).value))
        pitch_sign = -1.0 if bool(p("invert_pitch", False).value) else 1.0
        self.look = ZoneLook(cfg, pitch_sign)
        self.zone_topic = p("zone_topic", "/durance/ball_zone").value
        self._zone_gone_said = False
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)     # as fpga_perception publishes
        self.create_subscription(String, self.zone_topic, self.on_zone, qos)
        self.create_timer(1.0 / float(p("zone_rate_hz", 20.0).value), self.on_zone_tick)
        self.get_logger().info(
            f"mode look from {self.zone_topic}: yaw {cfg.look_yaw:.2f} rad, pitch {cfg.look_pitch:.2f} rad"
            f"{' (inverted)' if pitch_sign < 0 else ''}, camera {'on the trunk' if cfg.camera_on_trunk else 'fixed'}, "
            f"enabled {self.enabled}")
        self.send_requests(self.look.enable(self.now()))

    def on_zone(self, msg: String) -> None:
        before = self.look.zone
        try:
            self.look.set_zone(msg.data)
        except ValueError as e:
            self.get_logger().warn(str(e))
            return
        if self.look.zone != before:
            self.get_logger().info(f"ball zone: {self.look.zone}")

    def on_zone_tick(self) -> None:
        alive = self.count_publishers(self.zone_topic) > 0
        if self.look.zone is not None and not alive and not self._zone_gone_said:
            self.get_logger().warn(f"no publisher on {self.zone_topic} any more: treating it as 'none'")
        self._zone_gone_said = not alive
        self._stamp_us = self.get_clock().now().nanoseconds // 1000
        before = self.look.gesture
        self.send_requests(self.look.tick(self.now(), self._stamp_us, alive))
        self._reactions += self.look.gesture != before and self.look.gesture != "center"
        self._ticks += 1
        if self._ticks % 2 == 0 and self.debug_pub.get_subscription_count() > 0:
            self.debug_pub.publish(String(data=json.dumps({
                "state": self.look.ctrl.state, "pose": self.look.gesture, "zone": self.look.zone,
                "last_request": self._last_request, "reactions": self._reactions, "mode": "look",
                "enabled": bool(self.enabled)})))

    def send_requests(self, requests) -> None:
        for r in requests:
            if self.enabled:                  # disabled: send() logs what it would send
                self.get_logger().info(f"{r.cmd.name} {r.param} ({r.reason})")
            else:
                self.get_logger().info(f"({r.reason})")
            self.send(r.cmd, r.param)

    def shutdown(self) -> None:
        """Trunk straight and out of posture mode (look) before the node goes."""
        if self.mode == "look":
            self.send_requests(self.look.disable(self.now(), "node shutting down"))

    # --- common -----------------------------------------------------------------------------
    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_decision(self, msg: ReflexDecision) -> None:
        self._last_decision_t, self._silent = self.now(), False
        self._stamp_us = msg.header.stamp.sec * 1_000_000 + msg.header.stamp.nanosec // 1000
        self.act(self.fsm.update(Decision(msg.label, msg.confidence, self._stamp_us), self.now()))

    def publish_debug(self) -> None:
        if self.debug_pub.get_subscription_count() > 0:
            self.debug_pub.publish(String(data=json.dumps({
                "state": self.fsm.state, "pose": self._last_pose, "last_request": self._last_request,
                "reactions": self._reactions, "mode": self.fsm.cfg.mode, "enabled": bool(self.enabled)})))

    def on_timer(self) -> None:
        self.act(self.fsm.update(None, self.now()))        # lets the hold and the rest expire
        self._ticks += 1
        if self._ticks % 5 == 0:
            self.publish_debug()
        if (self._last_decision_t is not None and not self._silent
                and self.now() - self._last_decision_t > self.watchdog_s):
            self._silent = True
            self.get_logger().warn("perception is silent: sending StopMove")
            self.send(sport_api.STOP_MOVE)

    def act(self, action) -> None:
        if action is None:
            return
        for cmd, param in self.requests_for(action.name):
            self.send(cmd, param)
        self._last_pose = action.name
        self._reactions += action.name != "stand"
        self.publish_debug()
        self.get_logger().info(f"{action.label} -> {action.name}")

    def requests_for(self, pose: str) -> list:
        """Sport-mode requests that realize a pose (logic in sport_api.requests_for_pose)."""
        return sport_api.requests_for_pose(pose, self._last_pose, self.reaction, self.roll, self.vy)

    def send(self, cmd, param: str = "") -> None:
        self._last_request = f"{cmd.name} {param}".strip()
        if not self.enabled:
            self.get_logger().info(f"[disabled] would send {cmd.name} {param}")
            return
        req = Request()
        req.header.identity.id = int(self._stamp_us)
        req.header.identity.api_id = cmd.api_id
        req.parameter = param
        self.pub.publish(req)


def main():
    # Python's own SIGINT handling, not rclpy's: rclpy's shuts the context down at Ctrl-C, and
    # then the trunk could not be straightened on the way out (look mode).
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = BehaviorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.shutdown()
            time.sleep(0.2)              # let DDS deliver the last requests
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
