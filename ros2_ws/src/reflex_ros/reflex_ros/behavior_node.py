"""Behavior node: reflex decisions in, Unitree sport-mode requests out.

  subscribes  /reflex/decision      reflex_msgs/ReflexDecision
  publishes   /api/sport/request    unitree_api/Request
              /reflex/debug/behavior   std_msgs/String, JSON: state of the reflex and last request

It publishes the same message whether a simulator or a real Go2 is listening.

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

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from reflex.runtime import sport_api
from reflex.runtime.detectors import Decision
from reflex.runtime.reflex_fsm import FSMConfig, ReflexFSM
from reflex_msgs.msg import ReflexDecision
from unitree_api.msg import Request

MAX_VY = 0.5      # m/s, hard limit on lateral speed
MAX_ROLL = 0.5    # rad, hard limit on lean


class BehaviorNode(Node):
    def __init__(self):
        super().__init__("reflex_behavior")
        p = self.declare_parameter
        cfg = FSMConfig(mode=p("mode", "keeper").value, min_confidence=p("min_confidence", 0.6).value,
                        consecutive=p("consecutive", 2).value, hold_s=p("hold_s", 0.6).value,
                        refractory_s=p("refractory_s", 0.8).value)
        self.fsm = ReflexFSM(cfg)
        # How a sideways reaction is commanded: "euler" leans the trunk, "move" steps sideways.
        self.reaction = p("reaction", "euler").value
        self.roll = min(abs(p("euler_roll", 0.3).value), MAX_ROLL)
        self.vy = min(abs(p("move_vy", 0.3).value), MAX_VY)
        self.watchdog_s = p("watchdog_s", 0.5).value
        self.enabled = p("enabled", True).value
        self._last_pose, self._last_decision_t, self._silent = "stand", None, False
        self._stamp_us = 0
        self.pub = self.create_publisher(Request, sport_api.SPORT_REQUEST_TOPIC, 10)
        self.sub = self.create_subscription(ReflexDecision, "/reflex/decision", self.on_decision, 10)
        self.debug_pub = self.create_publisher(String, "/reflex/debug/behavior", 10)
        self._last_request, self._reactions, self._ticks = "-", 0, 0
        self.timer = self.create_timer(0.02, self.on_timer)
        self.get_logger().info(f"mode {cfg.mode}, reaction {self.reaction}, enabled {self.enabled}")

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
    rclpy.init()
    node = BehaviorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
