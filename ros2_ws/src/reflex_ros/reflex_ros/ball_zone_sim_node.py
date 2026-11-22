"""Stand-in for the Kria's ball_node: publishes a ball zone on /durance/ball_zone.

  publishes   /durance/ball_zone      std_msgs/String, `upper_left ... lower_right` or `none`,
                                      reliable + transient-local, only when the zone changes
                                      (what the follow node's input:=zone expects)

The real ball_node reads the zones network's spikes from SPLEAT on the FPGA. This node
gets the zone from one of two places, so `input:=zone` can be tested without the board:

  source:=snn   (default) subscribes /reflex/events_fullres (a recording or the live camera
                through the event_camera node) and runs the same zones network on the CPU
                (reflex/runtime/ball_snn.py, out/models/ball_zones_best.pt). A frame below
                min_p or min_pixels counts as `none`. Majority vote over the last `vote_window`
                decisions (reflex/runtime/zone_vote.py); the board's own vote rule is unknown.
  source:=keys  zones by hand, numpad layout: 7 8 9 upper row, 1 2 3 lower row, 0 or 5 none,
                q quit. Every key press is published, no vote.

    ros2 run reflex_ros ball_zone_sim                                  # with a recording playing
    ros2 run reflex_ros ball_zone_sim --ros-args -p source:=keys
"""
import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from reflex.runtime.zone_vote import ZoneVoteConfig, ZoneVoter

KEY_ZONES = {"7": "upper_left", "8": "upper_middle", "9": "upper_right",
             "1": "lower_left", "2": "lower_middle", "3": "lower_right",
             "0": "none", "5": "none"}


class BallZoneSimNode(Node):
    def __init__(self):
        super().__init__("ball_zone_sim")
        p = self.declare_parameter
        self.source = p("source", "snn").value
        topic = p("zone_topic", "/durance/ball_zone").value
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)        # as ball_node publishes
        self.pub = self.create_publisher(String, topic, qos)
        self.publish("none")
        if self.source == "keys":
            self.get_logger().info(f"keys -> {topic}: 7 8 9 upper row, 1 2 3 lower row, 0/5 none, q quit")
            return
        if self.source != "snn":
            raise ValueError(f"source:={self.source}: use snn or keys")

        import numpy as np
        from reflex.runtime.ball_snn import BallSNNConfig, BallSNNTracker
        from reflex_msgs.msg import EventWindow
        self._np = np
        d = BallSNNConfig()
        cfg = BallSNNConfig(checkpoint=p("checkpoint", d.checkpoint).value,
                            training_dir=p("training_dir", d.training_dir).value,
                            riverops_dir=p("riverops_dir", d.riverops_dir).value,
                            min_p=float(p("min_p", d.min_p).value),
                            min_pixels=int(p("min_pixels", d.min_pixels).value),
                            device=p("device", d.device).value)
        self.tracker = BallSNNTracker(cfg)
        if not self.tracker.zones:
            raise ValueError(f"{cfg.checkpoint} is a {self.tracker.architecture} network, not a zones network")
        vote = ZoneVoteConfig(window=int(p("vote_window", 5).value), min_votes=int(p("vote_min", 3).value))
        self.voter = ZoneVoter(vote)
        self.get_logger().info(
            f"zones network {cfg.checkpoint} on /reflex/events_fullres -> {topic}; P(zone) >= {cfg.min_p}, "
            f">= {cfg.min_pixels} event pixels, vote {vote.min_votes} of the last {vote.window}")
        sub_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(EventWindow, "/reflex/events_fullres", self.on_window, sub_qos)

    def on_window(self, msg) -> None:
        window = self._np.frombuffer(bytes(msg.data), dtype=self._np.uint8).reshape(
            msg.n_bins, msg.channels, msg.height, msg.width)
        target = self.tracker.predict(self.tracker.frame(window))
        zone = self.tracker.last.get("zone", "none") if target.confidence > 0 else "none"
        changed = self.voter.update(zone)
        if changed is not None:
            self.publish(changed)

    def publish(self, zone: str) -> None:
        self.pub.publish(String(data=zone))
        self.get_logger().info(f"zone {zone}")


def keyboard(node: BallZoneSimNode) -> None:
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            key = sys.stdin.read(1)
            if key in KEY_ZONES:
                node.publish(KEY_ZONES[key])
            elif key == "q":
                rclpy.shutdown()
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    rclpy.init()
    node = BallZoneSimNode()
    if node.source == "keys":
        if not sys.stdin.isatty():
            node.get_logger().error("source:=keys needs a terminal")
            return
        threading.Thread(target=keyboard, args=(node,), daemon=True).start()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
