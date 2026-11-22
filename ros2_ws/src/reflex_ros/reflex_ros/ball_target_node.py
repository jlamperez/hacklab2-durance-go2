"""Ball target from the SNN: full-resolution event window in, where the ball is out.

  subscribes  /reflex/events_fullres  reflex_msgs/EventWindow (1 bin, 320x320, no downsampling;
                                      the event camera publishes it only while someone listens)
  publishes   /reflex/target          reflex_msgs/Target, source "snn-ball" or "snn-zones"
              /reflex/debug/target    std_msgs/String, JSON: P(ball), pixel position or zone,
                                      spikes per class, timing

Drop-in replacement for the `target` node (centroid tracker): same output, so the follow
node does not change. The network is one of Theodor's ball detectors from feat/training,
loaded by reflex/runtime/ball_snn.py: the zone classifier (default,
out/models/ball_zones_best.pt; the target is the centre of one of six zones) or a
position network. No calibration step.

    ros2 run reflex_ros ball_target --ros-args -p checkpoint:=out/models/ball_zones_best.pt

Runs on the CPU (about 25 ms a frame on 4 arm64 cores; frames with fewer than
min_pixels event pixels skip the network). The subscription keeps only the
newest window, so a slow machine skips frames instead of falling behind.
"""
import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from reflex.runtime.ball_snn import BallSNNConfig, BallSNNTracker
from reflex_msgs.msg import EventWindow, Target


class BallTargetNode(Node):
    def __init__(self):
        super().__init__("reflex_ball_target")
        p = self.declare_parameter
        d = BallSNNConfig()
        cfg = BallSNNConfig(checkpoint=p("checkpoint", d.checkpoint).value,
                            training_dir=p("training_dir", d.training_dir).value,
                            riverops_dir=p("riverops_dir", d.riverops_dir).value,
                            min_p=float(p("min_p", d.min_p).value),
                            min_pixels=int(p("min_pixels", d.min_pixels).value),
                            device=p("device", d.device).value)
        self.tracker = BallSNNTracker(cfg)
        self._warned_window = False
        v = self.tracker.validation
        self.source = "snn-zones" if self.tracker.zones else "snn-ball"
        score = (f"; validation median error {v['median']:.0f} px" if "median" in v else
                 f"; validation balanced accuracy {v['balanced_accuracy']:.0%}" if "balanced_accuracy" in v else "")
        self.get_logger().info(
            f"SNN ball detector {self.tracker.architecture} from {cfg.checkpoint}: trained on "
            f"{self.tracker.window_ms:g} ms windows, {self.tracker.time_steps} time steps, "
            f"{self.tracker.channels} input channel(s), "
            f"{'P(zone)' if self.tracker.zones else 'P(ball)'} >= {cfg.min_p}, "
            f">= {cfg.min_pixels} event pixels" + score)
        self.pub = self.create_publisher(Target, "/reflex/target", 10)
        self.debug_pub = self.create_publisher(String, "/reflex/debug/target", 10)
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(EventWindow, "/reflex/events_fullres", self.on_window, qos)

    def on_window(self, msg: EventWindow) -> None:
        if not self._warned_window and abs(msg.window_ms - self.tracker.window_ms) > 1e-3:
            self.get_logger().info(f"windows are {msg.window_ms:g} ms, the network was trained on "
                                   f"{self.tracker.window_ms:g} ms; see reflex/runtime/ball_snn.py")
            self._warned_window = True
        window = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.n_bins, msg.channels, msg.height, msg.width)
        t_us = msg.header.stamp.sec * 1_000_000 + msg.header.stamp.nanosec // 1000
        target = self.tracker.predict(self.tracker.frame(window), t_us)
        out = Target()
        out.header = msg.header
        out.u, out.v, out.confidence, out.source = float(target.u), float(target.v), float(target.confidence), self.source
        self.pub.publish(out)
        if self.debug_pub.get_subscription_count() > 0:
            self.debug_pub.publish(String(data=json.dumps({
                "u": round(out.u, 3), "v": round(out.v, 3), "confidence": round(out.confidence, 3),
                "window_ms": msg.window_ms, **self.tracker.last})))


def main():
    rclpy.init()
    node = BallTargetNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
