"""Target node: event window in, where the target is in the image out.

  subscribes  /reflex/events_window   reflex_msgs/EventWindow
  publishes   /reflex/target          reflex_msgs/Target   (contract: reflex/runtime/target.py)
              /reflex/debug/target    std_msgs/String, JSON: what is behind the estimate

The no-learning CentroidTracker: the baseline, and the stand-in until the SNN that
outputs coordinates is ready. An SNN adapter (on a PC or on the Kria) publishes the
same /reflex/target message instead of this node.

It calibrates on startup: keep the scene still for the first seconds, as for the
simple detector. `ros2 service call /reflex/target/recalibrate std_srvs/srv/Trigger`
calibrates again.
"""
import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_srvs.srv import Trigger

from reflex.runtime.target import CentroidConfig, CentroidTracker
from reflex_msgs.msg import EventWindow, Target

SETTLE_S = 1.0     # first second skipped when calibrating: cameras settle


class TargetNode(Node):
    def __init__(self):
        super().__init__("reflex_target")
        p = self.declare_parameter
        d = CentroidConfig()
        cfg = CentroidConfig(zone_top=float(p("zone_top", d.zone_top).value),
                             min_share=float(p("min_share", d.min_share).value),
                             radius=int(p("radius", d.radius).value),
                             recent_bins=int(p("recent_bins", d.recent_bins).value))
        self.tracker = CentroidTracker(cfg)
        min_events = float(p("min_events", 0.0).value)       # > 0: fixed threshold, no calibration
        self.calibrate_s = float(p("calibrate_s", 1.5).value)
        if min_events > 0:
            cfg.min_events, self.calibrate_s = min_events, 0.0
        self._calibrate_for = max(self.calibrate_s, 2.0)
        self._calib, self._t0 = [], None
        self.pub = self.create_publisher(Target, "/reflex/target", 10)
        self.debug_pub = self.create_publisher(String, "/reflex/debug/target", 10)
        self.create_service(Trigger, "/reflex/target/recalibrate", self.on_recalibrate)
        self.create_subscription(EventWindow, "/reflex/events_window", self.on_window, qos_profile_sensor_data)
        if self.calibrate_s > 0:
            self.get_logger().info(f"calibrating: keep the scene still for {self.calibrate_s + SETTLE_S:.0f} s")

    def on_recalibrate(self, request, response):
        self._calib, self._t0, self.calibrate_s = [], None, self._calibrate_for
        response.success, response.message = True, f"calibrating: keep the scene still for {self.calibrate_s + SETTLE_S:.0f} s"
        self.get_logger().info(response.message)
        return response

    def on_window(self, msg: EventWindow) -> None:
        window = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.n_bins, msg.channels, msg.height, msg.width)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.calibrate_s > 0:
            self._t0 = t if self._t0 is None else self._t0
            if t - self._t0 < SETTLE_S + self.calibrate_s:
                if t - self._t0 >= SETTLE_S:
                    self._calib.append(window)
                return
            thr = self.tracker.calibrate(np.stack(self._calib))
            self.get_logger().info(f"calibrated on {len(self._calib)} still windows: at least {thr:.0f} events")
            self.calibrate_s, self._calib = 0.0, []
        t_us = msg.header.stamp.sec * 1_000_000 + msg.header.stamp.nanosec // 1000
        target = self.tracker.predict(window, t_us)
        out = Target()
        out.header = msg.header
        out.u, out.v, out.confidence, out.source = float(target.u), float(target.v), float(target.confidence), "centroid"
        self.pub.publish(out)
        if self.debug_pub.get_subscription_count() > 0:
            self.debug_pub.publish(String(data=json.dumps({
                "u": round(out.u, 3), "v": round(out.v, 3), "confidence": round(out.confidence, 3),
                **self.tracker.last})))


def main():
    rclpy.init()
    node = TargetNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
