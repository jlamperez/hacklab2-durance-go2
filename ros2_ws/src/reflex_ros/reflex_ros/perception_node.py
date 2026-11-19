"""Perception node: event window in, reflex decision out.

  subscribes  /reflex/events_window   reflex_msgs/EventWindow
  publishes   /reflex/decision        reflex_msgs/ReflexDecision
              /reflex/debug/perception   std_msgs/String, JSON: what is behind the decision
                                         (class probabilities, firing rate per layer, input activity)

The detector is chosen by a parameter, so the SNN and the no-learning detector are
swapped without touching anything else.
"""
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_srvs.srv import Trigger

from reflex.runtime.detectors import SimpleConfig, SimpleDetector, SNNDetector
from reflex_msgs.msg import EventWindow, ReflexDecision


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("reflex_perception")
        self.kind = self.declare_parameter("detector", "snn").value
        model = self.declare_parameter("model", "out/models/snn.pt").value
        device = self.declare_parameter("device", "cpu").value
        threshold = self.declare_parameter("simple_threshold", 0.0).value
        # Simple detector only: seconds of "nothing happening" used to set its threshold.
        self.calibrate_s = self.declare_parameter("calibrate_s", 1.5).value
        if self.kind == "snn":
            self.detector = SNNDetector(model, device=device)
            self.get_logger().info(f"SNN detector from {model} on {device}")
        elif self.kind == "simple":
            self.detector = SimpleDetector(SimpleConfig())
            if threshold > 0:
                self.detector.cfg.threshold = threshold
                self.calibrate_s = 0.0
            self.get_logger().info("simple detector (no learning)")
        else:
            raise ValueError(f"unknown detector {self.kind!r}")
        self._calib, self._t0 = [], None
        self._calibrate_for = float(self.calibrate_s)
        if self.kind == "simple":
            # ros2 service call /reflex/recalibrate std_srvs/srv/Trigger
            self.create_service(Trigger, "/reflex/recalibrate", self.on_recalibrate)
            if self.calibrate_s > 0:
                self.get_logger().info(f"calibrating: keep the scene still for {self.calibrate_s + 1.0:.0f} s")
        self.pub = self.create_publisher(ReflexDecision, "/reflex/decision", 10)
        self.debug_pub = self.create_publisher(String, "/reflex/debug/perception", 10)
        self.sub = self.create_subscription(EventWindow, "/reflex/events_window", self.on_window,
                                            qos_profile_sensor_data)

    def on_recalibrate(self, request, response):
        self._calib, self._t0, self.calibrate_s = [], None, max(self._calibrate_for, 2.0)
        response.success = True
        response.message = f"calibrating: keep the scene still for {self.calibrate_s + SETTLE_S:.0f} s"
        self.get_logger().info(response.message)
        return response

    def on_window(self, msg: EventWindow) -> None:
        window = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
            msg.n_bins, msg.channels, msg.height, msg.width)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.kind == "simple" and self.calibrate_s > 0:
            self._t0 = t if self._t0 is None else self._t0
            if t - self._t0 < SETTLE_S + self.calibrate_s:
                if t - self._t0 >= SETTLE_S:     # the first second is skipped: cameras settle
                    self._calib.append(window)
                return
            # Median background with a wide margin: a short movement during calibration
            # must not push the threshold out of reach.
            thr = self.detector.calibrate(np.stack(self._calib), margin=3.0, percentile=50.0)
            self.get_logger().info(f"simple detector calibrated on {len(self._calib)} idle windows: "
                                   f"threshold {thr:.0f} events")
            self.calibrate_s, self._calib = 0.0, []
        start = time.perf_counter()
        decision = self.detector.predict(window)
        out = ReflexDecision()
        out.header = msg.header
        out.label, out.confidence, out.detector = decision.label, float(decision.confidence), self.kind
        out.inference_ms = float((time.perf_counter() - start) * 1e3)
        self.pub.publish(out)
        if self.debug_pub.get_subscription_count() > 0:
            self.debug_pub.publish(String(data=json.dumps({
                "detector": self.kind, "label": out.label, "confidence": round(out.confidence, 4),
                "inference_ms": round(out.inference_ms, 2), "events": int(window.sum()),
                "input_active": round(float((window > 0).mean()), 5), **self.detector.last})))


SETTLE_S = 1.0


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
