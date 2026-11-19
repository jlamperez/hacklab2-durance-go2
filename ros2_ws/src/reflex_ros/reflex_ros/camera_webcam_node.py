"""Stand-in event camera: a normal webcam turned into events by the emulator.

  publishes  /reflex/events_window        reflex_msgs/EventWindow
             /reflex/debug/events_image   sensor_msgs/Image (only if someone subscribes)

Purpose: test the plumbing on real hardware (a Kria board, two machines talking over
ROS 2) before the real event camera is available. A 30 fps webcam is a poor event
camera, so the decisions it produces say nothing about the real sensor.

This node is also the template for the real camera adapter: replace `read_events`
with the sensor's SDK and keep the rest.
"""
from array import array
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from reflex.data.windows import WindowConfig
from reflex.events import repr as ev_repr
from reflex.events.emu import EmuConfig, EventEmulator
from reflex.runtime.windower import StreamWindower
from reflex_msgs.msg import EventWindow


class WebcamEventCamera(Node):
    def __init__(self):
        super().__init__("event_camera_webcam")
        p = self.declare_parameter
        device = p("device", 0).value
        fps = float(p("fps", 30.0).value)
        contrast = float(p("contrast_threshold", 0.25).value)
        self.cfg = WindowConfig()
        self.cap = cv2.VideoCapture(int(device), cv2.CAP_V4L2)
        # Ask for compressed frames: uncompressed, a USB 2 port limits many webcams to 7.5 fps.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {device}")
        # No synthetic noise: a real webcam brings its own.
        self.emu = EventEmulator(self.cfg.height, self.cfg.width,
                                 EmuConfig(c_on=contrast, c_off=contrast, noise_rate_hz=0.0, hot_pixel_frac=0.0))
        self.windower = StreamWindower(self.cfg)
        self.win_pub = self.create_publisher(EventWindow, "/reflex/events_window", qos_profile_sensor_data)
        self.img_pub = self.create_publisher(Image, "/reflex/debug/events_image", qos_profile_sensor_data)
        self.frames, self.events, self.t_report = 0, 0, time.monotonic()
        self.timer = self.create_timer(1.0 / fps, self.on_timer)
        self.get_logger().info(f"webcam {device} at {fps:.0f} fps emulating a {self.cfg.width}x{self.cfg.height} "
                               "event camera (plumbing test only)")

    def read_events(self):
        """Return (events, t_us) for everything since the last call, or None."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        t_us = self.get_clock().now().nanoseconds // 1000
        gray = cv2.cvtColor(cv2.resize(frame, (self.cfg.width, self.cfg.height), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return self.emu.step(gray, t_us), t_us

    def on_timer(self) -> None:
        got = self.read_events()
        if got is None:
            return
        events, t_us = got
        self.windower.push(events)
        window = self.windower.window(t_us)
        msg = EventWindow()
        msg.header.stamp.sec, msg.header.stamp.nanosec = t_us // 1_000_000, (t_us % 1_000_000) * 1000
        msg.header.frame_id = "event_camera"
        msg.window_ms = float(self.cfg.window_ms)
        msg.n_bins, msg.channels, msg.height, msg.width = self.cfg.shape
        msg.data = array("B", window.tobytes())   # array: fast path, bytes are checked one by one
        self.win_pub.publish(msg)
        if self.img_pub.get_subscription_count() > 0:
            bgr = ev_repr.render(ev_repr.to_frame(self.windower.recent(t_us), self.cfg.height, self.cfg.width), 1.5)
            img = Image()
            img.header = msg.header
            img.height, img.width, img.encoding, img.step = self.cfg.height, self.cfg.width, "bgr8", 3 * self.cfg.width
            img.data = array("B", bgr.tobytes())
            self.img_pub.publish(img)
        self.frames += 1
        self.events += len(events)
        if time.monotonic() - self.t_report > 5.0:
            dt = time.monotonic() - self.t_report
            self.get_logger().info(f"{self.frames / dt:.1f} windows/s, {self.events / dt / 1e3:.1f} k events/s")
            self.frames, self.events, self.t_report = 0, 0, time.monotonic()


def main():
    rclpy.init()
    node = WebcamEventCamera()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.cap.release()


if __name__ == "__main__":
    main()
