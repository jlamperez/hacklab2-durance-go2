"""Event camera: recordings played back, or the live GenX320 on the KV260.

  publishes  /reflex/events_window        reflex_msgs/EventWindow  (320x320 sensor, 80x80 cells)
             /reflex/events_fullres       reflex_msgs/EventWindow  (1 bin, 320x320, last `fullres_ms`;
                                          for the SNN ball detector, only while someone subscribes)
             /reflex/debug/events_image   sensor_msgs/Image (only if someone subscribes)
             /reflex/debug/replay         std_msgs/String, JSON: which take is playing

    ros2 run reflex_ros event_camera --ros-args -p source:=data/river/sat_follow2
    ros2 run reflex_ros event_camera --ros-args -p source:=live -p kria_host:=root@durance-kria.local

source: one or more `.h5` / `.bin` files or folders (comma-separated), or `live`.
Replay runs in real time (`speed`), in recording order, and loops. The session's still
scene (`none/empty*`) plays first (`still_first`), so the detectors calibrate on real
sensor noise; when looping it is part of every pass. Live listens on
`port` for despot_debug_camera, the way `river-tools capture` does; with `kria_host` it
starts the sender on the board over ssh. Stop river-tools capture first: both want the port.

Everything downstream (target, perception, follow, behavior, the web viewer) sees the
same topic as from the simulated camera, so the simulated dog reacts to real events.
Open loop for replays: the recording does not change when the simulated dog moves.
The logic is in reflex/events/sources.py.
"""
from array import array
import json
import time

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from reflex.data.windows import WindowConfig, make_window
from reflex.events import repr as ev_repr
from reflex.events.sources import LiveSource, ReplaySource, find_takes, still_take
from reflex.runtime.windower import StreamWindower
from reflex_msgs.msg import EventWindow


def _list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


class EventCamera(Node):
    def __init__(self):
        super().__init__("event_camera")
        p = self.declare_parameter
        source = p("source", "live").value
        width, height = int(p("width", 320).value), int(p("height", 320).value)
        flip_x, flip_y = bool(p("flip_x", False).value), bool(p("flip_y", False).value)
        rate_hz = float(p("rate_hz", 50.0).value)
        self.cfg = WindowConfig(height=height, width=width)
        # Full resolution, one bin: the ball SNN's input (reflex/runtime/ball_snn.py). Must be shorter
        # than the main window, which is all the windower keeps.
        fullres_ms = min(float(p("fullres_ms", 20.0).value), self.cfg.window_ms)
        self.fullres_cfg = WindowConfig(window_ms=fullres_ms, n_bins=1, downsample=1, height=height, width=width)
        self.windower = StreamWindower(self.cfg)
        log = self.get_logger().info
        if source == "live":
            self.src = LiveSource(port=int(p("port", 3333).value), kria_host=p("kria_host", "").value,
                                  width=width, height=height, flip_x=flip_x, flip_y=flip_y, log=log)
            self.src.start()
            self.live, self.epoch = True, 0
            what = f"live camera on TCP port {self.src.port}"
        else:
            takes = find_takes(_list(source), exclude=_list(p("exclude", "").value))
            # Detectors calibrate on the first seconds: give them the still scene, not the ball.
            if takes and bool(p("still_first", True).value):
                still = still_take(takes[0])
                if still is not None and still.path not in [t.path for t in takes[:1]]:
                    takes.insert(0, still)
            self.src = ReplaySource(takes, speed=float(p("speed", 1.0).value), loop=bool(p("loop", True).value),
                                    gap_s=float(p("gap_s", 1.0).value), lead_s=float(p("lead_s", 0.0).value),
                                    width=width, height=height, flip_x=flip_x, flip_y=flip_y)
            self.live = False
            what = f"{len(takes)} recordings from {source}"
            for t in takes:
                log(f"  {t.describe()}")
        self.win_pub = self.create_publisher(EventWindow, "/reflex/events_window", qos_profile_sensor_data)
        self.fullres_pub = self.create_publisher(EventWindow, "/reflex/events_fullres", qos_profile_sensor_data)
        self.img_pub = self.create_publisher(Image, "/reflex/debug/events_image", 2)
        self.take_pub = self.create_publisher(String, "/reflex/debug/replay", 10)
        self._take_name, self._finished_said = None, False
        self.windows, self.events, self.t_report = 0, 0, time.monotonic()
        self.create_timer(1.0 / rate_hz, self.on_timer, clock=Clock(clock_type=ClockType.STEADY_TIME))
        log(f"event camera: {what}, {width}x{height}{' mirrored' if flip_x else ''}"
            f"{' upside down' if flip_y else ''}, windows of {self.cfg.window_ms:.0f} ms at {rate_hz:.0f} Hz")

    def on_timer(self) -> None:
        now_us = time.monotonic_ns() // 1000
        if self.live and self.src.epoch != self.epoch:        # camera restarted: old events are void
            self.windower, self.epoch = StreamWindower(self.cfg), self.src.epoch
        ev = self.src.poll(now_us)
        t_end = self.src.t_end(now_us)
        if t_end is None:                                     # live, nothing received yet
            return
        self.windower.push(ev)
        self.events += len(ev)
        if not self.live:
            self.report_take(now_us)
        stamp_ns = self.get_clock().now().nanoseconds
        msg = EventWindow()
        msg.header.stamp.sec, msg.header.stamp.nanosec = stamp_ns // 1_000_000_000, stamp_ns % 1_000_000_000
        msg.header.frame_id = "event_camera"
        msg.window_ms = float(self.cfg.window_ms)
        msg.n_bins, msg.channels, msg.height, msg.width = self.cfg.shape
        msg.data = array("B", self.windower.window(t_end).tobytes())
        self.win_pub.publish(msg)
        if self.fullres_pub.get_subscription_count() > 0:
            full = EventWindow()
            full.header = msg.header
            full.window_ms = float(self.fullres_cfg.window_ms)
            full.n_bins, full.channels, full.height, full.width = self.fullres_cfg.shape
            recent = self.windower.recent(t_end, span_ms=self.fullres_cfg.window_ms)
            full.data = array("B", make_window(recent, t_end, self.fullres_cfg).tobytes())
            self.fullres_pub.publish(full)
        if self.img_pub.get_subscription_count() > 0:
            bgr = ev_repr.render(ev_repr.to_frame(self.windower.recent(t_end), self.cfg.height, self.cfg.width), 1.5)
            img = Image()
            img.header = msg.header
            img.height, img.width, img.encoding, img.step = self.cfg.height, self.cfg.width, "bgr8", 3 * self.cfg.width
            img.data = array("B", bgr.tobytes())
            self.img_pub.publish(img)
        self.windows += 1
        if time.monotonic() - self.t_report > 5.0:
            dt = time.monotonic() - self.t_report
            extra = ""
            if self.live:
                extra = "" if self.src.connected else " (camera not connected)"
                if self.src.dropped_late:
                    extra += f", {self.src.dropped_late} late events dropped"
            self.get_logger().info(f"{self.windows / dt:.1f} windows/s, {self.events / dt / 1e3:.1f} k events/s{extra}")
            self.windows, self.events, self.t_report = 0, 0, time.monotonic()

    def report_take(self, now_us: int) -> None:
        cur = self.src.current(now_us)
        name = None if cur is None else cur[0].name
        if name != self._take_name:
            self._take_name = name
            if cur is not None:
                take = cur[0]
                self.get_logger().info(f"playing {take.describe()}"
                                       + (f": {take.info['instruction']}" if take.info.get("instruction") else ""))
                self.take_pub.publish(String(data=json.dumps({"take": take.name, **{
                    k: v for k, v in take.info.items() if isinstance(v, (str, int, float))}})))
            else:
                self.take_pub.publish(String(data=json.dumps({"take": None})))
        if self.src.finished and not self._finished_said:
            self._finished_said = True
            self.get_logger().info("replay finished (loop:=false): publishing empty windows")


def main():
    rclpy.init()
    node = EventCamera()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node.live:
            node.src.close()


if __name__ == "__main__":
    main()
