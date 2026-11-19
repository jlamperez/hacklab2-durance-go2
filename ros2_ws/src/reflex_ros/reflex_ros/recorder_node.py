"""Record labelled clips from the live event stream, one key per take.

  subscribes  /reflex/events_window   reflex_msgs/EventWindow   (simulator, webcam or real camera)
              /reflex/record/cmd      std_msgs/String           (same commands as the keys, for scripts)
  writes      <out>/<session>/clip_NNN.npz

    pixi run -e ros ros-record session:=sat_am

Keys (sides are the DOG's left and right):
  l / c / r   start a take: a ball that will arrive on the left / center / right
  n           start a take of something that must NOT trigger a reflex
  SPACE       ball takes: press when the ball ARRIVES (marks the time and ends the take)
              "none" takes: ends the take
  x           cancel the take in progress
  u           undo: delete the last saved take
  q           quit

Labels are worked out later from the arrival mark (`pixi run build-real`), so a late or
early key press can be compensated without recording again.
"""
import sys
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from reflex.data.recording import ClipRecorder
from reflex_msgs.msg import EventWindow

KEYS = {"l": "left", "c": "center", "r": "right", "n": "none", " ": "hit", "x": "cancel", "u": "undo", "q": "quit"}


class Recorder(Node):
    def __init__(self):
        super().__init__("reflex_recorder")
        p = self.declare_parameter
        self.session = str(p("session", "session").value)
        out = str(p("out", "data/real").value)
        self.max_clip_s = float(p("max_clip_s", 20.0).value)
        self.rec: ClipRecorder | None = None
        self._dir = f"{out}/{self.session}"
        self._last_t_us, self._start_t_us, self._n = 0, 0, 0
        self.lock = threading.Lock()
        self.create_subscription(EventWindow, "/reflex/events_window", self.on_window, qos_profile_sensor_data)
        self.create_subscription(String, "/reflex/record/cmd", lambda m: self.command(m.data), 10)
        self.say(f"session '{self.session}' -> {self._dir}   keys: l c r n = start, SPACE = arrived/stop, "
                 "x cancel, u undo, q quit")

    def say(self, text: str) -> None:
        print(f"\r[rec] {text}\r", flush=True)       # \r: the terminal is in raw mode

    def on_window(self, msg: EventWindow) -> None:
        window = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.n_bins, msg.channels, msg.height, msg.width)
        t_us = msg.header.stamp.sec * 1_000_000 + msg.header.stamp.nanosec // 1000
        with self.lock:
            if self.rec is None:
                self.rec = ClipRecorder(self._dir, {"window_ms": float(msg.window_ms), "n_bins": int(msg.n_bins),
                                                    "height": int(msg.height), "width": int(msg.width)})
                self.say(f"stream is live ({msg.n_bins}x{msg.channels}x{msg.height}x{msg.width}); "
                         f"{len(self.rec.saved)} takes already in this session")
            self._last_t_us = t_us
            if self.rec.recording:
                self.rec.add(window, t_us)
                self._n += 1
                if (t_us - self._start_t_us) * 1e-6 > self.max_clip_s:
                    self.say(f"take longer than {self.max_clip_s:.0f} s: cancelled")
                    self.rec.cancel()

    def command(self, cmd: str) -> None:
        with self.lock:
            if cmd == "quit":
                raise SystemExit(0)
            if self.rec is None:
                self.say("no windows on /reflex/events_window yet")
                return
            rec = self.rec
            if cmd in ("left", "center", "right", "none"):
                rec.start(cmd)
                self._start_t_us, self._n = self._last_t_us, 0
                self.say(f"RECORDING {cmd.upper()}" + ("  (SPACE when the ball arrives)" if cmd != "none" else "  (SPACE to stop)"))
            elif cmd == "hit" and rec.recording:
                label, path = rec.label, rec.stop(self._last_t_us)
                if path is None:
                    self.say("nothing recorded")
                    return
                c = rec.counts()
                self.say(f"saved {path.name}: {label}, {self._n} windows, {(self._last_t_us - self._start_t_us) * 1e-6:.1f} s"
                         f"   totals: " + "  ".join(f"{k} {v}" for k, v in c.items()))
            elif cmd == "cancel" and rec.recording:
                rec.cancel()
                self.say("cancelled")
            elif cmd == "undo":
                path = rec.undo()
                self.say(f"deleted {path.name}" if path else "nothing to undo")


def keyboard(node: Recorder) -> None:
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            cmd = KEYS.get(sys.stdin.read(1).lower())
            if cmd == "quit":
                rclpy.shutdown()
                return
            if cmd:
                node.command(cmd)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    rclpy.init()
    node = Recorder()
    if sys.stdin.isatty():
        threading.Thread(target=keyboard, args=(node,), daemon=True).start()
    else:
        node.say("no keyboard (not a terminal): send commands on /reflex/record/cmd")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException, SystemExit):
        pass


if __name__ == "__main__":
    main()
