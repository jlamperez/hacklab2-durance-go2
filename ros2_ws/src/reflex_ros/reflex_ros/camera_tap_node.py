"""Camera tap: between despot and the FPGA server; fixes the input layout and shows the picture.

    despot_send_and_forget --TCP 3333--> camera_tap --TCP 3334--> spleat-server (FPGA)
                                            |
                                            +--> /reflex/debug/events_image  sensor_msgs/Image (rgb8)

On the Kria. despot always sends to 127.0.0.1:3333, so the server moves to 3334 (run.sh sets
it) and this node takes 3333. fpga_perception reads the results from the server on 3334.

Every frame is read from despot at once (despot is never held back), reordered from despot's
channel-row-column (CHW) to the row-column-channel (HWC) the FPGA reads (`to_hwc`, default on;
the values are not changed) and forwarded. Found on 22 Nov by replaying labelled takes into
the board (tools/replay_to_fpga.py --layout): HWC matches the laptop's emulation, CHW does
not; synthetic frames with both channels equal could not show it. The reorder works on the
payload bytes in place, without decoding and re-encoding the CBOR message (that took 7 ms per
2x320x320 frame on the board).

Frames are dropped, not queued, while the FPGA is busy: at most `max_inflight` (1) frames
without a reply. Otherwise a slow FPGA holds the TCP stream back, despot queues behind it and
the decisions lag further and further behind the camera (22 Nov, models/ball_v3: 200 frames/s
from the camera, about 15 ms per frame on the FPGA). A frame without a reply after
`reply_timeout_s` (0.2) counts as lost (the server drops frames itself when its pipeline is
full). The server's replies on this connection are counted and dropped.

The picture: the frames received since the last one summed, in despot's order, in the repo's
colours (reflex/events/repr.py): channel 0 red, channel 1 light blue (which channel is ON is
not documented by Durance; the repo calls 1 ON), with the network's zones marked as the camera
sees them: `zone_columns` x `zone_rows` (3 x 1 for models/zones3, 3 x 2 for models/ball_v3).
Frames larger than `max_side` (80) are summed in blocks down to it first: building a big
picture in Python delays the forwarding. At `image_hz` (10); the web dashboard (ros-web)
shows it under "Event camera".

Parameters: listen_port (3333), server_host (127.0.0.1), server_port (3334), to_hwc (true),
image_hz (10), scale (1), gain (64: brightness per event), max_side (80), zone_columns (3),
zone_rows (1), max_inflight (1, 0: forward every frame), reply_timeout_s (0.2), stats_s (10:
log frames read / forwarded / answered).
"""
from __future__ import annotations

import array
import socket
import struct
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from reflex.runtime.spleat_client import read_exact

# CBOR: text key "payload", then a byte string (major 2) with a 1, 2, 4 or 8-byte length.
_PAYLOAD_KEY = b"\x67payload"
_BYTES_LEN = {0x58: 1, 0x59: 2, 0x5A: 4, 0x5B: 8}


class CameraTapNode(Node):
    def __init__(self):
        super().__init__("reflex_camera_tap")
        p = self.declare_parameter
        self.listen_port = int(p("listen_port", 3333).value)
        self.server = (p("server_host", "127.0.0.1").value, int(p("server_port", 3334).value))
        # Kept small and slow on purpose: this node forwards every camera frame to the FPGA, and
        # building a big image in Python holds the interpreter, so frames queue behind it (seen
        # on 22 Nov with 240x240 at 15 Hz: the whole pipeline lagged). The browser scales it up.
        self.period = 1.0 / float(p("image_hz", 10.0).value)
        self.scale = int(p("scale", 1).value)
        self.gain = float(p("gain", 64.0).value)
        self.max_side = int(p("max_side", 80).value)
        self.zone_columns = int(p("zone_columns", 3).value)
        self.zone_rows = int(p("zone_rows", 1).value)
        # despot sends channel, row, column (CHW); the FPGA reads its input as row, column,
        # channel (HWC). Found on 22 Nov by replaying labelled takes into the board
        # (tools/replay_to_fpga.py --layout): HWC matches the laptop's emulation, CHW does not.
        self.to_hwc = bool(p("to_hwc", True).value)
        self.max_inflight = int(p("max_inflight", 1).value)
        self.reply_timeout = float(p("reply_timeout_s", 0.2).value)
        self.stats_s = float(p("stats_s", 10.0).value)
        self._inflight, self._last_sent = 0, 0.0
        self._stats = {"read": 0, "sent": 0, "replies": 0}
        self.pub = self.create_publisher(Image, "/reflex/debug/events_image", 2)
        self._acc, self._frames, self._lock = None, 0, threading.Lock()
        self._stop = threading.Event()
        self.get_logger().info(f"listening on {self.listen_port} for despot, forwarding to "
                               f"{self.server[0]}:{self.server[1]}")
        threading.Thread(target=self.serve, daemon=True).start()
        self.create_timer(self.period, self.publish_image)

    # ------------------------------------------------------------------ forwarding
    def serve(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self.listen_port))
        srv.listen(1)
        while not self._stop.is_set():
            client, addr = srv.accept()
            self.get_logger().info(f"camera client {addr[0]} connected")
            try:
                with client, socket.create_connection(self.server, timeout=5) as up:
                    up.settimeout(None)
                    with self._lock:
                        self._inflight = 0
                    threading.Thread(target=self.drain, args=(up,), daemon=True).start()
                    t_stats = time.monotonic()
                    while not self._stop.is_set():
                        head = read_exact(client, 4)
                        (n,) = struct.unpack(">I", head)
                        body = read_exact(client, n)
                        self._stats["read"] += 1
                        now = time.monotonic()
                        if self.stats_s > 0 and now - t_stats >= self.stats_s:
                            self.log_stats(now - t_stats)
                            t_stats = now
                        if not self.may_send(now):
                            continue                     # FPGA busy: drop, do not queue
                        frame, body = self.reorder(body)
                        up.sendall(head + body)          # first, before the picture
                        self._stats["sent"] += 1
                        if frame is not None:
                            self.add_frame(frame)
            except (OSError, ValueError) as e:
                self.get_logger().warn(f"camera link closed: {e}")

    def may_send(self, now: float) -> bool:
        if self.max_inflight <= 0:
            return True
        with self._lock:
            if self._inflight > 0 and now - self._last_sent > self.reply_timeout:
                self._inflight = 0                       # lost: the server dropped it
            if self._inflight >= self.max_inflight:
                return False
            self._inflight += 1
            self._last_sent = now
            return True

    def drain(self, sock: socket.socket) -> None:
        """Read the server's replies on this connection; each one frees a slot."""
        try:
            while True:
                (n,) = struct.unpack(">I", read_exact(sock, 4))
                read_exact(sock, n)
                with self._lock:
                    self._inflight = max(0, self._inflight - 1)
                self._stats["replies"] += 1
        except (OSError, ConnectionError):
            pass

    def log_stats(self, dt: float) -> None:
        s = self._stats
        self.get_logger().info(f"last {dt:.0f} s: {s['read'] / dt:.0f} frames/s from despot, "
                               f"{s['sent'] / dt:.0f}/s to the FPGA, {s['replies'] / dt:.0f}/s answered")
        self._stats = {"read": 0, "sent": 0, "replies": 0}

    def reorder(self, body: bytes):
        """(frame as 2 x H x W in despot's order, body to forward) or (None, body) if not a frame.

        The payload is rewritten in place: same length, so the CBOR heads stay valid.
        """
        k = body.find(_PAYLOAD_KEY)
        i = k + len(_PAYLOAD_KEY)
        if k < 0 or i >= len(body) or body[i] not in _BYTES_LEN:
            return None, body
        size = _BYTES_LEN[body[i]]
        n = int.from_bytes(body[i + 1:i + 1 + size], "big")
        start = i + 1 + size
        side = int(round((n / 8) ** 0.5))
        if start + n > len(body) or 8 * side * side != n:
            return None, body
        frame = np.frombuffer(body, "<f4", count=n // 4, offset=start).reshape(2, side, side)
        if not self.to_hwc:
            return frame, body
        out = bytearray(body)
        np.frombuffer(out, "<f4", count=n // 4, offset=start)[:] = frame.transpose(1, 2, 0).reshape(-1)
        return frame, bytes(out)

    def add_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._acc = frame.copy() if self._acc is None or self._acc.shape != frame.shape else self._acc + frame
            self._frames += 1

    # ------------------------------------------------------------------ picture
    def publish_image(self) -> None:
        with self._lock:
            acc, n, self._acc, self._frames = self._acc, self._frames, None, 0
        if acc is None:
            return
        side = acc.shape[-1]
        if side > self.max_side:                      # sum blocks: 320 -> 80 with blocks of 4
            b = -(-side // self.max_side)
            side = side // b
            acc = acc[:, :side * b, :side * b].reshape(2, side, b, side, b).sum(axis=(2, 4))
        ch0 = np.clip(acc[0] * self.gain / 255.0, 0, 1)[..., None]
        ch1 = np.clip(acc[1] * self.gain / 255.0, 0, 1)[..., None]
        red, blue = np.array([255, 70, 70], np.float32), np.array([90, 200, 255], np.float32)
        img = np.clip(red * ch0 + blue * ch1, 0, 255).astype(np.uint8)
        if self.scale > 1:
            img = img.repeat(self.scale, 0).repeat(self.scale, 1)
        h, w = img.shape[:2]
        for k in range(1, self.zone_columns):        # the network's zones
            img[:, k * w // self.zone_columns, :] = (90, 90, 255)
        for k in range(1, self.zone_rows):
            img[k * h // self.zone_rows, :, :] = (90, 90, 255)
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"despot {n} frames"
        msg.height, msg.width, msg.encoding, msg.step = img.shape[0], w, "rgb8", w * 3
        msg.data = array.array("B", img.tobytes())     # one copy; a bytes/list is checked per element
        self.pub.publish(msg)

    def destroy_node(self):
        self._stop.set()
        return super().destroy_node()


def main():
    rclpy.init()
    node = CameraTapNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
