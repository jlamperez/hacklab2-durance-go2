"""Camera tap: what the FPGA network sees, as a picture, without changing what it gets.

    despot_send_and_forget --TCP 3333--> camera_tap --TCP 3334--> spleat-server (FPGA)
                                            |
                                            +--> /reflex/debug/events_image  sensor_msgs/Image (rgb8)

On the Kria. despot always sends to 127.0.0.1:3333, so the server moves to 3334 (run.sh sets
it) and this node takes 3333. Every message is forwarded unchanged, byte for byte, as soon as
it is read; the server's replies on this connection are read and dropped (despot never reads
them either). fpga_perception reads the results from the server directly, on 3334.

The picture: the frames received since the last one summed, in the repo's colours
(reflex/events/repr.py): channel 0 red, channel 1 light blue, both white-ish (which channel is
ON is not documented by Durance; the repo calls 1 ON), scaled up `scale` times, with the image thirds marked:
the network's zones left / center / right, as the camera sees them. Published at `image_hz` (10), 80x80 (`scale` 1).
The web dashboard (ros-web) shows it under "Event camera".

Parameters: listen_port (3333), server_host (127.0.0.1), server_port (3334), image_hz (10),
scale (1), gain (64: brightness per event).
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

from reflex.runtime.spleat_client import cbor_decode, read_exact


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
                    threading.Thread(target=self.drain, args=(up,), daemon=True).start()
                    while not self._stop.is_set():
                        head = read_exact(client, 4)
                        (n,) = struct.unpack(">I", head)
                        body = read_exact(client, n)
                        up.sendall(head + body)          # unchanged, first
                        self.add_frame(body)
            except (OSError, ValueError) as e:
                self.get_logger().warn(f"camera link closed: {e}")

    @staticmethod
    def drain(sock: socket.socket) -> None:
        try:
            while sock.recv(65536):
                pass
        except OSError:
            pass

    def add_frame(self, body: bytes) -> None:
        try:
            payload = cbor_decode(body).get("payload")
            v = np.frombuffer(payload, "<f4")
            side = int(round((v.size / 2) ** 0.5))
            frame = v.reshape(2, side, side)
        except Exception:                          # noqa: BLE001 - the picture is best effort
            return
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
        ch0 = np.clip(acc[0] * self.gain / 255.0, 0, 1)[..., None]
        ch1 = np.clip(acc[1] * self.gain / 255.0, 0, 1)[..., None]
        red, blue = np.array([255, 70, 70], np.float32), np.array([90, 200, 255], np.float32)
        img = np.clip(red * ch0 + blue * ch1, 0, 255).astype(np.uint8)
        if self.scale > 1:
            img = img.repeat(self.scale, 0).repeat(self.scale, 1)
        w = img.shape[1]
        for x in (w // 3, 2 * w // 3):               # the network's zones, image thirds
            img[:, x, :] = (90, 90, 255)
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
