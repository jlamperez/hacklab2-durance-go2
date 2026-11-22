"""FPGA perception node: the target-zone SNN's output from SPLEAT, as reflex decisions.

  reads       TCP host:port (spleat-server / backward-server on the Kria, default 127.0.0.1:3333):
              one message per camera frame, the network's output spikes (4x1x1 for models/zones3,
              7x1x1 for models/ball_v3)
  publishes   /reflex/decision          reflex_msgs/ReflexDecision, one per FPGA message
              /durance/ball_zone        std_msgs/String, for 6-zone networks (models/ball_v3): the
                                        zone (upper_left ... lower_right, none), majority-voted,
                                        only when it changes, reliable + transient-local: what the
                                        follow node reads with mode:=look input:=zone
              /reflex/debug/fpga        std_msgs/String, JSON: values, window counts, latency

It takes the place of the `perception` node when the network runs on the FPGA. The camera
client on the board (despot_send_and_forget) feeds the server; this node only listens, like
Durance's gesture_node. The readout (reflex/runtime/zones_readout.py) sums each neuron's
spikes over the last `readout` frames and takes the most, ties to none: the rule the network
was trained and validated with.

The winning class is mapped to the behavior node's labels by column (`reflex_label`):
upper_left / lower_left -> left, *_middle -> center, *_right -> right. zones3's names pass
unchanged. The class name itself is in /reflex/debug/fpga.

confidence = the winner's spikes in the window / readout (the share of frames it fired in).
On 22 Nov, with balls thrown past the camera, zone decisions reached 0.5-0.55; the behavior
node's min_confidence must sit below that (the board launch file sets it). Not validated.

Parameters: zone_topic (/durance/ball_zone, '' to skip), vote_window (5), vote_min (3),
zone_hold_s (1.0: a zone stays published this long after its last vote before `none` is sent;
the network fires in short bursts, 0.1-0.2 s, too short for the follow node's gestures);
host, port, model_dir (meta.json or *_readout.json, + config.yaml of the deployed
network), readout (frames per decision, 0: the model folder's value), margin (a zone must beat
none by this many spikes), reconnect_s.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from reflex.runtime.spleat_client import ProtocolError, parse_message, read_message, server_shape
from reflex.runtime.zone_target import ZONES
from reflex.runtime.zone_vote import ZoneVoteConfig, ZoneVoter
from reflex.runtime.zones_readout import ZonesReadout, reflex_label
from reflex_msgs.msg import ReflexDecision


class FpgaPerceptionNode(Node):
    def __init__(self):
        super().__init__("reflex_fpga_perception")
        p = self.declare_parameter
        self.host = p("host", "127.0.0.1").value
        self.port = int(p("port", 3333).value)
        model_dir = Path(p("model_dir", "models/zones3").value)
        self.readout = ZonesReadout.from_model_dir(model_dir, float(p("margin", 0.0).value),
                                                   int(p("readout", 0).value))
        self.shape = server_shape(model_dir / "config.yaml")
        self.reconnect_s = float(p("reconnect_s", 1.0).value)
        self.pub = self.create_publisher(ReflexDecision, "/reflex/decision", 10)
        self.debug_pub = self.create_publisher(String, "/reflex/debug/fpga", 10)
        zone_topic = p("zone_topic", "/durance/ball_zone").value
        self.zone_pub, self.voter = None, None
        if zone_topic and set(self.readout.names) == set(ZONES) | {"none"}:
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.zone_pub = self.create_publisher(String, zone_topic, qos)
            self.voter = ZoneVoter(ZoneVoteConfig(int(p("vote_window", 5).value), int(p("vote_min", 3).value)))
            self.zone_pub.publish(String(data="none"))
            self.zone_hold_s = float(p("zone_hold_s", 1.0).value)
            self._zone, self._zone_t = "none", 0.0
            self.get_logger().info(f"6-zone network: zones on {zone_topic} "
                                   f"(vote {self.voter.cfg.min_votes} of {self.voter.cfg.window})")
        self._stop = threading.Event()
        self._sock = None
        self._n = 0
        self.get_logger().info(
            f"SPLEAT at {self.host}:{self.port}, classes {', '.join(self.readout.names)}, "
            f"readout {self.readout.readout} frames, margin {self.readout.margin:g}, "
            f"expected output {self.shape}")
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def run(self) -> None:
        while not self._stop.is_set() and rclpy.ok():
            try:
                with socket.create_connection((self.host, self.port), timeout=5) as sock:
                    # Blocking reads: a timeout in the middle of a message would lose bytes and
                    # desynchronise the stream. destroy_node() closes the socket to stop.
                    sock.settimeout(None)
                    self._sock = sock
                    self.get_logger().info("connected; waiting for frames (despot must run)")
                    self.readout.reset()
                    while not self._stop.is_set():
                        self.on_message(read_message(sock))
            except (OSError, ProtocolError) as e:
                if self._stop.is_set():
                    return
                self.get_logger().warn(f"SPLEAT connection: {e}; retrying in {self.reconnect_s:g} s",
                                       throttle_duration_sec=10.0)
                time.sleep(self.reconnect_s)

    def on_message(self, body: bytes) -> None:
        m = parse_message(body, self.shape)
        if m.get("error") or not m["shape_ok"]:
            self.get_logger().warn(f"message {m.get('id')}: error {m.get('error')!r}, "
                                   f"{m['n_values']} values", throttle_duration_sec=5.0)
            return
        d = self.readout.push(m["values"])
        if self.voter is not None:
            self.publish_zone(self.voter.update(d.label))
        out = ReflexDecision()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "fpga"
        out.label = reflex_label(d.label)
        out.confidence = float(d.counts[d.index] / self.readout.readout)
        out.detector = "fpga_zones"
        latency = m.get("latency")
        out.inference_ms = float(latency) / 1000.0 if isinstance(latency, (int, float)) else 0.0
        self.pub.publish(out)
        self._n += 1
        if self.debug_pub.get_subscription_count() > 0:
            self.debug_pub.publish(String(data=json.dumps({
                "id": m.get("id"), "values": [float(v) for v in m["values"].reshape(-1)],
                "counts": {n: int(c) for n, c in zip(self.readout.names, d.counts)},
                "class": d.label, "label": out.label, "confidence": round(out.confidence, 3),
                "latency_us": latency, "frames": d.frames})))
        if d.index != self.readout.none_index:
            self.get_logger().info(self.readout.format(d), throttle_duration_sec=0.2)

    def publish_zone(self, changed: str | None) -> None:
        """A new zone at once; `none` only after zone_hold_s without a zone."""
        now = time.monotonic()
        if self.voter.current != "none":
            self._zone_t = now                           # the vote still holds a zone
        if changed is not None and changed != "none" and changed != self._zone:
            self._zone = changed
        elif self._zone != "none" and self.voter.current == "none" and now - self._zone_t >= self.zone_hold_s:
            self._zone = "none"
        else:
            return
        self.zone_pub.publish(String(data=self._zone))
        self.get_logger().info(f"zone -> {self._zone}")

    def destroy_node(self):
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        return super().destroy_node()


def main():
    rclpy.init()
    node = FpgaPerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
