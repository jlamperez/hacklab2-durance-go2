"""Live dashboard in the browser, built on viser.

    pixi run -e ros ros-web            # then open http://localhost:8080
    pixi run -e ros ros-web port:=8081

It only listens: start it next to any of the launch files, on any machine of the ROS 2
network. The simulation is not run here; the bridge sends where each body is and the
browser draws it in 3D (drag to orbit, scroll to zoom).

  /reflex/sim/state             body poses from go2_sim_bridge -> 3D scene (empty with the real robot)
  /reflex/debug/events_image    events as a picture
  /reflex/debug/perception      class probabilities, firing rate per layer, input activity
  /reflex/decision              label and confidence (used when the debug topic is silent)
  /reflex/debug/behavior        reflex state and last request
  /api/sport/request            requests going to the robot
  /reflex/debug/latency_ms      newest event -> request, measured by the simulator bridge
  /reflex/sim/outcome           score of every shot
"""
import json
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, String

from reflex.runtime import sport_api
from reflex.runtime.detectors import LABELS
from reflex.viz.web_scene import body_meshes, unpack_state, world_meshes
from reflex_msgs.msg import ReflexDecision
from unitree_api.msg import Request

HISTORY = 300          # points kept in the plots
STATE_COLOURS = {"idle": "gray", "reacting": "red", "refractory": "yellow"}
OUTCOMES = ("correct", "wrong", "missed", "false_trigger", "quiet")


class WebViewer(Node):
    def __init__(self):
        super().__init__("reflex_web_viewer")
        import viser
        import viser.uplot as uplot
        port = int(self.declare_parameter("port", 8080).value)
        seed = int(self.declare_parameter("seed", 0).value)     # same seed as the bridge: same clutter
        self.server = viser.ViserServer(host="0.0.0.0", port=port, verbose=False)
        self.server.gui.configure_theme(control_layout="fixed", control_width="large", dark_mode=True)
        self.server.gui.set_panel_label("Go2 reflexes")
        self._build_scene(seed, self.declare_parameter("scene", "lab").value)   # as the bridge's
        gui = self.server.gui

        with gui.add_folder("Event camera"):
            self.image = gui.add_image(np.zeros((90, 160, 3), dtype=np.uint8), format="jpeg", jpeg_quality=80)
            self.activity = gui.add_markdown("waiting for `/reflex/debug/events_image`")
        with gui.add_folder("Decision"):
            self.decision = gui.add_markdown("waiting for `/reflex/decision`")
            self.bars = {}
            for label in LABELS:
                gui.add_markdown(f"{label}")
                self.bars[label] = gui.add_progress_bar(0.0, color="gray" if label == "none" else "orange")
        with gui.add_folder("Reflex"):
            self.reflex = gui.add_markdown("waiting for `/reflex/debug/behavior`")
            self.reflex_bar = gui.add_progress_bar(100.0, color="gray")
            self.requests = gui.add_markdown("no request yet")
        with gui.add_folder("Latency: newest event to request"):
            self.latency = gui.add_markdown("no reaction yet")
            x = np.arange(HISTORY, dtype=np.float64)
            self.latency_plot = gui.add_uplot(
                data=(x, np.full(HISTORY, np.nan)), aspect=2.4,
                series=(uplot.Series(label="reaction"), uplot.Series(label="ms", stroke="#f59f00", width=2,
                                                                     points={"show": True})),
                scales={"x": uplot.Scale(time=False)})
        with gui.add_folder("Network activity (sparsity)"):
            self.sparsity = gui.add_markdown("waiting for `/reflex/debug/perception`")
            self.rate_plot = gui.add_uplot(
                data=(x, *(np.full(HISTORY, np.nan) for _ in range(4))), aspect=2.0,
                series=(uplot.Series(label="window"), uplot.Series(label="input %", stroke="#868e96", width=1),
                        uplot.Series(label="layer 1 %", stroke="#4dabf7", width=2),
                        uplot.Series(label="layer 2 %", stroke="#f59f00", width=2),
                        uplot.Series(label="layer 3 %", stroke="#ff6b6b", width=2)),
                scales={"x": uplot.Scale(time=False)})
        with gui.add_folder("Score (simulated shots)"):
            self.score = gui.add_markdown("no shot scored yet")

        self._img = self._state = self._perception = self._behavior = self._decision = None
        self._dirty = set()
        self._lat = deque(maxlen=HISTORY)
        self._rates = deque(maxlen=HISTORY)
        self._requests = deque(maxlen=4)
        self._outcomes = dict.fromkeys(OUTCOMES, 0)
        self._last_perception_t = 0.0
        sub = self.create_subscription
        sub(Image, "/reflex/debug/events_image", self.on_image, 2)
        sub(Float32MultiArray, "/reflex/sim/state", self.on_state, 2)
        sub(String, "/reflex/debug/perception", self.on_perception, 10)
        sub(ReflexDecision, "/reflex/decision", self.on_decision, 10)
        sub(String, "/reflex/debug/behavior", self.on_behavior, 10)
        sub(Request, sport_api.SPORT_REQUEST_TOPIC, self.on_request, 10)
        sub(Float32, "/reflex/debug/latency_ms", self.on_latency, 10)
        sub(String, "/reflex/sim/outcome", self.on_outcome, 10)
        self.create_timer(1 / 20, self.refresh)
        actual = self.server.get_port()      # viser moves to the next free port if this one is taken
        if actual != port:
            self.get_logger().warn(f"port {port} is busy (another dashboard running?)")
        self.get_logger().info(f"dashboard at http://localhost:{actual} (other machines: http://<this-pc>:{actual})")

    # --- 3D scene ------------------------------------------------------------------------
    def _build_scene(self, seed: int, scene_name: str) -> None:
        from reflex.sim.scene import build_model
        scene = self.server.scene
        scene.set_up_direction("+z")
        model = build_model(np.random.default_rng(seed), scene_name)
        self.handles = {b: scene.add_mesh_trimesh(f"/body/{model.body(b).name or b}", mesh)
                        for b, mesh in body_meshes(model).items()}
        self.scene_note = scene.add_label("/note", "no simulator: 3D scene idle", position=(0.0, 0.0, 0.8))
        if scene_name == "park":     # textured scenery and a sky instead of the lab's grid
            del self.handles[0]
            scene.remove_by_name("/body/world")
            for name, mesh in world_meshes(model).items():
                scene.add_mesh_trimesh(f"/scenery/{name}", mesh)
            sky = np.linspace([115, 166, 235], [204, 222, 242], 64).astype(np.uint8)[:, None, :].repeat(64, 1)
            scene.set_background_image(sky)
        else:
            self._lab_floor(scene)

        @self.server.on_client_connect
        def _(client) -> None:       # same viewpoint as the recorded videos
            park = scene_name == "park"      # further back, clear of the goal behind the dog
            client.camera.position = (-2.3, -1.5, 1.3) if park else (-1.0, -1.7, 0.9)
            client.camera.look_at = (1.5, 0.2, 0.3) if park else (0.8, 0.1, 0.2)

    @staticmethod
    def _lab_floor(scene) -> None:
        scene.add_grid("/floor", width=20.0, height=20.0, plane="xy", cell_size=0.5, section_size=1.0,
                       plane_color=(110, 112, 108),
                       plane_opacity=1.0, cell_color=(90, 90, 90), section_color=(70, 70, 70))

    # --- topics: keep the newest, draw in refresh() --------------------------------------
    def on_image(self, msg: Image) -> None:
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self._img = img[..., ::-1] if msg.encoding == "bgr8" else img
        self._dirty.add("image")

    def on_state(self, msg: Float32MultiArray) -> None:
        self._state = unpack_state(np.frombuffer(msg.data, dtype=np.float32))
        self._dirty.add("state")

    def on_perception(self, msg: String) -> None:
        self._perception = json.loads(msg.data)
        self._last_perception_t = time.monotonic()
        rates = self._perception.get("firing_rate")
        self._rates.append([100 * self._perception.get("input_active", 0.0)]
                           + ([100 * r for r in rates[:3]] + [np.nan] * 3)[:3] if rates else
                           [100 * self._perception.get("input_active", 0.0), np.nan, np.nan, np.nan])
        self._dirty.update(("perception", "rates"))

    def on_decision(self, msg: ReflexDecision) -> None:
        self._decision = msg
        if time.monotonic() - self._last_perception_t > 1.0:     # debug topic silent (older perception node)
            self._perception = {"detector": msg.detector, "label": msg.label, "confidence": msg.confidence,
                                "inference_ms": msg.inference_ms}
            self._dirty.add("perception")

    def on_behavior(self, msg: String) -> None:
        self._behavior = json.loads(msg.data)
        self._dirty.add("behavior")

    def on_request(self, msg: Request) -> None:
        api_id = int(msg.header.identity.api_id)
        name = sport_api.BY_ID[api_id].name if api_id in sport_api.BY_ID else f"api_id {api_id}"
        self._requests.appendleft(f"`{name}` {msg.parameter}".strip())
        self._dirty.add("requests")
        # No simulator measuring it: use this machine's clock (needs clocks in sync, e.g. NTP).
        stamp_us = int(msg.header.identity.id)
        if stamp_us > 0 and self.count_publishers("/reflex/debug/latency_ms") == 0:
            ms = (self.get_clock().now().nanoseconds / 1000 - stamp_us) * 1e-3
            if 0.0 <= ms < 2000.0:
                self._add_latency(ms)

    def on_latency(self, msg: Float32) -> None:
        self._add_latency(float(msg.data))

    def _add_latency(self, ms: float) -> None:
        self._lat.append(ms)
        self._dirty.add("latency")

    def on_outcome(self, msg: String) -> None:
        if not msg.data.startswith("{"):      # the final "SUMMARY ..." line
            return
        outcome = json.loads(msg.data).get("outcome")
        if outcome in self._outcomes:
            self._outcomes[outcome] += 1
            self._dirty.add("score")

    # --- drawing, 20 times a second ------------------------------------------------------
    def refresh(self) -> None:
        dirty, self._dirty = self._dirty, set()
        if "state" in dirty:
            _, pos, quat = self._state
            self.scene_note.visible = False
            with self.server.atomic():
                for b, handle in self.handles.items():
                    if b < len(pos):
                        handle.position, handle.wxyz = pos[b], quat[b]
        if "image" in dirty:
            self.image.image = self._img
        if "perception" in dirty:
            p = self._perception
            scores = p.get("probs") or p.get("scores") or {p["label"]: p["confidence"]}
            for label, bar in self.bars.items():
                bar.value = float(np.clip(100 * scores.get(label, 0.0), 0, 100))
                if label != "none":
                    bar.color = "red" if p["label"] == label else "orange"
            what = "probability" if "probs" in p else "events / threshold" if "scores" in p else "confidence"
            self.decision.content = (f"**{p['label'].upper()}** {100 * p['confidence']:.0f} %  ·  detector "
                                     f"`{p['detector']}`  ·  {p['inference_ms']:.1f} ms  \nbars: {what}")
            if "events" in p:
                self.activity.content = f"{p['events']} events in the window · {100 * p['input_active']:.2f} % of cells active"
        if "rates" in dirty and len(self._rates) > 1:
            r = np.array(self._rates, dtype=np.float64)
            x = np.arange(len(r), dtype=np.float64)
            self.rate_plot.data = (x, r[:, 0], r[:, 1], r[:, 2], r[:, 3])
            last = r[-1]
            self.sparsity.content = ("the simple detector has no neurons; only the input activity is shown"
                                     if np.isnan(last[1]) else
                                     f"neurons firing per time step: layer 1 **{last[1]:.2f} %**, layer 2 "
                                     f"**{last[2]:.2f} %**, layer 3 **{last[3]:.2f} %**  ·  input {last[0]:.2f} %")
        if "behavior" in dirty:
            b = self._behavior
            self.reflex.content = (f"**{b['state'].upper()}**  ·  pose `{b['pose']}`  ·  mode `{b['mode']}`"
                                   f"  ·  {b['reactions']} reactions" + ("" if b["enabled"] else "  ·  **DISABLED**"))
            self.reflex_bar.color = STATE_COLOURS.get(b["state"], "gray")
        if "requests" in dirty:
            self.requests.content = "requests to the robot, newest first:  \n" + "  \n".join(self._requests)
        if "latency" in dirty:
            lat = np.array(self._lat, dtype=np.float64)
            self.latency.content = (f"last **{lat[-1]:.0f} ms**  ·  median {np.median(lat):.0f} ms  ·  "
                                    f"max {lat.max():.0f} ms  ·  {len(lat)} requests")
            if len(lat) > 1:
                self.latency_plot.data = (np.arange(len(lat), dtype=np.float64), lat)
        if "score" in dirty:
            o = self._outcomes
            self.score.content = (f"correct **{o['correct']}**  ·  wrong side {o['wrong']}  ·  missed {o['missed']}"
                                  f"  ·  false triggers {o['false_trigger']}  ·  quiet negatives {o['quiet']}")


def main():
    rclpy.init()
    node = WebViewer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
