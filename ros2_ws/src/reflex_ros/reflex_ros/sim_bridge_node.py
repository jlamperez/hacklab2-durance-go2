"""Simulated Go2: MuJoCo behind the same topics as the real robot and camera.

  subscribes  /api/sport/request      unitree_api/Request      (what the real Go2 listens to)
  publishes   /reflex/events_window   reflex_msgs/EventWindow  (what the camera adapter will publish)
              /reflex/debug/events_image  sensor_msgs/Image, the events drawn as a picture (for people)
              /clock                  simulated time
              /reflex/sim/outcome     std_msgs/String, one JSON line per shot

With robot_only it is just the robot (no camera, no balls, no /clock). Adding camera:=true
to that keeps the dog's simulated event camera on, e.g. to watch the events while driving.

It rolls balls (and hard negatives) at the dog, applies incoming requests after a
configurable actuation delay, and scores each shot. The behavior node cannot tell this
apart from the robot, which is the point: it is that node's test bench.
"""
from array import array
import json
import os
from collections import deque

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock as ClockMsg
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, String

from reflex.data.episodes import DOG_FRONT_X, THREAT_NEAR
from reflex.data.windows import WindowConfig, make_window
from reflex.events import repr as ev_repr
from reflex.events.emu import EmuConfig, EventEmulator
from reflex.runtime import sport_api
from reflex.runtime.reflex_fsm import BEHAVIORS
from reflex.runtime.sim_sport import PostureMode, pose_for_request
from reflex.sim.scene import BallShot, Go2Sim, to_gray
from reflex.viz.web_scene import pack_state
from reflex_msgs.msg import EventWindow
from unitree_api.msg import Request

WINDOW_STRIDE_STEPS = 10     # one window every 20 ms of simulated time
STATE_STRIDE_STEPS = 16      # body poses for the web viewer, about 30 per second
NEGATIVES = ("miss", "crossing", "walker")
# The walking policy covers about 75 % of the distance the real Go2 did for the same Move
# (y=0.3 for 1 s: 16 cm against 22 cm, 21 Nov 2026). Its x/y commands are scaled up by this.
RL_TRACKING_GAIN = 1.35


class SimBridge(Node):
    def __init__(self):
        super().__init__("go2_sim_bridge")
        p = self.declare_parameter
        # Default measured on the real Go2 (21 Nov 2026): a lean starts 70 ms after the command.
        # With it the simulated lean is 10/50/90 % done at 82/134/310 ms; the real one at 107/180/273.
        delay_ms = float(p("actuation_delay_ms", 70.0).value)
        if not 20.0 <= delay_ms <= 400.0:
            self.get_logger().warn(f"actuation_delay_ms={delay_ms:.0f} is outside the 20-400 ms range "
                                   "plausible for the real Go2")
        self.delay_s = delay_ms / 1e3
        self.period = float(p("shot_period_s", 3.0).value)
        self.neg_frac = float(p("negative_fraction", 0.3).value)
        self.max_shots = int(p("max_shots", 0).value)          # 0: run until stopped
        self.mode = p("mode", "keeper").value                  # only used to score reactions
        self.record = p("record", "").value                    # mp4 path, empty: no video
        seed = int(p("seed", 0).value)
        randomize = bool(p("randomize", False).value)          # camera mount, shake, sensor
        # robot_only: act as the robot and nothing else. No simulated camera, no balls, no
        # /clock: events come from somewhere else (a webcam node, a Kria board) in real time.
        self.robot_only = bool(p("robot_only", False).value)
        self.use_viewer = bool(p("viewer", False).value)       # MuJoCo 3D window (needs MUJOCO_GL=glfw)
        # camera: in robot_only mode, also publish what the dog's simulated event camera sees
        # (for driving it around and watching the events). Always on outside robot_only.
        self.camera = bool(p("camera", False).value) or not self.robot_only
        self.viewer = None
        # locomotion: "none" = postures only (default). "rl" = a Move request makes the dog
        # really walk, using the learned policy in models/go2_walk.onnx.
        self.locomotion = p("locomotion", "none").value
        scene = p("scene", "lab").value                        # "lab" or "park" (reflex/sim/scenery.py)
        self.homing = False

        self.rng = np.random.default_rng(seed)
        self.sim = Go2Sim(seed=seed, scene=scene)
        self.wcfg = WindowConfig()
        if self.locomotion == "rl":
            self.sim.enable_walking()
        elif self.locomotion != "none":
            raise ValueError(f"unknown locomotion {self.locomotion!r}")
        if randomize:
            self.sim.randomize_camera(self.rng)
            self.sim.set_shake(self.rng, float(self.rng.uniform(0.2, 1.5)))
        h, w = self.sim.eye_hw
        self.emu = EventEmulator(h, w, EmuConfig.randomized(self.rng) if randomize else EmuConfig(), self.rng)
        if self.camera:
            self.emu.reset(to_gray(self.sim.render_eye()), 0)
        self.recent = deque(maxlen=round(self.wcfg.window_ms * 1e-3 / self.sim.dt))

        # As the robot only, it publishes no camera data and no clock: creating those
        # publishers anyway would draw it as a second camera in rqt_graph.
        self.clock_pub = self.win_pub = self.img_pub = None
        if not self.robot_only:
            self.clock_pub = self.create_publisher(ClockMsg, "/clock", 10)
        if self.camera:
            self.win_pub = self.create_publisher(EventWindow, "/reflex/events_window", qos_profile_sensor_data)
            self.img_pub = self.create_publisher(Image, "/reflex/debug/events_image", 2)   # reliable: any viewer can subscribe
        self.out_pub = self.create_publisher(String, "/reflex/sim/outcome", 10)
        # For the web viewer: where every body is (reflex/viz/web_scene.py), and each latency.
        self.state_pub = self.create_publisher(Float32MultiArray, "/reflex/sim/state", 2)
        self.latency_pub = self.create_publisher(Float32, "/reflex/debug/latency_ms", 10)
        self.create_subscription(Request, sport_api.SPORT_REQUEST_TOPIC, self.on_request, 10)

        self.pending = []            # (apply_time_s, pose, api name, velocity or None)
        self.posture = PostureMode()   # Euler acts only after Pose(true), Move only without it
        self.moved = False           # the dog has walked since it was last on its spot
        self.pose = "stand"
        self.steps = 0
        self.shot = None             # dict describing the shot in flight
        self.next_shot_t = float("inf") if self.robot_only else 2.0   # quiet start: the simple detector calibrates
        self.results = []
        self.latencies_ms = []
        self.writer = None
        if self.use_viewer:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data)
            self.viewer.cam.lookat[:] = [0.0, 0.0, 0.25]
            self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = 2.2, 150, -15
            if self.locomotion == "rl":      # follow the dog
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = self.sim.model.body("base").id
                self.viewer.cam.distance = 3.0
        if self.record and not self.robot_only:
            import imageio.v2 as imageio
            self.writer = imageio.get_writer(self.record, fps=50, codec="libx264", quality=8, macro_block_size=2)
        # Wall-clock timer: this node produces /clock, so it cannot run on it.
        self.timer = self.create_timer(self.sim.dt, self.step, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(f"actuation delay {delay_ms:.0f} ms, shot every {self.period:.1f} s, "
                               f"{self.neg_frac:.0%} negatives, scoring as {self.mode}")

    # --- requests from the behavior node -------------------------------------------------
    def on_request(self, msg: Request) -> None:
        api_id = int(msg.header.identity.api_id)
        name = sport_api.BY_ID[api_id].name if api_id in sport_api.BY_ID else f"api_id {api_id}"
        if not self.posture.handle(api_id, msg.parameter):
            if api_id != sport_api.POSE.api_id:
                self.get_logger().info(f"{name}: accepted but no effect with posture mode "
                                       f"{'on' if self.posture.on else 'off'}, as on the real Go2")
            return
        pose = pose_for_request(api_id, msg.parameter)
        if pose is None:
            self.get_logger().warn(f"{name}: no equivalent in the simulator, ignored")
            return
        if self.homing:      # the dog is walking back to its spot between shots
            return
        velocity = None
        if api_id == sport_api.MOVE.api_id:
            v = json.loads(msg.parameter)
            velocity = (float(v.get("x", 0.0)), float(v.get("y", 0.0)), float(v.get("z", 0.0)))
        elif api_id == sport_api.STOP_MOVE.api_id:
            velocity = (0.0, 0.0, 0.0)
        stamp_us = int(msg.header.identity.id)
        if stamp_us > 0:   # time from the newest event to the request arriving here
            now_us = self.get_clock().now().nanoseconds / 1000 if self.robot_only else self.sim.time * 1e6
            self.latencies_ms.append((now_us - stamp_us) * 1e-3)
            self.latency_pub.publish(Float32(data=float(self.latencies_ms[-1])))
            if self.robot_only:
                self.get_logger().info(f"{name} -> {pose}; {self.latencies_ms[-1]:.0f} ms from newest event to request, "
                                       f"applied after {self.delay_s * 1e3:.0f} ms more")
        self.pending.append((self.sim.time + self.delay_s, pose, name, velocity))

    def apply_due(self) -> None:
        while self.pending and self.pending[0][0] <= self.sim.time:
            _, pose, name, velocity = self.pending.pop(0)
            if velocity is not None and self.locomotion == "rl":
                if not (self.sim.mode == "pose" and not any(velocity)):   # StopMove while posing: nothing to do
                    vx, vy, wz = velocity
                    self.sim.damped = False
                    self.sim.set_velocity(vx * RL_TRACKING_GAIN, vy * RL_TRACKING_GAIN, wz)
                    self.moved = True
            elif velocity is not None:
                # No walking policy: the dog keeps its posture and is moved as a whole with the
                # real Go2's sidestep profile (Go2Sim.set_slide).
                self.sim.damped = False
                self.sim.set_slide(*velocity)
                self.moved = self.moved or any(velocity)
            elif pose == "damp":
                self.sim.damped = True
            elif pose in ("sit", "hello"):
                self.sim.damped = False
                self.sim.play(pose)
            else:
                self.sim.damped = False
                self.sim.set_pose(pose)
            self.pose = pose
            if self.shot is not None and pose != "stand" and self.shot["reaction"] is None:
                self.shot["reaction"], self.shot["reaction_t"] = pose, self.sim.time

    # --- shots ---------------------------------------------------------------------------
    def start_shot(self) -> None:
        rng, t = self.rng, self.sim.time
        kind = str(rng.choice(NEGATIVES)) if rng.random() < self.neg_frac else "approach"
        shot = {"kind": kind, "label": "none", "t0": t, "reaction": None, "reaction_t": None, "walker": None}
        if kind in ("approach", "miss"):
            ball = BallShot.sample(rng, label=None if kind == "approach" else "none")
            self.sim.launch(ball)
            shot.update(label=ball.label, speed=ball.speed, t_end=t + 4.0)
        elif kind == "crossing":
            side = rng.choice([-1, 1])
            vy = -side * float(rng.uniform(2.0, 6.0))
            self.sim.launch_raw((float(rng.uniform(1.0, 4.0)), side * 2.6), (0.0, vy),
                                float(rng.uniform(0.05, 0.12)), rng.uniform(0.15, 1.0, 3))
            shot["t_end"] = t + 5.2 / abs(vy)
        else:
            side = rng.choice([-1, 1])
            shot["walker"] = (float(rng.uniform(1.5, 5.0)), side * 3.2, -side * float(rng.uniform(0.8, 1.8)))
            shot["t_end"] = t + 2.0
        self.shot = shot

    def finish_shot(self) -> None:
        s, t = self.shot, self.sim.time
        if s["kind"] == "approach":
            expected = BEHAVIORS[self.mode][s["label"]]
            outcome = "missed" if s["reaction"] is None else ("correct" if s["reaction"] == expected else "wrong")
        else:
            outcome = "quiet" if s["reaction"] is None else "false_trigger"
        res = {"kind": s["kind"], "label": s["label"], "reaction": s["reaction"], "outcome": outcome}
        if s["reaction_t"] is not None and s["kind"] == "approach":
            res["lead_ms"] = round((t - s["reaction_t"]) * 1e3)   # reaction started this long before arrival
        self.results.append(res)
        self.out_pub.publish(String(data=json.dumps(res)))
        self.get_logger().info(f"shot {len(self.results)}: {json.dumps(res)}")
        self.sim.park_ball()
        self.sim.park_walker()
        self.shot = None
        self.next_shot_t = t + self.period
        if self.max_shots and len(self.results) >= self.max_shots:
            self.summary()
            if self.writer is not None:
                self.writer.close()
            raise SystemExit(0)

    def summary(self) -> None:
        r = self.results
        balls = [x for x in r if x["kind"] == "approach"]
        negs = [x for x in r if x["kind"] != "approach"]
        count = lambda xs, o: sum(x["outcome"] == o for x in xs)   # noqa: E731
        leads = [x["lead_ms"] for x in balls if "lead_ms" in x]
        out = {
            "actuation_delay_ms": round(self.delay_s * 1e3), "balls": len(balls),
            "correct": count(balls, "correct"), "wrong_side": count(balls, "wrong"), "missed": count(balls, "missed"),
            "negatives": len(negs), "false_triggers": count(negs, "false_trigger"),
            "median_lead_ms": float(np.median(leads)) if leads else None,
            "median_event_to_request_ms": float(np.median(self.latencies_ms)) if self.latencies_ms else None,
        }
        self.get_logger().info("SUMMARY " + json.dumps(out))
        self.out_pub.publish(String(data="SUMMARY " + json.dumps(out)))

    # --- main loop -----------------------------------------------------------------------
    def step(self) -> None:
        sim = self.sim
        if self.shot is None and not self.robot_only and self.moved and self.pose == "stand" and not self.pending:
            # Between shots: walk back to the spot the balls are aimed at, then stand as before.
            self.homing = True
            if sim.walk_towards(0.0, 0.0, 0.0):
                sim.set_pose("stand")
                self.homing = self.moved = False
                self.next_shot_t = max(self.next_shot_t, sim.time + 1.5)
        if self.shot is None and sim.time >= self.next_shot_t and self.pose == "stand" and not self.moved:
            self.start_shot()
        self.apply_due()
        if self.shot is not None and self.shot["walker"] is not None:
            x, y0, vy = self.shot["walker"]
            sim.set_walker(x, y0 + vy * (sim.time - self.shot["t0"]))
        sim.step()
        self.steps += 1
        if self.steps % STATE_STRIDE_STEPS == 0 and self.state_pub.get_subscription_count() > 0:
            self.state_pub.publish(Float32MultiArray(data=array("f", pack_state(sim.model, sim.data).tobytes())))
        if self.robot_only and sim.mode == "walk" and self.steps % 1000 == 0:
            x, y, h = sim.base_pose()
            self.get_logger().info(f"dog at x {x:+.2f} m, y {y:+.2f} m, heading {np.degrees(h):+.0f} deg; "
                                   f"command {sim.command.round(2).tolist()}")
        if self.viewer is not None and self.steps % 8 == 0:
            if not self.viewer.is_running():
                os._exit(0)      # window closed; MuJoCo's viewer teardown is unsafe, leave directly
            self.viewer.sync()
        if not self.camera:
            return
        eye = sim.render_eye()
        t_us = int(sim.time * 1e6)
        self.recent.append(self.emu.step(to_gray(eye), t_us))

        if self.clock_pub is not None and self.steps % 5 == 0:
            msg = ClockMsg()
            msg.clock.sec, msg.clock.nanosec = t_us // 1_000_000, (t_us % 1_000_000) * 1000
            self.clock_pub.publish(msg)
        if self.steps % WINDOW_STRIDE_STEPS == 0:
            self.publish_window(t_us)
            if self.writer is not None:
                self.write_frame(eye)

        if self.shot is not None:
            s = self.shot
            arrived = s["kind"] == "approach" and sim.ball_xy()[0] - DOG_FRONT_X < THREAT_NEAR
            if arrived or sim.time >= s["t_end"]:
                self.finish_shot()

    def publish_window(self, t_us: int) -> None:
        cfg = self.wcfg
        window = make_window(np.concatenate(self.recent), t_us, cfg)
        msg = EventWindow()
        if self.robot_only:      # no /clock in this mode: everybody else runs on wall time
            msg.header.stamp = self.get_clock().now().to_msg()
        else:
            msg.header.stamp.sec, msg.header.stamp.nanosec = t_us // 1_000_000, (t_us % 1_000_000) * 1000
        msg.header.frame_id = "event_camera"
        msg.window_ms = float(cfg.window_ms)
        msg.n_bins, msg.channels, msg.height, msg.width = cfg.shape
        msg.data = array("B", window.tobytes())   # array: fast path, bytes are checked one by one
        self.win_pub.publish(msg)
        if self.img_pub.get_subscription_count() > 0:      # only draw it if someone is watching
            h, w = self.sim.eye_hw
            bgr = ev_repr.render(ev_repr.to_frame(np.concatenate(list(self.recent)[-10:]), h, w), saturation=1.5)
            img = Image()
            img.header = msg.header
            img.height, img.width, img.encoding, img.step = h, w, "bgr8", 3 * w
            img.data = array("B", bgr.tobytes())
            self.img_pub.publish(img)

    def write_frame(self, eye: np.ndarray) -> None:
        import cv2

        from reflex.sim.demo import FONT, _panel
        h, w = self.sim.eye_hw
        ev = ev_repr.render(ev_repr.to_frame(np.concatenate(list(self.recent)[-5:]), h, w), saturation=1.5)[..., ::-1]
        frame = np.hstack([_panel(self.sim.render_third(), "third person"), _panel(eye, "dog eye (RGB)"),
                           _panel(ev, "event camera (emulated)")])
        bar = np.zeros((30, frame.shape[1], 3), dtype=np.uint8)
        shot = f"{self.shot['kind']}/{self.shot['label']}" if self.shot else "-"
        done = [r["outcome"] for r in self.results]
        text = (f"t = {self.sim.time:5.1f} s   shot: {shot:16s} pose: {self.pose:10s} "
                f"correct {done.count('correct')}  wrong {done.count('wrong')}  missed {done.count('missed')}  "
                f"false triggers {done.count('false_trigger')}")
        cv2.putText(bar, text, (8, 21), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        self.writer.append_data(np.vstack([frame, bar]))


def main():
    rclpy.init()
    node = SimBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException, SystemExit):
        pass
    finally:
        if node.writer is not None:
            node.writer.close()


if __name__ == "__main__":
    main()
