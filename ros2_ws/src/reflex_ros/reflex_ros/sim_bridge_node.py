"""Simulated Go2: MuJoCo behind the same topics as the real robot and camera.

  subscribes  /api/sport/request      unitree_api/Request      (what the real Go2 listens to)
  publishes   /reflex/events_window   reflex_msgs/EventWindow  (what the camera adapter will publish)
              /clock                  simulated time
              /reflex/sim/outcome     std_msgs/String, one JSON line per shot

It rolls balls (and hard negatives) at the dog, applies incoming requests after a
configurable actuation delay, and scores each shot. The behavior node cannot tell this
apart from the robot, which is the point: it is that node's test bench.
"""
import json
from collections import deque

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock as ClockMsg
from std_msgs.msg import String

from reflex.data.episodes import DOG_FRONT_X, THREAT_NEAR
from reflex.data.windows import WindowConfig, make_window
from reflex.events import repr as ev_repr
from reflex.events.emu import EmuConfig, EventEmulator
from reflex.runtime import sport_api
from reflex.runtime.reflex_fsm import BEHAVIORS
from reflex.runtime.sim_sport import pose_for_request
from reflex.sim.scene import BallShot, Go2Sim, to_gray
from reflex_msgs.msg import EventWindow
from unitree_api.msg import Request

WINDOW_STRIDE_STEPS = 10     # one window every 20 ms of simulated time
NEGATIVES = ("miss", "crossing", "walker")


class SimBridge(Node):
    def __init__(self):
        super().__init__("go2_sim_bridge")
        p = self.declare_parameter
        delay_ms = float(p("actuation_delay_ms", 300.0).value)
        if not 200.0 <= delay_ms <= 400.0:
            self.get_logger().warn(f"actuation_delay_ms={delay_ms:.0f} is outside the 200-400 ms range "
                                   "assumed for the real Go2")
        self.delay_s = delay_ms / 1e3
        self.period = float(p("shot_period_s", 3.0).value)
        self.neg_frac = float(p("negative_fraction", 0.3).value)
        self.max_shots = int(p("max_shots", 0).value)          # 0: run until stopped
        self.mode = p("mode", "keeper").value                  # only used to score reactions
        self.record = p("record", "").value                    # mp4 path, empty: no video
        seed = int(p("seed", 0).value)
        randomize = bool(p("randomize", False).value)          # camera mount, shake, sensor

        self.rng = np.random.default_rng(seed)
        self.sim = Go2Sim(seed=seed)
        self.wcfg = WindowConfig()
        if randomize:
            self.sim.randomize_camera(self.rng)
            self.sim.set_shake(self.rng, float(self.rng.uniform(0.2, 1.5)))
        h, w = self.sim.eye_hw
        self.emu = EventEmulator(h, w, EmuConfig.randomized(self.rng) if randomize else EmuConfig(), self.rng)
        self.emu.reset(to_gray(self.sim.render_eye()), 0)
        self.recent = deque(maxlen=round(self.wcfg.window_ms * 1e-3 / self.sim.dt))

        self.clock_pub = self.create_publisher(ClockMsg, "/clock", 10)
        self.win_pub = self.create_publisher(EventWindow, "/reflex/events_window", qos_profile_sensor_data)
        self.out_pub = self.create_publisher(String, "/reflex/sim/outcome", 10)
        self.create_subscription(Request, sport_api.SPORT_REQUEST_TOPIC, self.on_request, 10)

        self.pending = []            # (apply_time_s, pose, api name)
        self.pose = "stand"
        self.steps = 0
        self.shot = None             # dict describing the shot in flight
        self.next_shot_t = 2.0       # quiet start: lets the simple detector calibrate
        self.results = []
        self.latencies_ms = []
        self.writer = None
        if self.record:
            import imageio.v2 as imageio
            self.writer = imageio.get_writer(self.record, fps=50, codec="libx264", quality=8, macro_block_size=2)
        # Wall-clock timer: this node produces /clock, so it cannot run on it.
        self.timer = self.create_timer(self.sim.dt, self.step, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(f"actuation delay {delay_ms:.0f} ms, shot every {self.period:.1f} s, "
                               f"{self.neg_frac:.0%} negatives, scoring as {self.mode}")

    # --- requests from the behavior node -------------------------------------------------
    def on_request(self, msg: Request) -> None:
        api_id = int(msg.header.identity.api_id)
        pose = pose_for_request(api_id, msg.parameter)
        name = sport_api.BY_ID[api_id].name if api_id in sport_api.BY_ID else f"api_id {api_id}"
        if pose is None:
            self.get_logger().warn(f"{name}: no equivalent in the simulator, ignored")
            return
        stamp_us = int(msg.header.identity.id)
        if stamp_us > 0:   # time from the newest event to the request arriving here
            self.latencies_ms.append(self.sim.time * 1e3 - stamp_us * 1e-3)
        self.pending.append((self.sim.time + self.delay_s, pose, name))

    def apply_due(self) -> None:
        while self.pending and self.pending[0][0] <= self.sim.time:
            _, pose, name = self.pending.pop(0)
            if pose == "damp":
                self.sim.damped = True
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
        if self.shot is None and sim.time >= self.next_shot_t and self.pose == "stand":
            self.start_shot()
        self.apply_due()
        if self.shot is not None and self.shot["walker"] is not None:
            x, y0, vy = self.shot["walker"]
            sim.set_walker(x, y0 + vy * (sim.time - self.shot["t0"]))
        sim.step()
        self.steps += 1
        eye = sim.render_eye()
        t_us = int(sim.time * 1e6)
        self.recent.append(self.emu.step(to_gray(eye), t_us))

        if self.steps % 5 == 0:
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
        msg.header.stamp.sec, msg.header.stamp.nanosec = t_us // 1_000_000, (t_us % 1_000_000) * 1000
        msg.header.frame_id = "event_camera"
        msg.window_ms = float(cfg.window_ms)
        msg.n_bins, msg.channels, msg.height, msg.width = cfg.shape
        msg.data = window.tobytes()
        self.win_pub.publish(msg)

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
