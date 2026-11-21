"""Guided recording session with the Durance event camera (follow-the-ball data plan).

The script walks through the takes of docs/data-plan-follow.md one by one. For each take
it shows what to do, you press SPACE when the action starts and SPACE again when it ends,
and the take is saved with its label. A short stretch before the start and after the end
is kept as well (--pre / --post), with the two marks stored next to it.

Runs on the laptop, in the environment where Durance's river-tools is installed (it uses
river-tools to start the camera on the Kria and to receive the event stream), from the
repo root:

    & "$env:USERPROFILE\\.venvs\\river-tools\\Scripts\\python.exe" -m capture.record --session sat_follow1

Keys (in the window):
    SPACE       start the take / end the take
    M           add a mark while recording (e.g. the ball crosses a tape line)
    ESC         cancel the take being recorded (nothing is saved)
    N           skip this take
    B           go back: delete the last saved take and record it again
    Q           quit (a take being recorded is discarded)

Output, in data/river_samples/<session>/:
    take_003_left.h5    events (t, x, y, p) and the take's label, plan fields and marks
                        as attributes; layout in reflex/events/h5.py
    takes.csv           one row per saved take; rerunning with the same session resumes

Needs h5py in the same environment.

Times in the marks are camera timestamps in microseconds, the same clock as the events.
"""
from __future__ import annotations

import argparse
import csv
import socket
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from reflex.events.h5 import write_h5

REPO = Path(__file__).resolve().parent.parent

# Same wire format as river_tools.capture.events (checked against river-tools 0.1.0).
EVENT_DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
CAMERA_SIZE = 320
COLORS = np.array([[255, 80, 40], [40, 160, 255]], dtype=np.uint8)


# --------------------------------------------------------------------------- plan

@dataclass(frozen=True)
class Take:
    phase: int
    label: str          # none / left / center / right / close, or a take name for time-labelled takes
    instruction: str
    distance_m: str = ""
    camera: str = "still"
    ball: str = ""
    time_labelled: bool = False   # label comes from the marks, not one label for the whole take


def build_plan() -> list[Take]:
    """The takes of docs/data-plan-follow.md, in recording order (45 takes).

    One take per variant; more repetitions come from a second session in another place.
    """
    t: list[Take] = []
    t.append(Take(0, "none", "Nothing moves. SPACE, wait about 5 s, SPACE.", ball="none"))
    t.append(Take(0, "none", "Switch a lamp on and off / walk a shadow past. SPACE before, SPACE after.",
                  ball="none"))
    t.append(Take(0, "none", "Point the camera at a screen or fluorescent lamp (flicker). SPACE, ~5 s, SPACE.",
                  ball="none"))

    # Phase 1: each zone three times at about 2 m (the room allows no more, and closer than
    # about 1.5 m the floor is below the picture); the style changes each time.
    styles = [("rolling", "slow roll back and forth"), ("rolling", "fast roll back and forth"),
              ("nudged", "nudge it with a foot / bounce it lightly")]
    for label in ("left", "center", "right"):
        for ball, how in styles:
            t.append(Take(1, label, f"Ball stays in the {label.upper()} zone at about 2 m: {how}. "
                                    "SPACE when it starts moving, SPACE when it stops.",
                          distance_m="2", ball=ball))
    for ball, how in styles:
        t.append(Take(1, "close", f"Ball CLOSE: as near as the camera still sees it (bottom edge of the "
                                  f"picture): {how}. SPACE start, SPACE end.",
                      distance_m="nearest visible", ball=ball))
    t += [Take(1, "none", "Ball lies STILL in view, nothing moves. SPACE, ~5 s, SPACE.", ball="still")
          for _ in range(2)]

    marks = " Press M each time it crosses a tape line."
    t += [Take(2, "sweep", "Move the ball slowly LEFT -> CENTER -> RIGHT: pull it on a thin dark string, or "
                           "roll it across from outside the picture. Keep people out of view." + marks,
               ball="rolling", time_labelled=True) for _ in range(2)]
    t += [Take(2, "approach", "Roll the ball from 2 m straight in until it leaves the bottom of the "
                              "picture." + marks,
               distance_m="2-nearest", ball="rolling", time_labelled=True) for _ in range(2)]
    for side in ("left", "right"):
        t.append(Take(2, "diagonal", f"Roll the ball from {side.upper()} at 2 m to the bottom center of the "
                                     "picture." + marks,
                      ball="rolling", time_labelled=True))
    t.append(Take(2, "leave", "Roll the ball out of view sideways (push it from outside the picture), then "
                              "nothing moves." + marks,
                  ball="rolling", time_labelled=True))

    t += [Take(3, "none", "No ball. Move the camera FORWARD slowly (chair / trolley).",
               camera="forward", ball="none") for _ in range(2)]
    for d in ("left", "right"):
        t.append(Take(3, "none", f"No ball. TURN the camera to the {d.upper()}.", camera=f"turn-{d}", ball="none"))
    for label in ("left", "center", "right"):
        t.append(Take(3, label, f"Ball lies still in the {label.upper()} zone. Move the camera FORWARD slowly.",
                      camera="forward", ball="still"))
    for label in ("left", "center", "right"):
        t.append(Take(3, label, f"Ball rolls in the {label.upper()} zone. Move the camera FORWARD slowly.",
                      camera="forward", ball="rolling"))
    for d in ("left", "right"):
        t.append(Take(3, "turn-to-ball", f"Ball lies still to the {d.upper()}. Turn the camera until the "
                                         "ball is centered." + marks,
                      camera=f"turn-{d}", ball="still", time_labelled=True))

    t += [Take(4, "none", "Feet / legs walk past the camera (one close, one far).", ball="none")
          for _ in range(2)]
    t.append(Take(4, "none", "Someone kicks the ball out of view, then only the legs move.", ball="kicked"))
    t += [Take(4, "none", "Roll a DIFFERENT object (bottle, box) past the camera.", ball="other")
          for _ in range(2)]
    t.append(Take(4, "none", "Drop a bag or jacket in view.", ball="none"))
    # Own label so it can later be treated as "none" (ignore) or as a target.
    drone = ["Fly the DRONE across the view, left to right, at 1-2 m.",
             "Fly the DRONE towards the camera and away again.",
             "Let the DRONE hover in view, then fly it out of the picture."]
    t += [Take(4, "drone", how + " SPACE start, SPACE end.", ball="none") for how in drone]
    # Added 21 Nov after a recording of this case; appended so earlier take numbers stay.
    t.append(Take(4, "drone", "Fly the DRONE in view while the camera MOVES (forward or turning). "
                              "SPACE start, SPACE end.", camera="moving", ball="none"))
    # Added 21 Nov: the trigger of the follow behavior (ball rolls, then comes to rest) and
    # the ball with a person in view. Appended so earlier take numbers stay.
    for zone in ("left", "left", "center", "center", "right", "right"):
        t.append(Take(1, f"stop-{zone}", f"Roll the ball into the {zone.upper()} zone (from outside the picture) "
                                         "and let it COME TO REST there. Press M the moment it stops; "
                                         "SPACE about 2 s later.",
                      distance_m="2", ball="rolls-stops", time_labelled=True))
    for zone in ("left", "right"):
        t.append(Take(4, zone, f"Ball rolls in the {zone.upper()} zone while a PERSON is visible on the other "
                               "side (standing or walking). Label is the ball's zone.",
                      distance_m="2", ball="rolling+person"))
    # Added 21 Nov: ball further away than the 2 m takes, as far back as the room allows.
    for zone in ("left", "center", "right"):
        t.append(Take(1, zone, f"Stand BEHIND the camera and roll the ball AWAY from it into the {zone.upper()} "
                               "zone, as far back as the room allows; let it come to rest. (Or roll it in from "
                               "the side, from outside the picture.) Nobody in view. SPACE before, SPACE after.",
                      distance_m="max (>2)", ball="rolls-away"))
    # Added 21 Nov after recordings of these cases; own labels, mapped later (follow / ignore).
    t.append(Take(4, "wave-ball", "A person WAVES a hand that HOLDS the ball. SPACE start, SPACE end.",
                  ball="in-hand"))
    t.append(Take(4, "wave-hand", "A person WAVES a closed hand, NO ball. SPACE start, SPACE end.",
                  ball="none"))
    return t


def build_cross_plan() -> list[Take]:
    """Plan "cross" (21 Nov): 10 takes each of the ball rolling in from the left, from the
    right, from the center towards the camera, and 10 without a ball (people, noise, nothing)."""
    t: list[Take] = []
    away = " Roll it from outside the picture; nobody in view. SPACE before it rolls, SPACE after it left or stopped."
    t += [Take(1, "from-left", "Roll the ball in from the LEFT side, across the picture." + away,
               distance_m="~2", ball="rolling") for _ in range(10)]
    t += [Take(1, "from-right", "Roll the ball in from the RIGHT side, across the picture." + away,
               distance_m="~2", ball="rolling") for _ in range(10)]
    t += [Take(1, "from-center", "Roll the ball from far back in the CENTER straight towards the camera." + away,
               distance_m="far->near", ball="rolling") for _ in range(10)]
    none = [("A PERSON walks through the picture (vary near / far, left->right and right->left).", "person")] * 4         + [("Someone moves ARMS or HANDS in the picture, no ball.", "arms")] * 2         + [("NOISE: switch a lamp on/off or move a shadow / reflection, nobody in view.", "light")] * 2         + [("NOTHING moves. SPACE, wait about 5 s, SPACE.", "idle")] * 2
    t += [Take(4, "no-ball", f"No ball. {how} SPACE start, SPACE end.", ball="none", camera="still")
          for how, _ in none]
    return t


def build_center_plan() -> list[Take]:
    """Plan "center" (21 Nov): the ball starts in the center and rolls to the left or right,
    or comes straight at the camera; plus the same 10 no-ball takes as plan "cross"."""
    t: list[Take] = []
    rule = " Nobody in view. SPACE before it rolls, SPACE after it left or stopped."
    t += [Take(1, "to-left", "Ball starts in the CENTER (far back) and rolls to the LEFT, out of the picture." + rule,
               distance_m="~2", ball="rolling") for _ in range(10)]
    t += [Take(1, "to-right", "Ball starts in the CENTER (far back) and rolls to the RIGHT, out of the picture." + rule,
               distance_m="~2", ball="rolling") for _ in range(10)]
    t += [Take(1, "from-center", "Roll the ball from far back in the CENTER straight towards the camera." + rule,
               distance_m="far->near", ball="rolling") for _ in range(10)]
    t += [tk for tk in build_cross_plan() if tk.label == "no-ball"]
    # Added 21 Nov: spare slots for straight throws recorded under another take number.
    t += [Take(1, "from-center", "Roll the ball from far back in the CENTER straight towards the camera." + rule,
               distance_m="far->near", ball="rolling") for _ in range(3)]
    return t


# Plans by name; a session folder remembers its plan in plan.txt.
PLANS = {"follow": build_plan, "cross": build_cross_plan, "center": build_center_plan}


def session_plan(folder: Path, requested: str | None = None) -> str:
    """Name of the session's plan: from plan.txt if the session exists, else `requested`."""
    f = folder / "plan.txt"
    if f.exists():
        name = f.read_text(encoding="utf-8").strip()
        if requested and requested != name:
            raise SystemExit(f"session {folder.name} uses plan {name!r}, not {requested!r}")
        return name
    return requested or "follow"


# --------------------------------------------------------------------------- session files

CSV_FIELDS = ["take", "file", "phase", "label", "distance_m", "camera", "ball", "time_labelled",
              "t_start_us", "t_end_us", "marks_us", "n_events", "recorded_at", "instruction"]


@dataclass
class SavedTake:
    take: int
    file: str
    row: dict = field(default_factory=dict)


class SessionStore:
    """The session folder: event files, metadata and takes.csv. No camera or GUI needed."""

    def __init__(self, folder: Path, plan: list[Take], plan_name: str = "follow"):
        self.folder, self.plan, self.plan_name = folder, plan, plan_name
        self.folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "plan.txt").exists():
            (folder / "plan.txt").write_text(plan_name, encoding="utf-8")
        self.csv_path = folder / "takes.csv"
        self.rows: list[dict] = []
        if self.csv_path.exists():
            with self.csv_path.open(newline="") as f:
                self.rows = list(csv.DictReader(f))

    def done(self) -> set[int]:
        return {int(r["take"]) for r in self.rows}

    def next_index(self, start: int = 0, allowed_phases: set[int] | None = None) -> int | None:
        done = self.done()
        for i in range(start, len(self.plan)):
            if i not in done and (allowed_phases is None or self.plan[i].phase in allowed_phases):
                return i
        return None

    def _write_csv(self) -> None:
        tmp = self.csv_path.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(sorted(self.rows, key=lambda r: int(r["take"])))
        tmp.replace(self.csv_path)

    def save(self, index: int, events: np.ndarray, t_start: int, t_end: int, marks: list[int]) -> SavedTake:
        take = self.plan[index]
        stem = f"take_{index:03d}_{take.label}"
        meta = {"take": index, **asdict(take), "t_start_us": t_start, "t_end_us": t_end,
                "marks_us": marks, "n_events": int(len(events)),
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "session": self.folder.name,
                "plan": self.plan_name}
        write_h5(self.folder / f"{stem}.h5", events, meta, CAMERA_SIZE, CAMERA_SIZE)
        row = {k: meta.get(k, "") for k in CSV_FIELDS}
        row["file"] = f"{stem}.h5"
        row["marks_us"] = " ".join(str(m) for m in marks)
        self.rows = [r for r in self.rows if int(r["take"]) != index] + [row]
        self._write_csv()
        return SavedTake(index, row["file"], row)

    def delete(self, index: int) -> None:
        for r in [r for r in self.rows if int(r["take"]) == index]:
            stem = Path(r["file"]).stem
            for suffix in (".h5", ".bin", ".json"):
                (self.folder / f"{stem}{suffix}").unlink(missing_ok=True)
        self.rows = [r for r in self.rows if int(r["take"]) != index]
        self._write_csv()


# --------------------------------------------------------------------------- live recording

class Recorder:
    """Receives event batches (from any thread) and cuts takes out of the stream.

    Keeps the last `pre_us` of events so a take starts a bit before SPACE was pressed, and
    keeps recording `post_us` after the end mark before the take is complete.
    """

    def __init__(self, pre_us: int, post_us: int):
        self.pre_us, self.post_us = pre_us, post_us
        self.lock = threading.Lock()
        self.recent: deque[np.ndarray] = deque()
        self.last_ts = -1
        self.image = np.zeros((CAMERA_SIZE, CAMERA_SIZE, 3), dtype=np.uint8)
        self.events_seen = 0
        self._take: list[np.ndarray] | None = None
        self.t_start = self.t_end = None
        self.marks: list[int] = []
        self._t_end_wall = 0.0

    @property
    def recording(self) -> bool:
        return self._take is not None

    def push(self, events: np.ndarray) -> None:
        if not len(events):
            return
        with self.lock:
            self.events_seen += len(events)
            self.last_ts = int(events["timestamp"][-1])
            ok = (events["x"] < CAMERA_SIZE) & (events["y"] < CAMERA_SIZE) & (events["polarity"] < 2)
            ev = events[ok]
            self.image[ev["y"], ev["x"]] = COLORS[ev["polarity"]]
            self.recent.append(events)
            while self.recent and int(self.recent[0]["timestamp"][-1]) < self.last_ts - self.pre_us:
                self.recent.popleft()
            if self._take is not None:
                self._take.append(events)

    def take_image(self) -> np.ndarray:
        with self.lock:
            img = self.image.copy()
            self.image[:] = 0
            return img

    def start(self) -> None:
        with self.lock:
            self.t_start = self.last_ts
            self.t_end, self.marks = None, []
            self._take = list(self.recent)

    def mark(self) -> None:
        with self.lock:
            if self._take is not None and self.t_end is None:
                self.marks.append(self.last_ts)

    def end(self) -> None:
        with self.lock:
            if self._take is not None and self.t_end is None:
                self.t_end, self._t_end_wall = self.last_ts, time.monotonic()

    def cancel(self) -> None:
        with self.lock:
            self._take, self.t_start, self.t_end, self.marks = None, None, None, []

    def finished(self) -> tuple[np.ndarray, int, int, list[int]] | None:
        """The complete take once the post-roll has passed, else None."""
        with self.lock:
            if self._take is None or self.t_end is None:
                return None
            # Few events arrive when nothing moves, so also stop on wall time.
            waited = time.monotonic() - self._t_end_wall
            if self.last_ts < self.t_end + self.post_us and waited < self.post_us / 1e6 + 1.0:
                return None
            ev = np.concatenate(self._take) if self._take else np.zeros(0, EVENT_DTYPE)
            lo, hi = self.t_start - self.pre_us, self.t_end + self.post_us
            ev = ev[(ev["timestamp"] >= lo) & (ev["timestamp"] <= hi)]
            out = (ev, self.t_start, self.t_end, list(self.marks))
            self._take, self.t_start, self.t_end, self.marks = None, None, None, []
            return out


def receive(port: int, host: str | None, recorder: Recorder, stop: threading.Event,
            status: dict) -> None:
    """Listen for the camera, start it over SSH, push every batch into the recorder."""
    from river_tools.capture.remote import start_camera, stop_camera
    from river_tools.capture.stream import receive_batches

    camera = None
    try:
        with socket.create_server(("", port)) as server:
            server.settimeout(0.2)
            status["text"] = f"Waiting for the camera on port {port}..."
            if host:
                camera = start_camera(host, port)
            while not stop.is_set():
                if camera is not None and camera.poll() is not None:
                    status["text"] = "Camera SSH session ended. Close the window and start again."
                    break
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                status["connected"] = True
                print(f"Camera connected from {addr[0]}.")
                with conn:
                    try:
                        for batch in receive_batches(conn, float("inf"), stop):
                            recorder.push(np.frombuffer(bytes(batch), dtype=EVENT_DTYPE))
                    except ConnectionError:
                        pass
                status["connected"] = False
                print("Camera disconnected.")
    finally:
        if camera is not None:
            stop_camera(camera)


# --------------------------------------------------------------------------- window

def run_gui(store: SessionStore, recorder: Recorder, phases: set[int] | None, start: int,
            port: int, host: str | None, order: list[int] | None = None) -> None:
    import matplotlib.pyplot as plt

    for k in list(plt.rcParams):          # free the keys matplotlib uses for its own shortcuts
        if k.startswith("keymap."):
            plt.rcParams[k] = []

    stop = threading.Event()
    status = {"text": "", "connected": False}
    worker = threading.Thread(target=receive, args=(port, host, recorder, stop, status), daemon=True)

    plan = store.plan
    if order is None:
        order = [i for i, t in enumerate(plan) if i >= start and (phases is None or t.phase in phases)]

    def next_in_order(after: int | None) -> int | None:
        """First take not done yet in `order`, after `after` (then wrapping round)."""
        done = store.done()
        pos = order.index(after) + 1 if after in order else 0
        for i in order[pos:] + order[:pos]:
            if i not in done:
                return i
        return None

    state = {"index": next_in_order(None), "history": [], "msg": ""}
    total = len(order)

    fig = plt.figure(figsize=(9, 9.5), facecolor="#0c121c")
    fig.canvas.manager.set_window_title(f"Guided capture - {store.folder.name}")
    ax = fig.add_axes([0.08, 0.27, 0.84, 0.52], facecolor="black")
    ax.set_xticks([]), ax.set_yticks([])
    view = ax.imshow(np.zeros((CAMERA_SIZE, CAMERA_SIZE, 3), np.uint8), interpolation="nearest")
    head = fig.text(0.08, 0.94, "", color="#91a0b3", fontsize=10)
    title = fig.text(0.08, 0.885, "", color="#edf2f7", fontsize=24, weight="bold")
    instr = fig.text(0.08, 0.81, "", color="#edf2f7", fontsize=11, wrap=True, va="bottom")
    stat = fig.text(0.08, 0.20, "", color="#6ce3c5", fontsize=14, weight="bold")
    msg = fig.text(0.08, 0.15, "", color="#91a0b3", fontsize=10)
    fig.text(0.08, 0.04, "SPACE start/end   M mark   ESC cancel   N skip   B back (redo last)   Q quit",
             color="#91a0b3", fontsize=9)

    def advance(from_index: int) -> None:
        state["index"] = next_in_order(from_index)

    def on_key(event) -> None:
        key = (event.key or "").lower()
        i = state["index"]
        print(f"key {event.key!r} on take {i}: recording={recorder.recording} "
              f"connected={status['connected']}", flush=True)
        if key == "q":
            recorder.cancel()
            plt.close(fig)
        elif i is None:
            return
        elif key == " ":
            if not recorder.recording:
                if not status["connected"]:
                    state["msg"] = "No camera stream yet."
                    return
                recorder.start()
                state["msg"] = ""
            elif recorder.t_end is None:
                recorder.end()
        elif key == "m" and recorder.recording:
            recorder.mark()
            state["msg"] = f"Mark {len(recorder.marks)} set."
        elif key == "escape" and recorder.recording:
            recorder.cancel()
            state["msg"] = "Take cancelled, nothing saved."
        elif key == "n" and not recorder.recording:
            state["msg"] = f"Skipped take {i}."
            advance(i)
        elif key == "b" and not recorder.recording and state["history"]:
            last = state["history"].pop()
            store.delete(last)
            state["index"] = last
            state["msg"] = f"Deleted take {last}; record it again."

    fig.canvas.mpl_connect("key_press_event", on_key)
    def on_close(_) -> None:
        print(f"window closed on take {state['index']} (recording={recorder.recording})", flush=True)
        stop.set()

    fig.canvas.mpl_connect("close_event", on_close)
    worker.start()
    plt.show(block=False)
    try:
        while plt.fignum_exists(fig.number):
            done = recorder.finished()
            if done is not None and state["index"] is not None:
                ev, t0, t1, marks = done
                i = state["index"]
                saved = store.save(i, ev, t0, t1, marks)
                print(f"Saved take {i} ({plan[i].label}): {len(ev)} events, "
                      f"{(t1 - t0) / 1e6:.1f} s -> {saved.file}")
                state["history"].append(i)
                state["msg"] = f"Saved {saved.file}: {len(ev)} events, {(t1 - t0) / 1e6:.1f} s."
                advance(i)

            i = state["index"]
            n_done = len(store.done() & set(order))
            if i is None:
                head.set_text(f"{store.folder.name}   {n_done}/{total} takes")
                title.set_text("All takes done")
                instr.set_text("Close the window or press Q.")
            else:
                t = plan[i]
                extra = "   ".join(s for s in (f"{t.distance_m} m" if t.distance_m else "",
                                               f"camera: {t.camera}", f"ball: {t.ball}" if t.ball else "") if s)
                head.set_text(f"{store.folder.name}   take {i}   phase {t.phase}   "
                              f"{n_done}/{total} done   {extra}")
                title.set_text(t.label.upper())
                instr.set_text(t.instruction)
            if not status["connected"]:
                stat.set_text(status["text"] or "Starting...")
                stat.set_color("#91a0b3")
            elif recorder.recording and recorder.t_end is not None:
                stat.set_text("Saving...")
                stat.set_color("#ffd166")
            elif recorder.recording:
                stat.set_text("RECORDING  -  SPACE when the action ends")
                stat.set_color("#ff4b4b")
            else:
                stat.set_text("Ready  -  SPACE when the action starts")
                stat.set_color("#6ce3c5")
            msg.set_text(state["msg"])
            view.set_data(recorder.take_image())
            fig.canvas.draw_idle()
            plt.pause(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join(timeout=10)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="session name, e.g. sat_follow1")
    ap.add_argument("--out", default=str(REPO / "data" / "river_samples"), help="parent folder of sessions")
    ap.add_argument("--host", default="root@durance-kria.local", help="SSH target of the Kria ('' = start it yourself)")
    ap.add_argument("--port", type=int, default=3333, help="port this laptop listens on")
    ap.add_argument("--phases", default="", help="only these phases, e.g. 0,1 (default: all)")
    ap.add_argument("--takes", default="", help="only these takes, in this order, e.g. 15,16,46-51,17")
    ap.add_argument("--start", type=int, default=0, help="first take index to offer")
    ap.add_argument("--pre", type=float, default=1.0, help="seconds kept before the start mark")
    ap.add_argument("--post", type=float, default=0.5, help="seconds kept after the end mark")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--plan", choices=sorted(PLANS), default=None,
                    help="which list of takes (default: the session's own, else 'follow')")
    args = ap.parse_args()

    plan_name = session_plan(Path(args.out) / args.session, args.plan)
    plan = PLANS[plan_name]()
    phases = {int(p) for p in args.phases.split(",") if p.strip()} or None
    order = None
    if args.takes:
        order = []
        for part in args.takes.split(","):
            lo, _, hi = part.strip().partition("-")
            order += list(range(int(lo), int(hi or lo) + 1))
        bad = [i for i in order if not 0 <= i < len(plan)]
        if bad:
            raise SystemExit(f"--takes: no such takes {bad} (plan has 0-{len(plan) - 1})")
    if args.list:
        for i, t in enumerate(plan):
            if (order is None or i in order) and (phases is None or t.phase in phases):
                print(f"{i:3d}  phase {t.phase}  {t.label:13s} {t.instruction}")
        return

    store = SessionStore(Path(args.out) / args.session, plan, plan_name)
    if store.done():
        print(f"Resuming {store.folder}: {len(store.done())} takes already saved.")
    recorder = Recorder(int(args.pre * 1e6), int(args.post * 1e6))
    run_gui(store, recorder, phases, args.start, args.port, args.host or None, order)
    print(f"{len(store.rows)} takes in {store.csv_path}")


if __name__ == "__main__":
    main()
