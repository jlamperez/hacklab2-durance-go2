# Recording real event-camera data

How we record, label, check and share data from the real camera (Prophesee GenX320 on the
Durance Kria KV260), and what we have so far. What to record and why is in
`docs/data-plan-follow.md`; Durance's tools are summarized in `docs/durance-riversdk.md`.

(`docs/recording.md` is a different path: recording windows from the ROS 2 pipeline with the
webcam stand-in. This document is about the real camera, without ROS.)

## How it fits together

```
GenX320 --MIPI--> Kria KV260 (despot_debug_camera) --TCP, raw events--> laptop
                    started over SSH by our script         port 3333     capture/record.py
                                                                          -> data/river_samples/<session>/
```

- The camera program on the board (`despot_debug_camera`, from Durance's image) streams **raw,
  unfiltered events at full 320x320 resolution** to the laptop. No frames, no fps: every event
  has its own timestamp in microseconds.
- Our scripts use Durance's `river-tools` package to start that program over SSH and to
  receive the stream. Nothing is stored on the board; the laptop holds the only copy.
- The FPGA pipeline (`despot_send_and_forget`) is a different path: it bins events into frames
  (default 100 per second, 80x80) for the SNN. Raw recordings can be turned into any such
  format later; the other way round is not possible.

## One-time setup (Windows laptop)

1. **Python environment with river-tools and h5py.** The `river-tools` wheel comes from
   Durance's download list (login required; proprietary, never commit or publish it). It asks for
   Python 3.10 but works on 3.13:
   ```powershell
   python -m venv $env:USERPROFILE\.venvs\river-tools
   & $env:USERPROFILE\.venvs\river-tools\Scripts\python.exe -m pip install --ignore-requires-python <path>\river_tools-0.1.0-py3-none-any.whl
   & $env:USERPROFILE\.venvs\river-tools\Scripts\python.exe -m pip install h5py
   ```
   (`pipx install` fails with the Microsoft Store Python; a plain venv avoids that.)
2. **Network.** Plug the Kria's own Ethernet port into the laptop. The board hands out an
   address (laptop gets 192.168.77.x); `ssh root@durance-kria.local` works without password.
3. **Firewall.** The board connects *to* the laptop on TCP port 3333. If Windows asks whether
   Python may accept connections, allow it. If the live view stays black, this is the first
   suspect.

Below, `$PY` stands for `& "$env:USERPROFILE\.venvs\river-tools\Scripts\python.exe"` and all
commands run from the repo root. The tools live in `capture/` and run as modules
(`-m capture.record`), so they find `reflex/` without installing it; the file format is in
`reflex/events/h5.py`.

## Each session

1. **Power the board**, wait about a minute. After every boot run the camera check once
   (Durance's workaround against a freeze on first start):
   ```powershell
   ssh root@durance-kria.local despot_test_camera     # ends with "Camera test successfully passed"
   ```
2. **Start the guided recorder:**
   ```powershell
   $PY -m capture.record --session sat_follow2                       # all takes not yet recorded
   $PY -m capture.record --session sat_follow2 --takes 46-51,17-20   # only these, in this order
   $PY -m capture.record --session x --list                         # print the plan
   ```
   The same session name resumes where it stopped. A new name starts a new session (use one
   per place / day, so a whole session can be held out for validation).
3. **Record.** Click into the window once (it must have keyboard focus). For each take the
   window shows the label in large letters and what to do.

   | Key | Does |
   |---|---|
   | SPACE | start the take / end the take (status turns red "RECORDING" in between) |
   | M | mark a moment while recording (ball crosses a zone line, ball comes to rest) |
   | ESC | cancel the take being recorded |
   | N | skip this take |
   | B | delete the last saved take and record it again |
   | Q | quit (closing the window also stops the camera on the board) |

   Each take keeps 1 s before the first SPACE and 0.5 s after the second, so the start and end
   of the action are not clipped. Every key press is logged to the console.
4. **Check** the takes right after (see "Replay" below).
5. **Shut the board down** when done: `ssh root@durance-kria.local poweroff`. LEDs and the fan
   stay on at full speed afterwards (the KV260 cannot cut its own power); unplug it then.

### Tips from the first sessions

- **Keep people out of the picture** unless the take asks for them: stand behind the camera
  and roll the ball away from it, roll it in from outside the picture, or pull it on a thin dark
  string. A hand next to the ball teaches the network to follow hands.
- **Distances:** our room allows about 2 m; closer than about 1.5 m the floor is below the
  picture at dog height. `close` therefore means "bottom edge of the picture".
- **Start before, stop after:** SPACE before the ball starts moving, SPACE once the action is
  over. Takes of 1–2 s are usable but thin; aim for 4–8 s.
- **Flicker** (screens, some lamps) floods the sensor with about 2.4 million events/s. One
  recording of it is enough; keep the demo area free of flickering lights.
- **Turning the camera** produces 50–100 times more events than a rolling ball (ego-motion).
  Walking forward slowly does not.
- **If a take shows something else than asked**, relabel it instead of deleting it (below).

## Replay

```powershell
$PY -m capture.replay data\river_samples\sat_follow2            # a session, in take order
$PY -m capture.replay data\river_samples\sat_follow2\take_045_drone.h5 --loop
$PY -m capture.replay data\durance_dataset\03 --loop            # Durance's gesture dataset
```

Shows the events, the label and instruction, and a timeline with the event rate, the
labelled stretch (green start, red end) and the M marks; click the timeline to jump. Keys:
SPACE pause, arrows seek / change take, +/- events per frame, [ ] speed, L loop, G thirds
of the image, Q quit. `--snapshot out.png --at 2.5` saves one frame without a window.

## Relabel, convert, summarize, share

```powershell
# a take that shows something else: move it to the right take of the plan (add that take to
# build_plan() in capture/record.py first if the case is new)
$PY -m capture.relabel data\river_samples\sat_follow2 54 57 --note "person waving a hand holding the ball"

# river-tools .bin recordings (river-tools capture) -> .h5; --remove deletes the .bin after a verified read-back
$PY -m capture.bin_to_h5 data\river_samples\samples

# summary tables of everything recorded (the snapshot below was made with this)
$PY -m capture.summarize --plan-session sat_follow2
```

To share: zip the session folders' `.h5` files and `takes.csv`; they are compressed already
(the 32 MB flicker take is the largest). `data\` is not in git. Durance's dataset and tools
must not be published or put in the repo.

## File format

One HDF5 file per take (`reflex/events/h5.py`), readable with h5py, MATLAB, HDFView:

| Path | Type | Content |
|---|---|---|
| `/events/t` | int64 | timestamp, microseconds, camera clock, sorted |
| `/events/x`, `/events/y` | uint16 | pixel, 0–319 |
| `/events/p` | uint8 | polarity 0 / 1 as sent by the camera (which one is ON is unverified) |
| root attributes | | `label`, `instruction`, `phase`, `distance_m`, `camera`, `ball`, `t_start_us`, `t_end_us`, `marks_us`, `session`, `width`, `height`, `relabelled_from` (if moved) |

```python
import h5py
with h5py.File("data/river_samples/sat_follow2/take_004_left.h5") as f:
    t, x, y, p = (f["events"][k][()] for k in "txyp")
    label, start, end = f.attrs["label"], f.attrs["t_start_us"], f.attrs["t_end_us"]
```

`takes.csv` per session has one row per take with the same fields. Labels: `left` / `center`
/ `right` = third of the image the ball is in; `close` = bottom edge; `none` = nothing to follow;
`stop-<zone>` = ball comes to rest (M at the moment it stops); `sweep`, `approach`, `diagonal`,
`leave`, `turn-to-ball` = time-labelled with M marks; `drone`, `wave-ball`, `wave-hand` = own
labels, to be mapped to follow / ignore later.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Kria not reachable`, `durance-kria.local` unknown | board off or still booting, or cable not in the board's own port; wait a minute after power-on |
| Window open, nothing saved | the window did not have keyboard focus: click into it; the console log shows every key that arrived |
| Live view stays black | Windows firewall blocks port 3333 for Python |
| Camera program hangs on start | run `despot_test_camera` once after boot |
| Fan at full speed after shutdown | normal for the KV260; unplug |

## What we have (snapshot 21 Nov 2026, evening)

Regenerate with `python -m capture.summarize`. One room, one camera position (stand at
dog height, not on the robot), one day.

| Session | Takes | Action time | Events | Size |
|---|---|---|---|---|
| sat_follow1 | 5 | 31 s | 13.7 M | 32 MB |
| sat_follow2 | 26 | 107 s | 16.9 M | 43 MB |
| samples (no labels) | 8 | | | 5 MB |

| Label | Takes | Action time | Median events/s |
|---|---|---|---|
| none | 11 | 61 s | 3,440 |
| left | 7 | 29 s | 1,916 |
| right | 4 | 9 s | 2,239 |
| center | 3 | 4 s | 1,472 |
| close | 3 | 7 s | 2,637 |
| drone (camera moving) | 1 | 19 s | 36,498 |
| wave-ball | 1 | 4 s | 104,503 |
| wave-hand | 1 | 4 s | 69,474 |

What the numbers say:

- Still scene: about 20–25 events/s of sensor noise (takes 15, 16). A ball at 2 m: a few
  thousand events/s. Camera turning: 160,000–220,000 events/s. Flicker: about 2.4 million.
- `sat_follow1` is the first trial with an older plan; its take 3 (`left`, 1 m) is unusable
  (ball below the picture). Its take 4 and the rest of `sat_follow2` are usable.
- Phase 1 ball takes in `sat_follow2` may show a hand or foot nudging the ball.

Not recorded yet (33 of 59 plan takes), most important first:

1. **Ball rolls and comes to rest** (46–51), the trigger of the follow behavior: none yet.
2. Sweeps, approaches, diagonals, leave (17–23): no time-labelled takes yet.
3. Ball far back (54–56), ball plus a person (52–53).
4. Feet, kick, other objects, bag (36–41); drone with the camera still (42–44).
5. Remaining moving-camera takes (24, 29, 30, 32, 34, 35).

Enough to test a model trained elsewhere (simulator) on left / center / right / close and on a
few distractors; not enough to train on, and nothing yet from the camera mounted on the robot.
