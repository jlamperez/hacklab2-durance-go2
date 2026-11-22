# Data collection plan: follow the ball

Goal: the Go2 follows a ball. The network looks at the event camera and says where the
ball is; the behavior turns and walks towards it. This plan says what to record, in which
order, and how to label it, so the takes can train and test that network.

Status: plan only. Nothing here has been recorded yet.

## What the network has to answer

Five outputs, one per window (50 ms of events):

| Label | Meaning | Robot does |
|---|---|---|
| `none` | no ball visible | keep the last command for a moment, then stop / search |
| `left` | ball in the left third of the image | turn left |
| `center` | ball in the middle third, not close | walk forward |
| `right` | ball in the right third | turn right |
| `close` | ball close in front (lower part of the image) | stop |

Five outputs also matches the output size in Durance's example server config
(`docs/durance-riversdk.md`); whether that is required is unverified. Left and right are
in the camera's (and dog's) frame.

The label comes from **where the ball is in the image**, not from where it is going. That
makes labelling simple and is all a follower needs.

### Two event-camera facts that shape the plan

1. **A still ball is invisible to a still camera.** Events only come from change. If the
   dog stands and the ball lies still, the network sees nothing; that is `none`, and the
   behavior has to remember where the ball was.
2. **When the dog moves, everything produces events.** Walking and turning flood the image
   with edges of the floor, walls and furniture. A follower is moving most of the time, so
   the data must include a moving camera (phase 3), or the network will only work while
   the dog stands still.

## Setup before recording

- **Camera height and tilt as on the dog**: about 30–40 cm above the floor, tilted slightly
  down. Measure and write down the real values; ask Durance where the camera sits on the dog.
- **Mark the zones on the floor** with tape. With the camera fixed, open the live view and
  lay tape where the image's thirds meet the floor (two lines fanning out from the camera),
  and a cross line where `close` starts. Add distance marks at 0.5, 1, 2 and 3 m.
- **Same ball as at the demo.** If there may be two different balls, record both.
- Record at full resolution (320x320, what river-tools saves). Downsampling to the 80x80
  the FPGA expects happens later in software.
- Before each session: run `despot_test_camera` on the board once after boot, then
  `river-tools capture --host root@durance-kria.local`.

## Phases

Takes are 8–10 s unless noted. Start recording, then move the ball; stop after the action.
Numbers are a first target; more variety beats more repetitions.

### Phase 0: sensor check (5 min)

| Take | What to do | Takes |
|---|---|---|
| idle | nothing moves, normal light | 3 |
| idle, lights | switch a lamp on and off, walk a shadow past | 2 |
| flicker | point at a screen or a fluorescent lamp if there is one | 1 |

Tells us the noise level and whether lights alone trigger anything.

### Phase 1: still camera, ball in one zone (core training set, about 20 min)

The ball stays inside one zone for the whole take: roll it back and forth, nudge it with a
foot, bounce it lightly. One label per take.

| Label | Distances | Takes per distance | Takes |
|---|---|---|---|
| `left` | 1, 2, 3 m | 3 | 9 |
| `center` | 1, 2, 3 m | 3 | 9 |
| `right` | 1, 2, 3 m | 3 | 9 |
| `close` | 0.3–0.6 m | 6 | 6 |
| `none`, ball still in view | any | 3 | 3 |

Vary speed (slow roll, fast roll) and direction (towards, away, sideways within the zone).

### Phase 2: still camera, ball moving between zones (test set, about 10 min)

Not for training at first: to watch the network live and to score it.

| Take | What to do | Takes |
|---|---|---|
| sweep | ball rolls slowly left -> center -> right and back | 3 |
| approach | ball rolls from 3 m straight in until `close` | 3 |
| diagonal | ball rolls from far left to close center, far right to close center | 4 |
| leave | ball rolls out of view sideways, then nothing | 2 |

Write down roughly when the ball crosses each tape line (or say it aloud and note the
times after); that is enough to label these by time later.

### Phase 3: moving camera (about 15 min)

Without the robot: put the camera on something that moves smoothly at dog height (a
chair or trolley, pushed by hand). Hand-held is a last resort; it shakes far more than the
dog.

| Take | Camera does | Ball | Label | Takes |
|---|---|---|---|---|
| walk, empty | moves forward slowly | none | `none` | 3 |
| turn, empty | turns left, turns right | none | `none` | 4 |
| walk, ball | moves forward | ball still in one zone | zone | 6 (2 per zone) |
| walk, ball moving | moves forward | ball rolls in one zone | zone | 6 |
| turn to ball | turns until the ball is centered | still ball | time-labelled | 4 |

This is the most important phase for following and the hardest to label; keep the ball in
one zone where you can.

### Phase 4: distractors (about 10 min), `none` except the drone

| Take | Takes |
|---|---|
| feet and legs walking past, close and far | 4 |
| someone kicks the ball out of view, then only the legs move | 2 |
| a different object rolled (bottle, box) | 3 |
| bag or jacket dropped in view | 2 |
| drone: across the view, towards and away, hovering then leaving (label `drone`) | 3 |

### Phase 5: on the real Go2 (when the robot is available)

Same structure as phase 3, but with the camera on the dog and the dog driven by joystick
or `ros-teleop`: standing, walking straight, turning on the spot, with and without the ball.
Also record the dog's own motion patterns (trotting in place, standing up, sitting down),
which are `none`. Ask before driving the robot; check speed limits with Durance.

## Sessions and splits

- Record **at least two sessions** in different conditions (other room or corner, other
  light, other time of day). Train on one, validate on the other, never on random takes
  from the same session: neighbouring windows are almost identical, so a random split
  looks far better than reality.
- Total for one session (phases 0–4): about one hour including setup.

## Guided recording (recommended)

Room constraint (21 Nov): only about 2 m of space, and with the camera at dog height the
floor closer than about 1.5 m is below the picture. The script therefore records every
zone at about 2 m, and `close` means "as near as the camera still sees the ball" (bottom
edge of the picture). The 1 / 2 / 3 m tables above are the plan for a bigger room.
Consequence for the robot: it loses sight of the ball about 1.5 m in front of it unless
the camera is tilted further down.

`capture/record.py` walks through a reduced version of this plan: 45 takes,
one per variant instead of the repetitions in the tables above (more repetitions come
from a second session), plus three drone takes. For
each take it shows the instruction, you press SPACE when the action starts and SPACE when
it ends, and it saves the take with its label. M adds a mark while recording (for the
time-labelled takes of phases 2 and 3), ESC cancels, N skips, B deletes the last take to
redo it, Q quits. Rerunning with the same session resumes where it stopped.

```powershell
& "$env:USERPROFILE\.venvs\river-tools\Scripts\python.exe" -m capture.record --session sat_follow1
#   --phases 0,1    only some phases       --list   print the plan and exit
#   --pre 1.0 --post 0.5    seconds kept before the start / after the end mark
```

It needs river-tools (Durance's wheel, not in the repo) and h5py in the same Python
environment. Output in `data/river_samples/<session>/`: one `take_NNN_<label>.h5` per take
and `takes.csv`.

### File format

The recordings are a true event stream, not frames: every event has its own timestamp
(microseconds, camera clock), full 320x320 resolution. Each take is a plain HDF5 file
(`reflex/events/h5.py`):

| Path | Type | Content |
|---|---|---|
| `/events/t` | int64 | timestamp, µs |
| `/events/x`, `/events/y` | uint16 | pixel, 0–319 |
| `/events/p` | uint8 | polarity 0 / 1 as sent by the camera (ON/OFF meaning unverified) |
| root attributes | | `label`, `instruction`, `phase`, `distance_m`, `camera`, `ball`, `t_start_us`, `t_end_us`, `marks_us`, `width`, `height`, ... |

```python
import h5py
with h5py.File("take_004_left.h5") as f:
    t, x, y, p = (f["events"][k][()] for k in "txyp")
    label, start, end = f.attrs["label"], f.attrs["t_start_us"], f.attrs["t_end_us"]
```

River-tools `.bin` recordings convert with `python -m capture.bin_to_h5 <folder or files>`
(verified read-back; `--remove` deletes the originals afterwards).

## Replaying takes

`capture/replay.py` plays a session folder (in take order, with label, instruction
and marks), a folder of plain river-tools samples, or single `.bin` files. The timeline
shows the event rate, the labelled stretch between the start and end mark, and the M
marks; click it to jump. Needs only numpy and matplotlib.

```powershell
& "$env:USERPROFILE\.venvs\river-tools\Scripts\python.exe" -m capture.replay data\river_samples\sat_follow1
#   SPACE pause   arrows seek / change take   +/- window   [/] speed   L loop   G thirds   Q quit
#   --snapshot out.png [--at 3.5]   save one frame of the first take and exit
```

## Take log (manual alternative)

river-tools only names files by time (`sample_YYYYMMDD_HHMMSS_ffffff.bin`) and prints the
name when a take starts. Keep one CSV per session next to the recordings,
`data/river_samples/<session>/takes.csv`:

```csv
file,phase,label,distance_m,camera,ball,notes
sample_20261121_130210_123456.bin,1,left,2,still,rolling,slow roll towards camera
sample_20261121_130231_654321.bin,2,sweep,,still,rolling,center at ~3 s; right at ~6 s
sample_20261121_130300_111111.bin,3,none,,turn-left,none,
```

- `label`: one of the five labels, or the take name (`sweep`, `approach`, ...) for
  time-labelled takes, with the crossing times in `notes`.
- `camera`: `still`, `forward`, `turn-left`, `turn-right`, `handheld`, `dog`.
- Delete failed takes right away and leave them out of the log.

## Afterwards

1. Convert the `.bin` files to the repo's event format (converter still to be written).
2. Look at a few takes per label with `pixi run view-data` to check the labels.
3. Cut windows; train with the session split above; watch phase 2 takes live.

The drone takes have their own label, `drone`: they can be mapped to `none` (ignore
drones) or kept as a target class later without recording again.

Open questions: where Durance mounts the camera on the dog and at which angle; whether
5 outputs is fixed by the FPGA design or only an example; how fast the dog may walk.
