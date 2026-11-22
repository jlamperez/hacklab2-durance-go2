# Technical overview

How the code on `main` fits together, as of commit `e034923` (22 Nov 2026, after merging
`merge-all`). `README.md` has the status table and every command, `docs/commands.md` the
recipes. This file explains how the parts connect: data flow, ROS 2 interfaces, the Kria
pipeline, the networks, and how a network gets onto the FPGA. Details and measurements stay
in the per-topic docs; each section points to them.

What is real and what is simulated, in one paragraph: our nodes have never driven the real
Go2 (sport-mode commands were sent to it by hand on 21 Nov, `docs/real-robot.md`). The real
GenX320 event camera has recorded about 100 labelled takes and, on 22 Nov, fed two of our
networks running on the Kria's FPGA. The whole pipeline runs on the Kria and drives the
**simulated** dog on the laptop over the cable. No accuracy figure for the live decisions
has been measured.

## 1. What the system does

Two behaviors share one pipeline (events → network → decision → Unitree sport-mode request):

| Behavior | Input | Output to the robot | Node | Logic |
|---|---|---|---|---|
| Reflex (goalkeeper or dodge) | `none / left / center / right` per window | lean (Pose + Euler roll), sidestep (Move) or crouch (StandDown) | `behavior` | `reflex/runtime/reflex_fsm.py`, `sport_api.requests_for_pose` |
| Follow / look | where the target is (`u, v, confidence`), a zone name, or a decision | walk bursts (Move), trunk yaw (Euler), or one of nine trunk gestures (Euler yaw + pitch) | `follow` | `reflex/runtime/follow.py` |

Both publish `unitree_api/Request` on `/api/sport/request`. Run one of them at a time: two
nodes commanding the same robot will fight.

## 2. The pipeline and its three seams

```
EVENT SOURCE                  INFERENCE                         BEHAVIOR               ROBOT
sim_bridge (MuJoCo)     ->    perception (PyTorch, CPU)    ->   behavior          ->   sim_bridge (MuJoCo)
camera_webcam (emulated)      target / ball_target (CPU)        follow                 real Go2 (same message)
event_camera (h5 / live)      fpga_perception (SPLEAT, FPGA)
despot on the Kria
```

Each column can be swapped without touching the others, because they only meet at ROS 2
topics:

| Seam | Code | Options |
|---|---|---|
| Event source | `reflex/events/io.py`, `sources.py`, nodes `sim_bridge`, `camera_webcam`, `event_camera`; on the board `despot_send_and_forget` | MuJoCo emulated camera, webcam through the emulator, recorded GenX320 takes, live GenX320 |
| Inference | `reflex/runtime/`, nodes `perception`, `target`, `ball_target`, `fpga_perception` | reflex SNN or no-learning detector on CPU, ball networks on CPU in integer arithmetic, a network on the FPGA |
| Robot | topic `/api/sport/request` | `sim_bridge` (MuJoCo, with an actuation delay, 70 ms by default as measured on the Go2), the real Go2 |

Nodes are thin. The logic lives in the `reflex/` Python package, where it is unit-tested
without ROS (201 tests, `pixi run test`).

Frame convention: dog at the origin, +x forward, +y left, +z up. Decision labels are in the
dog frame. `Target.u` and zone names are in the image as the camera sees it; with the camera
facing forward and not mirrored, image left is the dog's left. Whether despot's x runs the
same way as the raw events' x has not been checked on the board.

## 3. Deployments

| Setup | Event source | Inference | Robot | Run with | State |
|---|---|---|---|---|---|
| All on the laptop | MuJoCo camera | reflex SNN, no-learning or ball detector on CPU | MuJoCo | `pixi run -e ros ros-sim` | working; 12-shot smoke tests in `README.md` |
| Real events, simulated dog | recordings or live KV260 stream into `event_camera` | centroid tracker or ball SNN on CPU | MuJoCo, walking | `ros-events-sim` (`docs/events-sim.md`) | sources tested without ROS; launch not yet run under ROS |
| Whole pipeline on the Kria | live GenX320 via despot | zones3 or ball_v3 on the FPGA | MuJoCo on the laptop over Ethernet | `/opt/reflex/run.sh robot:=sim` + `ros-robot-sim` or `ros-look` on the laptop | runs (22 Nov); live decisions weak and unmeasured |
| Hackathon goal | live GenX320 via despot | network on the FPGA | real Go2 | `/opt/reflex/run.sh robot:=real enabled:=true` | not run |

## 4. ROS 2 interface

### Messages (`ros2_ws/src/reflex_msgs`)

| Message | Fields | Meaning |
|---|---|---|
| `EventWindow` | `header`, `window_ms`, `n_bins`, `channels` (2: OFF, ON), `height`, `width`, `uint8[] data` | event counts, row-major `(n_bins, channels, height, width)`; stamp = newest event |
| `ReflexDecision` | `header`, `label`, `confidence`, `detector`, `inference_ms` | `none / left / center / right` in the dog frame; stamp copied from the window |
| `Target` | `header`, `u`, `v`, `confidence`, `source` | `u` -1 left .. +1 right, `v` -1 top .. +1 bottom; confidence 0 means nothing seen. Contract in `reflex/runtime/target.py` |

Robot messages come from `unitree_api` (`Request`, `header.identity.api_id` plus a JSON
`parameter`). On the laptop they are built from `unitree_ros2`; on the Kria they come from
Durance's runtime in `/opt/go2-*`. The behavior node writes the time of the newest event into
`header.identity.id`, so the receiver can measure end-to-end latency.

### Topics

| Topic | Type | Published by | Read by |
|---|---|---|---|
| `/reflex/events_window` | `EventWindow` | `sim_bridge`, `camera_webcam`, `event_camera` | `perception`, `target`, `recorder` |
| `/reflex/events_fullres` | `EventWindow` (1 bin, 320x320) | `event_camera`, only while someone subscribes | `ball_target`, `ball_zone_sim` |
| `/reflex/decision` | `ReflexDecision` | `perception`, `fpga_perception` | `behavior`, `follow` (`input:=decision`), `web_viewer` |
| `/reflex/target` | `Target` | `target` (centroid), `ball_target` (SNN) | `follow` (`input:=target`) |
| `/durance/ball_zone` | `std_msgs/String`, reliable + transient-local, only on change | `fpga_perception` (6-zone networks), `ball_zone_sim` | `follow` (`input:=zone`, default of `ros-look`), `web_viewer` |
| `/api/sport/request` | `unitree_api/Request` | `behavior`, `follow`, `tricks`, `cmd_vel_to_sport` | `sim_bridge` or the Go2 |
| `/cmd_vel` | `geometry_msgs/Twist` | any teleop | `cmd_vel_to_sport` |
| `/reflex/debug/events_image` | `sensor_msgs/Image` | `sim_bridge`, `camera_webcam`, `event_camera`, `camera_tap` | `web_viewer`, `rqt_image_view` |
| `/reflex/debug/{perception,behavior,follow,target,fpga,replay}` | `String`, JSON | the node of that name | `web_viewer`, people |
| `/reflex/sim/state`, `/reflex/sim/outcome`, `/reflex/debug/latency_ms`, `/clock` | various | `sim_bridge` | `web_viewer` |

Services: `/reflex/recalibrate` (perception), `/reflex/target/recalibrate` (target),
`/reflex/follow/enable` (`std_srvs/SetBool`, follow).

### Nodes (`ros2_ws/src/reflex_ros`)

| Node | Role |
|---|---|
| `sim_bridge` | MuJoCo dog behind the robot's and camera's topics; rolls balls and hard negatives, applies requests after a delay, scores each shot. `robot_only` turns it into the robot alone |
| `camera_webcam` | webcam through the event emulator; plumbing test, says nothing about the real sensor |
| `event_camera` | GenX320 recordings (`.h5` / `.bin`, real time, looped, still scene first) or the live KV260 stream |
| `perception` | `EventWindow` → `ReflexDecision`; `detector:=snn|simple|ball` |
| `fpga_perception` | reads the SPLEAT server's output over TCP → `ReflexDecision` and, for 6-zone networks, `/durance/ball_zone` |
| `camera_tap` | on the Kria, between despot and the SPLEAT server: reorders frames CHW → HWC, drops frames while the FPGA is busy, publishes the picture |
| `target` | no-learning centroid tracker → `Target` |
| `ball_target` | the teammates' ball SNNs on CPU (default the zone classifier) → `Target` |
| `ball_zone_sim` | stand-in for a board-side zone publisher: zones from the CPU SNN or the keyboard |
| `behavior` | reflex state machine and all reflex safety logic |
| `follow` | move-stop-look controller; keyboard `f` start, `SPACE` stop, `!` Damp, `1`-`9` gestures |
| `tricks`, `cmd_vel_to_sport` | built-in Go2 tricks from the keyboard; `/cmd_vel` → Move |
| `recorder` | labelled takes from `/reflex/events_window` (webcam path) |
| `web_viewer` | browser dashboard on port 8080 (viser); listens only |

Launch files: `sim`, `robot_sim`, `camera`, `drive`, `follow_sim`, `events_sim` for the
laptop, and `kria` for the board.

## 5. The pipeline on the Kria

```
GenX320 --MIPI--> FPGA camera input --> /dev/video0 (raw events, 320x320)
   |
despot_send_and_forget (Durance)   frames per /root/metavision_camera.json
   | TCP 127.0.0.1:3333
camera_tap (ours)                  CHW -> HWC in place, max_inflight, picture at 10 Hz
   | TCP 3334
spleat-server / backward-server    <-> SNN on the FPGA (spleat bitstream, /root/weights.txt)
   | TCP 3334, one CBOR message per frame: {"id", "payload": float32 spikes, "latency", "error"}
fpga_perception (ours)             readout, vote, hold -> /reflex/decision, /durance/ball_zone
   |
behavior (ours)  or  follow on the laptop
   | /api/sport/request
robot:=sim  -> board eth0 192.168.77.1 -> laptop (ros-robot-sim behavior:=false dds_interface:=enp8s0)
robot:=real -> USB-Ethernet adapter 192.168.123.99 -> Go2 (/etc/cyclonedds/cyclonedds-go2.xml)
```

Source: `ros2_ws/src/reflex_ros/launch/kria.launch.py`, `scripts/kria/run.sh`. Durance's side
of this path (despot, the server, their `gesture_node` and `action_bridge`) is described in
`docs/fpga-pipeline.md` and `docs/despot.md`.

### Starting it: `run.sh`

`/opt/reflex/run.sh robot:=sim|real [model:=zones3|ball_v3] [name:=value ...]` does, in order:

1. runs `despot_test_camera` and stops unless it passes (skipped when despot already holds the camera);
2. moves spleat-server to port 3334 in `/root/.config/spleat-server/config.yaml` and restarts it if needed (3333 with `tap:=false`);
3. sources ROS 2 Humble, Durance's overlay in `/opt/go2-*` (for `unitree_api`) and ours;
4. checks that the chosen model's output count matches the server config, so it refuses to read zones3 spikes as ball_v3's or the reverse;
5. adds the model's defaults from `models/<model>/run_args` (command-line values win);
6. `exec ros2 launch reflex_ros kria.launch.py model_dir:=/opt/reflex/models/<model> ...`.

`kria.launch.py` starts `camera_tap`, despot 3 s later (so the tap is listening), then
`fpga_perception` and `behavior`, with `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and
`CYCLONEDDS_URI` pinned to the interface for `robot`. With `robot:=real` the behavior node
starts disabled and only logs what it would send; `enabled:=true` has to be passed on purpose.

### Installing it: `deploy.sh`

`scripts/kria/deploy.sh [root@192.168.77.1]` runs on the laptop. It cross-compiles
`reflex_msgs` and `reflex_ros` for aarch64 inside Durance's Docker image
(`durance/go2-yocto:humble`, built by `./manage.sh image` in their `go2-kria` folder), packs
the build, the pure-Python `reflex` package, the readout files of `models/zones3` and
`models/ball_v3` and `run.sh`, and unpacks them into `/opt/reflex` on the board (replacing
it). `unitree_api` is not built; the board has it.

### Input fixes found on the board

The network on the FPGA computes what the laptop's fixed-point emulation computes only once
these are right (`docs/fpga-zones3-2026-11-22.md` sections 10-12):

| Problem | Fix | Where |
|---|---|---|
| despot sends **sums** per 4x4 block; zones3 was trained on the mean | `scaling: 256` instead of 4096 in the server config (4096 / 16) | `models/zones3/config.yaml` |
| despot's noise filter removes events the training data had, and empty frames are skipped (15-30 frames/s, the behavior watchdog fired) | remove `noise_filtering_factor`; 100 frames/s | `metavision_camera.json`, `spleat-bundle --no-noise-filter` |
| the FPGA reads its input as **HWC**; despot sends **CHW** | `camera_tap` reorders the payload in place (`to_hwc`, default on) | `camera_tap_node.py`, test `tests/test_camera_tap_reorder.py` |
| a slow FPGA holds despot's TCP stream back and decisions lag further and further | `camera_tap` drops frames while `max_inflight` are unanswered (4 in the launch file) | `kria.launch.py` |
| riverops 0.2.0 exports signed weights; the board reads two's complement | `spleat-bundle` and `reflex.river.train` write two's complement (0-255, -16 → 240) | `reflex/deploy/spleat_bundle.py` |

### Readout: spikes to a decision

`reflex/runtime/zones_readout.py`. The server sends one spike count per output neuron per
frame. `fpga_perception` sums each neuron over the last `readout` frames and takes the
largest; ties go to `none`, and a zone must beat `none` by `margin` spikes. Confidence is the
winner's spikes divided by `readout` (the share of frames it fired in). For 6-zone networks
the winner is then majority-voted (`vote_window`, `vote_min`, `reflex/runtime/zone_vote.py`)
and held for `zone_hold_s` before `none` is published, because the network fires in bursts of
0.1-0.2 s, too short for the follow node's gestures. Zones map to decision labels by column
(`*_left` → `left`, `*_middle` → `center`, `*_right` → `right`).

| | zones3 | ball_v3 (`models/ball_v3/run_args`) |
|---|---|---|
| frames per decision | 20 (10 ms each) | 4 (the checkpoint's time steps) |
| margin | 0 | -10: any zone spike beats `none`, which fires every frame |
| vote | not used (3 classes) | none (`vote_window` 1) |
| hold | n/a | 0.5 s |
| behavior `min_confidence` | 0.45 in the launch file: zone decisions reached 0.5-0.55 with balls thrown past the camera; not validated | same |

## 6. The networks

| Network | Code | Input → output | Trained on | Runs where | State |
|---|---|---|---|---|---|
| Reflex SNN, 26.5k params | `reflex/models/snn.py`, `reflex/train.py` | 50 ms window, T x 2 x H x W → 4 classes | simulated episodes with hard negatives; optional fine-tune on webcam takes | PyTorch on CPU, `perception detector:=snn` | 81 % window accuracy on held-out simulated episodes; simulation only |
| zones3 | `reflex/river/` (port of Durance's Motion Buddy notebook) | 2x80x80, 10 ms frames → 4x1x1 `none/left/center/right` | 21 Nov real `cross` and `none` takes, auto-labelled per frame | **FPGA**, firmware job #44 | board output = emulation on synthetic frames; real camera at 100 frames/s, `none` with nothing in view, brief zone decisions with a ball; accuracy not measured. Validation: 81 % balanced accuracy, 38 % false triggers (same room, optimistic) |
| ball_v3 (zone classifier) | `Training/train_ball_zones.py` | 2x320x320 binary, 5 ms → 7x1x1 (2x3 zones + `none`) | teammates' dataset (big balls, zoomed crops) | **FPGA**, firmware `models/firmware/ball_zones_v3` (job number not recorded); camera at 50 Hz | output identical to the integer emulation on real frames; drives the simulated dog's look mode; reacts only to a ball close to the camera and moved fast |
| ball (U-Net) | `Training/train_ball_size.py` | 1x320x320 → 9x80x80 heatmap | Tübingen event-camera ball dataset | CPU, in the FPGA's integer arithmetic (`detector:=ball`) | full-size firmware failed place-and-route twice |
| ball_small (U-Net, ~5x fewer weights) | `Training/train_ball_small.py` | 320x320 → 9x80x80 | same | FPGA job #35 | **hangs the FPGA**; Durance's example runs on the same board, so the cause is the architecture (`docs/fpga-run-2026-11-22.md`) |

The pattern so far: the two networks shaped like Durance's example (a plain chain of
convolutions, a 1x1 output) run on the FPGA; the U-Nets with skips, upscaling and a spatial
output do not.

`reflex/runtime/river_net.py` runs any riverops export (`*_model.json` + `*_parameters.json`)
in the board's integer arithmetic in plain PyTorch. It is the reference every board result
is compared against (`tools/emulate_river_poke.py`, `tools/emulate_poke.py`).

## 7. Putting a network on the FPGA

Full reference: `docs/deploy-fpga.md`. The sequence used for zones3 and ball_v3:

1. **Train** with riverops: `python -m reflex.river.train ...` (zones3) or a script in `Training/`.
2. **Export** the River Model Graph and fixed-point parameters. For the `Training/` checkpoints:
   `pixi run --manifest-path tools/river/pixi.toml export-ball CKPT.pt models/<name>`, which also writes `PROVENANCE.md`.
3. **Bundle** for the board: `pixi run spleat-bundle MODEL.json PARAMS.json --firmware models/firmware/<job> [--frequency HZ] [--no-noise-filter]`
   writes `out/spleat/<model>/`: two's-complement `weights.txt`, the server's `config.yaml`
   (input shape, `scaling`, output channels), despot's camera config and a `deploy.yaml` for river-tools.
4. **Firmware**: submit the model JSON on riversdk.dev and download the bitstream by hand.
   One firmware serves any weights of the same architecture; a new architecture needs a new job.
   Record the job and the model's sha256 in `models/firmware/<job>/PROVENANCE.md`.
5. **Deploy**: `pixi run --manifest-path tools/river/pixi.toml -e tools river-tools deploy out/spleat/<model>/deploy.yaml root@192.168.77.1 --replace -y`,
   then `systemctl restart spleat-xmutil-loadapp spleat-server` on the board.
6. **Check against the emulation** before trusting anything live: `tools/spleat_poke.py` on
   the board, `tools/emulate_river_poke.py` on the laptop, same pattern and seed; then real
   recordings through `tools/replay_to_fpga.py`.
7. **Run**: `/opt/reflex/run.sh robot:=sim model:=<name>`. For a new model, add its readout
   files to `deploy.sh` and, if needed, a `run_args`.

Board tools, all standard-library Python because the board has no pip:
`spleat_sniff.py` (capture what despot sends), `spleat_poke.py` (one synthetic frame, wait for
the reply), `spleat_merge_proxy.py` (2 channels → 1, for one-channel networks), and on the
laptop `pixi run spleat-read` (decode the server's output live or from a capture).

## 8. Behavior and safety

The robot has no safety of its own in sport mode, so all of it is in our code, and none of it
replaces a person on the Unitree remote.

**Reflex** (`reflex_fsm.py`, `behavior_node.py`):
`idle → reacting (hold_s) → refractory (refractory_s) → idle`. A reaction needs `consecutive`
agreeing decisions above `min_confidence`; decisions are ignored while reacting or resting
(the dog's own motion floods the camera). `mode:=keeper` goes to the ball, `mode:=dodge` gets
out of the way. `reaction:=euler` leans the trunk (enter posture mode with Pose, then Euler
roll, max roll clamped); `reaction:=move` sidesteps with Move (lateral speed clamped). Center
is StandDown either way. If decisions stop arriving for `watchdog_s` (0.5 s), StopMove is sent
once. `enabled:=false` logs instead of publishing.

**Follow** (`follow.py`, `follow_node.py`): `disabled → looking → moving → settling → looking`,
and `halted` once the travel budget is used up. Stop-and-look, because an event camera on a
moving dog sees the whole room move. Three modes: `walk` (short Move bursts ended by
StopMove, forward only, towards a target roughly ahead, stops when it is close), `gaze`
(Euler yaw only) and `look` (nine fixed gestures, feet planted, hysteresis between them, the
camera-on-trunk correction so looking at the target does not make the dog look back). Three
inputs: `input:=target` (`/reflex/target`), `input:=decision` (`/reflex/decision`, e.g. zones3
on the FPGA, mapped to the level gestures) and `input:=zone` (`/durance/ball_zone`, mapped to
zone centres by `zone_target.py`). Starts disabled; StopMove on shutdown.

**Sport-mode commands** (`reflex/runtime/sport_api.py`): every `api_id` is read from
`unitree_ros2` at commit `668d1ec` and recorded with its constant name; never write one from
memory. Used here: Damp 1001, BalanceStand 1002, StopMove 1003, StandUp 1004, StandDown 1005,
RecoveryStand 1006, Euler 1007, Move 1008, Pose 1028, plus gestures and (behind
`allow_acrobatics:=true`) jumps and flips. Euler only acts in posture mode and Move only walks
outside it; `requests_for_pose` handles entering and leaving posture mode.

## 9. Simulation

`reflex/sim/scene.py` is a standing Go2 in MuJoCo with a ball launcher and a head-mounted
camera rendered at 500 Hz of simulated time; `reflex/events/emu.py` turns the frames into
events (a reduced ESIM/v2e model, not the real sensor). By default the dog does not walk:
reflexes are posture changes, sport mode's posture mode is modelled (`sim_sport.py`) and a
Move slides the standing dog with the real Go2's measured sidestep profile. Optional walking
is a learned policy (`locomotion.py`, `models/go2_walk.onnx`, trained with unitree_rl_mjlab),
off unless `locomotion:=rl`. `scenery.py` adds an optional park; the default lab scene stays
as it is.

Also in `reflex/sim/`: simulated tiny-whoop drone flights for training data
(`drone_behaviors.py`, `drone_recording.py`, `docs/simulation-drone.md`), headless follow
loops without ROS (`follow_sim.py`, `follow_demo.py`) and a second follow behavior
(`reflex/runtime/follow_fsm.py`: turn in place, greet when the ball stops), not wired into ROS.

## 10. Data

| Data | Format | Made by | Used by |
|---|---|---|---|
| Simulated episodes (approach, miss, crossing, slow, walker, idle) | `.npz` shards of windows | `pixi run gen-data` (`reflex/data/`) | reflex SNN |
| Webcam takes over ROS | `clip_NNN.npz` | `ros-record` | `build-real`, `finetune-real` |
| Real GenX320 takes | one HDF5 per take (`/events/t,x,y,p` + attributes, `reflex/events/h5.py`), `takes.csv` per session | `python -m capture.record` (Durance's `river-tools`, no ROS) | `tools/river_layout.py` → `reflex.river` |
| Auto-labels per 10 ms frame | in memory, reviewed as sheets | `reflex/river/autolabel.py`, `review.py` | zones3 training; `export_positions.py` for the ball SNNs |
| Synthetic river recordings | `.bin`/`.h5` + ground truth | `reflex/river/synth.py` | pretraining |

Recordings, datasets and river-tools are never committed (`data/` is ignored). The real
takes are not yet usable by `build-real`, which still expects the ROS recorder's format.

## 11. Code layout

| Path | Contents |
|---|---|
| `reflex/events/` | event format (`EVENT_DTYPE`: t in µs, x, y, p), emulator, HDF5 / `.bin` I/O, sources, representations |
| `reflex/data/` | simulated episodes, windowing, dataset generation, recorder format |
| `reflex/models/`, `reflex/train.py`, `reflex/eval.py` | reflex SNN, training, evaluation against the no-learning detector |
| `reflex/river/` | riverops training of the target-zone SNN, synthetic data, auto-labels, review |
| `reflex/runtime/` | detectors, readouts (`zones_readout`, `ball_readout`), `river_net` emulation, `spleat_client`, reflex FSM, follow controller and FSM, target contract, zone vote, sport-mode table |
| `reflex/deploy/` | `spleat_bundle.py`: riverops export → files for SPLEAT |
| `reflex/sim/`, `reflex/viz/` | MuJoCo scene, viewers, drone and follow simulations; web-viewer geometry |
| `Training/` | teammates' ball detectors (U-Nets, zone classifier) with int8 quantization-aware training |
| `capture/` | recording, replay, relabelling and summaries of real event-camera takes |
| `ros2_ws/src/` | `reflex_msgs`, `reflex_ros` (nodes and launch files) |
| `models/` | exports with `PROVENANCE.md`: `zones3`, `ball`, `ball_small`, `ball_v3`, `firmware/<job>`, `snn.json`, `go2_walk.*` |
| `scripts/`, `scripts/kria/` | setup, Kria sync, cross-compile and deploy, `run.sh` for the board |
| `tools/` | board probes, emulation comparisons, replay into the FPGA, `tools/river/` (river-tools environment, Python 3.10) |
| `docker/kria/` | older camera-side container for the board (webcam rehearsal) |
| `tests/` | 201 unit tests, no ROS needed |

Environments (pixi): `default` (Python, MuJoCo, PyTorch with CUDA), `ros` (adds ROS 2 Humble
from RoboStack with CycloneDDS), `kria` (arm64, camera side). `tools/river/` is a separate
pixi project for Durance's river-tools.

## 12. Open

- Our nodes have not driven the real Go2. Next step: `run.sh robot:=real` with the behavior
  disabled, read what it would send, then `enabled:=true` with someone on the remote; for
  look mode, `ros-look dry_run:=true` first. Yaw and pitch signs and reach of the look
  gestures are not checked on the real robot.
- No accuracy figure for live decisions from either FPGA network.
- ball_v3 only sees big, close, fast balls; it needs fine-tuning on our camera's data (small
  balls at the real distance, one step per frame with state kept). Same architecture, so only
  `weights.txt` changes.
- Image orientation through despot is unchecked.
- The U-Net hang on the FPGA is narrowed to its architecture, not to a layer.
- ball_v3's firmware job number is not recorded.
- Several ROS paths (`events_sim`, `ball_target`, `ball_zone_sim`) are unit-tested but have
  not been run under ROS.
