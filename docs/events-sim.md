# Real events, simulated dog

The `event_camera` node puts real GenX320 events on `/reflex/events_window`, the topic the
simulated camera and the webcam stand-in use. Everything downstream runs unchanged, so the
simulated Go2 shows what the follow node (or the reflex) would command for those events.

```bash
pixi run -e ros ros-build                                       # once: new node and launch file
pixi run -e ros ros-events-sim source:=data/river/sat_follow2   # a whole session, looping
pixi run -e ros ros-web                                         # optional: events + the dog in the browser
```

```
event_camera (replay or live, 320x320, windows of 50 ms at 50 Hz)
  -> /reflex/events_window
  behavior:=follow  -> target node -> /reflex/target -> follow node -> Move / StopMove -> walking dog
                       (tracker:=snn: ball_target on /reflex/events_fullres instead of the target node)
  behavior:=reflex  -> perception (simple detector) -> /reflex/decision -> behavior node -> lean / sidestep
```

## Sources

| `source:=` | What plays |
|---|---|
| `data/river` | every session, every take, in recording order |
| `data/river/sat_follow2` | one session |
| `data/river/sat_follow2/ball/take_007_center.h5` | one take (the session's still scene plays before it) |
| `a.h5,b.h5` | a list, comma-separated |
| `live` | the KV260's camera stream (below) |

Takes play back to back, `gap_s:=1.0` of silence between them, in a loop (`loop:=false`
stops after one pass). `speed:=0.5` plays at half speed. Takes that `overrides.json` drops
are left out; `exclude:='*take_002*'` leaves out more (fnmatch on the path or file name,
comma-separated; take 2 is the flicker test at 2.4 M events/s). The node logs each take as
it starts, with the instruction the operator was given, and publishes it on
`/reflex/debug/replay`.

**The still scene plays first** (`still_first:=true`). The target node and the simple
detector calibrate on the first 2.5 s they see. Calibrated on silence, the target node's
threshold sits at its floor of 5 events and the still scene alone gives 48 false sightings
(sat_follow2, at u = +0.86, likely hot pixels). Calibrated on `none/empty_000.h5` the
threshold is 11.3 and the still scenes give none.

## Live camera

Same protocol as `river-tools capture`, read from river-tools 0.1.0 (not from a spec): the
PC listens on TCP `port:=3333`, `despot_debug_camera` on the board connects and sends
batches of 13-byte events. Stop `river-tools capture` first: both want the port.

```bash
pixi run -e ros ros-events-sim source:=live kria_host:=root@durance-kria.local   # starts the sender over ssh
pixi run -e ros ros-events-sim source:=live    # or start it on the board yourself:
                                               #   despot_debug_camera --host <PC address> --port 3333
```

Or on the board, publishing for a robot side elsewhere (`kria-event-camera`, after
`despot_debug_camera --host 127.0.0.1 --port 3333`).

Windows keep moving when the scene is still and nothing arrives: the camera's clock is
mapped onto the PC's with the smallest delay seen. A camera restart (its clock jumps back)
starts over.

## What this shows and what it does not

- **Open loop for recordings.** When the simulated dog turns, the recording does not
  change. It shows which commands real events cause, not whether the dog would reach the ball.
- **Reflex:** `behavior:=reflex` works with the simple detector only. The reflex SNN was
  trained on 320x180 windows; this camera gives 320x320 (80x80 cells after downsampling).
- **Field of view.** The follow node turns `u` into an angle with `hfov_deg:=90`. The
  lens on Durance's camera is unknown; set it when known.
- **Left and right may be swapped.** In sat_follow2 the tracked ball sits right of centre
  in all three "left" takes (median u +0.08 to +0.27) and left of it in all three "right"
  takes (-0.04 to -0.13). Small offsets, consistent sign. If the zones meant the camera's
  left, the image is mirrored: `flip_x:=true`. Check with `ros-web` on one take.
- **Same topic as the real Go2.** The follow node starts enabled here and publishes on
  `/api/sport/request`. Keep the real robot off this ROS network, or use another
  `ROS_DOMAIN_ID`.

## With the SNN ball detector (`tracker:=snn`)

```bash
pixi run -e sim-arm ros-build
pixi run -e sim-arm ros-events-sim source:=data/river/sat_follow2 tracker:=snn viewer:=false
pixi run -e sim-arm ros-events-sim source:=data/river/sat_follow2 tracker:=snn mode:=look viewer:=false
pixi run -e sim-arm ros-web          # second shell (docker exec -it go2arm bash), http://localhost:8080
```

`ball_target` replaces the target node: one of Theodor's networks from `feat/training`,
loaded by `reflex/runtime/ball_snn.py`, publishes the same `/reflex/target`. Default
since 22 Nov: the zone classifier (`Training/train_ball_zones.py`, checkpoint
`out/models/ball_zones_best.pt`, architecture `zones-small-polarity-v2`, epoch 87,
validation balanced accuracy 95 % on the benchmark camera). It answers with one of six
zones (upper/lower x left/middle/right) or none; the target is the zone's centre
(u = -2/3, 0, +2/3; v = -1/2, +1/2) and the confidence is the zone's softmax probability.
Its input keeps the polarity (two channels, raw p 0 and 1; which one is ON on our GenX320
is unverified, swapping them changed nothing on sat_follow2). It needs the full
320x320 frame, which the 50 ms / 80x80 `/reflex/events_window` does not carry, so the
event camera also publishes `/reflex/events_fullres` (one bin, the last `fullres_ms`,
only while someone subscribes). Needs PyTorch, riverops (installed, or the unpacked wheel
in `riverops-0/`) and `Training/` (`git checkout origin/feat/training -- Training`).
Arguments: `ball_model:=PATH`, `ball_min_p:=0.55` (P(zone) for the zones network),
`ball_min_pixels:=20`, `fullres_ms:=30.0`.

The older position checkpoint `ball_small_best.pt` (`unet-position-small-v1`, one channel)
does not load with the current `Training/`, which builds the two-channel network; the
tracker says so and names the commit to load it with.

Zones network, without ROS, on sat_follow2 (22 Nov; frames every 20 ms with >= 20 event
pixels, ball takes; reference zone = median position of the frame's events, still
camera, so mostly the ball):

| window | P(zone) >= | zone named in | column agrees | row agrees |
|---|---|---|---|---|
| 10 ms | 0 | 38 % | 61 % | 79 % |
| 20 ms | 0 | 59 % | 80 % | 90 % |
| 20 ms | 0.55 | 30 % | 89 % | 100 % |
| 30 ms | 0.55 | 44 % | 94 % | 100 % |
| 40 ms | 0.55 | 55 % | 93 % | 100 % |

So `fullres_ms:=30` and 0.55. Flipping x did not help: no mirror between the benchmark
camera and ours. Per take at 30 ms / 0.55 (frames every 40 ms, all frames): a zone in
7-23 % of the ball takes' frames, nearly all lower zones; nothing in the still takes;
but 13-36 % in the none takes with people moving and 31 % in the drone take. Like the
position network it reacts to moving things, not only the ball; when it fires, the zone
is right. Look mode fed with it offline (follow controller, `camera_on_trunk` off): the
right takes end in `right_down`, two of three centre takes in `down`, left takes split
between `left_down` and `right_down`. A ball on the floor sits in the lower row, so look
mode only uses the three down gestures (and up ones for something high); never level.
About 20 ms a frame on x86 CPU. Not yet run under ROS.

Position network (`ball_small_best.pt`, 21 Nov). It was trained on 2 ms windows of a 1280x720 benchmark camera. Our recordings
have 1-3 k events/s, so 2 ms frames are nearly empty; 20 ms works better than 10 ms.
Without ROS, on sat_follow2 (frames every 40 ms, P(ball) >= 0.55, >= 20 event pixels), it
reported a ball in 28 % of the ball takes' frames, 16 % of the none takes' and 43 % of the
drone take's. The still takes give nothing, but take_025_none fires in about half its frames
at the image's left edge: it reacts to moving blobs, not only to the ball. Where both see
something, its position agrees with the centre of the events to about 0.06-0.10 in u, v.
About 25 ms a frame on 4 arm64 CPU cores (Docker on a Mac: not measured). Per take:
`python -m reflex.runtime.ball_snn data/river/sat_follow2` (`--zone-top 0.3` blanks the
top rows, as the centroid tracker does; it did not clearly help).

## With the zones network on the FPGA (`tracker:=fpga`)

`ball_node` (Denis's `gesture_node` package, C++, built in Durance's go2-kria workspace and run
on the Kria) reads the zones network's spikes from the SPLEAT server and publishes the
majority-voted zone as a string on `/durance/ball_zone` (`upper_left ... lower_right`, `none`),
only when it changes, transient-local. The follow node reads it directly with `input:=zone`
(the default of `ros-look`): it keeps the latest zone and feeds the controller its centre
(reflex/runtime/zone_target.py) at 20 Hz with confidence 1. `tracker:=fpga` starts the follow
node that way, without the event camera and the target node:

```bash
# on the Kria (go2-kria overlay sourced; readout file from `train_ball_zones.py export`,
# or out/models/ball_zones_readout.json)
ros2 run gesture_node ball_node --ros-args -p port:=3333 -p readout:=/root/ball_zones_readout.json
# on the Linux PC, same ROS_DOMAIN_ID and RMW (CycloneDDS) as the Kria
pixi run -e ros ros-events-sim tracker:=fpga mode:=look
# or with the real dog instead of the simulator
pixi run -e ros ros-look dry_run:=true        # f to start; drop dry_run once the log looks right
```

DDS does not leave Docker on a Mac, so this needs the Linux PC (or the nodes on the Kria).
Whether the 320x320 zones network runs on SPLEAT, and in which dtype the payload comes, is
not checked. The zone -> target -> gesture mapping is unit-tested (tests/test_zone_target.py);
the follow node's `input:=zone` and the launch option have not run under ROS.

## Without the board: a stand-in ball_node (`tracker:=fpga-sim`)

`ball_zone_sim` (node in reflex_ros, task `ros-ball-zone-sim`) publishes on
`/durance/ball_zone` with the same message and QoS as `ball_node`, so `input:=zone` can be
tried on a Mac or without the Kria. Two sources:

- `source:=snn` (default): the zones network on the CPU over `/reflex/events_fullres`, a frame
  below `min_p` / `min_pixels` counts as `none`, then a majority vote
  (reflex/runtime/zone_vote.py: a zone needs `vote_min:=3` of the last `vote_window:=5`
  decisions) and a message only when the zone changes. The board's own vote rule is unknown.
- `source:=keys`: numpad layout, 7 8 9 upper row, 1 2 3 lower row, 0 or 5 none. No vote.

```bash
# recording -> stand-in ball_node -> follow node (input:=zone, look mode) -> simulated dog
pixi run -e sim-arm ros-events-sim source:=data/river/sat_follow2 tracker:=fpga-sim mode:=look viewer:=false
# zones by hand, next to ros-robot-sim and ros-look camera_on_trunk:=false
pixi run -e sim-arm ros-ball-zone-sim source:=keys
```

`tracker:=fpga-sim` takes `ball_model`, `ball_min_p`, `ball_min_pixels`, `ball_vote_window:=5`,
`ball_vote_min:=3`. The vote is unit-tested (tests/test_zone_vote.py); the node and the launch
option have not run under ROS (22 Nov).

## Checked and not checked (21 Nov 2026)

Checked without ROS: the sources (tests/test_sources.py: protocol decoding split at every
byte, clock mapping, camera restart, replay timing and looping, take order and drops, a
real session through the windower), and sat_follow2 played through the same chain the
nodes run (replay, windower, centroid tracker calibrated like the target node, follow
controller). Ball takes give sightings in 25-45 % of windows and turns towards them;
still takes give none; the lamp take and the moving-camera takes give false sightings, as
expected for a tracker that has not learned anything.

Not checked: the node and the launch file under ROS (no ROS where they were written),
the live path with the real `despot_debug_camera` (only with a local test sender), replay
on the Kria (no h5py in the `kria` environment; live needs only NumPy).
