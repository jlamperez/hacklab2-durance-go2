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
