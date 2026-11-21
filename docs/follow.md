# Following a ball (or a drone)

The Go2 turns towards a moving target and walks after it, slowly, on the spot it can see.
Agreed with Durance: following an object is fine, following people is not, and it has to
be safe because the robot has no safeguards of its own in sport mode.

```
 events  ->  target estimate  ->  /reflex/target  ->  follow node  ->  /api/sport/request
             (centroid, or the SNN)  u, v, confidence    move, stop, look   Move / StopMove / Euler
```

## The contract: `/reflex/target`

`reflex_msgs/Target`, defined in `reflex/runtime/target.py`:

| Field | Meaning |
|---|---|
| `header.stamp` | time of the newest event behind the estimate |
| `u` | -1 left edge ... +1 right edge of the image, as the camera sees it |
| `v` | -1 top edge ... +1 bottom edge. For a ball on the floor, larger = closer |
| `confidence` | 0..1. 0 means nothing seen, and `u`, `v` mean nothing |
| `source` | `centroid`, `snn`, ... |

Anything that can locate the target publishes this message, and nothing downstream changes:

- the **target node** (`ros-target`, included in `ros-follow-sim`): `CentroidTracker`, no
  learning. Subtracts the still-scene background, finds the densest patch of events and
  takes its centre. Rejects frames where the events are spread over the whole image (the
  dog's own motion). Calibrates on a still scene when it starts.
- the **SNN** trained to output the ball's coordinates: its adapter (to write once the
  output format is known) converts the network's output to `u, v, confidence` and
  publishes the same message, on the PC or on the Kria. Questions for that adapter: which
  output channels hold x and y, in which range, and what says "no ball"?

## Move, stop, look

An event camera on a walking dog sees the whole room move; only a still camera sees just
the ball. So the follow controller (`reflex/runtime/follow.py`) works in bursts:

```
disabled --f--> looking --(2 confident sightings)--> moving (0.8 s) --> settling (0.8 s) --> looking ...
```

- **looking**: the dog stands still. Two confident sightings in a row trigger a burst.
- **moving**: one burst of `Move`, repeated every 0.1 s and ended by `StopMove`. Turn
  rate from the target's angle (`u` and the camera's field of view); forward speed only
  if the target is within 20 degrees of straight ahead and not close yet (`v` below
  `v_near`).
- **settling**: sightings are ignored while the request reaches the robot (actuation
  delay) and the camera calms down.

`mode:=gaze` keeps the feet still and turns the trunk instead (`Euler`, yaw only, within
`max_yaw`). It goes back to straight when nothing is seen for `lost_s`. For a hovering
drone: its propellers fire events all the time, a still ball does not.

## Safety

All in `FollowController`, plus hard caps in the node that no parameter can exceed:

| Limit | Default | Hard cap in the node |
|---|---|---|
| starts disabled, `f` to start, `SPACE` to stop, `!` Damp | | |
| forward speed, forward only | `max_vx:=0.3` m/s | 0.5 m/s |
| turn rate | `max_wz:=0.5` rad/s | 1.0 rad/s |
| one burst, always ended by StopMove | `move_s:=0.8` s | |
| commanded forward travel, then halted until `f` again | `max_travel_m:=3.0` m | |
| trunk yaw (gaze) | `max_yaw:=0.35` rad | 0.5 rad |
| stop walking when the target is close | `v_near:=0.6` | |
| ignore the upper image (people's bodies) | `zone_top:=0.3` (target node) | |

Not covered, so keep a person on the Unitree remote:

- the node crashing in the middle of a burst: whether a single `Move` keeps the real Go2
  walking until `StopMove` is not known. The node repeats Move every 0.1 s and sends
  StopMove on shutdown; check on the robot what one Move without a follow-up does.
- people: the centroid tracker follows anything that moves in the lower part of the
  image, legs included. The SNN should be trained with people walking through as "no
  target". Run the demo in a cleared area.
- obstacles: nothing looks for them.

## Running it

```bash
pixi run test                                       # includes tests/test_follow.py
pixi run follow-sim                                 # headless closed loop, no ROS: prints a summary
pixi run follow-sim --video out/follow.mp4

pixi run -e ros ros-build                           # once: new message reflex_msgs/Target, new nodes
pixi run -e ros ros-follow-sim                      # simulated dog + camera + scripted ball + target node
pixi run -e ros ros-follow                          # second terminal: f to start, SPACE to stop
pixi run -e ros ros-follow-sim follow:=true         # both in one, starts by itself
```

On the real robot (not tried yet):

1. Camera side publishes `/reflex/events_window` and the target node runs next to it, or
   the SNN adapter publishes `/reflex/target` directly.
2. On the PC connected to the Go2: `pixi run -e ros ros-follow dry_run:=true` first. It
   logs every request it would send. Move the ball and check the signs: ball on the right
   must give a negative turn rate.
3. Then without `dry_run`, with small limits: `ros-follow max_vx:=0.15 max_wz:=0.3 max_travel_m:=1.0`.
4. Set `hfov_deg` to the real lens, and `v_near` / `zone_top` to where the floor is in the
   real image (camera height and tilt decide them).
5. `mode:=gaze` first if the walking makes anyone nervous: the feet stay down.

## In simulation

`reflex/sim/follow_sim.py` moves a ball along a scripted path in front of the dog: rolling at
0.35 m/s between waypoints, pausing 1.5 s at each (invisible to a still event camera), going
left, right and away. Walking dog with the learned policy, emulated events from its head
camera, 300 ms actuation delay. Ball in view = within 30 degrees of the heading and 0.3 to
2.5 m away.

One run of 30 s each, seed 0, `pixi run follow-sim --camera-hz 50`:

| Dog | Ball in view | Median distance | Dog moved | Bursts |
|---|---|---|---|---|
| stands still (no following, baseline) | 37 % | 2.7 m | 0 m | 0 |
| follows, bursts of 0.5 s | 54 % | 2.3 m | 0.8 m | 10 |
| follows, bursts of 0.8 s (default) | 63 % | 2.2 m | 1.2 m | 10 |

It never fell and never halted. In a separate probe run (dog standing, then turning, then
walking), the tracker found the rolling ball in 114 of 115 windows at 1.4 to 2.5 m,
nothing while the ball paused, and nothing in any of the 35 windows while the dog turned.

The ball outruns the dog. One cycle (burst, settle, two sightings) takes about 1.7 s and
moves the dog at most 0.24 m, while the ball averages about 0.2 m/s with its pauses. For
the demo: roll the ball slowly and keep it within 2 m, or raise `max_vx` and `move_s`
once the real robot has been checked at the low settings.

Known gaps of the simulation: `Euler` yaw is not simulated (gaze mode only shows what it
would send); the dog standing under the walking policy sways more than a real Go2 in
balance stand, so the simulated camera is noisier after each burst than the real one
should be; the SNN is not in this loop.
