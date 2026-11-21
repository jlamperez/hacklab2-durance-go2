# Simulated drone recordings

Synthetic event-camera data for the planned behavior: the Go2 stands in pose mode (no walking)
and turns its "head" to follow a small drone. Everything here is simulation: MuJoCo renders
the robot's camera, an emulator turns the images into events. No behavior runs; the robot only
stands, and in `follow` / `sweep` mode the camera moves on a script.

## Run

```bash
python -m reflex.sim.drone_recording --list                                  # the behaviors
python -m reflex.sim.drone_recording --behavior orbit                        # one flight + video
python -m reflex.sim.drone_recording --behavior orbit --head follow --live   # watch it (slowed down)
python -m reflex.sim.drone_recording --behavior all --head still follow sweep --seeds 10 --no-video
```

Windows laptop: `& "$env:USERPROFILE\.venvs\reflex-sim\Scripts\python.exe" -m reflex.sim.drone_recording ...`
(about 4-12 s of wall time per simulated second with Intel Iris graphics; use the GPU machine
for large batches). Linux / pixi: `pixi run python -m reflex.sim.drone_recording ...`.

Output: `data/sim_drone/<behavior>_<head>_<seed>.h5` and `.bin`, `out/sim_drone/<...>.mp4`.

## Pieces

| Piece | Where | Notes |
|---|---|---|
| Drone | `reflex/sim/scene.py` (`set_drone`) | Sized like an Eachine UZ65 tiny whoop from approximate public specs: 65 mm motor diagonal, 31 mm two-blade props in ducts, ~85 mm across. Props spin at 250 rev/s (aliased at the render rate, like flickering prop discs). |
| Behaviors | `reflex/sim/drone_behaviors.py` | hover, cross, approach, orbit, figure8, zigzag, rise_drop, dart, leave_return, waypoints; parameters drawn per seed. Distances 0.7-3.5 m, speeds 0.3-3 m/s. |
| Camera | `MAST` in `reflex/sim/follow_demo.py` | 320x320 like the GenX320, on a mast above the front of the back (from Durance's photo), 12 deg down, 70 deg field of view (lens unknown: a guess). |
| Head | `Go2Sim.set_head`, `Head` in `drone_recording.py` | The Go2 has no head joint: "looking" is a trunk posture change (sport-mode Euler). Simulated by rotating the camera about the trunk center, legs planted. Limits 35 deg yaw, 20 deg pitch: a guess. `still`: no motion. `follow`: aims at the drone with 0.1-0.25 s delay and a rate limit (scripted from ground truth, not a behavior). `sweep`: pans and tilts on its own. |
| Sensor noise | `GENX320_NOISE` | Calibrated on our real recordings (still scene ~20-40 events/s), not the emulator default. |
| Speed-up | `Go2Sim(fast_eye=True)` | Head camera rendered without the robot's own meshes and shadows: 7 ms instead of 140 ms per image on Intel Iris. |

## Files

`.h5`: same layout as the real recordings (`/events/t, x, y, p`, attributes with
`behavior`, `head`, `seed`, `simulated`), plus ground truth per camera image:

| Dataset | Content |
|---|---|
| `/labels/t` | µs, same clock as the events |
| `/labels/x`, `/labels/y` | pixel of the drone center (-1 if behind the camera) |
| `/labels/visible` | 1 if inside the 320x320 image |
| `/labels/dist` | m from the camera |
| `/labels/head_yaw`, `/labels/head_pitch` | rad, the camera motion at that moment |

`.bin`: events only, river-tools format (`river-tools play`, Durance's notebook).

## Known limits

- Rendered images, not a real sensor: no motion blur model, no real lens, a simple emulator.
- The head is a camera rotation; the real trunk motion (and its speed, range, wobble) is not
  measured yet.
- Objects closer than about 0.4 m to the camera are clipped by the renderer (the behaviors keep
  the drone farther away).
- One indoor scene (`lab`); the drone flies against walls and boxes, not people or furniture.
