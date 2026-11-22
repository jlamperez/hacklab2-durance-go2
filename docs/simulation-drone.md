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
| Behaviors | `reflex/sim/drone_behaviors.py` | hover, cross, approach, orbit, figure8, zigzag, rise_drop, dart, leave_return, waypoints; parameters drawn per seed. Distances 1-5 m from the camera, weighted towards close (median about 2 m, `sample_distance`), speeds 0.3-3 m/s. `none`: no drone (negatives). |
| Flight | `Flight` in `reflex/sim/drone_behaviors.py` | The behavior's path is only where the pilot aims. The drone chases it like an underdamped controller (lags on fast moves, overshoots when stopping), drifts 5-10 cm around a hover point, jitters, bobs, makes a short jerk now and then, and tilts into its acceleration (up to 40 deg) with a wobbling heading. Measured over 20 flights: hovering about 8 cm off the point and 10 deg tilt (median), darts up to 40 cm off the path and 40 deg tilt. Parameters drawn per flight, plausible for a whoop, not fitted to the team's drone. `--calm` for a smoother pilot. |
| Room | `add_hall` in `reflex/sim/scenery.py`, `--scene hall` (default for drone runs) | Dark concrete floor (speckle, cloudy patches, saw-cut joints), four plain walls in different colours and distances, doors, posters, whiteboards, radiators, shelves and boxes, a ceiling; a patterned rug in front of the dog in half the rooms. Drawn per seed. The `lab` scene is unchanged. |
| Camera | `CAMERA_MOUNT` in `reflex/sim/drone_recording.py` | 320x320 like the GenX320, 76 deg field of view, on a small tripod 10 cm above the head and 10 cm behind its front (lens at 0.23, 0, 0.19 m in the base frame, about 0.5 m above the floor), as on the real robot. Level: the tilt is our assumption, not measured. The ball follow demo still uses its own `MAST`. |
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
| `/labels/radius_px` | half the drone's size in the image (about 9 px at 1 m, 2 px at 4 m: the drone is small) |
| `/labels/support` | events of the last 10 ms within `radius_px + 4` px of the center |

Labels come at the camera rate (500 per simulated second). Use the ones with `visible` and
enough `support` (say 5 or more): a hovering drone in front of a still camera is in view but
makes almost no events, and a window without events cannot show where it is.

`.bin`: events only, river-tools format (`river-tools play`, Durance's notebook).

## Generating a training set on several machines

`reflex/sim/drone_dataset.py` holds one fixed plan: 290 flights (the 10 behaviors with the
head still x20, following x5, sweeping x2, plus `none`, no drone, still x5 and sweeping x15),
each with its own seed and a split (232 train, 29 validation, 29 test). Every machine runs
the same command with its own number and takes every N-th flight:

```bash
python -m reflex.sim.drone_dataset --plan                    # what will be generated
python -m reflex.sim.drone_dataset --worker 0 --workers 4    # machine 1 (1, 2, 3 on the others)
```

Output: `data/sim_drone_ds/*.h5` and `.bin` plus `splits.csv`. Copy all machines' folders
into one; names never collide. An interrupted run resumes when started again (finished
flights are skipped, a half flight is not saved). One flight takes about 2-3 minutes on a
laptop in the hall scene, so 290 flights are 10-15 hours on one machine.

Not done yet: the team's ball trainer (`Training/train_ball_size.py`) reads only the
Tübingen ball dataset's layout (`projected/`, `hand_labeled/`, `gt_x/gt_y/gt_t`), not these
files. A loader for `/labels` (with the support filter and `splits.csv`) is needed first.

## Known limits

- Rendered images, not a real sensor: no motion blur model, no real lens, a simple emulator.
- The head is a camera rotation; the real trunk motion (and its speed, range, wobble) is not
  measured yet.
- Objects closer than about 0.4 m to the camera are clipped by the renderer (the behaviors keep
  the drone farther away).
- `visible` means inside the image; it does not check whether something hides the drone.
- The hall is built from boxes and noise textures: no furniture shapes, no people, no reflections.
- The flight model's numbers are plausible guesses; tune them against a real drone recording.
- Rendering in the hall is slower than in the lab (more objects and lights): about 15-20x real time on a laptop.
