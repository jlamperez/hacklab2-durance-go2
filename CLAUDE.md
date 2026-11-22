# hacklab2-durance-go2

Hackathon project: event-vision reflexes for a Unitree Go2. Durance AI track, HackLab #2,
CIC Berlin. Hacking runs Saturday 21 Nov 2026 11:00 to Sunday 22 Nov 17:45.

The track asks for the full loop: record event data, train a spiking neural network,
deploy it on a Kria KV260, and make the Go2 react through ROS 2.

## Rules for this repo

- **Everything in English**: code, comments, docs, commit messages.
- **Say what is simulated.** On 21 Nov 2026 sport-mode commands sent by hand moved the real
  Go2 (`docs/real-robot.md`); our nodes have not driven it. The real event camera has been used
  for recording only (`capture/`, about 100 takes, camera on a stand), and on 22 Nov to feed the
  FPGA (the team also ran Durance's own example model on the board). The small ball detector
  was loaded on the FPGA on 21 Nov; on 22 Nov, fed by the real camera through
  `tools/spleat_merge_proxy.py`, one inference came back, then the FPGA hung; Durance's example
  runs on the same board, so it is our bitstream (`docs/fpga-run-2026-11-22.md`). Keep the status table in `README.md` honest.
- **Never write a Unitree `api_id` from memory.** Each one is checked against
  `ros2_sport_client` in `unitree_ros2` and recorded with its source file and commit.
- Keep the model small and in plain PyTorch. It has to be ported to Durance's RIVERLib,
  whose supported layers we do not know yet. Assumptions are config, not hard-coded.

## Commands

```bash
pixi run setup          # fetch the Go2 MuJoCo model into third_party/
pixi run test           # unit tests
pixi run gen-data       # synthetic dataset -> data/synth
pixi run train          # train the SNN -> out/models/snn.pt
pixi run eval           # SNN vs simple detector on held-out windows
pixi run build-real data/real/*   # recorded sessions -> data/real_ds
pixi run finetune-real  # build + fine-tune the SNN on recorded data -> out/models/snn_real.pt
pixi run view-data PATH # look inside a dataset folder or an event recording; --sheet/--video save files
pixi run --manifest-path tools/river/pixi.toml export-ball CKPT.pt [OUT_DIR] [--script S --commit C]  # checkpoint -> models/ball*/ + PROVENANCE.md
pixi run spleat-bundle MODEL.json PARAMS.json [--firmware DIR --checkpoint PT]  # -> out/spleat/<model>/ + deploy.yaml for river-tools
pixi run spleat-read [--host H --save F | --file F]  # read the FPGA results from the Kria (port 3333), or decode saved ones
pixi run --manifest-path tools/river/pixi.toml -e tools river-tools deploy out/spleat/<model>/deploy.yaml root@192.168.77.1  # Durance's tool, Python 3.10
pixi run sim            # record out/demo.mp4 (RGB + emulated events)
pixi run view           # interactive 3D viewer, needs a display
pixi run drive          # same viewer with the walking policy: arrow keys drive the dog (optional mode)
pixi run -e ros ros-build   # build ros2_ws (unitree messages, reflex_msgs, reflex_ros)
pixi run -e ros ros-sim     # closed loop over ROS 2 in simulation; args: detector:=snn|simple|ball
                            #   max_shots:=N actuation_delay_ms:=70.0 record:=out/x.mp4
                            #   viewer:=true (MuJoCo window) scene:=park
pixi run -e ros ros-camera     # camera side: webcam as stand-in event camera + perception
pixi run -e ros ros-robot-sim  # robot side: behavior + MuJoCo dog acting only as the robot
pixi run -e ros ros-drive      # drive the walking dog over ROS 2; second terminal: ros-teleop
                               #   camera:=true adds the simulated event camera + detector while driving
pixi run -e ros ros-record session:=NAME   # record labelled takes from the live stream (docs/recording.md)
pixi run follow-sim     # dog follows a scripted ball, headless, no ROS; --video out/follow.mp4
pixi run -e ros ros-follow-sim # simulated dog + camera + scripted ball + target node (docs/follow.md)
pixi run -e ros ros-events-sim # recordings (source:=data/river/...) or live KV260 (source:=live) drive the
                               #   simulated dog; behavior:=follow|reflex, tracker:=centroid|snn (docs/events-sim.md)
pixi run -e ros ros-follow     # follow node, keyboard: f start, SPACE stop, ! Damp; starts disabled
pixi run -e ros ros-tricks     # keyboard for built-in tricks (Hello, Sit, ...); acrobatics off by default
pixi run -e ros ros-web        # dashboard in the browser (viser), http://localhost:8080; listens only
pixi run -e ros ros-shell      # shell with the workspace sourced, for ros2 / rqt commands
pixi run -e ros <cmd>   # anything that needs ROS 2 Humble
```

Real event camera (laptop with the Kria plugged in; Python env with Durance's `river-tools` and
`h5py`, not pixi; see `docs/recording-real-data.md`):

```bash
python -m capture.record --session NAME --plan follow|cross|center   # guided, labelled takes -> data/river_samples/NAME
python -m capture.replay data/river_samples/NAME                    # play takes with labels and marks
python -m capture.relabel data/river_samples/NAME SRC DST --note .. # fix a take's label
python -m capture.summarize --plan-session NAME                     # what is recorded / missing
python -m capture.bin_to_h5 FOLDER                                  # river-tools .bin -> .h5
```

Rendering is headless (`MUJOCO_GL=egl`). To check a visual result, open `out/demo.png`.

## Architecture

One pipeline: events → SNN → reflex decision → Go2 sport-mode command.

Three seams let simulation be swapped for the real thing:

| Seam | Before the hackathon | At the hackathon |
|---|---|---|
| Event source, `reflex/events/io.py`, `sources.py` | MuJoCo emulated camera, phone clips | `event_camera` node: GenX320 recordings, live KV260 stream |
| Inference backend, `reflex/runtime/` | PyTorch, ONNX on CPU, simple detector | SPLEAT server on the KV260 (RIVERLib?) |
| Robot, topic `/api/sport/request` | `go2_sim_bridge` on MuJoCo | the real Go2, same message |

- `reflex/events/`: event format (`EVENT_DTYPE`: t in µs, x, y, p), emulator, representations.
- `reflex/data/`: simulated episodes (threats and hard negatives), windowing, dataset generation.
- `reflex/models/snn.py`: the network. `reflex/runtime/`: detectors (SNN, no-learning, and
  the ball detector: Durance's riverops U-Net from `models/ball/`, run by `river_net.py` in
  the FPGA's integer arithmetic), reflex state machine, sport-mode command table.
- `reflex/sim/`: standing Go2, ball launcher, head camera. `scenery.py` is an optional
  outdoor scene (`scene:=park` / `--scene park`); the default lab scene must stay as it is. By default the dog does not walk:
  reflexes are posture changes, and a Move slides the standing dog with the real Go2's
  sidestep profile (`Go2Sim.set_slide`). Sport mode's posture mode is modelled
  (`sim_sport.PostureMode`). Optional walking (`locomotion.py`, `models/go2_walk.onnx`,
  a policy trained with unitree_rl_mjlab) is off unless `enable_walking()` / `locomotion:=rl`
  is used; keep it optional. Details in `docs/go2-control.md`.
- `capture/`: recording real event-camera data with Durance's `river-tools` (no ROS); file
  format in `reflex/events/h5.py` (one HDF5 per take: t, x, y, p + attributes). Never commit
  river-tools, Durance's dataset or recordings (`data/` is ignored).
- `ros2_ws/src/reflex_msgs`: `EventWindow`, `ReflexDecision`, `Target`.
- Following (`docs/follow.md`): `reflex/runtime/target.py` (the `u, v, confidence` contract and
  the centroid tracker), `reflex/runtime/follow.py` (move-stop-look controller, all safety
  limits), nodes `target` and `follow`. `reflex/runtime/ball_snn.py` + node `ball_target`: Theodor's
  ball SNN (`Training/` from feat/training, needs riverops) as a second target source, fed by
  `/reflex/events_fullres` (`tracker:=snn`, docs/events-sim.md).
- `ros2_ws/src/reflex_ros`: three nodes. `sim_bridge` (MuJoCo behind the robot's and the
  camera's topics, with a configurable actuation delay, scores every shot), `perception`
  (window -> decision), `behavior` (decision -> `unitree_api/msg/Request`, all safety
  logic). The behavior node publishes the same message to the simulator and to the robot.
  Nodes are thin; logic lives in `reflex/` where it is unit-tested without ROS.

Frame convention: dog at the origin, +x forward, +y left, +z up. Labels
(`none / left / center / right`) are in the dog frame.

## Environment

pixi, three environments: `default` (Python, MuJoCo, PyTorch with CUDA), `ros` (adds
ROS 2 Humble from RoboStack with CycloneDDS), both linux-64 only, and `kria` (arm64 and
linux-64: ROS 2 + OpenCV, camera side only, for the board). Humble because it is what
`unitree_ros2` supports. Whether RoboStack's CycloneDDS talks to the real Go2 is
unverified; `docker/` holds the fallback.

## Durance's toolchain

Notes from Durance's login-only docs (Kria login, Go2 networking, cross-compiling the ROS 2
node, deploying an SNN to the FPGA, camera config): `docs/durance-riversdk.md`. Read it
before working on the Kria, the real robot or the FPGA export. Where it conflicts with the
assumptions below, it wins.

## What we do not know yet

What is left to run on the real robot, and the Kria + webcam rehearsal: `docs/real-robot.md`.
How to look inside the running ROS 2 system: `docs/inspecting-ros.md`.
Every command, where it runs and recipes: `docs/commands.md` (keep it in sync with `pixi.toml`).
Putting a network on the FPGA (files, tools, board paths, steps): `docs/deploy-fpga.md`.
See `docs/questions-for-durance.md`. For sport mode versus low-level control, and why
`unitree_mujoco` is not used (yet), see `docs/go2-control.md`.

Durance's camera client on the board (`despot_send_and_forget`: 2 polarity channels, fixed),
and the adapter for one-channel networks: `docs/despot.md`.

Durance's KV260 image was read from its microSD on 21 Nov, not booted:
`docs/durance-kria-image.md`. It has ROS 2 Humble, OpenEB, Prophesee drivers (GenX320 and
IMX636) and an FPGA inference server, SPLEAT (`backward-server`). Still unknown: which
camera is fitted, SPLEAT's protocol, supported layers and weights format, and how it
relates to RIVERLib.
