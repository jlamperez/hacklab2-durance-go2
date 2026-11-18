# hacklab2-durance-go2

Hackathon project: event-vision reflexes for a Unitree Go2. Durance AI track, HackLab #2,
CIC Berlin. Hacking runs Saturday 21 Nov 2026 11:00 to Sunday 22 Nov 17:45.

The track asks for the full loop: record event data, train a spiking neural network,
deploy it on a Kria KV260, and make the Go2 react through ROS 2.

## Rules for this repo

- **Everything in English**: code, comments, docs, commit messages.
- **Say what is simulated.** Nothing here has touched the real robot, the real event camera
  or Durance's toolchain yet. Keep the status table in `README.md` honest.
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
pixi run sim            # record out/demo.mp4 (RGB + emulated events)
pixi run view           # interactive 3D viewer, needs a display
pixi run -e ros ros-build   # build ros2_ws (unitree messages, reflex_msgs, reflex_ros)
pixi run -e ros ros-sim     # closed loop over ROS 2 in simulation; args: detector:=snn|simple
                            #   max_shots:=N actuation_delay_ms:=300.0 record:=out/x.mp4
pixi run -e ros <cmd>   # anything that needs ROS 2 Humble
```

Rendering is headless (`MUJOCO_GL=egl`). To check a visual result, open `out/demo.png`.

## Architecture

One pipeline: events → SNN → reflex decision → Go2 sport-mode command.

Three seams let simulation be swapped for the real thing:

| Seam | Before the hackathon | At the hackathon |
|---|---|---|
| Event source, `reflex/events/io.py` | MuJoCo emulated camera, phone clips | Durance camera adapter |
| Inference backend, `reflex/runtime/` | PyTorch, ONNX on CPU, simple detector | RIVERLib on the KV260 |
| Robot, topic `/api/sport/request` | `go2_sim_bridge` on MuJoCo | the real Go2, same message |

- `reflex/events/`: event format (`EVENT_DTYPE`: t in µs, x, y, p), emulator, representations.
- `reflex/data/`: simulated episodes (threats and hard negatives), windowing, dataset generation.
- `reflex/models/snn.py`: the network. `reflex/runtime/`: detectors (SNN, no-learning),
  reflex state machine, sport-mode command table.
- `reflex/sim/`: standing Go2, ball launcher, head camera. The dog does not walk in
  simulation; reflexes are posture changes. Walking is sport mode's job on the real robot.
- `ros2_ws/src/reflex_msgs`: `EventWindow`, `ReflexDecision`.
- `ros2_ws/src/reflex_ros`: three nodes. `sim_bridge` (MuJoCo behind the robot's and the
  camera's topics, with a configurable actuation delay, scores every shot), `perception`
  (window -> decision), `behavior` (decision -> `unitree_api/msg/Request`, all safety
  logic). The behavior node publishes the same message to the simulator and to the robot.
  Nodes are thin; logic lives in `reflex/` where it is unit-tested without ROS.

Frame convention: dog at the origin, +x forward, +y left, +z up. Labels
(`none / left / center / right`) are in the dog frame.

## Environment

pixi, two environments: `default` (Python, MuJoCo, PyTorch with CUDA) and `ros`
(adds ROS 2 Humble from RoboStack with CycloneDDS). Humble because it is what
`unitree_ros2` supports. Whether RoboStack's CycloneDDS talks to the real Go2 is
unverified; `docker/` holds the fallback.

## What we do not know yet

See `docs/questions-for-durance.md`. For sport mode versus low-level control, and why
`unitree_mujoco` is not used (yet), see `docs/go2-control.md`. Camera model and resolution, recording format,
RIVERLib's supported layers, and the KV260 image are all assumptions until Saturday.
