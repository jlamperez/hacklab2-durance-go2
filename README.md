# hacklab2-durance-go2

Reflexes for a Unitree Go2 driven by an event camera and a spiking neural network (SNN).
Built for the Durance AI track of HackLab #2 (Berlin, 21–22 November 2026).

Pipeline: **events → SNN → reflex decision → Go2 sport-mode command over ROS 2**.

Main reflex: goalkeeper / dodge. The network classifies a fast-approaching object as
`none / left / center / right`; the dog either blocks it or steps out of the way.

## Status

Everything so far is simulation. Nothing has run on the real robot, the real event
camera or Durance's toolchain.

| Part | State |
|---|---|
| Python environment (pixi, MuJoCo, PyTorch) | working |
| Event-camera emulator (contrast-threshold model, randomized) | working, tested |
| MuJoCo scene: standing Go2, ball launcher, head camera | working, tested |
| ROS 2 Humble environment (RoboStack, CycloneDDS) | working; Unitree messages build and import |
| Sport-mode `api_id` table, checked against Unitree's client | done, tested; not tried on a robot |
| Synthetic dataset with hard negatives (miss, crossing, slow, walker, idle) | working, tested |
| SNN (23.7k parameters, plain PyTorch) and training | trains in ~90 s; 81 % window accuracy on held-out simulated episodes |
| No-learning detector and reflex state machine | working, tested |
| Closed loop (events → decision → dog reacts) | not started |
| ROS 2 nodes | not started |

## Quick start

```bash
pixi install
pixi run setup   # fetches the Go2 MuJoCo model and Unitree's ROS 2 messages
pixi run test
pixi run gen-data # 2000 simulated episodes -> data/synth (about 10 min, 63 MB)
pixi run train    # trains the SNN -> out/models/snn.pt
pixi run eval     # SNN vs the no-learning detector on the same held-out windows
pixi run sim     # writes out/demo.mp4: third person, dog-eye RGB, emulated events
pixi run view    # interactive 3D window (needs a display); keys S/C/L/R change the pose
```

Requires [pixi](https://pixi.sh). The default environment is Python only; ROS 2 Humble
lives in a separate environment (`pixi run -e ros ...`). Build the ROS 2 workspace with
`pixi run -e ros ros-build`.

## First results (simulation only)

4128 held-out windows from simulated episodes never seen in training:

| | No-learning detector | SNN |
|---|---|---|
| Missed threats | 93.5 % | 16.4 % |
| Wrong side | 1.8 % | 8.8 % |
| False alarms, all negatives | 1.1 % | 14.2 % |

The no-learning detector counts events in the lower part of the image. With the camera
shake we simulate, no threshold works: low enough to catch balls, it fires on everything
(at 100 events: 8 % missed, 81 % false alarms); high enough to stay quiet, it misses the
balls. The shake amplitude is a guess until we measure the real dog.

These are per-window numbers on emulated events. They say nothing yet about the real
camera, and the number that matters, correct reactions per ball through the state
machine, is not measured yet.
