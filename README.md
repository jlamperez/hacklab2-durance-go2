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
| SNN training | not started |
| ROS 2 nodes | not started |

## Quick start

```bash
pixi install
pixi run setup   # fetches the Go2 MuJoCo model and Unitree's ROS 2 messages
pixi run test
pixi run sim     # writes out/demo.mp4: third person, dog-eye RGB, emulated events
```

Requires [pixi](https://pixi.sh). The default environment is Python only; ROS 2 Humble
lives in a separate environment (`pixi run -e ros ...`). Build the ROS 2 workspace with
`pixi run -e ros ros-build`.
