# Reflex Arc

Reflexes for a Unitree Go2 driven by an event camera and a spiking neural network (SNN).
Built for the Durance AI track of HackLab #2 (Berlin, 21–22 November 2026).

Pipeline: **events → SNN → reflex decision → Go2 sport-mode command over ROS 2**.

Main reflex: goalkeeper / dodge. The network classifies a fast-approaching object as
`none / left / center / right`; the dog either blocks it or steps out of the way.

## Status

Early scaffold. What exists and what is simulated is tracked here as the project grows.

| Part | State |
|---|---|
| Python environment (pixi, MuJoCo, PyTorch) | working |
| Event-camera emulator | in progress |
| MuJoCo Go2 scene | in progress |
| SNN training | not started |
| ROS 2 nodes | not started |

## Quick start

```bash
pixi install
pixi run test
pixi run sim
```

Requires [pixi](https://pixi.sh). The default environment is Python only; ROS 2 Humble
lives in a separate environment (`pixi run -e ros ...`).
