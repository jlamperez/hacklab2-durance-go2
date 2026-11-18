# From simulation to the real robot

Status on 19 Nov 2026: everything runs in simulation. Nothing has touched the real Go2,
the real event camera, Durance's toolchain, or a Kria board.

## Target layout

```
        ------------- Kria KV260 -------------        -- Kria or laptop --      -- Go2 --
camera -> [camera adapter] -> /reflex/events_window
                              [perception + SNN] -> /reflex/decision -> [behavior] -> /api/sport/request -> robot
```

`go2_sim_bridge` disappears: in simulation it plays both the camera and the robot.

## What is left

| Piece | State | Left to do | Blocked on |
|---|---|---|---|
| Camera adapter | Template only: `camera_webcam_node.py` (webcam + emulator) | Replace `read_events` with the real sensor's SDK; keep windowing and publishing | Camera model, SDK, event format |
| Perception node | Works with `snn` (PyTorch) and `simple` | Third detector backend that runs the network on the FPGA | RIVERLib |
| Network | Trained on simulated events only | Fine-tune on real recordings; port to RIVERLib; `pixi run finetune-real` not written yet | Real data, RIVERLib layer support |
| Recording and labelling tool | Not written | Record events to `.npz` by session, label threat / side | Camera adapter |
| Behavior node | Done, tested against the simulator | No code. Check on the robot: Euler roll sign, which states accept each command | Robot |
| Launch for the real setup | `camera.launch.py` (camera side) exists; behavior runs alone with `ros2 run reflex_ros behavior` | A `real.launch.py` once the camera adapter exists | - |
| Network / DDS | Works on one machine only | Kria, laptop and Go2 on the same network (Go2: `192.168.123.x` over Ethernet, CycloneDDS). RoboStack's CycloneDDS against the Go2 is unverified | Robot |
| Kria software | Nothing installed | ROS 2 Humble + `reflex` + the two ROS packages on arm64; `docker/kria` not written | Durance's KV260 image |
| PC fallback | Not written | `docker/pc-humble`, in case RoboStack's DDS does not talk to the Go2 | - |

## First hour with the robot, in order

1. Find and test the emergency stop. Fenced area. Rolled balls, not thrown.
2. Robot standing in sport mode. Run behavior with `enabled:=false`: it logs what it would
   send and publishes nothing.
3. Send one lean by hand and watch the direction. If it leans the wrong way, flip the sign
   in `behavior_node.requests_for`.
4. Record the idle dog (standing, nothing in front). Its balance control already produces
   events: this sets the simple detector's threshold and becomes training negatives.
5. Measure command-to-motion delay and put it in `actuation_delay_ms` of the simulator.

## Rehearsal without Durance hardware: Kria + webcam

Goal: prove the part that does not depend on Durance. A Kria board runs the camera side,
a PC runs the robot side, and they talk over ROS 2 on the network.

A webcam at 30 fps is a poor event camera. This tests plumbing (arm64 install, DDS between
two machines, message sizes, latency), not detection quality.

Tested so far (19 Nov): both sides on the same PC with a Logitech C920. 30 windows/s,
decisions reach behavior, the simulated dog reacts, 16-28 ms from frame to request.
**Not tested: anything on a Kria, anything across two machines.**

### PC (robot side)

```bash
export ROS_DOMAIN_ID=42
pixi run -e ros ros-robot-sim            # behavior + MuJoCo dog in a 3D window
```

### Kria (camera side)

Assumes Ubuntu 22.04 for Kria with ROS 2 Humble from apt. The camera side needs only
numpy, OpenCV and ROS; no PyTorch (with `detector:=simple`), no MuJoCo.

```bash
sudo apt install ros-humble-ros-base ros-humble-rmw-cyclonedds-cpp python3-colcon-common-extensions \
                 python3-opencv python3-numpy python3-pip
git clone <this repo> && cd hacklab2-durance-go2
pip install --no-deps -e .
source /opt/ros/humble/setup.bash
cd ros2_ws && colcon build --packages-select reflex_msgs reflex_ros && cd ..
source ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch reflex_ros camera.launch.py detector:=simple device:=0
```

Wave a hand on one side of the webcam: the dog on the PC should lean.

### What to check

| Check | How |
|---|---|
| The PC sees the Kria's nodes | `ros2 node list` on the PC shows `event_camera_webcam` and `reflex_perception` |
| Rate holds over the network | `ros2 topic hz /reflex/decision` near 30 |
| Latency | The bridge logs "N ms from newest event to request". Only meaningful if both clocks are synchronized (chrony/NTP) |
| CPU load on the Kria | `top` while it runs |
| The SNN on the Kria's CPU | Needs PyTorch for arm64 (`pip install torch`), then `detector:=snn`. Gives a CPU baseline for `inference_ms` to compare with the FPGA later |

If the machines do not see each other: same `ROS_DOMAIN_ID`, same `RMW_IMPLEMENTATION`,
same subnet, multicast allowed by the switch and firewall (`sudo ufw allow from <subnet>`).

Only `/reflex/decision` (a few bytes) crosses the network in this layout. The 72 kB
windows stay inside the Kria, which is also how the real system should be laid out.
