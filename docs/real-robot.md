# From simulation to the real robot

Status on 19 Nov 2026: everything runs in simulation, and the camera side has run on a
Kria KV260 with a webcam standing in for the event camera. Nothing has touched the real
Go2, the real event camera or Durance's toolchain.

21 Nov 2026: sport-mode commands sent by hand from the Kria moved the real Go2; results
below. Our nodes have not driven the robot yet. Durance's KV260 microSD image has been read. It ships ROS 2
Humble, OpenEB, Prophesee drivers and an FPGA inference server called SPLEAT. Details and
what they change: `docs/durance-kria-image.md`.

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
| Perception node | Works with `snn` (PyTorch) and `simple` | Third detector backend: a client for Durance's SPLEAT server (`backward-server`) | SPLEAT protocol, port, input shape |
| Network | Trained on simulated events only | Fine-tune on real recordings (`pixi run finetune-real`); export to SPLEAT's `weights_file` | Real data, supported layers, weights format |
| Recording and labelling tool | `ros-record`, `build-real`, `finetune-real` written and tested on the simulator stream (`docs/recording.md`) | Try with real takes; it records the windows the adapter publishes, not raw events | Camera adapter |
| Behavior node | Done, tested against the simulator | No code. Check on the robot: Euler roll sign, which states accept each command | Robot |
| Launch for the real setup | `camera.launch.py` (camera side) exists; behavior runs alone with `ros2 run reflex_ros behavior` | A `real.launch.py` once the camera adapter exists | - |
| Network / DDS | Kria <-> PC works over CycloneDDS (19 Nov) | Kria, laptop and Go2 on the same network (Go2: `192.168.123.x` over Ethernet, CycloneDDS). RoboStack's CycloneDDS against the Go2 is unverified | Robot |
| Kria software | `docker/kria` image built and run on our own KV260 (Ubuntu, camera side, no PyTorch). Durance's image read: Yocto, ROS 2 Humble, Python 3.12, no Docker, no pixi, 788 MB free | Get our ROS packages onto Durance's image, or keep only the camera and SPLEAT on the board and run perception and behavior on the laptop | Booting the image; `unitree_api` and `colcon` are not on it |
| PC fallback | Not written | `docker/pc-humble`, in case RoboStack's DDS does not talk to the Go2 | - |

## First hour with the robot, in order

1. Find and test the emergency stop. Fenced area. Rolled balls, not thrown.
2. Robot standing in sport mode. Run behavior with `enabled:=false`: it logs what it would
   send and publishes nothing.
3. Send one lean by hand and watch the direction. **Done 21 Nov 2026:** Euler `x=+0.4`
   lowers the right side, `x=-0.4` the left (right-hand rule about +x). The code assumed
   the opposite; `sport_api.requests_for_pose` and `sim_sport.pose_for_request` are fixed.
   Move `y=+0.2` steps to the dog's left, as the code assumed.
4. Record the idle dog (standing, nothing in front). Its balance control already produces
   events: this sets the simple detector's threshold and becomes training negatives.
5. Measure command-to-motion delay and put it in `actuation_delay_ms` of the simulator.

## First session with the real Go2, 21 Nov 2026

Commands sent by hand from Durance's Kria with `ros2 topic pub`; none of our nodes ran
against the robot yet.

**Wiring.** Laptop to the Kria's built-in Ethernet (`eth0`, DHCP, `192.168.77.x`). Go2 to a
USB-Ethernet adapter on the Kria, which the image configures as `192.168.123.99` by MAC
(`/etc/systemd/network/10-unitree-go2.network`). Plugging the laptop into the adapter
instead gives it no address. Go2 at `192.168.123.161`.

**Software on the Kria.** Durance's `kria-runtime.tar.gz` (from `go2-kria-generated.zip`)
extracted to `/opt/go2` provides `unitree_api` / `unitree_go`. Shell setup:

```bash
source /opt/ros/humble/setup.bash && source /opt/go2/local_setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file:///etc/cyclonedds/cyclonedds-go2.xml
ros2 topic pub --once /api/sport/request unitree_api/msg/Request "{header: {identity: {api_id: 1016}}}"
```

The board runs BusyBox: no `timeout`, and `head` needs `-n`.

**Verified.**

| What | Result |
|---|---|
| Hello, Sit, StopMove, BalanceStand, StandUp, StandDown, Move, Euler, Pose | work; robot replies on `/api/sport/response` (`code 0` accepted, `-1` refused, e.g. RiseSit when not sitting) |
| Move `x` / `y` / `z` signs | +x forward, +y the dog's left, +z turns left: as the code assumed |
| Euler roll sign | +x lowers the right side. The code assumed the opposite; fixed |
| Euler needs posture mode | after walking, Euler is accepted but ignored until `Pose` (1028, `{"data": true}`) |
| Move in posture mode | turns the trunk instead of walking. Leave with Euler 0, `Pose false`, BalanceStand |
| Turning at 0.8 rad/s | slips and twists the body; 0.5 rad/s steps cleanly |
| A Move sent at 10 Hz for 2 s (`-r 10 -t 20`) | walks for about that long |

`behavior` now sends Pose(true) before a lean and Euler 0, Pose(false), BalanceStand on
the way back (`sport_api.requests_for_pose`).

**Not measured yet.** Delay from command to motion (the simulator assumes 300 ms) and how
fast a lean completes (visibly slow on the robot). `error_code` in `/sportmodestate` reads
100 and `foot_force` reads zeros; meaning unknown.

**Event camera.** `river-tools capture --host root@durance-kria.local` shows the live
view. Install with `uv tool install --python 3.10 --with PyQt5 river_tools-*.whl` (the
wheel needs Python 3.10; without PyQt5 matplotlib opens no window). Its window steals focus
every 20 ms; run it with `MATPLOTLIBRC` pointing at a `matplotlibrc` containing
`figure.raise_window: False`.

## Rehearsal without Durance hardware: Kria + webcam

Goal: prove the part that does not depend on Durance. A Kria board runs the camera side,
a PC runs the robot side, and they talk over ROS 2 on the network.

A webcam at 30 fps is a poor event camera. This tests plumbing (arm64 install, DDS between
two machines, message sizes, latency), not detection quality.

**Done on 19 Nov 2026** with a KV260 (Ubuntu 22.04, kernel 5.15 xilinx-zynqmp, 4 GB) and
a Logitech C920, PC and board on the same home network:

| Check | Result |
|---|---|
| Image build on the board | 4.5 min, 1.29 GB (`docker/kria/Dockerfile`) |
| PC sees the board's nodes | yes, with `ufw` active on the PC and no extra rules |
| `/reflex/decision` rate at the PC | 30.0 Hz, 1.9 kB/s |
| Newest frame on the Kria to request on the PC | 6-13 ms (both clocks NTP-synchronized) |
| Simulated dog reacts to motion in front of the webcam | yes (lean, crouch, back to stand) |
| CPU on the Kria | camera node about 85 % of one core, perception about 15 % |

Not done: the SNN on the board (needs PyTorch for arm64), anything with the FPGA.

Three problems found and fixed on the way, all of which would have cost time on Saturday:

1. **Publishing a large `uint8[]` from Python was 40x too slow.** Assigning `bytes` to the
   message field makes rclpy check the 72000 elements one by one: 0.25 s per window on the
   Kria (7 windows/s instead of 30). Assigning `array.array("B", ...)` takes the fast path.
2. **Docker's default bridge had no DNS during builds on this board.** `docker build
   --network=host` fixes it; `--pull=false` avoids a registry lookup on every rebuild.
3. **An rsync exclude named `data` also dropped `reflex/data`.** Excludes are now anchored
   (`/data`).

### PC (robot side)

```bash
pixi run -e ros ros-robot-sim            # behavior + MuJoCo dog in a 3D window
```

### Kria (camera side), with Docker

No sudo needed if the user is in the `docker` group. From the PC:

```bash
scripts/sync_kria.sh ubuntu@kria.local   # copy the code (under 1 MB) and build the image on the board
ssh -t ubuntu@kria.local docker run --rm -it --net=host --device /dev/video0 reflex-kria
```

The image holds ROS 2 Humble, CycloneDDS, OpenCV and this project's camera side, with
the default ROS domain (0). No PyTorch, no MuJoCo: it runs the no-learning detector.

Wave a hand on one side of the webcam: the dog on the PC should lean.

### Kria (camera side), with pixi

No sudo and no Docker: pixi installs into the user's home, and the same `pixi.toml` has a
`kria` environment for arm64 (ROS 2 Humble from RoboStack, OpenCV, numpy; no MuJoCo, no
PyTorch). The nodes run as normal processes, so devices and the FPGA are reached directly.

```bash
# on the PC
NO_DOCKER=1 scripts/sync_kria.sh ubuntu@kria.local
# on the board
curl -fsSL https://pixi.sh/install.sh | PIXI_NO_PATH_UPDATE=1 bash     # once; installs ~/.pixi/bin/pixi
cd hacklab2-durance-go2
~/.pixi/bin/pixi install -e kria --frozen       # 4.5 min, 2.4 GB
~/.pixi/bin/pixi run --frozen -e kria kria-build     # 1 min
~/.pixi/bin/pixi run --frozen -e kria kria-camera    # arguments: detector:=simple mirror:=true
```

Tried on 19 Nov 2026 on the same KV260: 30 windows/s, decisions reach the PC at 30 Hz,
the simulated dog reacts, camera node at about 70 % of one core. Same result as Docker.
`--frozen` uses the lock file as it is (the board had a newer pixi than the PC).

| | Docker | pixi |
|---|---|---|
| Needs sudo | no (user in the `docker` group) | no |
| Size on the board | 1.3 GB image | 2.4 GB environment |
| First setup | 4.5 min | 4.5 min + 1 min build |
| Devices, FPGA | passed in explicitly (`--device`, `--privileged`) | direct |
| Board without internet | load a saved image (`docker save` / `docker load`) | not without preparing a package cache first |
| Same commands as on the PC | no | yes |

### Kria, with apt (not tried)

Needs sudo. Ubuntu 22.04 with ROS 2 Humble from apt:

```bash
sudo apt install ros-humble-ros-base ros-humble-rmw-cyclonedds-cpp python3-colcon-common-extensions \
                 python3-opencv python3-numpy
source /opt/ros/humble/setup.bash
cd ros2_ws && colcon build --packages-select reflex_msgs reflex_ros && cd ..
source ros2_ws/install/setup.bash
export PYTHONPATH=$PWD RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch reflex_ros camera.launch.py detector:=simple device:=0
```

Direct access to the FPGA is simpler this way than from a container, which may matter
for Durance's toolchain.

### What to check

| Check | How |
|---|---|
| The PC sees the Kria's nodes | `ros2 node list` on the PC shows `event_camera_webcam` and `reflex_perception` |
| Rate holds over the network | `ros2 topic hz /reflex/decision` near 30 |
| Latency | The bridge logs "N ms from newest event to request". Only meaningful if both clocks are synchronized (chrony/NTP) |
| CPU load on the Kria | `top` while it runs |
| The SNN on the Kria's CPU | Needs PyTorch for arm64 (`pip install torch`), then `detector:=snn`. Gives a CPU baseline for `inference_ms` to compare with the FPGA later |

### ROS domain

Everything uses the default domain, 0, so nothing has to be exported. The Go2 is believed
to use 0 as well (not verified). On a network shared with other ROS 2 teams, pick a
number and set it on every machine: `export ROS_DOMAIN_ID=42` on the PC and
`docker run -e ROS_DOMAIN_ID=42 ...` on the board. A mismatch gives no error, only silence.

If the machines do not see each other: same `ROS_DOMAIN_ID`, same `RMW_IMPLEMENTATION`,
same subnet, multicast allowed by the switch and firewall (`sudo ufw allow from <subnet>`).

Only `/reflex/decision` (a few bytes) crosses the network in this layout. The 72 kB
windows stay inside the Kria, which is also how the real system should be laid out.
