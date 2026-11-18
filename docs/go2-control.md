# Controlling the Go2

Notes on how the robot is commanded and how our simulation relates to it. Statements
marked **unverified** have not been checked against a real robot or an official source.

## Two ways to control the robot

| | High level: sport mode | Low level: motor control |
|---|---|---|
| You send | A behavior: "move at 0.3 m/s", "stand down" | A target for each of the 12 motors, at about 500 Hz |
| Balance and gait | Unitree's controller | Yours |
| ROS 2 topic | `/api/sport/request` | `/lowcmd` |
| Message | `unitree_api/msg/Request` (`api_id` + JSON parameters) | `unitree_go/msg/LowCmd` (position, velocity, torque, gains per motor) |
| Feedback | `sportmodestate` | `lowstate` (also `lf/lowstate` at low frequency) |
| Example in `unitree_ros2` | `example/src/sport_mode_ctrl.cpp` | `example/src/low_level_ctrl.cpp` |

Topic and message names above were read from `unitree_ros2` at commit `668d1ec`.

Before using low-level control, sport mode has to be switched off so the two controllers
do not fight over the motors. The exact procedure for the Go2 at the hackathon is
**unverified**; ask before trying.

### Why this project uses sport mode

- **Time.** Low level means writing or training a controller that keeps the dog upright
  and moves it. That does not fit in 30 hours next to the camera, the SNN and the FPGA.
- **The track is about perception**, not locomotion. Sport mode gives the motion for free.
- **Safety.** A low-level bug can drop the robot or strain a motor. The robot is borrowed
  and shared.
- **Reliable demo.** Unitree's motions are tested; ours would not be.

### What sport mode costs: reaction time

Sport mode adds its own latency and ramps. The plan assumes about 300 ms from command to
motion; that number is an estimate, **unverified**, and is measured on Saturday. A
low-level "flinch" could start within tens of milliseconds, which would look much more
like a reflex. The camera and the SNN react in milliseconds, so the dog may be the
bottleneck.

How the plan deals with it:

1. Measure the real command-to-motion delay before deciding anything.
2. If a lateral step is slow, use posture commands (`Euler`), which should start sooner.
3. In the pitch, report perception latency separately from robot latency.

Middle path worth keeping in mind: stay in sport mode for everything, and if time and the
organizers allow it, try one single low-level gesture with the robot secured. Our
simulation already does exactly that (joint PD control), so starting code exists in
`reflex/sim/scene.py`.

## Sport-mode commands

The verified table lives in `reflex/runtime/sport_api.py`, with a test that compares it
to Unitree's header. Commands used or likely to be used:

| Command | `api_id` | Use |
|---|---|---|
| `Damp` | 1001 | Motors go limp: emergency stop |
| `BalanceStand` | 1002 | Stand, ready to move |
| `StopMove` | 1003 | Stop the current motion |
| `StandUp` / `StandDown` | 1004 / 1005 | Get up, lie down |
| `RecoveryStand` | 1006 | Get up after a fall |
| `Euler` | 1007 | Lean the trunk (roll, pitch, yaw) |
| `Move` | 1008 | Walk (vx, vy, vyaw) |
| `Hello` | 1016 | Wave |

Acrobatics present in the header, not in our table: `FrontFlip` 1030, `FrontJump` 1031,
`FrontPounce` 1032, `LeftFlip` 2041, `BackFlip` 2043, `HandStand` 2044. Whether the Go2 X
at the hackathon has them enabled is **unverified**. They are unsuitable as reflexes
(slow, unsafe near people, the camera sees the ceiling) but could work as a celebration
after a save.

There is no `BodyHeight` command in the client at this commit.

## Simulation: what we have and what is on the radar

### What we use now

Plain MuJoCo with the Go2 model from MuJoCo Menagerie. The dog holds one of four
hand-picked poses (`stand`, `crouch`, `lean_left`, `lean_right`) through a joint PD
controller. It does not walk. A bridge node will accept sport-mode requests and translate
them into those poses, so the behavior node publishes the same message to the simulator
and to the robot.

This validates message format, reflex logic, safety limits and rough timing. It does not
validate how the real Go2 moves.

### On the radar: `unitree_mujoco`

https://github.com/unitreerobotics/unitree_mujoco is Unitree's own simulator, built on
MuJoCo and `unitree_sdk2`.

- **Closer to the real robot at low level.** The simulated robot speaks the same DDS
  topics as the real one (`lowcmd`, `lowstate`), so a low-level controller can move from
  simulation to robot unchanged.
- **Not closer at high level.** It does not implement sport mode, so `Move`, `StandDown`
  and the rest do nothing there. Our bridge would still be needed.
- **Heavier to set up.** It needs `unitree_sdk2` built and DDS configured, and has no event
  camera or ball launcher.

Not installed or tried here; the points above come from its documentation.

Switch to it if we decide to do low-level control on the real robot: that is the case it
was built for. For the sport-mode plan it adds setup cost without adding fidelity.

### Optional: a dog that walks in simulation (`unitree_rl_mjlab`)

A dog that actually walks in simulation needs a locomotion policy. Unitree's
https://github.com/unitreerobotics/unitree_rl_mjlab trains one with reinforcement
learning (PPO, 4096 parallel robots on the GPU). It ships no pretrained Go2 policy, so it
has to be trained: `scripts/setup_rl.sh` installs it and prints the training command.

Measured on the RTX 5090: 0.61 s per iteration, about 2 hours for the default 10001
iterations. How many iterations are needed for a usable walk is **unverified**.

The trained policy (from `deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml`):

- Runs at 50 Hz. Input, 47 numbers: base angular velocity (3), projected gravity (3),
  velocity command vx, vy, yaw rate (3), gait phase (2), joint positions relative to
  default (12), joint velocities (12), previous action (12).
- Output, 12 numbers: joint position targets = default pose + 0.25 x action, tracked by a
  PD controller with gains 20/20/40 and damping 1/1/2 (abduction/hip/knee).
- Command ranges: vx -0.5..1.0 m/s, vy -0.5..0.5 m/s, yaw rate -1..1 rad/s.

Not integrated into our simulation yet. To do that: run the exported `policy.onnx` at
50 Hz inside `Go2Sim`, build the 47-number observation, and feed it a velocity command
(keyboard, or the `Move` request from the bridge). Its Go2 model differs from the
Menagerie one we use (joint order, default pose, gains), so the safest route is to use
its `go2.xml` in our scene.

This policy is low-level control: on a real robot it replaces sport mode. We do not plan
to deploy it; it only makes the simulated dog respond to `Move` like the real one would.
