# Commands: what to run, where, and for what

Everything is started with `pixi run`. There are three families, and telling them apart
removes most of the confusion.

| Family | Looks like | Where | Uses ROS 2 | Visible in the web dashboard |
|---|---|---|---|---|
| Plain | `pixi run NAME` | PC or laptop | no | no |
| ROS | `pixi run -e ros ros-NAME` | PC or laptop | yes | yes |
| Kria | `pixi run -e kria kria-NAME` | the Kria board | yes | yes, from the PC |

- `-e ros` selects the environment that contains ROS 2. Without it, `ros2` does not exist.
- All commands are run from the repository folder (`hacklab2-durance-go2`).
- Arguments are written `name:=value` after the command (ROS family only).
- Stop anything with `Ctrl+C` in its terminal.
- ROS programs find each other on their own if they share a network and `ROS_DOMAIN_ID`
  (unset means 0, used everywhere here). Nothing has to be told an address.

## One-time setup

| Command | When |
|---|---|
| `pixi install -a` | after cloning; downloads the environments |
| `pixi run setup` | after cloning; fetches the Go2 model into `third_party/` |
| `pixi run -e ros ros-build` | after cloning, and after changing a message, a launch file or `setup.py` |
| `pixi run test` | any time; 73 tests, about 25 s |

Not in git, copy by hand to a new machine: `out/models/snn.pt` (the trained network),
`data/` (datasets and recordings).

## Plain commands (no ROS)

| Command | What it does | Opens |
|---|---|---|
| `pixi run view` | balls rolled at a standing dog; you trigger postures | 3D window |
| `pixi run drive` | walk the dog with the arrow keys | 3D window |
| `pixi run sim` | records `out/demo.mp4`: RGB and emulated events side by side | nothing |
| `pixi run gen-data` | synthetic dataset into `data/synth` | nothing |
| `pixi run train` | trains the network into `out/models/snn.pt` | nothing |
| `pixi run eval` | network versus simple detector on held-out windows | nothing |
| `pixi run view-data PATH` | browse a dataset folder or a recording | window; `--sheet`, `--video` save files |
| `pixi run build-real data/real/*` | recorded sessions into a training set, `data/real_ds` | nothing |
| `pixi run finetune-real` | the previous step, then fine-tune into `out/models/snn_real.pt` | nothing |
| `pixi run rl-play` | watch the walking policy inside mjlab (optional extra) | 3D window; `--viewer viser` for the browser |

Keys, pressed in the 3D window: `X` stand, `C` crouch, `Q` / `E` lean, `Z` sit, `H` hello.
With `drive`: arrows, `A` / `D` sideways, `SPACE` stop.

## ROS commands on the PC

Two kinds: **launchers** start part of the pipeline, **tools** attach to whatever is running.

### Launchers (pick one)

| Command | Camera | Detector | Behavior | Robot |
|---|---|---|---|---|
| `ros-sim` | simulated | yes | yes | simulated, with balls and a score |
| `ros-camera` | webcam on this machine | yes | no | no |
| `ros-robot-sim` | no | no | yes | simulated, waits for decisions |
| `ros-drive` | no | no | no | simulated, walks on `/cmd_vel` |
| `ros-drive camera:=true` | simulated, moves with the dog | yes | no | simulated, walks on `/cmd_vel` |

Only `ros-sim` is complete on its own. The others do nothing visible until their partner runs:

| If you start | It does nothing until you also start | Why |
|---|---|---|
| `ros-sim` | nothing else needed | camera, detector, behavior and robot are all inside |
| `ros-robot-sim` | a camera: `kria-camera` on the Kria, or `ros-camera` on a PC with a webcam | the dog waits for decisions on `/reflex/decision` |
| `ros-camera` / `kria-camera` | a robot: `ros-robot-sim` (or the real Go2 with the behavior node) | decisions are published but nobody acts on them |
| `ros-drive` | a keyboard: `ros-teleop` (walking) and/or `ros-tricks` (Hello, Sit) | the dog stands still until something publishes `/cmd_vel` or a trick |

`ros-web` can be added to any of them; it never replaces a partner.

Arguments (defaults in brackets):

- `ros-sim`: `detector:=snn|simple` [snn], `max_shots:=N` [0, endless], `negative_fraction:=0.3`,
  `actuation_delay_ms:=300.0`, `record:=out/x.mp4`, `mode:=keeper|dodge`,
  `reaction:=euler|move`, `locomotion:=none|rl`, `move_vy:=0.3`, `hold_s:=0.6`,
  `shot_period_s:=3.0`, `seed:=0`, `randomize:=false`.
- `ros-camera`: `device:=0` (camera number), `detector:=simple|snn` [simple], `mirror:=false`,
  `fps:=30.0`, `contrast_threshold:=0.25`, `calibrate_s:=3.0`.
- `ros-robot-sim`: `viewer:=true` (false: no MuJoCo window), `reaction:=euler|move`,
  `locomotion:=none|rl`, `actuation_delay_ms:=300.0`, `enabled:=true` (false: only log
  what it would send), `min_confidence:=0.6`, `consecutive:=2`.
- `ros-drive`: `viewer:=true`, `actuation_delay_ms:=300.0`, `timeout_s:=0.0`,
  `camera:=false` (true: the dog's simulated event camera and the detector run while you
  drive), and with the camera `detector:=snn|simple` [snn].

### Tools (add as many as you want, each in its own terminal)

| Command | What it does | Keys go to |
|---|---|---|
| `ros-web` | dashboard at `http://localhost:8080`; `port:=8081` | the browser |
| `ros-teleop` | keyboard to `/cmd_vel`; needs `ros-drive` | its terminal: `i` forward, `,` back, `j` / `l` turn, `J` / `L` sideways, `k` stop |
| `ros-tricks` | keyboard to built-in tricks; `allow_acrobatics:=true` | its terminal: `h` Hello, `s` Sit, `u` RiseSit, `!` Damp |
| `ros-record session:=NAME` | record labelled takes from the live stream | its terminal: `l` `c` `r` `n` start, `SPACE` hit, `x` cancel, `u` undo, `q` quit |
| `ros-shell` | a shell where `ros2` and `rqt` work | — |

Inside `ros-shell`:

```bash
ros2 node list
ros2 topic list
ros2 topic echo /reflex/decision
ros2 topic hz /reflex/events_window
rqt_graph
ros2 run rqt_image_view rqt_image_view /reflex/debug/events_image
ros2 service call /reflex/recalibrate std_srvs/srv/Trigger     # simple detector only
```

## On the Kria

Reach it with `ssh ubuntu@kria.local`. pixi is at `~/.pixi/bin/pixi` (not on the PATH).

| Command, run on the board | What it does |
|---|---|
| `~/.pixi/bin/pixi run --frozen -e kria kria-build` | build the workspace (after copying new code) |
| `~/.pixi/bin/pixi run --frozen -e kria kria-camera` | webcam as event camera + detector; same arguments as `ros-camera` |
| `~/.pixi/bin/pixi run --frozen -e kria kria-shell` | a shell where `ros2` works |

`--frozen` makes pixi use the lock file as it is. Always pass it on the board: its pixi is
newer than the PC's, and without the flag it may try to rewrite the lock file.

Without logging in first, straight from a PC terminal (`-t` so that `Ctrl+C` reaches the board):

```bash
ssh -t ubuntu@kria.local 'cd hacklab2-durance-go2 && ~/.pixi/bin/pixi run --frozen -e kria kria-camera'
```

From the PC: `bash scripts/sync_kria.sh` copies the code to the board and rebuilds the
Docker image; `NO_DOCKER=1 bash scripts/sync_kria.sh` copies only (then `kria-build`).
The Docker route is in `docs/real-robot.md`.

## Recipes

Each line is one terminal.

**See the whole loop work, with a score**
```bash
pixi run -e ros ros-sim
pixi run -e ros ros-web            # optional
```

**Compare the network with the simple detector**
```bash
pixi run -e ros ros-sim detector:=snn max_shots:=10
pixi run -e ros ros-sim detector:=simple max_shots:=10     # afterwards, not at once
```

**Kria as the camera, simulated dog on the PC** (the rehearsal for Saturday)
```bash
~/.pixi/bin/pixi run --frozen -e kria kria-camera     # on the Kria
pixi run -e ros ros-robot-sim                # on the PC
pixi run -e ros ros-web                      # on the PC, optional
```

**Webcam on the PC as the camera** (needs a camera plugged into the PC)
```bash
pixi run -e ros ros-camera
pixi run -e ros ros-robot-sim
```

**Drive the dog and watch it in the browser**
```bash
pixi run -e ros ros-drive viewer:=false
pixi run -e ros ros-teleop
pixi run -e ros ros-web
```

**Tricks**
```bash
pixi run -e ros ros-drive
pixi run -e ros ros-tricks
```

**Record data, then fine-tune**
```bash
~/.pixi/bin/pixi run --frozen -e kria kria-camera     # or any other camera source
pixi run -e ros ros-record session:=sat_am
pixi run -e ros ros-record session:=sat_pm   # later, a second session
pixi run finetune-real --val-session sat_pm
```

**Retrain from scratch on synthetic data**
```bash
pixi run gen-data && pixi run train && pixi run eval
```

**Drive the dog and watch its event camera** (what the sensor sees when the dog itself moves)
```bash
pixi run -e ros ros-drive camera:=true       # add viewer:=false to skip the MuJoCo window
pixi run -e ros ros-teleop
pixi run -e ros ros-web                      # events, decision and network activity
```
Nothing acts on the decisions here: there is no behavior node. Expect false alarms while
walking (measured: about 15 % of windows, none standing still): the network was trained
with a still dog, and a moving camera makes the whole scene produce events. With the
camera on, the simulation runs at about 70 % of real time.

## When something does not work

| Symptom | Cause |
|---|---|
| `ros2: command not found`, `pixi: command not found` on the Kria | not inside `ros-shell`; on the Kria use `~/.pixi/bin/pixi` and `-e kria` |
| `cannot open camera 0` | no camera on that machine, or another number: `device:=2` |
| the dog does not react | no camera source running; simple detector calibrated while something moved (call `/reflex/recalibrate`); reflexes `enabled:=false` |
| recorder says "no windows yet" | no camera source running, or a different `ROS_DOMAIN_ID` |
| the web dashboard does not move | the simulator was started with a plain command (`pixi run view` / `drive`); use the `ros-` one |
| dashboard on port 8081 | another dashboard is still running on 8080 |
| keys do nothing | `ros-teleop`, `ros-tricks`, `ros-record`: their terminal must have focus; `view` / `drive`: the 3D window must |
| nodes on two machines do not see each other | different network or `ROS_DOMAIN_ID`; check with `ros2 node list` on both |
