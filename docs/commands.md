# Commands: what to run, where, and for what

Everything is started with `pixi run`. There are three families, and telling them apart
removes most of the confusion.

| Family | Looks like | Where | Uses ROS 2 | Visible in the web dashboard |
|---|---|---|---|---|
| Plain | `pixi run NAME` | PC or laptop | no | no |
| ROS | `pixi run -e ros ros-NAME` | PC or laptop | yes | yes |
| Kria | `pixi run -e kria kria-NAME` | the Kria board | yes | yes, from the PC |
| Capture | `python -m capture.NAME` | laptop with the Kria plugged in | no | no |
| ROS on a Mac | `pixi run -e sim-arm ros-NAME` | arm64 Docker container on Apple Silicon | yes | yes, http://localhost:8080 |

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
| `pixi run test` | any time; 86 tests, about 25 s |

Not in git, copy by hand to a new machine: `data/` (datasets and recordings).

The trained network is in git as `models/snn.pt` (with `models/snn.json`, its metrics on
simulated data). Commands default to `out/models/snn.pt`, where `pixi run train` writes;
on a new machine either `mkdir -p out/models && cp models/snn.* out/models/`, or pass
`model:=models/snn.pt`.

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
| `pixi run spleat-bundle MODEL.json PARAMS.json [--firmware models/firmware/JOB] [--checkpoint CKPT.pt]` | riverops export into the files for the KV260: `weights.txt`, `config.yaml`, `metavision_camera.json`, with `--firmware` the bitstream (refused if built from another model graph), and `deploy.yaml` for `river-tools deploy`; `--checkpoint` sets the camera frequency from the model's event window. README says where each value comes from | nothing; writes `out/spleat/<model>/` |
| `pixi run spleat-read [--host 192.168.77.1] [--save out/x.bin]` | read the FPGA results from the Kria's SPLEAT server (port 3333): size, values, `(x, y, P(ball))` per message. Needs the camera client running on the board. `--file F` decodes saved replies (`--save`, or `tools/spleat_merge_proxy.py --save` on the board). `--zones-dir models/zones3`: for the target-zone SNN, prints the 4 values and the winning zone over the last 20 frames (`reflex/runtime/zones_readout.py`; `--margin N` makes a zone beat `none` by N spikes) | nothing; prints per message |
| `scripts/kria/deploy.sh [root@192.168.77.1]` | cross-compile `reflex_msgs` and `reflex_ros` for the Kria with Durance's Docker image (`durance/go2-yocto:humble`, built once with `./manage.sh image` in their `go2-kria` folder) and install them, the `reflex` package and `models/zones3`'s readout files in `/opt/reflex` | `ros2_ws/install-yocto/`, `/opt/reflex` on the board |
| `ssh -t root@192.168.77.1 /opt/reflex/run.sh robot:=sim\|real [enabled:=true] [tap:=false]` | on the Kria: `despot_test_camera`, spleat-server on 3334, then `kria.launch.py`: despot, `camera_tap`, `fpga_perception`, `behavior`. `robot:=real` starts behavior disabled. With `model:=ball_v3` behavior runs `mode:=look` (six zones -> trunk yaw and pitch; `look_yaw:=0.15 look_pitch:=0.1 camera_on_trunk:=false invert_pitch:=false lost_s:=2.0`); `mode:=keeper` for the left / center / right reflexes | nothing |
| `pixi run -e ros ros-robot-sim behavior:=false dds_interface:=enp8s0` | laptop: the MuJoCo dog as the robot only, for `run.sh robot:=sim`; `dds_interface` is the laptop's interface towards the Kria | nothing |
| `python3 /root/spleat_sniff.py [--seconds 15]` (on the Kria, after `systemctl stop spleat-server`) | stands in for the SPLEAT server on port 3333 and records what the camera client sends; prints message keys and payload shapes. `--decode F` on the laptop | `/root/spleat_in.bin` |
| `python3 /root/spleat_merge_proxy.py [--save /root/replies.bin]` (on the Kria, server on port 3334) | between `despot_send_and_forget` and the server: merges its 2 polarity channels into 1, one frame in flight, prints the replies. Board commands in its docstring | replies file with `--save` |
| `python3 /root/spleat_poke.py [--pattern zeros\|pixel\|blob\|dense] [--count N] [--timeout 70]` (on the Kria) | one synthetic frame at a time to the SPLEAT server, waits for each reply; no camera needed. Durance's example: `--channels 2 --size 80 --port 3333` | nothing; prints per frame |
| `pixi run --manifest-path tools/river/pixi.toml export-ball CKPT.pt [OUT_DIR] [--script train_ball_small.py --commit SHA]` | ball-detector checkpoint into `<prefix>_model.json`, `_parameters.json/.txt`, `_readout.json` with the training code's own exporter, plus `PROVENANCE.md` (with the event window the checkpoint was trained on). Defaults: `train_ball_size.py` at `cc540f2`, into `models/ball`. Separate pixi environment; needs `third_party/riverops-0.2.0-py3-none-any.whl` (Durance, not in git) | nothing; writes OUT_DIR |
| `pixi run --manifest-path tools/river/pixi.toml -e tools river-tools deploy PROFILE.yaml root@192.168.77.1 [--check-only\|--replace]` | Durance's river-tools: copy the files of a profile (`spleat-bundle`'s `deploy.yaml`) to the Kria, verifying sha256; also `capture`, `play`. Needs `third_party/river_tools-0.1.0-py3-none-any.whl`, setup in `docs/deploy-fpga.md` | nothing; copies over SSH |
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
| `ros-follow-sim` | simulated, moves with the dog, a ball rolled by script | target node (centroid) | `ros-follow` (or `follow:=true`) | simulated, walks |
| `ros-events-sim` | GenX320 recordings, or the live KV260 camera | target node, or the ball SNN with `tracker:=snn` (or simple detector) | follow node, enabled (or reflex) | simulated, walks |

Only `ros-sim` is complete on its own. The others do nothing visible until their partner runs:

| If you start | It does nothing until you also start | Why |
|---|---|---|
| `ros-sim` | nothing else needed | camera, detector, behavior and robot are all inside |
| `ros-robot-sim` | a camera: `kria-camera` on the Kria, or `ros-camera` on a PC with a webcam | the dog waits for decisions on `/reflex/decision` |
| `ros-camera` / `kria-camera` | a robot: `ros-robot-sim` (or the real Go2 with the behavior node) | decisions are published but nobody acts on them |
| `ros-drive` | a keyboard: `ros-teleop` (walking) and/or `ros-tricks` (Hello, Sit) | the dog stands still until something publishes `/cmd_vel` or a trick |
| `ros-follow-sim` | `ros-follow`, then press `f` (or start it with `follow:=true`) | the follow node starts disabled, on purpose |
| `ros-events-sim source:=live` | the camera sender on the board (`kria_host:=...` starts it) | no events until `despot_debug_camera` connects; with a recording nothing else is needed |

`ros-web` can be added to any of them; it never replaces a partner.

Arguments (defaults in brackets):

- `ros-sim`: `detector:=snn|simple|ball` [snn] (`ball`: Durance's ball network from `ball_dir:=models/ball`
  [models/ball], or `models/ball_small`; its event window is read from the checkpoint,
  `device:=cuda` to run it on the GPU), `max_shots:=N` [0, endless], `negative_fraction:=0.3`,
  `actuation_delay_ms:=70.0`, `record:=out/x.mp4`, `mode:=keeper|dodge`,
  `reaction:=euler|move`, `locomotion:=none|rl`, `move_vy:=0.3`, `hold_s:=0.6`,
  `shot_period_s:=3.0`, `seed:=0`, `randomize:=false`, `viewer:=false` (true: also open the
  MuJoCo window; by default it runs with no window and you watch it with `ros-web`).
- `ros-camera`: `device:=0` (camera number), `detector:=simple|snn` [simple], `mirror:=false`,
  `fps:=30.0`, `contrast_threshold:=0.25`, `calibrate_s:=3.0`.
- `ros-robot-sim`: `viewer:=true` (false: no MuJoCo window), `reaction:=euler|move`,
  `locomotion:=none|rl`, `actuation_delay_ms:=70.0`, `enabled:=true` (false: only log
  what it would send), `min_confidence:=0.6`, `consecutive:=2`.
- `ros-drive`: `viewer:=true`, `actuation_delay_ms:=70.0`, `timeout_s:=0.0`,
  `camera:=false` (true: the dog's simulated event camera and the detector run while you
  drive), and with the camera `detector:=snn|simple|ball` [snn].

- `ros-follow-sim`: `follow:=false` (true: the follow node runs inside and starts by itself),
  `mode:=walk|gaze|look`, `vfov_deg:=60.0`, `viewer:=true`, `actuation_delay_ms:=300.0`, `target_start_s:=5.0` (the
  ball waits this long so the target node can calibrate), `zone_top:=0.3`. Details:
  `docs/follow.md`.
- `ros-events-sim`: `source:=PATH[,PATH]|live` [data/river], `behavior:=follow|reflex`,
  `speed:=1.0`, `loop:=true`, `gap_s:=1.0`, `still_first:=true` (the session's still scene
  plays first, for calibration), `exclude:='*take_002*'`, `flip_x:=false`, `flip_y:=false`,
  live: `port:=3333`, `kria_host:=root@durance-kria.local`; follow: `follow:=true`,
  `mode:=walk|gaze|look`, `hfov_deg:=90.0`, `vfov_deg:=90.0`, `camera_on_trunk:=false` (look
  mode; open loop), `zone_top:=0.3`, `min_events:=0.0`, `tracker:=centroid|snn|fpga|fpga-sim`; snn: `ball_model:=out/models/ball_zones_best.pt`, `ball_min_p:=0.55`,
  `ball_min_pixels:=20`, `fullres_ms:=30.0`; fpga-sim (stand-in ball_node, `/durance/ball_zone`): the snn
  arguments plus `ball_vote_window:=5`, `ball_vote_min:=3`; reflex:
  `reaction:=euler|move`, `min_confidence:=0.6`; robot: `viewer:=true`, `scene:=lab`,
  `actuation_delay_ms` [300 walking, 70 reflex]. Details: `docs/events-sim.md`.

Scenery: `ros-sim`, `ros-robot-sim` and `ros-drive` take `scene:=lab|park` [lab]. `park` is
an outdoor scene built from code (grass, a gravel path, trees, benches, a brick wall, houses);
see `reflex/sim/scenery.py`. Give `ros-web` the same value (`ros-web scene:=park`) or it draws
the lab. The browser shows the same textures, but with its own lighting and no shadows, so it
does not look identical to MuJoCo. The plain commands take it as `--scene park` (`pixi run view --scene park`, also
`drive` and `sim`). The network was trained on the lab scene only: in the park it scored
5 of 9 balls where the lab gave 9 of 9 (one run of 12 shots each), so the README numbers
are lab numbers.

![The park scene](img/park_scene.png)

### Tools (add as many as you want, each in its own terminal)

| Command | What it does | Keys go to |
|---|---|---|
| `ros-web` | dashboard at `http://localhost:8080`; `port:=8081` | the browser |
| `ros-teleop` | keyboard to `/cmd_vel`; needs `ros-drive` | its terminal: `i` forward, `,` back, `j` / `l` turn, `J` / `L` sideways, `k` stop |
| `ros-tricks` | keyboard to built-in tricks; `allow_acrobatics:=true` | its terminal: `h` Hello, `s` Sit, `u` RiseSit, `!` Damp |
| `ros-follow` | follow node: `/reflex/target` to Move / StopMove (or Euler with `mode:=gaze` / `mode:=look`); starts disabled. `max_vx:=0.3`, `max_wz:=0.5`, `max_travel_m:=3.0`, `dry_run:=true` | its terminal: `f` start, `SPACE` stop, `!` Damp, `q` quit |
| `ros-look` | follow node in look mode: nine trunk gestures (Euler yaw + pitch), never walks; starts disabled. Reads the FPGA ball zone from `/durance/ball_zone` (the Kria's `ball_node`; `input:=target` for `/reflex/target`, `input:=decision` for `/reflex/decision`: left / center / right from `fpga_perception`, use `min_confidence:=0.45`). `look_yaw:=0.3`, `look_pitch:=0.2`, `camera_on_trunk:=true`, `invert_pitch:=false`, `dry_run:=true` (`docs/follow.md`) | its terminal: `f` start, `SPACE` stop, `1`-`9` a gesture (numpad layout), `!` Damp, `q` quit |
| `ros-ball-zone-sim` | stand-in for the Kria's `ball_node` on `/durance/ball_zone`, for `ros-look` without the board. `source:=snn` (zones network on `/reflex/events_fullres`, `vote_window:=5`, `vote_min:=3`) or `source:=keys` | with `source:=keys`, its terminal: `7` `8` `9` upper row, `1` `2` `3` lower row, `0` none, `q` quit |
| `ros-event-camera` | event camera node alone, onto `/reflex/events_window`: `source:=PATH` (replay) or `source:=live`; same arguments as `ros-events-sim` | — |
| `ros-target` | target node alone: `/reflex/events_window` to `/reflex/target`; `zone_top:=0.0` for a drone | — |
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
ros2 topic echo /reflex/target                                  # where the target is (u, v, confidence)
ros2 topic echo /reflex/debug/follow                            # follow node: state, burst, travel
ros2 service call /reflex/follow/enable std_srvs/srv/SetBool "{data: false}"   # stop following
ros2 service call /reflex/target/recalibrate std_srvs/srv/Trigger
```

## Simulation on a Mac (Apple Silicon)

`default` and `ros` are `linux-64` only (PyTorch with CUDA). Under amd64 emulation (QEMU) in
Docker on a Mac, `sim_bridge` and `web_viewer` grow to the whole VM's memory and get
OOM-killed (seen 21 Nov 2026: ~7 GB each; building the lab scene natively takes ~270 MB).
`sim-arm` runs natively in an arm64 container: ROS 2, MuJoCo, the walking policy, the web
dashboard and PyTorch on the CPU (conda-forge's `cpu` build; the PyPI wheel for arm64 Linux
pulls in CUDA). So `ros-sim detector:=snn` works, and so do `test`, `gen-data`, `train` and
`eval`, slower than on the PC's GPU, and `ros-events-sim tracker:=snn` (the ball SNN,
`docs/events-sim.md`; also needs `riverops-0/` and `Training/`). The reflex SNN still does
not fit `ros-events-sim`: it expects 320x180 windows, not the camera's 320x320.
No GLFW windows: use `viewer:=false` and `ros-web`.

```bash
# on the Mac, from the repository folder
docker run -it --platform linux/arm64 --name go2arm \
  -v "$PWD":/ws -v go2arm-cache:/root/.cache -w /ws -p 8080:8080 \
  ghcr.io/prefix-dev/pixi:latest bash

# inside the container, once
pixi config set --global detached-environments true    # env in the volume, not in /ws/.pixi
apt-get update && apt-get install -y git libegl1 libegl-mesa0 libgl1-mesa-dri libglib2.0-0
bash scripts/setup.sh                                   # if third_party/ is missing (`pixi run setup` needs default)
rm -rf ros2_ws/build ros2_ws/install ros2_ws/log        # drop an amd64 build, if any
pixi run -e sim-arm ros-build

# then
pixi run -e sim-arm ros-events-sim source:=data/river/sat_follow2 viewer:=false
pixi run -e sim-arm ros-web          # second shell: docker exec -it go2arm bash
```

Later: `docker start -ai go2arm`. DDS stays inside the container; the real Go2 and the Kria
cannot be reached from Docker on macOS.

## On the Kria

Reach it with `ssh ubuntu@kria.local`. pixi is at `~/.pixi/bin/pixi` (not on the PATH).

| Command, run on the board | What it does |
|---|---|
| `~/.pixi/bin/pixi run --frozen -e kria kria-build` | build the workspace (after copying new code) |
| `~/.pixi/bin/pixi run --frozen -e kria kria-camera` | webcam as event camera + detector; same arguments as `ros-camera` |
| `~/.pixi/bin/pixi run --frozen -e kria kria-event-camera` | the real event camera onto `/reflex/events_window`; start `despot_debug_camera --host 127.0.0.1 --port 3333` first. Untested |
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

## Real event camera: capture (no pixi, no ROS)

Recording from the real GenX320 on the Kria with Durance's `river-tools`. Runs from the repo
folder in a Python environment that has `river-tools` and `h5py` (on the Windows laptop:
`& "$env:USERPROFILE\.venvs\river-tools\Scripts\python.exe"`). Setup, tips and the file format: `docs/recording-real-data.md`.

| Command | What it does | Opens |
|---|---|---|
| `ssh root@durance-kria.local despot_test_camera` | camera check, once after every boot of the board | nothing |
| `python -m capture.record --session NAME --plan PLAN` | guided, labelled takes into `data/river_samples/NAME`; plans `follow`, `cross`, `center`; resumes | window with live events |
| `python -m capture.record --session NAME --list` | prints the session's plan | nothing |
| `python -m capture.replay PATH` | plays a session, a folder of `.h5`/`.bin` files, or files | window; `--snapshot out.png` saves a frame |
| `python -m capture.relabel SESSION SRC DST --note TEXT` | moves a mislabelled take to the right take of the plan | nothing |
| `python -m capture.summarize --plan-session NAME` | Markdown tables: takes per session and label, missing takes | nothing |
| `python -m capture.bin_to_h5 PATH` | river-tools `.bin` -> `.h5`; `--remove` deletes the `.bin` after a verified read-back | nothing |

## Recipes

Each line is one terminal.

**See the whole loop work, with a score**
```bash
pixi run -e ros ros-sim
pixi run -e ros ros-web            # optional
```

**FPGA ball zones (ball_v3) turn the simulated dog's trunk** (`docs/fpga-ball-zones-v3-2026-11-22.md`).
The FPGA must hold ball_v3 (`out/spleat/ball_zones_model`, camera at 50 Hz). After
`scripts/kria/deploy.sh`, the board's behavior node does the looking (`mode:=look` from
`models/ball_v3/run_args`): no node on the laptop sends commands. `ros-web` needs, in its terminal, first:
`export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI='<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="enp8s0"/></Interfaces></General></Domain></CycloneDDS>'`
```bash
ssh -t root@192.168.77.1 /opt/reflex/run.sh robot:=sim model:=ball_v3   # Kria: camera, FPGA, zone, behavior (look)
pixi run -e ros ros-robot-sim behavior:=false dds_interface:=enp8s0 viewer:=false
pixi run -e ros ros-web             # http://localhost:8080: camera, 2x3 grid, zone, spikes per zone
```
The ball must be close to the camera and moved fast. To drive it from the laptop instead
(keyboard, gestures by hand): `run.sh ... enabled:=false` and `pixi run -e ros ros-look`, `f` to start.

**Same on the real Go2** (not run yet; `docs/real-robot.md`). With `robot:=real` the board's
DDS leaves on the Go2's adapter only: the laptop sees nothing, `ros-look` and `ros-web` cannot
be used. Only the board's behavior node can drive the dog.
```bash
ssh -t root@192.168.77.1 /opt/reflex/run.sh robot:=real model:=ball_v3                  # logs what it would send
ssh -t root@192.168.77.1 /opt/reflex/run.sh robot:=real model:=ball_v3 enabled:=true    # moves; small angles by default
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

**Watch the simulated dog follow the recorded ball** (`docs/events-sim.md`)
```bash
pixi run -e ros ros-events-sim source:=data/river/sat_follow2 exclude:='*take_002*'
pixi run -e ros ros-web                      # optional: the events next to the dog
```

**Live KV260 event camera, simulated dog**
```bash
pixi run -e ros ros-events-sim source:=live kria_host:=root@durance-kria.local
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
The same in the park, which gives the camera much more to see:
```bash
pixi run -e ros ros-drive camera:=true scene:=park
pixi run -e ros ros-teleop
pixi run -e ros ros-web scene:=park
```
Nothing acts on the decisions here: there is no behavior node. Expect false alarms while
walking (measured: about 15 % of windows, none standing still): the network was trained
with a still dog, and a moving camera makes the whole scene produce events. With the
camera on, the simulation runs at about 70 % of real time.

## When something does not work

| Symptom | Cause |
|---|---|
| the dog reacts to nothing you did, shots are scored oddly, latency shows absurd numbers | two launchers running at once (e.g. `ros-sim` while `ros-drive` is still open): they share topics. Stop one, or give one of them another network: `ROS_DOMAIN_ID=7 pixi run -e ros ros-sim` |
| `ros2: command not found`, `pixi: command not found` on the Kria | not inside `ros-shell`; on the Kria use `~/.pixi/bin/pixi` and `-e kria` |
| `cannot open camera 0` | no camera on that machine, or another number: `device:=2` |
| the dog does not react | no camera source running; simple detector calibrated while something moved (call `/reflex/recalibrate`); reflexes `enabled:=false` |
| recorder says "no windows yet" | no camera source running, or a different `ROS_DOMAIN_ID` |
| the web dashboard does not move | the simulator was started with a plain command (`pixi run view` / `drive`); use the `ros-` one |
| dashboard on port 8081 | another dashboard is still running on 8080 |
| keys do nothing | `ros-teleop`, `ros-tricks`, `ros-record`: their terminal must have focus; `view` / `drive`: the 3D window must |
| `event_camera`: `Address already in use` | `river-tools capture` (or another event camera node) holds port 3333; stop it or pass `port:=3334` |
| `event_camera`: `No module named 'h5py'` | the environment predates h5py in `pixi.toml`: `pixi install -a` (on the Kria, replay needs it too; live does not) |
| nodes on two machines do not see each other | different network or `ROS_DOMAIN_ID`; check with `ros2 node list` on both |
