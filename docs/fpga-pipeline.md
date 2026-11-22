# Camera -> FPGA -> ROS 2: how Durance's system works

How events from the GenX320 reach the SNN on the FPGA, where the result comes out, and how
Durance's example ROS 2 nodes turn it into Go2 commands. Then how our own node can plug in.
Written 21 Nov 2026.

Every statement says where it comes from:

| Mark | Source |
|---|---|
| **board** | seen on our Kria on 21 Nov (log of `despot_debug_camera` when `river-tools capture` started) |
| **docs** | Durance's pages on riversdk.dev ("Deploy an SNN", "Crosscompile the ROS2 node", the training notebook); summary in `docs/durance-riversdk.md` |
| **source** | Durance's ROS 2 workspace `go2-kria` (from `go2-kria-generated.zip`), read on 21 Nov |
| **inference** | ours, not confirmed |

`go2-kria` is Durance's code: keep it next to this repo (`../go2-kria`), never inside it.

## The whole path

```
GenX320 sensor
   | MIPI CSI-2
   v
FPGA, camera input (part of the board image)
   mipi_csi2_rx -> axis_tkeep_handler -> event_stream_smart_tra.. -> ps_host_if
   v
/dev/video0: raw events, 320x320, format 'PSE2'
   v
despot_send_and_forget (camera client)   reads /root/metavision_camera.json
   events -> 10 ms frames, 2 polarities, 80x80
   | TCP
   v
backward-server (spleat-server.service)  reads /root/.config/spleat-server/config.yaml
   <-> FPGA, SNN (spleat bitstream, /root/weights.txt)
   | TCP port 3333: one CBOR message per inference
   v
gesture_node        average of the last 50 results  -> /durance/gestures
action_bridge       highest score -> Go2 api_id      -> /api/sport/request -> Go2
obstacle_detection  lidar height map -> clear?       -> /durance/clearance (gates action_bridge)
```

## 1. Camera into the FPGA

- The sensor (`genx320 6-003c`) feeds a chain of stages: `mipi_csi2_rx_subsystem`,
  `axis_tkeep_handler`, `event_stream_smart_tra…` (name cut off in the log), then
  `ps_host_if`, which delivers the events to the processor as `/dev/video0`. **board**
- That these stages are FPGA blocks is **inference**: `mipi_csi2_rx_subsystem` is the name of
  Xilinx's MIPI receiver block, and addresses starting `a00…` are where blocks in the
  programmable logic usually sit on this chip. The board does not say so itself.
- On the path we observed, every stage is set to 320x320, format `0x5303`, at its input and
  its output, and our recordings have x and y from 0 to 319. So this chain does **not change
  the resolution**. **board**
- Not known: whether `event_stream_smart_tra…` filters or drops events (same format does not
  rule that out), and whether `despot_send_and_forget` sets these stages up differently from
  `despot_debug_camera` (we only saw the latter).

## 2. Frames into the SNN

- `despot_send_and_forget` reads `/root/metavision_camera.json`: `frequency: 100` (a frame
  every 10 ms), `downsampling: 4` (320x320 -> 80x80; allowed 1, 2, 4, 8, 16, 32, 64),
  `noise_filtering_factor: 0.5`, and the sensor biases. **docs**
- The camera does not do the downsampling: `despot_debug_camera` loaded the same file
  (biases applied, in the log) and still delivered full 320x320. **board**
- Where it happens (processor or FPGA) and how (sum or average per 4x4 block, capped or
  not) is not documented. The noise filter's method is not documented either.
- `backward-server` is the only program that talks to the SNN in the FPGA. It is started
  after `xmutil loadapp spleat` and reads the weights from `weights_file`. **docs**
- The camera client sends its frames to `backward-server` over TCP. **inference**, from
  Durance's ROS node never sending frames (section 3) and the README's "start the board's
  SPLEAT server and your camera client".
- Server config (`/root/.config/spleat-server/config.yaml`): `host: 0.0.0.0`, `port: 3333`,
  `weights_file`, `max_concurrency: 40`; `inputs: channels 2, height 80, width 80,
  scaling 4096.0`; `outputs: channels 5, height 1, width 1`. `scaling` is 2**fraction_bits of
  the potentials (4096 for Q4.12). Durance: "Use 0.0.0.0 for the host and 3333 for the port".
  **docs**

## 3. FPGA output -> TCP

From `gesture_node/src/spleat_client.cpp` and `cbor_payload.cpp` (the client) and
`qualia_sim/src/qualia_sim.cpp` (Durance's stand-in for the server). **source**

- The ROS node is a plain TCP client of `backward-server` (`host`, `port` launch arguments,
  default `127.0.0.1:3333`). It **only reads**; it never sends anything.
- Each message: 4-byte length (big-endian, at most 64 kB in the client), then a CBOR map.
- The map, as `qualia_sim` builds it: `{"id": <counter>, "payload": <bytes>, "latency": 0,
  "error": ""}`.
- `"payload"` holds the output values as raw `float32`, little-endian (`smoke.py` packs them
  with `<11f`). Number of values = the outputs in the server config.
- The client does not decode the whole map: it searches for the bytes of the key `"payload"`
  and reads the byte string after it. It drops any message whose payload is not exactly the
  expected size.
- If the connection drops, it reconnects every second.

**What a CBOR map is.** CBOR ("Concise Binary Object Representation", RFC 8949) is like
JSON, but stored as compact binary instead of text. A CBOR map is what JSON calls an object:
key -> value pairs. One message from the server, written as JSON:

```json
{"id": 42, "payload": <bytes>, "latency": 0, "error": ""}
```

The same in CBOR bytes. Each item starts with one byte giving its type and length:

```
A4                          map with 4 pairs
  62 69 64                  text "id" (2 letters)
  18 2A                     number 42
  67 70 61 79 6C 6F 61 64   text "payload" (7 letters)
  58 2C ...44 bytes...      byte string, 44 bytes long (0x2C = 44)
  ...                       "latency", "error"
```

Unlike JSON, CBOR can carry raw bytes, so the FPGA's `float32` outputs go into `payload`
unchanged, and messages stay small and quick to read. That is also why `gesture_node` can
skip decoding the map: it searches for the bytes of `"payload"`
(`67 70 61 79 6C 6F 61 64`) and reads the byte string after them. In Python,
`cbor2.loads(body)` turns a message into a `dict` (whether `cbor2` is installed on the
board's Python is not checked).

- Durance's simulated server sends a message every 2 s; the real rate is not documented. That
  `gesture_node` averages 50 messages suggests one per 10 ms frame. **inference**

What the payload values are on the real server (spike counts, spikes per time step or
membrane potentials) is not documented. The notebook says the summing of spikes over time is
outside the exported network, done by "the software using the exported model". That fits
the node averaging the last 50 messages. **docs + inference**

## 4. TCP -> ROS 2 -> Go2

Durance's launch file `bringup/launch/pipeline.launch.py` starts four nodes. **source**

| Node | Reads | Publishes | Does |
|---|---|---|---|
| `gesture_node` (C++) | TCP `host:port` | `/durance/gestures` (`msgs/Gestures`) | keeps the last `window` results (default 50), publishes their mean once the window is full; restarts the window after a reconnect |
| `action_bridge` (C++) | `/durance/gestures`, `/durance/clearance`, `/api/sport/response` | `/api/sport/request` | highest score -> `api_id` (table in `action_map.cpp`); only when clearance is `CLEAR`; ignores gestures while an action runs; stands up after sit / lay down; frees early if the robot rejects |
| `obstacle_detection` (C++) | `/utlidar/robot_pose`, `/utlidar/height_map_array` | `/durance/clearance` (`CLEAR` / `OBSTACLE` / `UNKNOWN`) | sweeps the lidar height map within `radius` around the robot |
| `profiling_node` (C++) | `/durance/gestures` | `/profiling/fps_gestures` | rate of gesture messages over 2 s |

- With `despot:=true` the launch file also starts the camera client itself
  (`despot_binary`, default `/usr/bin/despot_help`) and shuts everything down when it exits.
  The docs launch with `despot:=false` and start the camera client separately.
- Our board has no `despot_help` (22 Nov, tab completion in `/root`: `despot_debug_camera`,
  `despot_debug_camera_on_board`, `despot_send_and_forget`, `despot_test_camera`). With the
  launch defaults, the despot process fails to start, and since its exit triggers a shutdown,
  the whole pipeline would likely stop. Use `despot:=false`, or
  `despot_binary:=/usr/bin/despot_send_and_forget` (path not checked). **board + source**
- `despot_debug_camera_on_board`: seen on the board, not documented, not run yet.

## The despot binaries, read from the image

Static read on 22 Nov 2026 of `/usr/bin/despot_*` from `yocto-hackathon-img.gz` (root
partition, extracted with `debugfs`; not run, no disassembly: no aarch64 disassembler on the
laptop). Strings only, so **image + inference**. The binaries are Durance's: not in git.

| Binary | Source file (from strings) | Stages it contains |
|---|---|---|
| `despot_test_camera` | not named | camera check |
| `despot_debug_camera` | `app/debug_camera.cpp` | `event_serialization_stage`: raw events out (what `river-tools capture` reads) |
| `despot_send_and_forget` | `app/send_and_forget.cpp` | `frame_generation_debug_stage`, `spleat_server_tx_bridge`, SDL/ImGui display |
| `despot_debug_camera_on_board` | `app/debug_camera_on_board.cpp` | `frame_generation_stage`, `spleat_server_tx_bridge`, `spleat_server_rx_bridge`, `emulated_stages` (`EmulatedFrameGenerationStage`, `EmulatedSpleatStage`) |

What the strings say about `despot_send_and_forget`:

- **One argument**: `to_display` (argument parser strings "Unexpected argument:", "Missing
  value for:"). No help option. The config path is fixed: `/root/metavision_camera.json`.
- It reads `downsampling` ("Incorrect value of downsampling. Expected 1,2,4,8... Defaulting
  to 1."), `frequency`, `noise_filtering_factor` and `biases`, with the ranges it accepts:
  `bias_diff` 41-51, `bias_diff_on` 24-60, `bias_diff_off` 19-50, `bias_fo` 19-39,
  `bias_hpf` 0-127, `bias_refr` 0-127.
- The frame type is fixed at compile time: `EventCapturing<EventFrameVec, Width = 320,
  Height = 320, ChannelsNumber = 2, DownsampleFactor = 2>` (same in all three camera
  binaries). **Two channels (polarities) are built in**: no config key for channels. How
  this template `DownsampleFactor = 2` relates to the runtime `downsampling` key is unknown.
- The TX bridge connects to `127.0.0.1` (fixed string; the port is not a string, not found
  without disassembly), encodes with CBOR and puts the frame under the key `payload`; strings
  "PreProcess", "Reading input", "CBOR encoding", "Writing output". `send_and_forget` has no
  RX bridge: it does not read the results back (hence the name); the ROS node does.
- **Consequence for the small ball detector**: our server config says `channels: 1`, 320x320.
  If the client always sends 2 channels, the input size will not match the server's. What the
  server does then (reject, truncate) is unknown. The first run's logs will tell.
- `despot_debug_camera_on_board` has both bridges plus emulated stages: likely a test that
  runs the whole loop on the board, possibly without the camera or the FPGA. Not run.
- Also in the workspace: `debug_node` (`gesture_monitor`, prints the scores), `robot_sim` and
  `qualia_sim` (fake Go2 and fake server; `go2-kria/scripts/smoke.py` tests the pipeline
  with its own fake server and `robot_sim`), `unitree_api`,
  `unitree_go`.
- `go2-kria/kria-runtime.tar.gz` is a prebuilt runtime: it can be installed on the board
  without cross-compiling (`go2-kria/README.md`).
- With the Go2 connected, `action_bridge` sends real motion commands as soon as the space is
  clear.

## Durance's example does not fit together

- `msgs/Gestures.msg` is fixed at **11** scores in DVS-Gesture order (the public gesture
  dataset), so `gesture_node` expects a 44-byte payload. Durance's notebook model has **5**
  classes and their server config says `channels: 5` (20 bytes). With that model the node
  would log `no valid 44-byte payload in frame, dropping` and publish nothing. **source + docs**
- The notebook trains on **32x32** input. Its layers give a 1x1 output for 32x32 but 7x7 for
  80x80, and 320 -> 32 would need a downsampling of 10, which is not allowed. So the notebook
  model cannot be fed by the documented camera config. **docs, our arithmetic**
- The docs use port 41017 in one launch example and recommend 3333 everywhere else; the
  node's default is 3333.

So which model and config the team ran when "Durance's example worked" on our board is not
known. Ask whoever deployed it; `/root/.config/spleat-server/config.yaml` on the board shows
what the server actually uses.

## Our own node

Any program can be a client of `backward-server`. Our node replaces `gesture_node` and
`action_bridge`; the camera client, the server and the FPGA stay Durance's.

```
backward-server --TCP 3333--> spleat_bridge (ours) --/reflex/decision--> behavior (ours) --> /api/sport/request
```

**What it does**

1. Connect to `host:port` (default `127.0.0.1:3333`); reconnect every second if it drops.
2. Read 4 bytes (big-endian length), then that many bytes.
3. Decode the CBOR map (a CBOR library, or Durance's search for the `"payload"` key), take
   `payload` as little-endian `float32`. Check the count against our model's outputs; log
   and drop otherwise.
4. Keep a rolling window of the last N messages and turn it into a decision (below).
5. Publish `reflex_msgs/ReflexDecision` on `/reflex/decision`: `label`, `confidence`,
   `detector: "fpga"`, `inference_ms` (from the message's `latency`, if the server fills it;
   its unit is not documented, and `qualia_sim` always sends 0).
   No message carries a camera timestamp, so `header.stamp` is the receive time.

Our behavior node (`ros2_ws/src/reflex_ros/reflex_ros/behavior_node.py`) then works
unchanged: it already subscribes to `/reflex/decision` and holds all the safety logic.

**Decoding the output** depends on the model's last layer, set in the server config's
`outputs`:

| Model output | Server `outputs` | Payload | Decision |
|---|---|---|---|
| classes (Durance's way; ours: `none / left / center / right`) | `channels: 4, height: 1, width: 1` | 4 floats | mean over the window per class, highest wins; confidence = its share |
| `(x, y, P(ball))` | `channels: 3, height: 1, width: 1` | 12 bytes, 3 floats | only if the values are spike rates: `x`, `y` trained as rates (0 = left edge, 1 = right edge, never negative); `P(ball)` = rate of the third neuron |
| heat map (suggested for position) | `channels: 1, height: H, width: W` | H x W floats | position = cell with the most spikes (or spike-weighted centre); P(ball) = total or maximum |

A spiking last layer (LIF, hard reset) gives spikes, not coordinates. If the payload is
spike counts, `x` and `y` come out non-negative and coarse; a heat map suits a spiking
network better than regression. Whether a spatial output (`height`, `width` > 1) works on the
FPGA is not tested; the config format allows it.

**Where it runs**

- On the Kria, next to the server (`127.0.0.1`). The board's ROS 2 Humble runs Python nodes:
  Durance's `gesture_monitor` and `smoke.py` are Python, and `go2-kria/README.md` says the
  board runs Python 3.12. **source**
- `reflex_msgs` must be built for the board: add it to Durance's workspace (`go2-kria/src`)
  and build with `./manage.sh build` (see `docs/durance-riversdk.md`). `unitree_api` is
  already in their workspace.
- Our behavior node imports `reflex.runtime`, which imports numpy. Whether the board's Python
  has numpy is not checked.
- The server listens on `0.0.0.0`, so a node on the laptop could also connect to
  `durance-kria.local:3333` for development, without building anything for the board.
  Whether the server accepts several clients at once is not documented (`max_concurrency: 40`
  suggests it does).

**Rules**

- Do not run Durance's `action_bridge` at the same time: it publishes on the same
  `/api/sport/request`. Only our behavior node commands the robot.
- Our SNN has to be trained on frames built exactly like the board builds them (10 ms, 2
  polarities, 80x80 or whatever `downsampling` is set to, same summing or averaging). How the
  board shrinks frames is still open (section 2).

## Still open

| Question | How to settle it |
|---|---|
| What the payload values are: spike counts, spikes per step, potentials | Durance's mentors, or watch the messages of a running model |
| How often the server sends a message | same |
| How the board shrinks frames (sum or average, processor or FPGA), what the noise filter does | Durance's mentors; or the setup log `despot_send_and_forget` prints when started |
| What `event_stream_smart_tra…` does | Durance's mentors |
| Which model, config and port ran when the example "worked" | whoever deployed it; `config.yaml` on the board |
| Whether the server accepts more than one reading client | Durance's mentors |
| numpy on the board's Python | `python3 -c "import numpy"` on the Kria |
