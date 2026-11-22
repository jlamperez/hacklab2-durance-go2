# Despot: Durance's camera programs on the Kria

`despot_*` are the programs on Durance's KV260 image that read the GenX320 event camera.
One of them, `despot_send_and_forget`, is the camera client that feeds frames to the FPGA
inference server (SPLEAT, `backward-server`). It always sends **two channels**, one per
polarity, and there is no setting to change that. That is why our one-channel ball
detector needs an adapter (`tools/spleat_merge_proxy.py`).

Written 22 Nov 2026. Each statement says where it comes from:

| Mark | Source |
|---|---|
| **board** | run or seen on our Kria on 21-22 Nov |
| **image** | strings of the binaries, read from `yocto-hackathon-img.gz` (not disassembled: no aarch64 disassembler on the laptop) |
| **docs** | Durance's pages on riversdk.dev (`docs/durance-riversdk.md`) |
| **inference** | ours, not confirmed |

The binaries are Durance's: never commit them.

## The four programs

All in `/usr/bin/` (**board**). There is no help option and no `despot_help`, although
Durance's ROS 2 launch file uses `/usr/bin/despot_help` as its default
(`docs/fpga-pipeline.md`, section 4).

| Program | Source file (**image**) | What it does | Used by us |
|---|---|---|---|
| `despot_test_camera` | not named | sets the camera chain up, starts and stops the camera; ends with "Camera test successfully passed" | yes: once after every boot **and after every bitstream reload** |
| `despot_debug_camera` | `app/debug_camera.cpp` | raw events `(t, x, y, p)` over TCP to the laptop (`event_serialization_stage`) | yes: `river-tools capture`, our ~100 recordings |
| `despot_send_and_forget` | `app/send_and_forget.cpp` | events -> frames -> SPLEAT server; optional window with the frames (SDL + ImGui) | yes: 22 Nov, through our adapter |
| `despot_debug_camera_on_board` | `app/debug_camera_on_board.cpp` | same as `send_and_forget`, plus it reads the server's results back, plus emulated stages (`EmulatedFrameGenerationStage`, `EmulatedSpleatStage`) | no. Probably a test of the whole loop on the board, maybe without camera or FPGA (**inference**) |

Libraries `despot_send_and_forget` links (**image**): Prophesee's OpenEB
(`libmetavision_hal`, `libmetavision_sdk_stream`), `libudev`, SDL2, Mali (GPU, for the
window). It also contains Tracy, a profiler.

## How `despot_send_and_forget` works

```
GenX320 --MIPI--> camera blocks in the FPGA overlay --> /dev/video0 (raw events, 320x320, 'PSE2')
   |
   v  despot_send_and_forget
   1. finds the Prophesee media device and sets every block of the chain to 320x320
   2. opens the camera with the Metavision SDK
   3. applies biases (and noise filter) from /root/metavision_camera.json
   4. EventCapturing: events -> frame of 2 x 320 x 320, one frame per 1/frequency s
   5. FramesQueue -> SpleatServerTXBridge: "PreProcess", CBOR {"id", "payload"} -> TCP 127.0.0.1:3333
   6. with --to_display 1: draws the frames in a window
```

1. **Camera chain** (**board**, startup log). It looks for `/dev/media*` with driver
   `psee-video` and sets each entity's format:

   ```
   Found media device /dev/media0: driver=psee-video, model=Prophesee Video Pipeline, bus=platform:a0030000.ps_host_if
       Entities:
           ps_host_if output 0 (/dev/video0)
           a0050000.event_stream_smart_tra (/dev/v4l-subdev0)
           a0040000.axis_tkeep_handler (/dev/v4l-subdev1)
           a0010000.mipi_csi2_rx_subsystem (/dev/v4l-subdev2)
           genx320 6-003c (/dev/v4l-subdev3)
   Setting subdev format for genx320 6-003c (/dev/v4l-subdev3) pad 0 stream 0 to 320x320, code 0x5303
   ...
   Verified capture format for /dev/video0: 320x320, fourcc 'PSE2', sizeimage=1048576
   ```

   It also knows the IMX636 (`imx636 6-003c`, **image**). If the overlay is not loaded:
   "Could not find Prophesee media device. Is the FPGA overlay loaded?" (**image**).
2. **Camera** (**board**): "Camera started", then "Topology configured" about 0.6 s later.
3. **Biases** (**board**): read from the JSON, range-checked and applied. The ranges it
   accepts (**image**): `bias_diff` 41-51, `bias_diff_on` 24-60, `bias_diff_off` 19-50,
   `bias_fo` 19-39, `bias_hpf` 0-127, `bias_refr` 0-127.
4. **Frames.** One class builds them, with its sizes fixed at compile time (**image**, the
   same in the three camera programs):

   ```
   EventCapturing<EventFrameType, Width, Height, ChannelsNumber, DownsampleFactor>
     [with EventFrameType = EventFrameVec; Width = 320; Height = 320;
           ChannelsNumber = 2; DownsampleFactor = 2]
   ```

   `downsampling` from the JSON is read at run time ("Incorrect value of downsampling.
   Expected 1,2,4,8... Defaulting to 1.") and was respected: with `downsampling: 1` the
   frames are 320x320 (**board**). How the compile-time `DownsampleFactor = 2` relates to it
   is unknown.
5. **Sending.** A separate thread takes frames from a queue and sends them to `127.0.0.1`
   (fixed string, **image**), port 3333 (**board**: it connects to whatever listens there).
   It never reads anything back: hence "send and forget". The results go to the server's
   other clients (Durance's `gesture_node`, our `spleat-read`), or back on the sending
   connection, as our adapter saw.

Configuration: only `/root/metavision_camera.json`, a fixed path (**image**). Keys:
`frequency`, `downsampling`, `noise_filtering_factor`, `biases`. Arguments: only
`--to_display 0|1` (**image**; **docs** for the values).

## What it sends (captured)

`tools/spleat_sniff.py` took the server's place on port 3333 (**board**, 22 Nov,
`frequency: 500`, `downsampling: 1`):

```
stream: 20054548 bytes; first 32: 00 0c 80 12 a2 62 69 64 00 67 70 61 79 6c 6f 61 64 5a 00 0c 80 00 ...
message 0: 819218 bytes
  'id': 0
  'payload':
    bytes: 819200
      as float32: 204800 values, shapes ['2x320x320'], min 0.0, max 3.0, nonzero 87522, 4 distinct
message 1: 819218 bytes
  'id': 1
  'payload':
    bytes: 819200
      as float32: 204800 values, shapes ['2x320x320'], min 0.0, max 1.0, nonzero 58, 2 distinct
```

Read byte by byte:

| Bytes | Meaning |
|---|---|
| `00 0c 80 12` | message length, big-endian: 819218 |
| `a2` | CBOR map with 2 pairs |
| `62 69 64` `00` | `"id"`: 0 |
| `67 70 61 79 6c 6f 61 64` | `"payload"` |
| `5a 00 0c 80 00` | byte string of 819200 bytes |

- **819200 bytes = 2 x 320 x 320 x 4**: two channels of float32 (little-endian; the bytes
  `3f 80`, `40 00`, `40 40` are 1.0, 2.0, 3.0).
- Values are **event counts** per pixel and polarity: 0 to 3 in the first frame (which
  holds what piled up during startup), then 0/1 with a few dozen active pixels per 2 ms.
- Which channel is ON and which is OFF is not known.
- Only `id` and `payload`: no shape, no timestamp.
- Rate seen through the adapter: about 75-80 frames per second, with `frequency: 500` asking
  for 500 (**board**). Whether each frame still covers 2 ms is not known.

## The fixed two channels, and what we do about it

Three things show the channel count cannot be changed:

1. The type above has `ChannelsNumber = 2` as a template argument (**image**).
2. No configuration key for channels: the only keys read are `frequency`, `downsampling`,
   `noise_filtering_factor` and `biases` (**image**).
3. With our server config at `channels: 1`, every frame was refused (**board**):

   ```
   level=ERROR msg="error validating message" service=spleat-facade client=jolly-otter err="wrong payload size" msg_id=112964
   ```

The server checks the payload size against `config.yaml` and drops the frame before the
FPGA. Setting `channels: 2` would get frames past that check, but the bitstream (job #35)
was built for a one-channel first layer, so the result would be meaningless.

Two ways out:

| | Change | Cost |
|---|---|---|
| **Adapter** (done) | `tools/spleat_merge_proxy.py`: despot -> 3333 -> merge (any event = 1.0, what the model saw in training) -> 3334 -> server | the server moves to port 3334; one more process on the board; firmware and weights stay |
| Retrain | network with 2 input channels | new firmware job, new training |

With the adapter the server accepts our frames and the FPGA ran one inference; then the
hardware hung. That is a separate problem: `docs/fpga-run-2026-11-22.md`.

## Running it

```bash
despot_test_camera                         # after every boot and every bitstream reload
despot_send_and_forget --to_display 0      # 1 opens a window (needs a display)
```

- **After a bitstream reload** (`systemctl restart spleat-xmutil-loadapp`) the camera
  blocks are reloaded too, since they are in the same overlay. Without a new
  `despot_test_camera`, despot starts ("Camera started", "Topology configured") but sends
  nothing (**board**, 22 Nov).
- It connects to port 3333 before frames start. A listener that records only for a few
  seconds can see a connection and no data: our first sniff got 0 bytes in 3 s and 20 MB
  in 15 s (**board**).
- On exit it prints "SPLEAT server TX Bridge exited" and "Camera stopped".
- With Durance's ROS 2 launch file: `despot:=false` and start despot by hand, or
  `despot_binary:=/usr/bin/despot_send_and_forget`. The default binary does not exist.

## Still unknown

- What "PreProcess" does to a frame before sending.
- How the noise filter works, and how downsampling combines pixels (sum or mean).
- Which channel is which polarity.
- Why it runs at about 80 frames per second when asked for 500.
- What `despot_debug_camera_on_board` does when run.
