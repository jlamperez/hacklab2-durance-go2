# Ball zone classifier (ball_v3) on the FPGA, 22 Nov 2026

The teammates' zone classifier (`Training/train_ball_zones.py`, branch `feat/training`,
exported to `models/ball_v3/`: 2x320x320 binary polarity frames of 5 ms in, 10 `ConvLIF2d`
in a chain, 7x1x1 out: `upper_left ... lower_right, none`) put on the KV260 with its own
firmware (`models/firmware/ball_zones_v3/`, job number not recorded).

Result: **it runs on the board with the real camera and drives the follow node's look mode
in simulation, but only for a ball close to the camera and moved fast.** Sections 1-5 are the
first attempt (nothing came out); sections 6-9 how it was made to work.

## 1. Deployment

```bash
pixi run spleat-bundle models/ball_v3/ball_zones_model.json models/ball_v3/ball_zones_parameters.json \
  --out out/spleat/ball_zones_model --checkpoint models/ball_v3/ball_zones_best.pt \
  --firmware models/firmware/ball_zones_v3 --no-noise-filter
tools/river/.pixi/envs/tools/bin/river-tools deploy out/spleat/ball_zones_model/deploy.yaml root@192.168.77.1 --replace -y
ssh root@192.168.77.1 systemctl restart spleat-xmutil-loadapp spleat-server
```

`-y`: river-tools' confirmation prompt gets no input through `pixi run`. Server config:
2x320x320 in, scaling 4096 (not checked against despot's sums: with `downsampling: 1` a pixel
holds a count, mostly 0 or 1, while training used binary frames), 7x1x1 out. Camera: 200 Hz
(5 ms windows, from the checkpoint), downsampling 1, no noise filter. The zones3 files on the
board are kept in `/root/backup/zones3_job44/`.

## 2. Synthetic frames: board = emulation

The 320x320 input did not hang the FPGA (unlike the U-Net, `docs/fpga-run-2026-11-22.md`).
`tools/spleat_poke.py` (server restarted before each pattern) against
`tools/emulate_river_poke.py` (the integer emulation `reflex/runtime/river_net.py`, one step
per frame, state kept): zeros, one pixel, a disc, 5 % and 30 % random, 10 frames each, all 7
outputs: **identical when the bytes are read as HWC**, different as CHW. So this bitstream
reads HWC too. 4-28 ms per frame.

## 3. Real camera

Whole pipeline on the board: `run.sh robot:=sim model:=ball_v3` (despot -> `camera_tap` ->
FPGA -> `fpga_perception` -> `behavior`). Readout: spikes over the last 4 frames (the
checkpoint's time steps), ties to `none`, zones mapped to left / center / right by column.

- Throughput: despot sends 130-170 frames/s; about 30/s go to the FPGA (about 15 ms each).
  The first `camera_tap` reordered via a CBOR decode + encode (7 ms per frame) and held despot
  back, so frames queued. It now reorders the payload in place and drops frames while one is
  on the FPGA (`max_inflight` 1); every forwarded frame is answered.
- Output: 20 s, 375 frames, ball thrown or not (not recorded): **all 372 spikes on `none`**,
  none on any zone. The behavior node never received a zone.

## 4. Why: trained one way, run another

On the laptop, the same emulation on recorded takes (21 Nov `sat_cross1`, 5 ms binary frames):

| Take | as trained: 5 busiest frames, 4 steps each, state reset | as the board: 120 frames, 1 step each, state kept (zone spikes / `none`) |
|---|---|---|
| 004 from-left | none x5 | 10 / 77 |
| 014 from-right | lower_middle x5 | 19 / 91 |
| 024 from-center | lower_middle x5 | 18 / 90 |

The network was trained with each frame repeated for 4 steps and the state reset; the board
runs one step per frame and keeps the state, and with `max_inflight` 1 it sees about one
frame in five. Run that way, `none` dominates even on recordings. Also, as trained, these
three takes do not come out right (left and right both `lower_middle`); three takes only.

Sections 3-4 judged the network by the zones3 take labels (from-left, ...), which say where
the ball comes from, not which sixth of the image it is in; the next sections do not.

## 5. Per frame against the tracked ball position

The 21 Nov `cross` takes (ball rolling at about 2 m), each 5 ms frame scored against the zone of
the ball's tracked position (`reflex.river.autolabel`; its row is less trusted than its column):
as trained (4 steps, reset) 18-37 % correct in the lower row, 0-7 % in the upper, `none` 58 %;
as the board runs it, 4-24 % in the lower row and `none` 100 % (it never invents a zone).

## 6. What made it work live

1. **Every frame must reach the FPGA.** 30 s of despot's frames captured on the board
   (`sniff_sparse.py`, scratch) and run through the emulation: with every frame, the network
   fires `upper_right` 16 times while a ball is held up right (second 20-21); with one frame in
   three, which was what reached the FPGA, none of it. Camera to **50 Hz** (`frequency` 50,
   `spleat-bundle --frequency 50`; merging two frames in emulation kept 13 of the 16) and
   `camera_tap max_inflight` 4: 46-50 frames/s from despot, all answered.
2. **The board computes the same on real frames.** The 30 s capture replayed into the FPGA
   (pairs merged, as HWC): `upper_right` at 20.4-20.9 s as in the emulation; 1294 of 1525
   frames identical (the server state was not reset before).
3. **Readout.** The `none` neuron fires every frame, so a zone never had more spikes than
   `none` over 4 frames: nothing was published. `margin` -10 (any zone spike wins; a negative
   margin was ignored before, fixed in `zones_readout.py`), no vote, a zone held 0.5 s
   (`models/ball_v3/run_args`, read by `run.sh`).

## 7. The loop

`run.sh robot:=sim model:=ball_v3 enabled:=false` on the board: `fpga_perception` publishes
the zone on `/durance/ball_zone` (reliable, transient-local); on the laptop `ros-look` (follow
node, `mode:=look input:=zone`) turns the simulated dog's trunk towards it, `ros-web` shows the
camera picture with the 2x3 grid, the published zone outlined and the FPGA's spikes per zone.
On the laptop every terminal needs `CYCLONEDDS_URI` on `enp8s0` (docs/commands.md), and the
follow node starts disabled (`f`, or the `/reflex/follow/enable` service).

Afterwards the board's behavior node got the same look mode (`mode:=look`, set by
`models/ball_v3/run_args`; logic in `reflex/runtime/zone_look.py` on top of the follow
controller): it reads the six zones from `/durance/ball_zone`, so the row is no longer lost
(`/reflex/decision` only says left / center / right), and the loop needs no node on the laptop.
That is what lets it drive the real Go2, which the laptop cannot reach with `robot:=real`.
Unit-tested and run on the laptop against a hand-published zone; not yet run on the board.

Live, ball moved in front of the camera for a minute: 162 zone spikes, 10-19 per second while
it moved fast and close, 41 zone changes published; zones flicker between neighbours at a
border. Not measured: which share of the published zones was right.

## 8. Why only close and fast

Layer 0 is a 1x1 ConvLIF with bias 41: every pixel spikes every few frames with no event at
all, and a single event already makes it spike. What tells the ball from that background is
many pixels at once, through layer 1's 8x8 filters. The training data (the teammates' dataset,
`projected` and `hand_labeled`, zoomed crops, balls up to about 140 px across) have big balls; a
ball a few pixels wide is background to the network. Reasoning from the parameters, not
measured layer by layer.

## 9. Open

- Retrain (fine-tune) on our camera's data: small balls at the real distance, no zoom, 20 ms
  windows, one step per frame with state kept. Same architecture: only `weights.txt` changes.
- Accuracy of the published zones, measured.
- The real Go2: `run.sh robot:=real model:=ball_v3`, first without `enabled:=true` (docs/real-robot.md).
- The firmware's job number (`models/firmware/ball_zones_v3/PROVENANCE.md`).
