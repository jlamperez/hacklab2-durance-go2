# Ball zone classifier (ball_v3) on the FPGA, 22 Nov 2026

The teammates' zone classifier (`Training/train_ball_zones.py`, branch `feat/training`,
exported to `models/ball_v3/`: 2x320x320 binary polarity frames of 5 ms in, 10 `ConvLIF2d`
in a chain, 7x1x1 out: `upper_left ... lower_right, none`) put on the KV260 with its own
firmware (`models/firmware/ball_zones_v3/`, job number not recorded).

Result: **it runs and computes what the export says, but as the board runs it, it hardly
ever picks a zone.**

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

## 5. Open

- Retrain one step per frame with the state carried over (as `models/zones3`), or find out
  whether spleat-server can run several steps per frame.
- A measured ball test (side of each throw noted), and the image orientation.
- The firmware's job number (`models/firmware/ball_zones_v3/PROVENANCE.md`).
