# Provenance of the target-zone SNN (3 zones)

Trained on 22 Nov 2026, 09:11-09:27 CET, on CPU, with `reflex.river.train` (Philipp's port of
Durance's Motion Buddy notebook). Every file comes from that one run, with two exceptions found on the
board: `weights.txt` was re-encoded from `parameters.json` (see "weights.txt encoding"; the
run's original is `weights_signed.txt`), `config.yaml`'s input scaling was set to 256 and
`metavision_camera.json`'s noise filter removed (see "Input scale").

**Runs on the KV260's FPGA** (firmware job #44, `models/firmware/kv260_zones3_job44/`). On
synthetic frames its output spikes match the fixed-point emulation neuron by neuron. Not yet
fed by the real camera. Full log: `docs/fpga-zones3-2026-11-22.md`.

## What it does

Per 10 ms frame of the GenX320 (2 polarity channels, 320x320 binned to 80x80), is there a
ball and in which third of the image. Output 4x1x1 = `none / left / center / right`.
Spikes of the last layer are counted over the last 20 frames (200 ms); the class with the most
spikes wins. The neuron state carries from frame to frame, on the board too.

Architecture (`model.json`): 2x80x80 -> 5 `SpikingConvBN2d` in a chain -> 4x1x1.

| Layer | in -> out channels | kernel | stride | output |
|---|---|---|---|---|
| 0 | 2 -> 16 | 5 | 2 | 40x40 |
| 1 | 16 -> 32 | 3 | 2 | 20x20 |
| 2 | 32 -> 32 | 3 | 2 | 10x10 |
| 3 | 32 -> 64 | 3 | 2 | 5x5 |
| 4 | 64 -> 4 | 5 | 1 | 1x1 |

LIF with a fixed leak of 0.93 per step, threshold 1.0, hard reset, no bias in training
(biases come from folding BatchNorm). 39,752 parameters. No skips and no upscaling: the same
shape class as Durance's example, which runs on our board.

## How

```bash
# 1. Link the recordings into reflex.river's layout (data/river/<split>/<kind>/*.h5)
python3 tools/river_layout.py data/recordings_2026-11-21 data/recordings_2026-11-21_cross --out data/river

# 2. Train (pixi environment with riverops; CPU, about 1 min per epoch)
pixi run --manifest-path tools/river/pixi.toml python -u -m reflex.river.train \
  --data data/river --val-sessions val --test-sessions \
  --out out/river/zones3_v1 --epochs 15 --samples-per-epoch 4000 --workers 8 \
  > out/river/zones3_v1.log 2>&1

# 3. Copy the run here (this folder)
cp out/river/zones3_v1/{best.pt,model.json,parameters.json,weights.txt,config.yaml,metavision_camera.json,meta.json,metrics.json} models/zones3/
cp out/river/zones3_v1.log models/zones3/train.log
```

- Code: repo commit `4ebd86a827b6f9c1337d9bc7c68c9dff7d23d384` (branch `merge-all`), plus
  `tools/river_layout.py`, not yet committed when this was trained.
- Environment: `tools/river/pixi.toml`: python 3.12.16, riverops 0.2.0, torch 2.14.1+cpu,
  numpy 2.2.6.
- Settings (`meta.json`): 3 zones, labels from the blob tracker (`reflex/river/autolabel.py`),
  `norm mean` (counts / 16 per 4x4 block), 10 ms frames, 40-frame clips (400 ms), stride 5,
  label = majority of the last 5 frames, readout over the last 20 frames, channels
  16-32-32-64, Adam lr 3e-3, weight decay 1e-4, batch 32, 4000 samples per epoch (class
  balanced), 15 epochs, seed 42. Left/right mirroring as augmentation. The best epoch by
  validation balanced accuracy is kept.
- Full output: `train.log`.

## Data

The 21 Nov recordings: real GenX320 on the Kria, camera on a stand at dog height, one room
about 2 m deep, one evening (`capture/`, `docs/recording-real-data.md`). Not on the robot.

`tools/river_layout.py` sorts the takes:

- `ball`: `from-left`, `from-right`, `from-center` (sat_cross1), `to-left`, `to-right`,
  `from-center` (sat_cross2). The auto-labeller finds the ball in each 10 ms frame and gives
  its zone, or `none` where it finds no ball (before the throw, after it left).
- `none`: `no-ball` (sat_cross1: a person walking, arms, lamp/shadow, nothing moving) and
  `none` (sat_follow1, sat_follow2).
- Still scene for the noise calibration (`none/empty_*.h5`): sat_follow2 takes 15 and 16.
- Left out: the `follow` ball takes (ball lying still: a different task), `close`, `drone`,
  the `wave_takes` session, `samples` (no labels).
- Split: every 5th take by take number (4, 9, 14, ...) to `val`, the rest to `train`.

| Split | ball takes | none takes | windows: none / left / center / right |
|---|---|---|---|
| train | 49 | 17 + 2 still scene | 3416 / 602 / 1127 / 684 (5829) |
| val | 11 | 2 + 2 still scene | 690 / 159 / 274 / 115 (1238) |

Val takes: sat_cross1 4, 9 (from-left), 14, 19 (from-right), 24, 29 (from-center),
34, 39 (no-ball); sat_cross2 4, 9 (to-left), 14 (to-right), 24, 29 (from-center).

Validation is optimistic:

- Same room, same evening, same camera position as training.
- The 2 still-scene takes are in both splits (they are linked for the calibration and are
  also read as `none` windows).
- Only 2 real `no-ball` takes in val, so the false-trigger rate is noisy. During training it
  moved between 8 % and 41 % from one epoch to the next.

## Result

Validation per epoch (`train.log`):

| Epoch | bal. acc | false trigger | miss | wrong zone | |
|---|---|---|---|---|---|
| 1 | 75.9 % | 65.8 % | 1.8 % | 9.5 % | saved |
| 3 | 80.4 % | 46.1 % | 3.8 % | 8.4 % | saved |
| 7 | 70.7 % | 8.4 % | 35.2 % | 6.8 % | |
| 8 | 80.7 % | 21.2 % | 15.7 % | 8.2 % | saved |
| **9** | **82.6 %** | **29.3 %** | **7.7 %** | **8.0 %** | **saved, kept** |
| 13 | 79.4 % | 19.9 % | 16.4 % | 8.0 % | |
| 15 | 82.0 % | 35.1 % | 5.5 % | 9.3 % | |

Best epoch 9, validation (1238 windows):

| | float | fixed point, BN folded (as on the board) |
|---|---|---|
| balanced accuracy | 82.6 % | 81.2 % |
| recall none / left / center / right | 71 / 87 / 78 / 94 % | 62 / 89 / 83 / 90 % |
| false trigger (none -> a zone) | 29 % | 38 % |
| miss (zone -> none) | 8 % | 4 % |
| wrong zone | 8 % | 9 % |

Confusion, fixed point (rows: truth, columns: predicted none / left / center / right):

```
none    428  55 191  16
left      2 142  15   0
center   21  14 228  11
right     0   0  11 104
```

False triggers are the weak point: `none` taken as `center` most often.

Per-layer rescale to fit Q1.7 after folding BatchNorm (`fold_for_board`): 9.63, 1.24, 1, 1, 1.
It divides weights, bias and threshold of a layer by the same factor, so its spikes are
unchanged. Hence thresholds 425 and 3293 instead of 4096 in layers 0 and 1.

## weights.txt encoding

riverops 0.2.0's `export_params` gives weights and biases as signed integers (-128..127).
spleat-server on the KV260 reads them as **two's complement, unsigned 0-255** (-16 -> 240),
the encoding of Durance's example `weights.txt`. Tested on the board with both files and the
same synthetic frames: the two's complement file matches the fixed-point emulation exactly
for the first 6 frames; the signed one does not match from frame 0
(`docs/fpga-zones3-2026-11-22.md`).

- `weights.txt`: two's complement. It is `board_weights_text(parameters.json)`, which
  `reflex/river/train.py` now writes directly.
- `weights_signed.txt`: what the training run wrote, signed. Kept as a record; do not deploy.

Both are 39,613 values in this order: weights of layers 0-4 (800, 4608, 9216, 18432, 6400),
biases of layers 0-4 (16, 32, 32, 64, 4), then threshold and leak per layer (425 287,
3293 287, 4096 287, 4096 287, 4096 287). One per line, no trailing newline.

## Input scale: `config.yaml` scaling 256, not 4096

Trained with `norm mean` (event count / 16 per 4x4 block, as in Durance's notebook).
`despot_send_and_forget` with `downsampling: 4` sends the **sum** (integers 0-17 in a 15 s
capture with `tools/spleat_sniff.py`, 22 Nov). spleat-server multiplies its input by `scaling`
to make it fixed point, so `scaling: 256` (4096 / 16) gives the network the values it was
trained on, with no retraining. With 4096 it saw 16 times too much (and values above 8
saturate Q4.12). Checked with `spleat_poke.py` against the emulation: at 256 the board follows
the 1/16-scaled emulation (same silent first frames, `none`-dominated), not the unscaled one;
spike for spike it matches for 2-3 frames, then drifts (not explained).
`docs/fpga-zones3-2026-11-22.md` section 10.

**Noise filter off.** The recordings it was trained on are raw, unfiltered events. despot's
`noise_filtering_factor: 0.5` left a median of 3 events per 10 ms frame with a ball moving, and
despot skipped frames with no events (15-30 frames per second instead of 100). With the scaling
right, the network then hardly fired. `metavision_camera.json` here has the key removed
(Durance's docs: remove it for no filtering); with it removed despot sends 100 frames per second.

`config.yaml` was edited by hand for this (the training run wrote 4096). `reflex/river/train.py`
now writes 4096 / downsample² for `norm mean`. `spleat-bundle` still insists on 4096; do not use
it to regenerate this config.

## Files

| File | What it is | Where it goes |
|---|---|---|
| `best.pt` | checkpoint (float weights, BatchNorm, metrics, config) | stays here |
| `model.json` | River Model Graph | riversdk.dev firmware job (#44) -> bitstream |
| `weights.txt` | fixed-point parameters, two's complement | `/root/weights.txt` on the board |
| `weights_signed.txt` | the same, signed, as riverops exported it | do not deploy |
| `parameters.json` | the same values per layer, signed | |
| `config.yaml` | spleat-server config (2x80x80 in, 4x1x1 out, **scaling 256**, port 3333) | `/root/.config/spleat-server/config.yaml` |
| `metavision_camera.json` | camera config (`frequency` 100, `downsampling` 4, **no noise filter**) | `/root/metavision_camera.json` |
| `meta.json`, `metrics.json` | run configuration, validation metrics | stays here |
| `train.log` | training output | stays here |

## sha256

| File | sha256 |
|---|---|
| `best.pt` | `b89d73e631dfed9e3fcf12d944a405fa42d96964fd89b71b85d7fb4b3a8ec577` |
| `config.yaml` | `70a5826c292f09a2cb6a2acb96f7e7e444f2006f95e059420ebc289cd16f397f` (scaling 256; as trained, 4096: `012ba3bf…`) |
| `meta.json` | `2f8d3fa73751c9077cde9810cca761c60907c89208868505dafb80adb397fd1e` |
| `metavision_camera.json` | `1c0aa3bf86aa100f730b4c499a1730fc7a95d41aa9f7fa1ec28f1634edbc4789` (no noise filter; as trained, with `noise_filtering_factor` 0.5: `5a731420…`) |
| `metrics.json` | `b7d187d2b5ca0036d2ad84f78a30306a21d0bcc0811a2f3575fd29d31de966f0` |
| `model.json` | `ad7b79badf1df9345e3fcdd505e58fc7fa8ebbf07fcf2d5e1426f9e0fd24a2ca` |
| `parameters.json` | `32b33f82bb2c42777f352e23bfde1303f41bfb0a6eeaf18e099c930153c87f7e` |
| `train.log` | `0a2eb497ca6d6b89c0b5ecd1643a5b05eb7fef48473becd7f09d0d69a87c7865` |
| `weights.txt` | `65bacbcccfade5b3fbda59280603dae17ed3816637e1e5facb0856fce5b70fc0` |
| `weights_signed.txt` | `6196806c6c77e3393def9587600d6521c678e20f256dfa75dbd81bef4d0f5150` |
