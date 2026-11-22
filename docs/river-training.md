# Training the target-zone SNN for RIVER

`reflex/river/` is Durance's Motion Buddy training notebook turned into scripts and
pointed at our task: per 10 ms frame, is there a target, and if so in which horizontal
zone (`none / left / center / right` by default). Same riverops layers, loss, optimiser
and export as the notebook. What it produces goes to the Kria as described in Durance's
"Deploy an SNN" page.

Status (22 Nov): run with the real riverops 0.2.0 (CPU) on the real GenX320 recordings of
21 Nov, with no code change except the `weights.txt` encoding. The result, `models/zones3`
(3 zones, firmware job #44), **runs on the KV260's FPGA** and matches the fixed-point
emulation there (`docs/fpga-zones3-2026-11-22.md`). Commands, data and validation:
`models/zones3/PROVENANCE.md`. Not yet fed by the real camera on the board.

Our capture sessions (`capture/`, one folder per session with `takes.csv`) are linked into
the layout below by `tools/river_layout.py`. Its "sessions" are `train` and `val` splits,
since the 21 Nov takes are all one room and one evening.

`weights.txt`: weights and biases are written in two's complement (0-255), which is what the
board reads. riverops' `export_params` gives them signed (`board_weights_text` in `train.py`).

## Setup on the GPU box (once)

riverops wants Python 3.10. A plain venv, no pixi needed:

```bash
# from the Mac: copy the repo (no environments, no third_party)
rsync -av --exclude .pixi --exclude third_party --exclude out --exclude data \
  "$HOME/Git repos/hacklab2-durance-go2/" GPUBOX:hacklab2-durance-go2/
scp riverops-0.2.0-py3-none-any.whl GPUBOX:

# on the GPU box
cd hacklab2-durance-go2
python3.10 -m venv .venv-river && source .venv-river/bin/activate   # or: uv venv -p 3.10 .venv-river
pip install torch numpy matplotlib pytest      # pick the CUDA wheel that matches the driver
pip install ~/riverops-0.2.0-py3-none-any.whl
python -c "import torch, riverops; print(torch.cuda.is_available())"
python -m pytest -q tests/test_river.py
```

Run everything below from the repo root with the venv active.

## 1. First model from synthetic data (no recordings needed)

```bash
python -m reflex.river.synth --out data/river_synth            # 5 sessions, ~4 min
python -m reflex.river.train --data data/river_synth --out out/river/synth_v1
```

The last session is the test split, the one before it validation. The synthetic data is
a cartoon (an edge-event ball bouncing on a floor; walkers, waving hands, camera pans and
flicker as distractors), good enough to bring up the deployment and to pretrain.

## 2. Real recordings

Record with `river-tools capture --host root@durance-kria.local` (Durance's "Capture
Data" page), then sort the files into one folder per session and kind:

```
data/river/sat_pm/none/empty_000.bin   FIRST: the still scene, nothing moving, ~10 s
data/river/sat_pm/ball/*.bin           one moving ball, camera still
data/river/sat_pm/none/*.bin           no target: people walking, hands, camera turning
data/river/sat_pm/drone/*.bin          later, for the drone
data/river/sun_am/...
```

A session is one sitting: same place, light and mount. Splits are by session, never by
window.

### How target takes get their labels

Every frame of a `none` take is `none`. Target takes are labelled by a tracker,
`reflex/river/autolabel.py`:

1. The `none/empty*.bin` takes of the session calibrate it: pixels that fire in most
   frames are masked (hot pixels) and the detection threshold is set at twice the 99th
   percentile of pure noise. Without an empty take the threshold is a fixed guess.
2. Per take, 4x4-pixel cells busy in more than 30 % of the frames are masked (a
   flickering lamp, a screen, a hand waving in one spot).
3. Per 10 ms frame, the strongest compact patch of events (11x11 cells, about one ball)
   is the target. Compact means the patch holds most of the events of a 3x longer strip
   through it, so the edge of a person walking past or of a door frame during a camera
   pan does not qualify.
4. A tracker stays near the last position for 200 ms before searching the whole image
   again, so a second object does not steal the track. Runs shorter than 30 ms are
   dropped as noise and gaps up to 100 ms are bridged.

Recording rules that make this work: the camera does not move, the ball comes in from
outside the picture (the person throwing it stays out of view), and nothing else moves
in a ball take.

### Check the labels before training

```bash
python -m reflex.river.review data/river/sat_pm      # -> out/labels/sat_pm.png, one row per take
```

Each row shows where the events are along the image's horizontal axis over time (dog's
left at the top), the tracked target in red and the label band underneath. A good take
has the red line on the ball's trace. About 3 seconds per take.

Fix a bad take in `data/river/sat_pm/overrides.json` instead of re-recording or editing
files, then run `review` again:

```json
{"*":                 {"ignore": [[280, 0, 320, 320]]},
 "ball/clip_003.bin": {"drop": true},
 "ball/clip_007.bin": {"trim": [1.0, null], "zone": "right"}}
```

`drop` leaves the take out, `trim` keeps [start_s, end_s], `ignore` hides rectangles
(sensor pixels, image coordinates) from the tracker but not from the network, `zone`
forces the label wherever the target is detected. `"*"` applies to every take of the
session. Unknown keys are an error.

`python -m reflex.river.data data/river/sat_pm` prints the same per-take label shares
without the image. Every tracker setting can be tried from the command line
(`review --help`); `--method median` gives the old labels (median x of all events).

### How good the labels are (synthetic data only)

Measured with `review --auto`, which labels synthetic takes with the tracker and compares
with their ground truth. Tuned on 18 takes (seed 3), then checked on 18 new ones (seed
11, below). 10 s takes, about a third with a walker, waving hand, camera pan or flicker
next to the ball:

| Labeller | Frames agreeing | False target (truth none) | Missed (truth zone) | Wrong zone |
|---|---|---|---|---|
| median x of all events (before) | 64 % | 86 % | 3 % | 25 % |
| tracker (now) | 92 % | 1.5 % | 8 % | 1.1 % |

Most remaining misses are a ball rolling slowly (few events per 10 ms). The ground truth
calls the ball visible from 30 px/s, which is a choice of the generator, not a fact
about the sensor. Not yet measured on a real GenX320 recording: the first real session
decides whether the defaults hold, starting with the threshold the empty take gives.

### First real sessions (21 Nov, `sat_follow1`, `sat_follow2`)

Recorded by the team with river-tools' camera sender, camera on a stand at dog height,
one room. The files are HDF5 (`reflex-events-v1`) with a take label and key-press times
in the attributes; `takes.csv` in each session lists them. Sorted into
`data/river/<session>/{ball,none,drone}/` (take 0 of `sat_follow2` is the empty take),
the auto-labeller held up:

- **Calibration:** `sat_follow2`'s empty take gives a threshold of 10.3 events per patch
  (noise p99 5.1). `sat_follow1`'s take 0 has bursts of up to 50 events per frame and
  pushed the threshold to 52, which hides the ball, so it is filed as an ordinary `none`
  take there (fixed threshold 8).
- **Tracking:** on every still-camera ball take the red track sits on the ball's trace,
  bounces included. All `none` takes stay 100 % none, the screen flicker (~2 M ev/s) and
  the camera turns too.
- **Take labels are not frame labels.** The "left/center/right" takes say where the ball
  *started*: in all 10 of them the first tracked zone matches the take name (so the mount
  is not mirrored), but 8 of the 10 then roll across the whole picture. Use the tracker's
  per-frame labels, not the take name. `close` takes also cross the full width; the ball
  sits at y ~ 300 of 320 there versus ~ 250 at 2 m.
- **Dropped in `overrides.json`:** `sat_follow1` take 3 (ball below the picture) and
  `sat_follow2` takes 28, 31, 33 (ball with the camera moving forward: the whole scene
  fires and the tracker cannot find the ball). The camera-moving `none` takes stay in.
- **How much ball there is:** about 16 s in total (left 4.4 s, center 6.0 s, right 5.2 s)
  against 115 s of none, so ~310 target windows out of 2,400. Enough to fine-tune the
  synthetic model and to test it, not to train from scratch. A ball at 2 m gives only
  ~25-30 events per 10 ms.

Fine-tune from the synthetic model:

```bash
python -m reflex.river.train --data data/river --val-sessions sat_pm --test-sessions sun_am \
  --init out/river/synth_v1/best.pt --out out/river/ball_v1
```

Add `--data data/river data/river_synth` to keep the synthetic data in the mix.

### Ball positions for the ball SNN in `Training/`

```bash
python -m reflex.river.export_positions data/river --out data/river_positions
```

Writes the tracker's per-frame ball position (sensor pixels) as `gt_x`, `gt_y`, `gt_t`
into a copy of each ball take, and copies the `none` takes to `no_ball/`. Format and
caveats in `data/river_positions/README.md`. From the 21 Nov sessions: 13 takes, 1,493
labelled frames.

### Drone takes: the camera moves

Drone takes are recorded with the camera moving, and then the whole scene fires: the ball
tracker above takes whichever background corner is strongest. Kinds listed in
`LabelConfig.ego_kinds` (default `drone`) are therefore tracked by their motion relative to
the scene (`track_target_ego`, step 6 in `reflex/river/autolabel.py`):

1. Per 10 ms frame, the shift of the whole scene over the last 200 ms is the peak of the
   cross-correlation of active cells with those 200 ms earlier. Frames with few active
   cells count as camera still; busy frames without a clear peak as unknown.
2. A candidate (5x5-cell patch, about 20 px) is the drone if its cells are *not* where the
   scene shift puts them 200 ms earlier, but are found elsewhere, at least 2 cells off: it
   moved on its own. Patches that just came in over the image border do not count.
3. With the camera still, the drone is found as the ball is. Once locked, the track's
   search area moves with the scene. A candidate without motion of its own continues the
   track only right where the scene put it (a hovering drone), for at most 300 ms.
4. Busy frames with an unknown shift start no track.

`"ego_motion": true / false` in `overrides.json` switches this on or off per take. Every
setting starts with `ego_` and can be tried with `review --ego-...`.

Measured on `sat_follow2` (21 Nov). No hand labels exist, so these are frames where
something is found, not accuracy:

| Take | Drone tracker | Ball tracker |
|---|---|---|
| 26, 27: no drone, camera turning | 6.8 %, 5.7 % | 67 %, 66 % |
| 25: no drone, camera creeping forward | 38 % | 40 % |
| still-camera ball takes 3-14 (forced through it) | same zone label as the ball tracker in 93-99.5 % of frames | |
| 45: drone, camera tilting | 27 %, on the drone's arcs when checked by eye | 51 %, mostly background |

On synthetic data (tests): 97-100 % of the frames it labels are on the drone, but it
finds a slow drone in only about a third of the frames.

Limits, in the order they matter:

- **It misses more than it invents.** A drone that hovers while the camera moves looks
  like background and is held for 300 ms at most; frames it misses are labelled `none`.
  For training, use the found segments rather than whole takes, or record still-camera
  drone takes, which the ball logic labels well.
- **One translation for the whole image.** A pan or tilt fits; moving forward (expansion)
  or rolling does not, and those frames end up unknown or wrong.
- **With the camera nearly still, anything compact that moves is taken**, as for the
  ball: in take 25 an object on the floor; in take 45, at 0-2.5 s and 18.7-20.6 s, a large
  object at the left edge that may be the drone up close or a hand. Check the review
  image and `trim` or `ignore` what is not the drone.

## 3. A new target with only the last layer retrained

```bash
python -m reflex.river.train --data data/river --targets drone --init out/river/ball_v1/best.pt \
  --freeze-trunk --epochs 5 --out out/river/drone_head
```

Same architecture, so the same bitstream: only `weights.txt` changes on the board. Measure
the ball model on the drone test session first (zero-shot), then this one. With
`--targets drone`, ball recordings are left out; `--targets ball drone` trains one model
that turns toward either.

## What a run writes (`--out`)

| File | Use |
|---|---|
| `best.pt`, `meta.json` | checkpoint (best validation balanced accuracy) and every setting |
| `metrics.json` | val / test: accuracy, balanced accuracy, false-trigger rate (none -> a zone), miss rate (zone -> none), wrong-zone rate, confusion matrix; quantized val if it could run |
| `model.json` | architecture for riversdk.dev ("New firmware", board KV260, IP GenX320). Only needed when the architecture changes |
| `weights.txt` | -> `/root/weights.txt` on the Kria (no trailing empty line, as required) |
| `config.yaml` | -> `/root/.config/spleat-server/config.yaml` (input 2x80x80, scaling 4096, output C x 1 x 1) |
| `metavision_camera.json` | -> `/root/metavision_camera.json`: frequency and downsampling matching training |

The network: four stride-2 `SpikingConvBN2d` (16, 32, 32, 64 channels) take 80x80 to 5x5,
then a 5x5 convolution gives one output channel per class at 1x1. About 40k parameters.
Training loss: MSE between the spike count over the last `--readout` frames (default 20 =
200 ms) and a target of 80 % of those frames for the true class, 50 % for the neighbouring
zones and 20 % for the rest. On the robot, the ROS node should count spikes over the
same 200 ms and pick the class with most, with `none` when the counts are low.

## Assumptions to check with Durance

1. `river-tools capture` writes the same `.bin` format as the gesture dataset (4-byte
   size header, then x, y, polarity, timestamp). The reader raises if not.
2. What despot puts in each input pixel at downsampling 4: the mean of the 4x4 block's
   counts (the notebook's area interpolation, our default `--norm mean`), the sum, or 0/1.
3. Is the spleat-server's output a spike count per frame per class, and is an output
   wider than 1x1 supported (for more zones as positions instead of classes)?
4. The camera's x axis: if image left is not the dog's left, train with `--flip-x`.
5. Limits on channels, kernel size (the last layer uses 5x5) and layer count.

## Public data

The table-tennis dataset of Ziegler et al. 2024 ("Spiking Neural Networks for Fast-Moving
Object Detection on Neuromorphic Hardware Devices Using an Event-Based Camera") is public:
`github.com/cogsys-tuebingen/snn-edge-benchmark`, `dataset/`, 108 HDF5 files (18
hand-labelled, 90 projected from frame cameras), about 2.6 GB through Git LFS, no license
file. Events from a Prophesee EVK4 at 1280x720, ball positions as labels, ceiling-mounted
cameras over a table with a static background. Not converted here: the viewpoint and
sensor differ a lot from a GenX320 at dog height, so own recordings come first.

## Two fixes after the first run with the real riverops 0.2.0

Found on a short CPU run (2 epochs x 256 windows on synthetic data), so the numbers are
from an undertrained model; what matters is the size of each gap.

1. **BatchNorm running statistics lagged behind the weights.** The board runs the model
   with BatchNorm folded from its running statistics (eval mode). On the same 256 training
   windows: 56 % with batch statistics, 14 % with the running ones. `recalibrate_bn`
   re-estimates them on `--bn-calib-batches` (default 20) training batches before every
   validation, so the saved checkpoint and its validation score are what the board gets.
2. **Folded weights did not fit Q1.7.** With correct statistics, folding BatchNorm into
   the first layer gives |w| up to 16.5 (the event-count inputs are small, so BatchNorm
   scales them up); `to_fixed` clipped them to [-1, 127/128] and the fixed-point model
   fell to 25 %. `fold_for_board` divides each layer's weights, bias and threshold by the
   same factor so they fit. That is exact (input is frames or spikes, linear leak, reset
   to zero) and the threshold is a per-layer value in `weights.txt`. Fixed point went from
   26 % to 46 % against 49.5 % in float. Layers that need no rescale export identical to
   Durance's notebook.

The "fixed-point val" line of a run now emulates the folded, rescaled, fixed-point model,
not the unfused QAT model. What still costs accuracy (46 vs 49.5 %) is weight resolution;
fusion-aware QAT (`prepare_qat(model, cfg, fused_bn=True)`) for the last epochs is the
next thing to try. To check with Durance: does the hardware compare `v >= threshold` and
reset to zero, as riverops' fixed-point emulation does?
