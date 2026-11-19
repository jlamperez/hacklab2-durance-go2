# Recording real data and fine-tuning

Three steps: record takes, look at them, fine-tune. Works with anything that publishes
`/reflex/events_window`: the simulator, the webcam stand-in, the real camera adapter.

## 1. Record

With the camera side running (and the same `ROS_DOMAIN_ID`):

```bash
pixi run -e ros ros-record session:=sat_am
```

| Key | Action |
|---|---|
| `l` / `c` / `r` | Start a take: a ball that will arrive on the dog's left / center / right |
| `n` | Start a take of something that must not trigger a reflex |
| SPACE | Ball takes: press when the ball **arrives** (marks the time, ends the take). "None" takes: ends the take |
| `x` | Cancel the take in progress |
| `u` | Delete the last saved take |
| `q` | Quit |

Left and right are the **dog's**, not the operator's. Takes land in
`data/real/<session>/clip_NNN.npz` (git-ignored). A take longer than 20 s is cancelled.

One session = one stretch of recording under the same conditions (place, light, camera
mount). Start a new session name when those change, and keep one whole session aside for
validation.

What to record, per session:

- Balls on each side, at different speeds, sizes and colors. Each take only yields about
  0.4 s of "threat" windows (20 at 50 Hz), so many short takes beat a few long ones.
- "None" takes: the dog standing with nothing in front (do this first; it is also the
  noise calibration), people walking past, balls crossing sideways, balls that stop
  short, slow approaches.

## 2. Build and look

```bash
pixi run build-real data/real/*          # -> data/real_ds
pixi run view-data data/real_ds          # browse; --label left, --sheet out/real.png
```

Labels come from the arrival mark, when the set is built:

```
... none ... | left out (300 ms) | threat (400 ms) | left out
                                                   ^ arrival
```

If the key presses were consistently late or the ball is fast, change the rule without
recording again: `pixi run build-real data/real/* --threat-ms 300 --guard-ms 200`.

## 3. Fine-tune

```bash
pixi run finetune-real --val-session sat_pm
```

Builds the set, fine-tunes `out/models/snn.pt` (pretrained on simulation) for 15 epochs
and writes `out/models/snn_real.pt` with its metrics. `--val-session` holds a whole
session out: that is the number to quote. Without it, validation clips come from the
same sessions as training and the result is optimistic. Environment variables: `EPOCHS`,
`LR`, `INIT`.

Run the result: `detector:=snn model:=out/models/snn_real.pt`.

## Checked and not checked (19 Nov 2026)

Checked end to end with windows from the simulator and commands sent on
`/reflex/record/cmd`: recording, undo, building, viewing, fine-tuning with a held-out
session. On 1642 windows, 15 epochs took 2 s on the RTX 5090 and 41 s on the CPU, so the
laptop is a workable fallback.

Not checked: the keyboard itself (the test had no terminal), and whether the labelling
rule gives a good model. The test takes had arbitrary labels, so its accuracy means nothing.

Not written: export to ONNX or RIVERLib after fine-tuning.
