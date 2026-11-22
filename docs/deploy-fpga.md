# Putting a network on the FPGA: the files and where they come from

Every file Durance's KV260 needs to run one of our riverops networks on SPLEAT (the FPGA
inference server), which tool makes it, where it lives in this repo, and where it goes on
the board. Written on 21 Nov 2026 for the small ball detector. **Nothing here has been
loaded on the board until 21 Nov evening**: see "Current state" for what was run and
what has not been seen yet. How camera, server and Durance's ROS 2 nodes connect, and the
server's message format: `docs/fpga-pipeline.md`.

## Setup, once

Durance's two tools are proprietary wheels, not in git. Get them from Durance's download
list (`https://riversdk.dev/docs/hackathon/download/`, needs the hackathon login) and put
them in `third_party/` (gitignored):

| Wheel | Used for | pixi environment |
|---|---|---|
| `riverops-0.2.0-py3-none-any.whl` | `export-ball` (checkpoint -> JSON files) | `tools/river/pixi.toml`, default |
| `river_tools-0.1.0-py3-none-any.whl` | `river-tools deploy` (and `capture`, `play`) | `tools/river/pixi.toml`, `tools` |

```bash
cp ~/Downloads/riverops-0.2.0-py3-none-any.whl ~/Downloads/river_tools-0.1.0-py3-none-any.whl third_party/
pixi install --manifest-path tools/river/pixi.toml            # riverops, Python 3.12, torch CPU
pixi install --manifest-path tools/river/pixi.toml -e tools   # river-tools, Python 3.10
pixi run --manifest-path tools/river/pixi.toml -e tools river-tools --help
```

Two environments because river-tools requires Python 3.10 and riverops' torch is
installed for 3.12. They live in their own manifest, not in the main `pixi.toml`, so the
main workspace installs without the wheels. Durance's docs install river-tools with
`pipx`; here it is pixi, like everything else. Not needed: the riversdk.dev login for the
firmware job, which is done in the browser.

## The chain

```
checkpoint (.pt) ──export-ball──> model.json + parameters.json + readout.json
                                      │              │
                       riversdk.dev ──┘              │
                      (firmware job)                 │
                            │                        │
                firmware-kv260.tar.gz                │
                            └──────spleat-bundle─────┘
                                        │
       weights.txt, config.yaml, metavision_camera.json, spleat.bit.bin, spleat.dtbo, deploy.yaml
                                        │
                         river-tools deploy (pixi) ──> Kria
```

Two things are independent, and that is what makes iterating cheap:

- **The bitstream depends only on the model graph** (`*_model.json`: layers, channels,
  connections). It does not contain weights. One firmware job serves every retrained
  version of the same architecture.
- **The weights go separately**, as `weights.txt`. A new checkpoint of the same
  architecture only changes `weights.txt` (and `metavision_camera.json` if its event
  window changed).

## Files to the board

| File | Destination on the board | Made by | Changes when |
|---|---|---|---|
| `spleat.bit.bin` | `/usr/lib/firmware/xilinx/spleat/spleat.bit.bin` | riversdk.dev firmware job, from `*_model.json` | the architecture changes |
| `spleat.dtbo` | `/usr/lib/firmware/xilinx/spleat/spleat.dtbo` | same job, same archive | the architecture changes |
| `weights.txt` | `/root/weights.txt` | `spleat-bundle`, from `*_parameters.json` | any new checkpoint |
| `config.yaml` | `/root/.config/spleat-server/config.yaml` | `spleat-bundle`, from `*_model.json` | input or output size changes |
| `metavision_camera.json` | `/root/metavision_camera.json` | `spleat-bundle`; `frequency` from the checkpoint | input size or event window changes |

`shell.json` (same firmware folder on the board) is not in riversdk.dev's archive; the one
on Durance's card stays.

## Files that stay on the laptop

| File | Where | What it is for |
|---|---|---|
| `*_best.pt` | `models/ball*/` | the trained model; source of everything else |
| `*_model.json` | `models/ball*/` | River Model Graph: input of the firmware job and of `spleat-bundle` |
| `*_parameters.json`, `.txt` | `models/ball*/` | fixed-point weights; `.txt` is byte-identical to `weights.txt` |
| `*_readout.json` | `models/ball*/` | how to decode the network output (scales, no-ball logit); used off-chip |
| `PROVENANCE.md`, `export.log` | `models/ball*/` | how the export was made, versions, sha256 |
| `firmware-kv260.tar.gz`, `PROVENANCE.md` | `models/firmware/<board>_<model>_job<N>/` | the firmware as downloaded, and the job and model graph it was built from |
| `deploy.yaml` | `out/spleat/<model>/` | profile for `river-tools deploy`: the five board files, destinations, sha256 |
| `README.md`, `manifest.json` | `out/spleat/<model>/` | where every value of the board files comes from, checks, sha256 |

`models/` is in git; `out/` is generated and can be rebuilt with the commands below.

## Steps

1. **Export the checkpoint** (needs `third_party/riverops-0.2.0-py3-none-any.whl`):

   ```bash
   pixi run --manifest-path tools/river/pixi.toml export-ball models/ball_small/ball_small_best.pt \
       models/ball_small --script train_ball_small.py --commit <training commit>
   ```

2. **Firmware**, only if the architecture is new. Compare first:

   ```bash
   sha256sum models/ball_small/ball_small_model.json
   grep "Model file" models/firmware/*/PROVENANCE.md
   ```

   Same sha256: reuse that firmware. Otherwise, on riversdk.dev: "New firmware", upload
   `*_model.json`, board **KV260**, Prophesee IP **GenX320**, weights **Q1.7**, potentials
   **Q4.12** (the formats riverops exports; the form's default Q4.12 for weights would be
   wrong). Download the archive into `models/firmware/<board>_<model>_job<N>/` and write
   its `PROVENANCE.md` like the existing one: `spleat-bundle` reads the model sha256 from it.

3. **Board files**:

   ```bash
   pixi run spleat-bundle models/ball_small/ball_small_model.json models/ball_small/ball_small_parameters.json \
       --firmware models/firmware/kv260_ball_small_job35 --checkpoint models/ball_small/ball_small_best.pt
   ```

   It refuses firmware built from another model graph, and writes everything to
   `out/spleat/ball_small_model/`.

4. **Back up the board** (the Durance example is known to work):

   ```bash
   ssh root@192.168.77.1 'mkdir -p /root/backup && cp /root/weights.txt /root/metavision_camera.json \
       /root/.config/spleat-server/config.yaml /root/backup/ && cp -r /usr/lib/firmware/xilinx/spleat /root/backup/'
   ```

5. **Copy** with Durance's tool (it checks sha256 before copying and again on the board):

   ```bash
   pixi run --manifest-path tools/river/pixi.toml -e tools river-tools deploy out/spleat/ball_small_model/deploy.yaml root@192.168.77.1 --check-only
   pixi run --manifest-path tools/river/pixi.toml -e tools river-tools deploy out/spleat/ball_small_model/deploy.yaml root@192.168.77.1 --replace
   ```

6. **Restart and look**, on the board (`ss` is not on Durance's image):

   ```bash
   systemctl restart spleat-xmutil-loadapp spleat-server     # or reboot
   journalctl -u spleat-xmutil-loadapp -n 10                 # "spleat: Loaded with slot_handle 0"
   journalctl -u spleat-server -n 10                         # "read config" with our inputs/outputs
   grep -i ":0D05 " /proc/net/tcp6                           # port 3333 listening
   ```

7. **Feed it frames.** `despot_send_and_forget` always sends 2 channels; for a 1-channel
   network put `tools/spleat_merge_proxy.py` between despot and the server (its docstring
   has the board commands: server on port 3334, proxy on 3333). The replies come back on
   the proxy's connection; it saves them with `--save`, and `pixi run spleat-read --file F`
   decodes them on the laptop. `despot_test_camera` is needed once per boot and after every
   bitstream reload. **Read the results** from the laptop directly (if the network takes
   despot's frames as they are):

   ```bash
   pixi run spleat-read --save out/spleat_capture.bin        # Ctrl+C to stop
   ```

   Per message: size, number of values (9x80x80 expected), `latency`, min/max/sum, and
   `(x, y, P(ball))` decoded with the readout. It assumes the payload is the head's spike
   counts in channel, row, column order; the first real messages tell whether that holds.

**Back to Durance's example**: the files from before were saved on the board in
`/root/backup/durance-example/` (and on the laptop in `out/kria_backup/`); copy them back to
their places and restart the two services.

## Current state (22 Nov 2026)

| | |
|---|---|
| Firmware | `models/firmware/kv260_ball_small_job35/`: job #35, small U-Net, generated. The full-size network failed in place-and-route twice |
| Model | `models/ball_small/` holds an early QAT checkpoint (epoch 4, 2 ms window, not in git): good enough to test the board path, not to detect balls. A better one, trained on 5 ms windows, is expected |
| Board (21 Nov, about 23:20) | **deployed with river-tools**: sha256 of the five files checked on the board; bitstream loaded (`dfx-mgr-client: spleat: Loaded with slot_handle 0`, DMA channels created); `backward-server` running with our config (1x320x320 in, 9x80x80 out); port 3333 accepts connections. **Not seen yet**: a single inference. No camera client was running, so no frame reached the server and no message came out |
| Board (22 Nov) | despot's frames are 2x320x320 and the server refuses them; through `tools/spleat_merge_proxy.py` (channels merged, server on port 3334) they are accepted. **One inference came back** (9x80x80, `latency: 219331`), then the FPGA hung: `rx hardware status: ProxyTimeout`. After a bitstream reload the first frame did not come back. Logs: `docs/fpga-run-2026-11-22.md`; question to Durance in `docs/questions-for-durance.md` |

## Not known yet

- **What the camera client sends for our config.** The model was trained on one binary
  channel (1 where a pixel had any event) at 320x320. Durance's camera client builds frames
  of 2 polarities with event counts. Whether `channels: 1` makes it merge polarities, and
  whether it sends counts above 1, is unknown: the biggest risk for this model. The
  binary's strings (22 Nov) show 2 channels fixed at compile time, with no config key for
  channels (`docs/despot.md`): most likely it does not merge.
  **Seen on the board, 22 Nov**: with `despot_send_and_forget` running, `backward-server`
  receives messages every few ms and rejects each one: `level=ERROR msg="error validating
  message" service=spleat-facade err="wrong payload size"`. No inference ran. The sizes
  themselves were cut off in the log. **Captured on 22 Nov** (`tools/spleat_sniff.py` in
  place of the server, `downsampling: 1`, `frequency: 500`): 4-byte big-endian length + CBOR
  map with two keys, `{"id": n, "payload": <819200 bytes>}`; the payload is **2x320x320
  float32 little-endian event counts** (first frame values 0-3, later frames 0/1 with a few
  dozen non-zero pixels). Our server expects 1x320x320 (409600 bytes). Adapter:
  `tools/spleat_merge_proxy.py` (despot -> 3333 -> merge, any event = 1.0 -> 3334 -> server).
  **Run on the board, 22 Nov**: the server accepts the merged frames (`sending to spleat
  msg_id=0 shape=1x320x320`). First run: one reply, 9x80x80 float32, `latency: 219331`, then
  the FPGA never answered the second frame: `error reading through dma: rx hardware status:
  ProxyTimeout` (msg_id=1). With 40 frames in flight the server then refused everything
  ("pipeline is already full"). After reloading the bitstream and with one frame in flight
  (`--max-inflight 1`), the first frame got no reply in 20 s. So far: one inference out of
  two attempts, then the hardware hangs. Suspects: the 9x80x80 spatial output or the
  320x320 input (Durance's example is 2x80x80 in, 5x1x1 out). To ask Durance.
- **What the payload holds**: spike counts summed over the time steps (what the readout
  expects), spikes of one step, or potentials. `spleat-read` assumes the first.
- **Whether the server sends a 9x80x80 output** (230 kB per message). Durance's own client
  caps messages at 64 kB; spatial outputs are untested on the FPGA.
- **`frequency`**: one camera frame per event window, as in Durance's example (100 Hz,
  10 ms frames, `docs/fpga-pipeline.md`); the port is 3333 (the node's default).
- **Time steps per frame** and whether the server resets the neuron state between frames.
  Training feeds each frame for 4 steps from a zero state.
- **The server config format** on the real board: compare with the file already there
  before replacing it.
