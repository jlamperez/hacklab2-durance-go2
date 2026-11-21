# Putting a network on the FPGA: the files and where they come from

Every file Durance's KV260 needs to run one of our riverops networks on SPLEAT (the FPGA
inference server), which tool makes it, where it lives in this repo, and where it goes on
the board. Written on 21 Nov 2026 for the small ball detector. **Nothing here has been
loaded on the board yet**; the board side follows Durance's deploy page
(`docs/durance-riversdk.md`), not our own test.

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

6. **Restart and look**, on the board:

   ```bash
   systemctl restart spleat-xmutil-loadapp spleat-server     # or reboot
   journalctl -u spleat-server -n 50
   ss -tlnp | grep 3333
   ```

## Current state (21 Nov 2026, evening)

| | |
|---|---|
| Firmware | `models/firmware/kv260_ball_small_job35/`: job #35, small U-Net, generated. The full-size network failed in place-and-route twice |
| Model | `models/ball_small/` holds an early QAT checkpoint (epoch 4, 2 ms window, not in git): good enough to test the board path, not to detect balls. A better one, trained on 5 ms windows, is expected |
| Board | nothing deployed yet |

## Not known yet

- **How the network output reaches us.** Something has to send camera frames to the
  server and read the 9x80x80 output. Durance's ROS 2 package `bringup`
  (`go2-kria-generated.zip`, `ros2 launch bringup pipeline.launch.py port:=...`) does it for
  their example; whether it handles another output shape is unknown.
- **The port**: 3333 in the server config of the deploy page, 41017 in its launch example.
- **`frequency`**: we assume one camera frame per event window (Durance's example: 100 Hz,
  10 ms frames). Not confirmed.
- **Time steps per frame** and whether the server resets the neuron state between frames.
  Training feeds each frame for 4 steps from a zero state.
- **The server config format** on the real board: compare with the file already there
  before replacing it.
