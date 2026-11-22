# Questions for Durance at the opening

Ask these before hacking starts (opening ceremony is Saturday 09:30).

Some are partly answered by reading their KV260 microSD image on 21 Nov
(`docs/durance-kria-image.md`); those now ask for the part the card does not say.

1. Which camera is fitted: GenX320 (320×320) or IMX636 (1280×720)? The image has drivers
   for both and OpenEB 3.1.2. How do we read events live: V4L2 device, OpenEB C++, a ROS 2
   topic? Is there a recording format you already use (`.raw`, `.dat`)?
2. SPLEAT (`backward-server`): is it RIVERLib, or what RIVERLib runs on? Which layers and
   neuron types does it support? Limits on model size, time steps, quantization?
3. How do we produce the `weights_file` from a PyTorch model? Is there an exporter or a
   sample model?
4. What is `backward-server`'s protocol (CBOR over TCP? request and reply layout)? Is
   there a client library or sample in Python? What do `inputs` (`width`, `height`,
   `channels`, `scaling`) and `outputs` look like for the demo `application`?
5. Can we install our own packages on the image (no package manager, 788 MB free on root),
   or should our ROS nodes run on a laptop? Is `/opt/unitree` (the Go2 messages) supposed
   to be there, and is `durance_bringup` available?
6. Which USB-Ethernet adapter is the one with MAC `6c:6e:07:24:09:12` that the image
   expects for the Go2 network?
7. How many Go2 robots and boards are there, and how is robot time shared? Where is the
   camera mounted: on the dog (head, back, tilted?) or off the robot? Which lens and
   field of view?
8. Is the Go2 controlled through high-level sport mode? Is low-level motor control
   allowed? Which acrobatics (jump, flips) are enabled and permitted? Are there speed
   limits or a fenced area?
9. How is the track judged and what is the prize? What weighs most: latency, power,
   sparsity, originality of the behavior?
10. Do you have data already recorded with the event camera?
11. How is the board's power consumption measured?

## SPLEAT hangs on our model (asked on 22 Nov)

Logs: `docs/fpga-run-2026-11-22.md`.

> Your example model runs fine on our KV260: 10 frames out of 10, 0.5-30 ms each. Our
> small U-Net (firmware job #35, input 1x320x320, output 9x80x80, weights Q1.7, potentials
> Q4.12) hangs: after a clean boot, even an all-zero frame never comes back, and after
> about 60 s backward-server logs `error reading through dma: rx hardware status:
> ProxyTimeout` (kernel: `DMA timed out`). The frame goes out in 4 DMA transfers, 2 receive
> transfers complete, then nothing. Once, on the very first run, one inference did come
> back (latency 219331 us). Are spatial outputs (9x80x80) or 320x320 inputs supported by
> SPLEAT? Is there a size limit, or something in the model graph that stops the
> accelerator from ending the inference? Our graph has 3 skip connections and 3 layers with
> `upscaling: 2` (a U-Net); your example is a plain chain. Are skips and upscaling supported
> on the FPGA? (Weights encoding ruled out: signed, two's complement and 16-bit biases all
> hang the same way.)

Also worth asking: whether `despot_send_and_forget` can send one channel (it builds 2
polarities, fixed at compile time; we merge them in `tools/spleat_merge_proxy.py`).
Answered by our own test: `latency` is in microseconds.

Answer: not yet.

Our own follow-up (22 Nov, `docs/fpga-zones3-2026-11-22.md`): a Durance-shaped network
(2x80x80 -> 5 convolutions in a chain -> 4x1x1, job #44) runs on the same board, so the hang is
architectural. Still worth asking which of skips, upscaling, 320x320 input or a spatial output
SPLEAT supports. Also worth telling them: riverops 0.2.0's `export_params` writes signed
weights, while the board reads `weights.txt` as two's complement (0-255). Written signed, the
network runs but computes something else.

## Assumptions these answers will confirm or break

| Assumption in the code | Question | What the card says |
|---|---|---|
| Prophesee-style sensor, events as (t, x, y, p) | 1 | Consistent |
| LIF neuron, subtractive reset, conv/pool/linear only, 8–16 time steps, 8-bit weights | 2, 3 | Nothing |
| Inference called from Python in the perception node | 4 | A separate server, reached over the network |
| ROS 2 Humble with CycloneDDS on the board | 5 | Confirmed |
| Reflex triggers while the dog stands still, head-mounted camera | 7 | Nothing |
| Commands go to `/api/sport/request`; see `go2-control.md` | 8 | Board has a Go2 network (`192.168.123.99`), no Unitree messages |
