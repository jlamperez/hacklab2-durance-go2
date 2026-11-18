# Questions for Durance at the opening

Ask these before hacking starts (opening ceremony is Saturday 09:30).

1. Which camera and resolution? What is the recording format, and how do we read events
   live (ROS 2 topic, SDK)?
2. Which layers and neuron types does RIVERLib support? Limits on model size, time steps,
   quantization?
3. Does the KV260 image ship with ROS 2? Which distro? Can we install our own packages?
   Is Docker available on it?
4. How many Go2 robots and boards are there, and how is robot time shared? Where is the
   camera mounted on the dog?
5. Is the Go2 controlled through high-level sport mode? Are there speed limits or a
   fenced area?
6. How is the track judged and what is the prize? What weighs most: latency, power,
   sparsity, originality of the behavior?
7. Do you have data already recorded with the event camera?
8. How is the board's power consumption measured?

## Assumptions these answers will confirm or break

| Assumption in the code | Question |
|---|---|
| Prophesee-style sensor, events as (t, x, y, p) | 1 |
| LIF neuron, subtractive reset, conv/pool/linear only, 8–16 time steps, 8-bit weights | 2 |
| ROS 2 Humble with CycloneDDS on the board | 3 |
| Reflex triggers while the dog stands still, head-mounted camera | 4 |
| Commands go to `/api/sport/request` | 5 |
