# Looking inside the ROS 2 system

Two terminals, both in the repository root.

**Terminal 1** runs the system (simulator, perception, behavior):

```bash
pixi run -e ros ros-sim
```

**Terminal 2** is for looking:

```bash
pixi run -e ros ros-shell      # a shell with ROS 2 and this workspace ready
```

## The graph

```
                 /reflex/events_window          /reflex/decision          /api/sport/request
 go2_sim_bridge ----------------------> reflex_perception ---------> reflex_behavior ---------+
      ^   (EventWindow, 50 Hz sim time)      (ReflexDecision)      (unitree_api/Request)     |
      +--------------------------------------------------------------------------------------+
```

On the real robot the bridge is replaced by two things: a camera adapter that publishes
`/reflex/events_window`, and the Go2 itself listening on `/api/sport/request`.

`rqt_graph` draws this live.

## Commands

| What you want to see | Command |
|---|---|
| Which nodes are running | `ros2 node list` |
| Which topics exist and their types | `ros2 topic list -t` |
| The node/topic graph | `rqt_graph` |
| What the model outputs | `ros2 topic echo /reflex/decision` |
| Only the threats | `ros2 topic echo /reflex/decision --filter "m.label != 'none'"` |
| What the camera side sends (without the big array) | `ros2 topic echo /reflex/events_window --no-arr` |
| The events as a picture | `ros2 run rqt_image_view rqt_image_view /reflex/debug/events_image` |
| What is sent to the dog | `ros2 topic echo /api/sport/request` |
| The score of each shot | `ros2 topic echo /reflex/sim/outcome` |
| How often a topic publishes | `ros2 topic hz /reflex/decision` |
| Who publishes and who listens | `ros2 topic info /api/sport/request -v` |
| A message definition | `ros2 interface show reflex_msgs/msg/ReflexDecision` |
| A node's parameters | `ros2 param list /reflex_behavior` |

## What the messages mean

`/reflex/events_window` (`reflex_msgs/EventWindow`): the network's input. The last 50 ms
of events, as counts in 10 time steps x 2 polarities x 45 x 80 cells (72000 bytes).
`header.stamp` is the time of the newest event.

`/reflex/decision` (`reflex_msgs/ReflexDecision`): the detector's answer for one window.

```
label: none            # none / left / center / right, in the dog frame
confidence: 0.736      # 0..1 (softmax probability for the SNN)
detector: snn          # snn or simple
inference_ms: 8.9      # wall time spent in the detector
```

`/api/sport/request` (`unitree_api/Request`): the command to the Go2.
`header.identity.api_id` selects the command (see `reflex/runtime/sport_api.py`),
`parameter` is a JSON string, and `header.identity.id` carries the timestamp (µs) of the
newest event behind the decision, for latency measurements.

Most decisions are `none`: the behavior node only sends a request after two agreeing,
confident threats in a row.

Times are simulated time (`/clock`). The simulation runs slower than real time, so
`ros2 topic hz` shows about 25 Hz of wall time for a topic published at 50 Hz of
simulated time.
