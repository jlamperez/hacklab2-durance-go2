# One network input: the last `window_ms` of events, binned.
# header.stamp is the time of the newest event in the window.
std_msgs/Header header
float32 window_ms
uint32 n_bins      # time steps
uint32 channels    # 2: OFF, ON
uint32 height
uint32 width
uint8[] data       # event counts, row-major (n_bins, channels, height, width)
