# Output of a detector for one window.
# header.stamp is copied from the window: the time of the newest event it is based on.
std_msgs/Header header
string label          # none / left / center / right, in the dog frame
float32 confidence    # 0..1
string detector       # snn / simple
float32 inference_ms  # wall time spent in the detector
