# Where the target (ball, drone) is in the image. Contract: reflex/runtime/target.py.
# header.stamp is the time of the newest event the estimate is based on.
std_msgs/Header header
float32 u            # -1 left edge .. +1 right edge of the image, as the camera sees it
float32 v            # -1 top edge .. +1 bottom edge; for a ball on the floor, larger = closer
float32 confidence   # 0..1; 0 means nothing seen (u, v meaningless)
string source        # centroid / snn / ...
