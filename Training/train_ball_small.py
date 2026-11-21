"""Smaller ball detector: the spiking U-Net of train_ball_size.py with fewer channels.

The network takes the 320x320 sensor frame and downsamples it itself: enc1 is
a learned 8x8 stride-4 ConvLIF (320x320 -> 80x80), so nothing has to be done
to the events outside the network. The rest of the topology, strides, skips,
head, loss, data and readout are those of train_ball_size.py, so the receptive
field stays ~300 sensor px and only RIVER-lowerable ops are used. Channel
widths shrink from (16, 32, 64, 64) to WIDTHS per resolution level (80x80,
40x40, 20x20, 10x10). Everything is written under its own names (ball_small_*),
so the full-size model's checkpoint and export files are left alone::

    python train_ball_small.py
    python train_ball_small.py train --no-ball data/no_ball
    python train_ball_small.py export
    python train_ball_small.py visualize
    python train_ball_small.py predict data/recording.h5 --window-ms 5
"""

from pathlib import Path

from torch import nn

import train_ball_size as size
from riverops.nn import NearestUpsample

WIDTHS = (8, 16, 24, 32)


class SmallBallUNet(nn.Module):
    """``train_ball_size.BallUNet`` with channels WIDTHS at 80x80, 40x40, 20x20 and 10x10.

    enc1 does the 4x downsampling from the 320x320 input. Decoder outputs match
    the encoder widths at the same grid so the additive skips line up. Modules
    are registered in forward order, which is the export order.
    """

    def __init__(self, widths=WIDTHS):
        super().__init__()
        c1, c2, c3, c4 = widths
        conv = size._conv
        self.enc1 = conv(1, c1, 8, stride=4, padding=2, first_layer=True)
        self.enc2 = conv(c1, c2, 4, stride=2, padding=1)
        self.enc3 = conv(c2, c2, 3)
        self.enc4 = conv(c2, c3, 4, stride=2, padding=1)
        self.enc5 = conv(c3, c3, 3)
        self.enc6 = conv(c3, c4, 4, stride=2, padding=1)
        self.enc7 = conv(c4, c4, 3)
        self.up1 = NearestUpsample((2, 2))
        self.dec1 = conv(c4, c3, 3)
        self.mix1 = conv(c3, c3, 3)
        self.up2 = NearestUpsample((2, 2))
        self.dec2 = conv(c3, c2, 3)
        self.mix2 = conv(c2, c2, 3)
        self.up3 = NearestUpsample((2, 2))
        self.dec3 = conv(c2, c1, 3)
        self.mix3 = conv(c1, c1, 3)
        self.head = conv(c1, size.HEAD_CHANNELS, 1, padding=0)

    forward = size.BallUNet.forward


# train_ball_size looks these up when it runs, so its training, export,
# visualize and predict commands use the small network and its own file names.
size.BallUNet = SmallBallUNet
size.ARCHITECTURE = "unet-position-small-v1"
size.CHECKPOINT = Path("ball_small_best.pt")
size.EXPORT_PREFIX = "ball_small"
size.SAMPLE_FIGURE = Path("ball_small_sample.png")
size.TEST_VIDEO = Path("ball_small_test.mp4")
size.PREDICT_SUFFIX = "_small_pred.mp4"

if __name__ == "__main__":
    size.main()
