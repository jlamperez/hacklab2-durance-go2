"""Smaller ball detector: the spiking U-Net of train_ball_size.py with fewer channels.

The input is [2, 320, 320]: one binary channel per polarity (0 = OFF events,
1 = ON events) for each window. The first layer, merge, is a 1x1 ConvLIF that
turns the two channels into one 320x320 spike map; enc1 then downsamples it
itself with a learned 8x8 stride-4 ConvLIF (320x320 -> 80x80), so nothing has
to be done to the events outside the network beyond sorting them by
polarity. The rest of the topology, strides, skips,
head, loss, data and readout are those of train_ball_size.py, so the receptive
field stays ~300 sensor px and only RIVER-lowerable ops are used. Channel
widths shrink from (16, 32, 64, 64) to WIDTHS per resolution level (80x80,
40x40, 20x20, 10x10). It trains with int8 fake quantization (TRAIN_INT8), so
the weights it learns are the int8 weights the export writes. Everything is
written under its own names (ball_small_*), so the full-size model's
checkpoint and export files are left alone::

    python train_ball_small.py
    python train_ball_small.py train --no-ball data/no_ball
    python train_ball_small.py export
    python train_ball_small.py visualize
    python train_ball_small.py predict data/recording.h5 --window-ms 5
"""

from pathlib import Path

from torch import nn

import train_ball_size as size
import train_motion_buddy_fast as fast
from riverops.nn import NearestUpsample

WIDTHS = (8, 16, 24, 32)
POLARITIES = 2


class SmallBallUNet(nn.Module):
    """``train_ball_size.BallUNet`` with channels WIDTHS at 80x80, 40x40, 20x20 and 10x10.

    merge turns [B, T, 2, 320, 320] into [B, T, 1, 320, 320]; enc1 does the 4x
    downsampling from there. Decoder outputs match the encoder widths at the
    same grid so the additive skips line up. Modules are registered in forward
    order, which is the export order. ``unet_forward`` is the original
    ``BallUNet.forward``, taken before ``size.BallUNet`` is replaced below.
    """

    unet_forward = size.BallUNet.forward

    def __init__(self, widths=WIDTHS):
        super().__init__()
        c1, c2, c3, c4 = widths
        conv = size._conv
        self.merge = conv(POLARITIES, 1, 1, padding=0, first_layer=True)
        self.enc1 = conv(1, c1, 8, stride=4, padding=2)
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

    def forward(self, x):
        return self.unet_forward(self.merge(x))


# train_ball_size looks these up when it runs, so its training, export,
# visualize and predict commands use the small network and its own file names.
size.BallUNet = SmallBallUNet
size.ARCHITECTURE = "unet-position-small-polarity-v1"
fast.INPUT_CHANNELS = POLARITIES
size.CHECKPOINT = Path("ball_small_best.pt")
size.EXPORT_PREFIX = "ball_small"
size.SAMPLE_FIGURE = Path("ball_small_sample.png")
size.TEST_VIDEO = Path("ball_small_test.mp4")
size.PREDICT_SUFFIX = "_small_pred.mp4"
size.TRAIN_INT8 = True

if __name__ == "__main__":
    size.main()
