"""Training for Durance's RIVER toolchain (riverops), adapted from their Motion Buddy notebook.

Task: where is the target? `none` or one of `n_bins` horizontal zones (default
left / center / right), per 10 ms frame, from a GenX320 event stream.

- `data`:  read river `.bin` recordings, events -> frames, labels, windowed dataset
- `autolabel`: labels for recorded target takes (calibrated tracker, overrides.json)
- `review`: one image per session to check the labels; agreement with synthetic truth
- `synth`: synthetic bouncing-ball and distractor recordings in the same `.bin` format
- `model`: the conv-only spiking network built from `riverops.nn.SpikingConvBN2d`
- `train`: training, evaluation and export (`model.json`, `weights.txt`, `config.yaml`)

`data` and `synth` need only NumPy (and PyTorch for the Dataset); `model` and the
export need the riverops wheel.
"""
