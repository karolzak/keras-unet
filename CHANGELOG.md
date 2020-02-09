27.01.2020 (0.0.7)

- Modified `custom_unet` to not use a bias when using BatchNorm
- Added `SpatialDropout2D` to `custom_unet`. Regular Dropout does not perform as well as Spatial Dropout in CNNs. Compare [here ](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L178).
- Added test coverage to the new code.
- Added a __version__ to __init__.py


