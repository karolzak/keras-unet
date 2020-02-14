14.02.2020 (0.0.8)

- Bumped version to 0.0.8
- Added support for TF >= 2.0 and solved issue #14 
- From this point forward `keras-unet` will import `tf.keras` instead of regular `Keras` when TF >= 2.0 is present.
- Added `color` param to `keras_unet.utils.plot_imgs` function allowing to change the mask color when plotting on top of orginal image. Thanks @muminoff picking that up
- Added some docs scafoldings for `keras_unet.utils`

27.01.2020 (0.0.7)

- Modified `custom_unet` to not use a bias when using BatchNorm
- Added `SpatialDropout2D` to `custom_unet`. Regular Dropout does not perform as well as Spatial Dropout in CNNs. Compare [here ](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L178).
- Added test coverage to the new code.
- Added a __version__ to __init__.py


