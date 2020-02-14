name = "keras_unet"

__version__ = "0.0.7"
# TODO add __all__

# if tensorflow 2.x is present use tf.keras instead of Keras
try:
    from tensorflow import __version__ as TF_VERSION
    from packaging import version
    version.parse(TF_VERSION) >= version.parse("2.0.0")
    TF = True
except ImportError:
    TF = False
print('-----------------------------------------')
if TF:
    print('keras-unet init: TF version is >= 2.0.0 - using tf.keras instead of Keras')
else:
    print('keras-unet init: TF version is < 2.0.0 or not present - using Keras instead of tf.keras instead')
print('-----------------------------------------')

from . import models
from . import losses
from . import metrics
from . import metrics_np
from . import utils
