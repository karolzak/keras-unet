from keras_unet.models.custom_unet import conv2d_block
from keras import backend as K
import numpy as np


def test_conv2D_block_vanilla():
    v = K.variable(np.ones([1, 16, 16, 1]), dtype=np.float32)
    out = conv2d_block(v, use_batch_norm=True)
    assert out.shape.as_list() == [1, 16, 16, 16]


def test_conv2D_block_no_BN():
    v = K.variable(np.ones([1, 32, 32, 1]), dtype=np.float32)
    out = conv2d_block(v, use_batch_norm=False)
    assert out.shape.as_list() == [1, 32, 32, 16]


def test_conv2D_block_filters():
    v = K.variable(np.ones([1, 32, 32, 1]), dtype=np.float32)
    out = conv2d_block(v, use_batch_norm=False, filters=32)
    assert out.shape.as_list() == [1, 32, 32, 32]


def test_conv2D_block_elu():
    v = K.variable(np.ones([1, 32, 32, 1]), dtype=np.float32)
    out = conv2d_block(v, activation="elu")
    assert out.shape.as_list() == [1, 32, 32, 16]
