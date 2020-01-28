from keras_unet.models.custom_unet import conv2d_block, custom_unet
from keras import backend as K
import numpy as np
import pytest


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


def test_conv2D_block_standard_dropout():
    v = K.variable(np.ones([1, 16, 16, 1]), dtype=np.float32)
    out = conv2d_block(v, use_batch_norm=True, dropout_type="standard")
    assert out.shape.as_list() == [1, 16, 16, 16]


def test_conv2d_block_unsupported_dropout():
    v = K.variable(np.ones([1, 16, 16, 1]), dtype=np.float32)
    with pytest.raises(ValueError):
        out = conv2d_block(v, use_batch_norm=True, dropout_type="foo")


def test_custom_unet():
    model = custom_unet((224, 224, 3))
    model.compile("sgd", "binary_crossentropy", ["acc"])
    assert all(model.output_shape) == all((None, 224, 224, 1))
    del model

