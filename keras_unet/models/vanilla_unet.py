from keras_unet import TF
if TF:
    from tensorflow.keras.models import Model
    from tensorflow.keras.backend import int_shape
    from tensorflow.keras.layers import (
        BatchNormalization, Conv2D, Conv2DTranspose,
        MaxPooling2D, Dropout, Input, concatenate, Cropping2D
    )
else:
    from keras.models import Model
    from keras.backend import int_shape
    from keras.layers import (
        BatchNormalization, Conv2D, Conv2DTranspose,
        MaxPooling2D, Dropout, Input, concatenate, Cropping2D
    )

from .custom_unet import conv2d_block


def vanilla_unet(
    input_shape,
    num_classes=1,
    dropout=0.5, 
    filters=64,
    num_layers=4,
    output_activation='sigmoid'): # 'sigmoid' or 'softmax'

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs   

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='valid')
        down_layers.append(x)
        x = MaxPooling2D((2, 2), strides=2) (x)
        filters = filters*2 # double the number of filters with each layer

    x = Dropout(dropout)(x)
    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='valid')

    for conv in reversed(down_layers):
        filters //= 2 # decreasing number of filters with each layer 
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='valid') (x)
        
        ch, cw = get_crop_shape(int_shape(conv), int_shape(x))
        conv = Cropping2D(cropping=(ch, cw))(conv)

        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='valid')
    
    outputs = Conv2D(num_classes, (1, 1), activation=output_activation) (x)    
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = target[2] - refer[2]
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = target[1] - refer[1]
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)