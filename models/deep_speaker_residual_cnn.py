from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, add, Lambda, Dense

from utils.utils import ClippedRelu


def deep_speaker_residual_block(inp, filters, kernel_size=(3, 3), strides=(1, 1), clip_value=20):
    """ input ----> conv -> bn -> relu -> conv -> bn ----> relu --> output
                \_____________________________________/
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(inp)
    x = BatchNormalization()(x)
    x = ClippedRelu(clip_value=clip_value)(x)

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, name='{}_{}')(x)
    x = BatchNormalization()(x)
    x = add([x, inp], mode='sum')

    x = ClippedRelu(clip_value=clip_value)(x)
    return x


def deep_speaker_residual_cnn(inp, filters=None, kernel_size=(5, 5), strides=(2, 2), res_kernel_size=(3, 3),
                              res_strides=(1, 1), clip_value=20, res_repeat=3):
    """ input ---> [ conv -> relu -> [ res_block ]* ] * -> average -> affine -> length_normalization ---> embeddings
    """
    if filters is None:
        filters = [64, 128, 256, 512]

    x = None

    # [ conv -> relu -> [ res_block ]* ] *
    for i, f in enumerate(filters):
        if i == 0:
            x = Conv2D(filters=f, kernel_size=kernel_size, strides=strides)(inp)
        else:
            x = Conv2D(filters=f, kernel_size=kernel_size, strides=strides, name='conv_{}'.format(f))(x)
        x = ClippedRelu(clip_value=clip_value)(x)
        for j in range(res_repeat):
            x = deep_speaker_residual_block(x, f, kernel_size=res_kernel_size,
                                            strides=res_strides, clip_value=clip_value)

    # temporal average
    x = Lambda(lambda y: K.mean(y, axis=1))(x)
    # affine
    x = Dense(units=512)(x)
    # length normalization
    x = Lambda(lambda y: K.l2_normalize(y, axis=-1))(x)

    return x
