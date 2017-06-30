from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, Lambda, Dense, Dropout, MaxPool2D, Flatten, GlobalAveragePooling2D
from keras.models import Sequential

from utils.utils import ClippedRelu


def conv_bn_block(inp_shape, filters=5, kernel_size=(9, 5), strides=(1, 1), clip_value=6, dropout=0.75, name='conv_bn'):
    layers = [
        Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, input_shape=inp_shape, padding='same'),
        # BatchNormalization(),
        ClippedRelu(clip_value=clip_value),
        # Dropout(dropout)
    ]
    model = Sequential(layers, name=name)
    return model


def deep_voice_speaker_model(inp_shape, filters=32, kernel_size=(9, 5), strides=(1, 1), clip_value=6, dropout=0.75,
                             conv_rep=5, pool_size=(2, 2), maxpool_strides=(2, 2), dnn_units=16, num_speakers=100):
    """
    paper: Deep Voice 2: Multi-Speaker Neural Text-to-Speech
    focus : Speaker Discriminative Model
    url  : https://arxiv.org/pdf/1705.08947.pdf
    """

    model = Sequential()
    for i in range(conv_rep):
        if i == 0:
            model.add(conv_bn_block(inp_shape, filters=filters, kernel_size=kernel_size, strides=strides,
                                    clip_value=clip_value, dropout=dropout, name='conv_bn_{}'.format(i)))
        else:
            model.add(conv_bn_block(model.output_shape[1:], filters=filters, kernel_size=kernel_size, strides=strides,
                                    clip_value=clip_value, dropout=dropout, name='conv_bn_{}'.format(i)))
    # max pooling
    model.add(MaxPool2D(pool_size=pool_size, strides=maxpool_strides))

    # temporal average
    model.add(Lambda(lambda y: K.mean(y, axis=1), name="temporal_average"))

    # flatten
    model.add(Flatten(name="flatten"))

    # dnn
    model.add(Dense(units=dnn_units, activation='relu', name="dense"))

    # softmax
    model.add(Dense(units=num_speakers, activation='softmax'))
    return model
