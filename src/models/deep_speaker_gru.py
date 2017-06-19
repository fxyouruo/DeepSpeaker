from keras import backend as K
from keras.layers import Conv2D, Lambda, Dense, GRU


def deep_speaker_gru(inp, filters=64, kernel_size=(5, 5), strides=(2, 2), gru_filters=None):
    """ input ---> conv -> gru * -> average -> affine -> length_normalization ---> embeddings
    """
    # conv
    if gru_filters is None:
        gru_filters = [1024, 1024, 1024]

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(inp)
    # grus
    for i, f in enumerate(gru_filters):
        x = GRU(units=f, return_sequences=True)(x)
    # temporal average
    x = Lambda(lambda y: K.mean(y, axis=1))(x)
    # affine
    x = Dense(units=512)(x)
    # length normalization
    x = Lambda(lambda y: K.l2_normalize(y, axis=-1))(x)

    return x
