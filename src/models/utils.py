from keras.engine.topology import Layer
from keras import backend as K


class ClippedRelu(Layer):
    def __init__(self, clip_value, **kwargs):
        self.clip_value = clip_value
        super(ClippedRelu, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return K.minimum(K.maximum(x, 0), self.clip_value)
