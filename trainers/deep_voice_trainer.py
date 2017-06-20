import logging

from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.optimizers import SGD

from data_generators.timit_data_generator import TimitBatchGenerator
from models.deep_voice_speaker_model import deep_voice_speaker_model
from utils.utils import get_mfcc


def deep_voice_features(y, sr, n_mfcc=20, tgt_sr=16000.0, win_len=0.025,
                        hop_len=0.010, n_fft=512, n_mels=128, fmin=0.0, fmax=8000.0):
    return get_mfcc(y, sr, n_mfcc, tgt_sr, win_len, hop_len, n_fft, n_mels, fmin, fmax)


def train(inp_shape, train_batch_generator, val_batch_generator=None,
          init_lr=0.001, epochs=1000, steps_per_epoch=20, workers=4, **kwargs):
    model = deep_voice_speaker_model(inp_shape, **kwargs)

    model.compile(optimizer=SGD(lr=init_lr, decay=1e-6, momentum=0.9, nesterov=True),
                  loss=categorical_crossentropy, metrics=[categorical_accuracy])
    logging.info(model.summary())
    model.fit_generator(
        train_batch_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, workers=workers)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    """
    paper : Deep Voice 2: Multi-Speaker Neural Text-to-Speech
    focus : Speaker Discriminative Model
    url   : https://arxiv.org/pdf/1705.08947.pdf
    """

    num_speakers = 20

    train_timit_path = '/Users/venkatesh/datasets/timit/data/lisa/data/timit/raw/TIMIT/TRAIN/'
    train_data_gen = TimitBatchGenerator(train_timit_path, num_speakers=num_speakers, batch_size=32)

    # val_timit_path = '/Users/venkatesh/datasets/timit/data/lisa/data/timit/raw/TIMIT/TEST'
    # val_data_gen = TimitBatchGenerator(val_timit_path, num_speakers=10, batch_size=)

    train(inp_shape=(train_data_gen.frames, train_data_gen.dim, 1), train_batch_generator=train_data_gen.generator(),
          workers=4, num_speakers=num_speakers, conv_rep=2, dropout=0.0)
