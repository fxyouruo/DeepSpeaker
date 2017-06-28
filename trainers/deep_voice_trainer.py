import logging
import os
import time

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam

from data_generators.timit_data_generator import TimitBatchGenerator
from models.deep_voice_speaker_model import deep_voice_speaker_model


def train(inp_shape, train_batch_generator, val_batch_generator=None,
          init_lr=0.001, epochs=1000, steps_per_epoch=20, workers=4, runs_dir=None, **kwargs):
    if runs_dir is None:
        runs_dir = 'deep_voice_' + str(time.time())
    model = deep_voice_speaker_model(inp_shape, **kwargs)
    opt = Adam(lr=init_lr)
    model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=[categorical_accuracy])
    logging.info(model.summary())

    tb_callback = TensorBoard(log_dir=os.path.join(runs_dir, 'logs'), histogram_freq=0, write_graph=True,
                              write_images=True)
    checkpointer = ModelCheckpoint(filepath=os.path.join(runs_dir, 'weights.hdf5'), verbose=1)
    model.fit_generator(train_batch_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, workers=workers,
                        callbacks=[tb_callback, checkpointer])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    """
    paper : Deep Voice 2: Multi-Speaker Neural Text-to-Speech
    focus : Speaker Discriminative Model
    url   : https://arxiv.org/pdf/1705.08947.pdf
    """

    num_speakers = 20

    train_timit_path = '/Users/venkatesh/datasets/timit/data/lisa/data/timit/raw/TIMIT/TRAIN/'
    train_data_gen = TimitBatchGenerator(train_timit_path, num_speakers=num_speakers, batch_size=64)

    # val_timit_path = '/Users/venkatesh/datasets/timit/data/lisa/data/timit/raw/TIMIT/TEST'
    # val_data_gen = TimitBatchGenerator(val_timit_path, num_speakers=10, batch_size=64)

    train(inp_shape=(train_data_gen.frames, train_data_gen.dim, 1), train_batch_generator=train_data_gen.generator(),
          workers=4, num_speakers=num_speakers, conv_rep=2, dropout=0.0)
