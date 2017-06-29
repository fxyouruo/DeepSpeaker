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
          init_lr=0.001, epochs=1000, steps_per_epoch=20, val_steps=20, workers=4, runs_dir=None, **kwargs):
    if runs_dir is None:
        runs_dir = 'deep_voice_' + str(int(time.time()))
    model = deep_voice_speaker_model(inp_shape, **kwargs)
    opt = Adam(lr=init_lr)
    model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=[categorical_accuracy])
    logging.info(model.summary())
    tb_callback = TensorBoard(log_dir=os.path.join(runs_dir, 'logs'), write_images=True)
    checkpointer = ModelCheckpoint(filepath=os.path.join(runs_dir, 'weights.hdf5'), verbose=0, period=10)
    model.fit_generator(train_batch_generator, steps_per_epoch=steps_per_epoch,
                        epochs=epochs, workers=workers, callbacks=[tb_callback, checkpointer],
                        validation_data=val_batch_generator, validation_steps=val_steps)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    """
    paper : Deep Voice 2: Multi-Speaker Neural Text-to-SpeechA
    focus : Speaker Discriminative Model
    url   : https://arxiv.org/pdf/1705.08947.pdf
    """

    num_speakers = 20

    timit_path = '/Users/venkatesh/datasets/timit/data/lisa/data/timit/raw/TIMIT/TRAIN/'
    data_gen = TimitBatchGenerator(timit_path, num_speakers=num_speakers, frames=64, file_batch_size=1)

    train(inp_shape=(data_gen.frames, data_gen.dim, 1), train_batch_generator=data_gen.generator('train'),
          workers=1, num_speakers=num_speakers, conv_rep=2, dropout=0.0, steps_per_epoch=40,
          val_batch_generator=data_gen.generator('dev'), val_steps=20)

