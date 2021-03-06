import logging
import os
import time

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam

from data_generators.base_data_generator import BaseBatchGenerator
from models.deep_voice_speaker_model import deep_voice_speaker_model
from utils.utils import SaverCallback, LoggingCallback, mkdir
from keras.models import load_model


def train(inp_shape, train_batch_generator, val_batch_generator=None,
          init_lr=0.001, epochs=1000, steps_per_epoch=20, val_steps=20, workers=4, runs_dir=None, **kwargs):
    if runs_dir is None:
        runs_dir = 'deep_voice_' + str(int(time.time()))
    mkdir(runs_dir, True)
    model = deep_voice_speaker_model(inp_shape, **kwargs)
    opt = Adam(lr=init_lr)
    model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=[categorical_accuracy])
    logging.info(model.summary())
    tb_callback = TensorBoard(log_dir=os.path.join(runs_dir, 'logs'), write_images=True)
    lc = LoggingCallback()
    sc = SaverCallback(saver=tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=0.5),
                       save_path=runs_dir,
                       model=model, name='deep_voice_cnn')
    model.fit_generator(train_batch_generator, steps_per_epoch=steps_per_epoch,
                        epochs=epochs, workers=workers, callbacks=[tb_callback, lc, sc],
                        validation_data=val_batch_generator, validation_steps=val_steps,
                        verbose=0)


def main(_):
    mkdir(os.path.join(FLAGS.runs_dir, "training-log.txt"))
    logging.basicConfig(level=logging.INFO, filename=os.path.join(FLAGS.runs_dir, "training-log.txt"),
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info(FLAGS.__dict__)
    # logging.basicConfig(level=logging.INFO)

    """
    paper : Deep Voice 2: Multi-Speaker Neural Text-to-SpeechA
    focus : Speaker Discriminative Model
    url   : https://arxiv.org/pdf/1705.08947.pdf
    """

    num_speakers = FLAGS.num_speakers
    pkl_dir = FLAGS.pkl_dir
    data_gen = BaseBatchGenerator(FLAGS.data_path, num_speakers=num_speakers,
                                  frames=FLAGS.frames, id=FLAGS.id, pkl_dir=pkl_dir)
    train(inp_shape=(data_gen.frames, data_gen.dim, 1), train_batch_generator=data_gen.generator('train'),
          num_speakers=num_speakers, conv_rep=FLAGS.conv_rep, dropout=FLAGS.dp, steps_per_epoch=FLAGS.steps_per_epoch,
          val_batch_generator=data_gen.generator('dev'), val_steps=FLAGS.val_steps, runs_dir=FLAGS.runs_dir)


if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    # /Users/venkatesh/datasets/timit/data/lisa/data/timit/raw/TIMIT/TRAIN
    flags.DEFINE_string('runs_dir', "/Users/venkatesh/datasets/timit/data/lisa/data/timit/raw/TIMIT/TRAIN",
                        'Runs path for tensorboard')
    flags.DEFINE_string('data_path', "", 'Dataset path')
    flags.DEFINE_integer('num_speakers', 200, 'Number of speakers')
    flags.DEFINE_integer('frames', 64, 'Number of frames')
    flags.DEFINE_integer('conv_rep', 5, 'Number of conv layers')
    flags.DEFINE_integer('steps_per_epoch', 200, 'Steps for epoch')
    flags.DEFINE_integer('val_steps', 100, 'Validation steps')
    flags.DEFINE_float('dp', 0.0, 'Dropout')
    flags.DEFINE_string('pkl_dir', "", 'Temporary directory to store input data')
    flags.DEFINE_string('id', "TIMIT", 'Corpus identifier')
    tf.app.run()
