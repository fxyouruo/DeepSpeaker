import logging
import os
import random

import numpy as np
from keras.utils import to_categorical
from pydub import AudioSegment
from pydub.utils import mediainfo
from tensorflow.python.platform import gfile

from utils.utils import get_mfcc_v2, threadsafe_generator, normalize
from utils.utils import list_files, audio_predicate, remove_silence


class BaseBatchGenerator(object):
    np.random.seed(0)

    def feature_extractor(self, y, sr):
        mfccs = get_mfcc_v2(y, sr, delta=True, delta_delta=True)
        features = np.zeros(shape=(mfccs.shape[0] - self.frames, mfccs.shape[1] * self.frames))
        for i in range(mfccs.shape[0] - self.frames):
            features[i] = mfccs[i:i + self.frames].reshape(1, mfccs.shape[1] * self.frames)
        return normalize(features).reshape(-1, self.frames, mfccs.shape[1])

    def __init__(self, dir_path, num_speakers=100, frames=64, silence=False, batch_size=32, file_batch_size=16):
        self.frames = frames
        self.batch_size = batch_size
        self.num_speakers = num_speakers
        self.file_batch_size = file_batch_size
        files = list(list_files(dir_path, audio_predicate))
        speakers = list(set([os.path.basename(os.path.split(f)[0]) for f in files]))
        random.shuffle(speakers)
        speakers = dict(zip(speakers[:num_speakers], range(len(speakers[:num_speakers]))))

        self.speakers = speakers
        logging.info("Speakers: {}".format(self.speakers))
        logging.info("Number of speakers: {}".format(len(self.speakers)))

        speaker_files, cnt = {}, 0
        for f in files:
            speaker = os.path.basename(os.path.split(f)[0])
            if speaker in self.speakers:
                speaker_files[cnt] = f
                cnt += 1
        self.speaker_files = speaker_files

        speaker_features = {}
        for i, f in self.speaker_files.items():
            speaker = os.path.basename(os.path.split(f)[0])
            audio = AudioSegment.from_file(gfile.FastGFile(f))
            if silence:
                audio = remove_silence(audio, silence_threshold=-60.0)
            sr = float(mediainfo(f)['sample_rate'])
            speaker_feature = self.feature_extractor(np.array(audio.get_array_of_samples()), sr=sr)
            speaker_features[i] = (speaker_feature, np.repeat(self.speakers[speaker], speaker_feature.shape[0]))

        self.speaker_features = speaker_features
        logging.info("Speaker Features: {}".format(self.speaker_features.keys()))
        logging.info("Speaker Features: {}".format(self.speaker_files.keys()))
        self.dim = self.speaker_features.values()[0][0].shape[2]
        logging.info("Input dimensions: {}".format(self.dim))

    @threadsafe_generator
    def generator(self):
        speaker_files = self.speaker_files.keys()
        x = np.array([]).reshape((0, self.frames, self.dim))
        y = np.array([])
        while True:
            speakers_ids = random.sample(speaker_files, self.file_batch_size)
            logging.info("Speaker indices trained on: {}".format(speakers_ids))
            for j, i in enumerate(speakers_ids):
                x = np.vstack([x, self.speaker_features[i][0]])
                y = np.append(y, self.speaker_features[i][1])
            logging.info("Features shape: {}".format(x.shape))
            for i in range(x.shape[0] - self.batch_size):
                yield np.expand_dims(x[i: i + self.batch_size], -1), \
                      to_categorical(y[i: i + self.batch_size], num_classes=self.num_speakers)
