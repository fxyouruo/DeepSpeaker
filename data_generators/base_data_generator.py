import logging
import os
import random

import numpy as np
from keras.utils import to_categorical
from pydub import AudioSegment
from pydub.utils import mediainfo
from tensorflow.python.platform import gfile

from utils.utils import get_mfcc_v1, get_mfcc_v2, threadsafe_generator, get_energy
from utils.utils import list_files, audio_predicate, remove_silence, normalize


class BaseBatchGenerator(object):
    np.random.seed(0)

    @staticmethod
    def feature_extractor(y, sr):
        mfccs = get_mfcc_v2(y, sr, delta=True, delta_delta=True)
        return normalize(mfccs)

    def __init__(self, dir_path, num_speakers=100, frames=64, silence=True, batch_size=32):
        self.frames = frames
        self.batch_size = batch_size
        self.num_speakers = num_speakers
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
            print(speaker_feature.shape)
            speaker_features[i] = (speaker_feature, self.speakers[speaker])

        self.speaker_features = speaker_features
        logging.info("Speaker Features: {}".format(self.speaker_features.keys()))
        logging.info("Speaker Features: {}".format(self.speaker_files.keys()))
        self.dim = self.speaker_features.values()[0][0].shape[1]
        logging.info("Input dimensions: {}".format(self.dim))

    @threadsafe_generator
    def generator(self):
        speaker_files = self.speaker_files.keys()
        x = np.zeros(shape=(self.batch_size, self.frames, self.dim))
        y = np.zeros(shape=self.batch_size)
        while True:
            speakers_ids = random.sample(speaker_files, self.batch_size)
            for j, i in enumerate(speakers_ids):
                valid_frames = self.speaker_features[i][0].shape[0] - self.frames
                if valid_frames < 0:
                    continue
                start_frame = random.randint(0, valid_frames)
                x[j] = self.speaker_features[i][0][start_frame: start_frame + self.frames]
                y[j] = self.speaker_features[i][1]
            yield np.expand_dims(x, -1), to_categorical(y, num_classes=self.num_speakers)



