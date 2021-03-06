import logging
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from pydub import AudioSegment
from tensorflow.python.platform import gfile

from utils.utils import get_mfcc_v2, threadsafe_generator, normalize
from utils.utils import list_files, audio_predicate, remove_silence, mkdir


class BaseBatchGenerator(object):
    np.random.seed(0)

    def feature_extractor(self, y, sr):
        mfccs = get_mfcc_v2(y, sr, delta=True, delta_delta=True)
        if mfccs.shape[0] < self.frames:
            return None
        features = np.zeros(shape=(mfccs.shape[0] - self.frames, mfccs.shape[1] * self.frames))
        for i in range(mfccs.shape[0] - self.frames):
            features[i] = mfccs[i:i + self.frames].reshape(1, mfccs.shape[1] * self.frames)
        return normalize(features).reshape(-1, self.frames, mfccs.shape[1])

    def __init__(self, dir_path, num_speakers=100, frames=64, silence=True, val_frac=0.2, id="", pkl_dir=""):
        self.frames = frames
        self.num_speakers = num_speakers
        self.val_frac = val_frac
        self.id = id
        self.pkl_dir = os.path.join(pkl_dir, self.id)
        mkdir(self.pkl_dir, True)
        self.silence = silence

        files = list(list_files(dir_path, lambda x: audio_predicate(x) and x.find(self.id) != -1))
        speakers = list(set([os.path.basename(os.path.split(f)[0]) for f in files]))
        random.shuffle(speakers)
        speakers = dict(zip(speakers[:num_speakers], range(len(speakers[:num_speakers]))))

        self.speakers = speakers
        logging.info("Speakers: {}".format(self.speakers))
        logging.info("Number of speakers: {}".format(len(self.speakers)))

        speaker_files, speaker_files_count, cnt = {}, defaultdict(float), 0
        for f in files:
            speaker = os.path.basename(os.path.split(f)[0])
            if speaker in self.speakers:
                speaker_files[cnt] = f
                speaker_files_count[speaker] += 1.0
                cnt += 1
        self.speaker_files = speaker_files
        logging.info("Number of speaker files: {}".format(len(self.speaker_files)))

        speaker_train_files, speaker_dev_files, speaker_curr_count, featurized = [], [], defaultdict(float), {}
        for i, f in self.speaker_files.items():
            speaker = os.path.basename(os.path.split(f)[0])
            if speaker_curr_count[speaker] < (1.0 - self.val_frac) * speaker_files_count[speaker]:
                speaker_train_files.append(i)
                featurized[i] = False
                speaker_curr_count[speaker] += 1.0
            else:
                speaker_dev_files.append(i)
                featurized[i] = False
                speaker_curr_count[speaker] += 1.0
        self.speaker_train_files = speaker_train_files
        self.speaker_dev_files = speaker_dev_files
        self.featurized = featurized
        self.dim = 39

        logging.info("Train Speaker Files: {}".format(self.speaker_train_files))
        logging.info("Dev Speaker Files: {}".format(self.speaker_dev_files))
        logging.info("Input dimensions: {}".format(self.dim))

    def lazy_load(self, i):
        if self.featurized[i]:
            return pickle.load(tf.gfile.FastGFile(os.path.join(self.pkl_dir, '{}.pkl'.format(i))))
        else:
            f = self.speaker_files[i]
            speaker = os.path.basename(os.path.split(f)[0])
            # logging.info("Featurize {}: {}".format(i, f))
            audio = AudioSegment.from_file(gfile.FastGFile(f))
            if self.silence:
                audio = remove_silence(audio)
            sr = float(audio.frame_rate)
            speaker_feature = self.feature_extractor(np.array(audio.get_array_of_samples(), dtype=float), sr=sr)
            if speaker_feature is None:
                return None
            # logging.info("Speaker Feature Shape: {}".format(speaker_feature.shape))
            obj = (speaker_feature, np.repeat(self.speakers[speaker], speaker_feature.shape[0]))
            pickle.dump(obj, tf.gfile.FastGFile(os.path.join(self.pkl_dir, '{}.pkl'.format(i)), 'w'))
            self.featurized[i] = True
            return obj

    @threadsafe_generator
    def generator(self, id='train'):
        if id == 'train':
            speaker_files = self.speaker_train_files
        else:
            speaker_files = self.speaker_dev_files
        while True:
            x = np.array([]).reshape((0, self.frames, self.dim))
            y = np.array([])
            speakers_id = random.sample(speaker_files, 1)[0]
            tup = self.lazy_load(speakers_id)
            if tup is not None:
                x = np.vstack([x, tup[0]])
                y = np.append(y, tup[1])
                yield np.expand_dims(x, -1), to_categorical(y, num_classes=self.num_speakers)
