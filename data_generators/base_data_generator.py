import logging
import os
import random
from collections import defaultdict

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
        if mfccs.shape[0] < self.frames:
            return None
        features = np.zeros(shape=(mfccs.shape[0] - self.frames, mfccs.shape[1] * self.frames))
        for i in range(mfccs.shape[0] - self.frames):
            features[i] = mfccs[i:i + self.frames].reshape(1, mfccs.shape[1] * self.frames)
        return normalize(features).reshape(-1, self.frames, mfccs.shape[1])

    def __init__(self, dir_path, num_speakers=100, frames=64, silence=False, file_batch_size=16, val_frac=0.2):
        self.frames = frames
        self.num_speakers = num_speakers
        self.file_batch_size = file_batch_size
        self.val_frac = val_frac

        files = list(list_files(dir_path, lambda x: audio_predicate(x) and x.find('TIMIT') != -1))
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

        speaker_features_train, speaker_features_dev, speaker_curr_count = {}, {}, defaultdict(float)
        for i, f in self.speaker_files.items():
            logging.info("Featurize {}: {}".format(i, f))
            speaker = os.path.basename(os.path.split(f)[0])
            audio = AudioSegment.from_file(gfile.FastGFile(f))
            if silence:
                audio = remove_silence(audio, silence_threshold=-60.0)
            sr = float(audio.frame_rate)
            speaker_feature = self.feature_extractor(np.array(audio.get_array_of_samples(), dtype=float), sr=sr)
            logging.info("Speaker Feature Shape: {}".format(speaker_feature.shape))
            if speaker_feature is None:
                continue
            if speaker_curr_count[speaker] < (1.0 - self.val_frac) * speaker_files_count[speaker]:
                speaker_features_train[i] = (speaker_feature, np.repeat(self.speakers[speaker], speaker_feature.shape[0]))
                speaker_curr_count[speaker] += 1.0
            else:
                speaker_features_dev[i] = (speaker_feature, np.repeat(self.speakers[speaker], speaker_feature.shape[0]))
                speaker_curr_count[speaker] += 1.0
        self.speaker_features_train = speaker_features_train
        self.speaker_features_dev = speaker_features_dev
        self.dim = self.speaker_features_train.values()[0][0].shape[2]

        logging.info("Train Speaker Files: {}".format(self.speaker_features_train.keys()))
        logging.info("Dev Speaker: {}".format(self.speaker_features_dev.keys()))
        logging.info("Input dimensions: {}".format(self.dim))

    @threadsafe_generator
    def generator(self, id='train'):
        if id == 'train':
            tgt_features = self.speaker_features_train
            speaker_files = self.speaker_features_train.keys()
        else:
            tgt_features = self.speaker_features_dev
            speaker_files = self.speaker_features_dev.keys()
        while True:
            x = np.array([]).reshape((0, self.frames, self.dim))
            y = np.array([])
            speakers_ids = random.sample(speaker_files, self.file_batch_size)
            for i in speakers_ids:
                x = np.vstack([x, tgt_features[i][0]])
                y = np.append(y, tgt_features[i][1])
            yield np.expand_dims(x, -1), to_categorical(y, num_classes=self.num_speakers)
