import os
import threading

import librosa
import numpy as np
import python_speech_features
from keras import backend as K
from keras.engine.topology import Layer
from pydub import AudioSegment
from tensorflow.python.platform import gfile


class ClippedRelu(Layer):
    def __init__(self, clip_value, **kwargs):
        self.clip_value = clip_value
        super(ClippedRelu, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return K.minimum(K.maximum(x, 0), self.clip_value)


def normalize(x):
    return (x - np.mean(x)) / np.std(x)


def get_mel_spectrogram(y, sr, tgt_sr=16000, win_len=0.025,
                        hop_len=0.010, n_fft=512, n_mels=128, fmin=0.0, fmax=8000.0, log_mel=True):
    if sr != 16000.0:
        y = librosa.core.resample(y, orig_sr=sr, target_sr=16000)
    spectrogram, phase = librosa.magphase(
        librosa.stft(y, n_fft=n_fft, hop_length=int(hop_len * tgt_sr), win_length=int(win_len * tgt_sr)))
    mel_spectrogram = librosa.feature.melspectrogram(S=spectrogram, n_mels=n_mels, fmin=fmin, fmax=fmax)
    if log_mel:
        return librosa.power_to_db(mel_spectrogram).T
    else:
        return mel_spectrogram.T


def get_mfcc_v1(y, sr, n_mfcc=13, tgt_sr=16000, win_len=0.025,
                hop_len=0.010, n_fft=512, n_mels=22, fmin=0.0, fmax=None, delta=False, delta_delta=False):
    if sr != 16000.0:
        y = librosa.core.resample(y, orig_sr=sr, target_sr=16000)
    spectrogram, phase = librosa.magphase(
        librosa.stft(y, n_fft=n_fft, hop_length=int(hop_len * tgt_sr), win_length=int(win_len * tgt_sr)))
    mel_spectrogram = librosa.feature.melspectrogram(S=spectrogram, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), n_mfcc=n_mfcc)
    features = [mfccs]
    if delta:
        features.append(librosa.feature.delta(mfccs, order=1))
    if delta_delta:
        features.append(librosa.feature.delta(mfccs, order=2))
    return np.vstack(features).T


def get_mfcc_v2(y, sr, n_mfcc=13, tgt_sr=16000, win_len=0.025, hop_len=0.010, n_fft=512, n_mels=22,
                fmin=0.0, fmax=None, cep_lifter=22, pre_emph=0.97, win_func=lambda x: np.ones((x,)),
                append_energy=True, delta=True, delta_delta=True):
    if sr != 16000.0:
        y = librosa.core.resample(y, orig_sr=sr, target_sr=16000)
    mfccs = python_speech_features.mfcc(y, tgt_sr, winlen=win_len, winstep=hop_len, numcep=n_mfcc, nfilt=n_mels,
                                        nfft=n_fft, lowfreq=fmin, highfreq=fmax, preemph=pre_emph, ceplifter=cep_lifter,
                                        appendEnergy=append_energy, winfunc=win_func)
    features = [mfccs]
    if delta:
        features.append(python_speech_features.delta(mfccs, 1))
    if delta_delta:
        features.append(python_speech_features.delta(mfccs, 2))
    return np.hstack(features)


def get_energy(y, sr, tgt_sr=16000, win_len=0.025, hop_len=0.010, n_fft=512, delta=False, delta_delta=False):
    if sr != 16000.0:
        y = librosa.core.resample(y, orig_sr=sr, target_sr=16000)
    spectrogram, phase = librosa.magphase(
        librosa.stft(y, n_fft=n_fft, hop_length=int(hop_len * tgt_sr), win_length=int(win_len * tgt_sr)))
    energy = librosa.feature.rmse(S=spectrogram)
    features = [energy]
    if delta:
        features.append(librosa.feature.delta(energy, order=1))
    if delta_delta:
        features.append(librosa.feature.delta(energy, order=2))
    return np.vstack(features).T


def list_files(base_path, predicate):
    for folder, subs, files in gfile.Walk(base_path):
        for filename in files:
            if predicate(os.path.join(folder, filename)):
                yield (os.path.join(folder, filename))


def audio_predicate(fname):
    return fname.lower().endswith(".wav") or fname.lower().endswith(".mp3") or fname.lower().endswith(".aac")


def remove_silence(sound, silence_threshold=-50.0, chunk_size=10):
    clip = AudioSegment.empty()
    cur_start = 0
    trim_ms = 0
    while trim_ms + chunk_size < len(sound):
        if sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold:
            cur_end = trim_ms
            if cur_end != cur_start:
                clip += sound[cur_start:cur_end]
            trim_ms += chunk_size
            cur_start = trim_ms
        else:
            trim_ms += chunk_size
    if sound[cur_start:].dBFS > silence_threshold:
        clip += sound[cur_start:]
    return clip


class ThreadSafeIter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))

    return g
