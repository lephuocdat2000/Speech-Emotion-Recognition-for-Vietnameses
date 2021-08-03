import scipy.io.wavfile
import numpy as np
import sys
import glob 
import librosa
import os
import matplotlib.pyplot as plt
from scipy.stats import zscore
from IPython.display import Audio
### Time Distributed ConvNet imports ###
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelEncoder
from IPython.display import Image
from glob import glob
import pickle
import itertools
from sklearn.model_selection import train_test_split
from PIL import Image
from numpy import load
### Audioimport ###
import IPython

### Warning ###
import warnings

# reduce noise
#import noisereduce as nr
import soundfile as sf
from scipy.io import wavfile
warnings.filterwarnings('ignore')


from tensorflow import keras
model = keras.models.load_model('./models/vie-vie-model.h5')

#noise, sr = librosa.load('../noise.wav', sr=44100)

win_ts = 128
hop_ts = 64

# Split spectrogram into frames
def frame(x, win_step=128, win_size=64):
    nb_frames = 1 + int((x.shape[2] - win_size) / win_step)
    frames = np.zeros((x.shape[0], nb_frames, x.shape[1], win_size)).astype(np.float32)
    for t in range(nb_frames):
        frames[:,t,:,:] = np.copy(x[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float32)
    return frames

def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
    
    # Compute spectogram
    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2
    # Compute mel spectrogram
    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)
    # Compute log-mel spectrogram
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect
    
def predict_emotion_from_file(model, filename, chunk_step=16000, chunk_size=49100, predict_proba=False, sample_rate=16000):
        # Read audio file
        emotion1={0:'Angry',1:'Happy',2:'Sad',3:'Neutral'}
        y, sr = librosa.core.load(filename, sr=sample_rate, offset=0.5)
        # noise_reduced = nr.reduce_noise(audio_clip=y, noise_clip=noise, prop_decrease=1.0, verbose=False)
        # y= np.asarray(noise_reduced, dtype=np.float32)
        win_ts = 128
        hop_ts = 64
        max_pad_len = 49100
        if len(y) < max_pad_len:    
          y_padded = np.zeros(max_pad_len)
          y_padded[:len(y)] = y
          y = y_padded
        elif len(y) > max_pad_len:
          y = np.asarray(y[:max_pad_len])
        # Split audio signals into chunks
        chunks = frame(y.reshape(1, 1, -1), chunk_step, chunk_size)
        # Reshape chunks
        chunks = chunks.reshape(chunks.shape[1],chunks.shape[-1])
        # Z-normalization
        y = np.asarray(list(map(zscore, chunks)))

        # Compute mel spectrogram
        mel_spect = np.asarray(list(map(mel_spectrogram, y)))

        # Time distributed Framing
        mel_spect_ts = frame(mel_spect,hop_ts,win_ts)
        # Build X for time distributed CNN
        X = mel_spect_ts.reshape(mel_spect_ts.shape[0],
                                    mel_spect_ts.shape[1],
                                    mel_spect_ts.shape[2],
                                    mel_spect_ts.shape[3],
                                    1)
        # Predict emotion
        if predict_proba is True:
            predict = model.predict(X)
        else:
            predict = np.argmax(model.predict(X), axis=1)
            # predict = [emotion1.get(emotion) for emotion in predict]
        return predict




# emotion1={0:'Angry',1:'Happy',2:'Sad',3:'Neutral'}
# # file_emo='/content/0_2.wav'
# file_emo = 'F:/HKVI/ND/SER/DEMO/test (1).wav'
# predict=predict_emotion_from_file(model,file_emo)

# print(predict)