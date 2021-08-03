# Try for one file first
import librosa
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
from tqdm import tqdm
import pickle
from sklearn.preprocessing import MinMaxScaler
import librosa.display
import pandas as pd

ms.use('seaborn-muted')


def test(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    features_y = extract_features(y)

    df1 = pd.DataFrame(np.array(features_y).reshape(1, 8),
                       columns=['sig_mean', 'sig_std', 'rmse_mean', 'rmse_std', 'silence', 'harmonic', 'auto_corr_max',
                                'auto_corr_std'], dtype=float)
    df_pred = normalize_features(df1)

    with open('./SER/rf_classifier.pkl', 'rb') as file:
        model = pickle.load(file)

    pred = model.predict(df_pred)
    return pred


def extract_features(y):
    features_list_pred = list()
    features_list_pred.append(np.mean(abs(y)))
    features_list_pred.append(np.std(y))

    rmse = librosa.feature.rms(y + 0.0001)[0]
    features_list_pred.append(np.mean(rmse))
    features_list_pred.append(np.std(rmse))

    silence = 0
    for e in rmse:
        if e <= 0.4 * np.mean(rmse):
            silence += 1
    features_list_pred.append(silence / float(len(rmse)))

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    # print(np.mean(y_harmonic))
    features_list_pred.append(np.mean(y_harmonic) * 1000)

    cl = 0.45 * np.mean(abs(y))
    center_clipped = []
    for s in y:
        if s >= cl:
            center_clipped.append(s - cl)
        elif s <= -cl:
            center_clipped.append(s + cl)
        elif np.abs(s) < cl:
            center_clipped.append(0)

    new_autocorr = librosa.core.autocorrelate(np.array(center_clipped))
    features_list_pred.append(1000 * np.max(new_autocorr) / len(new_autocorr))
    features_list_pred.append(np.std(new_autocorr))

    return features_list_pred


def normalize_features(df1):
    df = pd.read_csv('./SER/modified_df.csv')
    df = df.append(df1)

    scalar = MinMaxScaler()
    df[df.columns[2:]] = scalar.fit_transform(df[df.columns[2:]])

    df_pred = pd.DataFrame(np.array(df.iloc[-1][3:]).reshape(1, 8),
                           columns=['sig_mean', 'sig_std', 'rmse_mean', 'rmse_std', 'silence', 'harmonic',
                                    'auto_corr_max', 'auto_corr_std'], dtype=float)

    return df_pred
