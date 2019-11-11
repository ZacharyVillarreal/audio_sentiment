import librosa
import librosa.display
from python_speech_features import mfcc, logfbank

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os


def data_source(x):
    if x.split('/')[1] == 'tess':
        return 'tess'
    elif x.split('/')[1] == 'savee':
        return 'savee'
    else:
        return 'ravdess'
    

def eda_data(df):
    df = pd.read_csv('data/data.csv')
    df = df[['filename', 'emotion']]
    df['emotion'] = df['emotion'].str.split('_')
    df['gender'] = df['emotion'].apply(lambda x : x[0])
    df['emotion'] = df['emotion'].apply(lambda x: x[1])
    df['type'] = df['filename'].apply(lambda x:data_source(x))
    
    return df[['filename', 'emotion', 'gender', 'type']]


def distributions(df, col, title):
    count = df.groupby(col).count()['filename']
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(count,labels= count.index.str.capitalize(),  autopct='%1.1f%%', textprops={'fontsize': 13})
    ax.set_title('Distribution of '+title, fontsize=15)


def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate) 
    Y = abs(np.fft.rfft(y)/n) #magnitude -- also /n so that it normalizes/scales the signal
    return (Y, freq)


def frequencies(df):
    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}
    emotion = 'happy sad angry neutral surprised fearful disgust'
    for i in emotion.split(' '):
        wav_file = df[df['emotion'] == i].iloc[0,0]
        signal, rate = librosa.load(wav_file, mono=True, duration=3, offset = .5) #wav file can detect sampling rate
        signals[i] = signal
        fft[i] = calc_fft(signal, rate)

        bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T #nfft=sampling_frequency/40, nfilt always 26
        fbank[i] = bank
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T #numcep = usually 1/2 of nfilt
        mfccs[i] = mel
    return signals, fft, fbank, mfccs