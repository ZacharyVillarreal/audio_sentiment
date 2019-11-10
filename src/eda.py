import librosa
import librosa.display
from python_speech_features import mfcc, logfbank

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os



def clean_radvess(df):
    df['filename'] = df['filename'].str.split('/')
    df['filename'] = df['filename'].apply(lambda x: x[2])
    df['emotion'] = df['filename'].apply(lambda x: x[6:-16])
    emotional_dict = {'01' : 'neutral', '02' : 'calm', '03' : 'happy', '04' : 'sad', '05' : 'angry', '06' : 'fearful', '07' : 'disgust', '08' : 'surprised'}
    df['emotion'] = df['emotion'].apply(lambda x: emotional_dict[x])
    return df


def clean_tess(df):
    emotions = ['neutral', 'fear', 'happy', 'angry', 'sad', 'ps', 'disgust']
    def emotion(x):
        for i in x[:-4].split('_'):
            if i in emotions:
                return i

    df['emotion'] = df['filename'].apply(lambda x: emotion(x))
    return df


def clean_savee(df):
    emotions = {'a' : 'angry', 'd' : 'disgust', 'f' : 'fearful', 'h' : 'happy', 'n' : 'neutral', 'sa' : 'sad', 'su' : 'surprised'}
    def emotion(x):
        for i in x[:-6].split('/'):
            if i in emotions:
                return emotions[i]

    df['emotion'] = df['filename'].apply(lambda x: emotion(x))
    return df


def uniform_emotion(x):
    emotion_dict = {'fear' : 'fearful', 'ps' : 'surprised', 'calm' : 'neutral'}
    if x in emotion_dict:
        return emotion_dict[x]
    else: 
        return x


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
        signal, rate = librosa.load('audio/'+wav_file, mono=True, duration=3, offset = .5) #wav file can detect sampling rate
        signals[i] = signal
        fft[i] = calc_fft(signal, rate)

        bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T #nfft=sampling_frequency/40, nfilt always 26
        fbank[i] = bank
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T #numcep = usually 1/2 of nfilt
        mfccs[i] = mel
    return signals, fft, fbank, mfccs