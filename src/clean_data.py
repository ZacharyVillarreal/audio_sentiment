
import pandas as pd
import os


def clean_ravdess(df):
    df_ravdess['emotion'] = df_ravdess['filename'].apply(lambda x: x[29:-16])
    emotional_dict = {'01' : 'neutral', '02' : 'calm', '03' : 'happy', '04' : 'sad', '05' : 'angry', '06' : 'fearful', '07' : 'disgust', '08' : 'surprised'}
    df_ravdess['emotion'] = df_ravdess['emotion'].apply(lambda x: emotional_dict[x])
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
    else: return x


def gender(x):
    if int(x[41:-4]) % 2 == 0:
        return 'female'
    else:
        return 'male'

from src.feature_extraction import feature_extraction

## feature extraction
df_ravdess = feature_extraction('audio/ravdess')
df_tess = feature_extraction('audio/tess')
df_savee = feature_extraction('audio/savee')

## save as csv
df_ravdess.to_csv('data/ravdess.csv', index=False)
df_tess.to_csv('data/tess.csv', index=False)
df_savee.to_csv('data/savee.csv', index=False)

## read csv
# df_tess = pd.read_csv('data/tess.csv')
# df_ravdess = pd.read_csv('data/ravdess.csv')
# df_savee = pd.read_csv('data/savee.csv')


## add gender columns
df_tess['gender'] = 'female'
df_savee['gender'] = 'male'
df_ravdess['gender'] = df_ravdess['filename'].apply(lambda x: gender(x))
df_ravdess.to_csv('data/ravdess_gender.csv', index=False)

## update emotions and clean df
df_tess = clean_tess(df_tess)
df_ravdess = clean_ravdess(df_ravdess)
df_savee = clean_savee(df_savee)
df_ravdess.to_csv('data/ravdess_clean.csv', index=False)

## concat all dfs
df = pd.concat([df_tess, df_ravdess, df_savee], axis=0)

## update emotions so that it's all the same
df['emotion'] = df['emotion'].apply(lambda x: uniform_emotion(x))
df = df[df['emotion'].notnull()]


## remove disgust
# df = df[df['emotion'] != 'disgust']

## add new column for gender_emotion
df['emotion'] = df['gender'] + '_' + df['emotion']
df = df.drop(columns='gender')


## output to csv
df.to_csv('data/data.csv', index=False)
