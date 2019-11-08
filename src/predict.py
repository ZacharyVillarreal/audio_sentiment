import pickle

import pandas as pd
import numpy as np
import os

import librosa

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling1D, Flatten, LSTM, Activation, Dropout, Dense

def predictions(filename, model_name):
    ## UNPICKLE
    model = pickle.load(open('pickles/' + model_name + '_model.p', 'rb'))
    encoder = pickle.load(open('pickles/' + model_name + '_encoder.p', 'rb'))
    scaler = pickle.load(open('pickles/' + model_name + '_scaler.p', 'rb'))

    from src.feature_extraction import feature_extraction_transform
    df = feature_extraction_transform([filename])

    X = scaler.transform(np.array(df.drop(columns = 'filename'), dtype = float))
    X = X.reshape(X.shape[0], X.shape[1],1)

    preds = model.predict(X, 
                         batch_size=5, 
                         verbose=1)

    preds1=preds.argmax(axis=1)
    predictions = (encoder.inverse_transform((preds1)))[0]
    predictions = {'name':filename[11:-4].split('_')[0], 'actual': filename[11:-4].split('_')[1], 'predicted' : predictions}

    return predictions


def accuracy(df):
    accuracy = df[df['correct'] == True].count()[1]/df.groupby('correct', as_index=False).count().sum()[1]
    return 'Accuracy = ' + str(round(accuracy*100,2)) + '%'



def pred_df(model): 
    paths = []
    for (dirpath, dirnames, filenames) in os.walk('live_audio'):
        for filename in filenames:
            if filename.endswith('.wav'): 
                paths.append(os.sep.join([dirpath, filename]))
            
            
    df = pd.DataFrame(columns=['name', 'actual', 'predicted'])
    for filename in paths:
        preds = predictions(filename, model)
        df = df.append(preds, ignore_index=True)
        df['correct'] = np.where(df['predicted'] == df['actual'],True, False)

    print(accuracy(df))
    
    return df


def pred_df_mf(model, people_dict): 
    paths = []
    for (dirpath, dirnames, filenames) in os.walk('live_audio'):
        for filename in filenames:
            if filename.endswith('.wav'): 
                paths.append(os.sep.join([dirpath, filename]))
            
            
    df = pd.DataFrame(columns=['name', 'actual', 'predicted'])
    for filename in paths:
        preds = predictions(filename, model)
        df = df.append(preds, ignore_index=True)

    df['gender'] = df['name'].apply(lambda x: people_dict[x])
    df['actual'] = df['gender'] + '_' + df['actual']
    df['correct'] = np.where(df['predicted'] == df['actual'],True, False)

    print(accuracy(df))
    
    return df


