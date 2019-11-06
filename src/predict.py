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

def predictions(filename):
    ## UNPICKLE
    model = pickle.load(open('model.p','rb'))
    encoder = pickle.load(open('encoder.p', 'rb'))

    from src.feature_extraction import feature_extraction
    df = feature_extraction([filename])

    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(df.drop(columns = 'filename'), dtype = float))
    X = X.reshape(X.shape[0], X.shape[1],1)

    preds = model.predict(X, 
                         batch_size=32, 
                         verbose=1)

    preds1=preds.argmax(axis=1)
    predictions = (encoder.inverse_transform((preds1)))
    preddf = pd.DataFrame({'predictedvalues': predictions})

    return 'Sentiment is ' + preddf['predictedvalues'][0]