import pandas as pd
import numpy as np
import os
import librosa

def mfcc_sep(x):
    lst = []
    for i in x:
        lst.append(np.mean(i))
    return lst


def feature_extraction(path):
    ## get file names
    # paths = []
    # for (dirpath, dirnames, filenames) in os.walk('audio'):
    #     for filename in filenames:
    #         if filename.endswith('.wav'): 
    #             paths.append(os.sep.join([dirpath, filename]))


    header = ['filename', 'chroma_stft', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate','mfcc']

    df = pd.DataFrame(columns = header)

    ## pull features
    for filename in path:
        lst = []
        file = filename
        y, sr = librosa.load(file, mono=True, duration=3)
        chroma_stft = librosa.feature.chroma_stft(y, sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
    #     mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13),axis=0)
        
        to_append = {'filename':filename, 'chroma_stft': np.mean(chroma_stft), 'spectral_centroid': np.mean(spec_cent), 
                'spectral_bandwidth':np.mean(spec_bw), 'rolloff':np.mean(rolloff), 'zero_crossing_rate':np.mean(zcr), 'mfcc':mfcc}
        df = df.append(to_append, ignore_index=True)

    df['mfcc'] = df['mfcc'].apply(lambda x: mfcc_sep(x))
    mfcc_list = ['mfcc'+str(i) for i in range(20)]
    df2 = pd.DataFrame(df['mfcc'].values.tolist(), columns = mfcc_list)
    df3 = pd.concat([df.drop(columns='mfcc'),df2], axis=1)

    return df3
