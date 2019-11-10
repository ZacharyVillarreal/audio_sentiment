def clean_radvess(df):
    df['filename'] = df['filename'].str.split('/')
    df['filename'] = df['filename'].apply(lambda x: x[2])
    df['emotion'] = df['filename'].apply(lambda x: x[6:-16])
    emotional_dict = {'01' : 'neutral', '02' : 'calm', '03' : 'happy', '04' : 'sad', '05' : 'angry', '06' : 'fearful', '07' : 'disgust', '08' : 'surprised'}
    df['emotion'] = df['emotion'].apply(lambda x: emotional_dict[x])
    return df.drop(columns = ['filename'])


def clean_tess(df):
    emotions = ['neutral', 'fear', 'happy', 'angry', 'sad', 'ps', 'disgust']
    def emotion(x):
        for i in x[:-4].split('_'):
            if i in emotions:
                return i

    df['emotion'] = df['filename'].apply(lambda x: emotion(x))
    return df.drop(columns = 'filename')



def clean_savee(df):
    emotions = {'a' : 'angry', 'd' : 'disgust', 'f' : 'fearful', 'h' : 'happy', 'n' : 'neutral', 'sa' : 'sad', 'su' : 'surprised'}
    def emotion(x):
        for i in x[:-6].split('/'):
            if i in emotions:
                return emotions[i]

    df['emotion'] = df['filename'].apply(lambda x: emotion(x))
    return df.drop(columns = 'filename')