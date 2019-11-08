def clean_radvess(df):
    ## split file name
    df['filename'] = df['filename'].str.split('/')
    ## take out 'audio'
    df['filename'] = df['filename'].apply(lambda x: x[2:])
    ## take out actor
    df['wav'] = df['filename'].apply(lambda x: x[1])
    ## take out '.wav'
    df['wav'] = df['wav'].apply(lambda x: x[:-4])
    ## decode file
    decode_file = ['modality', 'vocal_channel', 'emotion', 'emotional_intensity', 'statement', 'repetition', 'actor']
    df['wav'] = df['wav'].str.split('-')
    ## create new columns
    for idx, name in enumerate(decode_file):
        df[name] = df['wav'].apply(lambda x : x[idx]).astype(int)
    ## drop 'wav' and 'filename'

    emotional_dict = {1 : 'neutral', 2 : 'calm', 3 : 'happy', 4 : 'sad', 5 : 'angry', 6 : 'fearful', 7 : 'disgust', 8 : 'surprised'}
    df['emotion'] = df['emotion'].apply(lambda x: emotional_dict[x])
    return df.drop(columns = ['wav', 'filename', 'modality', 'vocal_channel', 'emotional_intensity', 'statement', 'repetition', 'actor'])



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