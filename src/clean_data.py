def clean_file(df):
    ## split file name
    df['filename'] = df['filename'].str.split('/')
    ## take out 'audio'
    df['filename'] = df['filename'].apply(lambda x: x[1:])
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