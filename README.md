# Gathering Sentiment from Audio

Many call center companies have so many recorded calls but no way to efficiently filter through them.  The idea for this model is to gather sentiment from audio.  Right now it's to gather it from sentences, but I would like to train it more going forward to use for call centers to be able to flag any angry customers or clients to reduce churn.

Data came from three separate datasets:

1. [RAVDESS](https://zenodo.org/record/1188976#.XcJFp1FKi00) - This is the Ryerson-Audio Visual Database of Emotional Speech and Song. It contains 1,440 speech files which is compiled from 12 male and 12 female actors expressing eight different emotions: calm, neutral, happy, sad, angry, fearful, surprise, and disgust.

2. [TESS](https://tspace.library.utoronto.ca/handle/1807/24487) - This dataset came from the University of Toronto and contains 2,800 speech files made by two female actresses expressing seven different emotions: neutral, happy, sad, angry, fearful, surprise, and disgust.

3. [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/) - This is the Surrey Audio-Visual Expressed Emotion Database. It contains a total of 477 recordings which consist of four male actors expressing 7 different emotions: neutral, happy, sad, angry, fearful, surprise, and disgust.


## EDA

![emotion_distribution](images/emotion_distribution.png?raw=true "Emotion Distribution")



![angry_wave](images/angry_wave.png?raw=true "Angry") ![disgust_wave](images/disgust_wave.png?raw=true "Disgust") ![fearful_wave](images/fearful_wave.png?raw=true "Fearful") ![happy_wave](images/happy_wave.png?raw=true "Happy") ![neutral_wave](images/neutral_wave.png?raw=true "Neutral") ![sad_wave](images/sad_wave.png?raw=true "Sad") ![surprised_wave](images/surprised_wave.png?raw=true "Surprised")




![angry_mfcc](images/angry_mfcc.png?raw=true "Angry") ![disgust_mfcc](images/disgust_mfcc.png?raw=true "Disgust") ![fearful_mfcc](images/fearful_mfcc.png?raw=true "Fearful") ![happy_mfcc](images/happy_mfcc.png?raw=true "Happy") ![neutral_mfcc](images/neutral_mfcc.png?raw=true "Neutral") ![sad_mfcc](images/sad_mfcc.png?raw=true "Sad") ![surprised_mfcc](images/surprised_mfcc.png?raw=true "Surprised")







## Preprocessing the Data

First I looped through all of the files and pulled loaded them using librosa. I pulled the features chroma_stft, spectral_centroid, spectra_bandwidth, rolloff, zcr, and 20 mfccs. Then I updated the target emotions using a label encoder and scaled my features. 


## Trial and Error





## Model

For the model I chose to use CNN. 


## Results

So far with CNN I got a 65% 

## Future Attempts
I want to try to use PCA on the mfccs and also try to use the information from a spectogram and see if that increases the accuracy of the predictions.
