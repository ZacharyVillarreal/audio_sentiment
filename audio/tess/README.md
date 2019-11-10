# Gathering Sentiment from Audio

In many call center companies they have so many recorded calls but no way to efficiently filter through them.  The idea for this model is to gather sentiment from audio.  Right now it's to gather it from sentences, but I would like to train it more going forward to use for call centers to be able to flag any angry customers or clients to reduce churn.

The data is from [RAVDESS](https://zenodo.org/record/1188976#.XcJFp1FKi00). It contains a dataset of 1,440 speech files which is compiled from 12 male and 12 female actors expressing each emotion: calm, happy, sad, angry, fearful, surprise, and disgust.


## EDA




## Preprocessing the Data

First I looped through all of the files and pulled loaded them using librosa. I pulled the features chroma_stft, spectral_centroid, spectra_bandwidth, rolloff, zcr, and 20 mfccs. Then I updated the target emotions using a label encoder and scaled my features. 



## Model

For the model I chose to use CNN. 


## Results

So far with CNN I got a 65% 

## Future Attempts
I want to try to use PCA on the mfccs and also try to use the information from a spectogram and see if that increases the accuracy of the predictions.
