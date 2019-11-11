# Gathering Sentiment from Audio

Many call center companies have recorded calls but no way to efficiently filter through them. The goal for this model is to gather sentiment from audio files in order to allow businesses to easily flag any angry calls. This could potentially reduce churn by calling attention to unhappy customers. It could also be used as a training tool to see if employees are effectively handling these calls.

In my specific case, I want to relate this to the home care industry.  The employee turnover rate is at an all time high of 82% in 2018. I want to use this model to flag any angry calls from both customers and caregivers in the field in an effort to reduce employee and customer churn.


### Data

The data used for this model came from three separate datasets:

1. [RAVDESS](https://zenodo.org/record/1188976#.XcJFp1FKi00) - This is the Ryerson-Audio Visual Database of Emotional Speech and Song. It contains 1,440 speech files which is compiled from 12 male and 12 female actors expressing eight different emotions: calm, neutral, happy, sad, angry, fearful, surprise, and disgust.

2. [TESS](https://tspace.library.utoronto.ca/handle/1807/24487) - This dataset came from the University of Toronto and contains 2,800 speech files made by two female actresses expressing seven different emotions: neutral, happy, sad, angry, fearful, surprise, and disgust.

3. [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/) - This is the Surrey Audio-Visual Expressed Emotion Database. It contains a total of 477 recordings which consist of four male actors expressing 7 different emotions: neutral, happy, sad, angry, fearful, surprise, and disgust.


## EDA

Since neutral and calm are very similar, I combined them together to show just neutral. There is an even distribution in emotions.<br/>


![emotion_distribution](images/emotion_distribution.png?raw=true "Emotion Distribution")
![gender_distribution](images/gender_distribution.png?raw=true "Gender Distribution")


Below are the waveplots for the combined files for each emotion.  The waveplots show the amplitude over time.
![angry_wave](images/angry_wave.png?raw=true "Angry") ![disgust_wave](images/disgust_wave.png?raw=true "Disgust") ![fearful_wave](images/fearful_wave.png?raw=true "Fearful") ![happy_wave](images/happy_wave.png?raw=true "Happy") ![neutral_wave](images/neutral_wave.png?raw=true "Neutral") ![sad_wave](images/sad_wave.png?raw=true "Sad") ![surprised_wave](images/surprised_wave.png?raw=true "Surprised")



Mel Frequency Cepstral Coefficient (MFCC) are the features that I extracted from the audio clip. It scales the frequencies to make the features match more closely to what humans hear.<br/>
![angry_mfcc](images/angry_mfcc.png?raw=true "Angry") ![disgust_mfcc](images/disgust_mfcc.png?raw=true "Disgust") ![fearful_mfcc](images/fearful_mfcc.png?raw=true "Fearful") ![happy_mfcc](images/happy_mfcc.png?raw=true "Happy") ![neutral_mfcc](images/neutral_mfcc.png?raw=true "Neutral") ![sad_mfcc](images/sad_mfcc.png?raw=true "Sad") ![surprised_mfcc](images/surprised_mfcc.png?raw=true "Surprised")





## Preprocessing the Data 

First I looped through all of the files and loaded them using librosa. I pulled the features chroma_stft, spectral_centroid, spectra_bandwidth, rolloff, zcr, and 20 MFCCs. Then I used a one hot encoder for the target emotions and scaled my features. 

I preprocessed my data multiple ways in order to find the highest accuracy. 
  1. Use all the data as is.
  2. Decrease the amount of emotions that were predicted.
      - The first way was to group together negative, positive, and neutral feelings. 
      - The second way was to drop different emotions.
  3. Split up the data between male and female with all emotions and then with both types of predicted emotions decreased.
  4. Use PCA to choose my MFCC features.


## Model

The base model that I used had all features using a CNN. I was having issues with overfitting to the training data so I added in more dropouts. I used four layers in my network and calculated loss using cross entropy.

For the final model I chose to split the data between men and women, and also drop the emotion of disgust. With this model I got about 81% validation accuracy using a CNN. When I input live data I got 45% accuracy on the new data. 

## Future Improvements
Future improvements for this project would be to add in more live data to help train the model. I would like to input call logs to try to interpret emotion after that. I will have to find a way to work around that there will be two different voices in the call logs and that the calls will be longer than the data I have in the training set.


<br/>
<br/>
<br/>

 Click [here](https://docs.google.com/presentation/d/1QjZdP__8h-kl2TJddOjDnTg-1VRrJiPaL3StbDlvLdk/edit?usp=sharing) for access to the powerpoint slides.
