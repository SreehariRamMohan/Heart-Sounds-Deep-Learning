#This script will convert a directory of .wav files to .npy data which can be used by a traditional machine learning model.
import matplotlib
matplotlib.use('TkAgg')
import librosa.display
import sklearn, librosa
import os
import glob
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def MFCC_from_wav(path_of_wav):
    #loading audio file
    x, fs = librosa.load(path_of_wav)
    mfccs = librosa.feature.mfcc(x, sr=fs)
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    return mfccs

def get_class(wav_path):
    global one_count
    global zero_count
    if "abnormal" in wav_path:
        #print("returning 1!")
        one_count += 1
        return 1
    else:
        #print("returning 0!")
        zero_count += 1
        return 0

root_dir = '/Users/sreeharirammohan/Desktop/heart_sounds'
MFCCs = []
labels = []
one_count = 0
zero_count = 0
progress = 0

pickle_filepath_X = "/Users/sreeharirammohan/Desktop/MFCCs_Data.npy"
pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/MFCC_Labels.npy"

all_img_paths = glob.glob('/Users/sreeharirammohan/Desktop/heart_sounds/**/*.wav', recursive=True) #glob.glob(os.path.join(root_dir, '*.*'))
print(all_img_paths)
print(str(len(all_img_paths)))
np.random.shuffle(all_img_paths)

if not os.path.exists(pickle_filepath_X) or not os.path.exists(pickle_filepath_Y):
    for i, wav_path in enumerate(all_img_paths):

        if(progress % 10 == 0):
            print(str(100*(progress/3240)) + str("% done"))

        #get features and label for this wav file
        individual_MFCC = MFCC_from_wav(wav_path)
        label = get_class(wav_path)

        #add features and label to the array
        MFCCs.append(individual_MFCC)
        labels.append(label)
        progress += 1

    print("Data below")

    #Format to numpy array, and create one-hot targets to encode labels
    X = np.array(MFCCs, dtype='float32')

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    Y = onehot_encoder.fit_transform(labels)
    print(Y)

    print("Saving")
    np.save(pickle_filepath_X, X)
    np.save(pickle_filepath_Y, Y)
    print("saved both numpy arrays")


else:
    print("We already created the MFCCs data, so we can retrieve it")
    X = np.load(pickle_filepath_X)
    Y = np.load(pickle_filepath_Y)
    print("Retrieved the data")




