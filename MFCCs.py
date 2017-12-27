import matplotlib
matplotlib.use('TkAgg')
import librosa.display
import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib
import os
import numpy
import re
from skimage import io
import glob
import numpy as np
import _pickle as cPickle
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split

#loading the audio file
x, fs = librosa.load('/Users/sreeharirammohan/Desktop/my_heartbeat.wav')

'''
plotting the general wavform audio
'''
#librosa.display.waveplot(x, sr=fs)
#plt.show()


'''
getting the MFCCs
'''
mfccs = librosa.feature.mfcc(x, sr=fs)
print(mfccs.shape) #(20, 1292)
#so the MFCC is 20 rows by 1292 rows

'''
displaying the MFCCs 
'''
#librosa.display.specshow(mfccs, sr=fs, x_axis='time')
#plt.show()


'''
MFFCCs scaled so that each coefficient dimension has zero mean and unit variance)
'''
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)

print(mfccs)
print(mfccs.mean(axis=1))
print(mfccs.var(axis=1))
librosa.display.specshow(mfccs, sr=fs, x_axis='time')
plt.show()