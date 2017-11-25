import numpy as np
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
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
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import keras
import h5py
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
from scipy.io import wavfile
import matplotlib.pyplot as plt
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
#below dependencies required for mel-spectrogram
import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import pylab
import librosa
import librosa.display
import numpy as np




class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
print("Starting ML Training")



print("Finished import statements")

'''
pickle_filepath_X = "/Users/sreeharirammohan/Desktop/all_data/allMelNumpyImages.npy"
pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/all_data/allMelNumpyLabels.npy"


X = np.load(pickle_filepath_X)
Y = np.load(pickle_filepath_Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


print(Y_train.shape)
print(Y_test.shape)

Y_train.reshape(2592, 2)
Y_test.reshape(648, 2)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
'''

def create_model(weights_path=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(3, 640, 480)))
    print("fail")
    model.add(Conv2D(64, (3, 3), activation='relu', dim_ordering="th"))
    print("fail2")

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model


print("created model")

def individualWavToMelSpectrogram(myAudioPath):
    sig, fs = librosa.load(myAudioPath)   
    # make pictures name 
    save_path = "/Users/sreeharirammohan/Desktop/temp_data/spectrogram.jpg"

    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()


def individualWavToSpectrogram(myAudio):
    print(myAudio)
    #Read file and get sampling freq [ usually 44100 Hz ]  and sound object
    samplingFreq, mySound = wavfile.read(myAudio)

    #Check if wave file is 16bit or 32 bit. 24bit is not supported
    mySoundDataType = mySound.dtype

    #We can convert our sound array to floating point values ranging from -1 to 1 as follows

    mySound = mySound / (2.**15)

    #Check sample points and sound channel for duel channel(5060, 2) or  (5060, ) for mono channel

    mySoundShape = mySound.shape
    samplePoints = float(mySound.shape[0])

    #Get duration of sound file
    signalDuration =  mySound.shape[0] / samplingFreq

    #If two channels, then select only one channel
    #mySoundOneChannel = mySound[:,0]

    #if one channel then index like a 1d array, if 2 channel index into 2 dimensional array
    if len(mySound.shape) > 1:
        mySoundOneChannel = mySound[:,0]
    else:
        mySoundOneChannel = mySound

    #Plotting the tone

    # We can represent sound by plotting the pressure values against time axis.
    #Create an array of sample point in one dimension
    timeArray = numpy.arange(0, samplePoints, 1)

    #
    timeArray = timeArray / samplingFreq

    #Scale to milliSeconds
    timeArray = timeArray * 1000

    plt.rcParams['agg.path.chunksize'] = 100000

    
    #Plot the tone
    plt.xlim((0, 8000))
    plt.figure(figsize=(0.875,0.375))
    plt.plot(timeArray, mySoundOneChannel, color='Black')
    #plt.xlabel('Time (ms)')
    #plt.ylabel('Amplitude')
    print("trying to save")
    plt.savefig('/Users/sreeharirammohan/Desktop/temp_data/' + 'spectrogram' + '.jpg')
    print("saved")
    print("not going to show now")
    #plt.show()

    #very important to prevent memory leaks!
    plt.close()


def convert_spectrogram_to_numpy(path_to_spectrogram):
    img = io.imread(path_to_spectrogram)
    return img
    

model = create_model('/Users/sreeharirammohan/Desktop/all_data/85model.h5')
print("Created model")


print("Here")
'''
root_dir = '/Users/sreeharirammohan/Desktop/heart_sounds/all-abnormal-wav'
all_img_paths = glob.glob(os.path.join(root_dir, '*.wav'))
for i, img_path in enumerate(all_img_paths):
    path_to_sound = img_path
    individualWavToMelSpectrogram(path_to_sound)
    numpy_image_from_spectrogram = np.array(convert_spectrogram_to_numpy('/Users/sreeharirammohan/Desktop/temp_data/spectrogram.jpg'))
    #numpy_image_from_spectrogram = numpy_image_from_spectrogram.swapaxes(0, 2)
    print("**********************************")
    print(numpy_image_from_spectrogram.shape)
    numpy_image_from_spectrogram = np.swapaxes(numpy_image_from_spectrogram, 2, 0)
    numpy_image_from_spectrogram = numpy_image_from_spectrogram[None, ...]
    prediction = model.predict_classes(numpy_image_from_spectrogram)
    if(prediction[0] == 0):
        print("Heartbeat NORMAL")
    else:
        print("Heartbeat ABNORMAL")
print("skipped")
'''

while True:
    path_to_sound = input("Enter the directory of a heart sound for me to analyze")
    individualWavToMelSpectrogram(path_to_sound)
    numpy_image_from_spectrogram = np.array(convert_spectrogram_to_numpy('/Users/sreeharirammohan/Desktop/temp_data/spectrogram.jpg'))
    #numpy_image_from_spectrogram = numpy_image_from_spectrogram.swapaxes(0, 2)
    numpy_image_from_spectrogram = np.swapaxes(numpy_image_from_spectrogram, 2, 0)
    numpy_image_from_spectrogram = numpy_image_from_spectrogram[None, ...]

    prediction = model.predict_classes(numpy_image_from_spectrogram)
    if(prediction[0] == 0):
        print("Heartbeat NORMAL")
    else:
        print("Heartbeat ABNORMAL")
       




