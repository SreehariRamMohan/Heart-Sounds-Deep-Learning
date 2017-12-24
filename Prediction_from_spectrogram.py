from __future__ import print_function
import numpy as np
np.random.seed(123)
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import MaxPooling2D
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import io
import numpy as np
np.random.seed(123)  # for reproducibility
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import MaxPooling2D
from skimage import io
import glob
import matplotlib
matplotlib.use('Agg') # No pictures displayed
import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed
import numpy as np

def convert_spectrogram_to_numpy(path_to_spectrogram):
    img = io.imread(path_to_spectrogram)
    return img

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

model = create_model('/Users/sreeharirammohan/Desktop/check_point_models/weights-best-031-0.88735.hdf5')
print("Created model")


def make_prediction_from_path(path):
    numpy_image_from_spectrogram = np.array(convert_spectrogram_to_numpy(path))
    print("Converted heartbeat data to numpy arr")

    numpy_image_from_spectrogram = np.swapaxes(numpy_image_from_spectrogram, 2, 0)
    numpy_image_from_spectrogram = numpy_image_from_spectrogram[None, ...]
    prediction = model.predict_classes(numpy_image_from_spectrogram)
    if (prediction[0] == 0):
        print("Heartbeat NORMAL")
    else:
        print("Heartbeat ABNORMAL")


'''
root_dir = '/Users/sreeharirammohan/Desktop/all_data/mel-all-abnormal'
all_img_paths = glob.glob(os.path.join(root_dir, '*.*'))

zeroCount = 0
oneCount = 0

for i, img_path in enumerate(all_img_paths):
    img = img[None, ...]
    prediction = model.predict_classes(img)
    if (prediction[0] == 0):
        print("Heartbeat NORMAL")
        zeroCount += 1
    else:
        print("Heartbeat ABNORMAL")
'''



print("Finished import statements")
pickle_filepath_X = "/Users/sreeharirammohan/Desktop/all_data/allMelNumpyImages.npy"
pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/all_data/allMelNumpyLabels.npy"

X = np.load(pickle_filepath_X)
Y = np.load(pickle_filepath_Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#X_train = numpy.swapaxes(X_train, 2, 3)
#X_test = numpy.swapaxes(X_test, 2, 3)
print("X_train shape: ")
print(X_train.shape)
print("-----")
print("X_test shape")
print(X_test.shape)
print("-----")
print("Y_train.shape")
print(Y_train.shape)
print("-----")
print("Y_test shape")
print(Y_test.shape)

Y_train.reshape(2592, 2)
Y_test.reshape(648, 2)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


y_pred = model.predict_classes(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix([ np.where(r==1)[0][0] for r in Y_test], y_pred)
print(cm)