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

pickle_filepath_X = "/Users/sreeharirammohan/Desktop/numpyImages.npy"
pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/numpyLabels.npy"


K.set_image_data_format('channels_first')


X = np.load(pickle_filepath_X)
Y = np.load(pickle_filepath_Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

Y_train.reshape(2592, 2)
Y_test.reshape(648, 2)



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(Y_train.shape)

model = Sequential()

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(3,875,375)))
print(model.output_shape)

model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)

