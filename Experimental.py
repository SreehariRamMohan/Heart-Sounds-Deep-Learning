from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
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
import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import ModelCheckpoint

print("Finished import statements")
pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/melNumpyLabels.npy"
pickle_filepath_X = "/Users/sreeharirammohan/Desktop/melNumpyImages.npy"

X = np.load(pickle_filepath_X)
Y = np.load(pickle_filepath_Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#bellow not needed in original mel-spec version
X_train = np.array(X_train)
X_test = np.array(X_test)

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

Y_train.reshape(993, 2)
Y_test.reshape(249, 2)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


X_train /= 255
X_test /= 255

#print(X_test)

def create_model(weights_path=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', padding="same", input_shape=(3, 640, 480)))
    model.add(Conv2D(64, (3, 3), activation='relu', dim_ordering="tf"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model


def more_complex_model(weights_path=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(3, 640, 480)))
    print("fail")
    model.add(Conv2D(64, (3, 3), activation='relu', dim_ordering="th"))
    print("fail2")
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model

print("created model")
K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_first')
model = create_model()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

filepath="/Users/sreeharirammohan/Desktop/check_point_models/weights-best-{epoch:03d}-{val_acc:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


#try with a lower learning rate, if I try a learning rate of below 10^-6 and it still
#doesn't work then I know something else is wrong. 
print("Compiled model")

history = model.fit(X_train, Y_train,
          batch_size=32,
          callbacks=callbacks_list,
          epochs=6,
          verbose=1,
          validation_data=(X_test, Y_test))

print("Fitted model")

print("Tring to save using hd5py")
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/Users/sreeharirammohan/Desktop/normalSizePLS.h5")
print("Saved model to disk")

print("Trying to display")

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

