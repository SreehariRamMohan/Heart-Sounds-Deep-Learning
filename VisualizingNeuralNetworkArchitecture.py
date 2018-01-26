from __future__ import print_function
import numpy as np
np.random.seed(123)
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import MaxPooling2D
import matplotlib
matplotlib.use('Agg') # No pictures displayed
from keras.utils import plot_model
import pydot
import graphviz

def create_model(weights_path=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(3, 640, 480)))
    model.add(Conv2D(64, (3, 3), activation='relu', dim_ordering="th"))
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


plot_model(model, to_file='/Users/sreeharirammohan/Desktop/model.png', show_shapes=True, show_layer_names=True)