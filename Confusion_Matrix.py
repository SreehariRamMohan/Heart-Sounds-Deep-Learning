from __future__ import print_function
from sklearn.metrics import confusion_matrix
import numpy as np
np.random.seed(123)
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import MaxPooling2D

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


print("Finished import statements")
pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/melNumpyLabels.npy"
pickle_filepath_X = "/Users/sreeharirammohan/Desktop/melNumpyImages.npy"

X = np.load(pickle_filepath_X)
Y = np.load(pickle_filepath_Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train.reshape(993, 2)
Y_test.reshape(249, 2)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


def create_model(self, weights_path=None):
    print("Creating Deep Learning Model for Analysis")
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(3, 640, 480)))
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



'''
basic example of plotting a confusion matrix is below
'''
y_pred = model.predict_classes(X_test)
print(y_pred)
cm = confusion_matrix(Y_test, y_pred)
cm.show()


'''

'''