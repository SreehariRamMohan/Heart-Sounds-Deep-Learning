from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import numpy as np
np.random.seed(123)  # for reproducibility
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import MaxPooling2D
from sklearn.model_selection import train_test_split

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


WEIGHTS_PATH = "/Users/sreeharirammohan/Desktop/check_point_models/weights-best-031-0.88735.hdf5"
WEIGHTS_PATH_PI = "/home/pi/Desktop/weights-best-031-0.88735.hdf5"

pickle_filepath_X = "/Users/sreeharirammohan/Desktop/all_data/allMelNumpyImages.npy"
pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/all_data/allMelNumpyLabels.npy"

pickle_filepath_X_pi = "/media/pi/3577-249A/MFCCs_Data.npy"
pickle_filepath_Y_pi = "/media/pi/3577-249A/MFCC_Labels.npy"

USING_RASPBERRY_PI = False

if USING_RASPBERRY_PI:
    pickle_filepath_X = pickle_filepath_X_pi
    pickle_filepath_Y = pickle_filepath_Y_pi

model = create_model(WEIGHTS_PATH)


X = np.load(pickle_filepath_X)
Y = np.load(pickle_filepath_Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("----------starting 5 time average benchmark----------")
times = []
import time
for i in range(0, 5):
    start = time.clock()
    y_pred = model.predict_classes(X_test)
    time_taken = time.clock() - start
    times.append(time_taken)
    print("On iteration " + str(i + 1) + " time taken was " + str(time_taken))
print("DONE WITH 5 TESTS")
print(times)
import statistics as s
print("Average time = " + str(s.mean(times)))
print("----------ending time benchmark----------")
