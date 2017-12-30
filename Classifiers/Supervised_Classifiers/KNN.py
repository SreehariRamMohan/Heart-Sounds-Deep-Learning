import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#retrieving the X and Y data stored as .npy files with pickle
pickle_filepath_X = "/Users/sreeharirammohan/Desktop/MFCCs_Data.npy"
pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/MFCC_Labels.npy"

pickle_filepath_X_pi = "/media/pi/3577-249A/MFCCs_Data.npy"
pickle_filepath_Y_pi = "/media/pi/3577-249A/MFCC_Labels.npy"

USING_RASPBERRY_PI = False

if USING_RASPBERRY_PI:
    pickle_filepath_X = pickle_filepath_X_pi
    pickle_filepath_Y = pickle_filepath_Y_pi

print("Retrieving X and Y training data")
X = np.load(pickle_filepath_X)
y = np.load(pickle_filepath_Y)

print(X.shape)
print(y.shape)

'''
Fix the dimensionality of the dataset, since sklearn can only take 2 dimensions
'''
nsamples, nx, ny = X.shape
X_2 = X.reshape((nsamples,nx*ny))
print(X_2.shape)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size = 0.2, random_state = 0)


# Fitting K-Nearest Neighbors Algorithm to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
print("----------starting 5 time average benchmark----------")
times = []
import time
for i in range(0, 5):
    start = time.clock()
    y_pred = classifier.predict(X_test)
    time_taken = time.clock() - start
    times.append(time_taken)
    print("On iteration " + str(i + 1) + " time taken was " + str(time_taken))
print("DONE WITH 5 TESTS")
print(times)
import statistics as s
print("Average time = " + str(s.mean(times)))
print("----------ending time benchmark----------")

#creating basic confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Getting the basic validated accuracy of the dummy classifier
accuracy = classifier.score(X_test, y_test)
print("Accuracy is " + str(accuracy*100) + "%")