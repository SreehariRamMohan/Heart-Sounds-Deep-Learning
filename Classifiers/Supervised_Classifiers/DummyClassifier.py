#This classifier is going to be used as a benchmark, sort of like a control to compare my other models
#This model randomly chooses an output class based on the frequency of "abnormal" and "normal" entries in the
#Training data, in fact it doesn't even look at the feature data!

from sklearn.dummy import DummyClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#retrieving the X and Y data stored as .npy files with pickle
pickle_filepath_X = "/Users/sreeharirammohan/Desktop/MFCCs_Data.npy"
pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/MFCC_Labels.npy"

pickle_filepath_X_pi = "/media/pi/3577-249A/MFCCs_Data.npy"
pickle_filepath_Y_pi = "/media/pi/3577-249A/MFCC_Labels.npy"

USING_RASPBERRY_PI = True

if USING_RASPBERRY_PI:
    pickle_filepath_X = pickle_filepath_X_pi
    pickle_filepath_Y = pickle_filepath_Y_pi



print("Retrieving X and Y training data")
y = np.load(pickle_filepath_Y)
print("Loaded Y data")
X = np.load(pickle_filepath_X)

'''
Fix the dimensionality of the dataset, since sklearn can only take 2 dimensions
'''
nsamples, nx, ny = X.shape
X_2 = X.reshape((nsamples,nx*ny))
print(X_2.shape)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size = 0.2, random_state = 0)

# Fitting Dummy Classifier to the Training set
from sklearn.dummy import DummyClassifier
dummy_classifier = DummyClassifier(strategy="uniform")
dummy_classifier.fit(X_train, y_train)

# Predicting the Test set results
print(X_test.shape)


print("----------starting time benchmark----------")
import time
start = time.clock()
y_pred = dummy_classifier.predict(X_test)
print(print(time.clock() - start))
print("----------ending time benchmark----------")


print(y_pred.shape)
#creating basic confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Getting the basic validated accuracy of the dummy classifier
accuracy = dummy_classifier.score(X_test, y_test)
print("Accuracy is " + str(accuracy*100) + "%")