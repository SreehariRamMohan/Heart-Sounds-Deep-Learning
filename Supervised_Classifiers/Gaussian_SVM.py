import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#retrieving the X and Y data stored as .npy files with pickle
pickle_filepath_X = "/Users/sreeharirammohan/Desktop/MFCCs_Data.npy"
pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/MFCC_Labels.npy"

print("Retrieving X and Y training data")
X = np.load(pickle_filepath_X)
y = np.load(pickle_filepath_Y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Gaussian (Radial) SVM Algorithm to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#creating basic confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix([ np.where(r==1)[0][0] for r in y_test], y_pred)
print(cm)

#Getting the basic validated accuracy of the dummy classifier
accuracy = classifier.score(X_test, y_test)
print("Accuracy is " + str(accuracy*100) + "%")