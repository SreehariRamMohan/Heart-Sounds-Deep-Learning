import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#retrieving the X and Y data stored as .npy files with pickle
pickle_filepath_X = "/Users/sreeharirammohan/Desktop/MFCCs_Data.npy"
pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/MFCC_Labels.npy"

print("Retrieving X and Y training data")
X = np.load(pickle_filepath_X)
y = np.load(pickle_filepath_Y)

'''
Fix the dimensionality of the dataset, since sklearn can only take 2 dimensions
'''
nsamples, nx, ny = X.shape
X_2 = X.reshape((nsamples,nx*ny))
print(X_2.shape)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size = 0.2, random_state = 0)

# Fitting Decision Tree Algorithm to the Training set
from sklearn.tree import DecisionTreeClassifier

#class_weight_dict = {0:1, 1:3.872180451}
#don't use class_weight_dict I found that the accuracy goes down to 63.8888888889%

classifier = DecisionTreeClassifier(criterion= 'entropy', random_state=123, class_weight=class_weight_dict)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#creating basic confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Getting the basic validated accuracy of the dummy classifier
accuracy = classifier.score(X_test, y_test)
print("Accuracy is " + str(accuracy*100) + "%")