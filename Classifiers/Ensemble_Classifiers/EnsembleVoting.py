import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

pickle_filepath_X = "/Users/sreeharirammohan/Desktop/MFCCs_Data.npy"
pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/MFCC_Labels.npy"

X = np.load(pickle_filepath_X)
y = np.load(pickle_filepath_Y)

nsamples, nx, ny = X.shape
X_2 = X.reshape((nsamples,nx*ny))
print(X_2.shape)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size = 0.2, random_state = 0)

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

votingClassifier = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('knn', clf3)], voting='hard')
votingClassifier = votingClassifier.fit(X_train, y_train)


print("----------starting 5 time average benchmark----------")
times = []
import time
for i in range(0, 5):
    start = time.clock()
    y_pred = votingClassifier.predict(X_test)
    time_taken = time.clock() - start
    times.append(time_taken)
    print("On iteration " + str(i + 1) + " time taken was " + str(time_taken))
print("DONE WITH 5 TESTS")
print(times)
import statistics as s
print("Average time = " + str(s.mean(times)))
print("----------ending time benchmark----------")




# Predicting the Test set results
y_pred = votingClassifier.predict(X_test)

#creating basic confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Getting the basic validated accuracy of the dummy classifier
accuracy = votingClassifier.score(X_test, y_test)
print("Accuracy is " + str(accuracy*100) + "%")