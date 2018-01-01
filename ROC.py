'''''
This script plots the ROC curve for the given model.

ROC curves are particularly helpful because they tell us how sensitivity and specificity are affected by 
various thresholds without having to manually change each threshold. 

ROC curve can help us choose a threshold that balances sensitivity and specifity in a way that makes sense for 
my particular context.

'''''


from __future__ import print_function
import numpy as np
np.random.seed(123)
import numpy as np
np.random.seed(123)
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import io
import numpy as np
np.random.seed(123)  # for reproducibility
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import MaxPooling2D
from skimage import io
import matplotlib
matplotlib.use('Agg')
import matplotlib
matplotlib.use('Agg')
import numpy as np

def convert_spectrogram_to_numpy(path_to_spectrogram):
    img = io.imread(path_to_spectrogram)
    return img

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

print("Finished import statements")
pickle_filepath_X = "/Users/sreeharirammohan/Desktop/all_data/allMelNumpyImages.npy"
pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/all_data/allMelNumpyLabels.npy"

X = np.load(pickle_filepath_X)
Y = np.load(pickle_filepath_Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

Y_train.reshape(2592, 2)
#Y_test.reshape(648, 2)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


y_pred = model.predict(X_test)


print("---------------ROC / AUC / Frequency---------------")

from sklearn.metrics import *
#store predicted probabilities for class 1
y_pred_proba = model.predict_proba(X_test)[:, 1]

'''
Plotting a histogram of predicted probabilities
'''
# histogram of predicted probabilities
plt.hist(y_pred_proba, bins=8)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')
plt.show()

'''
Basic example of plotting ROC curve

ROC curve shows relationship between True Positive Rate and False Positive Rate

ROC shows sensitivity vs (1-specificity) for all possible classification
thresholds from 0-1

'''

Y_test = [ np.where(r==1)[0][0] for r in Y_test]

from sklearn.metrics import *
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for heart abnormality classification')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()


def evaluate_threshold(threshold, tpr, fpr):
    print("senitivity: " + str(tpr[threshold > threshold][-1]))
    print("Specificity " + str(float(1 - fpr[threshold > threshold][-1])))



''''
AUC is the area under the ROC curve

A higher AUC score is a better classifier

Used as single number summary of performance of classifier

best possible AUC is 1

AUC is useful even when there is a high class imbalance

'''

# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(roc_auc_score(Y_test, y_pred_proba))
