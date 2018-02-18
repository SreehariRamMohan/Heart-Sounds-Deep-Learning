from __future__ import print_function
import numpy as np
np.random.seed(123)  # for reproducibility
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/all_data/allMelNumpyLabels.npy"
pickle_filepath_X = "/Users/sreeharirammohan/Desktop/all_data/allMelNumpyImages.npy"


X = np.load(pickle_filepath_X)
Y = np.load(pickle_filepath_Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Files Loaded")


#must reshape the numpy array to have only 2 dimensions.
print("Starting to reshape")
print(X_train.shape)
print(type(X_train.shape))
nsamples= X_train.shape[0]
rgb = X_train.shape[1]
nx = X_train.shape[2]
ny = X_train.shape[3]
X_train_2 = X_train.reshape((nsamples,nx*ny*rgb))

print("done reshaping X_train")

print(X_test.shape)
print(type(X_test.shape))
nsamples= X_test.shape[0]
rgb = X_test.shape[1]
nx = X_test.shape[2]
ny = X_test.shape[3]

X_test_2 = X_test.reshape((nsamples,nx*ny*rgb))
print("done reshaping")

# Applying PCA
#I had to use IncrementalPCA and process the data in batches because the default
#PCA sklearn module was used up too much CPA and RAM so it was killed by my MacOS system to prevent a crash
#Without using IncrementalPCA the following error occurrs: Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
from sklearn.decomposition import IncrementalPCA
pca = IncrementalPCA(n_components = 2, batch_size=16)
X_train = pca.fit_transform(X_train_2)
print("PCA on X_train complete")
X_test = pca.transform(X_test_2)
print("PCA on X_test complete")
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio: " + str(explained_variance))