import hypertools as hyp
import numpy as np

pickle_filepath_X = "/Users/sreeharirammohan/Desktop/MFCCs_Data.npy"

print("Retrieving X and Y training data")
X = np.load(pickle_filepath_X)

'''
Fix the dimensionality of the dataset, since sklearn can only take 2 dimensions
'''
nsamples, nx, ny = X.shape
X_2 = X.reshape((nsamples,nx*ny))
print(X_2.shape)

print("Plotting")
hyp.plot(X_2, '.', n_clusters=2)
print("done")