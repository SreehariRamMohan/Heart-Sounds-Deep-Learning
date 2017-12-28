import hypertools as hyp
import numpy as np

pickle_filepath_X = "/Users/sreeharirammohan/Desktop/MFCCs_Data.npy"

print("Retrieving X and Y training data")
X = np.load(pickle_filepath_X)

print("Plotting")
hyp.plot(X, '.', n_clusters=2)