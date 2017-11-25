one_count = 0
zero_count = 0


import os
import numpy
import re
from skimage import io
import glob
import numpy as np
import _pickle as cPickle
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split

#0 is normal
#1 is abnormal

X = numpy.array([])
y = list()

def get_class(img_path):
    global one_count
    global zero_count
    if "abnormal" in img_path:
        print("returning 1!")
        one_count += 1
        return 1
    else:
        print("returning 0!")
        zero_count += 1
        return 0

pickle_filepath_X = "/Users/sreeharirammohan/Desktop/all_data/allMelNumpyImages.npy"
pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/all_data/allMelNumpyLabels.npy"

if not os.path.exists(pickle_filepath_X) or not os.path.exists(pickle_filepath_Y):
    print("in the first if we are creating numpy arrays for images and labels")
    #If we haven't created the pickle file yet, create the dictionary
    #Computations here ..............
    root_dir = '/Users/sreeharirammohan/Desktop/all_data/combined'
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*.*'))
    print(all_img_paths)
    np.random.shuffle(all_img_paths)
    count = 0
    #X = np.empty((3240, 375, 875, 3), dtype="float32")
    print("done with creating empty numpy array of size " + str(X.shape))
    for i, img_path in enumerate(all_img_paths):
        count += 1
        print(str((i/3240)*100) + "%")
        #img is the numpy array
        img = io.imread(img_path)
        img = np.swapaxes(img, 2, 0)
        #X[i, ...] = img
        label = get_class(img_path)
        labels.append(label)
        imgs.append(img)


    X = np.array(imgs, dtype='float32')
    # Make one hot targets
    Y = np.eye(2, dtype='uint8')[labels]
    print(X.shape)
    print(Y.shape)
    #.......
    np.save(pickle_filepath_X, X)
    np.save(pickle_filepath_Y, Y)
    print("saved both numpy arrays")
    print("1 count is " + str(one_count))
    print("0 count is " + str(zero_count))
    
else:
    print("We have stored files, retrieving....")
    X = np.load(pickle_filepath_X)
    Y = np.load(pickle_filepath_Y)
    print(X.shape)
    print(Y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print("---------")
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    




