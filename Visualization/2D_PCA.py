from __future__ import print_function
import numpy as np
from setuptools.command.saveopts import saveopts
np.random.seed(123)  # for reproducibility
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import hypertools as hyp
import matplotlib.patches as mpatches



pickle_filepath_Y = "/Users/sreeharirammohan/Desktop/all_data/allMelNumpyLabels.npy"
pickle_filepath_X = "/Users/sreeharirammohan/Desktop/all_data/allMelNumpyImages.npy"
numpy_2D_PCA_X_test_filePath = os.getcwd() + "/" +"X_test_2D_PCA.npy"
numpy_2D_PCA_X_train_filePath = os.getcwd() + "/" +"X_train_2D_PCA.npy"


def saveFileInCurrentDirectory(name, numpyArrayToSave):
    directoryToSave = os.getcwd() + "/" + name + ".npy"
    np.save(directoryToSave, numpyArrayToSave)

def getColors(yNumpyArray):
    colors = []

    for row in yNumpyArray:
        print(str(row))
        print("Col 0 = " + str(row[0]))
        print("Col 1 = " + str(row[1]))

        if(row[0] == 1):
            #the heartbeat is normal
            colors.append('blue')
        else:
            colors.append('red')

    # for (x, y), value in np.ndenumerate(yNumpyArray):
    #     print("[" + str(x) + "," + str(y) + "] = " + str(value))
    #     if value == 1:
    #         colors.append('blue')
    #     else:
    #         colors.append('red')


    return colors


if(not os.path.exists(numpy_2D_PCA_X_test_filePath) or not os.path.exists(numpy_2D_PCA_X_train_filePath)):

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
    print("PCA object created")
    X_train = pca.fit_transform(X_train_2)
    print("PCA on X_train complete")
    X_test = pca.transform(X_test_2)
    print("PCA on X_test complete")
    explained_variance = pca.explained_variance_ratio_
    print("Explained Variance Ratio: " + str(explained_variance))

    #save the PCA 2D Data to pickle so we can avoid doing this computationally intensive task every single time.
    print("Starting to save")
    saveFileInCurrentDirectory("X_train_2D_PCA", X_train)
    saveFileInCurrentDirectory("X_test_2D_PCA", X_test)
    print("Done saving")
else:
    print("Files exist, retrieving")

    '''
    Pathway exists, so why bother re computing PCA? Exactly, we shouldn't!!
    simply reload the .npy files!
    '''
    #X_train_PCA = np.load(numpy_2D_PCA_X_train_filePath)
    X_test_PCA = np.load(numpy_2D_PCA_X_test_filePath)

    #X_train_PCA /= 255
    X_test_PCA /= 255

    print("Loaded files")

    X = np.load(pickle_filepath_X)
    Y = np.load(pickle_filepath_Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #numpy_test = np.random.randint(0, 2, size=(100, 2))

    colors = getColors(Y_test)

    print(str(len(colors)))

    #plot 2D scatter plot

    import matplotlib.pyplot as plt

    plt.scatter(X_test_PCA[:, 0], X_test_PCA[:, 1], c=colors)
    plt.title('PCA Dimensionality Reduction (2D) Data Scatter Plot')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    red_points = mpatches.Patch(color='red', label='Abnormal Heart Beats')
    blue_points = mpatches.Patch(color='blue', label='Normal Heart Beats')
    plt.legend(handles=[red_points, blue_points], loc='bottom right')
    plt.show()








