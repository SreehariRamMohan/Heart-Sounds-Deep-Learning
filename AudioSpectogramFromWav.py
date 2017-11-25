"""Plots
Time in MS Vs Amplitude in DB of a input wav signal
"""

#current bug: spectrogram images are repeated, they have titles like carhorn1923416BIT16BIT
#so the 16 bit gets repeated and the program re-reads them so I need to somehow figure out  how to fix this.

#current bug: The graphs scales aren't small enough. As a result, all of my graphs end up looking like the same
#black/grey rectangle. 


import numpy
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft
from pydub import AudioSegment
from PIL import Image
import io
import wave, os, glob
import soundfile as sf
import os
from os.path import basename
import soundfile


#below dependencies required for mel-spectrogram
import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import pylab
import librosa
import librosa.display
import numpy as np


def listAllWavNames(directory):
    for file in os.listdir(directory):
        if(file.endswith('.wav') and 'normal' in file):
            print(file)



def convertAllFilesInDirectoryTo16Bit(directory):
    for file in os.listdir(directory):
        if(file.endswith('.wav')):
            print('file is ' + file)
            nameSolo = file.rsplit('.', 1)[0]
            print(directory + nameSolo )
            data, samplerate = soundfile.read(directory + file)                
            soundfile.write('/Users/sreeharirammohan/Desktop/heartbeat-16bit/' + nameSolo + '|16|.wav', data, samplerate, subtype='PCM_16')
            print("converting " + file + "to 16 - bit")
            

def batchConvertFolderOfWav(directory):
    count = 0
    for file in os.listdir(directory):
        if(file.endswith('.wav')):
            nameSolo = basename(directory+file).rsplit('.', 1)[0]
            print('sending')

            #replace with appropriate method call name right below
            individualWavToMelSpectrogram(directory + '/' + file, nameSolo)
            print('finished ' + str(count))
            count += 1
            

def individualWavToMelSpectrogram(myAudioPath, fileNameToSaveTo):
    sig, fs = librosa.load(myAudioPath)   
    # make pictures name 
    save_path = '/Users/sreeharirammohan/Desktop/mel-all-normal/' + fileNameToSaveTo + '.jpg'

    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()

    

    
def individualWavToSpectrogram(myAudio, fileNameToSaveTo):
    print(myAudio)
    #Read file and get sampling freq [ usually 44100 Hz ]  and sound object
    samplingFreq, mySound = wavfile.read(myAudio)

    #Check if wave file is 16bit or 32 bit. 24bit is not supported
    mySoundDataType = mySound.dtype

    #We can convert our sound array to floating point values ranging from -1 to 1 as follows

    mySound = mySound / (2.**15)

    #Check sample points and sound channel for duel channel(5060, 2) or  (5060, ) for mono channel

    mySoundShape = mySound.shape
    samplePoints = float(mySound.shape[0])

    #Get duration of sound file
    signalDuration =  mySound.shape[0] / samplingFreq

    #If two channels, then select only one channel
    #mySoundOneChannel = mySound[:,0]

    #if one channel then index like a 1d array, if 2 channel index into 2 dimensional array
    if len(mySound.shape) > 1:
        mySoundOneChannel = mySound[:,0]
    else:
        mySoundOneChannel = mySound

    #Plotting the tone

    # We can represent sound by plotting the pressure values against time axis.
    #Create an array of sample point in one dimension
    timeArray = numpy.arange(0, samplePoints, 1)

    #
    timeArray = timeArray / samplingFreq

    #Scale to milliSeconds
    timeArray = timeArray * 1000

    plt.rcParams['agg.path.chunksize'] = 100000

    
    #Plot the tone
    plt.xlim((0, 8000))
    plt.figure(figsize=(0.875,0.375))
    ax1 = plt.axes(frameon=False)
    ax1.set_frame_on(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.axes.get_xaxis().set_visible(False)
    plt.plot(timeArray, mySoundOneChannel, color='Black')
    #plt.xlabel('Time (ms)')
    #plt.ylabel('Amplitude')
    print("trying to save")
    plt.savefig('/Users/sreeharirammohan/Desktop/smaller_test_train_data/abnormal/' + fileNameToSaveTo + '.jpg')
    print("saved")
    print("not going to show now")
    #plt.show()

    #very important to prevent memory leaks!
    plt.close()

#myAudio = "/Users/sreeharirammohan/Desktop/all-abnormal-wav/f0093.wav"
#individualWavToSpectrogram(myAudio, "abnormal")

directoryName = '/Users/sreeharirammohan/Desktop/heart_sounds/all-normal-wav'
batchConvertFolderOfWav(directoryName)


#listAllWavNames(directoryName)

#convertAllFilesInDirectoryTo16Bit('/Users/sreeharirammohan/Desktop/heartbeat-sounds/set_b')
