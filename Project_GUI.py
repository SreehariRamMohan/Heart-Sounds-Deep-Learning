from tkinter import *
from tkinter import ttk
import tkinter as tk
from PIL import ImageTk
import pyaudio
import wave
import numpy as np

np.random.seed(123)  # for reproducibility
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # No pictures displayed

import threading


import numpy
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
import pylab
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


class Project_GUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # recording attributes.
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.RECORD_SECONDS = 30
        self.WAVE_OUTPUT_FILENAME = "/Users/sreeharirammohan/Desktop/TensorFlow_Programs/Heart-Sounds-Deep-Learning/Sreehari_heartbeat.wav"
        self.CARDIOGRAM_OUTPUT_FILENAME = "/Users/sreeharirammohan/Desktop/TensorFlow_Programs/Heart-Sounds-Deep-Learning/cardiogram.jpg"
        self.MEL_SPECTROGRAM_OUTPUT_FILENAME = "/Users/sreeharirammohan/Desktop/TensorFlow_Programs/Heart-Sounds-Deep-Learning/mel-spectrogram.jpg"
        self.image_reference_1 = None
        self.image_reference_2 = None
        self.image_list = []
        '''
        when displaying photos
        photoType of 1 is the cardiogram
        photoType of 2 is the mel-spectrogram
        '''


        root = self
        root.title("Heart Analysis")

        label_title = Label(root, text="Heart Spectrogram Analysis Tool")
        label_title.pack(fil="x")

        topFrame = Frame(root)
        topFrame.pack(side=TOP)

        self.recordButton = Button(topFrame, text="Record", command=self.start)
        self.recordButton.pack(side=LEFT)

        self.progress = ttk.Progressbar(topFrame, orient="horizontal",
                                        length=200, mode="determinate")
        self.min = 0
        self.maxMin = 0
        self.progress.pack(side=RIGHT)

        diagnosis_label = Label(root, text="Diagnosis will show up here")
        diagnosis_label.pack()

        imageFrame = Frame(root)
        imageFrame.pack()

        #create a clear button to reset the application and get it ready for another patient
        self.clearButton = Button(root, text="Next Patient", command=self.clear)
        self.clearButton.pack()

        root.mainloop()


    def showImage(self, width, height, path, root, photoType):

        img = Image.open(path)
        img = img.resize((width, height), Image.ANTIALIAS)  # The (250, 250) is (height, width)
        img = ImageTk.PhotoImage(img)

        # The Label widget is a standard Tkinter widget used to display a text or image on the screen.
        panel = tk.Label(root, image=img)
        self.image_list.append(panel)
        # The Pack geometry manager packs widgets in rows or columns.
        panel.pack(side="bottom", fill="both", expand="yes")
        tk.Tk.update_idletasks(self)
        tk.Tk.update(self)

        if photoType == 2:
            self.image_reference_2 = img
        elif photoType == 1:
            self.image_reference_1 = img



    def clear(self):
        print("Clearing patient information")
        self.progress["value"] = 0 #clear the progress bars
        self.min = 0
        self.image_reference_1 = None
        self.image_reference_2 = None
        print(self.image_reference_1)
        print(self.image_reference_2)

        for image in self.image_list:
            image.config(image='') #clear the images on the screen

    def start(self):
        print("here")

        self.progress["value"] = 0
        self.maxMin = 30
        self.progress["maximum"] = 30


        '''
        I need the following 2 lines to happen at the same time
        the progress bar needs to update from 1-30 seconds(which is the time I am recording)
        '''
        self.update_progress()
        self.record_thread = threading.Thread(target=self.record, daemon=True)
        self.record_thread.start()

    def update_progress(self):
        '''simulate reading 500 bytes; update progress bar'''
        self.min += 0.1
        self.progress["value"] = self.min
        if self.min < self.maxMin:
            # read more bytes after 100 ms
            self.after(100, self.update_progress)

    def record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                             channels=self.CHANNELS,
                             rate=self.RATE,
                             input=True,
                             frames_per_buffer=self.CHUNK)

        print("* recording")

        frames = []

        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        # multi-thread the creation of the two new threads in python
        self.create_cardiogram_thread = threading.Thread(target=self.create_cardiogram, daemon=True)
        self.create_cardiogram_thread.start()

        self.create_mel_spectrogram_thread = threading.Thread(target=self.create_mel_spectrogram, daemon=True)
        self.create_mel_spectrogram_thread.start()




    def create_cardiogram(self):
        # Read file and get sampling freq [ usually 44100 Hz ]  and sound object
        samplingFreq, mySound = wavfile.read(self.WAVE_OUTPUT_FILENAME)
        # Check if wave file is 16bit or 32 bit. 24bit is not supported
        mySoundDataType = mySound.dtype
        # We can convert our sound array to floating point values ranging from -1 to 1 as follows
        mySound = mySound / (2. ** 15)

        # Check sample points and sound channel for duel channel(5060, 2) or  (5060, ) for mono channel

        mySoundShape = mySound.shape
        samplePoints = float(mySound.shape[0])

        # Get duration of sound file
        signalDuration = mySound.shape[0] / samplingFreq

        # If two channels, then select only one channel
        # mySoundOneChannel = mySound[:,0]

        # if one channel then index like a 1d array, if 2 channel index into 2 dimensional array
        if len(mySound.shape) > 1:
            mySoundOneChannel = mySound[:, 0]
        else:
            mySoundOneChannel = mySound

        # Plotting the tone

        # We can represent sound by plotting the pressure values against time axis.
        # Create an array of sample point in one dimension
        timeArray = numpy.arange(0, samplePoints, 1)

        #
        timeArray = timeArray / samplingFreq

        # Scale to milliSeconds
        timeArray = timeArray * 1000

        plt.rcParams['agg.path.chunksize'] = 100000

        # Plot the tone
        plt.xlim((0, 8000))
        plt.figure(figsize=(0.875, 0.375))
        ax1 = plt.axes(frameon=False)
        ax1.set_frame_on(False)
        ax1.axes.get_yaxis().set_visible(False)
        ax1.axes.get_xaxis().set_visible(False)
        plt.plot(timeArray, mySoundOneChannel, color='Black')
        # plt.xlabel('Time (ms)')
        # plt.ylabel('Amplitude')
        print("trying to save")
        plt.savefig(self.CARDIOGRAM_OUTPUT_FILENAME)
        print("saved")
        print("not going to show now")
        # plt.show()

        # very important to prevent memory leaks!
        plt.close()

        self.showImage(300, 200, self.CARDIOGRAM_OUTPUT_FILENAME, self, 2)




    def create_mel_spectrogram(self):
        sig, fs = librosa.load(self.WAVE_OUTPUT_FILENAME)
        # make pictures name
        save_path = self.MEL_SPECTROGRAM_OUTPUT_FILENAME

        pylab.axis('off')  # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
        S = librosa.feature.melspectrogram(y=sig, sr=fs)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
        pylab.close()

        self.showImage(200, 150, self.MEL_SPECTROGRAM_OUTPUT_FILENAME, self, 1)





app = Project_GUI()
app.mainloop()
