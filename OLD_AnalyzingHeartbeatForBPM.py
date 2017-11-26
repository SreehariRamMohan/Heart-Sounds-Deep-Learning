import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

measures = {}

def get_data(filename):
    dataset = pd.read_csv(filename)
    return dataset


def rolmean(dataset, hrw, fs):
    mov_avg = pd.rolling_mean(dataset.hart, window=(int(hrw*fs)))
    avg_hr = (np.mean(dataset.hart))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    #mov_avg = [x*1.2 for x in mov_avg]
    dataset['hart_rollingmean'] = mov_avg


def detect_peaks(dataset):
    window = []
    peaklist = []
    listpos = 0
    for datapoint in dataset.hart:
        rollingmean = dataset.hart_rollingmean[listpos]
        if (datapoint <= rollingmean) and (len(window) <= 1): #Here is the update in (datapoint <= rollingmean)
            listpos += 1
        elif (datapoint > rollingmean):
            window.append(datapoint)
            listpos += 1
        else:
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window)))
            peaklist.append(beatposition)
            window = []
            listpos += 1
    measures['peaklist'] = peaklist
    measures['ybeat'] = [dataset.hart[x] for x in peaklist]


def calc_RR(dataset, fs):
    peaklist = measures['peaklist']
    RR_list = []
    cnt = 0
    while (cnt < (len(peaklist) - 1)):
        RR_interval = (peaklist[cnt + 1] - peaklist[cnt])
        ms_dist = ((RR_interval / fs) * 1000.0)
        RR_list.append(ms_dist)
        cnt += 1

    RR_diff = []
    RR_sqdiff = []
    cnt = 0
    while (cnt < (len(RR_list) - 1)):
        RR_diff.append(abs(RR_list[cnt] - RR_list[cnt + 1]))
        RR_sqdiff.append(math.pow(RR_list[cnt] - RR_list[cnt + 1], 2))
        cnt += 1
    measures['RR_list'] = RR_list
    measures['RR_diff'] = RR_diff
    measures['RR_sqdiff'] = RR_sqdiff


def calc_ts_measures():
    RR_list = measures['RR_list']
    RR_diff = measures['RR_diff']
    RR_sqdiff = measures['RR_sqdiff']
    measures['bpm'] = 60000 / np.mean(RR_list)
    measures['ibi'] = np.mean(RR_list)
    measures['sdnn'] = np.std(RR_list)
    measures['sdsd'] = np.std(RR_diff)
    measures['rmssd'] = np.sqrt(np.mean(RR_sqdiff))
    NN20 = [x for x in RR_diff if (x > 20)]
    NN50 = [x for x in RR_diff if (x > 50)]
    measures['nn20'] = NN20
    measures['nn50'] = NN50
    measures['pnn20'] = float(len(NN20)) / float(len(RR_diff))
    measures['pnn50'] = float(len(NN50)) / float(len(RR_diff))

def get_sample_rate(dataset):
    # Simple way to get sample rate
    sampletimer = [x for x in dataset.timer]  # dataset.timer is a ms counter with start of recording at '0'
    measures['fs'] = ((len(sampletimer) / sampletimer[
        -1]) * 1000)  # Divide total length of dataset by last timer entry. This is in ms, so multiply by 1000 to get Hz value

    print("Measures fs is = " + str(measures['fs']))


#Don't forget to update our process() wrapper to include the new function
def process(dataset, hrw, fs):
    rolmean(dataset, hrw, fs)
    detect_peaks(dataset)
    calc_RR(dataset, fs)
    calc_ts_measures()
    plotter(dataset, "My Heartbeat Plot")



def plotter(dataset, title):
    peaklist = measures['peaklist']
    ybeat = measures['ybeat']
    plt.title(title)
    plt.plot(dataset.hart, alpha=0.5, color='blue', label="raw signal")
    plt.plot(dataset.hart_rollingmean, color='green', label="moving average")
    plt.scatter(peaklist, ybeat, color='red', label="average: %.1f BPM" % measures['bpm'])
    plt.legend(loc=4, framealpha=0.6)
    plt.show()


from scipy.signal import butter, lfilter  # Import the extra module required

# Define the filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frequeny is half the sampling frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


dataset = get_data('/Users/sreeharirammohan/Desktop/TensorFlow_Programs/Heart-Sounds-Deep-Learning/data2.csv')
dataset = dataset[6000:12000].reset_index(
    drop=True)  # For visibility take a subselection of the entire signal from samples 6000 - 12000 (00:01:00 - 00:02:00)

filtered = butter_lowpass_filter(dataset.hart, 2.5, 100.0,
                                 5)  # filter the signal with a cutoff at 2.5Hz and a 5th order Butterworth filter

# Plot it
plt.subplot(211)
plt.plot(dataset.hart, color='Blue', alpha=0.5, label='Original Signal')
plt.legend(loc=4)
plt.subplot(212)
plt.plot(filtered, color='Red', label='Filtered Signal')
plt.ylim(200,
         800)  # limit filtered signal to have same y-axis as original (filter response starts at 0 so otherwise the plot will be scaled)
plt.legend(loc=4)
plt.show()



import AnalyzingHeartbeatForBPM as hb #Assuming we named the file 'heartbeat.py'
dataset = hb.get_data("/Users/sreeharirammohan/Desktop/TensorFlow_Programs/Heart-Sounds-Deep-Learning/data2.csv")
get_sample_rate(dataset)
hb.process(dataset, 0.75, 100)

for key, value in hb.measures.items():
    print(str(key) + " ==> " + str(value))
