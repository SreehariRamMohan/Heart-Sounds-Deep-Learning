import heartbeat as hb
from scipy.io import wavfile as wav
import numpy
import pandas as pd


RAW_CSV_NAME = "/Users/sreeharirammohan/Desktop/TensorFlow_Programs/Heart-Sounds-Deep-Learning/pls.csv"
fs, data = wav.read('/Users/sreeharirammohan/Desktop/output_test.wav')
df = pd.DataFrame(data)
df.columns=['hart']
df.to_csv(RAW_CSV_NAME, index=False, header="hart" )

print("Sampling frequency " + str(fs))
hrdata = hb.get_data(RAW_CSV_NAME, column_name = 'hart')
measures = hb.process(data, 44100)

print(measures['bpm']) #returns BPM value
print(measures['lf/hf']) # returns LF:HF ratio

#Alternatively, use dictionary stored in module:
print(hb.measures['bpm']) #returns BPM value
print(hb.measures['lf/hf']) # returns LF:HF ratio
