import heartbeat_audio_experimental as hb_audio

measures = hb_audio.process('/Users/sreeharirammohan/Desktop/Sreehari_heartbeat.wav')
print(measures['bpm'])
hb_audio.plotter() #returns 72.016