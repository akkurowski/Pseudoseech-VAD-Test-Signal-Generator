import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ellip, lfilter, periodogram
import os

os.system('cls')

fs = 48_000
DECIM_FACTOR = 10

pseudovoice_signal          = np.load('voice_signal_floatpoint.npy')
words_presence_pattern  = np.load('words_presence_pattern.npy')
t_vec = np.arange(0,len(pseudovoice_signal))/fs

# plt.figure()
# plt.plot(words_presence_pattern)
# plt.show()
# exit()

speech_mask  = words_presence_pattern==1
silence_mask = words_presence_pattern==0

t_speech   = t_vec[speech_mask]
t_silence  = t_vec[silence_mask]

pseudovoice_signal_speech   = pseudovoice_signal[speech_mask]
pseudovoice_signal_silence  = pseudovoice_signal[silence_mask]

def make_clustered_plot(t_vec,y_vec,gap_length,c='red'):
    t_diff = np.diff(t_vec)*fs
    t_diff = np.min(np.array([t_diff,gap_length*np.ones_like(t_diff)]),axis=0)
    t_diff = t_diff - np.min(t_diff)
    t_diff = t_diff/np.max(t_diff)
    
    indices = np.where(t_diff>0.5)[0]
    ranges = []
    for i in range(1,len(indices)):
        id1,id2 = (indices[i-1]+1,indices[i])
        plt.plot(t_vec[id1:id2],y_vec[id1:id2],c=c)

plt.figure()
make_clustered_plot(t_silence[0:-1:DECIM_FACTOR],pseudovoice_signal_silence[0:-1:DECIM_FACTOR],DECIM_FACTOR*1.5,'blue')
make_clustered_plot(t_speech[0:-1:DECIM_FACTOR],pseudovoice_signal_speech[0:-1:DECIM_FACTOR],DECIM_FACTOR*1.5,'red')
plt.grid()
plt.xlabel('t [s]')
plt.plot([],[],c='blue',label='silence')
plt.plot([],[],c='red',label='speech')
plt.legend()

plt.show()