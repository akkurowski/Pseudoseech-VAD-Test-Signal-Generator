import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from scipy.io.wavfile import write
from scipy.signal import ellip, lfilter, periodogram

@nb.jit
def exp_avg(input,alpha):
    output  = np.zeros_like(input)
    acc     = input[0]
    for i in range(0,len(input)-1):
        output[i] = acc
        acc = acc*(1-alpha) + alpha*input[i+1]
    output[-1] = acc
    return output

@nb.jit
def word_pattern(length,word2sil_proba, sil2word_proba):
    output              = np.zeros(length)
    now_is_word_state   = np.random.randint(2)
    output[0] = now_is_word_state
    for i in range(1,length):
        if now_is_word_state:
            if np.random.rand()<word2sil_proba: now_is_word_state=0
        else:
            if np.random.rand()<sil2word_proba: now_is_word_state=1
        output[i] = now_is_word_state
    return output

def generate_bp_signal(fl,fh,fs,siglength):
    wl = fl/fs*2
    wh = fh/fs*2
    b, a = ellip(2, 5, 50, [wl,wh], 'bandpass')
    voice_subband = np.random.rand(siglength)*2-1
    voice_subband = lfilter(b,a,voice_subband)
    voice_subband = lfilter(b,a,voice_subband)
    voice_subband = voice_subband/np.max(np.abs(voice_subband))
    return voice_subband

# definitions from:
# http://resource.isvr.soton.ac.uk/staff/pubs/PubPDFs/BS%20EN%2060268-16.pdf
def generate_STIPA(siglength, fs):
    def generate_double_modsig(mf1, a1, mf2, a2, fs, siglength):
        t_vec   = np.arange(0,siglength)/fs
        signal1 = a1*np.sin(2*np.pi*mf1*t_vec)
        signal2 = a2*np.sin(2*np.pi*mf2*t_vec + np.pi)
        return signal1+signal2

    NUM_CARRIERS = 7
    fc_vec  = [125,250,500,1000,2000,4000,8000]
    mf1_vec = [1.60,1.00,0.63,2.00,1.25,0.80,2.50]
    mf2_vec = [8.00,5.00,3.15,10.0,6.25,4.00,12.5]

    output_signal = np.zeros(siglength)
    for i in range(NUM_CARRIERS):
        fc  = fc_vec[i]
        mf1 = mf1_vec[i]
        mf2 = mf2_vec[i]
        fl = fc/np.sqrt(4/3)
        fh = fc*np.sqrt(4/3)
        if fh>(fs/2): break
        carrier_signal = generate_bp_signal(fl,fh,fs,siglength)/NUM_CARRIERS
        carrier_signal *= generate_double_modsig(mf1, 0.5, mf2, 0.5, fs, siglength)
        output_signal += carrier_signal
        # print(fl,fh,carrier_signal[0:10])
    output_signal = output_signal/np.abs(np.max(output_signal))
    
    # f, Pxx_den = periodogram(output_signal,fs)
    # plt.figure()
    # plt.semilogy(f, Pxx_den)
    # plt.show()
    # exit()
    
    return output_signal

# --------------------------------------------

# fs              = 12_000 # [Sa/s]
fs              = 48_000 # [Sa/s]
SIM_TIME        = 5*60 # [s]
# SIM_TIME        = 15 # [s]

SPEECH_FMULT    = 5
SPEECH_SMOOTH   = 0.001
NUM_SINES       = 10

SPCH2SIL_PROBA  = 0.0001
SIL2SPCH_PROBA  = 0.000045
WORD_SMOOTH     = 0.0005

# NOISE_ALPHA     = 0.1
NOISE_ALPHA     = 0.0

WORDS_PATTERN_CLASS_THR = 0.02

# creation of speech envelope base signal
process_t = np.linspace(0,SIM_TIME,SIM_TIME*fs)
words_envelope = np.zeros_like(process_t)
for i in range(NUM_SINES):
    random_sine_signal = np.sin(2*np.pi*np.random.rand()*SPEECH_FMULT*process_t + 2*np.pi*np.random.rand()) + np.random.rand()
    random_sine_signal = np.sign(random_sine_signal)
    random_sine_signal = 0.5*(random_sine_signal + np.abs(random_sine_signal))
    words_envelope += random_sine_signal
words_envelope = words_envelope/NUM_SINES

# introduction of silence periods
word_presence_raw = word_pattern(len(process_t),SPCH2SIL_PROBA, SIL2SPCH_PROBA)
word_presence_pattern = exp_avg(word_presence_raw,WORD_SMOOTH)
words_envelope = exp_avg(words_envelope,SPEECH_SMOOTH)*word_presence_pattern

process_y = words_envelope*generate_STIPA(len(words_envelope),fs)
process_y = (1-NOISE_ALPHA)*process_y + NOISE_ALPHA*np.random.rand(len(process_y))


word_presence_pattern_binary = word_presence_pattern - WORDS_PATTERN_CLASS_THR
word_presence_pattern_binary = np.sign(word_presence_pattern_binary)
word_presence_pattern_binary = 0.5*(np.abs(word_presence_pattern_binary)+word_presence_pattern_binary)

print('speech percentage: ', 100*np.mean(word_presence_pattern_binary))

np.save('words_presence_pattern.npy',word_presence_pattern_binary)
np.save('voice_signal_floatpoint.npy',process_y)

write('pseudo-speech.wav', fs, (process_y*32000).astype('int16'))
plt.figure(figsize=(16,9))
plt.plot(process_t[0:-1:100],process_y[0:-1:100])
plt.plot(process_t[0:-1:100],word_presence_raw[0:-1:100])
plt.plot(process_t[0:-1:100],word_presence_pattern[0:-1:100])
plt.xlabel('t [s]')
plt.savefig('signal_visu.png')