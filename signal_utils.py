import librosa as lr
import numpy as np
import numpy.random as nr
import scipy as sp
import random

# def open_wav(filename, sample_rate, normal_const = 2 ** 15, DEBUG=0):
#     audio = o_audio * 1.0
#     if DEBUG > 0:
#         print("Reading from",filename)
#     # Parsing the wav file
#     f_sr,o_audio = sp.io.wavfile.read(filename)
#     print("Original Audio shape",audio.shape)

#     DOWNSAMPLE = False
#     if not f_sr == sample_rate:
#         DOWNSAMPLE = True
#         print("\nOriginal sampling aka frame rate:",f_sr)
#         # Resample to 8k Hz (Loses information)
#         # Try different resampling configs (kaiser best)
#         audio = lb.resample(audio,f_sr,sample_rate)
#         if DEBUG > 0:
#             print("Audio shape",audio.shape)

#     normalize = lambda x: x/(normal_const)
#     audio = normalize(audio)
    
#     original_len = audio.shape[0]
    
#     return audio

def lr_open_wav(filename, sample_rate, DEBUG=0):
    DOWNSAMPLE = False
    
    # Parsing the wav file
    signal, s_freq = lr.load(filename, sample_rate)
    audio = signal * 1.0
    
    # If need to resample to 8kHz
    if not s_freq == sample_rate:
        DOWNSAMPLE = True
        # Try different resampling configs (kaiser best)
        audio = lr.resample(audio, s_freq, sample_rate)

    if DEBUG > 0:
        print("Reading from",filename)
        print("Original Audio shape",audio.shape)
        print("\nOriginal sampling aka frame rate:",f_sr)
        print("Audio shape",audio.shape)
    
    original_len = audio.shape[0]
    return audio

# def get_power(waveform, win_len, hop_len):
#     spectro = lr.stft(waveform,
#                       n_fft=int(win_len),
#                       hop_length=hop_len,
#                       win_length=int(win_len))
#     # Assuming shape (freq, windows)
#     mag = np.abs(spectro)
#     p_frame_avg = np.average(mag**2,0) # Average power of each frame
#     p_wav = np.average(p_frame_avg) # Average power of the entire audio wave
#     return p


# In: Waveform
# Out: Spectrogram
def conv_to_spectro(audio):
    WINDOW_LENGTH = 256
    return lr.stft(audio,
                   n_fft=WINDOW_LENGTH,
                   hop_length=WINDOW_LENGTH//2,
                   win_length=int(WINDOW_LENGTH))

# In: 2D Spectrogram
# Out: Power
def get_power(spectro):
    if not len(spectro.shape) == 2:
        raise Exception("Expected 2D spectrogram, got shape " + str(spectro.shape))
        return -1
    mag = np.abs(spectro)         # Magnitude of spectro
    frame_avg = np.average(mag**2,0)  # Avg power of spectro
    power = np.average(frame_avg)     # Avg power of audio wave
    return power
  
# In: Dictionary of spectros 
# Out: Average power of all spectros
def get_avg_power(spectros):
    return np.average(power)
    for spectro in spectros.values(): # For each spectro in dictionary 
        power.append(get_power(spectro))
    return np.average(power)

# In: Spectrogram, average power to normalise to
# Out: Normalised Spectrogram
def normalise_power(spectro, avg_pw):
    normalised = ((avg_pw/get_power(spectro))**0.5)*spectro
    return normalised

# Input: Dictionary of spectros
# Output: Power-Normalised Dictionary of spectros
def normalise_power_batch(spectros):
    avg_pw = get_avg_power(spectros)    
    for key in spectros:
        spectros[key] = normalise_power(spectros[key],avg_pw)
    return spectros
    
# Generates noise with the same power as the signal
# In: tuple of audio shape, signal power, window length, hop length
# Out: noise with same power as audio, noise power
def generate_noise(signal, wl, hl):
    s_signal = conv_to_spectro(signal)
    noise = nr.random(signal.shape)
    noise = noise - np.average(noise) # Normalize
    s_noise = conv_to_spectro(noise)
    
    p_signal = su.get_power(s_signal)
    p_noise  = su.get_power(s_noise)
    
    # Scale waveform and power
    scaled_noise = noise*((p_signal/p_noise)**0.5)
    p_Snoise = p_noise*(p_signal/p_noise)
    return scaled_noise, p_Snoise

# Adds noise to an audio wave.
# In: 1D Audio wave, signal to noise ratio in dB
# Out: 1D Audio wave with noise, same power as original
def add_noise(audio,snr, wl=256, hl=128, DEBUG=0):
    ratio = 10 ** (snr/10)
    spectro = conv_to_spectro(audio)
    p_signal = su.get_power(spectro)
    scaled_noise, p_noise = generate_noise(audio, wl, hl)
    
    # Scale signal and noise to match SNR.
    # Also downscales the result to match original Power
    alpha = (ratio/(ratio+1))**0.5
    beta = 1/(ratio+1)**0.5
    result = (audio*alpha + scaled_noise*beta)
    
    if DEBUG > 0:
      print("Power signal vs noise", p_signal, p_noise)
      print("\nNoise Scaling factor",ratio)
      print("Scales: a", alpha, "b", beta)
    
    return result

# Function to crop the data in a variable shape array to the shortest sample length, start decided randomly
def cutoff(data):
    std_len = min(i.shape for i in data)[0]
    new_data = []
    for sample in data:
        start = random.randint(0,int(sample.shape[0] - std_len))
        cut_sample = sample[start:start+std_len]
        new_data.append(cut_sample)
    return np.array(new_data)

# Crop the data in a variable shape dictionary to the shortest sample length, start decided randomly
# In: Dictionary of spectrograms
# Out: Numpy array of cropped spectrograms
def cutoff_2d(data):
    keys = list(data.keys())
    std_len = min(x_trn[key].shape[1] for key in keys)
    new_data = []
    for key in x_trn:
        start = random.randint(0,int(x_trn[key].shape[1] - std_len))
        sample = x_trn[key]
        cut_sample = sample[:,start:start+std_len]
        new_data.append(cut_sample)
    return np.array(new_data)

# For each frame, grab the context window
# In: transposed spectrogram (X, 129)
# Out: 3D transposed spectrogram (X, 2C + 1, 129) where C = context
def get_context(i_spectro, hop=1, context=5):
    freq = int(i_spectro.shape[1])
    end = i_spectro.shape[0]
    chunks = np.empty((0,2*context+1,freq))
    i = 0
    while i < end:
        if i < context:
            b_windows = int(context - i)
            z = np.zeros((b_windows,freq),)
            back = np.append(z,i_spectro[:i+1],axis=0)
        else:
            back = i_spectro[i-context:i+1]
        if i + context > end:
            f_windows = i + context - end
            z = np.zeros((f_windows,freq),)
            front = np.append(i_spectro[i:],z,axis=0)
        else:
            front = i_spectro[i:i+context]
        
        chunk = np.expand_dims(np.append(back,front,axis=0),axis=0)
        #print("single",chunk.shape,"collection",chunks.shape)
        chunks = np.append(chunks,chunk,axis=0)
        
        i+=hop
    return chunks
    

# In: Numpy array of spectrograms
# Out: Numpy array of transposed spectrograms
def transpose_matrix(matrix):
    result = np.swapaxes(matrix,1,2)
    print("Finished transposing")
    return result

def stft_along_axis(data, window_len = 256, hop_len = 128):
    stft = lambda x: lr.stft(x, window_len, hop_len, window_len)
    labels = np.apply_along_axis(stft,1,data)
    return labels
    
def normalisation_concat(inputs):
    output = np.empty((inputs.shape[1],0))
    lengths = []
    sum = 0
    for array in inputs:
        output = np.concatenate((output, array), axis = 1)
    return output
    
# DEPRECATED CODE
# def normalisation_concat(inputs):
#     output = np.array([[]])
#     lengths = []
#     sum = 0
#     # Averages across the frequencies. Returns one value for eachEach timmealuemor array in inputs:
#     output = np.concatenate((output,array))
#     return output
    
# Takes in 3D data, concatenates it, then normalises it
def normalise_inputs(inputs):
    # Concatenate
    inputs_concat = normalisation_concat(inputs)
    
    averaged = np.mean(inputs_concat,axis=0) # Average of each frame across all frequencies
    var = np.var(averaged, axis = 0) # Variance of all frames
    mean = np.mean(averaged) # Average of averages.
    
    # Normalize 
    inputs = (inputs - mean)/var
    return inputs

def log_spectro(data):
    stft = stft_along_axis(data)
    log_spectro = np.log(abs(stft+1e-32))
    return log_spectro







