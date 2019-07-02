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
def stft(audio):
    WINDOW_LENGTH = 256
    return lr.stft(audio,
                   n_fft=WINDOW_LENGTH,
                   hop_length=WINDOW_LENGTH//2,
                   win_length=int(WINDOW_LENGTH))

# In: Dictionary of wav
# Out: Dictionary of corresponding spectrograms
def stft_batch(wav_dict):
    spectros = {}
    keys = wav_dict.keys()
    for key in keys:
        spectros[key] = stft(wav_dict[key])
    return spectros

# In: 2D Spectrogram
# Out: Power of Spectrogram
def get_power_spec(spectro):
    if not len(spectro.shape) == 2:
        raise Exception("Expected 2D spectrogram, got shape " + str(spectro.shape))
        return -1
    mag = np.abs(spectro)         # Magnitude of spectro
    frame_avg = np.average(mag**2,0)  # Avg power of spectro row-wise
    power = np.average(frame_avg)     # Avg power of entire spectro
    return power

# In: Wav
# Out: Power of wav
def get_power_wav(audio):
    spectro = lr.stft(audio,n_fft=256,hop_length=128)
    return get_power_spec(spectro) 

# In: Dictionary of wavs
# Out: Average power of all wavs
def get_avg_power(waves):
    power = []
    for wave in waves.values():
      power.append(get_power_wav(wave))
    return np.average(power)

# In: Wave file, avg pwr to normalise to
# Out: Normalised Spectrogram
def normalise_power(wave, avg_pw):
    normalised = ((avg_pw/get_power_wav(wave))**0.5)*wave
    return normalised

# Input: Dictionary of wavs
# Output: Power-normalised dic of waves
def normalise_power_batch(waves):
    avg_pw = get_avg_power(waves)    
    for key in waves.keys():
        waves[key] = normalise_power(waves[key],avg_pw)
    return waves
    
# Generates noise with the same power as the signal
# In: tuple of audio shape, signal power, window length, hop length
# Out: noise with same power as audio, noise power
def generate_noise(signal, wl, hl):
    s_signal = stft(signal)
    noise = nr.random(signal.shape)
    noise = noise - np.average(noise) # Normalize
    s_noise = stft(noise)
    
    p_signal = get_power_spec(s_signal)
    p_noise  = get_power_spec(s_noise)
    
    # Scale waveform and power
    scaled_noise = noise*((p_signal/p_noise)**0.5)
    p_Snoise = p_noise*(p_signal/p_noise)
    return scaled_noise, p_Snoise

# Adds noise to an audio wave.
# In: 1D Audio wave, signal to noise ratio in dB
# Out: 1D Audio wave with noise, same power as original
def add_noise(audio,snr, wl=256, hl=128, DEBUG=0):
    ratio = 10 ** (snr/10)
    spectro = stft(audio)
    p_signal = get_power_spec(spectro)
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
def cutoff_2d_batch(data):
    keys = list(data.keys())
    std_len = min(x_trn[key].shape[1] for key in keys)
    new_data = []
    for key in x_trn:
        start = random.randint(0,int(x_trn[key].shape[1] - std_len))
        sample = x_trn[key]
        cut_sample = sample[:,start:start+std_len]
        new_data.append(cut_sample)
    return np.array(new_data)

# # For each frame, grab the context window
# # In: Dict of normal spectrograms (129, X)
# # Out: Dict of 3D TRANSPOSED spectrograms (X, 2C + 1, 129) where C = context
def batch_get_context_spectro(spectros, context, hop):
  key_list = list(spectros.keys())
  out = {}
  for k in key_list:
    curr_spec = spectros[k]
    t_spec = np.swapaxes(curr_spec,0,1) #Transpose each spectrogram
    cont_lst = get_context_spectro(t_spec, context, hop, RETURN_AS_LIST = True)
    
    ext = 1
    for c in cont_lst:
      new_k = k +'-'+str(ext)
      out[new_k] = c
      ext += 1
      
  return out

#In: Single Transposed Spectro as np array
#Out: Context window of specified frame
def get_context_frame(t_spectro, frame_num, context=5):
    frames = t_spectro.shape[0]        # Num frames
    i = frame_num
    # start padding required
    if i < context:
          num_pad = int(context - i)
          pad = np.tile(t_spectro[0], (num_pad,1))        # Generate padding
          back = np.append(pad,t_spectro[:i+1], axis = 0) # Back context + middle
    else:
        back = t_spectro[i-context:i+1]                 # Back context + middle
    # end padding is required
    if i + context > frames-1:
        num_pad = i + context - frames + 1
        pad = np.tile(t_spectro[frames-1], (num_pad,1)) # Generate padding
        front = np.append(t_spectro[i+1:],pad, axis = 0) # Front context
    else:
        front = t_spectro[i+1:i+1+context] # Front context
    context_win = np.append(back,front,axis=0)
    return context_win

#In: Single Transposed Spectrogram as np array
#Out: np array of context windows w +- 5 frames with hop length 'hop'
def get_context_spectro(t_spectro, context=5, hop=1, RETURN_AS_LIST = True):
  freqs = int(t_spectro.shape[1])    # Frequencies
  frames = t_spectro.shape[0]        # Num frames
  if RETURN_AS_LIST == True:
    chunks = []
    i = 0
    while i < frames: # while index within spectro
      # start padding is required
      if i < context:
          num_pad = int(context - i)
          pad = np.tile(t_spectro[0], (num_pad,1))        # Generate padding
          back = np.append(pad,t_spectro[:i+1], axis = 0) # Back padding + middle
      else:
          back = t_spectro[i-context:i+1]                 # Back padding + middle
      # end padding is required
      if i + context > frames-1:
          num_pad = i + context - frames + 1
          pad = np.tile(t_spectro[frames-1], (num_pad,1)) # Generate padding
          front = np.append(t_spectro[i+1:],pad, axis = 0) # Front padding
      else:
          front = t_spectro[i+1:i+1+context] # Front padding
      chunk = np.append(back,front,axis=0)
      #print("single",chunk.shape,"collection",chunks.shape)
      chunks.append(chunk)
      i+=hop
    return chunks
  elif RETURN_AS_LIST == False: 
    chunks = np.empty((0,2*context+1,freqs)) # Empty 3D numpy array to store context windows
    i = 0
    while i < frames: # while index within spectro
      # start padding is required
      if i < context:
          num_pad = int(context - i)
          pad = np.tile(t_spectro[0], (num_pad,1))        # Generate padding
          back = np.append(pad,t_spectro[:i+1], axis = 0) # Back padding + middle
      else:
          back = t_spectro[i-context:i+1]                 # Back padding + middle

      # end padding is required
      if i + context > frames-1:
          num_pad = i + context - frames + 1
          pad = np.tile(t_spectro[frames-1], (num_pad,1)) # Generate padding
          front = np.append(t_spectro[i+1:],pad, axis = 0) # Front padding
      else:
          front = t_spectro[i+1:i+1+context] # Front padding
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

# In: Dictionary of spectrograms
# Out: Dictionary of transposed spectrograms
def transpose_batch(spectros_dict):
    transposed = {}
    keys = spectros_dict.keys()
    for key in keys:
        transposed[key] = spectros_dict[key].transpose()
    return transposed

def stft_along_axis(data, window_len = 256, hop_len = 128):
    stft = lambda x: lr.stft(x, window_len, hop_len, window_len)
    labels = np.apply_along_axis(stft,1,data)
    return labels
    
#In: Dictionary of spectrograms
#Out: Dictionary of normalised spectrograms, concatenated spectrograms, and array of average and variances
def normalise_spectros(inputs, DEBUG = False):
    # Concatenate to 1 array
    inputs_concat = normalise_concat(inputs, DEBUG) 
    averages = []
    stds = []
    rows = inputs_concat.shape[0]
    
    averages = np.mean(inputs_concat, axis=1)      # Average across all bins
    stds = np.std(inputs_concat, axis=1)   # Variance of all bins

    # Normalize 
    for key in inputs.keys():
      for row in range(rows):
        inputs[key][row] = (inputs[key][row]-averages[row])/stds[row]
      
    return inputs

# #In: Dictionary of spectrograms
# #Out: Numpy arr of concatenated spectrograms
# def normalise_concat(inputs, DEBUG = False):
#   output = np.empty((129,0))
#   count = 0
#   for spectro in inputs.values():
#     output = np.append(output, spectro, axis = 1)
#     count += 1
#     if DEBUG == True:
#         print(str(count) + ' array concatenated')
#   return output

#In: Dictionary of spectrograms
#Out: Numpy arr of concatenated spectrograms
def normalise_concat(spectros, DEBUG = False):
  length = 0
  # Obtain max length of concat
  for key in spectros.keys():
    length += spectros[key].shape[1]
    if DEBUG == True:
      print(str(length)+ 'current length') 

  count = 0
  arrayth = 0
  output = np.empty((129,length))

  # Concat arrays      
  for spectro in spectros.values():
    output[:,count:(count+spectro.shape[1])] = spectro
    count += spectro.shape[1]
    arrayth += 1
    if DEBUG == True:
        print(str(arrayth) + ' array concatenated')
  print(output.shape)
  return output

#In: Dictionary of spectrograms
#Out: Dictionry of lg_pwr spectrograms
def log_pwr_batch(data):
    keys = data.keys()
    for key in keys:
        data[key] = np.log(np.abs(data[key]+1e-32))
    return data
