import numpy as np
import scipy as sp
from scipy.io.wavfile import read, write
from scipy import signal

# create chunk object from input (sensor reading or audio file)
# filter chunk and export 3 new chunks (low, mid, high)
# play each chunk on a different audio device
# start playing new chunk when the current one ends

class Chunk:
    'Common base class for Chunks'
    chunk_count = 0

    def __init__(self, name, length, sample_rate, data):
        self.name = name
        self.length = length
        self.sample_rate = sample_rate
        self.data = data
        Chunk.chunk_count += 1

    def display_count(self):
        print
        "Total Chunks %d" % Chunk.chunk_count

#(samplerate, array) = read('C:\\Users\\Seth\\Documents\\school\\EGR334\\audio\\Lacrimosa.wav') # Reading the sound file.
(samplerate, array) = read('C:\\Users\\Seth\\Documents\\school\\EGR334\\audio\\Chicago.wav') # Reading the sound file.
#(Frequency, array) = read('C:\\Users\\mikef\\Desktop\\_Spring 2022\\EGR 334 T 9am\\Final Project\\5. Harris Heller - Floating Soul.wav') # Reading the sound file.

# split audio file into chunks
# create array of chunks

print(f"number of channels = {array.shape[1]}")
length = array.shape[0] / samplerate
print(f"length = {length}s")

time = np.linspace(0., length, array.shape[0])

# Only uses left channel
left = array[:, 0]
left_scaled = np.int16(left/np.max(np.abs(left)) * 32767)
write("OriginalAudio.wav", samplerate, left_scaled) # Saving it to the file.

FourierTransformation = sp.fft.fft(left) # Calculating the fourier transformation of the left signal

GuassianNoise = np.random.rand(len(FourierTransformation)) # Adding guassian Noise to the signal.

NewSound = left

# Scales the sound to to 16 bit
NewSound_scaled = np.int16(NewSound/np.max(np.abs(NewSound)) * 32767)
write("New-Sound-Added-With-Guassian-Noise.wav", samplerate, NewSound_scaled) # Saving it to the file.

##### Highpass Filter #####
b,a = signal.butter(5, 1000/(samplerate/2), btype='highpass') # ButterWorth filter 4350
high_data = signal.lfilter(b,a,NewSound_scaled)



##### Lowpass Filter #####
c,d = signal.butter(5, 380/(samplerate/2), btype='lowpass') # ButterWorth low-filter
low_data = signal.lfilter(c,d,NewSound_scaled) # Applying the filter to the signal

##### Bandpass Filter #####
e,f = signal.butter(5, [380/(samplerate/2),1000/(samplerate/2)], btype='band') # ButterWorth low-filter
band_data = signal.lfilter(e,f,NewSound_scaled) # Applying the filter to the signal

# scales all signals to 16 bit
high_scaled = np.int16(high_data/np.max(np.abs(high_data)) * 32767)
low_scaled = np.int16(low_data/np.max(np.abs(low_data)) * 32767)
band_scaled = np.int16(band_data/np.max(np.abs(band_data)) * 32767)



######### Saves Filtered Files #########
#write("HighSound.wav", samplerate, high_scaled) # Saving it to the file.
#write("LowSound.wav", samplerate, low_scaled) # Saving it to the file.
#write("BandSound.wav", samplerate, band_scaled) # Saving it to the file.