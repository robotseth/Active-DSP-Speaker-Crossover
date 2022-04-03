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

    def __init__(self, length=None, sample_rate=None, data=None):
        self.length = length
        self.sample_rate = sample_rate
        self.data = data
        Chunk.chunk_count += 1

    def display_count(self):
        print
        "Total Chunks %d" % Chunk.chunk_count

#(samplerate, array) = read('C:\\Users\\Seth\\Documents\\school\\EGR334\\audio\\Lacrimosa.wav') # Reading the sound file.
#(samplerate, raw_array) = read('C:\\Users\\Seth\\Documents\\school\\EGR334\\audio\\Chicago.wav') # Reading the sound file.
#(Frequency, array) = read('C:\\Users\\mikef\\Desktop\\_Spring 2022\\EGR 334 T 9am\\Final Project\\5. Harris Heller - Floating Soul.wav') # Reading the sound file.

# split audio file into chunks
def import_wav (file_path, chunk_size):
    global chunk_array
    scaled_data = 0
    (sample_rate, raw_array) = read(file_path) # reads the wave file into an array
    raw_length_ms = 1000 * raw_array.shape[0] / sample_rate
    num_chunks = int(raw_length_ms / chunk_size) + (raw_length_ms % chunk_size > 0)
    print(raw_length_ms)
    print(num_chunks)
    left_data = raw_array[:, 0]
    scaled_data = np.int16(left_data / np.max(np.abs(left_data)) * 32767)
    sub_arrays = np.array_split(scaled_data, num_chunks)
    #print(sub_arrays)
    for i in range(num_chunks):
        chunk = Chunk()
        chunk.data = sub_arrays[i]
        chunk.sample_rate = sample_rate
        chunk.length = len(sub_arrays[i])
        chunk_array.append(chunk)
    print(len(chunk_array))
    #print(chunk_array[0].data)
# create array of chunks

def filter (chunk, type):
    if type == "highpass" or type == "h" :
        b, a = signal.butter(5, 1000 / (chunk.sample_rate / 2), btype='highpass')  # ButterWorth filter 4350
    elif type == "lowpass" or type == "l" :
        b, a = signal.butter(5, 380 / (chunk.sample_rate / 2), btype='lowpass')  # ButterWorth low-filter
    elif type == "bandpass" or type == "b" :
        b, a = signal.butter(5, [380 / (chunk.sample_rate / 2), 1000 / (chunk.sample_rate / 2)], btype='band')  # ButterWorth low-filter
    else:
        print("Error - not valid filter type")
    data = signal.lfilter(b, a, chunk.data)
    data = np.int16(data / np.max(np.abs(data)) * 32767)
    return data

def filter_chunk (chunk):
    high_chunk = filter(chunk,'h')
    band_chunk = filter(chunk,'b')
    low_chunk = filter(chunk,'l')
    filtered_chunks = [high_chunk, band_chunk, low_chunk]
    return filtered_chunks

if __name__ == '__main__':
    chunk_array = []
    import_wav('C:\\Users\\Seth\\Documents\\school\\EGR334\\audio\\Chicago.wav', 10000)
    """
    for n in range(len(chunk_array)):
        filter(chunk_array[n], "h")
    """
    #print(chunk_array[0].data)
    [high_chunk, band_chunk, low_chunk] = filter_chunk(chunk_array[0])
    write("FirstChunkAudio.wav", chunk_array[0].sample_rate, low_chunk)  # Saving it to the file.
"""
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
"""