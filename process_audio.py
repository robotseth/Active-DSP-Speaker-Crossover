import numpy as np
import os
import scipy as sp
from scipy.io.wavfile import read, write
from scipy import signal
import sounddevice as sd
import threading
import time

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
    chunk_array = []
    scaled_data = 0
    (sample_rate, raw_array) = read(file_path) # reads the wave file into an array
    raw_length_ms = 1000 * raw_array.shape[0] / sample_rate
    num_chunks = int(raw_length_ms / chunk_size) + (raw_length_ms % chunk_size > 0)
    print("Length (ms): " + str(raw_length_ms))
    left_data = raw_array[:, 0]
    scaled_data = np.int16(left_data / np.max(np.abs(left_data)) * 32767)
    sub_arrays = np.array_split(scaled_data, num_chunks)
    for i in range(num_chunks):
        chunk = Chunk()
        chunk.data = sub_arrays[i]
        chunk.sample_rate = sample_rate
        chunk.length = raw_length_ms #len(sub_arrays[i])
        chunk_array.append(chunk)
    print("Number of chunks: " + str(len(chunk_array)))
    return chunk_array

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

def export_chunk (chunk, number, name):
    filename = f'chunk_{name}_{number}.wav'
    write_address = os.path.join("C:\\Users\\Seth\\Documents\\school\\EGR334\\exports\\", filename)
    write(write_address, chunk_array[number].sample_rate, chunk)  # Saving it to the file.

def play_chunk (chunk, sample_rate, audio_device):
        sd.play(chunk.data, sample_rate, device=4)
        #stream = sd.OutputStream(samplerate=chunk.sample_rate, device=audio_device, channels=1)
        sd.wait()

def play_chunk_array (chunk_array, sample_rate, audio_device):
    for n in range(len(chunk_array)):
        play_chunk(chunk_array[n], sample_rate, audio_device)

if __name__ == '__main__':
    threads = []  # list to hold threads
    chunk_array = import_wav('C:\\Users\\Seth\\Documents\\school\\EGR334\\audio\\Chicago.wav', 10000)
    high_chunks = []
    band_chunks = []
    low_chunks = []
    for n in range(len(chunk_array)):
        [high_chunk, band_chunk, low_chunk] = filter_chunk(chunk_array[n])
        high_chunks.append(high_chunk)
        band_chunks.append(band_chunk)
        low_chunks.append(low_chunk)
        # export_chunk(high_chunk,n,"high")
        # export_chunk(band_chunk, n,"band")
        # export_chunk(low_chunk, n,"low")
    print(len(high_chunks))
    low = threading.Thread(target=play_chunk_array, args=(low_chunks, chunk_array[n].sample_rate, 4))
    threads.append(low)
    band = threading.Thread(target=play_chunk_array, args=(band_chunks, chunk_array[n].sample_rate, 4))
    threads.append(band)
    high = threading.Thread(target=play_chunk_array, args=(high_chunks, chunk_array[n].sample_rate, 4))
    threads.append(high)
    for thread in threads:
        thread.start()
    for thread in threads:  # wait for all threads to finish
        thread.join()