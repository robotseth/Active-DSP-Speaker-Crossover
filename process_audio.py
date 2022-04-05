import numpy as np
import os
import scipy as sp
from scipy.io.wavfile import read, write
from scipy import signal
import sounddevice as sd
import threading
import time

out_buffer_low = []
out_buffer_band = []
out_buffer_high = []

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



""""############### NOTES / TO DO ###############"""
# MAIN LOGIC:
# each thread should start with a loop that constantly checks if new data has been added to an array of chunks.
# If it has, it should play the new chunks until it is out of data at which point it loops.
# The main loop should process an audio file for testing and will read sensor data in the real thing

# program a function that allows the playback threads to only play up until a certain number of chunks back so that a
# delay can be added for the live sensor reading

# look into not using play and trying stream or something so that there are not hickups on playback.
# Can you append to the playback array mid-playback?



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
    filtered_chunk = Chunk(chunk.length, chunk.sample_rate, data)
    return filtered_chunk

def filter_chunk (chunk):
    h_chunk = filter(chunk,'h')
    b_chunk = filter(chunk,'b')
    l_chunk = filter(chunk,'l')
    filtered_chunks = [h_chunk, b_chunk, l_chunk]
    return filtered_chunks


def input_stream ():
    pass

def load_output_buffer ():

    pass


def export_chunk (chunk, number, name):
    filename = f'chunk_{name}_{number}.wav'
    write_address = os.path.join("C:\\Users\\Seth\\Documents\\school\\EGR334\\exports\\", filename)
    write(write_address, chunk_array[number].sample_rate, chunk)  # Saving it to the file.


def play_chunk (chunk, audio_device):
    sd.play(chunk.data, chunk.sample_rate, device=audio_device)
    sd.wait()


def stream_chunk (audio_device, filter_type):

    if filter_type == 'low':
        stream = sd.OutputStream(device=audio_device, channels=1, callback=callback_low, blocksize=441)
    elif filter_type == 'band':
        stream = sd.OutputStream(device=audio_device, channels=1, callback=callback_band, blocksize=441)
    elif filter_type == 'high':
        stream = sd.OutputStream(device=audio_device, channels=1, callback=callback_high, blocksize=441)

    stream.start()


def play_chunk_array (chunk_array, audio_device):
    for n in range(len(chunk_array)):
        play_chunk(chunk_array[n], audio_device)


def async_play_chunks (chunk_array_index, audio_device):
    i = 1
    global global_chunk_array
    global lock
    while True:
        if not lock:
            rows, columns = global_chunk_array.shape
            #print("Num rows: " + str(rows))
            if i < rows:
                print("Num rows: " + str(rows))
                print("I is: " + str(i))
                play_chunk(global_chunk_array[chunk_array_index][i], audio_device)
                i += 1

def callback_low(outdata, frames, time, status):
    global out_buffer_low
    outdata[:] = out_buffer_low[0].data.reshape(len(out_buffer_low[0].data),1)
    out_buffer_low.pop(0)

def callback_band(outdata, frames, time, status):
    global out_buffer_band
    outdata[:] = out_buffer_band[0].data.reshape(len(out_buffer_band[0].data),1)
    out_buffer_band.pop(0)

def callback_high(outdata, frames, time, status):
    global out_buffer_high
    outdata[:] = out_buffer_high[0].data.reshape(len(out_buffer_high[0].data),1)
    out_buffer_high.pop(0)

if __name__ == '__main__':
    threads = []  # list to hold threads
    chunk_array = import_wav('C:\\Users\\Seth Altobelli\\Documents\\school\\EGR334\\EGR334\\input_wav.wav', 10)
    for n in range(len(chunk_array)):
        [high_chunk, band_chunk, low_chunk] = filter_chunk(chunk_array[n])
        out_buffer_high.append(high_chunk)
        out_buffer_band.append(band_chunk)
        out_buffer_low.append(low_chunk)
        # export_chunk(high_chunk,n,"high")
        # export_chunk(band_chunk, n,"band")
        # export_chunk(low_chunk, n,"low")
    print(out_buffer_high[200].data)
    low = threading.Thread(target=stream_chunk, args=(6, 'low'), daemon=True)
    threads.append(low)
    band = threading.Thread(target=stream_chunk, args=(6, 'band'), daemon=True)
    #threads.append(band)
    high = threading.Thread(target=stream_chunk, args=(6, 'high'), daemon=True)
    #threads.append(high)
    for thread in threads:
        thread.start()
    for thread in threads:  # wait for all threads to finish
        thread.join()
    time.sleep(100)


"""
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
    low = threading.Thread(target=play_chunk_array, args=(low_chunks, 4))
    threads.append(low)
    band = threading.Thread(target=play_chunk_array, args=(band_chunks, 4))
    threads.append(band)
    high = threading.Thread(target=play_chunk_array, args=(high_chunks, 4))
    threads.append(high)
    for thread in threads:
        thread.start()
    for thread in threads:  # wait for all threads to finish
        thread.join()
"""
