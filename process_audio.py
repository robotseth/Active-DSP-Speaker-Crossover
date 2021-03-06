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
global_sample_rate = 44100
input_device = 1
output_device_low = 6
output_device_band = 6
output_device_high = 6

# defines a Chunk class for generating chunk objects
# each Chunk is a section of audio and stores the audio data, sample rate, and chunk duration
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

# Imports a wave file and generates an array of Chunk objects with the data from the file
# the data is stored as numpy arrays
def import_wav (file_path, chunk_size):
    chunk_array = [] # defines an empty array to append Chunks to
    (sample_rate, raw_array) = read(file_path) # reads the wave file into an array
    raw_length_ms = 1000 * raw_array.shape[0] / sample_rate # calculates the length of the audio file
    num_chunks = int(raw_length_ms / chunk_size) + (raw_length_ms % chunk_size > 0) # divides the audio file length by the desired chunk duration and rounds up
    print("Length (ms): " + str(raw_length_ms)) # prints the length of the audio file
    left_data = raw_array[:, 0] # saves the left channel of the audio data
    scaled_data = np.int16(left_data / np.max(np.abs(left_data)) * 32767) # scales the data to fit at a 16 bit wav file
    sub_arrays = np.array_split(scaled_data, num_chunks) # splits the data into Chunks
    for i in range(num_chunks): # assigns data to the generated Chunks
        chunk = Chunk()
        chunk.data = sub_arrays[i]
        chunk.sample_rate = sample_rate
        chunk.length = raw_length_ms
        chunk_array.append(chunk)
    print("Number of chunks: " + str(len(chunk_array)))
    return chunk_array # returns an array of generated Chunks

# filters a Chunk and returns a filtered Chunk
def filter (chunk, type, gain):
    if type == "highpass" or type == "h" :
        b, a = signal.butter(5, 1000 / (chunk.sample_rate / 2), btype='highpass')  # defines a butterworth high-pass filter
    elif type == "lowpass" or type == "l" :
        b, a = signal.butter(5, 380 / (chunk.sample_rate / 2), btype='lowpass')  # defines a butterworth low-pass filter
    elif type == "bandpass" or type == "b" :
        b, a = signal.butter(5, [380 / (chunk.sample_rate / 2), 1000 / (chunk.sample_rate / 2)], btype='band')  # defines a butterworth band-pass filter
    else:
        print("Error - not valid filter type")
    data = signal.lfilter(b, a, chunk.data) # applies the defined filter to the data
    data = np.int16(data / np.max(np.abs(data)) * 32767) * gain # scales the data to fit within the 16 bit wav file size
    filtered_chunk = Chunk(chunk.length, chunk.sample_rate, data)
    return filtered_chunk # returns filtered chunk


def filter_chunk (chunk, gain):
    h_chunk = filter(chunk,'h', gain)
    b_chunk = filter(chunk,'b', gain)
    l_chunk = filter(chunk,'l', gain)
    filtered_chunks = [h_chunk, b_chunk, l_chunk]
    return filtered_chunks

def input_stream ():
    global input_device
    stream = sd.InputStream(device=input_device, channels=1, callback=callback_in, blocksize=4410, dtype=np.int16, samplerate=global_sample_rate)
    stream.start()

def callback_in (indata, frames, time, status):
    print(type(indata.copy()))
    print(indata.copy())
    global global_sample_rate
    chunk = Chunk()
    #print(indata.T[0])
    #print(type(indata))
    chunk.data = indata.copy().T[0]
    chunk.sample_rate = global_sample_rate
    #print(type(chunk.data))
    #load_output_buffer([chunk, chunk, chunk])
    [h_chunk, b_chunk, l_chunk] = filter_chunk(chunk, 1)
    load_output_buffer([h_chunk, b_chunk, l_chunk])

def load_output_buffer (chunk_array):
    global out_buffer_low
    global out_buffer_band
    global out_buffer_high
    #print(len(out_buffer_low))
    out_buffer_low.append(chunk_array[2])
    out_buffer_band.append(chunk_array[1])
    out_buffer_high.append(chunk_array[0])
    #print(len(out_buffer_low))


def export_chunk (chunk, number, name):
    global global_sample_rate
    filename = f'chunk_{name}_{number}.wav'
    #write_address = os.path.join("C:\\Users\\Seth\\Documents\\school\\EGR334\\exports\\", filename)
    write_address = os.path.join("C:\\Users\\Seth Altobelli\\Documents\\school\\EGR334\\EGR334\\discord\\", filename)
    chunk.data = np.int16(chunk.data / np.max(np.abs(chunk.data)) * 32767)
    write(write_address, global_sample_rate, chunk.data)  # Saving it to the file.


def join_chunks (chunks):
    global global_sample_rate
    data_array = []
    big_boy_chunk = Chunk()
    print(len(chunks))
    for i in range(len(chunks)):
        data_array = np.append(data_array, chunks[i].data)
        print(len(data_array))
    big_boy_chunk.data = data_array
    big_boy_chunk.sample_rate = global_sample_rate
    return big_boy_chunk


# Old function for playing chunks individually
def play_chunk (chunk, audio_device):
    sd.play(chunk.data, chunk.sample_rate, device=audio_device)
    sd.wait()

# Main playback function that runs on individual audio output threads. Takes a single argument that starts the sounddevice
# stream with the appropriate callback function and output device for the selected filter.
def stream_chunk (filter_type):
    global output_device_low
    global output_device_band
    global output_device_high

    # Select which callback function and output device to use.
    # The sounddevice OutputStream takes in a handle for a callback function which is responsible for filling the
    # output buffer array, and is called whenever the output buffer is empty.
    if filter_type == 'l':
        stream = sd.OutputStream(device=output_device_low, channels=1, callback=callback_low, blocksize=4410, dtype=np.int16, samplerate=global_sample_rate) #samplerate=out_buffer_low[0].sample_rate
    elif filter_type == 'b':
        stream = sd.OutputStream(device=output_device_band, channels=1, callback=callback_band, blocksize=4410, dtype=np.int16, samplerate=global_sample_rate)
    elif filter_type == 'h':
        stream = sd.OutputStream(device=output_device_high, channels=1, callback=callback_high, blocksize=4410, dtype=np.int16, samplerate=global_sample_rate)

    stream.start()

# Old function that plays an entire array of chunks in series
def play_chunk_array (chunk_array, audio_device):
    for n in range(len(chunk_array)):
        play_chunk(chunk_array[n], audio_device)

# Old function
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
    #print(type(out_buffer_low[0].data))
    print(out_buffer_low[0].data)
    #out_buffer_low[0].data = np.int16(out_buffer_low[0].data / np.max(np.abs(out_buffer_low[0].data)) * 32767)
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
    chunk_0 = Chunk()
    chunk_0.data = np.ones(4410)
    out_buffer_low = [chunk_0] * 1
    out_buffer_band = [chunk_0] * 1
    out_buffer_high = [chunk_0] * 1
    print(out_buffer_low)
    threads = []  # list to hold threads
    chunk_array = import_wav('C:\\Users\\Seth Altobelli\\Documents\\school\\EGR334\\EGR334\\audio\\Lacrimosa.wav', 100)

    for n in range(len(chunk_array)):
        [high_chunk, band_chunk, low_chunk] = filter_chunk(chunk_array[n], .5)
        out_buffer_high.append(high_chunk)
        out_buffer_band.append(band_chunk)
        out_buffer_low.append(low_chunk)
        # export_chunk(high_chunk,n,"high")
        # export_chunk(band_chunk, n,"band")
        # export_chunk(low_chunk, n,"low")
    #print(out_buffer_high[200].data)


    input = threading.Thread(target=input_stream, args=(), daemon=True)
    #threads.append(input)
    low = threading.Thread(target=stream_chunk, args=('l'), daemon=True)
    #threads.append(low)
    band = threading.Thread(target=stream_chunk, args=('b'), daemon=True)
    #threads.append(band)
    high = threading.Thread(target=stream_chunk, args=('h'), daemon=True)
    threads.append(high)
    for thread in threads:
        thread.start()
    for thread in threads:  # wait for all threads to finish
        thread.join()
    time.sleep(100)

    #export_chunk(join_chunks(out_buffer_low),0,"test")


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
