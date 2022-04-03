import numpy as np
import scipy as sp
from scipy.io.wavfile import read, write
from scipy import signal
import matplotlib.pyplot as plt

#(samplerate, array) = read('C:\\Users\\Seth\\Documents\\school\\EGR334\\audio\\Lacrimosa.wav') # Reading the sound file.
(samplerate, array) = read('C:\\Users\\Seth\\Documents\\school\\EGR334\\audio\\Chicago.wav') # Reading the sound file.

#(Frequency, array) = read('C:\\Users\\mikef\\Desktop\\_Spring 2022\\EGR 334 T 9am\\Final Project\\5. Harris Heller - Floating Soul.wav') # Reading the sound file.

print(f"number of channels = {array.shape[1]}")
length = array.shape[0] / samplerate
print(f"length = {length}s")

time = np.linspace(0., length, array.shape[0])
"""
plt.plot(time, array[:, 0], label="Left channel")
plt.plot(time, array[:, 1], label="Right channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()
"""

# Only uses left channel
left = array[:, 0]
left_scaled = np.int16(left/np.max(np.abs(left)) * 32767)
write("OriginalAudio.wav", samplerate, left_scaled) # Saving it to the file.

"""
plt.plot(left_scaled,'b')
plt.title('Original Signal Spectrum')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
"""

FourierTransformation = sp.fft.fft(left) # Calculating the fourier transformation of the left signal

'''
plt.stem(scale[0:5000], np.abs(FourierTransformation[0:5000]),'r')  # The size of our diagram
plt.title('Signal spectrum after FFT')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
'''

GuassianNoise = np.random.rand(len(FourierTransformation)) # Adding guassian Noise to the signal.

# does not add noise
#NewSound = GuassianNoise + left
NewSound = left


# Scales the sound to to 16 bit
NewSound_scaled = np.int16(NewSound/np.max(np.abs(NewSound)) * 32767)
write("New-Sound-Added-With-Guassian-Noise.wav", samplerate, NewSound_scaled) # Saving it to the file.


##### Highpass Filter #####
b,a = signal.butter(5, 1000/(samplerate/2), btype='highpass') # ButterWorth filter 4350
high_data = signal.lfilter(b,a,NewSound_scaled)

"""
plt.plot(filteredSignal,'g', alpha=0.25) # plotting the signal.
plt.title('Highpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
"""

##### Lowpass Filter #####
c,d = signal.butter(5, 380/(samplerate/2), btype='lowpass') # ButterWorth low-filter
low_data = signal.lfilter(c,d,NewSound_scaled) # Applying the filter to the signal
"""
plt.plot(newFilteredSignal,'k',alpha=0.25) # plotting the signal.
plt.title('Lowpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
"""
##### Bandpass Filter #####
e,f = signal.butter(5, [380/(samplerate/2),1000/(samplerate/2)], btype='band') # ButterWorth low-filter
band_data = signal.lfilter(e,f,NewSound_scaled) # Applying the filter to the signal

# scales all signals to 16 bit
high_scaled = np.int16(high_data/np.max(np.abs(high_data)) * 32767)
low_scaled = np.int16(low_data/np.max(np.abs(low_data)) * 32767)
band_scaled = np.int16(band_data/np.max(np.abs(band_data)) * 32767)



"""
plt.plot(high_scaled,'r',alpha=0.25) # plotting the signal.
plt.title('Lowpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
"""
######### Saves Filtered Files #########
#write("HighSound.wav", samplerate, high_scaled) # Saving it to the file.
#write("LowSound.wav", samplerate, low_scaled) # Saving it to the file.
#write("BandSound.wav", samplerate, band_scaled) # Saving it to the file.

# %%
"""
b, a = signal.butter(4, 100, 'low', analog=True)
w, h = signal.freqs(b, a)

plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()
"""

# %%