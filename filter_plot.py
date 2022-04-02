
# coding: utf-8

# In[9]:


import numpy as np
import scipy as sp
from scipy.io.wavfile import read, write
from scipy import signal
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')


# In[10]:


# Replace this with the location of your downloaded file.


#(samplerate, array) = read('C:\\Users\\Seth\\Documents\\school\\EGR334\\audio\\Lacrimosa.wav') # Reading the sound file.
(samplerate, array) = read('C:\\Users\\Seth\\Documents\\school\\EGR334\\audio\\Chicago.wav') # Reading the sound file.


#(Frequency, array) = read('C:\\Users\\mikef\\Desktop\\_Spring 2022\\EGR 334 T 9am\\Final Project\\5. Harris Heller - Floating Soul.wav') # Reading the sound file.


# In[11]:
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
left = array[:, 0]
left_scaled = np.int16(left/np.max(np.abs(left)) * 32767)
write("OriginalAudio.wav", samplerate, left_scaled) # Saving it to the file.

# In[12]:


#### always at zero, why? ####

plt.plot(left_scaled,'b')
plt.title('Original Signal Spectrum')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')


# In[13]:


#len(array) # length of our array
#left = array[0:len(left),1]


FourierTransformation = sp.fft.fft(left) # Calculating the fourier transformation of the left signal


# In[14]:


#scale = np.linspace(0, samplerate, left)
#print(scale)

# In[15]:
'''
plt.stem(scale[0:5000], np.abs(FourierTransformation[0:5000]),'r')  # The size of our diagram
plt.title('Signal spectrum after FFT')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
'''
# In[16]:


GuassianNoise = np.random.rand(len(FourierTransformation)) # Adding guassian Noise to the signal.


# In[17]:


#NewSound = GuassianNoise + left
NewSound = left


# In[18]:
NewSound_scaled = np.int16(NewSound/np.max(np.abs(NewSound)) * 32767)
write("New-Sound-Added-With-Guassian-Noise.wav", samplerate, NewSound_scaled) # Saving it to the file.


# In[19]:


b,a = signal.butter(5, 1000/(samplerate/2), btype='highpass') # ButterWorth filter 4350


# In[20]:


filteredSignal = signal.lfilter(b,a,NewSound_scaled)
"""
plt.plot(filteredSignal,'g', alpha=0.25) # plotting the signal.
plt.title('Highpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
"""
# In[21]:


c,d = signal.butter(5, 380/(samplerate/2), btype='lowpass') # ButterWorth low-filter
newFilteredSignal = signal.lfilter(c,d,filteredSignal) # Applying the filter to the signal
"""
plt.plot(newFilteredSignal,'k',alpha=0.25) # plotting the signal.
plt.title('Lowpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
"""

# In[22]:
filteredSignal_scaled = np.int16(filteredSignal/np.max(np.abs(filteredSignal)) * 32767)
newFilteredSignal_scaled = np.int16(newFilteredSignal/np.max(np.abs(newFilteredSignal)) * 32767)

plt.plot(newFilteredSignal_scaled,'r',alpha=0.25) # plotting the signal.
plt.title('Lowpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')

write("Filtered-Eagle-Sound.wav", samplerate, filteredSignal_scaled) # Saving it to the file.

# %%
b, a = signal.butter(4, 100, 'low', analog=True)
w, h = signal.freqs(b, a)
"""
plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency

"""
plt.show()

# %%