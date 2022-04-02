
# coding: utf-8

# In[9]:


import numpy as np
import scipy as sp
from scipy.io.wavfile import read
from scipy.io.wavfile import write     # Imported libaries such as numpy, scipy(read, write), matplotlib.pyplot
from scipy import signal
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')


# In[10]:


# Replace this with the location of your downloaded file.
(Frequency, array) = read('C:\\Users\\Seth\\Documents\\school\\EGR334\\audio\\Lacrimosa.wav') # Reading the sound file.
#(Frequency, array) = read('C:\\Users\\mikef\\Desktop\\_Spring 2022\\EGR 334 T 9am\\Final Project\\5. Harris Heller - Floating Soul.wav') # Reading the sound file.


# In[11]:


len(array) # length of our array
array = array[0:len(array),1]

# In[12]:


plt.plot(array)
plt.title('Original Signal Spectrum')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')


# In[13]:


FourierTransformation = sp.fft.fft(array) # Calculating the fourier transformation of the signal


# In[14]:


scale = np.linspace(0, Frequency, len(array))
print(scale)

# In[15]:


plt.stem(scale[0:5000], np.abs(FourierTransformation[0:5000]))  # The size of our diagram
plt.title('Signal spectrum after FFT')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')


# In[16]:


GuassianNoise = np.random.rand(len(FourierTransformation)) # Adding guassian Noise to the signal.


# In[17]:


NewSound = GuassianNoise + array


# In[18]:


write("New-Sound-Added-With-Guassian-Noise.wav", Frequency, NewSound) # Saving it to the file.


# In[19]:


b,a = signal.butter(5, 1000/(Frequency/2), btype='highpass') # ButterWorth filter 4350


# In[20]:


filteredSignal = signal.lfilter(b,a,NewSound)
plt.plot(filteredSignal) # plotting the signal.
plt.title('Highpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')


# In[21]:


c,d = signal.butter(5, 380/(Frequency/2), btype='lowpass') # ButterWorth low-filter
newFilteredSignal = signal.lfilter(c,d,filteredSignal) # Applying the filter to the signal
plt.plot(newFilteredSignal) # plotting the signal.
plt.title('Lowpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')


# In[22]:


write("New-Filtered-Eagle-Sound.wav", Frequency, newFilteredSignal) # Saving it to the file.

# %%
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
# %%