import pywt
import matplotlib.pyplot as plt
import numpy as np
import time

# >500hz
def lowpass_filter(signal,lvl=2):
  coeffs = pywt.wavedec(signal,'db2',level=lvl)
  coeffs[1:] = [np.zeros_like(d) for d in coeffs[1:]]
  return pywt.waverec(coeffs,'db2')

# <10hz
def highpass_filter(signal,lvl=2):
  # lowpass_hz_range = signal_hz/(2*5)
  coeffs = pywt.wavedec(signal,'db2',level=lvl)
  coeffs[0] = np.zeros_like(coeffs[0])
  return pywt.waverec(coeffs,'db2')

# features
def mav(x): return np.sum(np.abs(x))/len(x)

def ssi(x): return np.sum(np.square(x))

def rms(x): return np.sqrt(mav(x))

def var(x): return ssi(x)/(len(x)-1)

# etc...

# TODO; preprocessing algo for online/offline
# mother wavelet, high-low band pass filter
if __name__ == '__main__':
  # decomp area CAN be a hyperparam
  # see multilevel decomp: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
  x = np.linspace(0, 1, num=2048)
  sig = np.sin(250 * np.pi * x**2)
  # apply random oscillating noise function to f
  s = time.perf_counter()
  #recon = highpass_filter(sig)
  recon = lowpass_filter(sig)
  e = time.perf_counter()

  plt.plot(x, sig, label='original')
  plt.plot(x, recon, label='recon')
  plt.legend()
  plt.show()
