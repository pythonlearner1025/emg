import pywt
import matplotlib.pyplot as plt
import numpy as np

# TODO; preprocessing algo for online/offline
# mother wavelet, high-low band pass filter
if __name__ == '__main__':
  # decomp area CAN be a hyperparam
  # see multilevel decomp: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
  f = np.linspace(0,100,100)
  # apply random oscillating noise function to f
  noise_amplitude = np.random.rand()
  f_noisy = noise_amplitude * np.sin(f)

  approx,detail1,detail2 = pywt.wavedec(f_noisy, 'db1', level=2)

  # graph low, high
  plt.figure(figsize=(18,6))

  plt.subplot(1,3,1)
  plt.plot(approx)
  plt.title("noisy")

  plt.subplot(1,3,2)
  plt.plot(detail1)
  plt.title('approx')

  plt.subplot(1,3,3)
  plt.plot(detail2)
  plt.title("detail2")

  plt.show()
