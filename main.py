from serial import Serial
from datetime import datetime
import numpy as np
from typing import *
import time
import os

'''
high level goals: 
 - session data collection: associate with 1) time 2) extra conditions
 - visualize session data

- add overlap param

pipeline:
  sliding window - interval & overlap are hyperparameters
  preprocess - db2 mother wavelet transform 
'''
def receive():
  pipe = Serial('/dev/tty.Bluetooth-Incoming-Port',9600,timeout=5)
  time.sleep(1)
  data = []
  while 1:
    # data received should be x-sample long bytearray
    bs = pipe.read().decode().strip()
    if not pipe.is_open() or bs == -1: break
    print('read value: ')
    print(bs)
    data.append(bs)
  # do data processing
  sr = 4000
  window = 100
  chunks = [np.array(chunks[i:i+int(sr/1000*window)]) for i in range(0,len(data),int(sr/1000*window))]
  # write chunks 
  pipe.close()
  return chunks

def write(chunks: List[np.ndarray], dir='out'):
  f = 'sesh_'+datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  os.mkdir(os.path.join(dir,f))
  for i,chunk in enumerate(chunks):
    chunk.tofile(os.path.join(f,i))

if __name__ == '__main__':
  data = list(range(9992))
  sr = 4000
  window = 100
  chunks = [data[i:i+int(sr/1000*window)] for i in range(0,len(data),int(sr/1000*window))]
  print(len(chunks[-1]))
