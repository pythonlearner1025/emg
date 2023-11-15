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
  A) chad "just backprop the whole thing bro" deep learning
    CNN,LSTM,Transformers

  B) beta "meticulously handcraft features" classical analysis
    sliding window - interval & overlap are hyperparameters
    preprocess - db2 mother wavelet transform 
    log regression
'''

def receive():
  # call ls /dev/tty.* 
  port = '/dev/tty.usbmodem1101'
  print(f'connecting to port {port}')
  pipe = Serial(port,9600)
  while not pipe.is_open:
    time.sleep(0.1)
  data = []
  s = -1
  t=1
  while 1:
    # data received should be x-sample long bytearray
    b = pipe.read(size=6).decode().strip()
    if b and s == -1: s = time.perf_counter()
    if not pipe.is_open or b == -1: break
    value = float(b)
    #print('read value: ')
    #print(value)
    data.append(value)
    e = time.perf_counter()
    if e-s > t: break
  # do data processing
  sr = 1000
  window = 100

  if t==1:
    # how to get max hz of device: set delay to 0 in arduino, 
    print(f'analog device has max {len(data)} hz')

  print(f'num samples: {len(data)} at {sr} sr {t} secs')
  print(f'at {window}ms window, expecting {t*sr/window} chunks')
  chunks = [np.array(data[i:i+int(sr/1000*window)]) for i in range(0,len(data),int(sr/1000*window))]
  print(f'actual chunks {len(chunks)}')
  # write chunks 
  pipe.close()
  return chunks

def write(chunks: List[np.ndarray], dir='out'):
  f = 'sesh_'+datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  os.mkdir(os.path.join(dir,f))
  for i,chunk in enumerate(chunks):
    chunk.tofile(os.path.join(f,i))

if __name__ == '__main__':
  receive()