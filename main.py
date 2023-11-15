from serial import Serial
from datetime import datetime
from lib.pipe import *
import numpy as np
from typing import *
import time
import os

'''
high level goals: 
 - offline training
 - online prediction

offline training pipeline:
  A) chad "just differentiate" deep learning
    >preporcessing? 
    >CNN,LSTM,Transformers

  B) beta "meticulously handcraft features" classical analysis
    >preprocess - db2 mother wavelet transform OR low/high pass
    >log regression
'''
def receive(sr,window,overlap):
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
  assert overlap <= window

  if t==1:
    # how to get max hz of device: set delay to 0 in arduino, 
    print(f'analog device has max {len(data)} hz')

  print(f'num samples: {len(data)} at {sr} sr {t} secs')
  print(f'at {window} ms window, expecting {int((t*sr-overlap)/overlap)} chunks')
  chunks = [np.array(data[i:i+int(sr/1000*window)]) for i in range(0,len(data)-overlap,overlap)]
  print(f'actual chunks {len(chunks)}')
  pipe.close()
  return chunks

def write(chunks: List[np.ndarray], dir='out'):
  print(f'** writing {len(chunks)} chunks **')
  f = os.path.join(dir,'sesh_'+datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
  os.mkdir(f)
  for i,chunk in enumerate(chunks):
    chunk.tofile(os.path.join(f,str(i)))

if __name__ == '__main__':
  sampling_rate,window_size,overlap_size = 1000,100,50
  chunks = receive(sampling_rate,window_size,overlap_size)
  write(chunks)