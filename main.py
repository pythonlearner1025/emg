from serial import Serial
from datetime import datetime
import socket, json
from lib.pipe import *
import numpy as np
from typing import *
from pylsl import StreamInlet, resolve_stream
import time
import os

'''
vs. TEMG paper
- 12 double-diff electrodes used
- sr 2000hz
- 200ms window, so 400 samples
- 190ms overlap (10ms step)
- windows obtained during 5s classification period, 3s rest

vs. current cyton
- 1 dd electrodes
- sr 250hz 
- 500ms window, at 125 samples
- 250ms overlap
- regression to grip force
'''
def cyton_receive_lsl(window,overlap,channels=[0],sr=250):
  samples_thresh = sr//(1000//window)
  overlap_thresh = sr//(1000//overlap)
  streams = resolve_stream('type', 'EMG')
  inlet = StreamInlet(streams[0])  
  buff = []
  s = time.perf_counter()
  print(f'{samples_thresh} samples / {window}ms window')
  print(f'{overlap_thresh} samples / {overlap}ms overlap')
  while 1:
    chunk, _ = inlet.pull_chunk()
    if chunk:
       chunk = np.array(chunk)[:,channels].flatten()
       buff.extend(chunk.tolist())
       if len(buff) > samples_thresh:
          chunk = buff[:samples_thresh]
          buff = buff[-overlap_thresh:]
          # process chunk
          #print(len(chunk))
          #print(len(buff))
          assert len(chunk) == samples_thresh
          assert len(buff) == overlap_thresh
          e = time.perf_counter()
          print(f'{e-s:7.2f}')
          s = time.perf_counter()
          
def cyton_test_lsl():
  streams = resolve_stream('type', 'EMG')
  inlet = StreamInlet(streams[0])
  start = time.time()
  totalNumSamples = 0
  validSamples = 0
  numChunks = 0
  print( "Testing Sampling Rates..." )
  duration = 10
  while time.time() <= start + duration:
      # get chunks of samples
      chunk, timestamp = inlet.pull_chunk()
      if chunk:
          numChunks += 1
          totalNumSamples += len(chunk)
          i = 0
          for sample in chunk:
              print(sample, timestamp[i])
              validSamples += 1
              i += 1
  print( "Number of Chunks and Samples == {} , {}".format(numChunks, totalNumSamples) )
  print( "Valid Samples and Duration == {} / {}".format(validSamples, duration) )
  print( "Avg Sampling Rate == {}".format(validSamples / duration) )
  
def receive(sr,window,overlap,t=1):
  # call ls /dev/tty.* 
  port = '/dev/tty.usbmodem1101'
  print(f'connecting to port {port}')
  pipe = Serial(port,9600)
  data = []
  s=-1
  seen = 0
  while 1:
    b = pipe.read(size=4).decode().strip()
    if b and s == -1: s = time.perf_counter()
    if not pipe.is_open or b == -1: break
    value = int(b)
    print(value)
    data.append(value)
    e = time.perf_counter()
    if e-s > t: break
    if len(data) > seen*overlap+window:
      frame = data[seen*overlap:seen*overlap+window]
      # do processing here
  assert overlap <= window
  if t==1:
    # how to get max hz of device: set delay to 0 in arduino, 
    print(f'analog device has max {len(data)} hz')

  print(f'num samples: {len(data)} at {sr} sr {t} secs')
  print(f'at {window} ms window, expecting {int((t*sr-overlap)/overlap)} chunks')
  chunks = [np.array(data[i:i+int(sr/1000*window)],dtype=np.int16) for i in range(0,len(data)-overlap,overlap)]
  print(f'actual chunks {len(chunks)}')
  pipe.close()
  return chunks

def write(chunks: List[np.ndarray], dir='out'):
  print(f'** writing {len(chunks)} chunks **')
  f = os.path.join(dir,'sesh_'+datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
  os.mkdir(f)
  for i,chunk in enumerate(chunks):
    chunk.tofile(os.path.join(f,str(i)))

# udp limited to 25packet/s with packet size > 320...
def cyton_test_udp():
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  server_addr = ('localhost', 12345)
  sock.bind(server_addr)
  s = None
  c = 0
  while 1:
    data,_ = sock.recvfrom(4096)
    msg = data.decode('utf-8')
    try:
      json_data = json.loads(msg)
      if json_data.get('type') == 'timeSeriesFilt':
          # get from channel 0
          time_series_data = json_data.get('data')
          channel = np.abs(np.array(time_series_data[0]))
          avg = channel.sum()/len(channel)
          print(f'{avg:7.2f} uV avg')
          if not s: s = time.perf_counter()
          c += 1
      else:
          print("Received non-time-series data")
    except json.JSONDecodeError:
        print("Error decoding JSON")
    e = time.perf_counter()
    if s and e-s > 10:
      print(f'{c/(e-s)} frames per second')
 
if __name__ == '__main__':
  t = 0.1
  sampling_rate,window_size,overlap_size = 1000,100,50
  '''
  chunks = receive(
    sampling_rate,
    window_size,
    overlap_size,
    t=t
    )
  write(chunks)
  '''
  cyton_receive_lsl(500,250)