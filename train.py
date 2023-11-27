import numpy as np
import matplotlib.pyplot as plt
from lib.utils import lowpass_filter, highpass_filter
from scipy.optimize import curve_fit
from lib.models import MLP
from torch.optim import Adam
from torch.nn import MSELoss
import time
import torch
import os 

# load all windows from latest session
# train model
# save model parameters
def load(latest=True):
  cs = os.listdir('out')
  fs = [(os.path.join('out', dir), os.path.getctime(os.path.join('out', dir))) for dir in cs]
  target = sorted(fs,key=lambda x:x[1])[-1 if latest else 0]
  ret = []
  for f in os.listdir(target[0]):
    f = os.path.join(target[0],f)
    arr = np.fromfile(f,dtype=np.int16)
    print(arr)
    ret.append(arr)
  return ret

def main():
  sigs = load()
  print(f'loaded {len(sigs)} total signals with {len(sigs[0])} samples each')
  # get max, min, avg amp data 
  maxs,mins,avgs=[],[],[]
  for sig in sigs:
    print(sig)
    maxs.append(np.max(sig))
    mins.append(np.min(sig))
    avgs.append(np.average(sig))
  print(f'{sum(maxs)/len(maxs):7.2f} max {sum(mins)/len(mins):7.2f} min {sum(avgs)/len(avgs):7.2f} avgs')
  sample = 0
  x,y = list(range(len(sigs[sample]))), sigs[sample]
  y1 = highpass_filter(y)
  y2 = lowpass_filter(y)
  plt.plot(x,y,label='normal')
  plt.plot(x,y1,label='highpass')
  plt.plot(x,y2,label='lowpass')
  plt.legend()
  plt.show()

def log_fit(sig):
  def func(x,a,c):
    return a*np.log(c+x)
  popt,_ = curve_fit(func,list(range(len(sig))),sig)
  return lambda x:func(x,*popt)

def log_fit_test():
  x = np.arange(0,100,1)
  print(x)
  def func(x,a,c):
    return a*np.log(c+x)
  sig = func(x,2,1) + np.random.randn(*x.shape)
  #popt,_ = curve_fit(func,x,sig)
  f = log_fit(sig)
  plt.plot(x,sig)
  plt.plot(x,f(x))
  plt.show()

# map window -> force scalar... 
def test_grip_data(sr,window,overlap):
  grip_freq = 5
  one_grip_tm = 3
  #grip_model = np.log
  # total time = 5 * 3
  # min_amp = 0, max_amp = 1000
  l = []
  for _ in range(grip_freq*one_grip_tm):
    l.append(100*np.log(np.arange(1,one_grip_tm*sr+1,1)))
  x = np.stack(l).flatten().astype(np.float32) 
  x += 30*np.random.randn(*x.shape)
  print(f'x length: {len(x)}')
  chunks = [x[i:i+int(sr/1000*window)] for i in range(0,len(x)-overlap,overlap)]
  forces = [np.array(sum(chunk)/len(chunk),dtype=np.float32) for chunk in chunks]
  print(f'{len(chunks)} chunks of {len(chunks[0])}ms window')
  sig,X,Y = x,chunks,forces 
  return sig,X,Y

if __name__ == '__main__':
  #log_fit_test()
  sr,window,overlap=1000,400,200
  sig,X,Y = test_grip_data(sr,window,overlap)
  #plt.plot(list(range(len(sig))),sig)
  #plt.show()
  net = MLP(window)
  optim = Adam(net.parameters())
  loss_fn = MSELoss()
  epochs = 50
  print(f'trainin for {epochs} epochs')
  for i in range(epochs):
    ts,errs = [],[]
    for x,y in zip(X,Y):
      x,y = torch.from_numpy(x),torch.from_numpy(y)
      optim.zero_grad()
      s = time.perf_counter()
      pred = net(x)
      e = time.perf_counter()
      loss = loss_fn(pred,y)
      loss.backward()
      optim.step()
      errs.append(loss)
      ts.append(e-s)
    print(f'epoch {i}: {sum(errs)/len(errs):.2f} avg err {sum(ts)/len(ts)*1000:7.2f}ms inference tm') 

