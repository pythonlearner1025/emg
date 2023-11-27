import torch
from torch.utils.data import random_split
from torch.optim import Adam
from torch.nn import MSELoss
from lib.models import TEMG
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# TODO: 
# - write script to download dataset from putEMG

def train(file):
  data = np.load(file)
  train_size = int(0.8 * len(data))  # 80% of data for training
  val_size = len(data) - train_size  # Remaining 20% for validation
  train_dataset, val_dataset = random_split(data, [train_size, val_size])

  net = TEMG(64,layers=2,mlp_d=256)
  loss = MSELoss()
  optim = Adam(net.parameters())

  epochs = 50
  train_loss,val_loss = [],[]
  for i in range(epochs):
    ts = []
    for X,Y in tqdm(t:=iterator(np.array(train_dataset), BS=4, shuffle=True)):
      optim.zero_grad()
      X,Y = torch.tensor(X,dtype=torch.float32,requires_grad=True), torch.tensor(Y[:,0],dtype=torch.float32)
      o = net(X).squeeze()
      l = loss(o,Y)
      l.backward()
      optim.step()
      t.append(l.item())
      train_loss.append(l.item())

    if i % 4 == 0:
      vals = []
      for X,Y in tqdm(t:=iterator(np.array(val_dataset), BS=4, shuffle=False)):
        X,Y = torch.tensor(X,dtype=torch.float32,requires_grad=True), torch.tensor(Y[:,0],dtype=torch.float32)
        o = net(X).squeeze()
        l = loss(o,Y)
        vals.append(l.item())
      print(f'Epoch {i} | {sum(vals)/len(vals):7.2f} val loss')
      val_loss.append(sum(vals)/len(vals))
    print(f'Epoch {i} | {sum(ts)/len(ts):7.2f} train loss')

  plt.plot(list(range(len(train_loss))), train_loss, label='train loss')
  plt.plot(list(range(len(val_loss))), val_loss, label='val loss')
  plt.legend()
  plt.ylabel('MSE Loss')
  plt.show()

def iterator(data,BS=4,shuffle=True):
  B,C,features = data.shape
  X,Y = data[:,:,:-2],data[:,:,-2:]
  X,Y = X[:,:-(C%(features-2)),:],Y[:,:-(C%(features-2))]
  C = (C//(features-2))
  X = X.reshape(B,C,features-2*features-2)
  Y = Y.reshape(B,C,features-2,2)[:,:,0,:]
  assert X.shape[1] == Y.shape[1]

  if shuffle:
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X,Y = X[indices],Y[indices]

  for i in range(0, X.shape[0], BS):
    X_batch,Y_batch = X[i:i+BS],Y[i:i+BS]
    Y_batch = Y_batch[:,:,0]/Y_batch[:,:,1]
    Y_batch[Y_batch<0] = 0
    yield X_batch, Y_batch

def load_putemg_force(in_f,out_f,sr=1280,window=500,overlap=250,avg_last_n=10,adc_bit=10,gain=200):
  fs = [os.path.join(in_f,x) for x in os.listdir(in_f) if 'mvc' in x]

  def adc2uV(x):
    return x/(2**12)*(adc_bit**6/gain)

  # return N total chunks where N = total_samps // samples_per_window
  def chunk(buff):
    # return 2d list of size (samps_per_window,features)
    def listify(chunk):
      fs = [l['force'] for l in chunk[-avg_last_n:]]
      f = sum(fs)/len(fs)
      for x in chunk:
        x['force'] = f
      ret = []
      for c in chunk:
        l = list(c.values())
        l = l[2:] + l[:2]
        ret.append(l)
      return ret
    samps_per_window = int(window*(sr//1000))
    samps_per_overlap = int(overlap*(sr//1000))
    return [listify(buff[i:i+samps_per_window]) for i in range(0,len(buff),samps_per_overlap) 
            if i+samps_per_window<len(buff)]
  # middle band
  emgs = {f'EMG_{i}' for i in list(range(9,17))}

  all = []
  for j,f in enumerate(tqdm(t:=fs)):
    frame = pd.read_hdf(f)
    newrows = []

    for i in range(0,len(frame),5120//1280):
      if i >= len(frame): break
      newrow = dict()
      row = frame.iloc[i].to_dict()
      emgs = {x:adc2uV(row[x]) for x in row.keys() if x in emgs}
      newrow['force'] = row['FORCE_1'] + row['FORCE_2']
      newrow['mvc'] = row['FORCE_MVC']
      newrows.append({**newrow,**emgs})

    all.append(np.array(chunk(newrows)))
  res = np.vstack(all)
  np.save(out_f,res)

'''
putEMG dataset info
  - force data was sampled at 5120 Hz, with 12-bit A/D conversion 
  using 3 Hz high-pass and 900 Hz low-pass filter. 
  - All signals were pre-amplified with gain of 5, using amplifiers 
  placed on subject’s arm, resulting in total gain of 200. 
'''
if __name__ == '__main__':
  folder = 'putEMG/Data-HDF5'
  file = 'out/force.npy'
  load_putemg_force(folder,file,sr=1280,window=500,overlap=250,avg_last_n=10,adc_bit=10,gain=200)
  train(file)