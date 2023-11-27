import torch
from torch.utils.data import random_split
from torch.optim import Adam
from torch.nn import MSELoss
from lib.models import TEMG
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

def train():
  # print columns
  file = 'out/force.npy'
  # Splitting the dataset
  data = np.load(file)
  train_size = int(0.8 * len(data))  # 80% of data for training
  val_size = len(data) - train_size  # Remaining 20% for validation
  train_dataset, val_dataset = random_split(data, [train_size, val_size])

  net = TEMG(64,layers=2,mlp_d=256)
  loss = MSELoss()
  optim = Adam(net.parameters())

  # Creating DataLoaders for each set
  #train_iterator = iterator(np.array(train_dataset), BS=4, shuffle=True)

  epochs = 100
  for i in range(epochs):
    losses = []
    for X,Y in tqdm(t:=iterator(np.array(train_dataset), BS=4, shuffle=True)):
      optim.zero_grad()
      X,Y = torch.tensor(X,dtype=torch.float32,requires_grad=True), torch.tensor(Y[:,0],dtype=torch.float32)
      o = net(X).squeeze()
      l = loss(o,Y)
     # print(o)
     # print(Y)
      l.backward()
      optim.step()
      losses.append(l.item())

    if i % 4 == 0:
      ls = []
      for X,Y in tqdm(t:=iterator(np.array(val_dataset), BS=4, shuffle=False)):
        X,Y = torch.tensor(X,dtype=torch.float32,requires_grad=True), torch.tensor(Y[:,0],dtype=torch.float32)
        o = net(X).squeeze()
        l = loss(o,Y)
        ls.append(l.item())
      print(f'EPOCH{i}: {sum(ls)/len(ls):7.2f} avg val loss')

    #print(f'EPOCH {i}, {sum(losses)/len(losses):7.2f} avg loss')

'''
The data was sampled at 5120 Hz, with 12-bit A/D conversion using 3 Hz high-pass and 900 Hz low-pass filter. 
All signals were pre-amplified with gain of 5, using amplifiers placed on subject’s arm, resulting in total gain of 200. 
'''
def iterator(data,BS=16,shuffle=True):
  # B,500,9 --> B,X,8,8
  B,C,features = data.shape
  X = data[:,:,:-2]
  Y = data[:,:,-2:]
  X = X[:,:-(C%(features-2)),:]
  Y = Y[:,:-(C%(features-2))]
  C = (C//(features-2))
  X = X.reshape(B,C,features-2*features-2)
  Y = Y.reshape(B,C,features-2,2)[:,:,0,:]
  assert X.shape[1] == Y.shape[1]
  
  if shuffle:
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X,Y = X[indices],Y[indices]

  for i in range(0, X.shape[0], BS):
    X_batch = X[i:i + BS]
    Y_batch = Y[i:i + BS]
    Y_batch = Y_batch[:,:,0]/Y_batch[:,:,1]
    Y_batch[Y_batch<0] = 0
    yield X_batch, Y_batch

def load_putemg_force():
  f = 'putEMG/Data-HDF5'
  fs = [os.path.join(f,x) for x in os.listdir(f) if 'mvc' in x]

  def adc2uV(x):
    return x/(2**12)*(10**6/200)

  # return N total chunks where N = total_samps // samples_per_window
  def chunk(buff,window,overlap,sr=1280):

    # return 2d list of size (samps_per_window,features)
    def listify(chunk,n=10):
      fs = [l['force'] for l in chunk[-n:]]
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
  for j,f in enumerate(fs):
    if j == 10: break
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

    all.append(np.array(chunk(newrows,500,250)))
  res = np.vstack(all)
  print(res.shape)

  out = 'out/force.npy'
  np.save(out,res)

if __name__ == '__main__':
  #load_putemg_force()
  train()