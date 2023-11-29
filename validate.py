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

def train(normalize,file,epochs=50):
  data = np.load(file)
  train_size = int(0.8*len(data))  # 80% of data for training
  val_size = len(data)-train_size  # Remaining 20% for validation
  train_dataset, val_dataset = random_split(data, [train_size, val_size])

  net = TEMG(8*8 if normalize else 9*9,layers=2,mlp_d=256)
  loss = MSELoss()
  optim = Adam(net.parameters())

  train_loss,val_loss = [],[]
  for i in range(epochs):
    ts = []
    for n,(X,Y) in enumerate(tqdm(t:=iterator(np.array(train_dataset), BS=4, shuffle=True, normalize=normalize))):
      optim.zero_grad()
      X,Y = torch.tensor(X,dtype=torch.float32,requires_grad=True), torch.tensor(Y[:,0],dtype=torch.float32)
      o = net(X).squeeze()
      l = loss(o,Y)
      l.backward()
      optim.step()
      ts.append(l.item())

    train_loss.append(sum(ts)/len(ts))
    if i % 4 == 0:
      vals = []
      for n,(X,Y) in enumerate(tqdm(t:=iterator(np.array(val_dataset), BS=4, shuffle=False, normalize=normalize))):
        X,Y = torch.tensor(X,dtype=torch.float32,requires_grad=True), torch.tensor(Y[:,0],dtype=torch.float32)
        o = net(X).squeeze()
        l = loss(o,Y)
        vals.append(l.item())
      print(f'Epoch {i} | {sum(vals)/len(vals):7.2f} val loss')
      val_loss.append(sum(vals)/len(vals))
    print(f'Epoch {i} | {sum(ts)/len(ts):7.2f} train loss')

  # TODO: write train_loss and val_loss to two independent log files  
  with open(f'validation_train_loss_{normalize}.log','w') as f:
    for t in train_loss:
      f.write(f'{t}\n')
  with open(f'validation_val_loss_{normalize}.log','w') as f:
    for v in val_loss:
      f.write(f'{v}\n')

def read(normalize):
    train_loss_file,val_loss_file = None,None
    for file in os.listdir('.'):
      print(file)
      if file.startswith(f'validation_train_loss_{normalize}'): train_loss_file = file
      elif file.startswith(f'validation_val_loss_{normalize}'): val_loss_file = file
    if train_loss_file is None or val_loss_file is None: return None
    with open(train_loss_file, 'r') as f:
        train_loss = [float(line.strip()) for line in f.readlines()]
    with open(val_loss_file, 'r') as f:
        val_loss = [float(line.strip()) for line in f.readlines()]
    return normalize, train_loss, val_loss

def visualize(normalize,train_loss,val_loss):
  plt.subplot(1,2,1)
  plt.plot(list(range(len(train_loss))), train_loss, color='blue')
  plt.xlabel('Epochs')
  plt.ylabel('Mean Squared Error')
  plt.title("Train Loss")

  plt.subplot(1,2,2)
  ticks = list(range(0,len(val_loss)*4,4))
  plt.plot(ticks,val_loss, color='orange')
  labels = [str(i) for i in ticks]
  plt.xticks(ticks,labels)
  plt.xlabel('Epochs')
  plt.title("Validation Loss")
  
  plt.suptitle(f"{'Raw grip force' if not normalize else 'MVC normalized grip force'} prediction")
  plt.show()

def iterator(data,BS=4,shuffle=True,normalize=False):
  offset = 1 if not normalize else 2
  B,C,features = data.shape
  X,Y = data[:,:,:-offset],data[:,:,-offset:]
  X,Y = X[:,:-(C%(features-offset)),:],Y[:,:-(C%(features-offset))]
  C = (C//(features-offset))
  X = X.reshape(B,C,(features-offset)*(features-offset))
  Y = Y.reshape(B,C,features-offset,offset)[:,:,0,:]
  assert X.shape[1] == Y.shape[1]

  if shuffle:
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X,Y = X[indices],Y[indices]

  for i in range(0, X.shape[0], BS):
    X_batch,Y_batch = X[i:i+BS],Y[i:i+BS]
    if normalize:
      Y_batch = Y_batch[:,:,0]/Y_batch[:,:,1]
      Y_batch[Y_batch<0] = 0
    else:
      Y_batch = Y_batch[:,0,:]
    yield X_batch, Y_batch

'''
putEMG dataset info
  - force data was sampled at 5120 Hz, with 12-bit A/D conversion 
  using 3 Hz high-pass and 900 Hz low-pass filter. 
  - All signals were pre-amplified with gain of 5, using amplifiers 
  placed on subject’s arm, resulting in total gain of 200. 
'''
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


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--train',type=int,default=1)
  parser.add_argument('-n', '--normalize',type=int,default=1)
  parser.add_argument('-l', '--load_data',type=int,default=0)
  parser.add_argument('-v', '--visualize',type=int,default=0)
  args = parser.parse_args()
  folder = 'putEMG/Data-HDF5'
  file = 'out/force.npy'
  if args.load_data:
    load_putemg_force(folder,file,sr=1280,window=500,overlap=250,avg_last_n=10,adc_bit=10,gain=200)
  if args.train:
    train(args.normalize,file,epochs=10)
  if args.visualize:
    ret = read(args.normalize)
    if ret:
      n,train_loss,val_loss = ret 
      visualize(n,train_loss,val_loss)
    else:
      raise Exception("no validation and loss files found")