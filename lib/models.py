from torch import nn
from torch.nn import functional as F 
import torch
import numpy as np
import math

def pos_embedding(seq_sz,embedding_d):
  pos = np.zeros((seq_sz+1,embedding_d))
  for i in range(0,len(pos),2):
    n = np.arange(embedding_d)
    pos[i] = np.sin(i/10000**(2*n/embedding_d))
    if i+1 < len(pos):
      pos[i+1] = np.cos(i/10000**(2*n/embedding_d))
  return torch.tensor(pos)

class MLP(nn.Module):
  def __init__(self,win):
    super(MLP,self).__init__()
    self.l1 = nn.Linear(win,128)
    self.l2 = nn.Linear(128,64)
    self.l3 = nn.Linear(64,32)
    self.l4 = nn.Linear(32,1)

  def forward(self,x):
    x = self.l1(x)
    x = self.l2(x)
    x = self.l3(x)
    x = self.l4(x)
    return x

class Layer(nn.Module):
  def __init__(self,h,d,mlp_d=128):
    super(Layer,self).__init__()
    self.LN = nn.LayerNorm(d) 
    self.MSA = MultiHead(h,d)
    self.MLP = nn.Sequential(
      nn.Linear(d,mlp_d),
      nn.Linear(mlp_d,d),
      nn.GELU()
    )
  def forward(self,z):
    z = self.MSA(self.LN(z)) + z
    z = self.MLP(self.LN(z)) + z
    return z

class MultiHead(nn.Module):
  def __init__(self,h,d):
    super(MultiHead,self).__init__()
    self.h,self.d,self.dh = h,d,int(d/h)
    self.heads = [Head(d,self.dh) for _ in range(h)]
    self.w = nn.Parameter(torch.randn(h*self.dh,d))
  
  def forward(self,x):
    # B,S,h*dh
    all_heads = torch.cat([head(x) for head in self.heads],dim=-1)
    return all_heads @ self.w

class Head(nn.Module):
  def __init__(self,d,dh):
    super(Head,self).__init__()
    self.w = nn.Parameter(torch.randn(d,3*dh))
    self.d,self.dh = d,dh
  
  def forward(self,x):
    B,S,d = x.shape
    x = x@self.w
    # S,dh
    q,k,v = x[:,:,:self.dh],x[:,:,self.dh:2*self.dh],x[:,:,2*self.dh:3*self.dh]
    # B,S,S
    p = F.softmax((torch.bmm(q,k.transpose(1,2)))/math.sqrt(self.dh),dim=-1)
    out = p@v
    return out

class TEMG(nn.Module):
  def __init__(self,feat_sz,embedding_d=32,layers=1,heads=8,mlp_d=128):
    super(TEMG,self).__init__()
    # shared linear transformation
    self.lin = nn.Linear(feat_sz,embedding_d)
    self.cls_tok = nn.Parameter(torch.rand(1,1,embedding_d))
    self.layers = [Layer(heads,embedding_d,mlp_d=mlp_d) for _ in range(layers)]
    self.LN = nn.LayerNorm(embedding_d) 
    self.linear = nn.Linear(embedding_d,1)
    self.embedding_d = embedding_d
    
  def forward(self,x):
    B,S,_ = x.shape
    x = self.lin(x) # B,S,d

    # add cls -> B,S+1,d
    cls_toks = self.cls_tok.expand(B,-1,-1)
    x = torch.cat([cls_toks,x],dim=1)
    pos = pos_embedding(S,self.embedding_d)

    # add pos embed -> B,S+1,d
    x += pos.expand(B,-1,-1)
    for layer in self.layers:
      x = layer(x)
    return self.linear(self.LN(x)[:,0,:])#self.LN(x)[:,0,:]) 